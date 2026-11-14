# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Chunked and recurrent GDN compute.

Forked from recurrent_scan_v2.py with various optimizations applied.
"""

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config

# The `hc:` chunked path keeps decay gates in log2 space so each exponential is
# a single hardware `exp2` op on the TPU EUP rather than multi-op `jnp.exp`.
_LOG2_E = math.log2(math.e)


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last dim, accumulating in float32."""
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def _l2_norm_scaled_t(
    x: jax.Array, out_scale: float = 1.0, eps: float = 1e-6
) -> tuple[jax.Array, jax.Array]:
    """Transposes `[..., chunk, dim]`, L2-normalizes along `dim`, and scales.

    Summing along the transposed sublane axis keeps the reduction in `[1, chunk]`
    (1 vreg per 128 tokens) so `jax.lax.rsqrt` executes once per 128 tokens
    instead of emitting horizontal `vadd.xlane` + `vrsqrt` + `vrcp` on every
    `[8, 128]` input vreg.

    Args:
      x: Input array of shape `[..., chunk, dim]`.
      out_scale: Post-normalization scalar multiplier.
      eps: Variance floor added before `rsqrt`.

    Returns:
      Tuple `(xt_scaled, inv)` where `xt_scaled` has shape `[..., dim, chunk]` and
      `inv` is the unscaled unit-norm `rsqrt` of shape `[..., 1, chunk]`.
    """
    xt = jnp.swapaxes(x.astype(jnp.float32), -2, -1)
    inv = jax.lax.rsqrt(jnp.sum(xt * xt, axis=-2, keepdims=True) + eps)
    if out_scale != 1.0:
        return xt * (inv * jnp.float32(out_scale)), inv
    return xt * inv, inv


def _l2_norm_fast(x: jax.Array, out_scale: float = 1.0, eps: float = 1e-6) -> jax.Array:
    """L2-normalizes `[..., chunk, dim]` via transposed sublane reduction."""
    xt_s, _ = _l2_norm_scaled_t(x, out_scale=out_scale, eps=eps)
    return jnp.swapaxes(xt_s, -2, -1).astype(x.dtype)


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32  # pyrefly: ignore[bad-return]
        case 2:
            return jnp.int16  # pyrefly: ignore[bad-return]
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def fused_transpose_broadcast(
    x: jax.Array,
    src_dim: int,
    dst_dim: int,
    dst_size: int | None = None,
) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_shape[dst_dim] = dst_size if dst_size is not None else mask_shape[src_dim]
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


_HC_PREFIX = "hc:"
# MXU operand sites of the hillclimb path that flag `<site>1` limits to one
# float8 limb: K, Q, state_prev, L, v_diff, the block inverses M, the chain
# LHS W, the chain RHS (solved rows), u, K^T . beta_gb and (Q K^T . D).
_HC_ONE_LIMB_SITES = frozenset(
    {
        "k",
        "q",
        "kqs",
        "s",
        "l",
        "b",
        "m",
        "w",
        "xc",
        "mc",
        "y",
        "u",
        "kg",
        "oqk",
    }
)
_HC_BLOCK_SIZES = (8, 16, 32)
# Static power-of-two scale of K, Q, L, W, K^T . beta_gb and (Q K^T . D) at
# the MXU input of the hillclimb path. |K| <= 1 (L2-normalized) and |L| <= 1,
# so 256 keeps them inside the float8_e4m3fn range (448) with headroom while
# lifting them out of its subnormal range; it is folded into the gate
# exponents, so it costs no multiplies on the chunk x chunk maps.
_HC_SCALE = 256.0
_HC_LOG2_SCALE = 8.0


@dataclasses.dataclass(frozen=True)
class _HcOptions:
    """Options of an `hc:<flag>+<flag>...` hillclimb `gdn_mxu_fp8_mode`.

    The hillclimb path (`_chunked_gdn_per_seq_hc`) replaces the VPU forward
    substitution of `(I + L) u = v_diff` with either a block-Jacobi chain or a
    Schur block-doubling inverse and pushes MXU operands as single-pass bfloat16
    or one-/two-limb float8. Key flag groups:

    - `bj8` / `bj16` / `bj32`: base diagonal block size B inverted on the VPU
      (default 16).
    - `db<n>` (`n` in `32, 64, 128, 256`), `mdbl`, `fastdb`, `t2b`, `msolve`,
      `sdf8`, `d1sl`: Schur block-doubling inverse up to block size `n` (adapting
      Bao et al., FlyWheel, arXiv:2611.01426). `mdbl` doubles full `M_b` instead
      of `M_b - I`; `fastdb` packs strips with integer bitmasks and fuses final
      block assembly; `t2b` runs the two-block solve tail at `db64` on 128-token
      chunks; `msolve` computes `u = M y` directly; `sdf8` sets `s_l = 32` and
      `s_d = 8` (`s_l * s_d = 256`) to fold intermediate scaling; `d1sl` slices
      the second half of packed `M` instead of zero-padding `y1`.
    - `f8`, `chain8`, `<site>1`, `bf_<site>`, `lolo`, `split32`, `tlimb`, `lu1`,
      `zfp8`, `noclamp`, `exact`: MXU operand precision controls. `f8` splits
      operands into two `gdn_mxu_operand_dtype` limbs accumulated in the MXU
      result buffer; `tlimb` masks the top bfloat16 bits (`0xFFF0`) for exact
      float8 residuals on solve sites; `lu1` uses a single `u0` limb in the
      off-diagonal `L_10 @ u0` multiply while keeping two limbs for state and
      output updates; `zfp8` converts sub-32-row doubling tiles in bfloat16.
    - `vscale`, `vdbf`: dynamic per-column power-of-two scaling of `state_prev`
      and `v_diff` along the in-lane row axis (in packed bfloat16 under `vdbf`),
      keeping untuned checkpoints with `1000x` channel outliers inside `e4m3fn`
      range with zero cross-lane reductions.
    - `ss<n>` / `sb<n>` / `sm<n>` / `sq<n>`: power-of-two static scale folds on
      `state_prev`, `v_diff`, `M - I`, and `Q` in `Q K^T` (`sq` defaults to 256).
    - `mchain`: chained per-block MXU inverse (`u_i = y_i + (M_i - I) @ y_i`).
    - `g<n>`, `sub<n>`, `first<n>`, `last<n>`, `seqg`, `sfirst`, `dstate`,
      `lazyv`, `rhs0` / `lhs0`: head-group sizing and software-pipeline order.
    - `bfproj`, `bfop`, `bfoqk`, `bfl`, `rf256` / `rfnorm` / `fnorm`, `ggate` /
      `cfgate` / `fgate` / `lgate` / `fastg`, `fko`, `dkg`, `dlimb`, `ubf`,
      `ulw`: VPU layout, fast normalization/gate transposes, and bfloat16 staging.
    - `qpad`, `fstate`, `fout`: coprime-stride VMEM QKV scratchpad and per-head
      streaming state/output VMEM I/O.

    Requires `chunk_size` divisible by 32 and by B; smaller chunks fall back to
    the default chunked path (`_hc_active`).
    """

    block: int = 16
    f8: bool = False
    chain8: bool = False
    mchain: bool = False
    one_limb: frozenset[str] = frozenset()
    bf16_sites: frozenset[str] = frozenset()
    exact: bool = False
    group: int = 32
    split32: bool = False
    drop_lo_lo: bool = True
    last: int = 0
    first: int = 0
    rhs0: bool = True
    sfirst: bool = False
    fgate: bool = False
    lgate: bool = False
    cfgate: bool = False
    ggate: bool = False
    fnorm: bool = False
    rfnorm: bool = False
    rf256: bool = False
    fko: bool = False
    ulw: bool = False
    dlimb: bool = False
    dkg: bool = False
    ubf: bool = False
    dstate: bool = False
    fstate: bool = False
    fout: bool = False
    qpad: bool = False
    seqg: bool = False
    mdbl: bool = False
    fastdb: bool = False
    fastg: bool = False
    vscale: bool = False
    vdbf: bool = False
    zfp8: bool = False
    t2b: bool = False
    lazyv: bool = False
    msolve: bool = False
    sdf8: bool = False
    tlimb: bool = False
    bfop: bool = False
    bfproj: bool = False
    bfoqk: bool = False
    bfl: bool = False
    noclamp: bool = False
    d1sl: bool = False
    lu1: bool = False
    sub: int = 0
    db: int = 0
    state_scale: float = 1.0
    rhs_scale: float = 1.0
    inverse_scale: float = 1.0
    q_scale: float = _HC_SCALE


_HC_EXTRA_SCALES = {
    "ss": "state_scale",
    "sb": "rhs_scale",
    "sm": "inverse_scale",
    "sq": "q_scale",
}


def _hc_options(cfg: config.GDNConfig) -> _HcOptions | None:
    """Parses `cfg.gdn_mxu_fp8_mode`; None unless it is an `hc:` mode."""
    mode = cfg.gdn_mxu_fp8_mode
    if not mode.startswith(_HC_PREFIX):
        return None
    block, f8, chain8, mchain, exact = 16, False, False, False, False
    group, split32, drop_lo_lo, last, first = 32, False, True, 0, 0
    rhs0, sfirst, db, sub = True, False, 0, 0
    fgate, lgate, cfgate, ggate, fastg = False, False, False, False, False
    fnorm, rfnorm, rf256 = False, False, False
    fstate, fout, qpad, seqg = False, False, False, False
    mdbl, fastdb, t2b, msolve, sdf8, d1sl = (
        False,
        False,
        False,
        False,
        False,
        False,
    )
    vscale, vdbf, zfp8, lazyv = False, False, False, False
    tlimb, bfop, bfproj, bfoqk, bfl = False, False, False, False, False
    noclamp, lu1 = False, False
    fko, ulw, dlimb, dkg, ubf, dstate = False, False, False, False, False, False
    one_limb = set()
    bf16_sites = set()
    scales = {}
    for flag in mode[len(_HC_PREFIX) :].split("+"):
        if not flag:
            continue
        if flag.startswith("bj") and flag[2:].isdigit():
            block = int(flag[2:])
            if block not in _HC_BLOCK_SIZES:
                raise ValueError(
                    f"hc block size must be one of {_HC_BLOCK_SIZES}; got {flag!r}."
                )
        elif flag == "f8":
            f8 = True
        elif flag == "chain8":
            chain8 = True
        elif flag == "mchain":
            mchain = True
        elif flag == "exact":
            exact = True
        elif flag == "split32":
            split32 = True
        elif flag == "lolo":
            drop_lo_lo = False
        elif flag == "rhs0":
            rhs0 = True
        elif flag == "lhs0":
            rhs0 = False
        elif flag == "sfirst":
            sfirst = True
        elif flag == "fgate":
            fgate = True
        elif flag == "lgate":
            lgate = True
        elif flag == "cfgate":
            cfgate = True
        elif flag == "ggate":
            ggate = True
        elif flag == "fnorm":
            fnorm = True
        elif flag == "rfnorm":
            rfnorm = True
        elif flag == "rf256":
            rfnorm = True
            rf256 = True
        elif flag == "fko":
            fko = True
        elif flag == "ulw":
            ulw = True
        elif flag == "dlimb":
            dlimb = True
        elif flag == "dkg":
            dkg = True
        elif flag == "ubf":
            ubf = True
        elif flag == "dstate":
            dstate = True
        elif flag == "fstate":
            fstate = True
        elif flag == "fout":
            fout = True
        elif flag == "qpad":
            qpad = True
        elif flag == "seqg":
            seqg = True
        elif flag == "mdbl":
            mdbl = True
        elif flag == "fastdb":
            fastdb = True
        elif flag == "fastg":
            fastg = True
        elif flag == "vscale":
            vscale = True
        elif flag == "vdbf":
            vdbf = True
        elif flag == "zfp8":
            zfp8 = True
        elif flag == "t2b":
            t2b = True
        elif flag == "lazyv":
            lazyv = True
        elif flag == "msolve":
            msolve = True
        elif flag == "sdf8":
            sdf8 = True
        elif flag == "tlimb":
            tlimb = True
        elif flag == "bfop":
            bfop = True
        elif flag == "bfproj":
            bfop = True
            bfproj = True
        elif flag == "bfoqk":
            bfoqk = True
        elif flag == "bfl":
            bfl = True
        elif flag == "noclamp":
            noclamp = True
        elif flag == "d1sl":
            d1sl = True
        elif flag == "lu1":
            lu1 = True
        elif flag.startswith("sub") and flag[3:].isdigit():
            sub = int(flag[3:])
        elif flag.startswith("db") and flag[2:].isdigit():
            db = int(flag[2:])
            if db not in (32, 64, 128, 256):
                raise ValueError(
                    f"hc block-doubling size must be 32, 64, 128, or 256; got {flag!r}."
                )
        elif flag.startswith("first") and flag[5:].isdigit():
            first = int(flag[5:])
        elif flag.startswith("last") and flag[4:].isdigit():
            last = int(flag[4:])
        elif flag.startswith("g") and flag[1:].isdigit():
            group = int(flag[1:])
        elif flag.endswith("1") and flag[:-1] in _HC_ONE_LIMB_SITES:
            one_limb.add(flag[:-1])
        elif flag.startswith("bf_") and flag[3:] in _HC_ONE_LIMB_SITES:
            bf16_sites.add(flag[3:])
        elif flag[:2] in _HC_EXTRA_SCALES and flag[2:].isdigit():
            value = int(flag[2:])
            if value <= 0 or value & (value - 1):
                raise ValueError(f"hc scale {flag!r} must be a power of two.")
            scales[_HC_EXTRA_SCALES[flag[:2]]] = float(value)
        else:
            raise ValueError(f"Unknown hc flag {flag!r} in {mode!r}.")
    if f8 and (
        cfg.gdn_mxu_operand_dtype is None
        or jnp.dtype(cfg.gdn_mxu_operand_dtype).itemsize != 1
    ):
        raise ValueError(
            f"{mode!r} needs an 8-bit gdn_mxu_operand_dtype; got"
            f" {cfg.gdn_mxu_operand_dtype}."
        )
    if db and (mchain or block >= db):
        raise ValueError(
            f"{mode!r}: db{db} requires block < {db} and cannot combine with mchain."
        )
    return _HcOptions(
        block=block,
        f8=f8,
        chain8=chain8,
        mchain=mchain,
        one_limb=frozenset(one_limb),
        bf16_sites=frozenset(bf16_sites),
        exact=exact,
        group=group,
        split32=split32,
        drop_lo_lo=drop_lo_lo,
        last=last,
        first=first,
        rhs0=rhs0,
        sfirst=sfirst,
        fgate=fgate,
        lgate=lgate,
        cfgate=cfgate,
        ggate=ggate,
        fnorm=fnorm,
        rfnorm=rfnorm,
        rf256=rf256,
        fko=fko,
        ulw=ulw,
        dlimb=dlimb,
        dkg=dkg,
        ubf=ubf,
        dstate=dstate,
        fstate=fstate,
        fout=fout,
        qpad=qpad,
        seqg=seqg,
        mdbl=mdbl,
        fastdb=fastdb,
        fastg=fastg,
        vscale=vscale,
        vdbf=vdbf,
        zfp8=zfp8,
        t2b=t2b,
        lazyv=lazyv,
        msolve=msolve,
        sdf8=sdf8,
        tlimb=tlimb,
        bfop=bfop,
        bfproj=bfproj,
        bfoqk=bfoqk,
        bfl=bfl,
        noclamp=noclamp,
        d1sl=d1sl,
        lu1=lu1,
        sub=sub,
        db=db,
        **scales,
    )


# Operand sites of the short solve-chain matmuls (B LHS rows): bf16 unless
# `chain8`. `mc` / `y` are the block inverse and its RHS under `mchain`.
_HC_CHAIN_SITES = frozenset({"w", "xc", "mc", "y"})


def _hc_limbs(
    x: jax.Array,
    site: str,
    opts: _HcOptions,
    cfg: config.GDNConfig,
    *,
    allow_tlimb: bool = True,
) -> list[jax.Array]:
    """Splits float32 `x` into MXU operand limbs for `site`.

    Returns one bf16 limb unless `site` runs in float8 (`f8`, plus `chain8` for
    chain sites); then `[fp8(x)]` with flag `<site>1`, else `[hi, lo]` at `hi`'s
    scale so limb products sum without rescaling. `exact` returns `[x]`.

    Args:
      x: Float32 operand array.
      site: Short operand site tag (for example `"s"`, `"q"`, `"u"`).
      opts: Parsed hillclimb mode options.
      cfg: GDN operator configuration.
      allow_tlimb: Whether BF16 mantissa truncation is permitted for `site`.

    Returns:
      List of 1 or 2 limb arrays ready for `_hc_dot`.
    """
    if opts.exact:
        return [x]
    use_f8 = opts.f8 and (opts.chain8 or site not in _HC_CHAIN_SITES)
    if site in opts.bf16_sites:
        use_f8 = False
    if not use_f8:
        return [x.astype(jnp.bfloat16)]
    dtype = cfg.gdn_mxu_operand_dtype
    assert dtype is not None
    clamp_f8 = (
        opts.vscale
        and not opts.noclamp
        and jnp.dtype(dtype) == jnp.float8_e4m3fn
        and site in ("u", "y", "xc")
    )
    if site in opts.one_limb:
        if clamp_f8:
            x = jnp.clip(x, jnp.float32(-448.0), jnp.float32(448.0))
        return [x.astype(dtype)]
    if opts.split32:
        if clamp_f8:
            x = jnp.clip(x, jnp.float32(-448.0), jnp.float32(448.0))
        hi = x.astype(dtype)
        return [hi, (x - hi.astype(jnp.float32)).astype(dtype)]
    x_bf16 = x.astype(jnp.bfloat16)
    if clamp_f8:
        x_bf16 = jnp.clip(x_bf16, jnp.bfloat16(-448.0), jnp.bfloat16(448.0))
    if (
        allow_tlimb
        and opts.tlimb
        and jnp.dtype(dtype) == jnp.float8_e4m3fn
        and site in ("s", "b", "u", "y", "xc")
    ):
        u16 = jax.lax.bitcast_convert_type(x_bf16, jnp.uint16)
        hi_bf16 = jax.lax.bitcast_convert_type(
            jnp.bitwise_and(u16, jnp.uint16(0xFFF0)), jnp.bfloat16
        )
        return [hi_bf16.astype(dtype), (x_bf16 - hi_bf16).astype(dtype)]
    hi = x_bf16.astype(dtype)
    # bf16(x) - hi has at most 4 significant bits, so it is exact in bf16.
    return [hi, (x_bf16 - hi.astype(jnp.bfloat16)).astype(dtype)]


def _hc_col_pow2_scale(x: jax.Array, cutoff: float) -> tuple[jax.Array, jax.Array]:
    """Exact per-column pow2 `(scale, inv_scale)` keeping `|x * inv| < cutoff`.

    Reduces strictly along `axis=0` (in-lane on TPU, zero XLU cross-lane ops) and
    clamps `amax` into `[2^-96, 2^118]` before extracting IEEE-754 float32
    exponent bits so both `scale` and `inv_scale` stay finite normal powers of two
    with `scale * inv_scale == 1.0` bit-exact and nonzero column maxima mapped
    into `[cutoff / 2, cutoff)`.

    Args:
      x: Float32 2D array reduced along `axis=0`.
      cutoff: Power-of-two upper bound for `|x * inv_scale|`.

    Returns:
      Tuple `(scale, inv_scale)` of `[1, cols]` float32 exact powers of two.
    """
    amax = jnp.clip(
        jnp.max(jnp.abs(x), axis=0, keepdims=True),
        jnp.float32(2.0**-96),
        jnp.float32(2.0**118),
    )
    scaled_bits = jax.lax.bitcast_convert_type(
        amax * jnp.float32(2.0 / cutoff), jnp.int32
    )
    exp_bits = jnp.bitwise_and(scaled_bits, jnp.int32(0x7F800000))
    scale = jax.lax.bitcast_convert_type(exp_bits, jnp.float32)
    inv_scale = jax.lax.bitcast_convert_type(
        jnp.int32(0x7F000000) - exp_bits, jnp.float32
    )
    return scale, inv_scale


def _hc_col_pow2_scale_bf16(
    x_bf16: jax.Array, cutoff: float
) -> tuple[jax.Array, jax.Array]:
    """Exact per-column power-of-two `(scale_bf16, inv_bf16)` from `x_bf16`.

    Reduces `|x_bf16|` along `axis=0` in packed bfloat16 (2x VALU rate, zero XLU
    ops) and masks the IEEE-754 bfloat16 exponent directly in packed `uint16`
    without float32 conversions. Both `scale_bf16` and `inv_bf16` are normal exact
    powers of two with `scale_bf16 * inv_bf16 == 1.0` bit-exact, so
    `x_bf16 * inv_bf16` shifts exponents with zero mantissa rounding.

    Args:
      x_bf16: Bfloat16 2D array reduced along `axis=0`.
      cutoff: Power-of-two upper bound for `|x_bf16 * inv_bf16|`.

    Returns:
      Tuple `(scale_bf16, inv_bf16)` of `[1, cols]` bfloat16 powers of two.
    """
    amax = jnp.clip(
        jnp.max(jnp.abs(x_bf16), axis=0, keepdims=True),
        jnp.bfloat16(2.0**-96),
        jnp.bfloat16(2.0**118),
    )
    scaled_bits = jax.lax.bitcast_convert_type(
        amax * jnp.bfloat16(2.0 / cutoff), jnp.uint16
    )
    exp_bits = jnp.bitwise_and(scaled_bits, jnp.uint16(0x7F80))
    scale_bf16 = jax.lax.bitcast_convert_type(exp_bits, jnp.bfloat16)
    inv_bf16 = jax.lax.bitcast_convert_type(jnp.uint16(0x7F00) - exp_bits, jnp.bfloat16)
    return scale_bf16, inv_bf16


def _hc_uint_keep_mask(mask: jax.Array, dtype: jnp.dtype) -> jax.Array | None:
    """Unsigned integer bitmask (`0xFF`/`0xFFFF`) matching `dtype` itemsize."""
    sz = jnp.dtype(dtype).itemsize
    if sz == 1:
        return jnp.where(mask, jnp.uint8(0xFF), jnp.uint8(0))
    if sz == 2:
        return jnp.where(mask, jnp.uint16(0xFFFF), jnp.uint16(0))
    return None


def _hc_bitmask_zero(
    x: jax.Array, mask: jax.Array, u_mask: jax.Array | None
) -> jax.Array:
    """Zeros false positions of `x` via packed `vand.u32` when `u_mask` fits."""
    if u_mask is None:
        return jnp.where(mask, x, jnp.zeros_like(x))
    u = jax.lax.bitcast_convert_type(x, u_mask.dtype)
    return jax.lax.bitcast_convert_type(jnp.bitwise_and(u, u_mask), x.dtype)


def _hc_limbs_from_bf16(
    x_bf16: jax.Array,
    site: str,
    opts: _HcOptions,
    cfg: config.GDNConfig,
    *,
    allow_tlimb: bool = True,
) -> list[jax.Array]:
    """MXU operand limbs from an already-rounded bfloat16 tensor at `site`."""
    use_f8 = opts.f8 and (opts.chain8 or site not in _HC_CHAIN_SITES)
    if site in opts.bf16_sites or not use_f8:
        return [x_bf16]
    dtype = cfg.gdn_mxu_operand_dtype
    assert dtype is not None
    if site in opts.one_limb:
        return [x_bf16.astype(dtype)]
    if (
        allow_tlimb
        and opts.tlimb
        and jnp.dtype(dtype) == jnp.float8_e4m3fn
        and site in ("s", "b", "u", "y", "xc")
    ):
        u16 = jax.lax.bitcast_convert_type(x_bf16, jnp.uint16)
        hi_bf16 = jax.lax.bitcast_convert_type(
            jnp.bitwise_and(u16, jnp.uint16(0xFFF0)), jnp.bfloat16
        )
        return [hi_bf16.astype(dtype), (x_bf16 - hi_bf16).astype(dtype)]
    hi = x_bf16.astype(dtype)
    return [hi, (x_bf16 - hi.astype(jnp.bfloat16)).astype(dtype)]


def _hc_dot(
    lhs_limbs: list[jax.Array],
    rhs_limbs: list[jax.Array],
    dimension_numbers: jax.lax.DotDimensionNumbers,
    drop_lo_lo: bool = False,
    rhs_major: bool = False,
) -> jax.Array:
    """Computes the float32 sum of MXU limb dot products.

    Mosaic accumulates summed matmuls in the MXU result buffer without extra
    result pops or VPU adds. `drop_lo_lo` skips `lo @ lo` (within ~2^-8 of
    `hi @ hi`). `rhs_major` groups `rhs_limbs[0]` passes (`a0@b1 + a0@b0 + a1@b0`
    for 3-pass dots) to avoid reloading MSRA with `b0` after `b1`.

    Args:
      lhs_limbs: LHS operand limbs `[hi]` or `[hi, lo]`.
      rhs_limbs: RHS operand limbs `[hi]` or `[hi, lo]`.
      dimension_numbers: `jax.lax.dot_general` contracting/batch dimensions.
      drop_lo_lo: If True, omits limbs with `i + j >= max(len(lhs), len(rhs))`.
      rhs_major: If True, orders passes so `rhs_limbs[0]` uses stay adjacent.

    Returns:
      Float32 accumulated dot product.
    """
    num_limbs = max(len(lhs_limbs), len(rhs_limbs))
    pairs: list[tuple[int, int]] = []
    if rhs_major and len(rhs_limbs) > 1:
        if drop_lo_lo and len(lhs_limbs) == 2 and len(rhs_limbs) == 2:
            pairs = [(0, 1), (0, 0), (1, 0)]
        else:
            for j in range(len(rhs_limbs)):
                for i in range(len(lhs_limbs)):
                    if not (drop_lo_lo and i + j >= num_limbs):
                        pairs.append((i, j))
    else:
        for i in range(len(lhs_limbs)):
            for j in range(len(rhs_limbs)):
                if not (drop_lo_lo and i + j >= num_limbs):
                    pairs.append((i, j))
    acc = None
    for i, j in pairs:
        prod = jax.lax.dot_general(
            lhs_limbs[i],
            rhs_limbs[j],
            dimension_numbers,
            preferred_element_type=jnp.float32,
        )
        acc = prod if acc is None else acc + prod
    assert acc is not None
    return acc


@dataclasses.dataclass(frozen=True)
class _HcInverseConsts:
    """Constant masks and gather indices of `_hc_block_inverses`, built once."""

    # [B, chunk]: lane l belongs to diagonal block l // B.
    block_masks: tuple[jax.Array, ...]  # block_masks[i] = (lane // B == i)
    tree_masks: tuple[jax.Array, ...]  # binary-tree bit masks when `use_tree`
    identity: jax.Array  # (lane % B == row) * diag_scale as float32
    diag_scale: float
    # gather_idx[j] = (lane // B) * B + j over rows r0(j).., the lane of column
    # j of each block.
    gather_idx: tuple[jax.Array, ...]


def _hc_block_inverse_consts(
    block: int, chunk: int, diag_scale: float = 1.0, use_tree: bool = False
) -> _HcInverseConsts:
    """Builds constant masks and gather indices for `_hc_block_inverses`."""
    shape = (block, chunk)
    lanes = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    rows = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane_block = lanes // block
    num_blk = chunk // block
    block_masks = () if use_tree else tuple(lane_block == i for i in range(num_blk))
    t_masks = []
    if use_tree:
        bit = 0
        while (1 << bit) < num_blk:
            t_masks.append(jnp.bitwise_and(lane_block >> bit, 1) == 1)
            bit += 1
    identity = jnp.where(
        lanes % block == rows, jnp.float32(diag_scale), jnp.float32(0.0)
    )
    gather_idx = []
    for j in range(block - 1):
        r0 = ((j + 1) // 8) * 8
        gather_idx.append(lane_block[r0:, :] * block + j)
    return _HcInverseConsts(
        block_masks, tuple(t_masks), identity, diag_scale, tuple(gather_idx)
    )


def _hc_select_block_diag_strip(
    strict_lower: jax.Array, consts: _HcInverseConsts
) -> jax.Array:
    """Extracts packed `[B, strip]` diagonal blocks from `[strip, strip]`."""
    block = consts.identity.shape[0]
    chunk = consts.identity.shape[1]
    if consts.tree_masks:
        lvl = [
            strict_lower[i * block : (i + 1) * block, :] for i in range(chunk // block)
        ]
        for bmask in consts.tree_masks:
            nxt = []
            for k in range(0, len(lvl), 2):
                if k + 1 < len(lvl):
                    nxt.append(jnp.where(bmask, lvl[k + 1], lvl[k]))
                else:
                    nxt.append(lvl[k])
            lvl = nxt
        return lvl[0]
    # Lane group i takes the rows of diagonal block i: one select per block.
    coef = strict_lower[:block, :]
    for i in range(1, len(consts.block_masks)):
        coef = jnp.where(
            consts.block_masks[i],
            strict_lower[i * block : (i + 1) * block, :],
            coef,
        )
    return coef


def _hc_select_block_diag(mat: jax.Array, consts: _HcInverseConsts) -> jax.Array:
    """Extracts side-by-side `[B, chunk]` diagonal blocks across all strips."""
    strip = consts.identity.shape[1]
    chunk = mat.shape[0]
    if chunk == strip:
        return _hc_select_block_diag_strip(mat, consts)
    return jnp.concat(
        [
            _hc_select_block_diag_strip(mat[s : s + strip, s : s + strip], consts)
            for s in range(0, chunk, strip)
        ],
        axis=1,
    )


def _hc_block_inverses_from_coef(
    coef: jax.Array, consts: _HcInverseConsts
) -> jax.Array:
    """Forward substitution on pre-selected unscaled `[B, strip]` `coef`."""
    block = consts.identity.shape[0]
    # Forward substitution on the identity: after step j, row j is final and
    # the rows below it have column j's contribution removed. Rows <= j have a
    # zero coefficient (the block is strictly lower), so only the 8-sublane
    # slices from row j + 1 on are touched.
    x = consts.identity
    for j in range(block - 1):
        r0 = ((j + 1) // 8) * 8
        coef_j = jnp.take_along_axis(coef[r0:, :], consts.gather_idx[j], axis=1)
        x_tail = x[r0:, :] - coef_j * x[j : j + 1, :]
        x = x_tail if r0 == 0 else jnp.concat([x[:r0, :], x_tail], axis=0)
    return x


def _hc_all_block_deltas_from_coef(
    coef_full: jax.Array,
    consts: _HcInverseConsts,
    *,
    include_identity: bool = False,
) -> jax.Array:
    """Returns `M_c - I` (or `M_c`) from unscaled `[B, chunk]` `coef_full`."""
    strip = consts.identity.shape[1]
    chunk = coef_full.shape[1]
    if chunk == strip:
        inv = _hc_block_inverses_from_coef(coef_full, consts)
        return inv if include_identity else inv - consts.identity
    parts = []
    for s in range(0, chunk, strip):
        inv = _hc_block_inverses_from_coef(coef_full[:, s : s + strip], consts)
        parts.append(inv if include_identity else inv - consts.identity)
    return jnp.concat(parts, axis=1)


def _hc_block_inverses(
    strict_lower_scaled: jax.Array, scale: float, consts: _HcInverseConsts
) -> jax.Array:
    """Inverses of the unit lower-triangular diagonal blocks of I + L.

    Args:
      strict_lower_scaled: `scale * L` of one head, [chunk, chunk], with L
        strictly lower triangular.
      scale: Scale of `strict_lower_scaled`.
      consts: Masks and gather indices for the block size B and chunk.

    Returns:
      The block inverses side by side along lanes: `diag_scale * M_c` `[B, chunk]`
      with `M_c[r, i * B + c] = ((I + L_ii)^-1)[r, c]` for diagonal block i.
    """
    coef = _hc_select_block_diag_strip(strict_lower_scaled, consts)
    if scale != 1.0:
        coef = coef * jnp.float32(1.0 / scale)
    return _hc_block_inverses_from_coef(coef, consts)


def _hc_all_block_deltas(
    strict_lower_scaled: jax.Array,
    scale: float,
    consts: _HcInverseConsts,
    *,
    include_identity: bool = False,
) -> jax.Array:
    """Returns `M_c - I` (or `M_c` if `include_identity`) `[B, chunk]`."""
    strip = consts.identity.shape[1]
    chunk = strict_lower_scaled.shape[0]
    if chunk == strip:
        inv = _hc_block_inverses(strict_lower_scaled, scale, consts)
        return inv if include_identity else inv - consts.identity
    parts = []
    for s in range(0, chunk, strip):
        inv = _hc_block_inverses(
            strict_lower_scaled[s : s + strip, s : s + strip], scale, consts
        )
        parts.append(inv if include_identity else inv - consts.identity)
    return jnp.concat(parts, axis=1)


def _hc_limb_format(
    site: str, opts: _HcOptions, cfg: config.GDNConfig
) -> tuple[int, jnp.dtype]:
    """(number of limbs, dtype) that `_hc_limbs` produces at `site`."""
    if opts.exact:
        return 1, jnp.dtype(jnp.float32)
    use_f8 = opts.f8 and (opts.chain8 or site not in _HC_CHAIN_SITES)
    if site in opts.bf16_sites or not use_f8:
        return 1, jnp.dtype(jnp.bfloat16)
    assert cfg.gdn_mxu_operand_dtype is not None
    return (1 if site in opts.one_limb else 2), jnp.dtype(cfg.gdn_mxu_operand_dtype)


@dataclasses.dataclass
class _HcHead:
    """Per-V-head intermediates passed from chunk prep to state recurrence."""

    l_w: list[jax.Array]  # 256 L in the chain-LHS format (bf16 unless chain8)
    d_limbs: list[jax.Array]  # compact s_m (M - I), chain format (`mchain`)
    oqk_limbs: list[jax.Array]  # 256 s_q (Q K^T . D)
    kg_limbs: list[jax.Array]  # 256 K^T . beta_gb (empty when deferred to `tail`)
    kq_limbs: list[jax.Array]  # [256 K; 256 Q] of the KQ head (shared)
    v_s: jax.Array | None  # s_b v (None when v_diff_s is already set)
    ko_limbs: list[jax.Array] = dataclasses.field(default_factory=list)
    k_t_s: jax.Array | None = None  # shared 256 K^T when `kg_limbs` is deferred
    l_s: jax.Array | None = None  # 256 L in float32 (preconditioner variant)
    d_c: jax.Array | None = None  # compact s_m (M - I) in float32 (same)
    v_diff_s: jax.Array | None = None
    out_updated: jax.Array | None = None
    x_blocks: list[jax.Array] = dataclasses.field(default_factory=list)
    group_limbs: list[list[jax.Array]] = dataclasses.field(default_factory=list)
    d_blocks: list[list[jax.Array]] = dataclasses.field(default_factory=list)
    lw_blocks: list[list[jax.Array]] = dataclasses.field(default_factory=list)
    cs_limbs: dict[int, list[jax.Array]] = dataclasses.field(default_factory=dict)
    v_scale: jax.Array | None = None
    w_s: jax.Array | None = None
    c_vec: jax.Array | None = None


def _hc_g_lanes(gating_log: jax.Array, chunk: int) -> jax.Array:
    """Returns log2 cumulative decay `[1, chunk, lanes]` via Hillis-Steele."""
    g_lanes = gating_log.astype(jnp.float32) * _LOG2_E
    step = 1
    while step < chunk:
        if step < 8:
            shifted = jnp.pad(g_lanes[:, :-step, :], ((0, 0), (step, 0), (0, 0)))
            g_lanes = g_lanes + shifted
        else:
            g_lanes = jnp.concat(
                [g_lanes[:, :step, :], g_lanes[:, step:, :] + g_lanes[:, :-step, :]],
                axis=1,
            )
        step *= 2
    return g_lanes


def _hc_head_groups(cfg: config.GDNConfig, opts: _HcOptions) -> list[list[int]]:
    """Partitions V heads into software-pipeline groups in program order.

    Args:
      cfg: GDN kernel configuration with head counts and head dimensions.
      opts: Parsed hillclimb options (`group`, `first`, `last`).

    Returns:
      Ordered list of V-head index lists per pipeline group.
    """
    num_heads, vpk = cfg.num_v_heads, cfg.v_per_kq_head
    group = opts.group
    if group % vpk and vpk % group:
        raise ValueError(
            f"hc head group ({group}) must divide or be a multiple of"
            f" v_per_kq_head ({vpk})."
        )
    pair_v_heads = vpk % 2 == 0 and cfg.v_head_dim % 128 == 0
    unit = vpk if pair_v_heads else 1
    group = max(group, unit)
    last = opts.last
    if last % unit:
        raise ValueError(
            f"hc last group ({last}) must be a multiple of v_per_kq_head ({vpk})."
        )
    if last >= num_heads:
        last = 0
    main = num_heads - last
    first = opts.first
    if first % unit:
        raise ValueError(
            f"hc first group ({first}) must be a multiple of v_per_kq_head ({vpk})."
        )
    if first >= main:
        first = 0
    groups = []
    start = 0
    if first:
        groups.append(list(range(0, first)))
        start = first
    groups.extend(
        list(range(g0, min(g0 + group, main))) for g0 in range(start, main, group)
    )
    if last:
        groups.append(list(range(main, num_heads)))
    return groups


class _HcChunk:
    """One chunk of the hillclimb path: gate vectors plus the per-head stages.

    `prep` is state independent (decay maps, L, Q K^T . D, K^T . beta_gb, the
    block inverses); `state_part` needs the recurrent state ([K; Q] @ S,
    v_diff, out_updated); `tail` runs the solve chain and the state / output
    products. `_chunked_gdn_per_seq_hc` runs all three stages per head group.
    """

    def __init__(
        self,
        cfg: config.GDNConfig,
        opts: _HcOptions,
        g_lanes: jax.Array | None = None,  # [1, chunk, lanes] log2 cum. decay
        beta: jax.Array | None = None,  # [1, chunk, lanes]
        load_kq: Any = None,
        load_kq_scaled: Any = None,
        load_v: Any = None,
        load_state_v: Any = None,
        store_state_v: Any = None,
        store_out_slice: Any = None,
        out_step: int = 0,
        out_dtype: jnp.dtype | None = None,
        phase_hook: Any = None,
    ):
        self.cfg = cfg
        self.opts = opts
        self._dyn_scale = bool(opts.vscale)
        self._use_vdbf = bool(opts.vdbf and not opts.exact and not opts.split32)
        self._load_kq = load_kq
        self._load_kq_scaled = load_kq_scaled
        self._load_v = load_v
        self._load_state_v = load_state_v
        self._store_state_v = store_state_v
        self._store_out_slice = store_out_slice
        self._out_step = out_step
        self._out_dtype = out_dtype
        self._phase_hook = phase_hook
        self._out_flushed = 0
        self.num_heads = cfg.num_v_heads
        self.chunk = cfg.chunk_size
        self.block = opts.block
        self.v_dim = cfg.v_head_dim
        self.kq_dim = cfg.kq_head_dim
        if self.chunk % 32 or self.chunk % self.block:
            raise ValueError(
                "hc modes need chunk_size divisible by 32 and by the block size;"
                f" got chunk_size={self.chunk}, block={self.block}."
            )
        self.num_blocks = self.chunk // self.block
        self.scale = _HC_SCALE
        self.inv_scale = 1.0 / _HC_SCALE
        self.s_s, self.s_b = opts.state_scale, opts.rhs_scale
        self.s_m, self.s_q = opts.inverse_scale, opts.q_scale
        self.tb = min(opts.db, self.chunk) if opts.db else 0
        self.use_mdbl = bool(self.tb and opts.mdbl)
        self._fast_db = bool(self.tb and opts.fastdb)
        self._msolve = bool(self.use_mdbl and opts.msolve and not opts.exact)
        self._fold_sd = False
        if self.use_mdbl and opts.sdf8 and not opts.exact:
            self.s_l = 32.0
            self.s_d = 8.0
            self._fold_sd = True
        elif self.tb:
            self.s_l = self.scale
            self.s_d = self.s_m if self.s_m != 1.0 else self.scale
        else:
            self.s_l = self.scale
            self.s_d = 1.0
        self.inv_sd = 1.0 / self.s_d
        self.sd_over_scale = self.s_d / self.s_l
        self._fold_usd = bool(self._fold_sd and self._msolve)
        self._vd_cutoff = 16.0 if self._fold_usd else 32.0
        u_scale = self.s_d if self._fold_usd else 1.0
        self._ly_inv = 1.0 / (self.s_l * u_scale)
        self._use_bfoqk = bool(opts.bfoqk and not opts.exact and not opts.split32)
        self._use_bfl = bool(
            opts.bfl
            and self.tb
            and self._fast_db
            and self.use_mdbl
            and not opts.exact
            and not opts.split32
        )
        self._kkt_diag_map: dict[int, jax.Array] = {}
        # The solved rows u carry `s_b * u_scale`; s_m only scales the block-inverse
        # operand and is divided out of its products. Unscale factors of the final
        # products [256 K^T . beta_gb; s_q Q K^T . D] @ (s_b u_scale u):
        self.inv_state_new = self.inv_scale / (self.s_b * u_scale)
        self.inv_out_new = 1.0 / (self.s_b * self.s_q * u_scale)
        self.inv_s_m = 1.0 / self.s_m
        self.mm_dims = (((1,), (0,)), ((), ()))
        self.kkt_dims = (((1,), (1,)), ((), ()))
        self.pair_v_heads = cfg.v_per_kq_head % 2 == 0 and self.v_dim % 128 == 0
        self.half_v = cfg.v_per_kq_head // 2
        num_heads, chunk, block = self.num_heads, self.chunk, self.block

        # Gate columns. `g_col8` ([num_v_heads, chunk, 1], lane-replicated) is
        # the lane-dense g - 8 transposed to per-head column vectors, computed on
        # first use, and `gating_forward8` is its exp2 (the 2^-8 forward decay).
        # With `fgate`, `exp2` runs once on lane-dense `g_lanes8` (16 vregs per 128
        # tokens instead of 1,024) and each head slices its lane on demand without
        # spilling `[num_v_heads, chunk, 1]` gate arrays.
        self._g_col8 = None
        self._gating_forward8 = None
        self._gf_col_lanes8 = None
        self._gf_lanes8 = None
        self._gf_last_lanes = None
        self._gf_last_blk_idx = -1
        self._gf_last_blk: jax.Array | None = None
        self._g_t8_2d: jax.Array | None = None
        self._gf_t8_2d: jax.Array | None = None
        h_pad = 0
        if g_lanes is not None:
            h_pad = min(g_lanes.shape[2], max(8, ((num_heads + 7) // 8) * 8))
            g_l8 = g_lanes - _HC_LOG2_SCALE
            self.g_lanes8 = g_l8
            if opts.ggate:
                g_t8 = jnp.swapaxes(g_l8, 1, 2)[0, :h_pad, :]
                self._g_t8_2d = g_t8
                self._gf_t8_2d = jnp.exp2(g_t8)
                self._gf_last_lanes = jnp.exp2(g_lanes[:, -1:, :])
            elif opts.cfgate:
                self._gf_col_lanes8 = jnp.exp2(g_l8)
            elif opts.fgate or opts.lgate:
                self._gf_lanes8 = jnp.exp2(g_l8)
                self._gf_last_lanes = jnp.exp2(g_lanes[:, -1:, :])
            elif opts.fastg:
                self._gf_last_lanes = jnp.exp2(g_lanes[:, -1:, :])
        else:
            self.g_lanes8 = None
        # Row vectors [num_v_heads, 1, chunk] (per-head slices are free). beta
        # is folded into the decay exponent: the map is exp2(min(g_t - 8 - g_s +
        # log2 beta_s, log2 beta_s - 8)) = 2^-8 decay(s -> t) beta_s, which saves
        # the per-element multiply by beta; a zero beta (padding) gives log2 beta
        # = -inf and a zero map column.
        self._g_row_beta_2d: jax.Array | None = None
        self._clamp_row_2d: jax.Array | None = None
        self._beta_gb_2d: jax.Array | None = None
        if beta is not None:
            assert g_lanes is not None
            log2_beta = jnp.log2(beta)
            if opts.ggate:
                self._g_row_beta_2d = jnp.swapaxes(g_lanes - log2_beta, 1, 2)[
                    0, :h_pad, :
                ]
                self._clamp_row_2d = jnp.swapaxes(log2_beta - _HC_LOG2_SCALE, 1, 2)[
                    0, :h_pad, :
                ]
                self._beta_gb_2d = jnp.swapaxes(
                    beta * jnp.exp2(g_lanes[:, -1:, :] - g_lanes), 1, 2
                )[0, :h_pad, :]
            else:
                self.g_row_beta_t = jnp.swapaxes(g_lanes - log2_beta, 1, 2)[
                    0, :num_heads, None, :
                ]
                self.clamp_row = jnp.swapaxes(log2_beta - _HC_LOG2_SCALE, 1, 2)[
                    0, :num_heads, None, :
                ]
                self.beta_gb_row = jnp.swapaxes(
                    beta * jnp.exp2(g_lanes[:, -1:, :] - g_lanes), 1, 2
                )[0, :num_heads, None, :]

        mask_shape = (chunk, chunk)
        iota_r = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        iota_c = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        self.strictly_lower_mask = iota_r > iota_c
        self.lower_mask = iota_r >= iota_c
        self.block_diag_mask = (iota_r // block) == (iota_c // block)
        self._defer_limbs = bool(opts.dlimb and not opts.exact and not opts.split32)
        self._defer_kg = bool(opts.dkg or (opts.fgate and not opts.db))
        self._fuse_ko = bool(
            opts.fko
            and not self._defer_kg
            and _hc_limb_format("kg", opts, cfg) == _hc_limb_format("oqk", opts, cfg)
        )
        self._unscale_lw = bool(opts.ulw and opts.mchain)
        self._u_from_bf16 = bool(
            opts.ubf
            and opts.mchain
            and not opts.exact
            and not opts.split32
            and not (opts.f8 and opts.chain8)
            and "xc" not in opts.one_limb
        )
        # The solved rows feeding a chain matmul are converted once per packed
        # row group (16 bf16 / 32 float8 rows); a partial group is zero padded
        # and the LHS reads the matching extra lanes, which multiply the zero
        # rows.
        chain_f8 = opts.f8 and opts.chain8
        self.group_rows = max(block, 32 if chain_f8 else 16)
        self.blocks_per_group = self.group_rows // block
        self.reps = max(1, 32 // block)
        self.tile_rows = block * self.reps
        self._fuse_db_last = False
        self.db_tb_id: jax.Array | None = None
        self.db_half_id: jax.Array | None = None
        if self.tb:
            self._fuse_db_last = bool(
                self._fast_db
                and (chunk == self.tb or self.tb < 128)
                and not (self.tb == chunk and self.tb >= 256)
            )
            if self.use_mdbl and not self._msolve:
                if self._fuse_db_last:
                    hb = self.tb // 2
                    self.db_half_id = jnp.where(
                        (iota_c[:hb, :] % hb) == iota_r[:hb, :],
                        jnp.float32(self.s_d),
                        jnp.float32(0.0),
                    )
                else:
                    self.db_tb_id = jnp.where(
                        (iota_c[: self.tb, :] % self.tb) == iota_r[: self.tb, :],
                        jnp.float32(self.s_d),
                        jnp.float32(0.0),
                    )
            m_dt = _hc_limb_format("m", opts, cfg)[1]
            l_dt = _hc_limb_format("l", opts, cfg)[1]
            half_r = iota_r[: chunk // 2, :]
            half_c = iota_c[: chunk // 2, :]
            self.db_even_masks: dict[int, jax.Array] = {}
            self.db_even_umasks: dict[int, jax.Array | None] = {}
            self.db_z_rhs_masks: dict[
                int, tuple[jax.Array, jax.Array | None] | None
            ] = {}
            self.db_rhs_even_masks: dict[int, jax.Array | None] = {}
            self.db_rhs_even_umasks: dict[int, jax.Array | None] = {}
            self.db_rhs_strip_masks: dict[
                int, tuple[tuple[jax.Array, jax.Array | None], ...] | None
            ] = {}
            self.db_even_col: dict[int, jax.Array] = {}
            self.db_fuse_top_umask: jax.Array | None = None
            self.db_fuse_full_mask: tuple[jax.Array, jax.Array | None] | None = None
            b = block
            while b < self.tb:
                if not (2 * b == chunk and b >= 128):
                    m = chunk // (2 * b)
                    even_mask = (half_c // b) == 2 * (half_r // b)
                    self.db_even_masks[b] = even_mask
                    if self._fast_db and self.use_mdbl and b >= 32 and b % 32 == 0:
                        self.db_even_umasks[b] = _hc_uint_keep_mask(even_mask, l_dt)
                    else:
                        self.db_even_umasks[b] = None
                    if self._fast_db and self.use_mdbl and b < 32:
                        z_rhs = (((iota_r // b) % 2) == 1) & (
                            (iota_c // b) == (iota_r // b) - 1
                        )
                        self.db_z_rhs_masks[b] = (
                            z_rhs,
                            _hc_uint_keep_mask(z_rhs, l_dt),
                        )
                    else:
                        self.db_z_rhs_masks[b] = None
                    rhs_mask = (
                        None
                        if m == 1
                        else (((iota_r // b) % 2) == 0)
                        & ((iota_c // b) == (iota_r // b))
                    )
                    self.db_rhs_even_masks[b] = rhs_mask
                    use_strips = self._fast_db and m > 1 and b >= 32 and b % 32 == 0
                    self.db_rhs_even_umasks[b] = (
                        _hc_uint_keep_mask(rhs_mask, m_dt)
                        if (self._fast_db and rhs_mask is not None and not use_strips)
                        else None
                    )
                    if use_strips:
                        strips = []
                        for p in range(m):
                            sm = (iota_c[:b, :] // b) == (2 * p)
                            strips.append((sm, _hc_uint_keep_mask(sm, m_dt)))
                        self.db_rhs_strip_masks[b] = tuple(strips)
                    else:
                        self.db_rhs_strip_masks[b] = None
                    even_col = ((iota_c[:b, :] // b) % 2) == 0
                    self.db_even_col[b] = even_col
                    if self._fast_db and self._fuse_db_last and (2 * b == self.tb):
                        if b >= 32 and b % 32 == 0:
                            self.db_fuse_top_umask = _hc_uint_keep_mask(even_col, m_dt)
                        else:
                            fuse_full = (iota_r[: self.tb, :] >= b) | (
                                ((iota_c[: self.tb, :] // b) % 2) == 0
                            )
                            self.db_fuse_full_mask = (
                                fuse_full,
                                _hc_uint_keep_mask(fuse_full, m_dt),
                            )
                b *= 2
        inv_diag_s = (self.s_d if self.tb else self.s_m) if opts.fastdb else 1.0
        self.inverse_consts = _hc_block_inverse_consts(
            block, min(chunk, 128), diag_scale=inv_diag_s, use_tree=False
        )

    # --- gate columns ---------------------------------------------------------

    def _column(self, lanes: jax.Array) -> jax.Array:
        """Transposes `[1, chunk, lanes]` into `[num_v_heads, chunk, 1]`."""
        # Per-head lane slicing costs an XLU broadcast per vreg; transpose once.
        if self.num_heads <= 16:
            return jnp.swapaxes(lanes, 1, 2)[0, : self.num_heads, :, None]
        return fused_transpose_broadcast(
            lanes, src_dim=2, dst_dim=0, dst_size=self.num_heads
        )

    def _column_slice(self, lanes: jax.Array, h0: int, count: int) -> jax.Array:
        """[1, chunk, lanes] -> replicated [count, chunk, 1] for h0..h0+count."""
        if self.num_heads <= 16:
            return jnp.swapaxes(lanes, 1, 2)[0, h0 : h0 + count, :, None]
        mask_dtype = get_mask_dtype(lanes.dtype)
        mask_shape = (count, lanes.shape[1], lanes.shape[2])
        src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 2)
        dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, 0) + h0
        return jnp.where(src_mask == dst_mask, lanes, 0).sum(
            axis=2, keepdims=True, dtype=lanes.dtype
        )

    def _t2d_col_slice(self, t2d: jax.Array, h0: int, count: int) -> jax.Array:
        """Slices transposed `[heads, chunk]` into `[count, chunk, 1]`."""
        b0 = (h0 // 8) * 8
        b1 = min(t2d.shape[0], ((h0 + count + 7) // 8) * 8)
        cols = t2d[b0:b1, :, None]
        if b0 == h0 and b1 - b0 == count:
            return cols
        return cols[h0 - b0 : h0 - b0 + count]

    def _g_row_beta(self, v: int) -> jax.Array:
        if self._g_row_beta_2d is not None:
            return self._g_row_beta_2d[v : v + 1, :]
        return self.g_row_beta_t[v]

    def _clamp_r(self, v: int) -> jax.Array:
        if self._clamp_row_2d is not None:
            return self._clamp_row_2d[v : v + 1, :]
        return self.clamp_row[v]

    def _beta_gb(self, v: int) -> jax.Array:
        if self._beta_gb_2d is not None:
            return self._beta_gb_2d[v : v + 1, :]
        return self.beta_gb_row[v]

    @property
    def g_col8(self) -> jax.Array:
        if self._g_col8 is None:
            if self._g_t8_2d is not None:
                self._g_col8 = self._t2d_col_slice(self._g_t8_2d, 0, self.num_heads)
            else:
                assert self.g_lanes8 is not None
                self._g_col8 = self._column(self.g_lanes8)
        return self._g_col8

    @property
    def gating_forward8(self) -> jax.Array:
        """Returns `2^-8` forward decay `[chunk, num_v_heads]` for all V heads."""
        if self._gating_forward8 is None:
            if self._gf_t8_2d is not None:
                self._gating_forward8 = self._t2d_col_slice(
                    self._gf_t8_2d, 0, self.num_heads
                )
            elif self._gf_col_lanes8 is not None:
                self._gating_forward8 = self._column(self._gf_col_lanes8)
            elif self.opts.fastg:
                assert self.g_lanes8 is not None
                self._gating_forward8 = self._column(jnp.exp2(self.g_lanes8))
            else:
                self._gating_forward8 = jnp.exp2(self.g_col8)
        return self._gating_forward8

    def gating_forward(self, v: int) -> jax.Array:
        """2^-8 forward decay of head `v`: [chunk, 1] or [chunk, v_dim]."""
        if self._gf_lanes8 is not None:
            return self._gf_lanes8[0, :, v : v + 1]
        return self.gating_forward8[v]

    def gating_last(self, v: int) -> jax.Array:
        """Exact decay over the chunk of head `v`: [1, 1] or [1, v_dim]."""
        if self._gf_last_lanes is not None:
            return self._gf_last_lanes[0, :, v : v + 1]
        if self._gf_t8_2d is not None:
            b0 = (v // 8) * 8
            if b0 != self._gf_last_blk_idx or self._gf_last_blk is None:
                self._gf_last_blk_idx = b0
                self._gf_last_blk = (
                    self._t2d_col_slice(self._gf_t8_2d, b0, 8)[:, -1:, :] * self.scale
                )
            return self._gf_last_blk[v - b0]
        return self.gating_forward8[v, -1:] * self.scale

    # --- helpers -------------------------------------------------------------

    def limbs(
        self, x: jax.Array, site: str, *, allow_tlimb: bool = True
    ) -> list[jax.Array]:
        return _hc_limbs(x, site, self.opts, self.cfg, allow_tlimb=allow_tlimb)

    def limbs_bf16(self, x_bf16: jax.Array, site: str) -> list[jax.Array]:
        return _hc_limbs_from_bf16(x_bf16, site, self.opts, self.cfg)

    def dot(
        self,
        lhs_limbs: list[jax.Array],
        rhs_limbs: list[jax.Array],
        dimension_numbers: jax.lax.DotDimensionNumbers,
    ) -> jax.Array:
        return _hc_dot(
            lhs_limbs,
            rhs_limbs,
            dimension_numbers,
            self.opts.drop_lo_lo,
            self.opts.rhs0,
        )

    def chain_rhs(self, h: _HcHead) -> tuple[list[jax.Array], int]:
        """Packs buffered `h.x_blocks` into padded RHS limbs and row count."""
        bpg = self.blocks_per_group
        if self._u_from_bf16:
            while len(h.x_blocks) >= bpg:
                grp = h.x_blocks[:bpg]
                del h.x_blocks[:bpg]
                blk = jnp.concat(grp, axis=0) if bpg > 1 else grp[0]
                h.group_limbs.append([blk.astype(jnp.bfloat16)])
            parts = list(h.group_limbs)
            if h.x_blocks:
                pad = jnp.zeros(
                    (self.group_rows - len(h.x_blocks) * self.block, self.v_dim),
                    h.x_blocks[0].dtype,
                )
                parts.append(
                    [jnp.concat(h.x_blocks + [pad], axis=0).astype(jnp.bfloat16)]
                )
            rhs = [
                jnp.concat([p[0] for p in parts], axis=0)
                if len(parts) > 1
                else parts[0][0]
            ]
            return rhs, len(parts) * self.group_rows
        while len(h.x_blocks) // bpg > len(h.group_limbs):
            g = len(h.group_limbs)
            group = h.x_blocks[g * bpg : (g + 1) * bpg]
            h.group_limbs.append(
                self.limbs(
                    jnp.concat(group, axis=0) if len(group) > 1 else group[0], "xc"
                )
            )
        parts = list(h.group_limbs)
        tail = h.x_blocks[len(h.group_limbs) * bpg :]
        if tail:
            pad = jnp.zeros(
                (self.group_rows - len(tail) * self.block, self.v_dim), tail[0].dtype
            )
            parts.append(self.limbs(jnp.concat(tail + [pad], axis=0), "xc"))
        rhs = [
            jnp.concat([p[t] for p in parts], axis=0) if len(parts) > 1 else parts[0][t]
            for t in range(len(parts[0]))
        ]
        return rhs, len(parts) * self.group_rows

    # --- stages --------------------------------------------------------------

    def _get_kq(
        self,
        kqs: list[int],
        q_large: jax.Array | None,
        k_large: jax.Array | None,
    ) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
        if self._load_kq is not None:
            return self._load_kq(kqs)
        assert q_large is not None and k_large is not None
        return {kq: q_large[kq] for kq in kqs}, {kq: k_large[kq] for kq in kqs}

    def _get_v(self, vs: list[int], v_large: jax.Array | None) -> dict[int, jax.Array]:
        if self._load_v is not None:
            return self._load_v(vs)
        assert v_large is not None
        return {v: v_large[v] for v in vs}

    def _double_inverses_group(
        self,
        d_map: dict[int, jax.Array],
        l_map: dict[int, jax.Array | None],
        heads: list[int],
        built: dict[int, _HcHead] | None = None,
    ) -> dict[int, jax.Array]:
        """Stage-major Schur block-doubling of `s_d * (M - I)` up to `tb`.

        Adapts the exact block-triangular inverse recurrence of Bao et al.
        (FlyWheel, arXiv:2611.01426) to `D = s_d * (M - I)` coordinates in stacked
        strips, emitting each matmul stage across all `v in heads` before the
        dependent stage so every MXU round-trip is hidden behind group parallelism,
        and computing `D_odd @ Z` directly via `[max(b, 32), chunk] @ [chunk,
        chunk]` using block-column orthogonality without re-converting or masking
        `D_odd`.

        Args:
          d_map: Initial block-diagonal inverse residuals keyed by V-head index.
          l_map: Strictly lower-triangular `L` matrices keyed by V-head index.
          heads: V-head indices in the active head group.
          built: Optional pre-built `_HcHead` map holding cached `cs_limbs`.

        Returns:
          Dictionary mapping each `v in heads` to its doubled `s_d * (M - I)` array.
        """
        with jax.named_scope("hc_schur_block_doubling"):
            return self._double_inverses_group_impl(d_map, l_map, heads, built)

    def _double_inverses_group_impl(
        self,
        d_map: dict[int, jax.Array],
        l_map: dict[int, jax.Array | None],
        heads: list[int],
        built: dict[int, _HcHead] | None = None,
    ) -> dict[int, jax.Array]:
        chunk, mm_dims = self.chunk, self.mm_dims
        inv_sd, sd_over_scale = self.inv_sd, self.sd_over_scale
        neg_inv_scale = -self.inv_scale
        use_mdbl = self.use_mdbl
        cur = dict(d_map)
        b = self.block
        db_stage = 0
        num_db_stages = 0
        b_cnt = b
        while b_cnt < self.tb:
            num_db_stages += 1
            b_cnt *= 2
        while b < self.tb:
            if self._phase_hook is not None:
                self._phase_hook("db_stage", db_stage, num_db_stages)
            db_stage += 1
            if 2 * b == chunk and b >= 128:
                cs_map: dict[int, jax.Array] = {}
                d1_map: dict[int, list[jax.Array]] = {}
                p1_map: dict[int, jax.Array] = {}
                for v in heads:
                    d_v = cur[v]
                    if built is not None and b in built[v].cs_limbs:
                        cs_l = built[v].cs_limbs.pop(b)
                    else:
                        l_s = l_map[v]
                        assert l_s is not None
                        cs = l_s[b:, :b]
                        if not use_mdbl:
                            cs_map[v] = cs
                        cs_l = self.limbs(cs, "l")
                    d1_map[v] = self.limbs(d_v[:, b:], "m")
                    p1_map[v] = self.dot(cs_l, self.limbs(d_v[:, :b], "m"), mm_dims)
                z_map: dict[int, jax.Array] = {}
                p2_map: dict[int, jax.Array] = {}
                for v in heads:
                    p1_v = p1_map[v]
                    p1_s = (
                        p1_v
                        if ((self._fold_sd and use_mdbl) or inv_sd == 1.0)
                        else p1_v * inv_sd
                    )
                    z = p1_s if use_mdbl else cs_map[v] + p1_s
                    if not use_mdbl:
                        z_map[v] = z
                    p2_map[v] = self.dot(d1_map[v], self.limbs(z, "l"), mm_dims)
                for v in heads:
                    d_v = cur[v]
                    d0 = d_v[:, :b]
                    d1 = d_v[:, b:]
                    if use_mdbl:
                        bl = p2_map[v] * neg_inv_scale
                    else:
                        bl_sum = z_map[v] + p2_map[v] * inv_sd
                        bl = (
                            -bl_sum if sd_over_scale == 1.0 else -sd_over_scale * bl_sum
                        )
                    z_blk = jnp.zeros_like(d0)
                    cur[v] = jnp.concat(
                        [jnp.concat([d0, z_blk], axis=1), jnp.concat([bl, d1], axis=1)],
                        axis=0,
                    )
            else:
                m = chunk // (2 * b)
                even_mask = self.db_even_masks[b]
                even_umask = self.db_even_umasks[b]
                z_rhs_pair = self.db_z_rhs_masks[b]
                rhs_mask = self.db_rhs_even_masks[b]
                rhs_umask = self.db_rhs_even_umasks[b]
                strip_masks = self.db_rhs_strip_masks[b]
                even_col = self.db_even_col[b]
                cs_map = {}
                dl_map: dict[int, list[jax.Array]] = {}
                p1_map = {}
                use_zbf16 = (
                    self.opts.zfp8
                    and not self.opts.exact
                    and not self.opts.split32
                    and self._fast_db
                    and use_mdbl
                )
                z_f32_b = (
                    jnp.zeros((b, chunk), dtype=jnp.float32)
                    if (b < 32 and not (use_zbf16 and b == 16))
                    else None
                )
                z_bf16_b = (
                    jnp.zeros((b, chunk), dtype=jnp.bfloat16)
                    if (use_zbf16 and b == 16)
                    else None
                )
                for v in heads:
                    d_v = cur[v]
                    if built is not None and b in built[v].cs_limbs:
                        cs_l = built[v].cs_limbs.pop(b)
                    else:
                        l_s = l_map[v]
                        assert l_s is not None
                        if m == 1:
                            cs_stack = l_s[b : 2 * b, :]
                        else:
                            cs_stack = jnp.concat(
                                [
                                    l_s[(2 * p + 1) * b : (2 * p + 2) * b, :]
                                    for p in range(m)
                                ],
                                axis=0,
                            )
                        if not use_mdbl:
                            cs_map[v] = cs_stack
                        cs_l = self.limbs(cs_stack, "l")
                    if b < 32:
                        if z_bf16_b is not None:
                            tile32_bf16 = jnp.concat(
                                [d_v.astype(jnp.bfloat16), z_bf16_b], axis=0
                            )
                            d_l = self.limbs_bf16(tile32_bf16, "m")
                        elif use_zbf16 and 2 * b == 16:
                            assert z_f32_b is not None
                            pair_bf16 = jnp.concat([d_v, z_f32_b], axis=0).astype(
                                jnp.bfloat16
                            )
                            d_l = self.limbs_bf16(
                                jnp.concat([pair_bf16] * 2, axis=0), "m"
                            )
                        else:
                            assert z_f32_b is not None
                            pair = jnp.concat([d_v, z_f32_b], axis=0)
                            tile32 = (
                                pair
                                if 2 * b == 32
                                else jnp.concat([pair] * (32 // (2 * b)), axis=0)
                            )
                            d_l = self.limbs(tile32, "m")
                        rhs_even = []
                        for dl in d_l:
                            rep = jnp.concat([dl] * (chunk // 32), axis=0)
                            rhs_even.append(
                                rep
                                if rhs_mask is None
                                else _hc_bitmask_zero(rep, rhs_mask, rhs_umask)
                            )
                        dl_map[v] = d_l
                    else:
                        d_l = self.limbs(d_v, "m")
                        zb = jnp.zeros_like(d_l[0])
                        rhs_even = []
                        for dl in d_l:
                            if strip_masks is not None:
                                parts = []
                                for sm, sumask in strip_masks:
                                    parts.append(_hc_bitmask_zero(dl, sm, sumask))
                                    parts.append(zb)
                                rhs_even.append(jnp.concat(parts, axis=0))
                            else:
                                parts = []
                                for _ in range(m):
                                    parts.append(dl)
                                    parts.append(zb)
                                rep = jnp.concat(parts, axis=0)
                                rhs_even.append(
                                    rep
                                    if rhs_mask is None
                                    else _hc_bitmask_zero(rep, rhs_mask, rhs_umask)
                                )
                        dl_map[v] = d_l
                    p1_map[v] = self.dot(cs_l, rhs_even, mm_dims)

                zpack_map: dict[int, jax.Array] = {}
                p2_map = {}
                for v in heads:
                    p1_v = p1_map.pop(v)
                    p1_s = (
                        p1_v
                        if ((self._fold_sd and use_mdbl) or inv_sd == 1.0)
                        else p1_v * inv_sd
                    )
                    z_raw = p1_s if use_mdbl else cs_map.pop(v) + p1_s
                    dl_use = dl_map.pop(v)
                    if z_rhs_pair is not None:
                        if z_bf16_b is not None:
                            z_bf16 = z_raw.astype(jnp.bfloat16)
                            z_parts = []
                            for p in range(m):
                                z_parts.append(z_bf16_b)
                                z_parts.append(z_bf16[p * b : (p + 1) * b])
                            rhs_z = [
                                _hc_bitmask_zero(zl, z_rhs_pair[0], z_rhs_pair[1])
                                for zl in self.limbs_bf16(
                                    jnp.concat(z_parts, axis=0), "l"
                                )
                            ]
                        else:
                            assert z_f32_b is not None
                            z_parts = []
                            for p in range(m):
                                z_parts.append(z_f32_b)
                                z_parts.append(z_raw[p * b : (p + 1) * b])
                            rhs_z = [
                                _hc_bitmask_zero(zl, z_rhs_pair[0], z_rhs_pair[1])
                                for zl in self.limbs(jnp.concat(z_parts, axis=0), "l")
                            ]
                    elif use_mdbl and self._fast_db and b >= 32 and b % 32 == 0:
                        z_l = [
                            _hc_bitmask_zero(zl, even_mask, even_umask)
                            for zl in self.limbs(z_raw, "l")
                        ]
                        zb = jnp.zeros((b, chunk), dtype=z_l[0].dtype)
                        rhs_z = []
                        for zl in z_l:
                            if m == 1:
                                rhs_z.append(jnp.concat([zb, zl], axis=0))
                            else:
                                parts = []
                                for p in range(m):
                                    parts.append(zb)
                                    parts.append(zl[p * b : (p + 1) * b])
                                rhs_z.append(jnp.concat(parts, axis=0))
                    else:
                        z_strips = jnp.where(even_mask, z_raw, jnp.float32(0.0))
                        if not use_mdbl:
                            z_packed = z_strips[:b]
                            for p in range(1, m):
                                z_packed = z_packed + z_strips[p * b : (p + 1) * b]
                            zpack_map[v] = z_packed
                        if b < 32:
                            assert z_f32_b is not None
                            z_parts = []
                            for p in range(m):
                                z_parts.append(z_f32_b)
                                z_parts.append(z_strips[p * b : (p + 1) * b])
                            rhs_z = self.limbs(jnp.concat(z_parts, axis=0), "l")
                        else:
                            z_l = self.limbs(z_strips, "l")
                            zb = jnp.zeros((b, chunk), dtype=z_l[0].dtype)
                            rhs_z = []
                            for zl in z_l:
                                parts = []
                                for p in range(m):
                                    parts.append(zb)
                                    parts.append(zl[p * b : (p + 1) * b])
                                rhs_z.append(jnp.concat(parts, axis=0))
                    p2_map[v] = self.dot(dl_use, rhs_z, mm_dims)

                fuse_last = (
                    built is not None and self._fuse_db_last and (2 * b == self.tb)
                )
                for v in heads:
                    d_v = cur[v]
                    p2 = p2_map[v][:b, :] if b < 32 else p2_map[v]
                    if use_mdbl:
                        bl_packed = p2 * neg_inv_scale
                    else:
                        bl_sum = zpack_map[v] + p2 * inv_sd
                        bl_packed = (
                            -bl_sum if sd_over_scale == 1.0 else -sd_over_scale * bl_sum
                        )
                    if fuse_last:
                        assert built is not None
                        d_base = (
                            d_v - self.db_half_id
                            if self.db_half_id is not None
                            else d_v
                        )
                        d_bot = jnp.where(even_col, bl_packed, d_base)
                        if b >= 32 and b % 32 == 0 and self._fast_db:
                            raw_l = self.limbs(jnp.concat([d_base, d_bot], axis=0), "m")
                            built[v].d_blocks = [
                                [
                                    jnp.concat(
                                        [
                                            _hc_bitmask_zero(
                                                dl[:b, :],
                                                even_col,
                                                self.db_fuse_top_umask,
                                            ),
                                            dl[b:, :],
                                        ],
                                        axis=0,
                                    )
                                    for dl in raw_l
                                ]
                            ]
                        elif self.db_fuse_full_mask is not None:
                            fm, fum = self.db_fuse_full_mask
                            raw_l = self.limbs(jnp.concat([d_base, d_bot], axis=0), "m")
                            built[v].d_blocks = [
                                [_hc_bitmask_zero(dl, fm, fum) for dl in raw_l]
                            ]
                        else:
                            d_top = jnp.where(even_col, d_base, jnp.float32(0.0))
                            built[v].d_blocks = [
                                self.limbs(jnp.concat([d_top, d_bot], axis=0), "m")
                            ]
                        del cur[v]
                    else:
                        d_top = jnp.where(even_col, d_v, jnp.float32(0.0))
                        d_bot = jnp.where(even_col, bl_packed, d_v)
                        cur[v] = jnp.concat([d_top, d_bot], axis=0)
            b *= 2
        return cur

    def _finalize_db_heads(
        self,
        built: dict[int, _HcHead],
        d_init: dict[int, jax.Array],
        l_map: dict[int, jax.Array | None],
        heads: list[int],
    ) -> None:
        """Populates `d_blocks` and `lw_blocks` via `_double_inverses_group`."""
        d_tb_map = self._double_inverses_group(
            d_init, l_map, heads, built=built if self._fast_db else None
        )
        num_tb = self.chunk // self.tb
        for v in heads:
            h = built[v]
            if not h.d_blocks:
                d_tb = d_tb_map[v]
                if self.db_tb_id is not None:
                    d_tb = d_tb - self.db_tb_id
                if num_tb == 1 or self.tb < 128 or num_tb != 2:
                    # All `num_tb` diagonal inverse blocks sit side by side in `d_tb`
                    # (`[tb, chunk]`, `tb % 32 == 0`) and multiply zero-padded
                    # `[chunk, v_dim]` RHS panels in `tail`.
                    h.d_blocks = [self.limbs(d_tb, "m")]
                else:
                    h.d_blocks = [
                        self.limbs(d_tb[:, : self.tb], "m"),
                        self.limbs(d_tb[:, self.tb :], "m"),
                    ]
            if num_tb > 1 and not h.lw_blocks:
                l_s = l_map[v]
                assert l_s is not None
                l_fn = self.limbs_bf16 if l_s.dtype == jnp.bfloat16 else self.limbs
                is_2b = self.tb >= 128 or (self.opts.t2b and self.tb == 64)
                if is_2b and num_tb == 2:
                    h.lw_blocks = [l_fn(l_s[self.tb :, : self.tb], "l")]
                else:
                    # `l_s[tb:, :]` holds every strictly-lower block row without masking
                    # because `rhs_u` is zero on all unsolved blocks `>= bi * tb`.
                    h.lw_blocks = [l_fn(l_s[self.tb :, :], "l")]

    def _build_head(
        self,
        v: int,
        k_k_t: jax.Array,
        q_k_t: jax.Array,
        merged: list[jax.Array],
        k_t_s: jax.Array,
        v_s: jax.Array | None,
        v_diff_s: jax.Array | None = None,
        out_updated: jax.Array | None = None,
        g_col_v: jax.Array | None = None,
    ) -> _HcHead:
        """Builds per-head `_HcHead` tensors and block inverses for head `v`."""
        with jax.named_scope("hc_build_head_and_base_invert"):
            return self._build_head_impl(
                v, k_k_t, q_k_t, merged, k_t_s, v_s, v_diff_s, out_updated, g_col_v
            )

    def _build_head_impl(
        self,
        v: int,
        k_k_t: jax.Array,
        q_k_t: jax.Array,
        merged: list[jax.Array],
        k_t_s: jax.Array,
        v_s: jax.Array | None,
        v_diff_s: jax.Array | None = None,
        out_updated: jax.Array | None = None,
        g_col_v: jax.Array | None = None,
    ) -> _HcHead:
        if g_col_v is None:
            if self.opts.fgate:
                assert self.g_lanes8 is not None
                g_col_v = self.g_lanes8[0, :, v : v + 1]
            else:
                g_col_v = self.g_col8[v]
        map8 = jnp.exp2(jnp.minimum(g_col_v - self._g_row_beta(v), self._clamp_r(v)))
        map8_bf16 = (
            map8.astype(jnp.bfloat16)
            if (k_k_t.dtype == jnp.bfloat16 or q_k_t.dtype == jnp.bfloat16)
            else None
        )
        if k_k_t.dtype == jnp.bfloat16:
            assert map8_bf16 is not None
            kq = v // self.cfg.v_per_kq_head
            coef_v = self._kkt_diag_map[kq] * _hc_select_block_diag(
                map8, self.inverse_consts
            )
            d_c = _hc_all_block_deltas_from_coef(
                coef_v,
                self.inverse_consts,
                include_identity=bool(self.tb and self.use_mdbl),
            )
            l_s = k_k_t * map8_bf16
        else:
            l_s = k_k_t * map8
            d_c = _hc_all_block_deltas(
                l_s,
                self.s_l,
                self.inverse_consts,
                include_identity=bool(self.tb and self.use_mdbl),
            )
        if q_k_t.dtype == jnp.bfloat16:
            assert map8_bf16 is not None
            oqk_s = q_k_t * map8_bf16
        else:
            oqk_s = q_k_t * map8
        if self._fuse_ko:
            kg_part = (
                k_t_s * self._beta_gb(v).astype(jnp.bfloat16)
                if k_t_s.dtype == jnp.bfloat16
                else k_t_s * self._beta_gb(v)
            )
            if kg_part.dtype != oqk_s.dtype:
                ko_src = jnp.concat(
                    [kg_part.astype(jnp.bfloat16), oqk_s.astype(jnp.bfloat16)], axis=0
                )
            else:
                ko_src = jnp.concat([kg_part, oqk_s], axis=0)
            if (
                self._defer_limbs
                and "kg" not in self.opts.one_limb
                and "oqk" not in self.opts.one_limb
            ):
                ko_limbs = [ko_src.astype(jnp.bfloat16)]
            else:
                ko_limbs = (
                    self.limbs_bf16(ko_src, "kg")
                    if ko_src.dtype == jnp.bfloat16
                    else self.limbs(ko_src, "kg")
                )
            oqk_limbs = []
            kg_limbs = []
        elif self._defer_limbs:
            ko_limbs = []
            if "oqk" not in self.opts.one_limb:
                oqk_limbs = [oqk_s.astype(jnp.bfloat16)]
            else:
                oqk_limbs = (
                    self.limbs_bf16(oqk_s, "oqk")
                    if oqk_s.dtype == jnp.bfloat16
                    else self.limbs(oqk_s, "oqk")
                )
            if self._defer_kg:
                kg_limbs = []
            elif "kg" not in self.opts.one_limb:
                kg_limbs = [
                    k_t_s * self._beta_gb(v).astype(jnp.bfloat16)
                    if k_t_s.dtype == jnp.bfloat16
                    else (k_t_s * self._beta_gb(v)).astype(jnp.bfloat16)
                ]
            elif k_t_s.dtype == jnp.bfloat16:
                kg_limbs = self.limbs_bf16(
                    k_t_s * self._beta_gb(v).astype(jnp.bfloat16), "kg"
                )
            else:
                kg_limbs = self.limbs(k_t_s * self._beta_gb(v), "kg")
        else:
            ko_limbs = []
            oqk_limbs = (
                self.limbs_bf16(oqk_s, "oqk")
                if oqk_s.dtype == jnp.bfloat16
                else self.limbs(oqk_s, "oqk")
            )
            if self._defer_kg:
                kg_limbs = []
            elif k_t_s.dtype == jnp.bfloat16:
                kg_limbs = self.limbs_bf16(
                    k_t_s * self._beta_gb(v).astype(jnp.bfloat16), "kg"
                )
            else:
                kg_limbs = self.limbs(k_t_s * self._beta_gb(v), "kg")
        if self.tb:
            d_s = d_c if self.inverse_consts.diag_scale == self.s_d else d_c * self.s_d
            cs_limbs: dict[int, list[jax.Array]] = {}
            lw_blocks: list[list[jax.Array]] = []
            l_keep: jax.Array | None = l_s
            if self._fast_db and self.use_mdbl:
                l_fn = self.limbs_bf16 if l_s.dtype == jnp.bfloat16 else self.limbs
                hb = self.tb // 2
                if (
                    self.tb < 128
                    and self.chunk == 2 * self.tb
                    and hb >= 32
                    and hb % 32 == 0
                    and self.block <= hb
                ):
                    ltail = l_fn(l_s[hb:, :], "l")
                    cs_limbs[hb] = [
                        jnp.concat([lt[:hb, :], lt[2 * hb :, :]], axis=0)
                        for lt in ltail
                    ]
                    lw_cols = self.tb if self.opts.t2b else self.chunk
                    lw_blocks = [[lt[hb:, :lw_cols] for lt in ltail]]
                b = self.block
                while b < self.tb:
                    if b not in cs_limbs:
                        if 2 * b == self.chunk and b >= 128:
                            cs_limbs[b] = l_fn(l_s[b:, :b], "l")
                        else:
                            m = self.chunk // (2 * b)
                            if m == 1:
                                cs_s = l_s[b : 2 * b, :]
                            else:
                                cs_s = jnp.concat(
                                    [
                                        l_s[(2 * p + 1) * b : (2 * p + 2) * b, :]
                                        for p in range(m)
                                    ],
                                    axis=0,
                                )
                            cs_limbs[b] = l_fn(cs_s, "l")
                    b *= 2
                if not lw_blocks and self.chunk > self.tb:
                    if (
                        self.tb >= 128 or (self.opts.t2b and self.tb == 64)
                    ) and self.chunk == 2 * self.tb:
                        lw_blocks = [l_fn(l_s[self.tb :, : self.tb], "l")]
                    else:
                        lw_blocks = [l_fn(l_s[self.tb :, :], "l")]
                l_keep = None
            return _HcHead(
                l_w=[],
                d_limbs=[],
                oqk_limbs=oqk_limbs,
                kg_limbs=kg_limbs,
                kq_limbs=merged,
                v_s=v_s,
                ko_limbs=ko_limbs,
                k_t_s=k_t_s if self._defer_kg else None,
                l_s=l_keep,
                d_c=d_s,
                v_diff_s=v_diff_s,
                out_updated=out_updated,
                lw_blocks=lw_blocks,
                cs_limbs=cs_limbs,
            )
        if self.s_m != 1.0 and self.inverse_consts.diag_scale != self.s_m:
            d_c = d_c * self.s_m
        lw_src = l_s[self.block :, :] * self.inv_scale if self._unscale_lw else l_s
        return _HcHead(
            l_w=self.limbs(lw_src, "w"),
            d_limbs=self.limbs(d_c, "mc") if self.opts.mchain else [],
            oqk_limbs=oqk_limbs,
            kg_limbs=kg_limbs,
            kq_limbs=merged,
            v_s=v_s,
            ko_limbs=ko_limbs,
            k_t_s=k_t_s if self._defer_kg else None,
            v_diff_s=v_diff_s,
            out_updated=out_updated,
            l_s=None if self.opts.mchain else l_s,
            d_c=None if self.opts.mchain else d_c,
        )

    def _load_state(self, v: int, state_prev: jax.Array | None) -> jax.Array:
        if self._load_state_v is not None:
            return self._load_state_v(v).astype(jnp.float32)
        assert state_prev is not None
        return state_prev[v].astype(jnp.float32)

    def prep_kq(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        kqs: list[int],
    ) -> dict[int, tuple[jax.Array, jax.Array, list[jax.Array], jax.Array]]:
        """Per-KQ-head products of `prep`: K K^T, Q K^T, [K; Q] limbs, 256 K^T."""
        with jax.named_scope("hc_prep_kq_kkt_qkt"):
            return self._prep_kq_impl(q_large, k_large, kqs)

    def _prep_kq_impl(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        kqs: list[int],
    ) -> dict[int, tuple[jax.Array, jax.Array, list[jax.Array], jax.Array]]:
        f32 = jnp.float32
        scale = self.scale
        if self._load_kq_scaled is not None:
            q_s_map, q_oqk_map, k_s_map, kt_s_map = self._load_kq_scaled(
                kqs, scale, self.s_q
            )
            q_map = k_map = None
        else:
            q_s_map = q_oqk_map = k_s_map = kt_s_map = None
            q_map, k_map = self._get_kq(kqs, q_large, k_large)
        per_kq = {}
        for kq in kqs:
            # 256 K and 256 Q of this KQ head, converted once; 2^16 K K^T and 256
            # s_q Q K^T, masked (Q K^T takes its own s_q Q limbs so the state
            # projection stays at 256 and needs no extra unscale).
            if q_s_map is not None:
                assert k_s_map is not None and kt_s_map is not None
                q_s = q_s_map[kq]
                k_s = k_s_map[kq]
                k_t_s = kt_s_map[kq]
                k_limbs = self.limbs(k_s, "k")
                q_limbs = self.limbs(q_s, "q")
                q_limbs_oqk = (
                    q_limbs if q_oqk_map is None else self.limbs(q_oqk_map[kq], "q")
                )
            else:
                assert q_map is not None and k_map is not None
                q_h = q_map[kq].astype(f32)
                k_h = k_map[kq].astype(f32)
                if self.opts.rf256:
                    k_s = k_h
                    q_s = q_h
                else:
                    k_s = k_h * scale
                    q_s = q_h * scale
                if (
                    self.opts.bfop
                    and not self.opts.exact
                    and not self.opts.split32
                    and "kg" not in self.opts.one_limb
                ):
                    k_t_s = jnp.swapaxes(k_s.astype(jnp.bfloat16), 0, 1)
                else:
                    k_t_s = jnp.swapaxes(k_s, 0, 1)
                k_limbs = self.limbs(k_s, "k")
                q_limbs = self.limbs(q_s, "q")
                if self.s_q != scale:
                    q_limbs_oqk = self.limbs(
                        q_s * jnp.float32(self.s_q / scale)
                        if self.opts.rf256
                        else q_h * self.s_q,
                        "q",
                    )
                else:
                    q_limbs_oqk = q_limbs
            raw_kkt = jnp.where(
                self.strictly_lower_mask,
                self.dot(k_limbs, k_limbs, self.kkt_dims),
                0,
            )
            if self._use_bfl:
                self._kkt_diag_map[kq] = _hc_select_block_diag(
                    raw_kkt, self.inverse_consts
                ) * jnp.float32(self.inv_scale)
                k_k_t = (
                    raw_kkt * jnp.float32(self.s_l * self.inv_scale)
                    if self.s_l != scale
                    else raw_kkt
                ).astype(jnp.bfloat16)
            else:
                k_k_t = (
                    raw_kkt * jnp.float32(self.s_l * self.inv_scale)
                    if self.s_l != scale
                    else raw_kkt
                )
            q_k_t = jnp.where(
                self.lower_mask, self.dot(q_limbs_oqk, k_limbs, self.kkt_dims), 0
            )
            if self._use_bfoqk:
                q_k_t = q_k_t.astype(jnp.bfloat16)
            if "kqs" in self.opts.one_limb:
                merged = [jnp.concat([k_limbs[0], q_limbs[0]], axis=0)]
            else:
                merged = [
                    jnp.concat([kl, ql], axis=0) for kl, ql in zip(k_limbs, q_limbs)
                ]
            per_kq[kq] = (k_k_t, q_k_t, merged, k_t_s)
        return per_kq

    def _prep_batch(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        v_large: jax.Array | None,
        heads: list[int],
        per_kq: dict[int, Any] | None,
        load_v_now: bool,
    ) -> dict[int, _HcHead]:
        """Prepares `_HcHead` instances across all `v` in `heads`."""
        f32 = jnp.float32
        vpk = self.cfg.v_per_kq_head
        out: dict[int, _HcHead] = {}
        if (
            self.opts.ggate
            and not self.opts.fgate
            and self._g_t8_2d is not None
            and per_kq is None
        ):
            idx = 0
            while idx < len(heads):
                v0 = heads[idx]
                kq = v0 // vpk
                j = idx + 1
                while (
                    j < len(heads)
                    and heads[j] // vpk == kq
                    and heads[j] == v0 + (j - idx)
                ):
                    j += 1
                sub_heads = heads[idx:j]
                k_k_t, q_k_t, merged, k_t_s = self.prep_kq(q_large, k_large, [kq])[kq]
                v_sub = self._get_v(sub_heads, v_large) if load_v_now else None
                g_col_sub = self._t2d_col_slice(self._g_t8_2d, v0, len(sub_heads))
                for off, v in enumerate(sub_heads):
                    if self._phase_hook is not None:
                        self._phase_hook("prep_head", idx + off, len(heads))
                    if v_sub is not None:
                        v_arr = v_sub[v].astype(f32)
                        if self.s_b != 1.0:
                            v_arr = v_arr * self.s_b
                        v_s: jax.Array | None = v_arr
                    else:
                        v_s = None
                    out[v] = self._build_head(
                        v, k_k_t, q_k_t, merged, k_t_s, v_s, g_col_v=g_col_sub[off]
                    )
                del k_k_t, q_k_t, merged, k_t_s, v_sub, g_col_sub
                idx = j
        else:
            if per_kq is None:
                per_kq = self.prep_kq(
                    q_large, k_large, sorted({v // vpk for v in heads})
                )
            v_map = self._get_v(heads, v_large) if load_v_now else None
            g_col_grp = None
            h0 = heads[0] if heads else 0
            if (
                self.opts.lgate
                and not self.opts.fgate
                and self.g_lanes8 is not None
                and heads == list(range(h0, h0 + len(heads)))
            ):
                g_col_grp = self._column_slice(self.g_lanes8, h0, len(heads))
            for h_idx, v in enumerate(heads):
                if self._phase_hook is not None:
                    self._phase_hook("prep_head", h_idx, len(heads))
                k_k_t, q_k_t, merged, k_t_s = per_kq[v // vpk]
                if v_map is not None:
                    v_arr = v_map[v].astype(f32)
                    if self.s_b != 1.0:
                        v_arr = v_arr * self.s_b
                    v_s = v_arr
                else:
                    v_s = None
                out[v] = self._build_head(
                    v,
                    k_k_t,
                    q_k_t,
                    merged,
                    k_t_s,
                    v_s,
                    g_col_v=None if g_col_grp is None else g_col_grp[v - h0],
                )
            del g_col_grp
        if self.tb:
            d_init: dict[int, jax.Array] = {}
            l_map: dict[int, jax.Array | None] = {}
            for v in heads:
                h = out[v]
                assert h.d_c is not None
                d_init[v] = h.d_c
                l_map[v] = h.l_s
                h.d_c = None
                h.l_s = None
            self._finalize_db_heads(out, d_init, l_map, heads)
        return out

    def prep(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        v_large: jax.Array | None,
        heads: list[int],
        per_kq: dict[int, Any] | None = None,
        load_v_now: bool = True,
    ) -> dict[int, _HcHead]:
        """Runs state-independent prep (`L`, `Q K^T`, `K^T`, block inverses)."""
        eff_load_v = load_v_now and not self.opts.lazyv
        if self.opts.sub > 0 and per_kq is None:
            vpk = self.cfg.v_per_kq_head
            step = max(self.opts.sub, vpk)
            if step % vpk:
                step = ((step + vpk - 1) // vpk) * vpk
            if len(heads) > step:
                out: dict[int, _HcHead] = {}
                for i in range(0, len(heads), step):
                    out.update(
                        self._prep_batch(
                            q_large,
                            k_large,
                            v_large,
                            heads[i : i + step],
                            None,
                            eff_load_v,
                        )
                    )
                return out
        return self._prep_batch(q_large, k_large, v_large, heads, per_kq, eff_load_v)

    def _head_pairs(self, heads: list[int]) -> list[tuple[int, ...]]:
        vpk = self.cfg.v_per_kq_head
        if self.pair_v_heads:
            return [(v, v + self.half_v) for v in heads if (v % vpk) < self.half_v]
        return [(v,) for v in heads]

    def front_sfirst(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        v_large: jax.Array | None,
        heads: list[int],
        state_prev: jax.Array | None,
    ) -> dict[int, _HcHead]:
        """Emits `K K^T`, `Q K^T`, and `[K; Q] @ S` before V loads and inverses."""
        with jax.named_scope("hc_front_sfirst_state_and_prep"):
            return self._front_sfirst_impl(q_large, k_large, v_large, heads, state_prev)

    def _front_sfirst_impl(
        self,
        q_large: jax.Array | None,
        k_large: jax.Array | None,
        v_large: jax.Array | None,
        heads: list[int],
        state_prev: jax.Array | None,
    ) -> dict[int, _HcHead]:
        f32 = jnp.float32
        vpk, chunk, v_dim = self.cfg.v_per_kq_head, self.chunk, self.v_dim
        use_bfproj = bool(
            self.opts.bfproj and not self.opts.exact and not self.opts.split32
        )
        pairs = self._head_pairs(heads)
        kq_set: set[int] = set()
        for pair in pairs:
            for v in pair:
                kq_set.add(v // vpk)
        all_kqs = sorted(kq_set)
        per_kq = self.prep_kq(q_large, k_large, all_kqs)
        kq_states: dict[tuple[int, ...], jax.Array] = {}
        s_scales: dict[tuple[int, ...], list[jax.Array] | jax.Array] = {}
        for pair in pairs:
            merged = per_kq[pair[0] // vpk][2]
            if self.opts.vscale and not self.opts.exact and not self.opts.split32:
                s_bf16_parts = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    s_bf16_parts.append(s_tile.astype(jnp.bfloat16))
                s_pair_bf16 = (
                    jnp.concat(s_bf16_parts, axis=1)
                    if len(s_bf16_parts) > 1
                    else s_bf16_parts[0]
                )
                sc_pair_bf16, inv_pair_bf16 = _hc_col_pow2_scale_bf16(
                    s_pair_bf16, 256.0
                )
                s_limbs = self.limbs_bf16(s_pair_bf16 * inv_pair_bf16, "s")
                s_scales[pair] = (
                    sc_pair_bf16 if use_bfproj else sc_pair_bf16.astype(f32)
                )
            elif self.opts.vscale:
                s_limbs = []
                sc_list = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    sc_v, inv_v = _hc_col_pow2_scale(s_tile, 256.0)
                    sc_list.append(sc_v)
                    s_limbs.append(self.limbs(s_tile * inv_v, "s"))
                s_limbs = [
                    parts[0] if len(parts) == 1 else jnp.concat(parts, axis=1)
                    for parts in zip(*s_limbs)
                ]
                s_scales[pair] = sc_list
            elif self._defer_limbs and "s" not in self.opts.one_limb:
                s_bf16_parts = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    s_bf16_parts.append(s_tile.astype(jnp.bfloat16))
                s_pair_bf16 = (
                    jnp.concat(s_bf16_parts, axis=1)
                    if len(s_bf16_parts) > 1
                    else s_bf16_parts[0]
                )
                s_limbs = self.limbs_bf16(s_pair_bf16, "s")
            else:
                s_limbs = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    s_limbs.append(
                        self.limbs(
                            s_tile * self.s_s if self.s_s != 1.0 else s_tile, "s"
                        )
                    )
                s_limbs = [
                    parts[0] if len(parts) == 1 else jnp.concat(parts, axis=1)
                    for parts in zip(*s_limbs)
                ]
            kq_s = self.dot(merged, s_limbs, self.mm_dims)
            kq_states[pair] = kq_s.astype(jnp.bfloat16) if use_bfproj else kq_s

        built: dict[int, _HcHead] = {}
        cur_kq = -1
        gf_kq: jax.Array | None = None
        g_kq: jax.Array | None = None
        for p_idx, pair in enumerate(pairs):
            if self._phase_hook is not None:
                self._phase_hook("sfirst_pair", p_idx, len(pairs))
            kq_state = kq_states.pop(pair)
            sc_pair = s_scales.pop(pair, None)
            v_map = self._get_v(list(pair), v_large)
            p_kq = pair[0] // vpk
            if (
                self.opts.ggate
                and self._gf_t8_2d is not None
                and self._g_t8_2d is not None
                and all(v // vpk == p_kq for v in pair)
                and p_kq != cur_kq
            ):
                cur_kq = p_kq
                h0_kq = p_kq * vpk
                cnt_kq = min(vpk, self.num_heads - h0_kq)
                gf_kq = self._t2d_col_slice(self._gf_t8_2d, h0_kq, cnt_kq)
                g_kq = self._t2d_col_slice(self._g_t8_2d, h0_kq, cnt_kq)
            if use_bfproj and self._dyn_scale:
                assert sc_pair is not None
                vd_parts: list[jax.Array] = []
                outs_pair: dict[int, jax.Array] = {}
                g_cols_pair: dict[int, jax.Array | None] = {}
                for idx, v in enumerate(pair):
                    lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                    gf_v = (
                        gf_kq[v - cur_kq * vpk]
                        if (gf_kq is not None and v // vpk == cur_kq)
                        else self.gating_forward(v)
                    )
                    g_cols_pair[v] = (
                        g_kq[v - cur_kq * vpk]
                        if (g_kq is not None and v // vpk == cur_kq)
                        else None
                    )
                    gf_q = gf_v * (1.0 / self.s_s) if self.s_s != 1.0 else gf_v
                    if self.s_b == 1.0:
                        gf_k = gf_q
                    elif self.s_s != self.s_b:
                        gf_k = gf_v * (self.s_b / self.s_s)
                    else:
                        gf_k = gf_v
                    k_st_b = kq_state[:chunk, lanes]
                    q_st_b = kq_state[chunk:, lanes]
                    gf_k_b = gf_k.astype(jnp.bfloat16)
                    gf_q_b = gf_k_b if gf_k is gf_q else gf_q.astype(jnp.bfloat16)
                    v_raw = v_map.pop(v)
                    v_b = (
                        (v_raw.astype(f32) * self.s_b).astype(jnp.bfloat16)
                        if self.s_b != 1.0
                        else v_raw.astype(jnp.bfloat16)
                    )
                    sc_b = (
                        sc_pair[:, lanes]
                        if isinstance(sc_pair, jax.Array)
                        else sc_pair[idx].astype(jnp.bfloat16)
                    )
                    if gf_k is gf_q:
                        gf_sc_b = gf_k_b * sc_b
                        k_proj_b = k_st_b * gf_sc_b
                        out_u = q_st_b * gf_sc_b
                    else:
                        k_proj_b = (k_st_b * gf_k_b) * sc_b
                        out_u = (q_st_b * gf_q_b) * sc_b
                    outs_pair[v] = out_u
                    vd_parts.append(v_b - k_proj_b)
                del kq_state
                vd_cat = (
                    jnp.concat(vd_parts, axis=1) if len(vd_parts) > 1 else vd_parts[0]
                )
                del vd_parts
                v_sc_cat, v_inv_cat = _hc_col_pow2_scale_bf16(vd_cat, self._vd_cutoff)
                vd_scaled_cat = vd_cat * v_inv_cat
                del vd_cat
                for idx, v in enumerate(pair):
                    lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                    k_k_t, q_k_t, merged, k_t_s = per_kq[v // vpk]
                    h_v = self._build_head(
                        v,
                        k_k_t,
                        q_k_t,
                        merged,
                        k_t_s,
                        None,
                        v_diff_s=vd_scaled_cat[:, lanes],
                        out_updated=outs_pair.pop(v),
                        g_col_v=g_cols_pair.pop(v),
                    )
                    h_v.v_scale = v_sc_cat[:, lanes]
                    built[v] = h_v
            elif use_bfproj:
                for idx, v in enumerate(pair):
                    lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                    gf_v = (
                        gf_kq[v - cur_kq * vpk]
                        if (gf_kq is not None and v // vpk == cur_kq)
                        else self.gating_forward(v)
                    )
                    g_col_v = (
                        g_kq[v - cur_kq * vpk]
                        if (g_kq is not None and v // vpk == cur_kq)
                        else None
                    )
                    gf_q = gf_v * (1.0 / self.s_s) if self.s_s != 1.0 else gf_v
                    if self.s_b == 1.0:
                        gf_k = gf_q
                    elif self.s_s != self.s_b:
                        gf_k = gf_v * (self.s_b / self.s_s)
                    else:
                        gf_k = gf_v
                    k_st_b = kq_state[:chunk, lanes]
                    q_st_b = kq_state[chunk:, lanes]
                    gf_k_b = gf_k.astype(jnp.bfloat16)
                    gf_q_b = gf_k_b if gf_k is gf_q else gf_q.astype(jnp.bfloat16)
                    v_raw = v_map.pop(v)
                    v_b = (
                        (v_raw.astype(f32) * self.s_b).astype(jnp.bfloat16)
                        if self.s_b != 1.0
                        else v_raw.astype(jnp.bfloat16)
                    )
                    k_proj_b = k_st_b * gf_k_b
                    out_u = q_st_b * gf_q_b
                    k_k_t, q_k_t, merged, k_t_s = per_kq[v // vpk]
                    built[v] = self._build_head(
                        v,
                        k_k_t,
                        q_k_t,
                        merged,
                        k_t_s,
                        None,
                        v_diff_s=v_b - k_proj_b,
                        out_updated=out_u,
                        g_col_v=g_col_v,
                    )
                del kq_state
            else:
                for idx, v in enumerate(pair):
                    lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                    gf_v = (
                        gf_kq[v - cur_kq * vpk]
                        if (gf_kq is not None and v // vpk == cur_kq)
                        else self.gating_forward(v)
                    )
                    g_col_v = (
                        g_kq[v - cur_kq * vpk]
                        if (g_kq is not None and v // vpk == cur_kq)
                        else None
                    )
                    gf_q = gf_v * (1.0 / self.s_s) if self.s_s != 1.0 else gf_v
                    if self.s_b == 1.0:
                        gf_k = gf_q
                    elif self.s_s != self.s_b:
                        gf_k = gf_v * (self.s_b / self.s_s)
                    else:
                        gf_k = gf_v
                    v_col_scale = None
                    k_state = kq_state[:chunk, lanes]
                    q_state = kq_state[chunk:, lanes]
                    v_s = v_map.pop(v).astype(f32)
                    if self.s_b != 1.0:
                        v_s = v_s * self.s_b
                    if sc_pair is not None:
                        sc_v = (
                            sc_pair[:, lanes].astype(f32)
                            if isinstance(sc_pair, jax.Array)
                            else sc_pair[idx].astype(f32)
                        )
                        if gf_k is gf_q:
                            gf_sc = gf_k * sc_v
                            k_proj = k_state * gf_sc
                            out_updated = q_state * gf_sc
                        else:
                            k_proj = (k_state * gf_k) * sc_v
                            out_updated = (q_state * gf_q) * sc_v
                    else:
                        k_proj = k_state * gf_k
                        out_updated = q_state * gf_q
                    if self.opts.bfop and not self.opts.exact:
                        out_updated = out_updated.astype(jnp.bfloat16)
                    v_diff_s = v_s - k_proj
                    if self._dyn_scale:
                        if self._use_vdbf:
                            vd_bf16 = v_diff_s.astype(jnp.bfloat16)
                            v_col_scale, v_inv_bf16 = _hc_col_pow2_scale_bf16(
                                vd_bf16, self._vd_cutoff
                            )
                            v_diff_s = vd_bf16 * v_inv_bf16
                        else:
                            v_col_scale, v_col_inv = _hc_col_pow2_scale(
                                v_diff_s, self._vd_cutoff
                            )
                            v_diff_s = v_diff_s * v_col_inv
                    k_k_t, q_k_t, merged, k_t_s = per_kq[v // vpk]
                    h_v = self._build_head(
                        v,
                        k_k_t,
                        q_k_t,
                        merged,
                        k_t_s,
                        None,
                        v_diff_s=v_diff_s,
                        out_updated=out_updated,
                        g_col_v=g_col_v,
                    )
                    h_v.v_scale = v_col_scale
                    built[v] = h_v
                del kq_state
        self._kkt_diag_map.clear()
        del per_kq, gf_kq, g_kq
        ordered = {v: built[v] for v in heads}
        if self.tb:
            d_init: dict[int, jax.Array] = {}
            l_map: dict[int, jax.Array | None] = {}
            for v in heads:
                h = ordered[v]
                assert h.d_c is not None
                d_init[v] = h.d_c
                l_map[v] = h.l_s
                h.d_c = None
                h.l_s = None
            self._finalize_db_heads(ordered, d_init, l_map, heads)
        return ordered

    def state_part(
        self,
        hs: dict[int, _HcHead],
        state_prev: jax.Array | None,
        v_large: jax.Array | None = None,
    ) -> None:
        """256 s_s [K; Q] @ state_prev, then s_b v_diff and out_updated.

        Two V heads of a KQ head go side by side on lanes when the pairing fits.

        Args:
          hs: Active per-V-head state structs updated in place.
          state_prev: Optional full `[num_v_heads, kq_head_dim, v_head_dim]` state.
          v_large: Optional preloaded V tensor when `load_v` callback is unused.
        """
        with jax.named_scope("hc_state_part_kq_proj_vdiff"):
            self._state_part_impl(hs, state_prev, v_large)

    def _state_part_impl(
        self,
        hs: dict[int, _HcHead],
        state_prev: jax.Array | None,
        v_large: jax.Array | None = None,
    ) -> None:
        chunk, v_dim = self.chunk, self.v_dim
        vpk = self.cfg.v_per_kq_head
        pairs = self._head_pairs(list(hs))
        cur_gf_kq = -1
        gf_kq: jax.Array | None = None
        use_bfproj = bool(
            self.opts.bfproj and not self.opts.exact and not self.opts.split32
        )
        for pair in pairs:
            s_sc_parts: list[jax.Array] = []
            s_sc_pair: jax.Array | None = None
            if self.opts.vscale and not self.opts.exact and not self.opts.split32:
                s_bf16_parts = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    s_bf16_parts.append(s_tile.astype(jnp.bfloat16))
                s_pair_bf16 = (
                    jnp.concat(s_bf16_parts, axis=1)
                    if len(s_bf16_parts) > 1
                    else s_bf16_parts[0]
                )
                sc_pair_bf16, inv_pair_bf16 = _hc_col_pow2_scale_bf16(
                    s_pair_bf16, 256.0
                )
                s_limbs = self.limbs_bf16(s_pair_bf16 * inv_pair_bf16, "s")
                s_sc_pair = (
                    sc_pair_bf16 if use_bfproj else sc_pair_bf16.astype(jnp.float32)
                )
            elif self.opts.vscale:
                s_scaled = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    sc_v, inv_v = _hc_col_pow2_scale(s_tile, 256.0)
                    s_sc_parts.append(sc_v)
                    s_scaled.append(s_tile * inv_v)
                if (
                    self._defer_limbs
                    and "s" not in self.opts.one_limb
                    and len(s_scaled) > 1
                ):
                    s_pair_bf16 = jnp.concat(
                        [ss.astype(jnp.bfloat16) for ss in s_scaled], axis=1
                    )
                    s_limbs = self.limbs_bf16(s_pair_bf16, "s")
                else:
                    s_limb_parts = [self.limbs(ss, "s") for ss in s_scaled]
                    s_limbs = [
                        parts[0] if len(parts) == 1 else jnp.concat(parts, axis=1)
                        for parts in zip(*s_limb_parts)
                    ]
            elif self._defer_limbs and "s" not in self.opts.one_limb:
                s_bf16_parts = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    if self.s_s != 1.0:
                        s_tile = s_tile * self.s_s
                    s_bf16_parts.append(s_tile.astype(jnp.bfloat16))
                s_pair_bf16 = (
                    jnp.concat(s_bf16_parts, axis=1)
                    if len(s_bf16_parts) > 1
                    else s_bf16_parts[0]
                )
                s_limbs = self.limbs_bf16(s_pair_bf16, "s")
            else:
                s_limbs = []
                for v in pair:
                    s_tile = self._load_state(v, state_prev)
                    s_limbs.append(
                        self.limbs(
                            s_tile * self.s_s if self.s_s != 1.0 else s_tile, "s"
                        )
                    )
                s_limbs = [jnp.concat(parts, axis=1) for parts in zip(*s_limbs)]
            kq_state = self.dot(hs[pair[0]].kq_limbs, s_limbs, self.mm_dims)
            if use_bfproj:
                kq_bf16: jax.Array | None = kq_state.astype(jnp.bfloat16)
            else:
                kq_bf16 = None
            v_map = (
                self._get_v(list(pair), v_large) if hs[pair[0]].v_s is None else None
            )
            p_kq = pair[0] // vpk
            if (
                self.opts.ggate
                and self._gf_t8_2d is not None
                and all(v // vpk == p_kq for v in pair)
                and p_kq != cur_gf_kq
            ):
                cur_gf_kq = p_kq
                h0_kq = p_kq * vpk
                gf_kq = self._t2d_col_slice(
                    self._gf_t8_2d, h0_kq, min(vpk, self.num_heads - h0_kq)
                )
            for idx, v in enumerate(pair):
                lanes = slice(idx * v_dim, (idx + 1) * v_dim)
                # k_state and q_state are 256 s [K; Q] S with s the state scale; the
                # 2^-8 is in gating_forward8, the rest folds into its per-head
                # copies.
                gf_v = (
                    gf_kq[v - cur_gf_kq * vpk]
                    if (gf_kq is not None and v // vpk == cur_gf_kq)
                    else self.gating_forward(v)
                )
                gf_q = gf_v * (1.0 / self.s_s) if self.s_s != 1.0 else gf_v
                if self.s_b == 1.0:
                    gf_k = gf_q
                elif self.s_s != self.s_b:
                    gf_k = gf_v * (self.s_b / self.s_s)
                else:
                    gf_k = gf_v
                hs[v].kq_limbs = []
                if kq_bf16 is not None:
                    v_src = hs[v].v_s
                    hs[v].v_s = None
                    if v_src is None:
                        assert v_map is not None
                        v_b = (
                            (v_map[v].astype(jnp.float32) * self.s_b).astype(
                                jnp.bfloat16
                            )
                            if self.s_b != 1.0
                            else v_map[v].astype(jnp.bfloat16)
                        )
                    else:
                        v_b = v_src.astype(jnp.bfloat16)
                    k_st_b = kq_bf16[:chunk, lanes]
                    q_st_b = kq_bf16[chunk:, lanes]
                    gf_k_b = gf_k.astype(jnp.bfloat16)
                    gf_q_b = gf_k_b if gf_k is gf_q else gf_q.astype(jnp.bfloat16)
                    if s_sc_pair is not None or s_sc_parts:
                        sc_b = (
                            s_sc_pair[:, lanes]
                            if s_sc_pair is not None
                            else s_sc_parts[idx].astype(jnp.bfloat16)
                        )
                        if gf_k is gf_q:
                            gf_sc_b = gf_k_b * sc_b
                            k_proj_b = k_st_b * gf_sc_b
                            out_upd = q_st_b * gf_sc_b
                        else:
                            k_proj_b = (k_st_b * gf_k_b) * sc_b
                            out_upd = (q_st_b * gf_q_b) * sc_b
                    else:
                        k_proj_b = k_st_b * gf_k_b
                        out_upd = q_st_b * gf_q_b
                    vd_bf16 = v_b - k_proj_b
                    if self._dyn_scale:
                        v_col_scale, v_inv_bf16 = _hc_col_pow2_scale_bf16(
                            vd_bf16, self._vd_cutoff
                        )
                        hs[v].v_diff_s = vd_bf16 * v_inv_bf16
                        hs[v].v_scale = v_col_scale
                    else:
                        hs[v].v_diff_s = vd_bf16
                else:
                    k_state = kq_state[:chunk, lanes]
                    q_state = kq_state[chunk:, lanes]
                    v_s = hs[v].v_s
                    if v_s is None:
                        assert v_map is not None
                        v_s = v_map[v].astype(jnp.float32)
                        if self.s_b != 1.0:
                            v_s = v_s * self.s_b
                    hs[v].v_s = None
                    if s_sc_pair is not None or s_sc_parts:
                        sc_v = (
                            s_sc_pair[:, lanes].astype(jnp.float32)
                            if s_sc_pair is not None
                            else s_sc_parts[idx]
                        )
                        if gf_k is gf_q:
                            gf_sc = gf_k * sc_v
                            k_proj = k_state * gf_sc
                            out_upd = q_state * gf_sc
                        else:
                            k_proj = (k_state * gf_k) * sc_v
                            out_upd = (q_state * gf_q) * sc_v
                    else:
                        k_proj = k_state * gf_k
                        out_upd = q_state * gf_q
                    v_diff_s = v_s - k_proj
                    if self._dyn_scale:
                        if self._use_vdbf:
                            vd_bf16 = v_diff_s.astype(jnp.bfloat16)
                            v_col_scale, v_inv_bf16 = _hc_col_pow2_scale_bf16(
                                vd_bf16, self._vd_cutoff
                            )
                            hs[v].v_diff_s = vd_bf16 * v_inv_bf16
                        else:
                            v_col_scale, v_col_inv = _hc_col_pow2_scale(
                                v_diff_s, self._vd_cutoff
                            )
                            hs[v].v_diff_s = v_diff_s * v_col_inv
                        hs[v].v_scale = v_col_scale
                    else:
                        hs[v].v_diff_s = v_diff_s
                if (
                    self.opts.bfop
                    and not self.opts.exact
                    and out_upd.dtype != jnp.bfloat16
                ):
                    hs[v].out_updated = out_upd.astype(jnp.bfloat16)
                else:
                    hs[v].out_updated = out_upd
        del gf_kq

    def tail(
        self,
        hs: dict[int, _HcHead],
        state_prev: jax.Array | None,
        outs: dict[int, jax.Array],
        states: dict[int, jax.Array],
    ) -> None:
        """Solve chain and the state / output products of every head in `hs`."""
        with jax.named_scope("hc_solve_and_state_out_tail"):
            self._tail_impl(hs, state_prev, outs, states)

    def _tail_impl(
        self,
        hs: dict[int, _HcHead],
        state_prev: jax.Array | None,
        outs: dict[int, jax.Array],
        states: dict[int, jax.Array],
    ) -> None:
        heads = list(hs)
        block, num_blocks = self.block, self.num_blocks
        mm_dims, inv_scale, s_m = self.mm_dims, self.inv_scale, self.s_m
        pre_u_limbs: dict[int, list[jax.Array]] = {}
        t2b_u_pending: (
            tuple[
                dict[int, jax.Array],
                dict[int, list[jax.Array]],
                dict[int, jax.Array],
                bool,
                float,
            ]
            | None
        ) = None
        if self.tb:
            tb = self.tb
            num_tb = self.chunk // tb
            inv_sd = self.inv_sd
            ly_inv = self._ly_inv
            skip_usd = self._fold_usd or inv_sd == 1.0
            do_clamp = self._dyn_scale and not self.opts.noclamp
            if num_tb == 1:
                du_map: dict[int, jax.Array] = {}
                for v in heads:
                    h = hs[v]
                    assert h.v_diff_s is not None
                    vd = h.v_diff_s
                    b_l = (
                        self.limbs_bf16(vd, "b")
                        if vd.dtype == jnp.bfloat16
                        else self.limbs(vd, "b")
                    )
                    if self._msolve:
                        h.v_diff_s = None
                    du_map[v] = self.dot(h.d_blocks[0], b_l, mm_dims)
                for v in heads:
                    h = hs[v]
                    du_v = du_map[v] if skip_usd else du_map[v] * inv_sd
                    if self._msolve:
                        h.x_blocks.append(du_v)
                    else:
                        assert h.v_diff_s is not None
                        h.x_blocks.append(h.v_diff_s.astype(jnp.float32) + du_v)
                        h.v_diff_s = None
                    h.d_blocks = []
            elif (tb >= 128 or (self.opts.t2b and tb == 64)) and num_tb == 2:
                packed_d = tb < 128
                v0_map: dict[int, jax.Array] = {}
                du0_map: dict[int, jax.Array] = {}
                for v in heads:
                    h = hs[v]
                    assert h.v_diff_s is not None
                    vd = h.v_diff_s
                    v0 = vd[:tb, :]
                    h.v_diff_s = vd[tb:, :]
                    if not self._msolve:
                        v0_map[v] = v0
                    d0_l = (
                        [d[:, :tb] for d in h.d_blocks[0]]
                        if packed_d
                        else h.d_blocks[0]
                    )
                    b0_l = (
                        self.limbs_bf16(v0, "b")
                        if v0.dtype == jnp.bfloat16
                        else self.limbs(v0, "b")
                    )
                    du0_map[v] = self.dot(d0_l, b0_l, mm_dims)
                u0_map: dict[int, list[jax.Array]] = {}
                ly0_map: dict[int, jax.Array] = {}
                for v in heads:
                    h = hs[v]
                    du0_raw = du0_map.pop(v)
                    du0_s = du0_raw if skip_usd else du0_raw * inv_sd
                    u0 = (
                        du0_s
                        if self._msolve
                        else v0_map.pop(v).astype(jnp.float32) + du0_s
                    )
                    u0_l = self.limbs(u0, "u", allow_tlimb=not self.opts.lu1)
                    u0_map[v] = u0_l
                    ly0_map[v] = self.dot(
                        h.lw_blocks[0], u0_l[:1] if self.opts.lu1 else u0_l, mm_dims
                    )
                    h.lw_blocks = []
                del v0_map, du0_map
                y1_map: dict[int, jax.Array] = {}
                du1_map: dict[int, jax.Array] = {}
                for v in heads:
                    h = hs[v]
                    assert h.v_diff_s is not None
                    v1 = h.v_diff_s
                    h.v_diff_s = None
                    ly0_s = ly0_map.pop(v) * ly_inv
                    if (
                        self._msolve
                        and self._use_vdbf
                        and v1.dtype == jnp.bfloat16
                        and not do_clamp
                    ):
                        y1_l = self.limbs_bf16(v1 - ly0_s.astype(jnp.bfloat16), "b")
                    else:
                        y1 = v1.astype(jnp.float32) - ly0_s
                        if not self._msolve:
                            y1_map[v] = y1
                        if self._use_vdbf:
                            y1_bf16 = y1.astype(jnp.bfloat16)
                            if do_clamp:
                                y1_bf16 = jnp.clip(
                                    y1_bf16, jnp.bfloat16(-448.0), jnp.bfloat16(448.0)
                                )
                            y1_l = self.limbs_bf16(y1_bf16, "b")
                        else:
                            y1_src = (
                                jnp.clip(y1, jnp.float32(-448.0), jnp.float32(448.0))
                                if (do_clamp and not self.opts.exact)
                                else y1
                            )
                            y1_l = self.limbs(y1_src, "b")
                    if packed_d and self.opts.d1sl:
                        du1_map[v] = self.dot(
                            [d[:, tb:] for d in h.d_blocks[0]], y1_l, mm_dims
                        )
                    elif packed_d:
                        z_tb = jnp.zeros((tb, self.v_dim), dtype=y1_l[0].dtype)
                        du1_map[v] = self.dot(
                            h.d_blocks[0],
                            [jnp.concat([z_tb, yl], axis=0) for yl in y1_l],
                            mm_dims,
                        )
                    else:
                        du1_map[v] = self.dot(h.d_blocks[1], y1_l, mm_dims)
                    h.d_blocks = []
                del ly0_map
                t2b_u_pending = (du1_map, u0_map, y1_map, skip_usd, inv_sd)
            else:
                for bi in range(num_tb):
                    if bi > 0:
                        ly_map: dict[int, jax.Array] = {}
                        for v in heads:
                            h = hs[v]
                            u_prev = h.group_limbs
                            z_tail = jnp.zeros(
                                ((num_tb - bi) * tb, self.v_dim),
                                dtype=u_prev[0][0].dtype,
                            )
                            rhs_u = [
                                jnp.concat([ub[t] for ub in u_prev] + [z_tail], axis=0)
                                for t in range(len(u_prev[0]))
                            ]
                            if num_tb == 2:
                                lw_bi = h.lw_blocks[0]
                            else:
                                lw_bi = [
                                    w[(bi - 1) * tb : bi * tb, :]
                                    for w in h.lw_blocks[0]
                                ]
                            ly_map[v] = self.dot(lw_bi, rhs_u, mm_dims)
                    else:
                        ly_map = {}
                    y_map: dict[int, jax.Array] = {}
                    du_map = {}
                    for v in heads:
                        h = hs[v]
                        assert h.v_diff_s is not None
                        vd = h.v_diff_s
                        y_cur = vd[:tb, :]
                        h.v_diff_s = vd[tb:, :] if bi + 1 < num_tb else None
                        if bi > 0:
                            y_f32 = y_cur.astype(jnp.float32) - ly_map.pop(v) * ly_inv
                            y_map[v] = y_f32
                            if self._use_vdbf:
                                y_bf16 = y_f32.astype(jnp.bfloat16)
                                if do_clamp:
                                    y_bf16 = jnp.clip(
                                        y_bf16,
                                        jnp.bfloat16(-448.0),
                                        jnp.bfloat16(448.0),
                                    )
                                y_l = self.limbs_bf16(y_bf16, "b")
                            else:
                                y_src = (
                                    jnp.clip(
                                        y_f32, jnp.float32(-448.0), jnp.float32(448.0)
                                    )
                                    if (do_clamp and not self.opts.exact)
                                    else y_f32
                                )
                                y_l = self.limbs(y_src, "b")
                        else:
                            y_map[v] = y_cur.astype(jnp.float32)
                            y_l = (
                                self.limbs_bf16(y_cur, "b")
                                if y_cur.dtype == jnp.bfloat16
                                else self.limbs(y_cur, "b")
                            )
                        y_full = []
                        for yl in y_l:
                            parts = []
                            if bi > 0:
                                parts.append(
                                    jnp.zeros((bi * tb, self.v_dim), dtype=yl.dtype)
                                )
                            parts.append(yl)
                            if bi + 1 < num_tb:
                                parts.append(
                                    jnp.zeros(
                                        ((num_tb - 1 - bi) * tb, self.v_dim),
                                        dtype=yl.dtype,
                                    )
                                )
                            y_full.append(jnp.concat(parts, axis=0))
                        du_map[v] = self.dot(h.d_blocks[0], y_full, mm_dims)
                    for v in heads:
                        h = hs[v]
                        du_s = du_map[v] if skip_usd else du_map[v] * inv_sd
                        u_bi = du_s if self._msolve else y_map[v] + du_s
                        h.group_limbs.append(self.limbs(u_bi, "u"))
                        if bi + 1 == num_tb:
                            h.d_blocks = []
                            h.lw_blocks = []
                            h.v_diff_s = None
                for v in heads:
                    u_parts = hs[v].group_limbs
                    hs[v].group_limbs = []
                    pre_u_limbs[v] = [
                        jnp.concat([ub[t] for ub in u_parts], axis=0)
                        for t in range(len(u_parts[0]))
                    ]
        elif self.opts.mchain:
            # u_i = y_i + (M_i - I) @ y_i with y_i = v_diff_i - 2^-8 L~[i, :i] @
            # u_<i: two chained matmuls per block, all heads of the group per step.
            for i in range(num_blocks):
                start = i * block
                for v in heads:
                    h = hs[v]
                    assert h.v_diff_s is not None
                    y_raw = h.v_diff_s[start : start + block, :]
                    if i > 0:
                        y_f32 = y_raw.astype(jnp.float32)
                        rhs, contraction = self.chain_rhs(h)
                        if self._unscale_lw:
                            r0 = start - block
                            lhs = [w[r0 : r0 + block, :contraction] for w in h.l_w]
                            y = y_f32 - self.dot(lhs, rhs, mm_dims)
                        else:
                            lhs = [
                                w[start : start + block, :contraction] for w in h.l_w
                            ]
                            y = y_f32 - self.dot(lhs, rhs, mm_dims) * inv_scale
                        if self._use_vdbf:
                            y_bf16 = y.astype(jnp.bfloat16)
                            if self._dyn_scale and not self.opts.noclamp:
                                y_bf16 = jnp.clip(
                                    y_bf16, jnp.bfloat16(-448.0), jnp.bfloat16(448.0)
                                )
                            y_l = self.limbs_bf16(y_bf16, "y")
                        else:
                            y_l = self.limbs(y, "y")
                    else:
                        y = y_raw.astype(jnp.float32)
                        y_l = (
                            self.limbs_bf16(y_raw, "y")
                            if y_raw.dtype == jnp.bfloat16
                            else self.limbs(y, "y")
                        )
                    d_i = [d[:, start : start + block] for d in h.d_limbs]
                    dy = self.dot(d_i, y_l, mm_dims)
                    u_blk = y + dy * self.inv_s_m if s_m != 1.0 else y + dy
                    h.x_blocks.append(u_blk)
        else:
            # Preconditioner: s_m (M - I) tiled to 32-row pieces, converted once,
            # expanded to the block-diagonal [chunk, chunk] LHS with a constant
            # mask. W~ = 256 (L + (M - I) L) (block row i, columns < i B; the
            # diagonal blocks are never read) and c~ = s_b (v_diff + (M - I)
            # v_diff); L and v_diff are added back in float32.
            reps, tile_rows, chunk = self.reps, self.tile_rows, self.chunk
            fuse_lb = _hc_limb_format("l", self.opts, self.cfg) == _hc_limb_format(
                "b", self.opts, self.cfg
            )
            dcw_map: dict[int, tuple[jax.Array, jax.Array]] = {}
            for v in heads:
                h = hs[v]
                assert h.d_c is not None and h.l_s is not None
                assert h.v_diff_s is not None
                d_tile = jnp.concat([h.d_c] * reps, axis=0) if reps > 1 else h.d_c
                d_full = []
                for limb in self.limbs(d_tile, "m"):
                    full = jnp.concat([limb] * (chunk // tile_rows), axis=0)
                    d_full.append(
                        jnp.where(self.block_diag_mask, full, jnp.zeros_like(full))
                    )
                l_l = self.limbs(h.l_s, "l")
                b_l = (
                    self.limbs_bf16(h.v_diff_s, "b")
                    if h.v_diff_s.dtype == jnp.bfloat16
                    else self.limbs(h.v_diff_s, "b")
                )
                if fuse_lb:
                    rhs_lb = [jnp.concat([ll, bl], axis=1) for ll, bl in zip(l_l, b_l)]
                    dcw = self.dot(d_full, rhs_lb, mm_dims)
                    dcw_map[v] = (dcw[:, :chunk], dcw[:, chunk:])
                else:
                    dcw_map[v] = (
                        self.dot(d_full, l_l, mm_dims),
                        self.dot(d_full, b_l, mm_dims),
                    )
            for v in heads:
                h = hs[v]
                assert h.l_s is not None and h.v_diff_s is not None
                dw, dc = dcw_map[v]
                vd_f32 = h.v_diff_s.astype(jnp.float32)
                if s_m != 1.0:
                    h.w_s = h.l_s + dw * self.inv_s_m
                    h.c_vec = vd_f32 + dc * self.inv_s_m
                else:
                    h.w_s = h.l_s + dw
                    h.c_vec = vd_f32 + dc
                h.l_s = None
                h.d_c = None
                h.x_blocks.append(h.c_vec[:block, :])
            # Chain: u_i = c_i - 2^-8 W~[i, :i] @ u_<i, all heads per step.
            for i in range(1, num_blocks):
                start = i * block
                for v in heads:
                    h = hs[v]
                    assert h.w_s is not None and h.c_vec is not None
                    rhs, contraction = self.chain_rhs(h)
                    lhs = self.limbs(h.w_s[start : start + block, :contraction], "w")
                    prev = self.dot(lhs, rhs, mm_dims)
                    h.x_blocks.append(
                        h.c_vec[start : start + block, :] - prev * inv_scale
                    )
        # [state_new; out_new] = 256 s_b [K^T . beta_gb; s_q Q K^T . D] @ u in
        # one product per head so u is pushed once per limb pair.
        for h_idx, v in enumerate(heads):
            if self._phase_hook is not None:
                self._phase_hook("tail_head", h_idx, len(heads))
            h = hs[v]
            assert h.out_updated is not None
            if t2b_u_pending is not None:
                p_du1, p_u0, p_y1, p_skip_usd, p_inv_sd = t2b_u_pending
                du1_raw = p_du1.pop(v)
                du1_s = du1_raw if p_skip_usd else du1_raw * p_inv_sd
                u1 = du1_s if self._msolve else p_y1.pop(v) + du1_s
                u1_l = self.limbs(u1, "u")
                u_limbs = [
                    jnp.concat([a, b], axis=0) for a, b in zip(p_u0.pop(v), u1_l)
                ]
            elif v in pre_u_limbs:
                u_limbs = pre_u_limbs.pop(v)
            elif self._u_from_bf16:
                u_parts = [g[0] for g in h.group_limbs]
                for xb in h.x_blocks:
                    u_parts.append(xb.astype(jnp.bfloat16))
                h.group_limbs = []
                h.x_blocks = []
                u_limbs = self.limbs_bf16(jnp.concat(u_parts, axis=0), "u")
            else:
                u_limbs = self.limbs(jnp.concat(h.x_blocks, axis=0), "u")
            if h.ko_limbs:
                if (
                    self._defer_limbs
                    and "kg" not in self.opts.one_limb
                    and "oqk" not in self.opts.one_limb
                ):
                    ko_use = self.limbs_bf16(h.ko_limbs[0], "kg")
                else:
                    ko_use = h.ko_limbs
                h.ko_limbs = []
                new = self.dot(ko_use, u_limbs, mm_dims)
                state_new = new[: self.kq_dim]
                out_new = new[self.kq_dim :]
            elif (
                self._defer_limbs
                and not self.opts.exact
                and not self.opts.split32
                and _hc_limb_format("kg", self.opts, self.cfg)
                == _hc_limb_format("oqk", self.opts, self.cfg)
                and "kg" not in self.opts.one_limb
                and "oqk" not in self.opts.one_limb
            ):
                if h.kg_limbs:
                    kg_bf16 = h.kg_limbs[0]
                    h.kg_limbs = []
                else:
                    assert h.k_t_s is not None
                    beta_gb_v = self._beta_gb(v)
                    kg_bf16 = (
                        h.k_t_s * beta_gb_v.astype(jnp.bfloat16)
                        if h.k_t_s.dtype == jnp.bfloat16
                        else (h.k_t_s * beta_gb_v).astype(jnp.bfloat16)
                    )
                    h.k_t_s = None
                oqk_bf16 = h.oqk_limbs[0]
                h.oqk_limbs = []
                ko_bf16 = jnp.concat([kg_bf16, oqk_bf16], axis=0)
                new = self.dot(self.limbs_bf16(ko_bf16, "kg"), u_limbs, mm_dims)
                state_new = new[: self.kq_dim]
                out_new = new[self.kq_dim :]
            else:
                kg_limbs = h.kg_limbs
                h.kg_limbs = []
                if not kg_limbs:
                    assert h.k_t_s is not None
                    beta_gb_v = self._beta_gb(v)
                    if h.k_t_s.dtype == jnp.bfloat16:
                        kg_limbs = self.limbs_bf16(
                            h.k_t_s * beta_gb_v.astype(jnp.bfloat16), "kg"
                        )
                    else:
                        kg_limbs = self.limbs(h.k_t_s * beta_gb_v, "kg")
                    h.k_t_s = None
                elif self._defer_limbs and "kg" not in self.opts.one_limb:
                    kg_limbs = self.limbs_bf16(kg_limbs[0], "kg")
                oqk_limbs = (
                    self.limbs_bf16(h.oqk_limbs[0], "oqk")
                    if self._defer_limbs and "oqk" not in self.opts.one_limb
                    else h.oqk_limbs
                )
                h.oqk_limbs = []
                if len(kg_limbs) == len(oqk_limbs):
                    lhs_limbs = [
                        jnp.concat([a, b], axis=0) for a, b in zip(kg_limbs, oqk_limbs)
                    ]
                    new = self.dot(lhs_limbs, u_limbs, mm_dims)
                    state_new = new[: self.kq_dim]
                    out_new = new[self.kq_dim :]
                else:
                    state_new = self.dot(kg_limbs, u_limbs, mm_dims)
                    out_new = self.dot(oqk_limbs, u_limbs, mm_dims)
            v_sc = h.v_scale
            h.v_scale = None
            if v_sc is not None:
                s_mult: jax.Array | float = v_sc.astype(jnp.float32) * jnp.float32(
                    self.inv_state_new
                )
            else:
                s_mult = self.inv_state_new
            s_prev_v = self._load_state(v, state_prev)
            s_new = s_prev_v * self.gating_last(v) + state_new * s_mult
            if self._store_state_v is not None:
                self._store_state_v(v, s_new)
            else:
                states[v] = s_new
            out_raw = h.out_updated
            h.out_updated = None
            if out_raw.dtype == jnp.bfloat16 and self.opts.bfop and not self.opts.exact:
                o_mult_bf16 = (
                    v_sc.astype(jnp.bfloat16) * jnp.bfloat16(self.inv_out_new)
                    if v_sc is not None
                    else jnp.bfloat16(self.inv_out_new)
                )
                out_bf16 = out_raw + out_new.astype(jnp.bfloat16) * o_mult_bf16
                if self._store_out_slice is not None and self._out_dtype is not None:
                    out_v = out_bf16.astype(self._out_dtype)
                else:
                    out_v = out_bf16.astype(jnp.float32)
            else:
                o_mult: jax.Array | float = (
                    v_sc.astype(jnp.float32) * jnp.float32(self.inv_out_new)
                    if v_sc is not None
                    else self.inv_out_new
                )
                out_base = (
                    out_raw.astype(jnp.float32)
                    if out_raw.dtype != jnp.float32
                    else out_raw
                )
                out_v = out_base + out_new * o_mult
                if self._store_out_slice is not None and self._out_dtype is not None:
                    out_v = out_v.astype(self._out_dtype)
            outs[v] = out_v
            if self._store_out_slice is not None:
                step = self._out_step
                while self._out_flushed + step <= self.num_heads and all(
                    (self._out_flushed + r) in outs for r in range(step)
                ):
                    v0 = self._out_flushed
                    blk = jnp.stack(
                        [outs.pop(v0 + r) for r in range(step)], axis=0
                    ).swapaxes(0, 1)
                    self._store_out_slice(v0, blk)
                    self._out_flushed += step


def _chunked_gdn_per_seq_hc(
    q_large: jax.Array | None,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array | None,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array | None,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array | None,  # [1, chunk, lanes]
    beta: jax.Array | None,  # [1, chunk, lanes]
    state_prev: jax.Array | None,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
    opts: _HcOptions,
    load_kq: Any = None,
    load_kq_scaled: Any = None,
    load_v: Any = None,
    load_state_v: Any = None,
    store_state_v: Any = None,
    store_out_slice: Any = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: Any = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Block-Jacobi / Schur-doubling / float8-limb chunked GDN solver."""
    # Same contract as `chunked_gdn_per_seq`. It differs in four ways:
    # 1. Static power-of-two operand scales folded into the gate exponents: K and
    #    Q enter the MXU as 256 K and 256 Q, the decay map carries an extra 2^-8
    #    and beta, so L = K K^T . D and Q K^T . D come out scaled with no
    #    per-element multiplies, and gating_forward carries the 2^-8 that
    #    unscales [K; Q] @ state_prev.
    # 2. The solve of (I + L) u = v_diff inverts B x B diagonal blocks on the VPU
    #    in a compact layout (`_hc_block_inverses`) and finishes either by Schur
    #    block-doubling (`db<n>`), by chained per-block MXU multiplies (`mchain`),
    #    or by preconditioning once with `(M - I) @ [L | v_diff]`.
    # 3. Every MXU operand goes through `_hc_limbs` (one bf16 pass, or one/two
    #    float8 limbs accumulated in the MXU result buffer), with optional dynamic
    #    in-lane column scaling (`vscale`) for outlier-safe checkpoint inputs.
    # 4. Program order is stage-major within head groups, either overlapped across
    #    consecutive groups or sequential per group (`seqg`) so finished groups
    #    retire from registers before the next group loads.
    if cfg.chunk_size % 8:
        raise ValueError("hc modes need a lane-dense chunk (multiple of 8).")
    ctx = _HcChunk(
        cfg,
        opts,
        None if gating_log is None else _hc_g_lanes(gating_log, cfg.chunk_size),
        beta,
        load_kq=load_kq,
        load_kq_scaled=load_kq_scaled,
        load_v=load_v,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )
    groups = _hc_head_groups(cfg, opts)
    outs: dict[int, jax.Array] = {}
    states: dict[int, jax.Array] = {}

    def front(heads: list[int]) -> dict[int, _HcHead]:
        if opts.sfirst:
            return ctx.front_sfirst(q_large, k_large, v_large, heads, state_prev)
        hs = ctx.prep(q_large, k_large, v_large, heads, load_v_now=not opts.dstate)
        ctx.state_part(hs, state_prev, v_large)
        return hs

    if opts.seqg:
        for g_idx, grp in enumerate(groups):
            if phase_hook is not None:
                phase_hook("group_start", g_idx, len(groups))
            hs = front(grp)
            ctx.tail(hs, state_prev, outs, states)
            if phase_hook is not None:
                phase_hook("group_end", g_idx, len(groups))
    elif opts.dstate and not opts.sfirst and len(groups) > 1:
        if phase_hook is not None:
            phase_hook("group_start", 0, len(groups))
        pending = ctx.prep(q_large, k_large, v_large, groups[0], load_v_now=False)
        for g in range(1, len(groups)):
            ctx.state_part(pending, state_prev, v_large)
            if phase_hook is not None:
                phase_hook("group_start", g, len(groups))
            nxt = ctx.prep(q_large, k_large, v_large, groups[g], load_v_now=False)
            ctx.tail(pending, state_prev, outs, states)
            if phase_hook is not None:
                phase_hook("group_end", g - 1, len(groups))
            pending = nxt
        ctx.state_part(pending, state_prev, v_large)
        ctx.tail(pending, state_prev, outs, states)
        if phase_hook is not None:
            phase_hook("group_end", len(groups) - 1, len(groups))
    else:
        if phase_hook is not None:
            phase_hook("group_start", 0, len(groups))
        pending = front(groups[0])
        for g in range(1, len(groups)):
            if phase_hook is not None:
                phase_hook("group_start", g, len(groups))
            nxt = front(groups[g])
            ctx.tail(pending, state_prev, outs, states)
            if phase_hook is not None:
                phase_hook("group_end", g - 1, len(groups))
            pending = nxt
        ctx.tail(pending, state_prev, outs, states)
        if phase_hook is not None:
            phase_hook("group_end", len(groups) - 1, len(groups))
    out = (
        None
        if store_out_slice is not None
        else jnp.stack([outs[v] for v in range(cfg.num_v_heads)], axis=0)
    )
    state = (
        None
        if store_state_v is not None
        else jnp.stack([states[v] for v in range(cfg.num_v_heads)], axis=0)
    )
    return out, state


def hc_qpad_shape(
    cfg: config.GDNConfig, num_lanes: int = 128
) -> tuple[int, int, int, int] | None:
    """Coprime-stride `(1, chunk, 1, ((dim // lanes) | 1) * lanes)` scratchpad."""
    opts = _hc_active(cfg)
    if (
        cfg.mode != config.GDNMode.PER_SEQ
        or opts is None
        or not opts.qpad
        or cfg.use_recurrent
        or cfg.is_kda
        or cfg.seq_tile_size != 1
        or cfg.window_size != 1
        or cfg.chunk_size % 8
        or cfg.dim_size % num_lanes
        or cfg.kq_head_dim % num_lanes
        or cfg.v_head_dim % num_lanes
    ):
        return None
    pad_cols = (cfg.dim_size // num_lanes) | 1
    return 1, cfg.chunk_size, 1, pad_cols * num_lanes


def hc_stream_io(
    cfg: config.GDNConfig, out_itemsize: int, num_lanes: int = 128
) -> tuple[bool, int]:
    """Returns `(use_fstate, out_step)` for streaming state/out VMEM I/O."""
    opts = _hc_active(cfg)
    if (
        cfg.mode != config.GDNMode.PER_SEQ
        or opts is None
        or cfg.use_recurrent
        or cfg.is_kda
        or cfg.seq_tile_size != 1
        or cfg.window_size != 1
    ):
        return False, 0
    packing = 4 // jnp.dtype(cfg.dtypes.recurrent_state).itemsize
    use_fstate = bool(
        opts.fstate
        and (cfg.kq_head_dim // packing) % 8 == 0
        and cfg.v_head_dim % num_lanes == 0
    )
    out_step = 8 * (4 // out_itemsize)
    if not opts.fout or cfg.num_v_heads % out_step:
        out_step = 0
    return use_fstate, out_step


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Args:
      strict_lower: L, the strictly lower-triangular part of the system matrix,
        [num_heads, chunk, chunk]. Its diagonal and upper triangle must be zero;
        the unit diagonal is implicit.
      rhs: Right-hand side [num_heads, chunk, n].
      block_size: Diagonal block size B. Blocks are solved in order; each block
        first removes the coupling to the already-solved rows with one matmul and
        then runs a dense in-block forward substitution.

    Returns:
      x with the dtype of `strict_lower`.
    """
    out_dtype = strict_lower.dtype
    chunk = strict_lower.shape[-1]
    block_size = min(block_size, chunk)
    num_blocks = (chunk + block_size - 1) // block_size
    rhs_f32 = rhs.astype(jnp.float32)

    def local_forward_sub(l_block: jax.Array, b_block: jax.Array) -> jax.Array:
        # Column-oriented (right-looking) substitution over the whole dense
        # [num_heads, B, n] block: after step j, row j is final and the rows below
        # it have column j's contribution removed. Rows <= j are untouched because
        # their coefficient in the strictly lower block is zero, so each step is
        # one multiply-subtract over B rows held densely in sublanes. A row-by-row
        # chain costs one vreg op per head per (row, column) pair instead, which
        # made the solve VALU-bound at 8- and 16-row blocks (few V heads per core).
        x = b_block
        for j in range(l_block.shape[1] - 1):
            x = x - l_block[:, :, j : j + 1] * x[:, j : j + 1, :]
        return x

    x_blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, chunk)
        target_b = rhs_f32[:, start:end, :]
        if i > 0:
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jax.lax.dot(
                strict_lower[:, start:end, :start],
                solved_x,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            target_b = target_b - prev_sum

        l_block = strict_lower[:, start:end, start:end].astype(jnp.float32)
        x_blocks.append(local_forward_sub(l_block, target_b).astype(out_dtype))

    return jnp.concatenate(x_blocks, axis=1)


def _pack_v_head_pairs(x: jax.Array, v_per_kq_head: int) -> jax.Array:
    """Packs the two V-head halves of each KQ head group along the lane axis."""
    half_v = v_per_kq_head // 2
    group_starts = range(0, x.shape[0], v_per_kq_head)
    first_half = jnp.concat([x[s : s + half_v] for s in group_starts], axis=0)
    second_half = jnp.concat(
        [x[s + half_v : s + v_per_kq_head] for s in group_starts], axis=0
    )
    return jnp.concat([first_half, second_half], axis=-1)


def _unpack_v_head_pairs(x_pairs: jax.Array, v_per_kq_head: int) -> jax.Array:
    """Unpacks V-head pairs from the lane axis back to the head axis."""
    half_v = v_per_kq_head // 2
    first_half, second_half = jnp.split(x_pairs, 2, axis=-1)
    heads = []
    for s in range(0, x_pairs.shape[0], half_v):
        heads += [first_half[s : s + half_v], second_half[s : s + half_v]]
    return jnp.concat(heads, axis=0)


def _dot_kq_with_v_heads(
    kq_operand: jax.Array,  # [num_kq_heads, ...]
    v_operand: jax.Array,  # [num_v_heads, ..., v_head_dim]
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> jax.Array:
    """Contracts a per-KQ-head operand with a per-V-head operand.

    An even number of V heads per KQ head is packed in pairs along the lane
    axis, so that each matmul fills 2 * v_head_dim MXU columns.
    """
    v_per_kq_head = v_operand.shape[0] // kq_operand.shape[0]
    pair_v_heads = v_per_kq_head % 2 == 0 and v_operand.shape[-1] % 128 == 0
    kq_repeats = v_per_kq_head // 2 if pair_v_heads else v_per_kq_head
    kq_operand = jnp.repeat(kq_operand, kq_repeats, axis=0)
    if pair_v_heads:
        v_operand = _pack_v_head_pairs(v_operand, v_per_kq_head)
    out = jax.lax.dot(
        kq_operand,
        v_operand,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    if pair_v_heads:
        out = _unpack_v_head_pairs(out, v_per_kq_head)
    return out


def _hc_active(cfg: config.GDNConfig) -> _HcOptions | None:
    """Returns `_HcOptions` when `cfg` has a multiple-of-block `hc:` chunk."""
    opts = _hc_options(cfg)
    if opts is None or cfg.chunk_size % 32 or cfg.chunk_size % opts.block:
        return None
    return opts


def chunked_gdn_per_seq(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, 1, num_v_heads]
    beta: jax.Array,  # [1, 1, num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [num_heads, chunk, head_dim]."""
    hc_opts = _hc_active(cfg)
    if hc_opts is not None:
        out, state = _chunked_gdn_per_seq_hc(
            q_large, k_large, v_large, gating_log, beta, state_prev, cfg, hc_opts
        )
        assert out is not None and state is not None
        return out, state

    with jax.named_scope("gdn_prefill_decay_maps"):
        # Compute cumulative sum of decay.
        # [1, 1, num_v_heads]
        g_cum_sum_list = [gating_log[:, :1]]
        for row in range(1, cfg.chunk_size):
            g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[:, row : row + 1])
        # [1, chunk, num_v_heads]
        g_cum_sum_log = jnp.concat(g_cum_sum_list, axis=1)

        # [num_v_heads, chunk, 1]
        g_cum_sum_log = fused_transpose_broadcast(g_cum_sum_log, src_dim=2, dst_dim=0)
        g_cum_sum_log = g_cum_sum_log[: cfg.num_v_heads]
        beta = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)
        beta_large = beta[: cfg.num_v_heads]

        # [num_v_heads, 1, chunk]
        g_cum_sum_log_t = fused_transpose_broadcast(g_cum_sum_log, src_dim=1, dst_dim=2)
        beta_t = fused_transpose_broadcast(beta_large, src_dim=1, dst_dim=2)
        # [num_v_heads, chunk, chunk]
        g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t
        gating_map = jnp.exp(g_cum_sum_diff_log)
        # [num_v_heads, chunk, 1]
        gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])
        # [num_v_heads, chunk, 1]
        gating_forward = jnp.exp(g_cum_sum_log)
        # [num_v_heads, 1, 1]
        gating_last = gating_forward[:, -1:]

        mask_dtype = get_mask_dtype(cfg.dtypes.compute)
        iota_r = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 1)
        iota_c = jax.lax.broadcasted_iota(mask_dtype, gating_map.shape, 2)
        strictly_lower_mask = iota_r > iota_c
        lower_mask = iota_r >= iota_c
        # [num_v_heads, chunk, chunk]
        gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0)

    # Contract K @ K^T once per KQ head before repeating across V heads.
    # [num_kq_heads, chunk, chunk]
    with jax.named_scope("gdn_prefill_kkt_matmul"):
        k_k_t = jax.lax.dot(
            k_large,
            k_large,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtypes.compute)

    with jax.named_scope("gdn_prefill_strict_lower_L"):
        # [num_v_heads, chunk, chunk]
        beta_k_k_t = jnp.repeat(k_k_t, cfg.v_per_kq_head, axis=0) * beta_t
        # Strictly lower-triangular part L of the column-scaled system T = I + L;
        # the solve below takes L directly, so T is never materialized.
        gating_beta_k_k_t = gating_map_masked * beta_k_k_t

    # Contract unscaled [K; Q] against state_prev before the triangular solve so
    # the solve operates on the single v_head_dim-wide residual v_diff.
    with jax.named_scope("gdn_prefill_kq_state_proj"):
        # [num_kq_heads, 2 * chunk, kq_head_dim]
        merged_kq = jnp.concat([k_large, q_large], axis=1)
        # [num_v_heads, 2 * chunk, v_head_dim]
        kq_state = _dot_kq_with_v_heads(
            merged_kq, state_prev, (((2,), (1,)), ((0,), (0,)))
        )

        # [num_v_heads, chunk, v_head_dim]
        k_state, q_state = jnp.split(kq_state, 2, axis=1)
        out_updated = q_state * gating_forward

    # Form unscaled innovation residual v_diff = V - gf * (K @ S_prev) in fp32,
    # solve the column-scaled system (I + L) @ x = v_diff directly without
    # materializing T^-1, and post-multiply by beta_large.
    with jax.named_scope("gdn_prefill_v_diff_residual"):
        # [num_v_heads, chunk, v_head_dim]
        v_diff = v_large - k_state * gating_forward

    with jax.named_scope("gdn_prefill_triangular_solve"):
        u_ws = (
            solve_triangular_rhs(
                gating_beta_k_k_t, v_diff, block_size=cfg.triangular_block_size
            )
            * beta_large
        )

    # Absorb gating_backward into u_ws so the unscaled K (transposed once per KQ
    # head) is shared by all of its V heads.
    with jax.named_scope("gdn_prefill_state_update"):
        # [num_v_heads, chunk, v_head_dim]
        u_ws_gated = u_ws * gating_backward
        # [num_v_heads, kq_head_dim, v_head_dim]
        state_new = _dot_kq_with_v_heads(
            k_large, u_ws_gated, (((1,), (1,)), ((0,), (0,)))
        )

        # [num_v_heads, kq_head_dim, v_head_dim]
        state_updated = state_prev * gating_last
        state = state_updated + state_new

    with jax.named_scope("gdn_prefill_qk_intra_map"):
        # [num_kq_heads, chunk, chunk]
        out_qk = jax.lax.dot(
            q_large,
            k_large,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        ).astype(cfg.dtypes.compute)
        # NOTE: must perform repeat after matmul to reduce required compute.
        # [num_v_heads, chunk, chunk]
        out_qk = jnp.repeat(out_qk, cfg.v_per_kq_head, axis=0)
        out_qk *= gating_map
        out_qk = jnp.where(lower_mask, out_qk, 0)

    with jax.named_scope("gdn_prefill_intra_inter_out"):
        # [num_v_heads, chunk, v_head_dim]
        out_new = jax.lax.dot(
            out_qk,
            u_ws,
            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        out = out_updated + out_new

    return out, state


def compute_gates(
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> jax.Array:
    """Computes log-space decay gate `gating_log` from `a`, `a_log`, and `dt_bias`."""
    gate_dtype = jnp.float32 if cfg.use_f32_gates else cfg.dtypes.compute
    a = a.astype(gate_dtype)
    a_log = a_log.astype(gate_dtype)
    if cfg.use_dt_bias and dt_bias is not None:
        a = a + dt_bias.astype(gate_dtype)
    gating_log = -jnp.exp(a_log) * jax.nn.softplus(a)
    if cfg.is_kda and cfg.gate_lower_bound is not None:
        gating_log = jnp.maximum(gating_log, cfg.gate_lower_bound)
    return gating_log.astype(cfg.dtypes.compute)


def chunked_gdn_prepare(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Element-wise preamble of `chunked_gdn`: masks, norms, scale and gates."""
    # Returns q, k (L2-normalized, q scaled by kq_head_dim**-0.5), v, gating_log
    # (natural-log decay, 0 on padding) and beta (0 on padding), all in
    # `cfg.dtypes.compute`, with the padding rows of each sequence tile zeroed.
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota = jax.lax.broadcasted_iota(
        mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
    )
    mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

    # [seqs, num_kq_heads, chunk, kq_head_dim]
    q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
    k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
    # [seqs, num_v_heads, chunk, v_head_dim]
    v_large = jnp.where(mask, v_large.astype(cfg.dtypes.compute), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)

    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    if cfg.use_dt_bias and dt_bias is not None:
        dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

    # NOTE: Any element-wise computations should occur before repeat.
    opts = _hc_active(cfg)
    q_scale = cfg.kq_head_dim**-0.5
    if opts is not None and opts.fnorm and cfg.chunk_size % 8 == 0:
        q_large = _l2_norm_fast(q_large, out_scale=q_scale)
        k_large = _l2_norm_fast(k_large)
    elif opts is not None and opts.rfnorm:
        qf = q_large.astype(jnp.float32)
        kf = k_large.astype(jnp.float32)
        inv_q = jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
        inv_k = jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
        pre_s = _HC_SCALE if opts.rf256 else 1.0
        q_large = (qf * (inv_q * jnp.float32(q_scale * pre_s))).astype(q_large.dtype)
        k_mul = inv_k * jnp.float32(pre_s) if opts.rf256 else inv_k
        k_large = (kf * k_mul).astype(k_large.dtype)
    else:
        q_large = l2_norm(q_large)
        q_large *= q_scale
        k_large = l2_norm(k_large)

    # [seqs, 1, chunk, num_v_heads]
    beta = jax.nn.sigmoid(b_large)
    gating_log = compute_gates(a_large, a_log, dt_bias, cfg)

    beta = jnp.where(mask, beta, 0)
    # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
    # be masked to 1 since it signifies that strength of state from previous row
    # will be 1 (i.e., no decay) if current row is invalid.
    gating_log = jnp.where(mask, gating_log, 0)
    return q_large, k_large, v_large, gating_log, beta


def chunked_gdn(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform chunked GDN over input [seq, num_heads, chunk, head_dim]."""
    # The returned state is [seq, 1, num_v_heads, kq_head_dim, v_head_dim]: the
    # chunked path only produces the final state, so it is never taken when more
    # than one checkpoint per sequence is required.

    assert not cfg.use_recurrent
    q_large, k_large, v_large, gating_log, beta = chunked_gdn_prepare(
        real_sizes,
        q_large,
        k_large,
        v_large,
        b_large,
        a_large,
        a_log,
        dt_bias,
        cfg,
    )

    out_list = []
    state_list = []
    for idx in range(cfg.seq_tile_size):
        out, state = chunked_gdn_per_seq(
            q_large[idx],
            k_large[idx],
            v_large[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out.swapaxes(0, 1))
        state_list.append(state)
    out = jnp.stack(out_list, axis=0)
    # Single checkpoint per sequence, laid out like the recurrent path.
    state = jnp.stack(state_list, axis=0)[:, jnp.newaxis]
    return out, state


def chunked_gdn_stream(
    real_sizes: jax.Array,
    q_large: jax.Array,
    k_large: jax.Array,
    v_large: jax.Array,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
    load_state_v: Any = None,
    store_state_v: Any = None,
    store_out_slice: Any = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: Any = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """`chunked_gdn` with streaming state/output I/O (`fstate`/`fout`)."""
    assert not cfg.use_recurrent and cfg.seq_tile_size == 1
    opts = _hc_active(cfg)
    assert opts is not None
    q_large, k_large, v_large, gating_log, beta = chunked_gdn_prepare(
        real_sizes,
        q_large,
        k_large,
        v_large,
        b_large,
        a_large,
        a_log,
        dt_bias,
        cfg,
    )
    out, state = _chunked_gdn_per_seq_hc(
        q_large[0],
        k_large[0],
        v_large[0],
        gating_log[0],
        beta[0],
        None if state_prev is None else state_prev[0],
        cfg,
        opts,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )
    out_ret = None if out is None else out.swapaxes(0, 1)[jnp.newaxis]
    state_ret = None if state is None else state[jnp.newaxis, jnp.newaxis]
    return out_ret, state_ret


def chunked_gdn_lazy(
    real_sizes: jax.Array,
    load_raw_kq: Any,
    load_raw_v: Any,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array | None,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
    load_state_v: Any = None,
    store_state_v: Any = None,
    store_out_slice: Any = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: Any = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """`chunked_gdn` loading Q, K, and V per `hc` head group (`qpad`)."""
    assert not cfg.use_recurrent and cfg.seq_tile_size == 1
    opts = _hc_active(cfg)
    assert opts is not None
    compute = cfg.dtypes.compute
    mask_dtype = get_mask_dtype(compute)
    iota = jax.lax.broadcasted_iota(mask_dtype, (cfg.chunk_size, 1), 0)
    mask_2d = iota < real_sizes[0].astype(mask_dtype)
    mask_3d = mask_2d[jnp.newaxis, :, :]
    q_scale = cfg.kq_head_dim**-0.5

    def load_kq(
        kqs: list[int],
    ) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
        qs_raw, ks_raw = load_raw_kq(kqs)
        qs: dict[int, jax.Array] = {}
        ks: dict[int, jax.Array] = {}
        for kq in kqs:
            qh = jnp.where(mask_2d, qs_raw[kq].astype(compute), 0)
            kh = jnp.where(mask_2d, ks_raw[kq].astype(compute), 0)
            qs[kq] = l2_norm(qh) * q_scale
            ks[kq] = l2_norm(kh)
        return qs, ks

    def load_kq_scaled(
        kqs: list[int], scale: float, s_q: float
    ) -> tuple[
        dict[int, jax.Array],
        dict[int, jax.Array] | None,
        dict[int, jax.Array],
        dict[int, jax.Array],
    ]:
        qs_raw, ks_raw = load_raw_kq(kqs)
        q_s_map: dict[int, jax.Array] = {}
        q_oqk_map: dict[int, jax.Array] | None = {} if s_q != scale else None
        k_s_map: dict[int, jax.Array] = {}
        kt_s_map: dict[int, jax.Array] = {}
        kt_bf16 = (
            opts.bfop
            and not opts.exact
            and not opts.split32
            and "kg" not in opts.one_limb
        )
        for kq in kqs:
            qh = jnp.where(mask_2d, qs_raw[kq].astype(compute), 0)
            kh = jnp.where(mask_2d, ks_raw[kq].astype(compute), 0)
            if opts.fnorm and cfg.chunk_size % 8 == 0:
                qt_s, _ = _l2_norm_scaled_t(qh, out_scale=q_scale * scale)
                q_s = jnp.swapaxes(qt_s, 0, 1)
                kt_raw, _ = _l2_norm_scaled_t(kh, out_scale=scale)
                k_s = jnp.swapaxes(kt_raw, 0, 1)
                kt_s = kt_raw.astype(jnp.bfloat16) if kt_bf16 else kt_raw
            else:
                qf = qh.astype(jnp.float32)
                kf = kh.astype(jnp.float32)
                inv_q = jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
                inv_k = jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
                q_s = (qf * (inv_q * jnp.float32(q_scale * scale))).astype(compute)
                k_s = (kf * (inv_k * jnp.float32(scale))).astype(compute)
                kt_s = jnp.swapaxes(k_s.astype(jnp.bfloat16) if kt_bf16 else k_s, 0, 1)
            q_s_map[kq] = q_s
            if q_oqk_map is not None:
                q_oqk_map[kq] = q_s * jnp.float32(s_q / scale)
            kt_s_map[kq] = kt_s
            k_s_map[kq] = k_s
        return q_s_map, q_oqk_map, k_s_map, kt_s_map

    def load_v(vs: list[int]) -> dict[int, jax.Array]:
        vs_raw = load_raw_v(vs)
        return {v: jnp.where(mask_2d, vs_raw[v].astype(compute), 0) for v in vs}

    b_seq = b_large[0].astype(compute)
    a_seq = a_large[0].astype(compute)
    a_log = a_log.reshape(1, 1, -1).astype(compute)
    if cfg.use_dt_bias and dt_bias is not None:
        dt_bias = dt_bias.reshape(1, 1, -1).astype(compute)
    beta = jnp.where(mask_3d, jax.nn.sigmoid(b_seq), 0)
    gating_log = jnp.where(mask_3d, compute_gates(a_seq, a_log, dt_bias, cfg), 0)
    out, state = _chunked_gdn_per_seq_hc(
        None,
        None,
        None,
        gating_log,
        beta,
        None if state_prev is None else state_prev[0],
        cfg,
        opts,
        load_kq=None if (opts.fnorm or opts.rfnorm) else load_kq,
        load_kq_scaled=load_kq_scaled if (opts.fnorm or opts.rfnorm) else None,
        load_v=load_v,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )
    out_ret = None if out is None else out.swapaxes(0, 1)[jnp.newaxis]
    state_ret = None if state is None else state[jnp.newaxis, jnp.newaxis]
    return out_ret, state_ret


def run_hc_chunk_step(
    *,
    load_raw_kq: Any,
    load_raw_v: Any,
    cfg: config.GDNConfig,
    b_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    a_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    a_log: jax.Array | None = None,  # [lanes]
    dt_bias: jax.Array | None = None,  # [lanes]
    gating_log_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    beta_lanes: jax.Array | None = None,  # [1, chunk, lanes]
    kq_prescaled_256: bool = False,
    load_state_v: Any = None,
    store_state_v: Any = None,
    store_out_slice: Any = None,
    out_step: int = 0,
    out_dtype: jnp.dtype | None = None,
    phase_hook: Any = None,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Runs one full-chunk `hc` recurrence step with optional phase hook."""
    # Supports both standard inputs (`b_lanes`, `a_lanes`, unnormalized Q/K) and
    # pre-computed prologue inputs (`gating_log_lanes`, `beta_lanes`, and Q/K
    # pre-scaled by `256 * l2_norm` with Q also scaled by `d_k ** -0.5` when
    # `kq_prescaled_256=True`).
    opts = _hc_active(cfg)
    if opts is None:
        raise ValueError(f"Expected active hc mode; got {cfg.gdn_mxu_fp8_mode!r}.")
    compute = cfg.dtypes.compute
    q_scale = cfg.kq_head_dim**-0.5
    kt_bf16 = (
        opts.bfop and not opts.exact and not opts.split32 and "kg" not in opts.one_limb
    )

    def load_kq_std(
        kqs: list[int],
    ) -> tuple[dict[int, jax.Array], dict[int, jax.Array]]:
        qs_raw, ks_raw = load_raw_kq(kqs)
        qs: dict[int, jax.Array] = {}
        ks: dict[int, jax.Array] = {}
        for kq in kqs:
            qh = qs_raw[kq].astype(compute)
            kh = ks_raw[kq].astype(compute)
            if kq_prescaled_256:
                inv256 = jnp.float32(1.0 / _HC_SCALE)
                qs[kq] = (qh.astype(jnp.float32) * inv256).astype(compute)
                ks[kq] = (kh.astype(jnp.float32) * inv256).astype(compute)
            else:
                qs[kq] = l2_norm(qh) * q_scale
                ks[kq] = l2_norm(kh)
        return qs, ks

    def load_kq_scaled(
        kqs: list[int], scale: float, s_q: float
    ) -> tuple[
        dict[int, jax.Array],
        dict[int, jax.Array] | None,
        dict[int, jax.Array],
        dict[int, jax.Array],
    ]:
        qs_raw, ks_raw = load_raw_kq(kqs)
        q_s_map: dict[int, jax.Array] = {}
        q_oqk_map: dict[int, jax.Array] | None = {} if s_q != scale else None
        k_s_map: dict[int, jax.Array] = {}
        kt_s_map: dict[int, jax.Array] = {}
        for kq in kqs:
            if kq_prescaled_256:
                q_f32 = qs_raw[kq].astype(jnp.float32)
                k_f32 = ks_raw[kq].astype(jnp.float32)
                if scale != _HC_SCALE:
                    ratio = jnp.float32(scale / _HC_SCALE)
                    q_f32 = q_f32 * ratio
                    k_f32 = k_f32 * ratio
                q_s = q_f32.astype(compute)
                k_s = k_f32.astype(compute)
            else:
                qf = qs_raw[kq].astype(jnp.float32)
                kf = ks_raw[kq].astype(jnp.float32)
                inv_q = jax.lax.rsqrt(jnp.sum(qf * qf, axis=-1, keepdims=True) + 1e-6)
                inv_k = jax.lax.rsqrt(jnp.sum(kf * kf, axis=-1, keepdims=True) + 1e-6)
                q_s = (qf * (inv_q * jnp.float32(q_scale * scale))).astype(compute)
                k_s = (kf * (inv_k * jnp.float32(scale))).astype(compute)
            kt_s = jnp.swapaxes(k_s.astype(jnp.bfloat16) if kt_bf16 else k_s, 0, 1)
            q_s_map[kq] = q_s
            if q_oqk_map is not None:
                q_oqk_map[kq] = q_s * jnp.float32(s_q / scale)
            kt_s_map[kq] = kt_s
            k_s_map[kq] = k_s
        return q_s_map, q_oqk_map, k_s_map, kt_s_map

    def load_v(vs: list[int]) -> dict[int, jax.Array]:
        vs_raw = load_raw_v(vs)
        return {v: vs_raw[v].astype(compute) for v in vs}

    if gating_log_lanes is not None and beta_lanes is not None:
        gating_log = gating_log_lanes.astype(compute)
        beta = beta_lanes.astype(compute)
    else:
        assert b_lanes is not None and a_lanes is not None
        assert a_log is not None and dt_bias is not None
        b_c = b_lanes.astype(compute)
        a_c = a_lanes.astype(compute)
        al_c = a_log.reshape(1, 1, -1).astype(compute)
        dt_c = dt_bias.reshape(1, 1, -1).astype(compute)
        beta = jax.nn.sigmoid(b_c)
        gating_log = -jnp.exp(al_c) * jax.nn.softplus(a_c + dt_c)

    use_scaled = bool(opts.fnorm or opts.rfnorm or kq_prescaled_256)
    return _chunked_gdn_per_seq_hc(
        None,
        None,
        None,
        gating_log,
        beta,
        None,
        cfg,
        opts,
        load_kq=None if use_scaled else load_kq_std,
        load_kq_scaled=load_kq_scaled if use_scaled else None,
        load_v=load_v,
        load_state_v=load_state_v,
        store_state_v=store_state_v,
        store_out_slice=store_out_slice,
        out_step=out_step,
        out_dtype=out_dtype,
        phase_hook=phase_hook,
    )


def recurrent_gdn_per_seq(
    q_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact: jax.Array,  # [num_kq_heads, chunk, 1, kq_head_dim]
    k_compact_t: jax.Array,  # [num_kq_heads, chunk, kq_head_dim, 1]
    v_compact: jax.Array,  # [num_v_heads, chunk, 1, v_head_dim]
    gating_log: jax.Array,  # [num_v_heads, chunk, 1, 1]
    beta: jax.Array,  # [num_v_heads, chunk, 1, 1]
    state: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [num_heads, chunk, 1, head_dim].

    The returned state stacks the post-token state of the last
    `cfgs.window_size` window positions ([window_size, num_v_heads,
    kq_head_dim, v_head_dim]); with `window_size == 1` that is just the final
    state.
    """

    out_list = []
    state_list = []
    for c_idx in range(cfgs.chunk_size):
        # [num_v_heads, 1, kq_head_dim]
        q_curr = q_compact[:, c_idx]
        q_curr = jnp.repeat(q_curr, cfgs.v_per_kq_head, axis=0)
        k_curr = k_compact[:, c_idx]
        k_curr = jnp.repeat(k_curr, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, v_head_dim]
        v_curr = v_compact[:, c_idx]

        # [num_v_heads, kq_head_dim, 1]
        k_curr_t = k_compact_t[:, c_idx]
        k_curr_t = jnp.repeat(k_curr_t, cfgs.v_per_kq_head, axis=0)

        # [num_v_heads, 1, 1]
        beta_curr = beta[:, c_idx]
        gating_curr = gating_log[:, c_idx]

        # [num_v_heads, kq_head_dim, v_head_dim]
        with jax.named_scope("gdn_decode_state_decay"):
            state_updated = state * gating_curr

        # [num_v_heads, 1, v_head_dim]
        with jax.named_scope("gdn_decode_fused_kq_state_proj"):
            v_updated = jax.lax.dot(
                k_curr,
                state_updated,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.dtypes.compute)

        # [num_v_heads, 1, v_head_dim]
        with jax.named_scope("gdn_decode_delta_innovation"):
            v_diff = v_curr - v_updated
            v_new = beta_curr * v_diff

        # [num_v_heads, kq_head_dim, v_head_dim]
        # NOTE: Multiplication with k_curr_t needs to be deferred as much as
        # possible as it expands the dimension size by kq_head_dim.
        with jax.named_scope("gdn_decode_rank1_state_update"):
            state_new = k_curr_t * v_new
            # [num_v_heads, kq_head_dim, v_head_dim]
            state = state_updated + state_new

            # [num_v_heads, 1, v_head_dim]
            out = jax.lax.dot(
                q_curr,
                state,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.dtypes.compute)

        out_list.append(out[:, 0, :])
        # NOTE: Rows past real_sizes are masked out by the caller, so the state
        # stops changing there and trailing checkpoints simply repeat the state
        # of the last real token. Checkpoints of the positions dropped here are
        # dead code the compiler eliminates.
        if c_idx >= cfgs.chunk_size - cfgs.window_size:
            state_list.append(state)

    return jnp.stack(out_list, axis=0), jnp.stack(state_list, axis=0)


def recurrent_gdn(
    real_sizes: jax.Array,
    q_compact: jax.Array,
    k_compact: jax.Array,
    v_compact: jax.Array,
    b_compact: jax.Array,
    a_compact: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform recurrent GDN over input [seq, num_heads, chunk, 1, head_dim].

    The returned state has one checkpoint per window position: [seq,
    window_size, num_v_heads, kq_head_dim, v_head_dim]. Positions >=
    real_sizes repeat the last valid state; they are never written back to
    HBM.
    """

    with jax.named_scope("gdn_decode_mask_and_norm_qk"):
        mask_dtype = get_mask_dtype(cfg.dtypes.compute)
        iota = jax.lax.broadcasted_iota(
            mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1, 1), 2
        )
        mask = iota < real_sizes.reshape(-1, 1, 1, 1, 1).astype(mask_dtype)

        # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
        q_compact = jnp.where(mask, q_compact.astype(cfg.dtypes.compute), 0)
        k_compact = jnp.where(mask, k_compact.astype(cfg.dtypes.compute), 0)
        # [seqs, num_v_heads, chunk, 1, v_head_dim]
        v_compact = jnp.where(mask, v_compact.astype(cfg.dtypes.compute), 0)

        b_compact = b_compact.astype(cfg.dtypes.compute)
        a_compact = a_compact.astype(cfg.dtypes.compute)

        a_log = a_log.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)
        if cfg.use_dt_bias and dt_bias is not None:
            dt_bias = dt_bias.reshape(1, 1, 1, 1, -1).astype(cfg.dtypes.compute)

        # [seqs, num_kq_heads, chunk, 1, kq_head_dim]
        q_compact = l2_norm(q_compact)
        q_scale = cfg.kq_head_dim**-0.5
        q_compact *= q_scale
        k_compact = l2_norm(k_compact)
        k_compact_t = fused_transpose_broadcast(k_compact, src_dim=4, dst_dim=3)

    with jax.named_scope("gdn_decode_gate_activations"):
        beta = jax.nn.sigmoid(b_compact)
        gating_log = compute_gates(a_compact, a_log, dt_bias, cfg)

        beta = jnp.where(mask, beta, 0)
        # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
        # be masked to 1 since it signifies that strength of state from previous row
        # will be 1 (i.e., no decay) if current row is invalid.
        gating_log = jnp.where(mask, gating_log, 0)
        gating_log = jnp.exp(gating_log)

        beta = fused_transpose_broadcast(beta, src_dim=4, dst_dim=1)
        beta = beta[:, : cfg.num_v_heads]
        gating_log = fused_transpose_broadcast(gating_log, src_dim=4, dst_dim=1)
        gating_log = gating_log[:, : cfg.num_v_heads]

    if cfg.chunk_size == 1 and cfg.window_size == 1:
        # Optimized decode path: fused [k, q] projection against state.
        q_curr = jnp.repeat(q_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
        q_curr = jnp.expand_dims(q_curr, axis=2)

        k_curr = jnp.repeat(k_compact[:, :, 0, 0, :], cfg.v_per_kq_head, axis=1)
        k_curr = jnp.expand_dims(k_curr, axis=2)

        k_curr_t = jnp.repeat(k_compact_t[:, :, 0, :, 0], cfg.v_per_kq_head, axis=1)
        k_curr_t = jnp.expand_dims(k_curr_t, axis=3)

        v_curr = jnp.expand_dims(v_compact[:, :, 0, 0, :], axis=2)
        beta_curr = beta[:, :, 0, 0, 0].reshape(
            cfg.seq_tile_size, cfg.num_v_heads, 1, 1
        )
        gating_curr = gating_log[:, :, 0, 0, 0].reshape(
            cfg.seq_tile_size, cfg.num_v_heads, 1, 1
        )

        # 1. State decay update
        with jax.named_scope("gdn_decode_state_decay"):
            state_updated = state_prev * gating_curr

        # 2. Fused [k, q] projection against state
        with jax.named_scope("gdn_decode_fused_kq_state_proj"):
            b_size, n_v_heads = cfg.seq_tile_size, cfg.num_v_heads
            kq_merged = jnp.concat([k_curr, q_curr], axis=2)  # [B, n_v, 2, d_k]
            kq_merged_flat = kq_merged.reshape(-1, 2, cfg.kq_head_dim)
            state_updated_flat = state_updated.reshape(
                -1, cfg.kq_head_dim, cfg.v_head_dim
            )

            kq_proj_flat = jax.lax.dot_general(
                kq_merged_flat,
                state_updated_flat,
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            ).astype(cfg.dtypes.compute)  # [B * n_v, 2, d_v]
            kq_proj = kq_proj_flat.reshape(b_size, n_v_heads, 2, cfg.v_head_dim)

            v_updated = kq_proj[:, :, :1, :]
            out_decayed = kq_proj[:, :, 1:, :]

        # 3. Output difference
        with jax.named_scope("gdn_decode_delta_innovation"):
            v_diff = v_curr - v_updated
            v_new = beta_curr * v_diff

            qk_dot = jnp.sum(q_curr * k_curr, axis=-1, keepdims=True)  # [B, n_v, 1, 1]
            out = out_decayed + qk_dot * v_new  # [B, n_v, 1, d_v]

        # 4. State rank-1 update
        with jax.named_scope("gdn_decode_rank1_state_update"):
            state_new = k_curr_t * v_new  # [B, n_v, d_k, d_v]
            new_recurrent_state = state_updated + state_new

        # Match Pallas output shape [B, 1, n_v, d_v]:
        out = out.swapaxes(1, 2)
        # Output single checkpoint per sequence, matching [seq, window_size, ...]
        return out, jnp.expand_dims(new_recurrent_state, axis=1)

    out_list = []
    new_state_list = []

    for idx in range(cfg.seq_tile_size):
        out, state = recurrent_gdn_per_seq(
            q_compact[idx],
            k_compact[idx],
            k_compact_t[idx],
            v_compact[idx],
            gating_log[idx],
            beta[idx],
            state_prev[idx],
            cfg,
        )
        out_list.append(out)
        new_state_list.append(state)

    out = jnp.stack(out_list, axis=0)
    new_recurrent_state = jnp.stack(new_state_list, axis=0)

    return out, new_recurrent_state
