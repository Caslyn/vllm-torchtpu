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
"""Chunked and recurrent GDN compute."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.gdn.v3 import config, vmem_ldst


def l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize along the last dim, accumulating in float32."""
    x_f32 = x.astype(jnp.float32)
    norm = jnp.sqrt(jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    return (x_f32 / norm).astype(x.dtype)


def get_mask_dtype(dtype: jnp.dtype) -> jnp.dtype:
    match jnp.dtype(dtype).itemsize:
        case 4:
            return jnp.int32
        case 2:
            return jnp.int16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def solve_triangular_rhs(
    strict_lower: jax.Array, rhs: jax.Array, block_size: int
) -> jax.Array:
    """Solves the unit lower-triangular system (I + L) @ x = rhs in blocks.

    Args:
        strict_lower: L, the strictly lower-triangular part of the system
            matrix, [num_heads, chunk, chunk]. Its diagonal and upper triangle
            must be zero; the unit diagonal is implicit.
        rhs: Right-hand side [num_heads, chunk, n].
        block_size: Diagonal block size B. Blocks are solved in order; each
            block first removes the coupling to the already-solved rows with
            one matmul and then runs a dense in-block forward substitution.

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
        # [num_heads, B, n] block: after step j, row j is final and the rows
        # below it have column j's contribution removed. Rows <= j are
        # untouched because their coefficient in the strictly lower block is
        # zero, so each step is one multiply-subtract over B rows held densely
        # in sublanes. A row-by-row chain costs one vreg op per head per
        # (row, column) pair instead, which made the solve VALU-bound at 8-
        # and 16-row blocks (few V heads per core).
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


def fused_transpose_broadcast(x: jax.Array, src_dim: int, dst_dim: int) -> jax.Array:
    """Perform 1D transpose where results are broadcasted along src_dim."""
    assert x.shape[dst_dim] == 1

    dtype = x.dtype
    mask_dtype = get_mask_dtype(dtype)
    mask_shape = list(x.shape)
    mask_size = mask_shape[src_dim]
    mask_shape[dst_dim] = mask_size
    src_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, src_dim)
    dst_mask = jax.lax.broadcasted_iota(mask_dtype, mask_shape, dst_dim)
    mask = src_mask == dst_mask
    return jnp.where(mask, x, 0).sum(axis=src_dim, keepdims=True, dtype=dtype)


_LOG2E = 1.4426950408889634
_FP8_E4M3_MAX = 448.0


def _two_limb_split(
    x: jax.Array,
    dtype: jnp.dtype,
    *,
    scale_per_row: bool = False,
    axis: int = -1,
    fast_scale: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits `x` into two `dtype` limbs `(hi, lo)` and `scale`."""
    x_f32 = x.astype(jnp.float32)
    fmax = float(jnp.finfo(dtype).max)
    if scale_per_row:
        if fast_scale:
            scale = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True) * (
                1.0 / fmax
            ) + jnp.float32(1e-30)
            inv_scale = 1.0 / scale
        else:
            amax = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True)
            scale = jnp.where(amax > 0.0, amax / fmax, 1.0)
            inv_scale = 1.0 / scale
        x_scaled = x_f32 * inv_scale
    else:
        scale = jnp.ones((), dtype=jnp.float32)
        x_scaled = x_f32
    hi = x_scaled.astype(dtype)
    res = x_scaled - hi.astype(jnp.float32)
    lo = res.astype(dtype)
    return hi, lo, scale


def _dot_two_limb(
    a_hi: jax.Array,
    a_lo: jax.Array,
    a_scale: jax.Array,
    b_hi: jax.Array,
    b_lo: jax.Array,
    b_scale: jax.Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    *,
    num_terms: int = 3,
    b_lo_scale: jax.Array | float = 1.0,
) -> jax.Array:
    """3-matmul (or 2/1-matmul) two-limb dot product accumulating in float32."""
    p_hh = jax.lax.dot(
        a_hi,
        b_hi,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    if num_terms == 1:
        return p_hh * (a_scale * b_scale)
    p_hl = jax.lax.dot(
        a_hi,
        b_lo,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    if b_lo_scale is not None and not (
        isinstance(b_lo_scale, float) and b_lo_scale == 1.0
    ):
        p_hl = p_hl * b_lo_scale
    if num_terms == 2:
        return (p_hh + p_hl) * (a_scale * b_scale)
    p_lh = jax.lax.dot(
        a_lo,
        b_hi,
        dimension_numbers=dimension_numbers,
        preferred_element_type=jnp.float32,
    )
    return (p_hh + p_hl + p_lh) * (a_scale * b_scale)


def _quantize_rows(
    x: jax.Array, dtype: jnp.dtype, axis: int = -1, *, fast: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Per-row dynamic quantization along `axis`, returning `(x_q, scale)`."""
    x_f32 = x.astype(jnp.float32)
    fmax = float(jnp.finfo(dtype).max)
    if fast:
        scale = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True) * (
            1.0 / fmax
        ) + jnp.float32(1e-30)
        return (x_f32 / scale).astype(dtype), scale
    amax = jnp.max(jnp.abs(x_f32), axis=axis, keepdims=True)
    scale = jnp.where(amax > 0.0, amax / fmax, 1.0)
    return (x_f32 / scale).astype(dtype), scale


@dataclasses.dataclass(frozen=True)
class _HcOptions:
    """Parsed options for the `hc:...` chunked GDN experimental mode."""

    base_block: int = 16
    double_to: int = 64
    use_8bit_inv: bool = False
    inv_2limb: bool = False
    inv_dyn_scale: bool = False
    state_2limb: bool = False
    state_dyn_scale: bool = False
    u_2limb: bool = False
    u_dyn_scale: bool = False
    kq_2limb: bool = False
    row_fixed_scale: float = 0.0
    doubling_8bit: bool = False
    doubling_2limb: bool = False
    merged_doubling: bool = False
    direct_kg: bool = False
    rhs_zero_diag: bool = False
    state_1_limb: bool = False
    u_1_limb: bool = False
    k_1_limb: bool = False
    q_1_limb: bool = False
    kq_fixed_scale: float = 0.0
    state_fixed_scale: float = 0.0
    fast_scale: bool = False
    fast_db: bool = False
    two_block_step: bool = False
    state_first: bool = False
    merged_solve: bool = False
    solve_df8: bool = False
    tail_2limb: bool = False
    bf16_proj: bool = False
    bf16_oqk: bool = False
    bf16_l: bool = False
    no_exp_clamp: bool = False
    group_gating: bool = False
    zero_fp8: bool = False
    qpad: bool = False
    fused_out: bool = False
    fused_state: bool = False
    lu_1_limb: bool = False
    v_col_scale: bool = False
    v_diff_bf16: bool = False
    seq_group: bool = False
    subchunk_size: int = 0


def _hc_options(cfg: config.GDNConfig) -> _HcOptions | None:
    mode = cfg.gdn_mxu_fp8_mode
    if not mode.startswith("hc"):
        return None
    spec = mode[3:] if mode.startswith("hc:") else ""
    tags = set(spec.split("+")) if spec else set()
    base_block = 16
    double_to = 64
    row_fixed_scale = 0.0
    kq_fixed_scale = 0.0
    state_fixed_scale = 0.0
    subchunk_size = 0
    for t in tags:
        if t.startswith("bj"):
            base_block = int(t[2:])
        elif t.startswith("db"):
            double_to = int(t[2:])
        elif t.startswith("rf"):
            row_fixed_scale = float(t[2:])
        elif t.startswith("kqs"):
            kq_fixed_scale = float(t[3:])
        elif t.startswith("sq"):
            state_fixed_scale = float(t[2:])
        elif t.startswith("g") and t[1:].isdigit():
            subchunk_size = int(t[1:])
    return _HcOptions(
        base_block=base_block,
        double_to=double_to,
        use_8bit_inv=("f8" in tags or "df8" in tags or "2l" in tags or "d2l" in tags),
        inv_2limb=("2l" in tags or "d2l" in tags),
        inv_dyn_scale=("df8" in tags or "d2l" in tags),
        state_2limb=("s2l" in tags or "sd2l" in tags or "all2l" in tags),
        state_dyn_scale=("sd2l" in tags or "all2l" in tags),
        u_2limb=("u2l" in tags or "ud2l" in tags or "all2l" in tags),
        u_dyn_scale=("ud2l" in tags or "all2l" in tags),
        kq_2limb=("kq2l" in tags or "all2l" in tags),
        row_fixed_scale=row_fixed_scale,
        doubling_8bit=("dbl8" in tags or "dbl2l" in tags),
        doubling_2limb=("dbl2l" in tags),
        merged_doubling=("mdbl" in tags),
        direct_kg=("dkg" in tags),
        rhs_zero_diag=("rhs0" in tags),
        state_1_limb=("s1" in tags),
        u_1_limb=("u1" in tags),
        k_1_limb=("k1" in tags),
        q_1_limb=("q1" in tags),
        kq_fixed_scale=kq_fixed_scale,
        state_fixed_scale=state_fixed_scale,
        fast_scale=("fscale" in tags),
        fast_db=("fastdb" in tags),
        two_block_step=("t2b" in tags),
        state_first=("sfirst" in tags),
        merged_solve=("msolve" in tags),
        solve_df8=("sdf8" in tags),
        tail_2limb=("tlimb" in tags),
        bf16_proj=("bfproj" in tags),
        bf16_oqk=("bfoqk" in tags),
        bf16_l=("bfl" in tags),
        no_exp_clamp=("noclamp" in tags),
        group_gating=("ggate" in tags),
        zero_fp8=("zfp8" in tags),
        qpad=("qpad" in tags),
        fused_out=("fout" in tags),
        fused_state=("fstate" in tags),
        lu_1_limb=("lu1" in tags),
        v_col_scale=("vscale" in tags),
        v_diff_bf16=("vdbf" in tags),
        seq_group=("seqg" in tags),
        subchunk_size=subchunk_size,
    )


def _hc_active(cfg: config.GDNConfig) -> bool:
    return (
        cfg.gdn_mxu_operand_dtype is not None
        and _hc_options(cfg) is not None
        and not cfg.is_kda
    )


def hc_qpad_shape(cfg: config.GDNConfig) -> tuple[int, int] | None:
    """If `hc:...` with `+qpad` is active, returns `(chunk_size, v_head_dim)`."""
    if not _hc_active(cfg):
        return None
    opts = _hc_options(cfg)
    assert opts is not None
    if not opts.qpad or opts.subchunk_size > 0:
        return None
    return (cfg.chunk_size, cfg.v_head_dim)


def hc_stream_io(cfg: config.GDNConfig) -> tuple[bool, bool, bool]:
    """Returns `(stream_enabled, use_fout, use_fstate)` for the `hc:...` path."""
    if not _hc_active(cfg) or cfg.seq_tile_size != 1:
        return (False, False, False)
    opts = _hc_options(cfg)
    assert opts is not None
    if opts.subchunk_size > 0:
        return (False, False, False)
    stream = opts.seq_group
    return (stream, stream and opts.fused_out, stream and opts.fused_state)


def _invert_unit_lower_blocks(
    strict_lower: jax.Array, block_size: int, *, zero_diag: bool = False
) -> jax.Array:
    """Inverts each diagonal `[B, B]` block of `(I + L)` via column sweep.

    Args:
        strict_lower: `[..., num_blocks, B, B]` strictly lower-triangular blocks
            (or full `[..., C, C]` matrix reshaped by caller).
        block_size: Diagonal block size `B`.
        zero_diag: If True, returns `M_inv - I` (strictly lower-triangular, zero
            diagonal and upper triangle) instead of `M_inv`.
    """
    b = block_size
    eye = jnp.eye(b, dtype=jnp.float32)
    m_inv = -strict_lower.astype(jnp.float32)
    for j in range(1, b - 1):
        col_j = m_inv[..., :, j : j + 1]
        row_j = m_inv[..., j : j + 1, :]
        m_inv = m_inv + col_j * row_j
    if zero_diag:
        return m_inv
    return m_inv + eye


def _double_inverted_blocks(
    m_inv: jax.Array,
    l_full: jax.Array,
    cur_b: int,
    target_b: int,
    mxu_dtype: jnp.dtype,
    opts: _HcOptions,
) -> jax.Array:
    """Doubles diagonal inverse block size from `cur_b` up to `target_b`."""
    c = l_full.shape[-1]
    eye_full = jnp.eye(c, dtype=jnp.float32)
    b = cur_b
    while b < target_b:
        num_pairs = c // (2 * b)
        m_pairs = m_inv.reshape(-1, num_pairs, 2, b, b)
        m00 = m_pairs[:, :, 0]
        m11 = m_pairs[:, :, 1]
        l_blocks = l_full.reshape(-1, num_pairs, 2, b, num_pairs, 2, b)
        # Extract the sub-diagonal block L_{2k+1, 2k} for each pair k:
        pair_idx = jnp.arange(num_pairs)
        l10 = l_blocks[:, pair_idx, 1, :, pair_idx, 0, :]

        if opts.merged_doubling and opts.doubling_8bit and not opts.doubling_2limb:
            # Fuse `T = L10 @ M00` by contracting `L_sub` (the sub-diagonal rows
            # of `L + I`, holding `[L10, I_b]` in columns `[2k*b : (2k+2)*b]`)
            # against `block_diag(M00, M11)` or by computing `L10 @ M00` and
            # `-M11 @ T` in 8-bit directly with `M00 = I + M00_strict`:
            if opts.fast_db:
                # Since M00 and M11 are unit lower-triangular (`I + M_strict`),
                # their entries are bounded well within FP8 range (or scaled by
                # a fixed power-of-two factor `1/4`), avoiding any `amax` reduction!
                inv_scale = jnp.float32(4.0)
                scale = jnp.float32(0.25)
                l10_q = (l10.astype(jnp.float32) * inv_scale).astype(mxu_dtype)
                m00_q = (m00 * inv_scale).astype(mxu_dtype)
                m11_q = (m11 * inv_scale).astype(mxu_dtype)
                t_mat = jax.lax.dot(
                    l10_q,
                    m00_q,
                    dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                    preferred_element_type=jnp.float32,
                ) * (scale * scale)
                t_q = (t_mat * inv_scale).astype(mxu_dtype)
                m10 = -jax.lax.dot(
                    m11_q,
                    t_q,
                    dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                    preferred_element_type=jnp.float32,
                ) * (scale * scale)
            else:
                l10_q, l10_s = _quantize_rows(
                    l10, mxu_dtype, axis=-1, fast=opts.fast_scale
                )
                m00_q, m00_s = _quantize_rows(
                    m00, mxu_dtype, axis=-2, fast=opts.fast_scale
                )
                t_mat = jax.lax.dot(
                    l10_q,
                    m00_q,
                    dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                    preferred_element_type=jnp.float32,
                ) * (l10_s * m00_s)
                m11_q, m11_s = _quantize_rows(
                    m11, mxu_dtype, axis=-1, fast=opts.fast_scale
                )
                t_q, t_s = _quantize_rows(
                    t_mat, mxu_dtype, axis=-2, fast=opts.fast_scale
                )
                m10 = -jax.lax.dot(
                    m11_q,
                    t_q,
                    dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                    preferred_element_type=jnp.float32,
                ) * (m11_s * t_s)
        elif opts.doubling_2limb:
            l10_hi, l10_lo, l10_s = _two_limb_split(
                l10, mxu_dtype, scale_per_row=True, axis=-1
            )
            m00_hi, m00_lo, m00_s = _two_limb_split(
                m00, mxu_dtype, scale_per_row=True, axis=-2
            )
            t_mat = _dot_two_limb(
                l10_hi,
                l10_lo,
                l10_s,
                m00_hi,
                m00_lo,
                m00_s,
                (((2,), (1,)), ((0, 1), (0, 1))),
            )
            m11_hi, m11_lo, m11_s = _two_limb_split(
                m11, mxu_dtype, scale_per_row=True, axis=-1
            )
            t_hi, t_lo, t_s = _two_limb_split(
                t_mat, mxu_dtype, scale_per_row=True, axis=-2
            )
            m10 = -_dot_two_limb(
                m11_hi,
                m11_lo,
                m11_s,
                t_hi,
                t_lo,
                t_s,
                (((2,), (1,)), ((0, 1), (0, 1))),
            )
        elif opts.doubling_8bit:
            l10_q, l10_s = _quantize_rows(l10, mxu_dtype, axis=-1, fast=opts.fast_scale)
            m00_q, m00_s = _quantize_rows(m00, mxu_dtype, axis=-2, fast=opts.fast_scale)
            t_mat = jax.lax.dot(
                l10_q,
                m00_q,
                dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                preferred_element_type=jnp.float32,
            ) * (l10_s * m00_s)
            m11_q, m11_s = _quantize_rows(m11, mxu_dtype, axis=-1, fast=opts.fast_scale)
            t_q, t_s = _quantize_rows(t_mat, mxu_dtype, axis=-2, fast=opts.fast_scale)
            m10 = -jax.lax.dot(
                m11_q,
                t_q,
                dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                preferred_element_type=jnp.float32,
            ) * (m11_s * t_s)
        else:
            t_mat = jax.lax.dot(
                l10.astype(jnp.bfloat16),
                m00.astype(jnp.bfloat16),
                dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                preferred_element_type=jnp.float32,
            )
            m10 = -jax.lax.dot(
                m11.astype(jnp.bfloat16),
                t_mat.astype(jnp.bfloat16),
                dimension_numbers=(((2,), (1,)), ((0, 1), (0, 1))),
                preferred_element_type=jnp.float32,
            )
        zeros = jnp.zeros_like(m00)
        top = jnp.concatenate([m00, zeros], axis=-1)
        bot = jnp.concatenate([m10, m11], axis=-1)
        m_inv = jnp.concatenate([top, bot], axis=-2)
        b *= 2
    del eye_full
    return m_inv


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


class _HcChunk:
    """Staged per-KQ-group high-chunk FP8 GDN state machine."""

    def __init__(
        self,
        q_large: jax.Array,
        k_large: jax.Array,
        v_large: jax.Array,
        g_cum_log2: jax.Array,
        beta_large: jax.Array,
        state_prev: jax.Array,
        cfg: config.GDNConfig,
        opts: _HcOptions,
        *,
        out_slot_ref: Any = None,
        state_slot_ref: Any = None,
        col_offset: int = 0,
        row_offset: int = 0,
    ):
        self.q_large = q_large
        self.k_large = k_large
        self.v_large = v_large
        self.g_cum_log2 = g_cum_log2
        self.beta_large = beta_large
        self.state_prev = state_prev
        self.cfg = cfg
        self.opts = opts
        self.out_slot_ref = out_slot_ref
        self.state_slot_ref = state_slot_ref
        self.col_offset = col_offset
        self.row_offset = row_offset
        self.mxu_dtype = jnp.dtype(cfg.gdn_mxu_operand_dtype)
        self.c = q_large.shape[1]
        self.nv = v_large.shape[0]
        self.nq = q_large.shape[0]
        self.v_per_kq = self.nv // self.nq
        self.b = min(opts.base_block, self.c)
        self.target_b = min(max(opts.double_to, self.b), self.c)
        self.num_base_blocks = self.c // self.b
        self.num_blocks = self.c // self.target_b
        self.pair_v = self.v_per_kq % 2 == 0 and v_large.shape[-1] % 128 == 0
        self.kq_repeats = self.v_per_kq // 2 if self.pair_v else self.v_per_kq

    def prepare(self) -> None:
        opts = self.opts
        with jax.named_scope("hc_cum_decay_log2"):
            g_cum_log2_t = fused_transpose_broadcast(
                self.g_cum_log2, src_dim=1, dst_dim=2
            )
            beta_t = fused_transpose_broadcast(self.beta_large, src_dim=1, dst_dim=2)
            g_diff_log2 = self.g_cum_log2 - g_cum_log2_t

            mask_dtype = get_mask_dtype(self.cfg.dtypes.compute)
            iota_r = jax.lax.broadcasted_iota(mask_dtype, g_diff_log2.shape, 1)
            iota_c = jax.lax.broadcasted_iota(mask_dtype, g_diff_log2.shape, 2)
            strictly_lower_mask = iota_r > iota_c
            lower_mask = iota_r >= iota_c

            if opts.no_exp_clamp:
                gating_map = jnp.exp2(g_diff_log2)
            else:
                g_diff_clamped = jnp.where(
                    lower_mask, jnp.minimum(g_diff_log2, 0.0), -80.0
                )
                gating_map = jnp.exp2(g_diff_clamped)
            self.gating_map_lower = jnp.where(lower_mask, gating_map, 0.0)
            self.gating_map_strict = jnp.where(strictly_lower_mask, gating_map, 0.0)
            self.gating_backward = jnp.exp2(-g_diff_log2[..., -1:])
            self.gating_forward = jnp.exp2(self.g_cum_log2)
            self.gating_last = self.gating_forward[:, -1:]
            self.beta_t = beta_t

    def prep_kq(self) -> None:
        opts = self.opts
        mxu_dtype = self.mxu_dtype
        with jax.named_scope("hc_prep_kq_fp8_kkt_qkt"):
            if opts.bf16_proj:
                k_bf16 = self.k_large.astype(jnp.bfloat16)
                q_bf16 = self.q_large.astype(jnp.bfloat16)
                if opts.qpad:
                    qk_cat = jnp.concatenate([k_bf16, q_bf16], axis=1)
                    qk_kt = jax.lax.dot(
                        qk_cat,
                        k_bf16,
                        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
                    self.k_k_t = qk_kt[:, : self.c, :]
                    self.q_k_t = qk_kt[:, self.c :, :]
                else:
                    self.k_k_t = jax.lax.dot(
                        k_bf16,
                        k_bf16,
                        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
                    self.q_k_t = jax.lax.dot(
                        q_bf16,
                        k_bf16,
                        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
                if opts.kq_fixed_scale > 0.0:
                    kq_s_val = jnp.float32(1.0 / opts.kq_fixed_scale)
                    kq_inv_s = jnp.float32(opts.kq_fixed_scale)
                    self.k_hi = (self.k_large.astype(jnp.float32) * kq_inv_s).astype(
                        mxu_dtype
                    )
                    self.q_hi = (self.q_large.astype(jnp.float32) * kq_inv_s).astype(
                        mxu_dtype
                    )
                    self.k_s = kq_s_val
                    self.q_s = kq_s_val
                    if opts.k_1_limb:
                        self.k_lo = jnp.zeros_like(self.k_hi)
                    else:
                        self.k_lo = (
                            self.k_large.astype(jnp.float32) * kq_inv_s
                            - self.k_hi.astype(jnp.float32)
                        ).astype(mxu_dtype)
                    if opts.q_1_limb:
                        self.q_lo = jnp.zeros_like(self.q_hi)
                    else:
                        self.q_lo = (
                            self.q_large.astype(jnp.float32) * kq_inv_s
                            - self.q_hi.astype(jnp.float32)
                        ).astype(mxu_dtype)
            elif opts.kq_fixed_scale > 0.0:
                kq_s_val = jnp.float32(1.0 / opts.kq_fixed_scale)
                kq_inv_s = jnp.float32(opts.kq_fixed_scale)
                k_scaled = self.k_large.astype(jnp.float32) * kq_inv_s
                q_scaled = self.q_large.astype(jnp.float32) * kq_inv_s
                self.k_hi = k_scaled.astype(mxu_dtype)
                self.k_lo = (
                    jnp.zeros_like(self.k_hi)
                    if opts.k_1_limb
                    else (k_scaled - self.k_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                self.k_s = kq_s_val
                self.q_hi = q_scaled.astype(mxu_dtype)
                self.q_lo = (
                    jnp.zeros_like(self.q_hi)
                    if opts.q_1_limb
                    else (q_scaled - self.q_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                self.q_s = kq_s_val
                self.k_k_t = _dot_two_limb(
                    self.k_hi,
                    self.k_lo,
                    self.k_s,
                    self.k_hi,
                    self.k_lo,
                    self.k_s,
                    (((2,), (2,)), ((0,), (0,))),
                    num_terms=(1 if opts.k_1_limb else 3),
                )
                self.q_k_t = _dot_two_limb(
                    self.q_hi,
                    self.q_lo,
                    self.q_s,
                    self.k_hi,
                    self.k_lo,
                    self.k_s,
                    (((2,), (2,)), ((0,), (0,))),
                    num_terms=(
                        1
                        if (opts.k_1_limb and opts.q_1_limb)
                        else (2 if opts.q_1_limb else 3)
                    ),
                )
            else:
                self.k_hi, self.k_lo, self.k_s = _two_limb_split(
                    self.k_large,
                    mxu_dtype,
                    scale_per_row=True,
                    axis=-1,
                    fast_scale=opts.fast_scale,
                )
                self.q_hi, self.q_lo, self.q_s = _two_limb_split(
                    self.q_large,
                    mxu_dtype,
                    scale_per_row=True,
                    axis=-1,
                    fast_scale=opts.fast_scale,
                )
                self.k_k_t = _dot_two_limb(
                    self.k_hi,
                    self.k_lo,
                    self.k_s,
                    self.k_hi,
                    self.k_lo,
                    jnp.swapaxes(self.k_s, -1, -2),
                    (((2,), (2,)), ((0,), (0,))),
                    num_terms=(1 if opts.k_1_limb else 3),
                )
                self.q_k_t = _dot_two_limb(
                    self.q_hi,
                    self.q_lo,
                    self.q_s,
                    self.k_hi,
                    self.k_lo,
                    jnp.swapaxes(self.k_s, -1, -2),
                    (((2,), (2,)), ((0,), (0,))),
                    num_terms=(
                        1
                        if (opts.k_1_limb and opts.q_1_limb)
                        else (2 if opts.q_1_limb else 3)
                    ),
                )

    def base_invert(self) -> None:
        with jax.named_scope("hc_base_block_invert"):
            beta_k_k_t = jnp.repeat(self.k_k_t, self.v_per_kq, axis=0) * self.beta_t
            self.l_full = self.gating_map_strict * beta_k_k_t
            b = self.b
            num_base_blocks = self.num_base_blocks
            l_grid = self.l_full.reshape(
                self.nv, num_base_blocks, b, num_base_blocks, b
            )
            blk_idx = jnp.arange(num_base_blocks)
            l_diag_blocks = l_grid[:, blk_idx, :, blk_idx, :]
            self.m_inv = _invert_unit_lower_blocks(
                l_diag_blocks,
                b,
                zero_diag=(self.opts.rhs_zero_diag and self.target_b == b),
            )

    def double_blocks(self) -> None:
        with jax.named_scope("hc_schur_block_doubling"):
            if self.target_b > self.b:
                self.m_inv = _double_inverted_blocks(
                    self.m_inv,
                    self.l_full,
                    self.b,
                    self.target_b,
                    self.mxu_dtype,
                    self.opts,
                )
                if self.opts.rhs_zero_diag:
                    self.m_inv = self.m_inv - jnp.eye(self.target_b, dtype=jnp.float32)

    def state_vscale(self) -> None:
        opts = self.opts
        mxu_dtype = self.mxu_dtype
        with jax.named_scope("hc_state_vscale"):
            state_for_kq = (
                _pack_v_head_pairs(self.state_prev, self.v_per_kq)
                if self.pair_v
                else self.state_prev
            )
            if opts.v_col_scale:
                s_f32 = state_for_kq.astype(jnp.float32)
                s_col_max = jnp.max(jnp.abs(s_f32), axis=1, keepdims=True)
                self.s_col_scale = s_col_max * (1.0 / _FP8_E4M3_MAX) + jnp.float32(
                    1e-30
                )
                s_norm = s_f32 / self.s_col_scale
                self.s_hi = s_norm.astype(mxu_dtype)
                self.s_lo = (
                    jnp.zeros_like(self.s_hi)
                    if opts.state_1_limb
                    else (s_norm - self.s_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                self.s_s = jnp.ones((), dtype=jnp.float32)
            elif opts.state_fixed_scale > 0.0:
                s_s_val = jnp.float32(1.0 / opts.state_fixed_scale)
                s_inv_s = jnp.float32(opts.state_fixed_scale)
                s_scaled = state_for_kq.astype(jnp.float32) * s_inv_s
                self.s_hi = s_scaled.astype(mxu_dtype)
                self.s_lo = (
                    jnp.zeros_like(self.s_hi)
                    if opts.state_1_limb
                    else (s_scaled - self.s_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                self.s_s = s_s_val
            else:
                self.s_hi, self.s_lo, self.s_s = _two_limb_split(
                    state_for_kq,
                    mxu_dtype,
                    scale_per_row=opts.state_dyn_scale,
                    axis=1,
                    fast_scale=opts.fast_scale,
                )

    def kq_proj(self) -> None:
        opts = self.opts
        with jax.named_scope("hc_kq_state_proj"):
            kq_hi = jnp.repeat(
                jnp.concatenate([self.k_hi, self.q_hi], axis=1), self.kq_repeats, axis=0
            )
            kq_lo = jnp.repeat(
                jnp.concatenate([self.k_lo, self.q_lo], axis=1), self.kq_repeats, axis=0
            )
            if opts.kq_fixed_scale > 0.0:
                kq_s = self.k_s
            else:
                kq_s = jnp.repeat(
                    jnp.concatenate([self.k_s, self.q_s], axis=1),
                    self.kq_repeats,
                    axis=0,
                )
            kq_Both_1 = opts.k_1_limb and opts.q_1_limb
            state_num_terms = (
                1 if (opts.state_1_limb and kq_Both_1) else (2 if kq_Both_1 else 3)
            )
            kq_state = _dot_two_limb(
                kq_hi,
                kq_lo,
                kq_s,
                self.s_hi,
                self.s_lo,
                self.s_s,
                (((2,), (1,)), ((0,), (0,))),
                num_terms=state_num_terms,
            )
            if opts.v_col_scale:
                kq_state = kq_state * self.s_col_scale
            if self.pair_v:
                kq_state = _unpack_v_head_pairs(kq_state, self.v_per_kq)
            k_state, q_state = jnp.split(kq_state, 2, axis=1)
            self.out_updated = q_state * self.gating_forward
            self.v_diff = self.v_large - k_state * self.gating_forward

    def v_diff_scale(self) -> None:
        opts = self.opts
        with jax.named_scope("hc_v_diff_and_col_scale"):
            if opts.v_col_scale:
                v_col_max = jnp.max(jnp.abs(self.v_diff), axis=1, keepdims=True)
                self.v_col_s = v_col_max * (1.0 / (0.25 * _FP8_E4M3_MAX)) + jnp.float32(
                    1e-30
                )
                self.v_diff_work = self.v_diff / self.v_col_s
            else:
                self.v_col_s = jnp.ones((), dtype=jnp.float32)
                self.v_diff_work = self.v_diff

    def solve_tail(self) -> None:
        opts = self.opts
        mxu_dtype = self.mxu_dtype
        target_b = self.target_b
        num_blocks = self.num_blocks
        nv = self.nv
        c = self.c

        with jax.named_scope("hc_solve_tail"):

            def _apply_m_inv_block(
                m_blk: jax.Array, rhs_blk: jax.Array, *, is_merged: bool = False
            ) -> jax.Array:
                rhs_base = rhs_blk if opts.rhs_zero_diag else 0.0
                if not opts.use_8bit_inv:
                    return rhs_base + jax.lax.dot(
                        m_blk.astype(jnp.bfloat16),
                        rhs_blk.astype(jnp.bfloat16),
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
                if opts.inv_2limb:
                    m_hi, m_lo, m_s = _two_limb_split(
                        m_blk,
                        mxu_dtype,
                        scale_per_row=opts.inv_dyn_scale,
                        axis=-1,
                        fast_scale=opts.fast_scale,
                    )
                    r_hi, r_lo, r_s = _two_limb_split(
                        rhs_blk,
                        mxu_dtype,
                        scale_per_row=opts.inv_dyn_scale,
                        axis=-2,
                        fast_scale=opts.fast_scale,
                    )
                    return rhs_base + _dot_two_limb(
                        m_hi, m_lo, m_s, r_hi, r_lo, r_s, (((2,), (1,)), ((0,), (0,)))
                    )
                if opts.inv_dyn_scale:
                    m_q, m_s = _quantize_rows(
                        m_blk, mxu_dtype, axis=-1, fast=opts.fast_scale
                    )
                    r_q, r_s = _quantize_rows(
                        rhs_blk, mxu_dtype, axis=-2, fast=opts.fast_scale
                    )
                    return rhs_base + jax.lax.dot(
                        m_q,
                        r_q,
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    ) * (m_s * r_s)
                if opts.row_fixed_scale > 0.0:
                    m_inv_scale = jnp.float32(4.0 if opts.rhs_zero_diag else 2.0)
                    m_scale = jnp.float32(0.25 if opts.rhs_zero_diag else 0.5)
                    m_q = (m_blk * m_inv_scale).astype(mxu_dtype)
                    if opts.v_col_scale and not is_merged:
                        r_hi = rhs_blk.astype(mxu_dtype)
                        if opts.tail_2limb:
                            r_lo = (rhs_blk - r_hi.astype(jnp.float32)).astype(
                                mxu_dtype
                            )
                            return rhs_base + _dot_two_limb(
                                m_q,
                                m_q,
                                m_scale,
                                r_hi,
                                r_lo,
                                jnp.ones((), dtype=jnp.float32),
                                (((2,), (1,)), ((0,), (0,))),
                                num_terms=2,
                            )
                        return (
                            rhs_base
                            + jax.lax.dot(
                                m_q,
                                r_hi,
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            )
                            * m_scale
                        )
                    if opts.solve_df8 or is_merged:
                        if opts.tail_2limb:
                            r_hi, r_lo, r_s = _two_limb_split(
                                rhs_blk,
                                mxu_dtype,
                                scale_per_row=True,
                                axis=-2,
                                fast_scale=opts.fast_scale,
                            )
                            return rhs_base + _dot_two_limb(
                                m_q,
                                m_q,
                                m_scale,
                                r_hi,
                                r_lo,
                                r_s,
                                (((2,), (1,)), ((0,), (0,))),
                                num_terms=2,
                            )
                        r_q, r_s = _quantize_rows(
                            rhs_blk, mxu_dtype, axis=-2, fast=opts.fast_scale
                        )
                        return rhs_base + jax.lax.dot(
                            m_q,
                            r_q,
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            preferred_element_type=jnp.float32,
                        ) * (m_scale * r_s)
                    r_scale = jnp.float32(opts.row_fixed_scale / _FP8_E4M3_MAX)
                    r_inv_scale = jnp.float32(_FP8_E4M3_MAX / opts.row_fixed_scale)
                    r_scaled = rhs_blk * r_inv_scale
                    if opts.tail_2limb:
                        r_hi = r_scaled.astype(mxu_dtype)
                        r_lo = ((r_scaled - r_hi.astype(jnp.float32)) * 16.0).astype(
                            mxu_dtype
                        )
                        return rhs_base + _dot_two_limb(
                            m_q,
                            m_q,
                            m_scale,
                            r_hi,
                            r_lo,
                            r_scale,
                            (((2,), (1,)), ((0,), (0,))),
                            num_terms=2,
                            b_lo_scale=1.0 / 16.0,
                        )
                    r_q = r_scaled.astype(mxu_dtype)
                    return rhs_base + jax.lax.dot(
                        m_q,
                        r_q,
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    ) * (m_scale * r_scale)
                return rhs_base + jax.lax.dot(
                    m_blk.astype(mxu_dtype),
                    rhs_blk.astype(mxu_dtype),
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )

            if num_blocks == 2 and opts.two_block_step:
                rhs_blocks = self.v_diff_work.reshape(nv, 2, target_b, -1)
                x_all_0 = _apply_m_inv_block(
                    self.m_inv.reshape(nv * 2, target_b, target_b),
                    rhs_blocks.reshape(nv * 2, target_b, -1),
                ).reshape(nv, 2, target_b, -1)
                x0 = x_all_0[:, 0]
                x1_init = x_all_0[:, 1]
                l10 = self.l_full[:, target_b:, :target_b]
                if opts.merged_solve:
                    m11 = self.m_inv[:, 1]
                    m11_q = (
                        m11 * jnp.float32(4.0 if opts.rhs_zero_diag else 2.0)
                    ).astype(mxu_dtype)
                    m11_s = jnp.float32(0.25 if opts.rhs_zero_diag else 0.5)
                    l10_q = (l10 * jnp.float32(4.0)).astype(mxu_dtype)
                    l10_s = jnp.float32(0.25)
                    m11_l10 = jax.lax.dot(
                        m11_q,
                        l10_q,
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    ) * (m11_s * l10_s)
                    if opts.rhs_zero_diag:
                        m11_l10 = l10 + m11_l10
                    x0_q = (x0 * jnp.float32(_FP8_E4M3_MAX / 64.0)).astype(mxu_dtype)
                    x0_s = jnp.float32(64.0 / _FP8_E4M3_MAX)
                    m11_l10_q = (m11_l10 * jnp.float32(4.0)).astype(mxu_dtype)
                    corr1 = jax.lax.dot(
                        m11_l10_q,
                        x0_q,
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    ) * (jnp.float32(0.25) * x0_s)
                    x1 = x1_init - corr1
                else:
                    if not opts.use_8bit_inv or opts.bf16_l:
                        prev_sum = jax.lax.dot(
                            l10.astype(jnp.bfloat16),
                            x0.astype(jnp.bfloat16),
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            preferred_element_type=jnp.float32,
                        )
                    elif opts.v_col_scale:
                        l_q = (l10 * jnp.float32(4.0)).astype(mxu_dtype)
                        x_hi = x0.astype(mxu_dtype)
                        if opts.tail_2limb:
                            x_lo = (x0 - x_hi.astype(jnp.float32)).astype(mxu_dtype)
                            prev_sum = _dot_two_limb(
                                l_q,
                                l_q,
                                jnp.float32(0.25),
                                x_hi,
                                x_lo,
                                jnp.ones((), dtype=jnp.float32),
                                (((2,), (1,)), ((0,), (0,))),
                                num_terms=2,
                            )
                        else:
                            prev_sum = jax.lax.dot(
                                l_q,
                                x_hi,
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            ) * jnp.float32(0.25)
                    elif opts.row_fixed_scale > 0.0:
                        l_q = (l10 * jnp.float32(4.0)).astype(mxu_dtype)
                        if opts.solve_df8:
                            if opts.tail_2limb:
                                x_hi, x_lo, x_s = _two_limb_split(
                                    x0,
                                    mxu_dtype,
                                    scale_per_row=True,
                                    axis=-2,
                                    fast_scale=opts.fast_scale,
                                )
                                prev_sum = _dot_two_limb(
                                    l_q,
                                    l_q,
                                    jnp.float32(0.25),
                                    x_hi,
                                    x_lo,
                                    x_s,
                                    (((2,), (1,)), ((0,), (0,))),
                                    num_terms=2,
                                )
                            else:
                                x_q, x_s = _quantize_rows(
                                    x0, mxu_dtype, axis=-2, fast=opts.fast_scale
                                )
                                prev_sum = jax.lax.dot(
                                    l_q,
                                    x_q,
                                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                    preferred_element_type=jnp.float32,
                                ) * (jnp.float32(0.25) * x_s)
                        else:
                            r_scale = jnp.float32(opts.row_fixed_scale / _FP8_E4M3_MAX)
                            r_inv_scale = jnp.float32(
                                _FP8_E4M3_MAX / opts.row_fixed_scale
                            )
                            x_scaled = x0 * r_inv_scale
                            if opts.tail_2limb:
                                x_hi = x_scaled.astype(mxu_dtype)
                                x_lo = (
                                    (x_scaled - x_hi.astype(jnp.float32)) * 16.0
                                ).astype(mxu_dtype)
                                prev_sum = _dot_two_limb(
                                    l_q,
                                    l_q,
                                    jnp.float32(0.25),
                                    x_hi,
                                    x_lo,
                                    r_scale,
                                    (((2,), (1,)), ((0,), (0,))),
                                    num_terms=2,
                                    b_lo_scale=1.0 / 16.0,
                                )
                            else:
                                x_q = x_scaled.astype(mxu_dtype)
                                prev_sum = jax.lax.dot(
                                    l_q,
                                    x_q,
                                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                    preferred_element_type=jnp.float32,
                                ) * (jnp.float32(0.25) * r_scale)
                    else:
                        l_q, l_s = _quantize_rows(
                            l10, mxu_dtype, axis=-1, fast=opts.fast_scale
                        )
                        x_q, x_s = _quantize_rows(
                            x0, mxu_dtype, axis=-2, fast=opts.fast_scale
                        )
                        prev_sum = jax.lax.dot(
                            l_q,
                            x_q,
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            preferred_element_type=jnp.float32,
                        ) * (l_s * x_s)
                    corr1 = _apply_m_inv_block(self.m_inv[:, 1], prev_sum)
                    x1 = x1_init - corr1
                x_solved = jnp.concatenate([x0, x1], axis=1)
            else:
                x_blocks: list[jax.Array] = []
                for idx in range(num_blocks):
                    start = idx * target_b
                    end = min((idx + 1) * target_b, c)
                    rhs_blk = self.v_diff_work[:, start:end, :]
                    if idx > 0:
                        solved_x = jnp.concatenate(x_blocks, axis=1)
                        l_off = self.l_full[:, start:end, :start]
                        if not opts.use_8bit_inv:
                            prev_sum = jax.lax.dot(
                                l_off.astype(jnp.bfloat16),
                                solved_x.astype(jnp.bfloat16),
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            )
                        elif opts.inv_2limb:
                            l_hi, l_lo, l_s = _two_limb_split(
                                l_off,
                                mxu_dtype,
                                scale_per_row=opts.inv_dyn_scale,
                                axis=-1,
                                fast_scale=opts.fast_scale,
                            )
                            x_hi, x_lo, x_s = _two_limb_split(
                                solved_x,
                                mxu_dtype,
                                scale_per_row=opts.inv_dyn_scale,
                                axis=-2,
                                fast_scale=opts.fast_scale,
                            )
                            prev_sum = _dot_two_limb(
                                l_hi,
                                l_lo,
                                l_s,
                                x_hi,
                                x_lo,
                                x_s,
                                (((2,), (1,)), ((0,), (0,))),
                            )
                        elif opts.inv_dyn_scale:
                            l_q, l_s = _quantize_rows(
                                l_off, mxu_dtype, axis=-1, fast=opts.fast_scale
                            )
                            x_q, x_s = _quantize_rows(
                                solved_x, mxu_dtype, axis=-2, fast=opts.fast_scale
                            )
                            prev_sum = jax.lax.dot(
                                l_q,
                                x_q,
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            ) * (l_s * x_s)
                        elif opts.row_fixed_scale > 0.0:
                            l_q = (l_off * jnp.float32(4.0)).astype(mxu_dtype)
                            r_scale = jnp.float32(opts.row_fixed_scale / _FP8_E4M3_MAX)
                            r_inv_scale = jnp.float32(
                                _FP8_E4M3_MAX / opts.row_fixed_scale
                            )
                            x_q = (solved_x * r_inv_scale).astype(mxu_dtype)
                            prev_sum = jax.lax.dot(
                                l_q,
                                x_q,
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            ) * (jnp.float32(0.25) * r_scale)
                        else:
                            prev_sum = jax.lax.dot(
                                l_off.astype(mxu_dtype),
                                solved_x.astype(mxu_dtype),
                                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                                preferred_element_type=jnp.float32,
                            )
                        rhs_blk = rhs_blk - prev_sum
                    x_blk = _apply_m_inv_block(self.m_inv[:, idx], rhs_blk)
                    x_blocks.append(x_blk)
                x_solved = jnp.concatenate(x_blocks, axis=1)

            if opts.v_col_scale:
                self.u_ws_norm = x_solved * self.beta_large
                self.u_ws = self.u_ws_norm * self.v_col_s
            else:
                self.u_ws = x_solved * self.beta_large

    def u_limbs(self) -> None:
        opts = self.opts
        mxu_dtype = self.mxu_dtype
        with jax.named_scope("hc_u_limbs"):
            if opts.v_col_scale:
                u_hi = self.u_ws_norm.astype(mxu_dtype)
                u_lo = (
                    jnp.zeros_like(u_hi)
                    if opts.u_1_limb
                    else (self.u_ws_norm - u_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                u_s = jnp.ones((), dtype=jnp.float32)
            else:
                u_hi, u_lo, u_s = _two_limb_split(
                    self.u_ws,
                    mxu_dtype,
                    scale_per_row=opts.u_dyn_scale,
                    axis=1,
                    fast_scale=opts.fast_scale,
                )

            if opts.direct_kg:
                ug_src = self.u_ws_norm if opts.v_col_scale else self.u_ws
                u_ws_gated = ug_src * self.gating_backward
                if opts.v_col_scale:
                    ug_hi = u_ws_gated.astype(mxu_dtype)
                    ug_lo = (
                        jnp.zeros_like(ug_hi)
                        if opts.u_1_limb
                        else (u_ws_gated - ug_hi.astype(jnp.float32)).astype(mxu_dtype)
                    )
                    ug_s = jnp.ones((), dtype=jnp.float32)
                else:
                    ug_hi, ug_lo, ug_s = _two_limb_split(
                        u_ws_gated,
                        mxu_dtype,
                        scale_per_row=opts.u_dyn_scale,
                        axis=1,
                        fast_scale=opts.fast_scale,
                    )
                if self.pair_v:
                    ug_hi = _pack_v_head_pairs(ug_hi, self.v_per_kq)
                    ug_lo = _pack_v_head_pairs(ug_lo, self.v_per_kq)
                    if opts.u_dyn_scale and not opts.v_col_scale:
                        ug_s = _pack_v_head_pairs(ug_s, self.v_per_kq)
                self.ug_hi, self.ug_lo, self.ug_s = ug_hi, ug_lo, ug_s
            self.u_hi, self.u_lo, self.u_s = u_hi, u_lo, u_s

    def state_out_update(self) -> None:
        opts = self.opts
        mxu_dtype = self.mxu_dtype
        with jax.named_scope("hc_fused_state_and_out_update"):
            out_qk = (
                jnp.repeat(self.q_k_t, self.v_per_kq, axis=0) * self.gating_map_lower
            )
            if opts.bf16_oqk:
                out_new = jax.lax.dot(
                    out_qk.astype(jnp.bfloat16),
                    self.u_ws.astype(jnp.bfloat16),
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                if opts.direct_kg:
                    k_rep_hi = jnp.repeat(self.k_hi, self.kq_repeats, axis=0)
                    k_rep_lo = jnp.repeat(self.k_lo, self.kq_repeats, axis=0)
                    k_rep_s = (
                        self.k_s
                        if opts.kq_fixed_scale > 0.0
                        else jnp.repeat(
                            jnp.swapaxes(self.k_s, -1, -2), self.kq_repeats, axis=0
                        )
                    )
                    state_new = _dot_two_limb(
                        k_rep_hi,
                        k_rep_lo,
                        k_rep_s,
                        self.ug_hi,
                        self.ug_lo,
                        self.ug_s,
                        (((1,), (1,)), ((0,), (0,))),
                        num_terms=(
                            1
                            if (opts.u_1_limb and opts.k_1_limb)
                            else (2 if opts.k_1_limb else 3)
                        ),
                    )
                    if opts.v_col_scale:
                        if self.pair_v:
                            state_new = _unpack_v_head_pairs(state_new, self.v_per_kq)
                        state_new = state_new * self.v_col_s
                    elif self.pair_v:
                        state_new = _unpack_v_head_pairs(state_new, self.v_per_kq)
                else:
                    u_ws_gated = self.u_ws * self.gating_backward
                    state_new = _dot_kq_with_v_heads(
                        self.k_large.astype(jnp.bfloat16),
                        u_ws_gated.astype(jnp.bfloat16),
                        (((1,), (1,)), ((0,), (0,))),
                    )
                self.state = self.state_prev * self.gating_last + state_new
                self.out = self.out_updated + out_new
                return

            if opts.kq_fixed_scale > 0.0:
                oqk_s_val = jnp.float32(1.0 / opts.kq_fixed_scale)
                oqk_inv_s = jnp.float32(opts.kq_fixed_scale)
                oqk_scaled = out_qk.astype(jnp.float32) * oqk_inv_s
                oqk_hi = oqk_scaled.astype(mxu_dtype)
                oqk_lo = (
                    jnp.zeros_like(oqk_hi)
                    if opts.lu_1_limb
                    else (oqk_scaled - oqk_hi.astype(jnp.float32)).astype(mxu_dtype)
                )
                oqk_s = oqk_s_val
            else:
                oqk_hi, oqk_lo, oqk_s = _two_limb_split(
                    out_qk,
                    mxu_dtype,
                    scale_per_row=opts.u_dyn_scale,
                    axis=-1,
                    fast_scale=opts.fast_scale,
                )

            if opts.direct_kg and self.pair_v and opts.kq_fixed_scale > 0.0:
                k_rep_hi = jnp.repeat(
                    jnp.swapaxes(self.k_hi, 1, 2), self.kq_repeats, axis=0
                )
                k_rep_lo = jnp.repeat(
                    jnp.swapaxes(self.k_lo, 1, 2), self.kq_repeats, axis=0
                )
                oqk_hi_p = _pack_v_head_pairs(oqk_hi, self.v_per_kq)
                oqk_lo_p = _pack_v_head_pairs(oqk_lo, self.v_per_kq)
                u_hi_p = _pack_v_head_pairs(self.u_hi, self.v_per_kq)
                u_lo_p = _pack_v_head_pairs(self.u_lo, self.v_per_kq)

                lhs_hi = jnp.concatenate([k_rep_hi, oqk_hi_p], axis=0)
                lhs_lo = jnp.concatenate([k_rep_lo, oqk_lo_p], axis=0)
                rhs_hi = jnp.concatenate([self.ug_hi, u_hi_p], axis=0)
                rhs_lo = jnp.concatenate([self.ug_lo, u_lo_p], axis=0)

                if opts.u_dyn_scale and not opts.v_col_scale:
                    u_s_p = _pack_v_head_pairs(self.u_s, self.v_per_kq)
                    rhs_s = jnp.concatenate([self.ug_s, u_s_p], axis=0)
                else:
                    rhs_s = self.u_s
                lhs_both_1 = opts.k_1_limb and opts.lu_1_limb
                comb_terms = (
                    1 if (opts.u_1_limb and lhs_both_1) else (2 if lhs_both_1 else 3)
                )
                comb = _dot_two_limb(
                    lhs_hi,
                    lhs_lo,
                    self.k_s,
                    rhs_hi,
                    rhs_lo,
                    rhs_s,
                    (((2,), (1,)), ((0,), (0,))),
                    num_terms=comb_terms,
                )
                state_new_p, out_new_p = jnp.split(comb, 2, axis=0)
                state_new = _unpack_v_head_pairs(state_new_p, self.v_per_kq)
                out_new = _unpack_v_head_pairs(out_new_p, self.v_per_kq)
                if opts.v_col_scale:
                    state_new = state_new * self.v_col_s
                    out_new = out_new * self.v_col_s
            else:
                out_new = _dot_two_limb(
                    oqk_hi,
                    oqk_lo,
                    oqk_s,
                    self.u_hi,
                    self.u_lo,
                    self.u_s,
                    (((2,), (1,)), ((0,), (0,))),
                    num_terms=(
                        1
                        if (opts.u_1_limb and opts.lu_1_limb)
                        else (2 if opts.lu_1_limb else 3)
                    ),
                )
                if opts.v_col_scale:
                    out_new = out_new * self.v_col_s
                if opts.direct_kg:
                    k_rep_hi = jnp.repeat(self.k_hi, self.kq_repeats, axis=0)
                    k_rep_lo = jnp.repeat(self.k_lo, self.kq_repeats, axis=0)
                    k_rep_s = (
                        self.k_s
                        if opts.kq_fixed_scale > 0.0
                        else jnp.repeat(
                            jnp.swapaxes(self.k_s, -1, -2), self.kq_repeats, axis=0
                        )
                    )
                    state_new = _dot_two_limb(
                        k_rep_hi,
                        k_rep_lo,
                        k_rep_s,
                        self.ug_hi,
                        self.ug_lo,
                        self.ug_s,
                        (((1,), (1,)), ((0,), (0,))),
                        num_terms=(
                            1
                            if (opts.u_1_limb and opts.k_1_limb)
                            else (2 if opts.k_1_limb else 3)
                        ),
                    )
                    if self.pair_v:
                        state_new = _unpack_v_head_pairs(state_new, self.v_per_kq)
                    if opts.v_col_scale:
                        state_new = state_new * self.v_col_s
                else:
                    u_ws_gated = self.u_ws * self.gating_backward
                    state_new = _dot_kq_with_v_heads(
                        self.k_large.astype(jnp.bfloat16),
                        u_ws_gated.astype(jnp.bfloat16),
                        (((1,), (1,)), ((0,), (0,))),
                    )

            self.state = self.state_prev * self.gating_last + state_new
            self.out = self.out_updated + out_new

            if self.out_slot_ref is not None:
                v_dim = self.v_large.shape[-1]
                r0 = self.row_offset
                r1 = r0 + self.c
                for h_local in range(self.nv):
                    col = (self.col_offset + h_local) * v_dim
                    self.out_slot_ref[0, r0:r1, 0, col : col + v_dim] = self.out[
                        h_local
                    ].astype(self.out_slot_ref.dtype)
            if self.state_slot_ref is not None:
                h0 = self.col_offset
                h1 = h0 + self.nv
                self.state_slot_ref[0, h0:h1] = self.state.astype(
                    self.state_slot_ref.dtype
                )

    def out_state(self) -> tuple[jax.Array, jax.Array]:
        self.u_limbs()
        self.state_out_update()
        return self.out, self.state


def _chunked_gdn_per_seq_hc(
    q_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    k_large: jax.Array,  # [num_kq_heads, chunk, kq_head_dim]
    v_large: jax.Array,  # [num_v_heads, chunk, v_head_dim]
    gating_log: jax.Array,  # [1, chunk, num_v_heads]
    beta: jax.Array,  # [1, chunk, num_v_heads]
    state_prev: jax.Array,  # [num_v_heads, kq_head_dim, v_head_dim]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Chunked GDN using 2-limb 8-bit MXU contractions and block-doubling inverse."""
    opts = _hc_options(cfg)
    assert opts is not None
    assert cfg.gdn_mxu_operand_dtype is not None

    with jax.named_scope("hc_cum_decay_log2"):
        gating_log2 = gating_log * _LOG2E
        g_cum_list = [gating_log2[:, :1]]
        for row in range(1, cfg.chunk_size):
            g_cum_list.append(g_cum_list[-1] + gating_log2[:, row : row + 1])
        g_cum_log2_compact = jnp.concat(g_cum_list, axis=1)

        if opts.group_gating and cfg.num_kq_heads > 1:
            v_per_kq = cfg.v_per_kq_head
            g_cum_heads = [
                fused_transpose_broadcast(
                    g_cum_log2_compact[:, :, h * v_per_kq : (h + 1) * v_per_kq],
                    src_dim=2,
                    dst_dim=0,
                )[:v_per_kq]
                for h in range(cfg.num_kq_heads)
            ]
            beta_heads = [
                fused_transpose_broadcast(
                    beta[:, :, h * v_per_kq : (h + 1) * v_per_kq],
                    src_dim=2,
                    dst_dim=0,
                )[:v_per_kq]
                for h in range(cfg.num_kq_heads)
            ]
            g_cum_log2 = jnp.concatenate(g_cum_heads, axis=0)
            beta_large = jnp.concatenate(beta_heads, axis=0)
        else:
            g_cum_log2 = fused_transpose_broadcast(
                g_cum_log2_compact, src_dim=2, dst_dim=0
            )[: cfg.num_v_heads]
            beta_large = fused_transpose_broadcast(beta, src_dim=2, dst_dim=0)[
                : cfg.num_v_heads
            ]

    if opts.subchunk_size > 0 and opts.subchunk_size < cfg.chunk_size:
        sub_c = opts.subchunk_size
        num_sub = cfg.chunk_size // sub_c
        state_cur = state_prev
        out_sub_list = []
        for s_idx in range(num_sub):
            r0 = s_idx * sub_c
            r1 = r0 + sub_c
            g_sub_base = (
                jnp.zeros_like(g_cum_log2[:, :1, :])
                if s_idx == 0
                else g_cum_log2[:, r0 - 1 : r0, :]
            )
            g_sub = g_cum_log2[:, r0:r1, :] - g_sub_base
            if opts.seq_group and cfg.num_kq_heads > 1:
                v_per_kq = cfg.v_per_kq_head
                outs_g = []
                states_g = []
                for h in range(cfg.num_kq_heads):
                    vs = slice(h * v_per_kq, (h + 1) * v_per_kq)
                    ks = slice(h, h + 1)
                    hc_g = _HcChunk(
                        q_large[ks, r0:r1],
                        k_large[ks, r0:r1],
                        v_large[vs, r0:r1],
                        g_sub[vs],
                        beta_large[vs, r0:r1],
                        state_cur[vs],
                        cfg,
                        opts,
                    )
                    if opts.state_first:
                        hc_g.prepare()
                        hc_g.state_vscale()
                        hc_g.prep_kq()
                        hc_g.kq_proj()
                        hc_g.v_diff_scale()
                        hc_g.base_invert()
                        hc_g.double_blocks()
                        hc_g.solve_tail()
                    else:
                        hc_g.prepare()
                        hc_g.prep_kq()
                        hc_g.base_invert()
                        hc_g.double_blocks()
                        hc_g.state_vscale()
                        hc_g.kq_proj()
                        hc_g.v_diff_scale()
                        hc_g.solve_tail()
                    o_g, st_g = hc_g.out_state()
                    outs_g.append(o_g)
                    states_g.append(st_g)
                out_sub_list.append(jnp.concatenate(outs_g, axis=0))
                state_cur = jnp.concatenate(states_g, axis=0)
            else:
                hc_s = _HcChunk(
                    q_large[:, r0:r1],
                    k_large[:, r0:r1],
                    v_large[:, r0:r1],
                    g_sub,
                    beta_large[:, r0:r1],
                    state_cur,
                    cfg,
                    opts,
                )
                if opts.state_first:
                    hc_s.prepare()
                    hc_s.state_vscale()
                    hc_s.prep_kq()
                    hc_s.kq_proj()
                    hc_s.v_diff_scale()
                    hc_s.base_invert()
                    hc_s.double_blocks()
                    hc_s.solve_tail()
                else:
                    hc_s.prepare()
                    hc_s.prep_kq()
                    hc_s.base_invert()
                    hc_s.double_blocks()
                    hc_s.state_vscale()
                    hc_s.kq_proj()
                    hc_s.v_diff_scale()
                    hc_s.solve_tail()
                o_s, state_cur = hc_s.out_state()
                out_sub_list.append(o_s)
        return jnp.concatenate(out_sub_list, axis=1), state_cur

    if opts.seq_group and cfg.num_kq_heads > 1:
        v_per_kq = cfg.v_per_kq_head
        outs = []
        states = []
        for h in range(cfg.num_kq_heads):
            vs = slice(h * v_per_kq, (h + 1) * v_per_kq)
            ks = slice(h, h + 1)
            hc = _HcChunk(
                q_large[ks],
                k_large[ks],
                v_large[vs],
                g_cum_log2[vs],
                beta_large[vs],
                state_prev[vs],
                cfg,
                opts,
            )
            if opts.state_first:
                hc.prepare()
                hc.state_vscale()
                hc.prep_kq()
                hc.kq_proj()
                hc.v_diff_scale()
                hc.base_invert()
                hc.double_blocks()
                hc.solve_tail()
            else:
                hc.prepare()
                hc.prep_kq()
                hc.base_invert()
                hc.double_blocks()
                hc.state_vscale()
                hc.kq_proj()
                hc.v_diff_scale()
                hc.solve_tail()
            o_h, s_h = hc.out_state()
            outs.append(o_h)
            states.append(s_h)
        return jnp.concatenate(outs, axis=0), jnp.concatenate(states, axis=0)

    hc = _HcChunk(
        q_large, k_large, v_large, g_cum_log2, beta_large, state_prev, cfg, opts
    )
    if opts.state_first:
        hc.prepare()
        hc.state_vscale()
        hc.prep_kq()
        hc.kq_proj()
        hc.v_diff_scale()
        hc.base_invert()
        hc.double_blocks()
        hc.solve_tail()
    else:
        hc.prepare()
        hc.prep_kq()
        hc.base_invert()
        hc.double_blocks()
        hc.state_vscale()
        hc.kq_proj()
        hc.v_diff_scale()
        hc.solve_tail()
    return hc.out_state()


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
    if _hc_active(cfg):
        return _chunked_gdn_per_seq_hc(
            q_large, k_large, v_large, gating_log, beta, state_prev, cfg
        )

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
    """Masks, L2-normalizes Q/K, and computes `gating_log` and `beta`."""
    assert not cfg.use_recurrent

    with jax.named_scope("gdn_prefill_mask_and_cast"):
        mask_dtype = get_mask_dtype(cfg.dtypes.compute)
        iota = jax.lax.broadcasted_iota(
            mask_dtype, (cfg.seq_tile_size, 1, cfg.chunk_size, 1), 2
        )
        mask = iota < real_sizes.reshape(-1, 1, 1, 1).astype(mask_dtype)

        opts = _hc_options(cfg) if _hc_active(cfg) else None
        v_dtype = (
            jnp.bfloat16
            if (opts is not None and opts.v_diff_bf16)
            else cfg.dtypes.compute
        )
        q_large = jnp.where(mask, q_large.astype(cfg.dtypes.compute), 0)
        k_large = jnp.where(mask, k_large.astype(cfg.dtypes.compute), 0)
        v_large = jnp.where(mask, v_large.astype(v_dtype), 0)

        b_large = b_large.astype(cfg.dtypes.compute)
        a_large = a_large.astype(cfg.dtypes.compute)

        a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
        if cfg.use_dt_bias and dt_bias is not None:
            dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)

    with jax.named_scope("gdn_prefill_l2_norm_qk"):
        # NOTE: Any element-wise computations should occur before repeat.
        q_large = l2_norm(q_large)
        q_scale = cfg.kq_head_dim**-0.5
        q_large *= q_scale
        k_large = l2_norm(k_large)

    with jax.named_scope("gdn_prefill_gate_activations"):
        # [seqs, 1, chunk, num_v_heads]
        beta = jax.nn.sigmoid(b_large)
        gating_log = compute_gates(a_large, a_log, dt_bias, cfg)

        beta = jnp.where(mask, beta, 0)
        # NOTE: Masked gating_log will evaluate to jnp.exp(0)=1. gating (decay) must
        # be masked to 1 since it signifies that strength of state from previous row
        # will be 1 (i.e., no decay) if current row is invalid.
        gating_log = jnp.where(mask, gating_log, 0)
    return q_large, k_large, v_large, gating_log, beta


def run_hc_chunk_step(hc: _HcChunk, opts: _HcOptions) -> None:
    if opts.state_first:
        hc.prepare()
        hc.state_vscale()
        hc.prep_kq()
        hc.kq_proj()
        hc.v_diff_scale()
        hc.base_invert()
        hc.double_blocks()
        hc.solve_tail()
    else:
        hc.prepare()
        hc.prep_kq()
        hc.base_invert()
        hc.double_blocks()
        hc.state_vscale()
        hc.kq_proj()
        hc.v_diff_scale()
        hc.solve_tail()
    hc.u_limbs()
    hc.state_out_update()


def chunked_gdn_stream(
    real_sizes: jax.Array,
    qkv_vmem_ref: Any,
    b_large: jax.Array,
    a_large: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    recurrent_in_slot_ref: Any,
    carry_recurrent_scratch_ref: Any,
    is_first_tile: jax.Array,
    has_initial_state: jax.Array,
    out_slot_ref: Any,
    state_out_ref: Any,
    cfg: config.GDNConfig,
) -> tuple[jax.Array | None, jax.Array | None]:
    """Streams Q/K/V and recurrent state per KQ group directly from/to VMEM refs."""
    from jax.experimental.pallas import tpu as pltpu

    opts = _hc_options(cfg)
    assert opts is not None
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_1d = jax.lax.broadcasted_iota(mask_dtype, (cfg.chunk_size, 1), 0)
    r_size_0 = real_sizes[0].astype(mask_dtype)
    mask_2d = iota_1d < r_size_0
    mask_4d = jnp.expand_dims(jnp.expand_dims(mask_2d, 0), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)
    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    if cfg.use_dt_bias and dt_bias is not None:
        dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    beta = jnp.where(mask_4d, jax.nn.sigmoid(b_large), 0)[0]
    gating_log = jnp.where(mask_4d, compute_gates(a_large, a_log, dt_bias, cfg), 0)[0]

    gating_log2 = gating_log * _LOG2E
    g_cum_list = [gating_log2[:, :1]]
    for row in range(1, cfg.chunk_size):
        g_cum_list.append(g_cum_list[-1] + gating_log2[:, row : row + 1])
    g_cum_log2_compact = jnp.concat(g_cum_list, axis=1)

    v_per_kq = cfg.v_per_kq_head
    q_cols, k_cols, v_cols = vmem_ldst.qkv_head_cols(cfg)
    q_scale = cfg.kq_head_dim**-0.5
    v_dtype = jnp.bfloat16 if opts.v_diff_bf16 else cfg.dtypes.compute

    outs = [] if out_slot_ref is None else None
    states = [] if state_out_ref is None else None

    for h in range(cfg.num_kq_heads):
        h0 = h * v_per_kq
        h1 = h0 + v_per_kq
        g_cum_h = fused_transpose_broadcast(
            g_cum_log2_compact[:, :, h0:h1], src_dim=2, dst_dim=0
        )[:v_per_kq]
        beta_h = fused_transpose_broadcast(beta[:, :, h0:h1], src_dim=2, dst_dim=0)[
            :v_per_kq
        ]

        q_h = vmem_ldst.load_qkv_col(qkv_vmem_ref, q_cols[h]).astype(cfg.dtypes.compute)
        k_h = vmem_ldst.load_qkv_col(qkv_vmem_ref, k_cols[h]).astype(cfg.dtypes.compute)
        q_h = jnp.expand_dims(jnp.where(mask_2d, l2_norm(q_h) * q_scale, 0), 0)
        k_h = jnp.expand_dims(jnp.where(mask_2d, l2_norm(k_h), 0), 0)

        v_h_list = [
            jnp.where(
                mask_2d,
                vmem_ldst.load_qkv_col(qkv_vmem_ref, v_cols[h0 + vi]).astype(v_dtype),
                0,
            )
            for vi in range(v_per_kq)
        ]
        v_h = jnp.stack(v_h_list, axis=0)

        hbm_s = pltpu.bitcast(
            recurrent_in_slot_ref[0, 0, h0:h1], cfg.dtypes.recurrent_state
        )
        prev_s = jnp.where(has_initial_state, hbm_s, 0)
        if carry_recurrent_scratch_ref is not None:
            carry_s = carry_recurrent_scratch_ref[0, h0:h1].astype(prev_s.dtype)
            prev_s = jnp.where(is_first_tile, prev_s, carry_s)

        hc = _HcChunk(
            q_h,
            k_h,
            v_h,
            g_cum_h,
            beta_h,
            prev_s,
            cfg,
            opts,
            out_slot_ref=out_slot_ref,
            state_slot_ref=state_out_ref,
            col_offset=h0,
        )
        run_hc_chunk_step(hc, opts)
        if outs is not None:
            outs.append(hc.out)
        if states is not None:
            states.append(hc.state)

    out_ret = (
        jnp.expand_dims(jnp.concatenate(outs, axis=0).swapaxes(0, 1), 0)
        if outs is not None
        else None
    )
    state_ret = (
        jnp.expand_dims(jnp.expand_dims(jnp.concatenate(states, axis=0), 0), 0)
        if states is not None
        else None
    )
    return out_ret, state_ret


def chunked_gdn_lazy(
    real_sizes: jax.Array,
    qkv_vmem_ref: Any,
    b_large: jax.Array,
    a_large: jax.Array,
    state_prev: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Chunked GDN loading Q/K/V columns per KQ group lazily from `qkv_vmem_ref`."""
    opts = _hc_options(cfg)
    assert opts is not None
    mask_dtype = get_mask_dtype(cfg.dtypes.compute)
    iota_1d = jax.lax.broadcasted_iota(mask_dtype, (cfg.chunk_size, 1), 0)
    r_size_0 = real_sizes[0].astype(mask_dtype)
    mask_2d = iota_1d < r_size_0
    mask_4d = jnp.expand_dims(jnp.expand_dims(mask_2d, 0), 0)

    b_large = b_large.astype(cfg.dtypes.compute)
    a_large = a_large.astype(cfg.dtypes.compute)
    a_log = a_log.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    if cfg.use_dt_bias and dt_bias is not None:
        dt_bias = dt_bias.reshape(1, 1, 1, -1).astype(cfg.dtypes.compute)
    beta = jnp.where(mask_4d, jax.nn.sigmoid(b_large), 0)[0]
    gating_log = jnp.where(mask_4d, compute_gates(a_large, a_log, dt_bias, cfg), 0)[0]

    gating_log2 = gating_log * _LOG2E
    g_cum_list = [gating_log2[:, :1]]
    for row in range(1, cfg.chunk_size):
        g_cum_list.append(g_cum_list[-1] + gating_log2[:, row : row + 1])
    g_cum_log2_compact = jnp.concat(g_cum_list, axis=1)

    v_per_kq = cfg.v_per_kq_head
    g_cum_heads = [
        fused_transpose_broadcast(
            g_cum_log2_compact[:, :, h * v_per_kq : (h + 1) * v_per_kq],
            src_dim=2,
            dst_dim=0,
        )[:v_per_kq]
        for h in range(cfg.num_kq_heads)
    ]
    beta_heads = [
        fused_transpose_broadcast(
            beta[:, :, h * v_per_kq : (h + 1) * v_per_kq],
            src_dim=2,
            dst_dim=0,
        )[:v_per_kq]
        for h in range(cfg.num_kq_heads)
    ]

    q_cols, k_cols, v_cols = vmem_ldst.qkv_head_cols(cfg)
    q_scale = cfg.kq_head_dim**-0.5
    v_dtype = jnp.bfloat16 if opts.v_diff_bf16 else cfg.dtypes.compute

    outs = []
    states = []
    state_0 = state_prev[0]
    for h in range(cfg.num_kq_heads):
        vs = slice(h * v_per_kq, (h + 1) * v_per_kq)
        q_h = vmem_ldst.load_qkv_col(qkv_vmem_ref, q_cols[h]).astype(cfg.dtypes.compute)
        k_h = vmem_ldst.load_qkv_col(qkv_vmem_ref, k_cols[h]).astype(cfg.dtypes.compute)
        q_h = jnp.expand_dims(jnp.where(mask_2d, l2_norm(q_h) * q_scale, 0), 0)
        k_h = jnp.expand_dims(jnp.where(mask_2d, l2_norm(k_h), 0), 0)
        v_h_list = [
            jnp.where(
                mask_2d,
                vmem_ldst.load_qkv_col(qkv_vmem_ref, v_cols[h * v_per_kq + vi]).astype(
                    v_dtype
                ),
                0,
            )
            for vi in range(v_per_kq)
        ]
        v_h = jnp.stack(v_h_list, axis=0)
        hc = _HcChunk(
            q_h,
            k_h,
            v_h,
            g_cum_heads[h],
            beta_heads[h],
            state_0[vs],
            cfg,
            opts,
        )
        run_hc_chunk_step(hc, opts)
        outs.append(hc.out)
        states.append(hc.state)

    out = jnp.expand_dims(jnp.concatenate(outs, axis=0).swapaxes(0, 1), 0)
    state = jnp.expand_dims(jnp.expand_dims(jnp.concatenate(states, axis=0), 0), 0)
    return out, state


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
    """Perform chunked GDN over input [seq, num_heads, chunk, head_dim].

    The returned state is [seq, 1, num_v_heads, kq_head_dim, v_head_dim]: the
    chunked path only produces the final state, so it is never taken when
    more than one checkpoint per sequence is required.
    """
    q_large, k_large, v_large, gating_log, beta = chunked_gdn_prepare(
        real_sizes, q_large, k_large, v_large, b_large, a_large, a_log, dt_bias, cfg
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
