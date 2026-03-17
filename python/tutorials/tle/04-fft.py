"""
1D FFT with Triton and TLE (TLE Tutorial)
=======================================

This tutorial implements a simple 1D complex FFT over the last dimension of an
(M, N) tensor and compares Triton vs TLE kernels against torch.fft.fft. If
`cuda.tile` is available, it also runs a cuTile FFT kernel adapted from NVIDIA's
cutile-python tests.

Notes
-----
- N must be a power-of-two (<= 1024) for this tutorial implementation.
- Complex values are represented as two float32 arrays (real/imag).
- The kernels implement iterative Cooley-Tukey DIT with a bit-reversal copy.
- Twiddle factors are precomputed on the host and read from global memory.
- TLE uses a register-only path for small N to reduce shared-memory traffic.
- cuTile path is optional and requires `cuda.tile` + `cupy`; it uses a 3-factor
  decomposition with precomputed DFT/twiddle tables.
"""

# %%
# Setup
# -----

import argparse
import math
import sys
from typing import Tuple

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

try:
    import cuda.tile as ct  # type: ignore
    import cupy as cp  # type: ignore
    _HAVE_CUTILE = True
except Exception:  # pragma: no cover - optional dependency
    ct = None
    cp = None
    _HAVE_CUTILE = False

DEVICE = triton.runtime.driver.active.get_active_torch_device()
PI = math.pi

_BITREV_CACHE: dict[Tuple[int, torch.device], torch.Tensor] = {}
_TWIDDLE_CACHE: dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}
_CUTILE_CONST_CACHE: dict[Tuple[int, torch.device], tuple] = {}
_FFT_REG_THRESHOLD = 256


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2(n: int) -> int:
    return n.bit_length() - 1


def _bitrev_indices(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    cached = _BITREV_CACHE.get(key)
    if cached is not None:
        return cached
    log_n = _log2(n)
    idx = torch.arange(n, device=device, dtype=torch.int32)
    rev = torch.zeros_like(idx)
    tmp = idx.clone()
    for _ in range(log_n):
        rev = (rev << 1) | (tmp & 1)
        tmp = tmp >> 1
    _BITREV_CACHE[key] = rev
    return rev


def _twiddle_tables(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (n, device)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached
    log_n = _log2(n)
    tw_real = torch.empty((n - 1, ), device=device, dtype=torch.float32)
    tw_imag = torch.empty((n - 1, ), device=device, dtype=torch.float32)
    offset = 0
    for stage in range(log_n):
        m = 1 << (stage + 1)
        half = m >> 1
        j = torch.arange(half, device=device, dtype=torch.float32)
        angle = (-2.0 * PI / m) * j
        tw_real[offset:offset + half] = torch.cos(angle)
        tw_imag[offset:offset + half] = torch.sin(angle)
        offset += half
    _TWIDDLE_CACHE[key] = (tw_real, tw_imag)
    return tw_real, tw_imag


def _prepare_input(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.is_complex():
        if x.dtype not in (torch.complex64, torch.complex128):
            raise ValueError(f"unsupported complex dtype: {x.dtype}")
        x = x.to(torch.complex64)
        real = x.real.contiguous()
        imag = x.imag.contiguous()
    else:
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError(f"unsupported dtype: {x.dtype}")
        x = x.to(torch.float32)
        real = x.contiguous()
        imag = torch.zeros_like(real)
    return real, imag


# %%
# cuTile helpers (optional)
# -------------------------


def _choose_factors(n: int) -> Tuple[int, int, int]:
    log_n = _log2(n)
    exps = [0, 0, 0]
    for _ in range(log_n):
        idx = exps.index(min(exps))
        exps[idx] += 1
    return (1 << exps[0], 1 << exps[1], 1 << exps[2])


def _cupy_from_torch(t: torch.Tensor):
    if cp is None:
        raise RuntimeError("cupy is not available")
    try:
        return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t))
    except AttributeError:
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(t))


def _torch_from_cupy(a):
    try:
        return torch.utils.dlpack.from_dlpack(a.toDlpack())
    except AttributeError:
        return torch.utils.dlpack.from_dlpack(a.to_dlpack())


def _cutile_constants(n: int, device: torch.device):
    if not _HAVE_CUTILE:
        raise RuntimeError("cuda.tile is not available")
    key = (n, device)
    cached = _CUTILE_CONST_CACHE.get(key)
    if cached is not None:
        return cached

    f0, f1, f2 = _choose_factors(n)
    dtype = torch.float32

    def dft_matrix(size: int) -> torch.Tensor:
        k = torch.arange(size, device=device, dtype=dtype)[:, None]
        n_idx = torch.arange(size, device=device, dtype=dtype)[None, :]
        angle = (-2.0 * PI / size) * (k * n_idx)
        w_real = torch.cos(angle)
        w_imag = torch.sin(angle)
        return torch.stack((w_real, w_imag), dim=-1).contiguous()

    w0 = dft_matrix(f0)
    w1 = dft_matrix(f1)
    w2 = dft_matrix(f2)

    k0 = torch.arange(f0, device=device, dtype=dtype)[:, None, None]
    i1 = torch.arange(f1, device=device, dtype=dtype)[None, :, None]
    i2 = torch.arange(f2, device=device, dtype=dtype)[None, None, :]
    idx = i1 * f2 + i2
    angle0 = (-2.0 * PI / n) * (k0 * idx)
    t0 = torch.stack((torch.cos(angle0), torch.sin(angle0)), dim=-1).contiguous()
    t0 = t0.reshape(f0, f1 * f2, 2)

    k1 = torch.arange(f1, device=device, dtype=dtype)[:, None]
    i2 = torch.arange(f2, device=device, dtype=dtype)[None, :]
    angle1 = (-2.0 * PI / (f1 * f2)) * (k1 * i2)
    t1 = torch.stack((torch.cos(angle1), torch.sin(angle1)), dim=-1).contiguous()

    stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(stream.cuda_stream):
        w0_cp = _cupy_from_torch(w0)
        w1_cp = _cupy_from_torch(w1)
        w2_cp = _cupy_from_torch(w2)
        t0_cp = _cupy_from_torch(t0)
        t1_cp = _cupy_from_torch(t1)

    cached = (w0_cp, w1_cp, w2_cp, t0_cp, t1_cp, f0, f1, f2)
    _CUTILE_CONST_CACHE[key] = cached
    return cached


if _HAVE_CUTILE:
    # SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
    # SPDX-License-Identifier: Apache-2.0
    ConstInt = ct.Constant[int]

    @ct.kernel
    def fft_kernel_cutile(  # noqa: C901
        x_packed_in,
        y_packed_out,
        W0,
        W1,
        W2,
        T0,
        T1,
        N: ConstInt,
        F0: ConstInt,
        F1: ConstInt,
        F2: ConstInt,
        BS: ConstInt,
        D: ConstInt,
    ):
        F0F1 = F0 * F1
        F1F2 = F1 * F2
        F0F2 = F0 * F2
        bid = ct.bid(0)

        X_ri = ct.reshape(
            ct.load(x_packed_in, index=(bid, 0, 0), shape=(BS, N * 2 // D, D)),
            (BS, N, 2),
        )
        X_r = ct.reshape(ct.extract(X_ri, index=(0, 0, 0), shape=(BS, N, 1)), (BS, F0, F1, F2))
        X_i = ct.reshape(ct.extract(X_ri, index=(0, 0, 1), shape=(BS, N, 1)), (BS, F0, F1, F2))

        W0_ri_loaded = ct.reshape(ct.load(W0, index=(0, 0, 0), shape=(F0, F0, 2)), (F0, F0, 2))
        W0_r_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(0, 0, 0), shape=(F0, F0, 1)), (1, F0, F0))
        W0_i_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(0, 0, 1), shape=(F0, F0, 1)), (1, F0, F0))

        W1_ri_loaded = ct.reshape(ct.load(W1, index=(0, 0, 0), shape=(F1, F1, 2)), (F1, F1, 2))
        W1_r_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(0, 0, 0), shape=(F1, F1, 1)), (1, F1, F1))
        W1_i_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(0, 0, 1), shape=(F1, F1, 1)), (1, F1, F1))

        W2_ri_loaded = ct.reshape(ct.load(W2, index=(0, 0, 0), shape=(F2, F2, 2)), (F2, F2, 2))
        W2_r_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(0, 0, 0), shape=(F2, F2, 1)), (1, F2, F2))
        W2_i_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(0, 0, 1), shape=(F2, F2, 1)), (1, F2, F2))

        T0_ri_loaded = ct.reshape(ct.load(T0, index=(0, 0, 0), shape=(F0, F1F2, 2)), (F0, F1F2, 2))
        T0_r_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 0), shape=(F0, F1F2, 1)), (N, 1))
        T0_i_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 1), shape=(F0, F1F2, 1)), (N, 1))

        T1_ri_loaded = ct.reshape(ct.load(T1, index=(0, 0, 0), shape=(F1, F2, 2)), (F1, F2, 2))
        T1_r_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 0), shape=(F1, F2, 1)), (F1F2, 1))
        T1_i_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 1), shape=(F1, F2, 1)), (F1F2, 1))

        X_r = ct.reshape(X_r, (BS, F0, F1F2))
        X_i = ct.reshape(X_i, (BS, F0, F1F2))
        X_r_ = ct.reshape(ct.matmul(W0_r_tile, X_r) - ct.matmul(W0_i_tile, X_i), (BS, N, 1))
        X_i_ = ct.reshape(ct.matmul(W0_i_tile, X_r) + ct.matmul(W0_r_tile, X_i), (BS, N, 1))

        X_r = T0_r_tile * X_r_ - T0_i_tile * X_i_
        X_i = T0_i_tile * X_r_ + T0_r_tile * X_i_
        X_r = ct.permute(ct.reshape(X_r, (BS, F0, F1, F2)), (0, 2, 3, 1))
        X_i = ct.permute(ct.reshape(X_i, (BS, F0, F1, F2)), (0, 2, 3, 1))

        X_r = ct.reshape(X_r, (BS, F1, F0F2))
        X_i = ct.reshape(X_i, (BS, F1, F0F2))
        X_r_ = ct.reshape(ct.matmul(W1_r_tile, X_r) - ct.matmul(W1_i_tile, X_i), (BS, F1F2, F0))
        X_i_ = ct.reshape(ct.matmul(W1_i_tile, X_r) + ct.matmul(W1_r_tile, X_i), (BS, F1F2, F0))

        X_r = T1_r_tile * X_r_ - T1_i_tile * X_i_
        X_i = T1_i_tile * X_r_ + T1_r_tile * X_i_
        X_r = ct.permute(ct.reshape(X_r, (BS, F1, F2, F0)), (0, 2, 3, 1))
        X_i = ct.permute(ct.reshape(X_i, (BS, F1, F2, F0)), (0, 2, 3, 1))

        X_r = ct.reshape(X_r, (BS, F2, F0F1))
        X_i = ct.reshape(X_i, (BS, F2, F0F1))
        X_r_ = ct.matmul(W2_r_tile, X_r) - ct.matmul(W2_i_tile, X_i)
        X_i_ = ct.matmul(W2_i_tile, X_r) + ct.matmul(W2_r_tile, X_i)

        X_r = ct.permute(ct.reshape(X_r_, (BS, F2, F0, F1)), (0, 1, 3, 2))
        X_i = ct.permute(ct.reshape(X_i_, (BS, F2, F0, F1)), (0, 1, 3, 2))
        X_r = ct.reshape(X_r, (BS, N, 1))
        X_i = ct.reshape(X_i, (BS, N, 1))

        Y_ri = ct.reshape(ct.cat((X_r, X_i), axis=-1), (BS, N * 2 // D, D))
        ct.store(y_packed_out, index=(bid, 0, 0), tile=Y_ri)


def cutile_fft(x: torch.Tensor) -> torch.Tensor:
    if not _HAVE_CUTILE:
        raise RuntimeError("cuda.tile/cupy not available")
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    m, n = x.shape
    if not _is_power_of_two(n):
        raise ValueError(f"N={n} must be a power-of-two")
    if n > 1024:
        raise ValueError(f"N={n} too large for this tutorial kernel (max 1024)")

    in_real, in_imag = _prepare_input(x)
    w0, w1, w2, t0, t1, f0, f1, f2 = _cutile_constants(n, x.device)
    d_pack = 2
    bs_tile = 32 if m >= 32 else m

    x_ri = torch.stack((in_real, in_imag), dim=-1).contiguous()
    stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(stream.cuda_stream):
        x_cp = _cupy_from_torch(x_ri)
        y_cp = cp.empty_like(x_cp)
        grid = (triton.cdiv(m, bs_tile), 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            fft_kernel_cutile,
            (x_cp, y_cp, w0, w1, w2, t0, t1, n, f0, f1, f2, bs_tile, d_pack),
        )
        y_torch = _torch_from_cupy(y_cp)

    return torch.complex(y_torch[..., 0], y_torch[..., 1])


# %%
# Kernels (Triton)
# ---------------


@triton.jit
def fft_kernel_triton(
    in_real,
    in_imag,
    bitrev,
    twiddle_real,
    twiddle_imag,
    buf0_real,
    buf0_imag,
    buf1_real,
    buf1_imag,
    stride_in,
    stride_buf,
    n_rows,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, N)
    row_valid = row < n_rows
    mask = row_valid & (offs < N)

    rev = tl.load(bitrev + offs, mask=offs < N, other=0)
    in_real_ptrs = in_real + row * stride_in + rev
    in_imag_ptrs = in_imag + row * stride_in + rev
    vals_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
    vals_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

    buf0_real_ptrs = buf0_real + row * stride_buf + offs
    buf0_imag_ptrs = buf0_imag + row * stride_buf + offs
    tl.store(buf0_real_ptrs, vals_real, mask=mask)
    tl.store(buf0_imag_ptrs, vals_imag, mask=mask)

    buf_a_real = buf0_real
    buf_a_imag = buf0_imag
    buf_b_real = buf1_real
    buf_b_imag = buf1_imag

    if LOG_N % 2 == 1:
        m = 2
        half = 1
        idx = offs
        pos = idx & (m - 1)
        j = pos & (half - 1)
        base = idx - pos
        even_idx = base + j
        odd_idx = even_idx + half

        even_ptrs_real = buf_a_real + row * stride_buf + even_idx
        even_ptrs_imag = buf_a_imag + row * stride_buf + even_idx
        odd_ptrs_real = buf_a_real + row * stride_buf + odd_idx
        odd_ptrs_imag = buf_a_imag + row * stride_buf + odd_idx

        u_real = tl.load(even_ptrs_real, mask=mask, other=0.0)
        u_imag = tl.load(even_ptrs_imag, mask=mask, other=0.0)
        v_real = tl.load(odd_ptrs_real, mask=mask, other=0.0)
        v_imag = tl.load(odd_ptrs_imag, mask=mask, other=0.0)

        base_tw = 0
        tw_idx = base_tw + j
        tw_real = tl.load(twiddle_real + tw_idx, mask=mask, other=1.0)
        tw_imag = tl.load(twiddle_imag + tw_idx, mask=mask, other=0.0)

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        add_mask = pos < half
        out_real = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
        out_imag = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)

        out_ptrs_real = buf_b_real + row * stride_buf + idx
        out_ptrs_imag = buf_b_imag + row * stride_buf + idx
        tl.store(out_ptrs_real, out_real, mask=mask)
        tl.store(out_ptrs_imag, out_imag, mask=mask)
        tl.debug_barrier()

        buf_a_real, buf_b_real = buf_b_real, buf_a_real
        buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag

    if LOG_N % 2 == 1:
        for r4 in tl.static_range((LOG_N - 1) // 2):
            stage_s = 2 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = buf_a_real + row * stride_buf + i0
            ptr0_imag = buf_a_imag + row * stride_buf + i0
            ptr1_real = buf_a_real + row * stride_buf + i1
            ptr1_imag = buf_a_imag + row * stride_buf + i1
            ptr2_real = buf_a_real + row * stride_buf + i2
            ptr2_imag = buf_a_imag + row * stride_buf + i2
            ptr3_real = buf_a_real + row * stride_buf + i3
            ptr3_imag = buf_a_imag + row * stride_buf + i3

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

            out_ptrs_real = buf_b_real + row * stride_buf + idx
            out_ptrs_imag = buf_b_imag + row * stride_buf + idx
            tl.store(out_ptrs_real, out_real, mask=mask)
            tl.store(out_ptrs_imag, out_imag, mask=mask)
            tl.debug_barrier()

            buf_a_real, buf_b_real = buf_b_real, buf_a_real
            buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag
    else:
        for r4 in tl.static_range(LOG_N // 2):
            stage_s = 1 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = buf_a_real + row * stride_buf + i0
            ptr0_imag = buf_a_imag + row * stride_buf + i0
            ptr1_real = buf_a_real + row * stride_buf + i1
            ptr1_imag = buf_a_imag + row * stride_buf + i1
            ptr2_real = buf_a_real + row * stride_buf + i2
            ptr2_imag = buf_a_imag + row * stride_buf + i2
            ptr3_real = buf_a_real + row * stride_buf + i3
            ptr3_imag = buf_a_imag + row * stride_buf + i3

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

            out_ptrs_real = buf_b_real + row * stride_buf + idx
            out_ptrs_imag = buf_b_imag + row * stride_buf + idx
            tl.store(out_ptrs_real, out_real, mask=mask)
            tl.store(out_ptrs_imag, out_imag, mask=mask)
            tl.debug_barrier()

            buf_a_real, buf_b_real = buf_b_real, buf_a_real
            buf_a_imag, buf_b_imag = buf_b_imag, buf_a_imag


# %%
# Kernels (TLE)
# -------------


@triton.jit
def fft_kernel_tle(
    in_real,
    in_imag,
    bitrev,
    twiddle_real,
    twiddle_imag,
    out_real,
    out_imag,
    stride_in,
    stride_out,
    n_rows,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, N)
    row_valid = row < n_rows
    mask = row_valid & (offs < N)

    smem_a_real = tle.gpu.alloc(
        [N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_a_imag = tle.gpu.alloc(
        [N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_b_real = tle.gpu.alloc(
        [N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_b_imag = tle.gpu.alloc(
        [N],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    rev = tl.load(bitrev + offs, mask=offs < N, other=0)
    in_real_ptrs = in_real + row * stride_in + rev
    in_imag_ptrs = in_imag + row * stride_in + rev
    vals_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
    vals_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

    smem_a_real_ptrs = tle.gpu.local_ptr(smem_a_real, (offs, ))
    smem_a_imag_ptrs = tle.gpu.local_ptr(smem_a_imag, (offs, ))
    tl.store(smem_a_real_ptrs, vals_real, mask=mask)
    tl.store(smem_a_imag_ptrs, vals_imag, mask=mask)
    tl.debug_barrier()

    smem_in_real = smem_a_real
    smem_in_imag = smem_a_imag
    smem_out_real = smem_b_real
    smem_out_imag = smem_b_imag

    if LOG_N % 2 == 1:
        m = 2
        half = 1
        idx = offs
        pos = idx & (m - 1)
        j = pos & (half - 1)
        base = idx - pos
        even_idx = base + j
        odd_idx = even_idx + half

        even_ptrs_real = tle.gpu.local_ptr(smem_in_real, (even_idx, ))
        even_ptrs_imag = tle.gpu.local_ptr(smem_in_imag, (even_idx, ))
        odd_ptrs_real = tle.gpu.local_ptr(smem_in_real, (odd_idx, ))
        odd_ptrs_imag = tle.gpu.local_ptr(smem_in_imag, (odd_idx, ))

        u_real = tl.load(even_ptrs_real, mask=mask, other=0.0)
        u_imag = tl.load(even_ptrs_imag, mask=mask, other=0.0)
        v_real = tl.load(odd_ptrs_real, mask=mask, other=0.0)
        v_imag = tl.load(odd_ptrs_imag, mask=mask, other=0.0)

        base_tw = 0
        tw_idx = base_tw + j
        tw_real = tl.load(twiddle_real + tw_idx, mask=mask, other=1.0)
        tw_imag = tl.load(twiddle_imag + tw_idx, mask=mask, other=0.0)

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        add_mask = pos < half
        out_real_val = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
        out_imag_val = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)

        out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx, ))
        out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx, ))
        tl.store(out_ptrs_real, out_real_val, mask=mask)
        tl.store(out_ptrs_imag, out_imag_val, mask=mask)
        tl.debug_barrier()

        smem_in_real, smem_out_real = smem_out_real, smem_in_real
        smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag

    if LOG_N % 2 == 1:
        for r4 in tl.static_range((LOG_N - 1) // 2):
            stage_s = 2 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = tle.gpu.local_ptr(smem_in_real, (i0, ))
            ptr0_imag = tle.gpu.local_ptr(smem_in_imag, (i0, ))
            ptr1_real = tle.gpu.local_ptr(smem_in_real, (i1, ))
            ptr1_imag = tle.gpu.local_ptr(smem_in_imag, (i1, ))
            ptr2_real = tle.gpu.local_ptr(smem_in_real, (i2, ))
            ptr2_imag = tle.gpu.local_ptr(smem_in_imag, (i2, ))
            ptr3_real = tle.gpu.local_ptr(smem_in_real, (i3, ))
            ptr3_imag = tle.gpu.local_ptr(smem_in_imag, (i3, ))

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real_val = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag_val = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

            out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx, ))
            out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx, ))
            tl.store(out_ptrs_real, out_real_val, mask=mask)
            tl.store(out_ptrs_imag, out_imag_val, mask=mask)
            tl.debug_barrier()

            smem_in_real, smem_out_real = smem_out_real, smem_in_real
            smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag
    else:
        for r4 in tl.static_range(LOG_N // 2):
            stage_s = 1 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            ptr0_real = tle.gpu.local_ptr(smem_in_real, (i0, ))
            ptr0_imag = tle.gpu.local_ptr(smem_in_imag, (i0, ))
            ptr1_real = tle.gpu.local_ptr(smem_in_real, (i1, ))
            ptr1_imag = tle.gpu.local_ptr(smem_in_imag, (i1, ))
            ptr2_real = tle.gpu.local_ptr(smem_in_real, (i2, ))
            ptr2_imag = tle.gpu.local_ptr(smem_in_imag, (i2, ))
            ptr3_real = tle.gpu.local_ptr(smem_in_real, (i3, ))
            ptr3_imag = tle.gpu.local_ptr(smem_in_imag, (i3, ))

            x0_real = tl.load(ptr0_real, mask=mask, other=0.0)
            x0_imag = tl.load(ptr0_imag, mask=mask, other=0.0)
            x1_real = tl.load(ptr1_real, mask=mask, other=0.0)
            x1_imag = tl.load(ptr1_imag, mask=mask, other=0.0)
            x2_real = tl.load(ptr2_real, mask=mask, other=0.0)
            x2_imag = tl.load(ptr2_imag, mask=mask, other=0.0)
            x3_real = tl.load(ptr3_real, mask=mask, other=0.0)
            x3_imag = tl.load(ptr3_imag, mask=mask, other=0.0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real_val = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag_val = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))

            out_ptrs_real = tle.gpu.local_ptr(smem_out_real, (idx, ))
            out_ptrs_imag = tle.gpu.local_ptr(smem_out_imag, (idx, ))
            tl.store(out_ptrs_real, out_real_val, mask=mask)
            tl.store(out_ptrs_imag, out_imag_val, mask=mask)
            tl.debug_barrier()

            smem_in_real, smem_out_real = smem_out_real, smem_in_real
            smem_in_imag, smem_out_imag = smem_out_imag, smem_in_imag

    out_real_ptrs = out_real + row * stride_out + offs
    out_imag_ptrs = out_imag + row * stride_out + offs
    smem_final_real_ptrs = tle.gpu.local_ptr(smem_in_real, (offs, ))
    smem_final_imag_ptrs = tle.gpu.local_ptr(smem_in_imag, (offs, ))
    out_vals_real = tl.load(smem_final_real_ptrs, mask=mask, other=0.0)
    out_vals_imag = tl.load(smem_final_imag_ptrs, mask=mask, other=0.0)
    tl.store(out_real_ptrs, out_vals_real, mask=mask)
    tl.store(out_imag_ptrs, out_vals_imag, mask=mask)


@triton.jit
def fft_kernel_tle_reg(
    in_real,
    in_imag,
    bitrev,
    twiddle_real,
    twiddle_imag,
    out_real,
    out_imag,
    stride_in,
    stride_out,
    n_rows,
    N: tl.constexpr,
    LOG_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    offs = tl.arange(0, N)
    row_valid = row < n_rows
    mask = row_valid & (offs < N)

    rev = tl.load(bitrev + offs, mask=offs < N, other=0)
    in_real_ptrs = in_real + row * stride_in + rev
    in_imag_ptrs = in_imag + row * stride_in + rev
    x_real = tl.load(in_real_ptrs, mask=mask, other=0.0)
    x_imag = tl.load(in_imag_ptrs, mask=mask, other=0.0)

    if LOG_N % 2 == 1:
        m = 2
        half = 1
        idx = offs
        pos = idx & (m - 1)
        j = pos & (half - 1)
        base = idx - pos
        even_idx = base + j
        odd_idx = even_idx + half

        u_real = tl.gather(x_real, even_idx, axis=0)
        u_imag = tl.gather(x_imag, even_idx, axis=0)
        v_real = tl.gather(x_real, odd_idx, axis=0)
        v_imag = tl.gather(x_imag, odd_idx, axis=0)

        tw_real = tl.load(twiddle_real + j, mask=mask, other=1.0)
        tw_imag = tl.load(twiddle_imag + j, mask=mask, other=0.0)

        v_tw_real = v_real * tw_real - v_imag * tw_imag
        v_tw_imag = v_real * tw_imag + v_imag * tw_real

        add_mask = pos < half
        out_real_val = tl.where(add_mask, u_real + v_tw_real, u_real - v_tw_real)
        out_imag_val = tl.where(add_mask, u_imag + v_tw_imag, u_imag - v_tw_imag)
        x_real = out_real_val
        x_imag = out_imag_val

    if LOG_N % 2 == 1:
        for r4 in tl.static_range((LOG_N - 1) // 2):
            stage_s = 2 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            x0_real = tl.gather(x_real, i0, axis=0)
            x0_imag = tl.gather(x_imag, i0, axis=0)
            x1_real = tl.gather(x_real, i1, axis=0)
            x1_imag = tl.gather(x_imag, i1, axis=0)
            x2_real = tl.gather(x_real, i2, axis=0)
            x2_imag = tl.gather(x_imag, i2, axis=0)
            x3_real = tl.gather(x_real, i3, axis=0)
            x3_imag = tl.gather(x_imag, i3, axis=0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real_val = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag_val = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))
            x_real = out_real_val
            x_imag = out_imag_val
    else:
        for r4 in tl.static_range(LOG_N // 2):
            stage_s = 1 + r4 * 2
            m = 1 << (stage_s + 1)
            quarter = m >> 2
            half = m >> 1
            three_quarter = quarter + half

            idx = offs
            pos = idx & (m - 1)
            j = pos & (quarter - 1)
            base = idx - pos
            i0 = base + j
            i1 = i0 + quarter
            i2 = i1 + quarter
            i3 = i2 + quarter

            x0_real = tl.gather(x_real, i0, axis=0)
            x0_imag = tl.gather(x_imag, i0, axis=0)
            x1_real = tl.gather(x_real, i1, axis=0)
            x1_imag = tl.gather(x_imag, i1, axis=0)
            x2_real = tl.gather(x_real, i2, axis=0)
            x2_imag = tl.gather(x_imag, i2, axis=0)
            x3_real = tl.gather(x_real, i3, axis=0)
            x3_imag = tl.gather(x_imag, i3, axis=0)

            base_tw1 = (1 << (stage_s - 1)) - 1
            base_tw2 = (1 << stage_s) - 1
            tw1_idx = base_tw1 + j
            tw2_idx = base_tw2 + j
            tw1_real = tl.load(twiddle_real + tw1_idx, mask=mask, other=1.0)
            tw1_imag = tl.load(twiddle_imag + tw1_idx, mask=mask, other=0.0)
            tw2_real = tl.load(twiddle_real + tw2_idx, mask=mask, other=1.0)
            tw2_imag = tl.load(twiddle_imag + tw2_idx, mask=mask, other=0.0)

            t1_real = x1_real * tw1_real - x1_imag * tw1_imag
            t1_imag = x1_real * tw1_imag + x1_imag * tw1_real
            t3_real = x3_real * tw1_real - x3_imag * tw1_imag
            t3_imag = x3_real * tw1_imag + x3_imag * tw1_real

            u0_real = x0_real + t1_real
            u0_imag = x0_imag + t1_imag
            u1_real = x0_real - t1_real
            u1_imag = x0_imag - t1_imag
            v0_real = x2_real + t3_real
            v0_imag = x2_imag + t3_imag
            v1_real = x2_real - t3_real
            v1_imag = x2_imag - t3_imag

            v0_tw_real = v0_real * tw2_real - v0_imag * tw2_imag
            v0_tw_imag = v0_real * tw2_imag + v0_imag * tw2_real
            w3_real = tw2_imag
            w3_imag = -tw2_real
            v1_tw_real = v1_real * w3_real - v1_imag * w3_imag
            v1_tw_imag = v1_real * w3_imag + v1_imag * w3_real

            o0_real = u0_real + v0_tw_real
            o0_imag = u0_imag + v0_tw_imag
            o2_real = u0_real - v0_tw_real
            o2_imag = u0_imag - v0_tw_imag
            o1_real = u1_real + v1_tw_real
            o1_imag = u1_imag + v1_tw_imag
            o3_real = u1_real - v1_tw_real
            o3_imag = u1_imag - v1_tw_imag

            m0 = pos < quarter
            m1 = (pos >= quarter) & (pos < half)
            m2 = (pos >= half) & (pos < three_quarter)
            out_real_val = tl.where(m0, o0_real, tl.where(m1, o1_real, tl.where(m2, o2_real, o3_real)))
            out_imag_val = tl.where(m0, o0_imag, tl.where(m1, o1_imag, tl.where(m2, o2_imag, o3_imag)))
            x_real = out_real_val
            x_imag = out_imag_val

    out_real_ptrs = out_real + row * stride_out + offs
    out_imag_ptrs = out_imag + row * stride_out + offs
    tl.store(out_real_ptrs, x_real, mask=mask)
    tl.store(out_imag_ptrs, x_imag, mask=mask)


# %%
# Python wrappers
# ---------------


def triton_fft(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    m, n = x.shape
    if not _is_power_of_two(n):
        raise ValueError(f"N={n} must be a power-of-two")
    if n > 1024:
        raise ValueError(f"N={n} too large for this tutorial kernel (max 1024)")

    in_real, in_imag = _prepare_input(x)
    bitrev = _bitrev_indices(n, x.device)
    tw_real, tw_imag = _twiddle_tables(n, x.device)
    log_n = _log2(n)

    buf0_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
    buf0_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)
    buf1_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
    buf1_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)

    grid = (m, )
    fft_kernel_triton[grid](
        in_real,
        in_imag,
        bitrev,
        tw_real,
        tw_imag,
        buf0_real,
        buf0_imag,
        buf1_real,
        buf1_imag,
        in_real.stride(0),
        buf0_real.stride(0),
        m,
        N=n,
        LOG_N=log_n,
        num_warps=4,
        num_stages=1,
    )

    if log_n % 2 == 0:
        out_real = buf0_real
        out_imag = buf0_imag
    else:
        out_real = buf1_real
        out_imag = buf1_imag

    return torch.complex(out_real, out_imag)


def tle_fft(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "input must be on CUDA"
    assert x.ndim == 2, "input must be 2D (M, N)"
    m, n = x.shape
    if not _is_power_of_two(n):
        raise ValueError(f"N={n} must be a power-of-two")
    if n > 1024:
        raise ValueError(f"N={n} too large for this tutorial kernel (max 1024)")

    in_real, in_imag = _prepare_input(x)
    bitrev = _bitrev_indices(n, x.device)
    tw_real, tw_imag = _twiddle_tables(n, x.device)
    log_n = _log2(n)

    out_real = torch.empty((m, n), device=x.device, dtype=torch.float32)
    out_imag = torch.empty((m, n), device=x.device, dtype=torch.float32)

    grid = (m, )
    if n == _FFT_REG_THRESHOLD:
        fft_kernel_tle_reg[grid](
            in_real,
            in_imag,
            bitrev,
            tw_real,
            tw_imag,
            out_real,
            out_imag,
            in_real.stride(0),
            out_real.stride(0),
            m,
            N=n,
            LOG_N=log_n,
            num_warps=4,
            num_stages=1,
        )
    else:
        fft_kernel_tle[grid](
            in_real,
            in_imag,
            bitrev,
            tw_real,
            tw_imag,
            out_real,
            out_imag,
            in_real.stride(0),
            out_real.stride(0),
            m,
            N=n,
            LOG_N=log_n,
            num_warps=4,
            num_stages=1,
        )

    return torch.complex(out_real, out_imag)


# %%
# Correctness check
# -----------------


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def _make_input(m: int, n: int, dtype: torch.dtype, complex_input: bool) -> torch.Tensor:
    if complex_input:
        real = torch.randn((m, n), device=DEVICE, dtype=torch.float32)
        imag = torch.randn((m, n), device=DEVICE, dtype=torch.float32)
        return torch.complex(real, imag)
    return torch.randn((m, n), device=DEVICE, dtype=dtype)


def run_correctness(m: int, n: int, dtype: torch.dtype, complex_input: bool):
    torch.manual_seed(0)
    x = _make_input(m, n, dtype, complex_input)
    ref = torch.fft.fft(x.to(torch.complex64))

    y_triton = triton_fft(x)
    y_tle = tle_fft(x)
    if _HAVE_CUTILE:
        y_cutile = cutile_fft(x)

    torch.testing.assert_close(y_triton, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(y_tle, ref, rtol=1e-3, atol=1e-3)
    if _HAVE_CUTILE:
        torch.testing.assert_close(y_cutile, ref, rtol=1e-3, atol=1e-3)
        print("Correctness check passed (triton/tle/cutile).")
    else:
        print("Correctness check passed (triton/tle).")


if "--only_unit_test" in sys.argv:
    _args = argparse.Namespace(M=128, N=256, dtype="float32", complex_input=False)
    _dtype = _get_dtype(_args.dtype)
    run_correctness(_args.M, _args.N, _dtype, _args.complex_input)
    sys.exit(0)

# %%
# Benchmark
# ---------

_BENCH_PROVIDERS = ["triton", "tle", "torch"]
_BENCH_NAMES = ["Triton", "TLE", "Torch"]
_BENCH_STYLES = [("blue", "-"), ("orange", "-"), ("green", "-")]
if _HAVE_CUTILE:
    _BENCH_PROVIDERS.insert(2, "cutile")
    _BENCH_NAMES.insert(2, "cuTile")
    _BENCH_STYLES.insert(2, ("black", "-"))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[64, 128, 256, 512, 1024],
        x_log=True,
        line_arg="provider",
        line_vals=_BENCH_PROVIDERS,
        line_names=_BENCH_NAMES,
        styles=_BENCH_STYLES,
        ylabel="ms",
        plot_name="tle-fft-performance",
        args={"M": 4096},
    ))
def benchmark(M, N, provider, dtype, complex_input):
    x = _make_input(M, N, dtype, complex_input)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.fft.fft(x.to(torch.complex64)), quantiles=quantiles)
    elif provider == "cutile":
        _cutile_constants(int(N), x.device)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: cutile_fft(x), quantiles=quantiles)
    elif provider == "tle":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tle_fft(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_fft(x), quantiles=quantiles)

    return ms, max_ms, min_ms


# %%
# Main
# ----


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096, help="num rows")
    parser.add_argument("--N", type=int, default=256, help="FFT size (power-of-two)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--complex_input", action="store_true", help="use complex input")
    parser.add_argument("--show_plots", action="store_true", help="show plots in benchmark")
    args = parser.parse_args(argv)

    dtype = _get_dtype(args.dtype)
    if not _is_power_of_two(args.N):
        raise ValueError(f"N={args.N} must be a power-of-two")

    check_m = min(args.M, 256)
    check_n = min(args.N, 256)
    run_correctness(check_m, check_n, dtype, args.complex_input)

    benchmark.run(print_data=True, show_plots=args.show_plots, dtype=dtype, complex_input=args.complex_input)


if __name__ == "__main__":
    main()
