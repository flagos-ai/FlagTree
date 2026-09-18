"""Shared kernels for MUSA RLC tails/masks, FlagGems-contract, and latency gates."""

from __future__ import annotations

import os
import statistics
import tempfile
from contextlib import contextmanager

import triton
import triton.language as tl

FLAGGEMS_RAND_BLOCK = 1024
FLAGGEMS_PAD_BLOCK_H = 64
FLAGGEMS_PAD_BLOCK_W = 16
FLAGGEMS_RAND_WARPS = 16
FLAGGEMS_PAD_WARPS = 4


@contextmanager
def rlc_compile_env(enhance, phase_mask=15, cache_dir=None, extra=None):
    keys = {
        "FLAGTREE_MUSA_RLC_ENHANCE": "1" if enhance else "0",
        "FLAGTREE_MUSA_RLC_PHASE_MASK": str(phase_mask),
    }
    if extra:
        keys.update(extra)
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="musa-rlc-")
    keys["TRITON_CACHE_DIR"] = cache_dir
    install_rlc_policy_hook()
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in keys.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield cache_dir
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def count_convert_layout(ttgir: str) -> int:
    return ttgir.count("ttg.convert_layout")


_POLICY_ENV = (
    ("FLAGTREE_MUSA_RLC_MIN_WRITEBACK_BITS", "ttg.rlc-minimum-writeback-bits"),
    ("FLAGTREE_MUSA_RLC_CONVERT_MIN_ELEMENTS", "ttg.rlc-convert-minimum-elements"),
    ("FLAGTREE_MUSA_RLC_CONVERT_MIN_ELEMENT_BITS", "ttg.rlc-convert-minimum-element-bits"),
    ("FLAGTREE_MUSA_RLC_CONVERT_COST_PER_BYTE", "ttg.rlc-convert-cost-per-byte"),
    ("FLAGTREE_MUSA_RLC_CACHED_LOAD_COST_PER_BYTE", "ttg.rlc-cached-load-cost-per-byte"),
    ("FLAGTREE_MUSA_RLC_EXPENSIVE_MATH_COST_PER_BYTE", "ttg.rlc-expensive-math-cost-per-byte"),
    ("FLAGTREE_MUSA_RLC_INTER_WARP_REDUCE_COST", "ttg.rlc-inter-warp-reduce-cost"),
    ("FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO",
     "ttg.rlc-atomic-writeback-max-elements-per-thread-ratio"),
)


def install_rlc_policy_hook():
    """Apply env cost overrides on the isolated site without replacing compiler.py."""
    from triton.backends.mthreads import compiler as musa_compiler
    from triton._C.libtriton import ir

    if getattr(musa_compiler, "_flagtree_rlc_policy_hooked", False):
        return
    original = musa_compiler.MUSABackend.make_ttgir

    def hooked(mod, metadata, opt, arch, capability):
        enhance = os.environ.get("FLAGTREE_MUSA_RLC_ENHANCE", "") in ("1", "true", "True")
        if enhance:
            builder = ir.builder(mod.context)
            for env_name, attr in _POLICY_ENV:
                raw = os.environ.get(env_name)
                if raw and int(raw) > 0:
                    mod.set_attr(attr, builder.get_int32_attr(int(raw)))
        return original(mod, metadata, opt, arch, capability)

    musa_compiler.MUSABackend.make_ttgir = staticmethod(hooked)
    musa_compiler._flagtree_rlc_policy_hooked = True


@triton.jit
def masked_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_iter * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def flaggems_rand_kernel(out_ptr, N, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + i4
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)
    scale = 2.3283064365386963e-10
    r0 = r0.to(tl.float32) * scale
    r1 = r1.to(tl.float32) * scale
    r2 = r2.to(tl.float32) * scale
    r3 = r3.to(tl.float32) * scale
    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    tl.store(out_ptr + off0, r0, mask=off0 < N)
    tl.store(out_ptr + off1, r1, mask=off1 < N)
    tl.store(out_ptr + off2, r2, mask=off2 < N)
    tl.store(out_ptr + off3, r3, mask=off3 < N)


@triton.jit
def flaggems_uniform_kernel(
    out_ptr, N, philox_seed, philox_offset, from_, to, BLOCK: tl.constexpr
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + i4
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)
    scale = 2.3283064365386963e-10
    width = to - from_
    r0 = r0.to(tl.float32) * scale * width + from_
    r1 = r1.to(tl.float32) * scale * width + from_
    r2 = r2.to(tl.float32) * scale * width + from_
    r3 = r3.to(tl.float32) * scale * width + from_
    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    tl.store(out_ptr + off0, r0, mask=off0 < N)
    tl.store(out_ptr + off1, r1, mask=off1 < N)
    tl.store(out_ptr + off2, r2, mask=off2 < N)
    tl.store(out_ptr + off3, r3, mask=off3 < N)


@triton.jit
def flaggems_replication_pad3d_kernel(
    x_ptr,
    out_ptr,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    pad_l,
    pad_t,
    pad_f,
    stride_xn,
    stride_xc,
    stride_xd,
    stride_xh,
    stride_xw,
    stride_on,
    stride_oc,
    stride_od,
    stride_oh,
    stride_ow,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ncd = tl.program_id(2)
    d_idx = pid_ncd % D_out
    nc_idx = pid_ncd // D_out
    c_idx = nc_idx % C
    n_idx = nc_idx // C
    iz = d_idx - pad_f
    iz = tl.where(iz < 0, 0, iz)
    iz = tl.where(iz > D_in - 1, D_in - 1, iz)
    x_base = x_ptr + n_idx * stride_xn + c_idx * stride_xc + iz * stride_xd
    out_base = out_ptr + n_idx * stride_on + c_idx * stride_oc + d_idx * stride_od
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    iy = offs_h - pad_t
    iy = tl.where(iy < 0, 0, iy)
    iy = tl.where(iy > H_in - 1, H_in - 1, iy)
    ix = offs_w - pad_l
    ix = tl.where(ix < 0, 0, ix)
    ix = tl.where(ix > W_in - 1, W_in - 1, ix)
    x_offset = iy[:, None] * stride_xh + ix[None, :] * stride_xw
    out_offset = offs_h[:, None] * stride_oh + offs_w[None, :] * stride_ow
    mask = (offs_h[:, None] < H_out) & (offs_w[None, :] < W_out)
    vals = tl.load(x_base + x_offset, mask=mask)
    tl.store(out_base + out_offset, vals, mask=mask)


def launch_masked_gemm(a, b, out, block_m, block_n, block_k, num_warps=4):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    kernel = masked_gemm_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return kernel


def launch_flaggems_rand(out, philox_seed=1234, philox_offset=0):
    n = out.numel()
    grid = (triton.cdiv(n, FLAGGEMS_RAND_BLOCK * 4), )
    kernel = flaggems_rand_kernel[grid](
        out,
        n,
        philox_seed,
        philox_offset,
        FLAGGEMS_RAND_BLOCK,
        num_warps=FLAGGEMS_RAND_WARPS,
        num_stages=1,
    )
    return kernel


def launch_flaggems_uniform(
    out, philox_seed=1234, philox_offset=0, from_=-1.0, to=1.0
):
    n = out.numel()
    grid = (triton.cdiv(n, FLAGGEMS_RAND_BLOCK * 4), )
    kernel = flaggems_uniform_kernel[grid](
        out,
        n,
        philox_seed,
        philox_offset,
        from_,
        to,
        FLAGGEMS_RAND_BLOCK,
        num_warps=FLAGGEMS_RAND_WARPS,
        num_stages=1,
    )
    return kernel


def launch_flaggems_pad(x, out, pad_l, pad_t, pad_f):
    n, c, d_in, h_in, w_in = x.shape
    _, _, d_out, h_out, w_out = out.shape
    grid = (
        triton.cdiv(w_out, FLAGGEMS_PAD_BLOCK_W),
        triton.cdiv(h_out, FLAGGEMS_PAD_BLOCK_H),
        n * c * d_out,
    )
    kernel = flaggems_replication_pad3d_kernel[grid](
        x,
        out,
        d_in,
        h_in,
        w_in,
        d_out,
        h_out,
        w_out,
        pad_l,
        pad_t,
        pad_f,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        c,
        FLAGGEMS_PAD_BLOCK_H,
        FLAGGEMS_PAD_BLOCK_W,
        num_warps=FLAGGEMS_PAD_WARPS,
        num_stages=1,
    )
    return kernel


@triton.jit
def flaggems_softmax_kernel(out_ptr, in_ptr, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(in_ptr + pid * N + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + pid * N + offs, (num / den).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def flaggems_rms_norm_kernel(out_ptr, in_ptr, w_ptr, N, eps, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(in_ptr + pid * N + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * w
    tl.store(out_ptr + pid * N + offs, y, mask=mask)


@triton.jit
def flaggems_dropout_kernel(x_ptr, y_ptr, n, p, philox_seed, philox_offset, BLOCK: tl.constexpr):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 = c0 + i4
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)
    scale = 2.3283064365386963e-10
    r0 = r0.to(tl.float32) * scale
    r1 = r1.to(tl.float32) * scale
    r2 = r2.to(tl.float32) * scale
    r3 = r3.to(tl.float32) * scale
    keep0 = r0 > p
    keep1 = r1 > p
    keep2 = r2 > p
    keep3 = r3 > p
    inv = 1.0 / (1.0 - p)
    off0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off1 = off0 + BLOCK
    off2 = off1 + BLOCK
    off3 = off2 + BLOCK
    x0 = tl.load(x_ptr + off0, mask=off0 < n, other=0.0)
    x1 = tl.load(x_ptr + off1, mask=off1 < n, other=0.0)
    x2 = tl.load(x_ptr + off2, mask=off2 < n, other=0.0)
    x3 = tl.load(x_ptr + off3, mask=off3 < n, other=0.0)
    tl.store(y_ptr + off0, x0 * inv * keep0, mask=off0 < n)
    tl.store(y_ptr + off1, x1 * inv * keep1, mask=off1 < n)
    tl.store(y_ptr + off2, x2 * inv * keep2, mask=off2 < n)
    tl.store(y_ptr + off3, x3 * inv * keep3, mask=off3 < n)


@triton.jit
def flaggems_gelu_and_mul_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, (x * cdf * y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def flaggems_addmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_iter * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
    bias = tl.load(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)
    out = acc * alpha + bias * beta
    tl.store(
        out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def launch_flaggems_softmax(inp, out, block_n=128):
    m, n = inp.shape
    kernel = flaggems_softmax_kernel[(m, )](
        out, inp, m, n, block_n, num_warps=4, num_stages=1)
    return kernel


def launch_flaggems_rms_norm(inp, weight, out, eps=1e-5):
    m, n = inp.shape
    block_n = 1
    while block_n < n:
        block_n *= 2
    kernel = flaggems_rms_norm_kernel[(m, )](
        out, inp, weight, n, eps, block_n, num_warps=4, num_stages=1)
    return kernel


def launch_flaggems_dropout(inp, out, p=0.1, philox_seed=7, philox_offset=0, block=1024):
    n = inp.numel()
    grid = (triton.cdiv(n, block * 4), )
    kernel = flaggems_dropout_kernel[grid](
        inp, out, n, p, philox_seed, philox_offset, block, num_warps=16, num_stages=1)
    return kernel


def launch_flaggems_gelu_and_mul(x, y, out, block=1024):
    n = x.numel()
    grid = (triton.cdiv(n, block), )
    kernel = flaggems_gelu_and_mul_kernel[grid](
        x, y, out, n, block, num_warps=4, num_stages=1)
    return kernel


def launch_flaggems_addmm(a, b, c, out, alpha=1.0, beta=1.0, block_m=128, block_n=128, block_k=64):
    m, k = a.shape
    n = b.shape[1]
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    kernel = flaggems_addmm_kernel[grid](
        a, b, c, out, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        alpha, beta, block_m, block_n, block_k, num_warps=4, num_stages=1)
    return kernel


def median_ms(fn, warmup=10, reps=50):
    import torch

    torch.musa.synchronize()
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    event_cls = getattr(torch.musa, "Event", None)
    samples = []
    if event_cls is not None:
        for _ in range(reps):
            start = event_cls(enable_timing=True)
            end = event_cls(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.musa.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        import time
        for _ in range(reps):
            torch.musa.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.musa.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p20_ms": samples[max(0, int(0.2 * (len(samples) - 1)))],
        "p80_ms": samples[min(len(samples) - 1, int(0.8 * (len(samples) - 1)))],
        "n": len(samples),
        "timer": "musa_event" if event_cls is not None else "perf_counter_sync",
        "samples_ms": samples,
    }
