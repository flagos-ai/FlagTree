# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import get_available_triton_backend
from tilegym.backend import register_impl
from tilegym.logger import get_logger

# Constants & Type Aliases

logger = get_logger(__name__)

# Capability Probe


def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend in [
        "cuda",
        "tileir",
    ]


def _supports_host_descriptor():
    return _is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# Kernel Helpers


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


def _bmm_memref_set_block_size_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    if not isinstance(nargs.get("a_ptr"), TensorDescriptor):
        return
    if nargs.get("transpose_a", False):
        nargs["a_ptr"].block_shape = [1, BLOCK_K, BLOCK_M]
    else:
        nargs["a_ptr"].block_shape = [1, BLOCK_M, BLOCK_K]
    if nargs.get("transpose_b", False):
        nargs["b_ptr"].block_shape = [1, BLOCK_N, BLOCK_K]
    else:
        nargs["b_ptr"].block_shape = [1, BLOCK_K, BLOCK_N]
    nargs["c_ptr"].block_shape = [1, BLOCK_M, BLOCK_N]
    if EPILOGUE_SUBTILE:
        nargs["c_ptr"].block_shape = [1, BLOCK_M, BLOCK_N // 2]
    else:
        nargs["c_ptr"].block_shape = [1, BLOCK_M, BLOCK_N]


@triton.jit
def _bmm_calculate_pid(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_q = pid // (num_pid_m * num_pid_n)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid = pid % (num_pid_m * num_pid_n)
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_q, pid_m, pid_n


# Autotune Config


def _bmm_memref_get_configs(pre_hook=None):
    capability = torch.cuda.get_device_capability()
    if get_available_triton_backend() == "oait":
        if capability in [(12, 0), (12, 1)]:
            return [
                triton.Config(
                    {
                        "BLOCK_M": BM,
                        "BLOCK_N": BN,
                        "BLOCK_K": BK,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": SUBTILE,
                        "occupancy": 1,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )  #
                # To reduce the CI time, we pre-tune the whole config sets in local Geforce RTX 5080 360w machine and select the best config to here.
                for BM in [64]  #
                for BN in [64]  #
                for BK in [64, 128]  #
                for s in ([2, 3])  #
                for w in [4]  #
                for SUBTILE in [True, False]  #
            ]
        elif capability == (9, 0):
            return [
                triton.Config(
                    dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=SUBTILE, occupancy=occ),
                    pre_hook=pre_hook,
                    num_stages=s,
                )
                for BM in [64, 128, 256]
                for BN in [64, 128, 256]
                for BK in [64]
                for SUBTILE in [False]
                for occ in [1, 2]
                for s in [2, 3]
            ]
        elif capability[0] == 8:
            # A100/A10 (sm_80/sm_86): no TMA, no clusters, num_ctas=1
            # Tuned: 128x128 BK=64 occ=2 stages=2 warps=4 is dominant winner
            return [
                triton.Config(
                    dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=False, occupancy=occ),
                    pre_hook=pre_hook,
                    num_stages=s,
                    num_warps=4,
                ) for BM in [128] for BN in [128] for BK in [32, 64] for occ in [2] for s in [2, 4]
            ]
        else:
            return [
                triton.Config(
                    {
                        "BLOCK_M": BM,
                        "BLOCK_N": BN,
                        "BLOCK_K": BK,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": SUBTILE,
                        "occupancy": 1,
                    },
                    num_stages=s,
                    num_warps=w,
                    pre_hook=pre_hook,
                )  #
                # To reduce the CI time, we pre-tune the whole config sets in local gb100 850w machine and select the best config to here.
                for BM in [128, 256]  #
                for BN in [256]  #
                for BK in [64]  #
                for s in ([2, 3])  #
                for w in [4]  #
                for SUBTILE in [True, False]  #
            ]
    else:
        if capability in [(12, 0), (12, 1)]:
            return [
                triton.Config(
                    {
                        "BLOCK_M": BM,
                        "BLOCK_N": BN,
                        "BLOCK_K": BK,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": SUBTILE,
                        "occupancy": occ,
                    },
                    pre_hook=pre_hook,
                    num_stages=s,
                )  #
                for BM in [64, 128]  #
                for BN in [64, 128]  #
                for BK in [32, 64]  #
                for SUBTILE in [False]  #
                for occ in [1, 2, 4]
                for s in [10]
            ]
        elif capability == (9, 0):
            return [
                triton.Config(
                    dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=SUBTILE, occupancy=occ),
                    pre_hook=pre_hook,
                    num_stages=s,
                    num_warps=w,
                    num_ctas=c,
                )
                for BM in [256]
                for BN in [128]
                for BK in [64]
                for SUBTILE in [False]
                for occ in [2]
                for s in [2, 3]
                for w in [4]
                for c in [2]
            ]
        elif capability[0] == 8:
            # A100/A10 (sm_80/sm_86): no TMA, no clusters, num_ctas=1
            # Tuned: 128x128 BK=64 occ=2 stages=2 warps=4 is dominant winner
            return [
                triton.Config(
                    dict(BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=False, occupancy=occ),
                    pre_hook=pre_hook,
                    num_stages=s,
                    num_warps=4,
                ) for BM in [128] for BN in [128] for BK in [32, 64] for occ in [2] for s in [2, 4]
            ]
        else:
            return [
                triton.Config(
                    dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GROUP_SIZE_M=8, EPILOGUE_SUBTILE=False, occupancy=1),
                    pre_hook=pre_hook,
                    num_ctas=2,
                    num_stages=s,
                ) for s in [3, 4, 5]
            ]


# Device Kernels

# Adapted from https://github.com/openai/triton

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#     ],
#     key=['M', 'N', 'K'],
# )


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                      num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
})
@triton.jit
def _bmm_kernel_naive(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_aq,
    stride_am,
    stride_ak,
    stride_bq,
    stride_bk,
    stride_bn,
    stride_cq,
    stride_cm,
    stride_cn,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the bmm C = A x B.
    A has shape (Q, M, K), B has shape (Q, K, N) and C has shape (Q, M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # a_ptrs is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # b_ptrs is a block of [BLOCK_SIZE_K, BLOCK_SIZE_n] pointers
    # see above `Pointer Arithmetics` section for details
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    pid_q = tl.program_id(axis=1)
    if transpose_a:
        a_ptrs = a_ptr + pid_q * stride_aq + offs_am[:, None] * stride_ak + offs_k[None, :] * stride_am

    else:
        a_ptrs = a_ptr + pid_q * stride_aq + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    if transpose_b:
        b_ptrs = b_ptr + pid_q * stride_bq + offs_k[:, None] * stride_bn + offs_bn[None, :] * stride_bk
    else:
        b_ptrs = b_ptr + pid_q * stride_bq + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        # We accumulate along the K dimension
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block
        if transpose_a:
            a_ptrs += BLOCK_SIZE_K * stride_am
        else:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if transpose_b:
            b_ptrs += BLOCK_SIZE_K * stride_bn
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk
    # you can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION:
        accumulator = ACTIVATION(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    pid_q = tl.program_id(axis=1)
    c_ptrs = c_ptr + pid_q * stride_cq + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=_bmm_memref_get_configs(pre_hook=_bmm_memref_set_block_size_hook),
    key=["Q", "M", "N", "K", "transpose_a", "transpose_b"],
)
@triton.jit
def _bmm_kernel_memref(
    a_ptr,
    b_ptr,
    c_ptr,  #
    Q,
    M,
    N,
    K,  #
    stride_aq,
    stride_am,
    stride_ak,  #
    stride_bq,
    stride_bk,
    stride_bn,  #
    stride_cq,
    stride_cm,
    stride_cn,  #
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    occupancy: tl.constexpr,
):
    if isinstance(a_ptr, tl.tensor_descriptor):
        dtype = a_ptr.type.block_type.element_ty
    else:
        dtype = c_ptr.dtype.element_ty
    pid = tl.program_id(axis=0)

    # Create tensor descriptors for a, b, c
    if transpose_a:
        # For transpose_a, we need to swap M and K dimensions
        a_desc = _maybe_make_tensor_desc(
            a_ptr,
            shape=[Q, K, M],
            strides=[stride_aq, stride_am, stride_ak],
            block_shape=[1, BLOCK_K, BLOCK_M],
        )
    else:
        a_desc = _maybe_make_tensor_desc(
            a_ptr,
            shape=[Q, M, K],
            strides=[stride_aq, stride_am, stride_ak],
            block_shape=[1, BLOCK_M, BLOCK_K],
        )

    if transpose_b:
        # For transpose_b, we need to swap K and N dimensions
        b_desc = _maybe_make_tensor_desc(
            b_ptr,
            shape=[Q, N, K],
            strides=[stride_bq, stride_bk, stride_bn],
            block_shape=[1, BLOCK_N, BLOCK_K],
        )
    else:
        b_desc = _maybe_make_tensor_desc(
            b_ptr,
            shape=[Q, K, N],
            strides=[stride_bq, stride_bk, stride_bn],
            block_shape=[1, BLOCK_K, BLOCK_N],
        )

    if EPILOGUE_SUBTILE:
        c_desc = _maybe_make_tensor_desc(
            c_ptr,
            shape=[Q, M, N],
            strides=[stride_cq, stride_cm, stride_cn],
            block_shape=[1, BLOCK_M, BLOCK_N // 2],
        )
    else:
        c_desc = _maybe_make_tensor_desc(
            c_ptr,
            shape=[Q, M, N],
            strides=[stride_cq, stride_cm, stride_cn],
            block_shape=[1, BLOCK_M, BLOCK_N],
        )

    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N) * Q
    num_programs = tl.num_programs(0)
    # for loop for static scheduling
    for current_pid in tl.range(pid, total_tiles, num_programs, flatten=True):
        pid_q, pid_m, pid_n = _bmm_calculate_pid(current_pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if transpose_a:
                # Load and transpose: from [1, BLOCK_K, BLOCK_M] to [1, BLOCK_M, BLOCK_K]
                a = a_desc.load([pid_q * 1, k * BLOCK_K, pid_m * BLOCK_M])
                a = tl.trans(a, (0, 2, 1))
            else:
                a = a_desc.load([pid_q * 1, pid_m * BLOCK_M, k * BLOCK_K])

            if transpose_b:
                # Load and transpose: from [1, BLOCK_N, BLOCK_K] to [1, BLOCK_K, BLOCK_N]
                b = b_desc.load([pid_q * 1, pid_n * BLOCK_N, k * BLOCK_K])
                b = tl.trans(b, (0, 2, 1))
            else:
                b = b_desc.load([pid_q * 1, k * BLOCK_K, pid_n * BLOCK_N])

            # Reshape 3D tensors to 2D for tl.dot (mmaV5 expects 2D inputs)
            a_2d = tl.reshape(a, (BLOCK_M, BLOCK_K))
            b_2d = tl.reshape(b, (BLOCK_K, BLOCK_N))
            acc = tl.dot(a_2d, b_2d, acc)

        pid_q, pid_m, pid_n = _bmm_calculate_pid(current_pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M)

        # Reshape 2D accumulator back to 3D for epilogue processing
        acc = tl.reshape(acc, (1, BLOCK_M, BLOCK_N))
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(acc, (1, BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 1, 3, 2))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([pid_q * 1, pid_m * BLOCK_M, pid_n * BLOCK_N], c0)
            c1 = acc1.to(dtype)
            c_desc.store([pid_q * 1, pid_m * BLOCK_M, pid_n * BLOCK_N + BLOCK_N // 2], c1)
        else:
            c = acc.to(dtype)
            c_desc.store([pid_q * 1, pid_m * BLOCK_M, pid_n * BLOCK_N], c)


# Host Launchers & Public API


def bmm_fn(a, b, transpose_a=False, transpose_b=False):
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    assert Q_A == Q_B, "incompatible dimensions"
    K = K_A
    Q = Q_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    # allocates output
    c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        Q,
    )
    _bmm_kernel_naive[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        transpose_a,
        transpose_b,
        ACTIVATION=None,
    )
    return c


@register_impl("bmm", backend="triton")
def bmm_memref(a, b, transpose_a=False, transpose_b=False, static_persistent=None, **kwargs):
    if static_persistent is None:
        static_persistent = True
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    assert Q_A == Q_B, "incompatible dimensions"
    K = K_A
    Q = Q_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)
    assert static_persistent, "only support static persistent mode"

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Add host descriptor support
    if _supports_host_descriptor():
        dummy_block = [1, 1, 1]
        a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    else:
        a_desc = a
        b_desc = b
        c_desc = c

    # get grid size for static persistence
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    if get_available_triton_backend() == "oait":
        num_ctas = 1
    else:
        cap = torch.cuda.get_device_capability()
        if cap in [(12, 0), (12, 1)] or cap[0] < 9:
            num_ctas = 1  # Clusters require sm_90+
        else:
            num_ctas = 2

    grid = lambda META: (min(
        NUM_SMS // num_ctas,
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * Q,
    ) * META["occupancy"], )
    _bmm_kernel_memref[grid](
        a_desc,
        b_desc,
        c_desc,  #
        Q,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )

    return c


class _BMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):
        c = bmm_memref(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        da = bmm_memref(dy, b, transpose_b=True)
        db = bmm_memref(a, dy, transpose_a=True)
        return da, db


def bmm_memref_fwd_bwd(a, b):
    return _BMM.apply(a, b)
