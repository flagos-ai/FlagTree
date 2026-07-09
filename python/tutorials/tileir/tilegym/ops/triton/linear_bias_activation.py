# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports

import itertools
import math
import os
from typing import Optional
from typing import Union

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import register_impl
from tilegym.ops.triton.activation.gelu import gelu_backward
from tilegym.ops.triton.activation.gelu import gelu_tanh_forward
from tilegym.ops.triton.activation.relu import relu_backward
from tilegym.ops.triton.activation.relu import relu_forward
from tilegym.ops.triton.activation.silu import silu_backward
from tilegym.ops.triton.activation.silu import silu_forward

from .matmul_perf_model import early_config_prune
from .matmul_perf_model import estimate_matmul_time

# Constants & Type Aliases

supported_act_types = {
    "relu": {"fwd": relu_forward, "bwd": relu_backward},
    "gelu": {"fwd": gelu_tanh_forward, "bwd": gelu_backward},
    "silu": {"fwd": silu_forward, "bwd": silu_backward},
}


@triton.jit
def _act_bwd_func(act_inp, dy, ACT_TYPE: tl.constexpr):
    if ACT_TYPE == "gelu":
        return gelu_backward(act_inp, dy)
    if ACT_TYPE == "relu":
        return relu_backward(act_inp, dy)
    if ACT_TYPE == "silu":
        return silu_backward(act_inp, dy)
    return dy


# Autotune Config


def _init_to_zero(name):
    return lambda nargs: nargs[name].zero_() if torch.is_tensor(nargs[name]) else None


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    # Set block shapes for input tensors
    if not isinstance(nargs.get("A"), TensorDescriptor):
        return

    # A tensor (input): [M, K] for normal, [K, M] for transpose
    if nargs.get("transpose_a", False):
        nargs["A"].block_shape = [BLOCK_K, BLOCK_M]
    else:
        nargs["A"].block_shape = [BLOCK_M, BLOCK_K]

    # B tensor (weight): [K, N] for normal, [N, K] for transpose
    if nargs.get("transpose_b", False):
        nargs["B"].block_shape = [BLOCK_N, BLOCK_K]
    else:
        nargs["B"].block_shape = [BLOCK_K, BLOCK_N]

    # C tensor (output): [M, N]
    nargs["C"].block_shape = [BLOCK_M, BLOCK_N]

    # Act_in tensor: [M, N]
    if "Act_in" in nargs and isinstance(nargs["Act_in"], TensorDescriptor):
        nargs["Act_in"].block_shape = [BLOCK_M, BLOCK_N]

    # Dropout_mask tensor: [M, N]
    if "Dropout_mask" in nargs and isinstance(nargs["Dropout_mask"], TensorDescriptor):
        nargs["Dropout_mask"].block_shape = [BLOCK_M, BLOCK_N]


def _host_descriptor_bias_bwd_pre_hook(nargs):
    # Initialize bias to zero for backward pass
    _init_to_zero("Bias")(nargs)

    # Set up TMA descriptors
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]

    # Set block shapes for input tensors
    if not isinstance(nargs.get("A"), TensorDescriptor):
        return

    # A tensor (input): [M, K] for normal, [K, M] for transpose
    if nargs.get("transpose_a", False):
        nargs["A"].block_shape = [BLOCK_K, BLOCK_M]
    else:
        nargs["A"].block_shape = [BLOCK_M, BLOCK_K]

    # B tensor (weight): [K, N] for normal, [N, K] for transpose
    if nargs.get("transpose_b", False):
        nargs["B"].block_shape = [BLOCK_N, BLOCK_K]
    else:
        nargs["B"].block_shape = [BLOCK_K, BLOCK_N]

    # C tensor (output): [M, N]
    nargs["C"].block_shape = [BLOCK_M, BLOCK_N]

    # Bias tensor: [N]
    if "Bias" in nargs and isinstance(nargs["Bias"], TensorDescriptor):
        nargs["Bias"].block_shape = [BLOCK_N]

    # Act_in tensor: [M, N]
    if "Act_in" in nargs and isinstance(nargs["Act_in"], TensorDescriptor):
        nargs["Act_in"].block_shape = [BLOCK_M, BLOCK_N]

    # Dropout_mask tensor: [M, N]
    if "Dropout_mask" in nargs and isinstance(nargs["Dropout_mask"], TensorDescriptor):
        nargs["Dropout_mask"].block_shape = [BLOCK_M, BLOCK_N]


def _matmul_act_bias_bwd_get_configs_io_bound():
    configs = []
    for num_stages in [2, 6]:
        for block_m in [
                16,
        ]:
            for block_k in [
                    32,
            ]:
                for block_n in [
                        32,
                ]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k),
                            num_stages=num_stages,
                            num_warps=num_warps,
                            pre_hook=_host_descriptor_bias_bwd_pre_hook,
                        ))
    return configs


def _get_configs_io_bound():
    configs = []
    for num_stages in [2, 3]:
        for block_m in [
                32,
        ]:
            for block_k in [32, 64]:
                for block_n in [
                        128,
                ]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config(
                            dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k),
                            num_stages=num_stages,
                            num_warps=num_warps,
                            pre_hook=_host_descriptor_pre_hook,
                        ))
    return configs


matmul_act_bias_bwd_autotune = triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config(
            dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32),
            num_stages=3,
            num_warps=8,
            pre_hook=_host_descriptor_bias_bwd_pre_hook,
        ),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # good for int8
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2, pre_hook=_host_descriptor_bias_bwd_pre_hook),
    ] + _matmul_act_bias_bwd_get_configs_io_bound(),
    key=["M", "N", "K", "align_a", "align_b"],
    prune_configs_by={
        "early_config_prune": early_config_prune,
        "perf_model": estimate_matmul_time,
        "top_k": 10,
    },
)

matmul_autotune = triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config(dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32), num_stages=3, num_warps=8,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=32), num_stages=3, num_warps=8,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=256, BLOCK_N=64, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=64, BLOCK_N=256, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=128, BLOCK_N=32, BLOCK_K=32), num_stages=4, num_warps=4,
                      pre_hook=_host_descriptor_pre_hook),
        triton.Config(dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=32), num_stages=5, num_warps=2,
                      pre_hook=_host_descriptor_pre_hook),
        # good for int8
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4, pre_hook=_host_descriptor_pre_hook),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2, pre_hook=_host_descriptor_pre_hook),
    ] + _get_configs_io_bound(),
    key=["M", "N", "K", "align_a", "align_b"],
    # prune_configs_by={
    #     'early_config_prune': early_config_prune,
    #     'perf_model': estimate_matmul_time,
    #     'top_k': 10,
    # },
)

matmul_act_bias_bwd_heuristics = triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
})

matmul_heuristics = triton.heuristics({
    "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
})

# Device Kernels


# TODO: add split_k support
@matmul_autotune
@matmul_heuristics
@triton.jit
def matmul_bias_activation_dropout_fwd(
    A,
    B,
    C,
    Bias,
    Act_in,
    Dropout_mask,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    align_a,
    align_b,
    dropout_seed,
    dropout_prob,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    IS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    IS_GRAD: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
):
    # Determine dtype from tensor descriptors or fallback to C dtype
    if isinstance(A, tl.tensor_descriptor):
        dtype = A.type.block_type.element_ty
    else:
        dtype = C.dtype.element_ty

    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # Calculate explicit offsets for tensor descriptors
    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k * BLOCK_K

        # Load using tensor descriptors with explicit offsets
        if transpose_a:
            # For transposed A: load (BLOCK_K, BLOCK_M) then transpose
            a = A.load([offset_k, offset_am])
            a = tl.trans(a)  # Transpose from (BLOCK_K, BLOCK_M) to (BLOCK_M, BLOCK_K)
        else:
            a = A.load([offset_am, offset_k])

        if transpose_b:
            # For transposed B: load (BLOCK_N, BLOCK_K) then transpose
            b = B.load([offset_bn, offset_k])
            b = tl.trans(b)  # Transpose from (BLOCK_N, BLOCK_K) to (BLOCK_K, BLOCK_N)
        else:
            b = B.load([offset_k, offset_bn])

        acc += tl.dot(a, b, out_dtype=dot_out_dtype)

    # Calculate output offsets
    offset_cm = pid_m * BLOCK_M
    offset_cn = pid_n * BLOCK_N

    # handles write-back
    store_acc = acc.to(dtype)
    if IS_BIAS:
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_mask = cols < N
        # Load bias using standard pointer access for now (bias is 1D)
        bias = tl.load(Bias + cols, mask=bias_mask)
        bias = bias[None, :]
        store_acc += bias
    if ACT_TYPE is not None:
        if IS_GRAD:
            Act_in.store([offset_cm, offset_cn], store_acc)
        if ACT_TYPE == "gelu":
            store_acc = gelu_tanh_forward(store_acc).to(dtype)
        elif ACT_TYPE == "relu":
            store_acc = relu_forward(store_acc).to(dtype)
        elif ACT_TYPE == "silu":
            store_acc = silu_forward(store_acc).to(dtype)
    if IS_DROPOUT:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        random = tl.rand(
            dropout_seed,
            rm[:, None] * stride_cm + rn[None, :] * stride_cn,
            n_rounds=7,
        )
        x_keep = random > dropout_prob
        dropout_0 = 0.0
        dropout_0 = dropout_0.to(dtype)

        Dropout_mask.store([offset_cm, offset_cn], x_keep.to(tl.int8))
        store_acc = tl.where(
            x_keep,
            (store_acc / (1 - dropout_prob)).to(dtype),
            dropout_0,
        )
    C.store([offset_cm, offset_cn], store_acc)


@triton.autotune(
    configs=[
        # fmt: off
        triton.Config(dict(BLOCK_M=32, BLOCK_N=32), num_stages=1, num_warps=4,
                      pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=16, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=1, num_warps=1, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=1, num_warps=1, pre_hook=_host_descriptor_bias_bwd_pre_hook),
        # fmt: on
    ] if not os.getenv("EXHAUSTIVE_BWD", False) else [
        triton.Config(
            dict(BLOCK_M=block_m, BLOCK_N=block_n),
            num_stages=num_stages,
            num_warps=num_warps,
            pre_hook=_host_descriptor_bias_bwd_pre_hook,
        ) for block_m, block_n, num_warps, num_stages in itertools.product([64, 128], [32, 128], [1, 4, 16], [1, 4])
    ],
    key=["M", "N"],
)
@triton.jit
def _kernel_bias_act_bwd(
    # fmt: off
    Dy,  # pointer to output gradient
    Bias,  # pointer to the biases gradient
    Act_in,  # pointer to store activation input for bprop
    M,  # number of rows in DY
    N,  # number of columns in DY
    input_row_stride,  # row stride
    input_col_stride,  # col stride
    IS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # fmt: on
):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    # TODO re-order program ID for better L2 performance
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # Calculate explicit offsets for tensor descriptors
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    dy = Dy.load([offset_m, offset_n])

    if ACT_TYPE is not None:
        act_inp = Act_in.load([offset_m, offset_n])
        dy = _act_bwd_func(act_inp, dy, ACT_TYPE)
        Dy.store([offset_m, offset_n], dy)

    if IS_BIAS:
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.sum(dy, axis=0)
        bias_mask = rn < N
        # Use standard pointer access for bias gradient accumulation (atomic add)
        tl.atomic_add(Bias + (rn * input_col_stride), acc, mask=bias_mask)


# TODO: add split_k support
@matmul_act_bias_bwd_autotune
@matmul_act_bias_bwd_heuristics
@triton.jit
def matmul_dropout_activation_bias_bwd(
    # fmt: off
    A,
    B,
    C,
    Bias,
    Act_in,
    Dropout_mask,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    align_a,
    align_b,
    dropout_prob,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    IS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    IS_DROPOUT: tl.constexpr,
    # fmt: on
):
    # Determine dtype from tensor descriptors or fallback to C dtype
    if isinstance(A, tl.tensor_descriptor):
        dtype = A.type.block_type.element_ty
    else:
        dtype = C.dtype.element_ty

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # Calculate explicit offsets for tensor descriptors
    offset_am = pid_m * BLOCK_M
    offset_bn = pid_n * BLOCK_N

    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k * BLOCK_K

        # Load using tensor descriptors with explicit offsets
        if transpose_a:
            # For transposed A: load (BLOCK_K, BLOCK_M) then transpose
            a = A.load([offset_k, offset_am])
            a = tl.trans(a)  # Transpose from (BLOCK_K, BLOCK_M) to (BLOCK_M, BLOCK_K)
        else:
            a = A.load([offset_am, offset_k])

        if transpose_b:
            # For transposed B: load (BLOCK_N, BLOCK_K) then transpose
            b = B.load([offset_bn, offset_k])
            b = tl.trans(b)  # Transpose from (BLOCK_N, BLOCK_K) to (BLOCK_K, BLOCK_N)
        else:
            b = B.load([offset_k, offset_bn])

        acc += tl.dot(a, b, out_dtype=dot_out_dtype)

    # Calculate output offsets
    offset_cm = pid_m * BLOCK_M
    offset_cn = pid_n * BLOCK_N

    # handles write-back
    store_acc = acc.to(dtype)
    if IS_DROPOUT:
        x_keep = Dropout_mask.load([offset_cm, offset_cn])
        dropout_0 = 0.0
        dropout_0 = dropout_0.to(dtype)
        store_acc = tl.where(
            x_keep,
            (store_acc / (1.0 - dropout_prob)).to(dtype),
            dropout_0,
        )
    if ACT_TYPE is not None:
        act_in = Act_in.load([offset_cm, offset_cn])
        if ACT_TYPE == "gelu":
            store_acc = gelu_backward(act_in, store_acc).to(dtype)
        elif ACT_TYPE == "relu":
            store_acc = relu_backward(act_in, store_acc).to(dtype)
        elif ACT_TYPE == "silu":
            store_acc = silu_backward(act_in, store_acc).to(dtype)
    if IS_BIAS:
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_mask = cols < N
        dacc_sum = tl.sum(store_acc, axis=0)
        # Use standard pointer access for bias gradient accumulation (atomic add)
        tl.atomic_add(Bias + cols, dacc_sum.to(tl.float32), mask=bias_mask)
    C.store([offset_cm, offset_cn], store_acc)


# Host Helpers


def get_tensor_alignment(tensor):
    address = tensor.data_ptr()
    alignment = (((address - 1) ^ address) + 1) >> 1
    return alignment


def allocate_tma_aligned_2d(shape, dtype, device):
    """Allocate a 2D tensor whose row stride times element size is 16-byte aligned.

    ``TensorDescriptor`` (TMA) asserts ``(stride * elem_bytes) % 16 == 0`` for
    every non-innermost dimension. For 2D contiguous tensors that means the
    inner dimension ``N`` must satisfy ``N * itemsize % 16 == 0``. When ``N``
    does not already meet this constraint (e.g. fp32 with ``N=258``), allocate
    padded storage on the inner dimension and return a sliced view with the
    requested logical ``shape`` but a padded ``stride(0)``. The returned tensor
    is non-contiguous when padding is applied; call ``.contiguous()`` before
    passing it to kernels that require contiguity.
    """
    assert len(shape) == 2, "allocate_tma_aligned_2d expects a 2D shape"
    m, n = shape
    itemsize = torch.empty(0, dtype=dtype).element_size()
    if (n * itemsize) % 16 == 0:
        return torch.empty((m, n), dtype=dtype, device=device)
    # Pad inner dim so padded_n * itemsize is a multiple of 16 bytes.
    align_elems = 16 // math.gcd(16, itemsize)
    padded_n = ((n + align_elems - 1) // align_elems) * align_elems
    return torch.empty((m, padded_n), dtype=dtype, device=device)[:, :n]


def pad_for_tma_2d(t):
    """Return ``t`` (or a padded copy of it) with a 16-byte aligned row stride.

    Used to wrap a caller-provided 2D tensor before constructing a
    ``TensorDescriptor``. When ``t`` is already row-major with a 16-byte
    aligned ``stride(0)`` it is returned unchanged; otherwise a padded buffer
    is allocated, ``t``'s contents are copied into it, and a non-contiguous
    view of the padded buffer with the original logical shape is returned.
    ``None`` is propagated.
    """
    if t is None:
        return t
    assert t.dim() == 2, "pad_for_tma_2d expects a 2D tensor"
    if t.stride(-1) != 1:
        # Non row-major; compact first so we own the inner stride.
        t = t.contiguous()
    itemsize = t.element_size()
    if (t.stride(0) * itemsize) % 16 == 0:
        return t
    m, n = t.shape
    align_elems = 16 // math.gcd(16, itemsize)
    padded_n = ((n + align_elems - 1) // align_elems) * align_elems
    padded = torch.empty((m, padded_n), dtype=t.dtype, device=t.device)
    padded[:, :n] = t
    return padded[:, :n]


# Host Launchers & Public API


@register_impl("linear_bias_activation", backend="triton")
def linear_bias_act_dropout(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    act_type: Union[str, None],
    is_grad_enabled: bool = True,
    dropout_seed: int = 1234,
    dropout_prob: float = 0.0,
    kernel_configs: dict = None,
):
    r"""
    Applies Linear (+Bias) + Activation + Dropout (Optional) on the input.
    Only supports limited activation types

    Args:
        input: Tensor
        weight: Tensor
        bias: Tensor or None
        act_type: String. Supported Activation types: ``relu``, ``gelu``. Default is None
        is_grad_enabled: Bool
        dropout_seed: Int
        dropout_prob: Float

    Shape:
        input: (*, in_features)
        weight: (out_features, in_features)
        bias: (out_features) or None
    """
    assert act_type is None or act_type in supported_act_types.keys(), (
        f"Activation Type: {act_type} is not supported by the fused linear layer")
    return _LinearBiasActDropout.apply(
        input,
        weight,
        bias,
        act_type,
        is_grad_enabled,
        dropout_seed,
        dropout_prob,
    )


def _linear_bias_act_dropout_fwd_fn(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    act_type: Union[str, None],
    is_grad_enabled: bool,
    dropout_seed: int,
    dropout_prob: float,
):
    M, K_A = input.shape
    N, K_B = weight.shape
    # K_B, N = w.shape
    assert act_type is None or act_type in supported_act_types.keys(), (
        f"Activation Type: {act_type} is not supported by the fused linear layer")
    assert K_A == K_B, "incompatible dimensions"
    K = K_A
    assert input.is_contiguous(), "matrix input must be contiguous"
    assert weight.is_contiguous(), "matrix weight must be contiguous"
    # allocates output
    c = torch.empty((M, N), device=input.device, dtype=input.dtype)

    act_inp = None
    is_dropout = False
    dropout_mask = None
    if is_grad_enabled:
        if act_type is not None:
            act_inp = torch.empty((M, N), device=input.device, dtype=input.dtype)
        if dropout_prob > 0.0:
            is_dropout = True
            # here use int8 because tma descriptor does not support bool, also torch bool/int8 both occupy 1 byte (8 bits).
            # see https://discuss.pytorch.org/t/why-are-torch-bool-s-elements-1-byte-and-not-1-bit/204299
            dropout_mask = torch.empty((M, N), device=input.device, dtype=torch.int8)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Add host descriptor support
    dummy_block = [1, 1]
    desc_input = TensorDescriptor(input, input.shape, input.stride(), dummy_block)
    desc_weight = TensorDescriptor(weight, weight.shape, weight.stride(), dummy_block)
    desc_c = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    # Create descriptors for optional tensors
    desc_act_inp = (TensorDescriptor(act_inp, act_inp.shape, act_inp.stride(), dummy_block)
                    if act_inp is not None else act_inp)
    desc_dropout_mask = (TensorDescriptor(
        dropout_mask,
        dropout_mask.shape,
        dropout_mask.stride(),
        dummy_block,
    ) if dropout_mask is not None else dropout_mask)

    # 2D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    matmul_bias_activation_dropout_fwd[grid](
        desc_input,
        desc_weight,
        desc_c,
        bias,
        desc_act_inp,
        desc_dropout_mask,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        c.stride(0),
        c.stride(1),
        get_tensor_alignment(input) % 32,
        get_tensor_alignment(weight) % 32,
        dropout_seed,
        dropout_prob,
        transpose_a=False,
        transpose_b=True,
        dot_out_dtype=tl.float32,
        GROUP_M=8,
        IS_BIAS=torch.is_tensor(bias),
        ACT_TYPE=act_type,
        IS_GRAD=is_grad_enabled,
        IS_DROPOUT=is_dropout,
    )
    return c, act_inp, dropout_mask


class _LinearBiasActDropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, w, bias, act_type, is_grad_enabled, dropout_seed, dropout_prob):
        if a.dim() > 2:
            inp_shape = a.shape
            inp = a.view(-1, inp_shape[-1])
        else:
            inp = a
        out, act_inp, dropout_mask = _linear_bias_act_dropout_fwd_fn(inp, w, bias, act_type, is_grad_enabled,
                                                                     dropout_seed, dropout_prob)
        if a.dim() > 2:
            out = out.view(*inp_shape[:-1], out.shape[-1])
        if is_grad_enabled:
            if torch.is_tensor(bias):
                ctx.is_bias = True
            else:
                ctx.is_bias = False
            ctx.is_dropout = dropout_prob > 0.0
            ctx.dropout_prob = dropout_prob
            ctx.act_type = act_type
            ctx.save_for_backward(inp, w, bias, act_inp, dropout_mask)
        return out

    @staticmethod
    def backward(ctx, dy):
        pass
