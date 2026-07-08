# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Imports
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from tilegym.backend import get_available_triton_backend
from tilegym.backend import mark_perf_ready
from tilegym.backend import register_impl
from tilegym.logger import get_logger

# Constants & Type Aliases

logger = get_logger(__name__)


def _disable_autotune():
    return os.getenv("TILEGYM_DISABLE_AUTOTUNE", "0") == "1"


# Capability Probe


def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend in [
        "cuda",
        "tileir",
    ]


def _supports_host_descriptor():
    return _is_cuda() and torch.cuda.get_device_capability()[0] >= 8


# Kernel Helpers


@triton.jit
def _swizzle_2d(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# Autotune Config


def _get_cuda_autotune_config(pre_hook=None):
    if get_available_triton_backend() == "nvt":
        config = [
            triton.Config(dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=4),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, occupancy=2),
                          num_stages=5),
        ]
    else:
        config = [
            triton.Config(dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=5,
                          num_warps=2),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=5,
                          num_warps=2),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                          num_warps=4),
            triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                          num_warps=4),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                          num_warps=4),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                          num_warps=4),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=4,
                          num_warps=4),
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8), num_stages=3,
                          num_warps=8),
        ]
    return config


def _get_pure_ptr_autotune_config():
    if torch.cuda.get_device_capability() == (8, 0):
        return _get_cuda_autotune_config(pre_hook=None)
    else:
        config = [
            triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8), num_stages=3)
        ]
        return config


def _host_descriptor_pre_hook(nargs):
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", False)
    BLOCK_SIZE_M = nargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = nargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = nargs["BLOCK_SIZE_K"]
    if not isinstance(nargs.get("A"), TensorDescriptor):
        return
    if nargs.get("transpose_a", False):
        nargs["A"].block_shape = [BLOCK_SIZE_K, BLOCK_SIZE_M]
    else:
        nargs["A"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_K]
    if nargs.get("transpose_b", False):
        nargs["B"].block_shape = [BLOCK_SIZE_N, BLOCK_SIZE_K]
    else:
        nargs["B"].block_shape = [BLOCK_SIZE_K, BLOCK_SIZE_N]
    nargs["C"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]
    if EPILOGUE_SUBTILE:
        nargs["C"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N // 2]
    else:
        nargs["C"].block_shape = [BLOCK_SIZE_M, BLOCK_SIZE_N]


# from https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py
def _matmul_get_configs(pre_hook=None):
    if get_available_triton_backend() == "oait":
        # fmt: off
        # default config from triton
        config = [
            triton.Config(dict(BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_SIZE_M=8), num_stages=s, num_warps=w, pre_hook=pre_hook)
            for BM in [128]
            for BN in [128, 256]
            for BK in [64, 128]
            for s in [3, 4]
            for w in [4, 8]
        ]
        # some configs for fp32
        config.extend(
            [
                triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=8), num_stages=2, num_warps=4, pre_hook=pre_hook),
                triton.Config(dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8), num_stages=2, num_warps=4, pre_hook=pre_hook),
                triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8), num_stages=2, num_warps=4, pre_hook=pre_hook),
                triton.Config(dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8), num_stages=2, num_warps=4, pre_hook=pre_hook),
            ]
        )
        return config
    if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
        # Default config from triton
        config = [
            triton.Config(
                dict(BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_SIZE_M=8, occupancy=occ),
                pre_hook=pre_hook,
            )
            for BM in [64, 128]
            for BN in [64]
            for BK in [64, 32]
            for occ in [1, 2, 4]
        ]
        config.extend(
            [
                triton.Config(
                    dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8), pre_hook=pre_hook
                )
            ]
        )
        return config
    elif torch.cuda.get_device_capability()[0] == 8:
        # SM80 (A100): acc_regs = BM*BN/(warps*32) must be <= 255.
        # Restrict to safe tiles; use fewer stages to fit SMEM.
        return [
            triton.Config(
                {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 8, "occupancy": 1},
                pre_hook=pre_hook,
                num_ctas=1,
                num_warps=8,
                num_stages=s,
            )
            for BM, BN in [(128, 256), (256, 128), (128, 128)]
            for BK in [64, 128]
            for s in [2, 3, 4]
        ]
        # fmt: on
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": BM,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "occupancy": 1,
                },
                pre_hook=pre_hook,
                num_ctas=NUM_CTAS,
                num_warps=NUM_WARPS,
                # For the 256x256x64 tile, we use num_stages=5.
                num_stages=5,
            )
            for BM in [128, 256, 512]
            for BN in [128, 256, 512]
            for BK in [32, 64, 128]
            for NUM_CTAS in ([1, 2, 4] if torch.cuda.get_device_capability()[0] > 8 else [1])
            for NUM_WARPS in [4, 8]
        ]
        # fmt: on


def _prune_configs_by_shape(configs, named_args):
    """Drop autotune configs whose block shape vastly exceeds the problem shape.

    Background
    ----------
    The TMA matmul autotune grid on sm_90+ is
    ``3 BM * 3 BN * 3 BK * 3 NUM_CTAS * 2 NUM_WARPS = 162`` configs. The dtype
    whitelist below (``early_config_prune`` / ``early_config_prune_for_persistent``)
    reduces this sharply for fp16/fp32/fp8, but dtypes it doesn't cover (e.g.
    ``bf16``) keep the full grid. On small shapes such as M=N=K=64 every cold
    miss spawns a ``tileiras`` compile; at 162 variants this easily blows the
    300s pytest-timeout.

    Policy
    ------
    A tile larger than ``2 * dim`` contributes no extra parallelism
    (``cdiv(dim, block)`` is already 1), so we drop BLOCK_SIZE_{M,N,K} beyond
    that cap. To guarantee a non-empty result when the problem dim is smaller
    than the smallest legal block, the cap is floored to the minimum block
    size present in the surviving config list. If pruning would empty the list
    we fall back to the input configs unchanged.
    """
    if not configs:
        return configs
    M = named_args.get("M")
    N = named_args.get("N")
    K = named_args.get("K")
    # Only apply when we actually know the problem dims. If the kernel is ever
    # launched without these positional args (shouldn't happen, but defensive)
    # we leave the set untouched.
    if M is None or N is None or K is None:
        return configs

    def _kw(c, name, default):
        return c.kwargs.get(name, default)

    min_bm = min(_kw(c, "BLOCK_SIZE_M", 1) for c in configs)
    min_bn = min(_kw(c, "BLOCK_SIZE_N", 1) for c in configs)
    min_bk = min(_kw(c, "BLOCK_SIZE_K", 1) for c in configs)

    bm_cap = max(min_bm, 2 * M)
    bn_cap = max(min_bn, 2 * N)
    bk_cap = max(min_bk, 2 * K)

    pruned = [
        c for c in configs if _kw(c, "BLOCK_SIZE_M", 0) <= bm_cap and _kw(c, "BLOCK_SIZE_N", 0) <= bn_cap
        and _kw(c, "BLOCK_SIZE_K", 0) <= bk_cap
    ]
    return pruned or configs


def _early_config_prune(configs, named_args, **kwargs):
    if get_available_triton_backend() == "nvt" and torch.cuda.get_device_capability() not in [
        (12, 0),
        (12, 1),
    ]:
        dtype = kwargs.get("DTYPE")
        gpu_capability = torch.cuda.get_device_capability()
        supports_multi_cta = gpu_capability[0] > 8

        tune_space = []
        if dtype == torch.float32:
            # TODO: Remove (256, 256, 64, 2) after resolving other perf issues.  # {$nv-TODO}
            if supports_multi_cta:
                tune_space = [
                    (512, 256, 32, 4),
                    (256, 256, 32, 2),
                    (256, 256, 64, 2),
                    (128, 512, 32, 4),
                    (512, 512, 64, 4),
                    (256, 512, 32, 2),
                    (512, 256, 64, 2),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [32]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # For GPU capability < 9.0, only NUM_CTAS=1 is supported
                tune_space = [
                    (512, 256, 32, 1),
                    (256, 256, 32, 1),
                    (256, 256, 64, 1),
                    (128, 512, 32, 1),
                    (512, 512, 64, 1),
                    (256, 512, 32, 1),
                    (512, 256, 64, 1),
                ]
        elif dtype == torch.float16:
            if supports_multi_cta:
                tune_space = [
                    (128, 512, 64, 4),
                    (256, 256, 64, 1),
                    (256, 256, 64, 2),
                    (256, 256, 64, 4),
                    (256, 256, 32, 2),
                    (256, 512, 64, 2),
                    (512, 512, 128, 4),
                    (512, 256, 64, 2),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [64]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # SM80 (A100): restrict to safe tiles (acc_regs = BM*BN/(8*32) <= 255).
                tune_space = [
                    (128, 256, 64, 1),
                    (256, 128, 64, 1),
                    (128, 128, 64, 1),
                    (128, 256, 128, 1),
                    (256, 128, 128, 1),
                    (128, 128, 128, 1),
                ]
        elif dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if supports_multi_cta:
                tune_space = [
                    (256, 256, 64, 2),
                    (256, 256, 128, 4),
                    (128, 256, 128, 2),
                    (256, 256, 128, 1),
                    (256, 256, 128, 2),
                    (512, 128, 128, 4),
                    (256, 128, 128, 2),
                    (128, 128, 128, 1),
                    (512, 512, 128, 4),
                    (512, 256, 128, 2),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [128]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # For GPU capability < 9.0, only NUM_CTAS=1 is supported
                tune_space = [
                    (256, 256, 64, 1),
                    (256, 256, 128, 1),
                    (128, 256, 128, 1),
                    (512, 128, 128, 1),
                    (256, 128, 128, 1),
                    (128, 128, 128, 1),
                    (512, 512, 128, 1),
                    (512, 256, 128, 1),
                ]
        else:
            # Unknown dtype (e.g. bf16): skip the dtype whitelist, but still
            # apply shape-based pruning below to keep the autotune search
            # space reasonable on small problems.
            return _prune_configs_by_shape(configs, named_args)
        pruned_configs = []
        for config in configs:
            BLOCK_SIZE_M = config.kwargs.get("BLOCK_SIZE_M")
            BLOCK_SIZE_N = config.kwargs.get("BLOCK_SIZE_N")
            BLOCK_SIZE_K = config.kwargs.get("BLOCK_SIZE_K")
            NUM_CTAS = config.num_ctas
            NUM_WARPS = config.num_warps
            # Support both 4-tuples and 5-tuples (with num_warps)
            if (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_CTAS) in tune_space or (
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    NUM_CTAS,
                    NUM_WARPS,
            ) in tune_space:
                pruned_configs.append(config)
        return _prune_configs_by_shape(pruned_configs, named_args)
    else:
        return _prune_configs_by_shape(configs, named_args)


def _matmul_tma_persistent_get_configs(pre_hook=None):
    # fmt: off
    if get_available_triton_backend() == "oait":
        if torch.cuda.get_device_capability() == (10, 0):
            # Keep persistent transpose-A candidates below the sm_100 shared-memory limit.
            return [
                triton.Config(
                    {
                        'BLOCK_SIZE_M': BM,
                        'BLOCK_SIZE_N': BN,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": SUBTILE,
                        "occupancy": 1,
                    },
                    num_stages=2,
                    num_warps=w,
                    pre_hook=pre_hook,
                )
                for BM in [64, 128]
                for BN in [64, 128]
                for w in [4, 8]
                for SUBTILE in [True, False]
            ]
        return [
            triton.Config(
                {
                    'BLOCK_SIZE_M': BM,
                    'BLOCK_SIZE_N': BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": SUBTILE,
                    "occupancy": 1,
                },
                num_stages=s,
                num_warps=w,
                pre_hook=pre_hook,
            )  #
            for BM in [128]
            for BN in [128, 256]
            for BK in [64, 128]
            for s in ([2, 3, 4])
            for w in [4, 8]
            for SUBTILE in [True, False]
        ]
    if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
        configs = [
            triton.Config(
                {
                    'BLOCK_SIZE_M': BM,
                    'BLOCK_SIZE_N': BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "EPILOGUE_SUBTILE": SUBTILE,
                    "occupancy": occ,
                },
                pre_hook=pre_hook,
            )  #
            for BM in [64, 128]
            for BN in [64]
            for BK in [64]
            for SUBTILE in [True, False]
            for occ in [1, 2, 4]
        ]

        configs.extend(
            [
                triton.Config(
                    {
                        'BLOCK_SIZE_M': 256,
                        'BLOCK_SIZE_N': 256,
                        "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": False,
                        "occupancy": 1,
                    },
                    pre_hook=pre_hook,
                )
            ]
        )
        return configs
    elif torch.cuda.get_device_capability()[0] == 8:
        # SM80 (A100): same register-safety constraint as _matmul_get_configs.
        # Also explore EPILOGUE_SUBTILE which OAI selects for persistent kernels.
        return [
            triton.Config(
                {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK,
                 'GROUP_SIZE_M': 8, 'EPILOGUE_SUBTILE': subtile, 'occupancy': 1},
                pre_hook=pre_hook, num_ctas=1, num_warps=8, num_stages=s,
            )
            for BM, BN in [(128, 256), (256, 128), (128, 128)]
            for BK in [64, 128]
            for s in [2, 3, 4]
            for subtile in [True, False]
        ]
    else:
        # For dev and debug
        if _disable_autotune():
            return [
                triton.Config(
                    {
                        'BLOCK_SIZE_M': 256,
                        'BLOCK_SIZE_N': 256,
                        "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": False,
                        "occupancy": 1
                    },
                    num_stages=5,
                    pre_hook=pre_hook,
                    num_ctas=2,
                )
            ]
        else:
            return [
                triton.Config(
                    {
                        'BLOCK_SIZE_M': BM,
                        'BLOCK_SIZE_N': BN,
                        "BLOCK_SIZE_K": BK,
                        "GROUP_SIZE_M": 8,
                        "EPILOGUE_SUBTILE": False,
                        "occupancy": 1},
                    pre_hook=pre_hook,
                    num_ctas=NUM_CTAS,
                    num_warps=NUM_WARPS,
                    # For the 256x256x64 tile, we use num_stages=5.
                    num_stages=5,
                )
                for BM in [128, 256, 512]
                for BN in [128, 256, 512]
                for BK in [32, 64, 128]
                for NUM_CTAS in ([1, 2, 4] if torch.cuda.get_device_capability()[0] > 8 else [1])
                for NUM_WARPS in [4, 8]
            ]
            # fmt: on


def _early_config_prune_for_persistent(configs, named_args, **kwargs):
    if get_available_triton_backend() == "nvt" and torch.cuda.get_device_capability() not in [
        (12, 0),
        (12, 1),
    ]:
        dtype = kwargs.get("DTYPE")
        gpu_capability = torch.cuda.get_device_capability()
        supports_multi_cta = gpu_capability[0] > 8

        tune_space = []
        if dtype == torch.float32:
            # TODO: Remove (256, 256, 64, 2) after resolving other perf issues.  # {$nv-TODO}
            if supports_multi_cta:
                tune_space = [
                    (512, 256, 32, 4),
                    (256, 256, 32, 2),
                    (256, 256, 64, 2),
                    (128, 512, 32, 4),
                    (256, 256, 128, 2),
                    (256, 256, 64, 1),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [32]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # For GPU capability < 9.0, only NUM_CTAS=1 is supported
                tune_space = [
                    (512, 256, 32, 1),
                    (256, 256, 32, 1),
                    (256, 256, 64, 1),
                    (128, 512, 32, 1),
                    (256, 256, 128, 1),
                ]
        elif dtype == torch.float16:
            if supports_multi_cta:
                tune_space = [
                    (128, 512, 64, 4),
                    (256, 256, 64, 2),
                    (256, 256, 64, 1),
                    (256, 256, 64, 4),
                    (256, 256, 128, 2),
                    (512, 256, 32, 4),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [64]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # SM80 (A100): restrict to safe tiles (acc_regs = BM*BN/(8*32) <= 255).
                tune_space = [
                    (128, 256, 64, 1),
                    (256, 128, 64, 1),
                    (128, 128, 64, 1),
                    (128, 256, 128, 1),
                    (256, 128, 128, 1),
                    (128, 128, 128, 1),
                ]
        elif dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if supports_multi_cta:
                tune_space = [
                    (256, 256, 128, 4),
                    (128, 256, 128, 2),
                    (256, 256, 128, 2),
                    (512, 128, 128, 4),
                    (256, 128, 128, 2),
                    (256, 256, 64, 2),
                    (256, 256, 32, 2),
                ]
                if gpu_capability == (9, 0):
                    block_m_sizes = [128, 256]
                    block_n_sizes = [128, 256]
                    block_k_sizes = [128]
                    num_ctas = [1]
                    num_warps = [8]
                    tune_space = [(bm, bn, bk, nc, nw)
                                  for bm in block_m_sizes
                                  for bn in block_n_sizes
                                  for bk in block_k_sizes
                                  for nc in num_ctas
                                  for nw in num_warps]
            else:
                # For GPU capability < 9.0, only NUM_CTAS=1 is supported
                tune_space = [
                    (256, 256, 128, 1),
                    (128, 256, 128, 1),
                    (512, 128, 128, 1),
                    (256, 128, 128, 1),
                    (256, 256, 64, 1),
                    (256, 256, 32, 1),
                ]
        else:
            # Unknown dtype (e.g. bf16): skip the dtype whitelist, but still
            # apply shape-based pruning below to keep the autotune search
            # space reasonable on small problems.
            return _prune_configs_by_shape(configs, named_args)
        pruned_configs = []
        for config in configs:
            BLOCK_SIZE_M = config.kwargs.get("BLOCK_SIZE_M")
            BLOCK_SIZE_N = config.kwargs.get("BLOCK_SIZE_N")
            BLOCK_SIZE_K = config.kwargs.get("BLOCK_SIZE_K")
            NUM_CTAS = config.num_ctas
            NUM_WARPS = config.num_warps
            # Support both 4-tuples and 5-tuples (with num_warps)
            if (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_CTAS) in tune_space or (
                    BLOCK_SIZE_M,
                    BLOCK_SIZE_N,
                    BLOCK_SIZE_K,
                    NUM_CTAS,
                    NUM_WARPS,
            ) in tune_space:
                pruned_configs.append(config)
        return _prune_configs_by_shape(pruned_configs, named_args)
    else:
        return _prune_configs_by_shape(configs, named_args)


# Device Kernels


@triton.autotune(
    configs=_get_pure_ptr_autotune_config(),
    key=["M", "N", "K", "transpose_a", "transpose_b"],
)
@triton.jit
def _matmul_kernel_pure_ptr(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
):  # fmt: skip
    # Initialize offsets
    pid_m, pid_n = _swizzle_2d(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if transpose_a:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_ak + offs_k[None, :] * stride_am)
    else:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    if transpose_b:
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bn + offs_bn[None, :] * stride_bk)
    else:
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # KV loop
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        if transpose_a:
            a_ptrs += BLOCK_SIZE_K * stride_am
        else:
            a_ptrs += BLOCK_SIZE_K * stride_ak
        if transpose_b:
            b_ptrs += BLOCK_SIZE_K * stride_bn
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    # store output c
    if ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=_matmul_get_configs(pre_hook=_host_descriptor_pre_hook),
    key=["M", "N", "K", "transpose_a", "transpose_b", "DTYPE"],
    prune_configs_by={"early_config_prune": _early_config_prune},
)
@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid_m, pid_n = _swizzle_2d(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    num_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    dtype = C.type.block_type.element_ty

    for k in range(num_tiles):
        offs_k = k * BLOCK_SIZE_K
        if transpose_a:
            a = A.load([offs_k, offs_am])
            a = tl.trans(a)
        else:
            a = A.load([offs_am, offs_k])

        if transpose_b:
            b = B.load([offs_bn, offs_k])
            b = tl.trans(b)
        else:
            b = B.load([offs_k, offs_bn])
        sum = tl.dot(a, b, sum)

    sum = sum.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    C.store([offs_cm, offs_cn], sum)


@triton.autotune(
    configs=_matmul_tma_persistent_get_configs(pre_hook=_host_descriptor_pre_hook),
    key=["M", "N", "K", "transpose_a", "transpose_b", "DTYPE"],
    prune_configs_by={"early_config_prune": _early_config_prune_for_persistent},
)
@triton.jit
def _static_persistent_matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    transpose_a: tl.constexpr,
    transpose_b: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    DTYPE: tl.constexpr,
    occupancy: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Static persistent matmul kernel: C = A @ B with static scheduling."""
    """
    This kernel is adapted from triton tutorial, for oait get the best performance on that.
    For NVTriton, we don't need to introdce `tile_id_c`, `flatten`, `warp_specialize`, `EPILOGUE_SUBTILE`.
    """

    dtype = C.type.block_type.element_ty
    # Get program ID
    start_pid = tl.program_id(axis=0)

    # Calculate total number of tiles
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_programs = tl.num_programs(0)

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue in oait. In NVTiton, we don't need to introdce this,
    # as well as `flatten` and `warp_specialize`. We don't need to recalculate the pid_m and pid_n using tile_id_c
    tile_id_c = start_pid - num_programs
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Static persistent scheduling loop
    for tile_id in tl.range(start_pid, num_tiles, num_programs, flatten=True):
        # Calculate tile coordinates using GROUP_SIZE_M grouping
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, num_programs)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # K-dimension loop
        for k_tile in range(k_tiles):
            offs_k = k_tile * BLOCK_SIZE_K
            # Load A tile
            if transpose_a:
                # A is transposed: load from (K, M) layout
                a = A.load([offs_k, offs_am])
                a = tl.trans(a)  # Convert to (BLOCK_SIZE_M, BLOCK_SIZE_K)
            else:
                # A is normal: load from (M, K) layout
                a = A.load([offs_am, offs_k])

            # Load B tile
            if transpose_b:
                # B is transposed: load from (N, K) layout
                b = B.load([offs_bn, offs_k])
                b = tl.trans(b)  # Convert to (BLOCK_SIZE_K, BLOCK_SIZE_N)
            else:
                # B is normal: load from (K, N) layout
                b = B.load([offs_k, offs_bn])

            # Matrix multiplication and accumulation
            accumulator = tl.dot(a, b, accumulator)
        tile_id_c += num_programs
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, num_programs)
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
        # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
        # memory to increase our stage count.
        # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            C.store([offs_am_c, offs_bn_c], c0)
            c1 = acc1.to(dtype)
            C.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        else:
            # Convert to output dtype and store
            result = accumulator.to(dtype)
            C.store([offs_am_c, offs_bn_c], result)


# Host Launchers & Public API


def matmul_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    kernel_configs: dict = None,
):
    if transpose_a:
        K_A, M = a.shape
    else:
        M, K_A = a.shape

    if transpose_b:
        N, K_B = b.shape
    else:
        K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    K = K_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"
    # allocates output
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

    _matmul_kernel_pure_ptr[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION="",
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    return c


class MatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, transpose_a=False, transpose_b=False):
        c = matmul_fn(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
        ctx.save_for_backward(a, b)
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b
        return c

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        transpose_a = ctx.transpose_a
        transpose_b = ctx.transpose_b

        # Compute gradients with appropriate transposes
        # da = dy @ b^T (or handle transposes based on forward transposes)
        # db = a^T @ dy (or handle transposes based on forward transposes)
        if transpose_a:
            # Forward was: c = a^T @ b
            da = matmul_fn(b, dy, transpose_a=transpose_b, transpose_b=True)
        else:
            # Forward was: c = a @ b
            da = matmul_fn(dy, b, transpose_b=not transpose_b)

        if transpose_b:
            # Forward was: c = a @ b^T
            db = matmul_fn(dy, a, transpose_a=True, transpose_b=transpose_a)
        else:
            # Forward was: c = a @ b
            db = matmul_fn(a, dy, transpose_a=not transpose_a)

        return da, db, None, None


@register_impl("matmul", backend="triton")
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a=False,
    trans_b=False,
    static_persistent=None,
    use_tma=False,
    **kwargs,
):
    if use_tma:
        if static_persistent is None:
            static_persistent = True

        # Get matrix dimensions
        if trans_a:
            K, M = a.shape
        else:
            M, K = a.shape
        if trans_b:
            N, KB = b.shape
        else:
            KB, N = b.shape
        assert K == KB, f"Incompatible matrices: K dimension of A is {K}, K dimension of B is {KB}"

        # Create output tensor
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        # Add host descriptor support
        if _supports_host_descriptor():
            dummy_block = [1, 1]
            desc_a = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
            desc_b = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
            desc_c = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        else:
            desc_a = a
            desc_b = b
            desc_c = c
            raise NotImplementedError(
                "Only support host descriptor for now, need to modify kernel code to support non-host descriptor")

        # Grid calculation
        if static_persistent:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            grid = lambda META: (min(
                NUM_SMS // META["num_ctas"],
                triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ) * META["occupancy"], )
            kernel = _static_persistent_matmul_kernel
        else:
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
            kernel = _matmul_kernel

        logger.debug(
            f"[triton] calling matmul_kernel: use_tma: {use_tma}, static_persistent: {static_persistent}, trans_a: {trans_a}, trans_b: {trans_b}"
        )
        kernel[grid](
            desc_a,
            desc_b,
            desc_c,
            M,
            N,
            K,
            transpose_a=trans_a,
            transpose_b=trans_b,
            DTYPE=a.dtype,
        )
        return c
    else:
        return MatmulFunction.apply(a, b, trans_a, trans_b)


# Backend Registration & Perf Markers

mark_perf_ready("matmul", "nvt")
