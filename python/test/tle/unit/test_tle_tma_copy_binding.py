# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Keep existing builder call forms when adding an optional copy policy."""

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.language.core import builtin, _unwrap_if_constexpr


@builtin
def _copy_with_binding(descriptor, destination, logical_shape, policy, _semantic=None):
    offsets = [_semantic.to_tensor(0).handle for _ in destination.shape]
    arguments = [descriptor.handle, destination.handle, offsets, None, -1]
    if _unwrap_if_constexpr(logical_shape):
        arguments.append(list(destination.shape))
    if _unwrap_if_constexpr(policy):
        _semantic.builder.create_tma_copy(*arguments, eviction_policy=_semantic._str_to_eviction_policy(policy))
    else:
        _semantic.builder.create_tma_copy(*arguments)


@triton.jit
def _binding_kernel(logical_shape: tl.constexpr, policy: tl.constexpr):
    # Parser-only kernel: the address is never accessed on a device.
    pointer = tl.full((), 0, tl.uint64).to(tl.pointer_type(tl.uint8))
    shared = tle.gpu.alloc([8, 8], tl.float16, scope=tle.gpu.smem)
    descriptor = tle.gpu.reinterpret_tensor_map(pointer, shared)
    _copy_with_binding(descriptor, shared, logical_shape, policy)


@pytest.mark.skip_if_tle_unsupported("gpu.reinterpret_tensor_map")
@pytest.mark.parametrize("logical_shape", [False, True])
@pytest.mark.parametrize("policy", ["", "evict_first", "evict_last"])
def test_tma_copy_binding_preserves_call_forms(logical_shape, policy):
    module = run_parser(
        _binding_kernel,
        kwargs={"logical_shape": logical_shape, "policy": policy, "num_warps": 4},
        target=GPUTarget("cuda", 90, 32),
    )
    source = module.str_nodebug()
    assert source.count("ttg.tma_copy") == 1
    if policy:
        assert policy in source
    if logical_shape:
        assert "tle.logical_copy_shape = array<i64: 8, 8>" in source
