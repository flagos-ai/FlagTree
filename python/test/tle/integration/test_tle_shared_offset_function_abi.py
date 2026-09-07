# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit(noinline=True)
def _write_view(view, source):
    lane = tl.arange(0, 32)
    pointer = tle.gpu.local_ptr(view, (lane,), [32])
    tl.store(pointer, tl.load(source + lane))


@triton.jit(noinline=True)
def _roundtrip_view(view, source, output):
    _write_view(view, source)
    tl.debug_barrier()
    lane = tl.arange(0, 32)
    pointer = tle.gpu.local_ptr(view, (lane,), [32])
    tl.store(output + lane, tl.load(pointer))


@triton.jit
def _aliased_views(source, output, offset: tl.constexpr, repeats):
    arena = tle.gpu.alloc([256], dtype=tl.float32, scope=tle.gpu.smem,
                          nv_mma_shared_layout=False)
    first = arena.subslice([offset], [32])
    second = arena.subslice([offset + 64], [32])
    for _ in range(repeats):
        _roundtrip_view(first, source, output)
        _roundtrip_view(second, source + 32, output + 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.alloc", "gpu.local_ptr")
@pytest.mark.parametrize("offset", [0, 32, 64])
def test_shared_subview_arguments_cross_nested_calls_as_relative_offsets(offset):
    source = torch.randn(64, dtype=torch.float32, device="cuda")
    output = torch.full_like(source, float("nan"))
    compiled = _aliased_views[(1,)](source, output, offset, 2, num_warps=4)
    torch.testing.assert_close(output, source, rtol=0, atol=0)
    assert compiled.metadata.ptxas_spill_store_bytes == 0
    assert compiled.metadata.ptxas_spill_load_bytes == 0
    llvm = compiled.asm["llir"]
    for name in ("_write_view", "_roundtrip_view"):
        definitions = re.findall(r"^define internal .*" + name + r".*", llvm, re.M)
        assert len(definitions) == 1
        assert "ptr addrspace(3)" not in definitions[0]
        assert "i32" in definitions[0]
