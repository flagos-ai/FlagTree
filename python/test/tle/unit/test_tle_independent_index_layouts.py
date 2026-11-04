# Copyright 2026 FlagOS Contributors

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

import triton._C.libtriton as libtriton


_LAYOUT_A = tl.constexpr(tle.gpu.BlockEncoding([1], [32], [8], [0]))
_LAYOUT_B = tl.constexpr(tle.gpu.BlockEncoding([2], [32], [8], [0]))
_LAYOUT_2D_A = tl.constexpr(
    tle.gpu.BlockEncoding([1, 4], [4, 8], [8, 1], [1, 0]))
_LAYOUT_2D_B = tl.constexpr(
    tle.gpu.BlockEncoding([1, 2], [2, 16], [8, 1], [1, 0]))

@pytest.mark.parametrize("owner, name", [
    (tle.gpu, "rematerialize_index"),
    (tle.gpu.core, "rematerialize_index"),
    (libtriton.ir.builder, "create_tle_gpu_rematerialize_index"),
])
def test_explicit_layouts_do_not_require_manual_rematerialization_api(owner, name):
    assert not hasattr(owner, name)


@triton.jit
def _independent_index_layout_roots(output_a, output_b):
    common = tl.arange(0, 128)
    index_a = tle.gpu.set_layout(common, _LAYOUT_A)
    index_b = tle.gpu.set_layout(common, _LAYOUT_B)
    tl.store(output_a + index_a, index_a)
    tl.store(output_b + index_b, index_b + 1)


@triton.jit
def _independent_layout_roots_with_cse_constants(output_a, output_b):
    common = tl.arange(0, 32)
    index_a = tle.gpu.set_layout(common[None, :], _LAYOUT_2D_A)
    index_b = tle.gpu.set_layout(common[None, :], _LAYOUT_2D_B)

    # Both additions initially share the same CSE'd dense tensor constant.
    # Encoding propagation must rematerialize that constant instead of joining
    # the two explicit layout domains through it.
    offset_a = index_a + 32
    offset_b = index_b + 32
    tl.store(output_a + offset_a, offset_a)
    tl.store(output_b + offset_b, offset_b + 1)


@triton.jit
def _independent_layout_roots_with_shared_scalar_source(output):
    common = tl.arange(0, 32)
    index_a = tle.gpu.set_layout(common[None, :], _LAYOUT_2D_A)
    index_b = tle.gpu.set_layout(common[None, :], _LAYOUT_2D_B)

    # The frontend CSEs both tensor splats of the same scalar pointer. A scalar
    # splat is a layout-polymorphic source and must be rebuilt for each domain.
    tl.store(output + index_a, index_a)
    tl.store(output + index_b + 32, index_b + 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
def test_shared_indices_accept_independent_explicit_layouts():
    output_a = torch.empty((128,), device="cuda", dtype=torch.int32)
    output_b = torch.empty_like(output_a)
    _independent_index_layout_roots[(1,)](output_a, output_b, num_warps=8)
    expected = torch.arange(128, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(output_a, expected)
    torch.testing.assert_close(output_b, expected + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
def test_layout_domains_do_not_join_through_cse_constants():
    output_a = torch.full((64,), -1, device="cuda", dtype=torch.int32)
    output_b = torch.full_like(output_a, -1)
    _independent_layout_roots_with_cse_constants[(1,)](
        output_a, output_b, num_warps=8)
    expected = torch.arange(32, 64, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(output_a[32:], expected)
    torch.testing.assert_close(output_b[32:], expected + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.require_tle("gpu.set_layout")
def test_layout_domains_do_not_join_through_cse_scalar_splats():
    output = torch.full((64,), -1, device="cuda", dtype=torch.int32)
    _independent_layout_roots_with_shared_scalar_source[(1,)](
        output, num_warps=8)
    torch.testing.assert_close(
        output, torch.arange(64, device="cuda", dtype=torch.int32))
