# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

import torch
import torch_npu
import pytest
import test_common

# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


def torch_index_select(x0, dim, indices):
    """PyTorch reference implementation using torch.index_select."""
    return torch.index_select(x0, dim, indices)


# ============================================================================
# Triton Kernel Implementations
# ============================================================================


@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim, g_stride: tl.constexpr, indice_length: tl.constexpr,
                               g_block: tl.constexpr, g_block_sub: tl.constexpr, other_block: tl.constexpr):
    """
    Manual implementation using extension.get_element and extension.insert_slice.

    This is a baseline implementation without using the index_select_simd intrinsic.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Manual gather: iterate over each index
            for i in range(0, g_block_sub):
                gather_offset = extension.get_element(indices, (i, )) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = extension.insert_slice(tmp_buf, val[None, :], offsets=(i, 0), sizes=(1, other_block),
                                                 strides=(1, 1))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


@triton.jit
def index_select_extension_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr, other_numel: tl.constexpr,
                                  g_stride: tl.constexpr, indice_length: tl.constexpr, g_block: tl.constexpr,
                                  g_block_sub: tl.constexpr, other_block: tl.constexpr):
    """
    Implementation using extension.index_select_simd intrinsic.

    This uses the hardware-accelerated SIMD index_select operation.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Use extension index_select_simd
            tmp_buf = extension.index_select_simd(src=in_ptr, dim=dim, index=indices, src_shape=(other_numel, g_stride),
                                                  src_offset=(-1, 0), read_shape=(-1, other_block))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


@triton.jit
def index_select_auto_kernel(in_ptr, indices_ptr, out_ptr, dim: tl.constexpr, other_numel: tl.constexpr,
                             g_stride: tl.constexpr, indice_length: tl.constexpr, g_block: tl.constexpr,
                             g_block_sub: tl.constexpr, other_block: tl.constexpr):
    """
    Auto-lowering implementation using standard tl.load with computed offsets.

    This lets the compiler automatically handle the index_select pattern.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Auto-lowering: compute offsets and use standard load
            src_offsets = indices[:, None] * g_stride + other_idx[None, :]
            tmp_buf = tl.load(in_ptr + src_offsets, g_mask[:, None] & other_mask[None, :])

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :], tmp_buf,
                     g_mask[:, None] & other_mask[None, :])


# ============================================================================
# Host Functions
# ============================================================================


def triton_index_select(x0, dim, indices, impl='extension', num_vec_core=48):
    """
    Triton implementation of index_select.

    Args:
        x0: Source tensor
        dim: Dimension to select from
        indices: Indices to select
        impl: Implementation to use ('manual', 'extension', or 'auto')
        num_vec_core: Number of vector cores to use

    Returns:
        Output tensor with selected indices
    """
    sz = list(x0.shape)
    sz[dim] = len(indices)
    out = torch.empty(tuple(sz), dtype=x0.dtype).npu()

    g_stride = x0.stride(dim)
    indice_length = indices.numel()
    g_block = (indice_length - 1) // num_vec_core + 1

    # Calculate UB space allocation
    enable_multi_buffer = True
    ub_size = 125 * 1024 // (2 if enable_multi_buffer else 1)
    other_block = g_stride
    g_block_sub = ub_size // (
        # max memory consumption: arith.select + other (mask handling in auto)
        x0.element_size() * g_stride * 3 + indices.element_size())
    if g_block_sub < 1:
        other_block = (ub_size - indices.element_size()) // x0.element_size()
        g_block_sub = 1

    # Select kernel based on implementation
    if impl == 'manual':
        kernel = index_select_manual_kernel
        kernel[num_vec_core, 1, 1](x0, indices, out, dim, g_stride=g_stride, indice_length=indice_length,
                                   g_block=g_block, g_block_sub=g_block_sub, other_block=other_block, multibuffer=False)
    elif impl == 'extension':
        kernel = index_select_extension_kernel
        kernel[num_vec_core, 1,
               1](x0, indices, out, dim, other_numel=sz[0], g_stride=g_stride, indice_length=indice_length,
                  g_block=g_block, g_block_sub=g_block_sub, other_block=other_block)
    elif impl == 'auto':
        kernel = index_select_auto_kernel
        kernel[num_vec_core, 1,
               1](x0, indices, out, dim, other_numel=sz[0], g_stride=g_stride, indice_length=indice_length,
                  g_block=g_block, g_block_sub=g_block_sub, other_block=other_block)
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    return out


# ============================================================================
# Test Parameters
# ============================================================================

# Test cases with various shapes and sizes
# Format: (src_shape, dim, indice_shape, dtype)
INDEX_SELECT_TEST_CASES = [
    # Small scale tests (< 100us)
    ((10, 16), 0, (1024, ), "float32"),  # 27us fp32
    ((20669, 8), 0, (2048, ), "float32"),  # 31us fp32
    ((1034, 16), 0, (48000, ), "float32"),  # 38us fp32
    ((270098, 32), 0, (512, ), "float32"),  # 40us fp32
    ((1000000, 8), 0, (21329, ), "float32"),  # 44us fp32

    # Medium scale tests (100us - 2000us)
    ((22717, 8), 0, (278400, ), "float32"),  # 76us fp32
    ((2000000, 32), 0, (270244, ), "float32"),  # 155us fp32
    ((500000, 37), 0, (324344, ), "float32"),  # 266us +- 10us fp32
    ((270610, 32), 0, (2928000, ), "float32"),  # 553us fp32
    ((3200, 16), 0, (1971940, ), "float32"),  # 1104us fp32
    ((3200, 1, 37), 0, (1022226, ), "float32"),  # 1965us +- 65us fp32

    # Large scale tests (> 2000us)
    ((500000, 240), 0, (375144, ), "float32"),  # 1984us +- 5us fp32 48core
    ((323357, 37), 0, (1022226, ), "float32"),  # 3500us +- 50us fp32
    ((480000, 16), 0, (3943880, ), "float32"),  # 3636 +- 42us fp32
    ((480000, 32), 0, (3943880, ), "float32"),  # 3678 +- 5us fp32
    ((480000, 64), 0, (3943880, ), "float32"),  # 5392 +- 20us fp32
    ((378103, 240), 0, (1992337, ), "float32"),  # 10000us fp32
    ((374035, 240), 0, (1971940, ), "float32"),  # 10100us fp32
]

# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.parametrize("src_shape, dim, indice_shape, dtype", INDEX_SELECT_TEST_CASES)
def test_index_select_manual(src_shape, dim, indice_shape, dtype):
    """Test manual implementation using extension.get_element and extension.insert_slice."""
    x0 = test_common.generate_tensor(shape=src_shape, dtype=dtype).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()

    torch_ref = torch_index_select(x0, dim, indices)
    triton_cal = triton_index_select(x0, dim, indices, impl='manual', num_vec_core=40)

    assert torch.equal(torch_ref, triton_cal), \
        f"Manual implementation failed for shape={src_shape}, dim={dim}, indices={indice_shape}"


@pytest.mark.parametrize("src_shape, dim, indice_shape, dtype", INDEX_SELECT_TEST_CASES)
def test_index_select_extension(src_shape, dim, indice_shape, dtype):
    """Test extension.index_select_simd implementation."""
    x0 = test_common.generate_tensor(shape=src_shape, dtype=dtype).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()

    torch_ref = torch_index_select(x0, dim, indices)
    triton_cal = triton_index_select(x0, dim, indices, impl='extension', num_vec_core=48)

    test_common.validate_cmp(dtype, triton_cal, torch_ref)


@pytest.mark.parametrize("src_shape, dim, indice_shape, dtype", INDEX_SELECT_TEST_CASES)
def test_index_select_auto(src_shape, dim, indice_shape, dtype):
    """Test auto-lowering implementation using standard tl.load."""
    x0 = test_common.generate_tensor(shape=src_shape, dtype=dtype).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()

    torch_ref = torch_index_select(x0, dim, indices)
    triton_cal = triton_index_select(x0, dim, indices, impl='auto', num_vec_core=48)

    test_common.validate_cmp(dtype, triton_cal, torch_ref)


# Quick smoke test
if __name__ == "__main__":
    test_index_select_extension((500000, 37), 0, (324344, ), "float32")
    print("extension implementation passed")

    test_index_select_auto((500000, 37), 0, (324344, ), "float32")
    print("auto-lowering implementation passed")
