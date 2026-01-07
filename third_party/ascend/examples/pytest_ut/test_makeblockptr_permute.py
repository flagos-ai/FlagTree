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

import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common

@pytest.mark.parametrize('shape', [(1, 4, 2)])
@pytest.mark.parametrize('permute_order', [(2, 0, 1)])
def test_makeblockptr_order(shape, permute_order):

    @triton.jit
    def triton_kernel(in0_ptr: tl.tensor, # of tl.pointer_type
                out0_ptr: tl.tensor, # of tl.pointer_type
                in0_stride0: int, in0_stride1: int, in0_stride2: int, # strides for in0
                in0_stride_order0: tl.constexpr, in0_stride_order1: tl.constexpr, in0_stride_order2: tl.constexpr, # stride order for in0
                out0_stride0: int, out0_stride1: int, out0_stride2: int, # strides for out0
                out0_stride_order0: tl.constexpr, out0_stride_order1: tl.constexpr, out0_stride_order2: tl.constexpr, # stride order for out0
                s0: int, s1: int, s2: int,
                tile_size0: tl.constexpr, tile_size1: tl.constexpr, tile_size2: tl.constexpr,
                ):
        tile_id0 = tl.program_id(axis=0)
        tile_id1 = tl.program_id(axis=1)
        tile_id2 = tl.program_id(axis=2)
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        offset2 = (tile_id2 * tile_size2).to(tl.int32)
        in0_bptr = tl.make_block_ptr(in0_ptr,
                                    (s0, s1, s2),
                                    (in0_stride0, in0_stride1, in0_stride2),
                                    (offset0, offset1, offset2),
                                    (tile_size0, tile_size1, tile_size2),
                                    order=(in0_stride_order0, in0_stride_order1, in0_stride_order2))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0, in0_stride_order1, in0_stride_order2)).to(in0_ptr.type.element_ty)

        out0 = in0

        out0_bptr = tl.make_block_ptr(out0_ptr, (s0, s1, s2), (out0_stride0, out0_stride1, out0_stride2), (offset0, offset1, offset2), (tile_size0, tile_size1, tile_size2),
                                    order=(out0_stride_order0, out0_stride_order1, out0_stride_order2))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0, out0_stride_order1, out0_stride_order2))

    def triton_func(in0: torch.Tensor, permute_order):
        # in fact, it adjusts the layout metadata instead of doing a real permutation.
        in0_permuted_tmp = in0.permute(permute_order)
        in0_permuted_shape = in0_permuted_tmp.size()
        in0_permuted_strides = in0_permuted_tmp.stride()
        in0_stride_order = [len(permute_order)-1-i for i in permute_order]
        shape = (in0_permuted_shape[0], in0_permuted_shape[1], in0_permuted_shape[2])
        tile_sizes = (shape[0], shape[1], shape[2])
        out0 = torch.empty(shape, dtype=in0.dtype, device=in0.device)
        out0_strides = out0.stride()
        out0_stride_order = [len(permute_order)-1-i for i in range(len(permute_order))]
        grid = (shape[0]//tile_sizes[0], shape[1]//tile_sizes[1], shape[2]//tile_sizes[2])
        triton_kernel[grid](
                in0, out0,
                in0_permuted_strides[0], in0_permuted_strides[1], in0_permuted_strides[2], # stride for in0
                in0_stride_order[0], in0_stride_order[1], in0_stride_order[2], # stride order for in0
                out0_strides[0], out0_strides[1], out0_strides[2], # stride for out0
                out0_stride_order[0], out0_stride_order[1], out0_stride_order[2], # stride orderfor out0
                shape[0], shape[1], shape[2], # task indexing space
                tile_size0=tile_sizes[0],
                tile_size1=tile_sizes[1],
                tile_size2=tile_sizes[2],
            )
        return out0

    x0 = torch.randint(0, 9, shape, dtype=torch.int32).npu()
    torch_ref = torch.permute(x0, permute_order)
    triton_cal = triton_func(x0, permute_order)
    test_common.validate_cmp("int32", triton_cal, torch_ref)
