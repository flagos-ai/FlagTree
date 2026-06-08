# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

__all__ = [
    "fixpipe",
    "create_address_space",
]

import enum
from typing import (TypeVar, List, Union)

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.core as tl
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl

T = TypeVar('T')


def create_address_space(address_space: ascend_ir.AddressSpace,
                         builder: ascend_ir.ascendnpu_ir_builder) -> ir.attribute:
    return builder.get_target_attribute(address_space)


class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX


def create_sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _semantic=None):
    if isinstance(event_id, int):
        _semantic.builder.sync_block_set(sender, receiver,
                                         _semantic.to_tensor(tl.constexpr(event_id)).handle, sender_pipe.value,
                                         receiver_pipe.value)
    elif isinstance(event_id, tl.constexpr):
        _semantic.builder.sync_block_set(sender, receiver,
                                         _semantic.to_tensor(event_id).handle, sender_pipe.value, receiver_pipe.value)
    else:
        _semantic.builder.sync_block_set(sender, receiver, event_id.handle, sender_pipe.value, receiver_pipe.value)


def create_sync_block_wait(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _semantic=None):
    if isinstance(event_id, int):
        _semantic.builder.sync_block_wait(sender, receiver,
                                          _semantic.to_tensor(tl.constexpr(event_id)).handle, sender_pipe.value,
                                          receiver_pipe.value)
    elif isinstance(event_id, tl.constexpr):
        _semantic.builder.sync_block_wait(sender, receiver,
                                          _semantic.to_tensor(event_id).handle, sender_pipe.value, receiver_pipe.value)
    else:
        _semantic.builder.sync_block_wait(sender, receiver, event_id.handle, sender_pipe.value, receiver_pipe.value)


def sub_vec_id(_semantic=None) -> tl.tensor:
    return tl.tensor(_semantic.builder._ascend_builder.create_get_sub_vec_id(), tl.int64)


def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic=None):
    if not _semantic.builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space != al.ascend_address_space.L1:
            raise TypeError("dst's AddressSpace must be L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def copy(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic=None):
    if not _semantic.builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if isinstance(src, tl.tensor) or isinstance(dst, tl.tensor):
        raise TypeError("tensor not support yet")
    if src.shape != dst.shape:
        raise TypeError("src and dst must have same shape")
    if src.dtype != dst.dtype:
        raise TypeError("src and dst need to have the same type")
    if isinstance(src, bl.buffer) and isinstance(dst, bl.buffer):
        if src.space != al.ascend_address_space.UB:
            raise TypeError("src's AddressSpace must be UB")
        if dst.space not in (al.ascend_address_space.L1, al.ascend_address_space.UB):
            raise TypeError("dst's AddressSpace must be UB or L1")
        _semantic.builder.create_copy_buffer(src.handle, dst.handle)
    else:
        raise TypeError("src and dst must be tl.tensor or bl.buffer")


def fixpipe(
    src: tl.tensor,
    dst,
    dma_mode,
    dual_dst_mode,
    pre_quant_mode,
    pre_relu_mode,
    _semantic=None,
):
    if dst is None:
        result = _semantic.builder._ascend_builder.create_fixpipe(
            src.handle,
            dst,
            dma_mode.value,
            dual_dst_mode.value,
            pre_quant_mode.value,
            pre_relu_mode.value,
        )
        if dual_dst_mode == al.FixpipeDualDstMode.ROW_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 1 and new_shape[0] > 0:
                new_shape[0] = new_shape[0] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        elif dual_dst_mode == al.FixpipeDualDstMode.COLUMN_SPLIT:
            new_shape = list(src.type.shape)
            if len(new_shape) >= 2 and new_shape[1] > 0:
                new_shape[1] = new_shape[1] // 2
            new_type = tl.block_type(src.type.element_ty, new_shape)
            return tl.tensor(result, new_type)
        else:
            return tl.tensor(result, src.type)
    else:
        _semantic.builder._ascend_builder.create_fixpipe(
            src.handle,
            dst.handle,
            dma_mode.value,
            dual_dst_mode.value,
            pre_quant_mode.value,
            pre_relu_mode.value,
        )


def debug_barrier(sync_mode: str, _semantic=None) -> None:
    target = tl.tensor(_semantic.builder.get_int64(0), tl.int64)
    attr = _semantic.builder.get_string_attr(sync_mode)
    _semantic.builder.create_debug_barrier(target.handle, "SYNC_IN_VF", attr)


def conv1d(input_tensor: tl.tensor, weight_tensor: tl.tensor, bias: Union[tl.tensor, None], stride: int,
           padding_size: int, dilation: int, groups: int, output_shape, _semantic=None) -> tl.tensor:
    bias_handle = None if bias is None else bias.handle

    element_ty = input_tensor.type.element_ty
    output_ty = tl.block_type(element_ty, output_shape)
    out = _semantic.builder.create_conv1d(input_tensor.handle, weight_tensor.handle, bias_handle, stride, padding_size,
                                          dilation, groups, output_ty.to_ir(_semantic.builder))
    return tl.tensor(out, output_ty)
