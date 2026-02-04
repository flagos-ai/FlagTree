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
    "ascend_address_space", "builtin", "CORE", "copy_from_ub_to_l1", "debug_barrier", "fixpipe", "FixpipeDMAMode",
    "FixpipeDualDstMode", "FixpipePreQuantMode", "FixpipePreReluMode", "int64", "is_builtin", "MODE", "PIPE",
    "sub_vec_id", "sub_vec_num", "sync_block_all", "sync_block_set", "sync_block_wait", "SYNC_IN_VF"
]

import enum
from typing import TypeVar, List, Union
from functools import wraps

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.core as tl

import triton.extension.buffer.language as bl
from triton.language.core import _constexpr_to_value
from triton.backends.ascend.driver import NPUUtils

from . import semantic as semantic

PIPE = semantic.PIPE

T = TypeVar("T")

TRITON_BUILTIN = "__triton_builtin__"
ASCEND_BUILTIN = "__ascend_builtin__"


def builtin(fn: T) -> T:
    """Mark a function as a buffer language builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_builder" not in kwargs or kwargs["_builder"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_builder` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    # also set triton_builtin to true so that CodeGenerator will recognize this function
    setattr(wrapper, TRITON_BUILTIN, True)
    setattr(wrapper, ASCEND_BUILTIN, True)

    return wrapper


def is_builtin(fn) -> bool:
    """Is this a registered ascend language builtin function?"""
    return getattr(fn, ASCEND_BUILTIN, False)


class int64(int):
    """
    For custom op, python int argument will be converted to int32 by default,
    if a device-side int64 is required, you can pass an al.int64(x) to it.
    """

    def __new__(cls, value):
        obj = int.__new__(cls, value)
        obj.type = tl.int64
        return obj


class CORE(enum.Enum):
    VECTOR = ascend_ir.CoreType.VECTOR
    CUBE = ascend_ir.CoreType.CUBE
    CUBE_OR_VECTOR = ascend_ir.CoreType.CUBE_OR_VECTOR
    CUBE_AND_VECTOR = ascend_ir.CoreType.CUBE_AND_VECTOR


class PIPE(enum.Enum):
    PIPE_S = ascend_ir.PIPE.PIPE_S
    PIPE_V = ascend_ir.PIPE.PIPE_V
    PIPE_M = ascend_ir.PIPE.PIPE_M
    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1
    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2
    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3
    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL
    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX


class MODE(enum.Enum):
    SIMD = ascend_ir.MODE.SIMD
    SIMT = ascend_ir.MODE.SIMT
    MIX = ascend_ir.MODE.MIX


class ascend_address_space_base(bl.address_space):

    def __init__(self, address_space_value: ascend_ir.AddressSpace) -> None:
        super().__init__()
        self.real_address_space = address_space_value

    def to_ir(self, builder: ir.builder) -> ir.attribute:
        return builder.get_target_attribute(self.real_address_space)


class ascend_address_space_group:

    def __init__(self):
        for k, v in {k: v
                     for k, v in ascend_ir.AddressSpace.__dict__.items()
                     if isinstance(v, ascend_ir.AddressSpace)}.items():
            setattr(self, k, ascend_address_space_base(v))


ascend_address_space = ascend_address_space_group()


@builtin
def sub_vec_id(_builder=None) -> tl.tensor:
    """
    Get the Vector Core index on the AI Core.
    """
    return semantic.sub_vec_id(_builder)


@builtin
def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _builder: None) -> None:
    """
    Copies data from the Unified Buffer (UB) to the L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located in L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    return semantic.copy_from_ub_to_l1(src, dst, _builder)


def create_sync_block(sender, receiver, event_id, is_set: bool, sender_pipe=None, receiver_pipe=None, _builder=None):
    sender = _constexpr_to_value(sender)
    receiver = _constexpr_to_value(receiver)
    assert isinstance(sender, str) and (sender == "cube"
                                        or sender == "vector"), f"ERROR: sender = {sender}, only supports cube/vector"
    assert isinstance(receiver, str) and (receiver == "cube" or receiver
                                          == "vector"), f"ERROR: receiver = {receiver}, only supports cube/vector"
    if isinstance(event_id, int):
        assert (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    if sender == receiver:
        raise ValueError(f'Unexpected pair: {sender} -> {receiver}, only supports cube -> vector or vector -> cube')
    if sender_pipe is None and receiver_pipe is None:
        if sender == "cube":
            sender_pipe = PIPE.PIPE_FIX
            receiver_pipe = PIPE.PIPE_MTE2
        if sender == "vector":
            sender_pipe = PIPE.PIPE_MTE3
            receiver_pipe = PIPE.PIPE_MTE2
    if not isinstance(sender_pipe, PIPE) or not isinstance(receiver_pipe, PIPE):
        raise TypeError("sender_pipe and receiver_pipe must be instances of PIPE enum")
    if is_set:
        return semantic.create_sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe, _builder)
    return semantic.create_sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _builder=None):
    return create_sync_block(sender, receiver, event_id, True, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _builder=None):
    return create_sync_block(sender, receiver, event_id, False, sender_pipe, receiver_pipe, _builder)


@builtin
def sync_block_all(mode, event_id, _builder=None):
    mode = _constexpr_to_value(mode)
    event_id = _constexpr_to_value(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode in ("all_cube", "all_vector", "all",
                    "all_sub_vector"), f"ERROR: mode = {mode}, only supports all_cube/all_vector/all/all_sub_vector"
    _builder.sync_block_all(mode, event_id)


class FixpipeDMAMode(enum.Enum):
    NZ2DN = ascend_ir.FixpipeDMAMode.NZ2DN
    NZ2ND = ascend_ir.FixpipeDMAMode.NZ2ND
    NZ2NZ = ascend_ir.FixpipeDMAMode.NZ2NZ


class FixpipeDualDstMode(enum.Enum):
    NO_DUAL = ascend_ir.FixpipeDualDstMode.NO_DUAL
    COLUMN_SPLIT = ascend_ir.FixpipeDualDstMode.COLUMN_SPLIT
    ROW_SPLIT = ascend_ir.FixpipeDualDstMode.ROW_SPLIT


class FixpipePreQuantMode(enum.Enum):
    NO_QUANT = ascend_ir.FixpipePreQuantMode.NO_QUANT
    F322BF16 = ascend_ir.FixpipePreQuantMode.F322BF16
    F322F16 = ascend_ir.FixpipePreQuantMode.F322F16
    S322I8 = ascend_ir.FixpipePreQuantMode.S322I8


class FixpipePreReluMode(enum.Enum):
    LEAKY_RELU = ascend_ir.FixpipePreReluMode.LEAKY_RELU
    NO_RELU = ascend_ir.FixpipePreReluMode.NO_RELU
    NORMAL_RELU = ascend_ir.FixpipePreReluMode.NORMAL_RELU
    P_RELU = ascend_ir.FixpipePreReluMode.P_RELU


@builtin
def fixpipe(
    src: tl.tensor,
    dst: bl.buffer,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _builder=None,
) -> None:
    """
    Directly store a tensor on L0C to a local buffer via fixpipe.
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies.
    Currently support:
        - L0C to UB (for Ascend910_95 sereies)

    :param src: the source tensor, Must be located in the l0C memory region.
    :type src: tl.tensor
    :param dst: The destination buffer, Must be located in the UB memory region.
    :type dst: bl.buffer
    :param dma_mode: DMA transfer mode, "nz2nd" enables NZ to ND layout transformation
    :type dma_mode: str
    """
    if not _builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if not isinstance(src, tl.tensor):
        raise TypeError("src is not of tensor type")
    elif not isinstance(dst, bl.buffer):
        raise TypeError("dst is not of buffer type")
    if dst.space != ascend_address_space.UB:
        raise TypeError("dst must be located in the UB memory region")

    if len(dst.shape) == 2 and (dst.type.element_ty == tl.float32 or dst.type.element_ty == tl.int32):
        N = dst.shape[1]
        if N % 8 != 0:
            raise ValueError("32b Fixpipe last dim must be aligned to 8")
        if (dma_mode != FixpipeDMAMode.NZ2ND) and (N % 16 != 0):
            raise ValueError("32b non-NZ2ND Fixpipe last dim must be aligned to 16")
        if (dual_dst_mode == FixpipeDualDstMode.COLUMN_SPLIT) and (N % 32 != 0):
            raise ValueError("32b Column split dual Fixpipe last dim must be aligned to 32")
        M = dst.shape[0]
        if (dma_mode == FixpipeDMAMode.NZ2DN) and (M % 8 != 0):
            raise ValueError("32b NZ2DN Fixpipe first dim must be aligned to 8")
    dst16bits = (dst.type.element_ty == tl.float16 or dst.type.element_ty == tl.int16
                 or dst.type.element_ty == tl.bfloat16)
    if len(dst.shape) == 2 and dst16bits:
        N = dst.shape[1]
        if N % 16 != 0:
            raise ValueError("16b Fixpipe last dim must be aligned to 16")
        M = dst.shape[0]
        if (dma_mode == FixpipeDMAMode.NZ2DN) and (M % 16 != 0):
            raise ValueError("16b NZ2DN Fixpipe first dim must be aligned to 16")

    return semantic.fixpipe(src, dst, dma_mode, dual_dst_mode, FixpipePreQuantMode.NO_QUANT, FixpipePreReluMode.NO_RELU,
                            _builder)


class SYNC_IN_VF(enum.Enum):
    VV_ALL = enum.auto()
    VST_VLD = enum.auto()
    VLD_VST = enum.auto()
    VST_VST = enum.auto()
    VS_ALL = enum.auto()
    VST_LD = enum.auto()
    VLD_ST = enum.auto()
    VST_ST = enum.auto()
    SV_ALL = enum.auto()
    ST_VLD = enum.auto()
    LD_VST = enum.auto()
    ST_VST = enum.auto()


@builtin
def debug_barrier(
    sync_mode: SYNC_IN_VF,
    _builder=None,
) -> None:
    return semantic.debug_barrier(sync_mode.name, _builder)


@builtin
def sub_vec_num(_builder=None) -> tl.constexpr:
    """
    Get the Vector Core Num on one AI Core.
    """
    npuUtils = NPUUtils()
    cube_num = npuUtils.get_aivector_core_num()
    vector_num = npuUtils.get_aicore_num()
    const_val = cube_num // vector_num
    return tl.constexpr(const_val)
