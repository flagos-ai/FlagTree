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
    "ascend_address_space", "builtin", "CORE", "copy_from_ub_to_l1", "copy", "debug_barrier", "fixpipe",
    "FixpipeDMAMode", "FixpipeDualDstMode", "FixpipePreQuantMode", "FixpipePreReluMode", "int64", "is_builtin", "MODE",
    "PIPE", "IteratorType", "sub_vec_id", "sub_vec_num", "sync_block_all", "sync_block_set", "sync_block_wait",
    "SYNC_IN_VF", "conv1d"
]

import enum
from typing import TypeVar, List, Union
from functools import wraps

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.core as tl

import triton.extension.buffer.language as bl
from triton.language.core import _unwrap_if_constexpr
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
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
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


class IteratorType(enum.Enum):
    Parallel = ascend_ir.IteratorType.Parallel
    Broadcast = ascend_ir.IteratorType.Broadcast
    Transpose = ascend_ir.IteratorType.Transpose
    Reduction = ascend_ir.IteratorType.Reduction
    Interleave = ascend_ir.IteratorType.Interleave
    Deinterleave = ascend_ir.IteratorType.Deinterleave
    Inverse = ascend_ir.IteratorType.Inverse
    Pad = ascend_ir.IteratorType.Pad
    Concat = ascend_ir.IteratorType.Concat
    Gather = ascend_ir.IteratorType.Gather
    Cumulative = ascend_ir.IteratorType.Cumulative
    Opaque = ascend_ir.IteratorType.Opaque


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
def sub_vec_id(_semantic=None) -> tl.tensor:
    """
    Get the Vector Core index on the AI Core.
    """
    return semantic.sub_vec_id(_semantic)


@builtin
def copy_from_ub_to_l1(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic: None) -> None:
    """
    Copies data from the Unified Buffer (UB) to the L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located in L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    from warnings import warn
    warn("copy_from_ub_to_l1 is deprecated, please use copy instead.")
    return semantic.copy_from_ub_to_l1(src, dst, _semantic)


@builtin
def copy(src: Union[tl.tensor, bl.buffer], dst: Union[tl.tensor, bl.buffer], _semantic: None) -> None:
    """
    Copies data from the Unified Buffer (UB) to the Unified Buffer (UB) or L1 Buffer.

    :param src: The source data located in the Unified Buffer.
    :type src: tl.tensor | bl.buffer
    :param dst: The destination buffer located Unified Buffer (UB) or L1 memory.
    :type dst: tl.tensor | bl.buffer
    """
    return semantic.copy(src, dst, _semantic)


def create_sync_block(sender, receiver, event_id, is_set: bool, sender_pipe=None, receiver_pipe=None, _semantic=None):
    sender = _unwrap_if_constexpr(sender)
    receiver = _unwrap_if_constexpr(receiver)
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
        return semantic.create_sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)
    return semantic.create_sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _semantic=None):
    return create_sync_block(sender, receiver, event_id, True, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _semantic=None):
    return create_sync_block(sender, receiver, event_id, False, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_all(mode, event_id, _semantic=None):
    mode = _unwrap_if_constexpr(mode)
    event_id = _unwrap_if_constexpr(event_id)
    assert isinstance(mode, str), f"mode: {mode} is not string"
    assert isinstance(event_id, int) and (event_id >= 0) and (event_id < 16), f"event_id: {event_id} should be 0 ~ 15"
    assert mode in ("all_cube", "all_vector", "all",
                    "all_sub_vector"), f"ERROR: mode = {mode}, only supports all_cube/all_vector/all/all_sub_vector"
    _semantic.builder.sync_block_all(mode, event_id)


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
    dst: Union[None, bl.buffer] = None,
    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,
    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,
    _semantic=None,
) -> Union[None, tl.tensor]:
    """
    Directly store a tensor on L0C to a local buffer via fixpipe.
    Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies.
    Currently support:
        - L0C to UB (for Ascend910_95 sereies)

    :param src: the source tensor, Must be located in the l0C memory region.
    :type src: tl.tensor
    :param dst: The destination buffer in UB. If None, an empty tensor will be created.
                Must be buffer type in UB if provided.
    :type dst: bl.buffer | None
    :param dma_mode: DMA transfer mode, "nz2nd" enables NZ to ND layout transformation
    :type dma_mode: FixpipeDMAMode
    :param dual_dst_mode: Dual destination mode for split operations
    :type dual_dst_mode: FixpipeDualDstMode
    :return: If dst is None, returns the created tensor; otherwise returns None.
    :rtype: tl.tensor | None
    """
    if not _semantic.builder.is_910_95():
        raise RuntimeError("this feature is only supported on Ascend910_95")
    if not isinstance(src, tl.tensor):
        raise TypeError("src is not of tensor type")

    if dst is not None:
        if not isinstance(dst, bl.buffer):
            raise TypeError("dst must be buffer type or None")
        if dst.space != ascend_address_space.UB:
            raise TypeError("dst's AddressSpace must be UB")
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

    return semantic.fixpipe(
        src,
        dst,
        dma_mode,
        dual_dst_mode,
        FixpipePreQuantMode.NO_QUANT,
        FixpipePreReluMode.NO_RELU,
        _semantic,
    )


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
    _semantic=None,
) -> None:
    return semantic.debug_barrier(sync_mode.name, _semantic)


@builtin
def sub_vec_num(_semantic=None) -> tl.constexpr:
    """
    Get the Vector Core Num on one AI Core.
    """
    npuUtils = NPUUtils()
    cube_num = npuUtils.get_aivector_core_num()
    vector_num = npuUtils.get_aicore_num()
    const_val = cube_num // vector_num
    return tl.constexpr(const_val)


@builtin
def conv1d(input: tl.tensor, weight: tl.tensor, bias: tl.tensor = None, stride=None, padding_size=None, dilation=None,
           groups=None, _semantic=None) -> tl.tensor:
    """
    Applies a 1D convolution over an input signal.

    :param input: Input tensor of shape (N, C_in, L_in) or (C_in, L_in). N is a batch size, C denotes a number of channels, L is a length of signal sequence.
    :type input: tensor
    :param weight: Weight tensor of shape (C_out, C_in // groups, kernel_size).
    :type weight: tensor
    :param bias: Bias tensor of shape (C_out) or None. Default: None.
    :type bias: tensor or None
    :param stride: The stride of the convolution kernel. Can be an int or a 1-element tuple.
    :type stride: int or Tuple[int]
    :param padding_size: Padding added to both sides of the input. Can be an int, a 1-element tuple, or a string. Can be a string {'valid', 'same'}, single number or a one-element tuple.
        ``padding_size='valid'`` is the same as no padding.
        ``padding_size='same'`` pads the input so the output has the same shape as the input. However, this mode doesn't support any stride values other than 1.
    :type padding_size: int, Tuple[int], or str
    :param dilation: The spacing between kernel elements. Can be an int or a 1-element tuple.
    :type dilation: int or Tuple[int]
    :param groups: Number of blocked connections from input to output channels.
    :type groups: int

    **Example:**

    .. code-block:: python

        @triton.jit
        def conv_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, N, C, L, K, BLOCK_SIZE: tl.constexpr):
            # Load a tile of input and weight
            input_block = tl.load(input_ptr + ...)
            weight_block = tl.load(weight_ptr + ...)

            # Perform 1D convolution
            # Using default stride=1, padding_size=0, dilation=1, groups=1
            conv_output = al.conv1d(
                input_block,
                weight_block,
                bias=None,
                stride=1,
                padding_size=0,
                dilation=1,
                groups=1,
            )

            # Store the result
            tl.store(output_ptr + ..., conv_output)

    :return: The output tensor of shape (N, C_out, L_out).
    :rtype: tensor
    """

    stride = _unwrap_if_constexpr(stride)
    padding_size = _unwrap_if_constexpr(padding_size)
    dilation = _unwrap_if_constexpr(dilation)
    groups = _unwrap_if_constexpr(groups)

    # Set default value
    stride = stride if stride is not None else 1
    padding_size = padding_size if padding_size is not None else 0
    dilation = dilation if dilation is not None else 1
    groups = groups if groups is not None else 1

    if type(bias).__name__ == 'constexpr':
        bias = getattr(bias, 'value', bias)
    if bias is not None:
        assert len(bias.shape) == 1, f"bias must be a 1D tensor (C_out), got {len(bias.shape)}D"
    assert isinstance(groups, int), f"groups must be an integer, got {groups}"

    def _check_and_normalize_1d_param(param, name):
        if param is None:
            return None
        if isinstance(param, (list, tuple)):
            assert len(param) == 1, f"{name} must be an integer or a 1-element tuple, got {param}"
            return param[0]
        assert isinstance(param, int), f"{name} must be an integer or a 1-element tuple, got {type(param)}"
        return param

    stride = _check_and_normalize_1d_param(stride, 'stride')
    dilation = _check_and_normalize_1d_param(dilation, 'dilation')

    is_batched = len(input.shape) == 3
    L_in = input.shape[-1]
    K = weight.shape[2]

    if isinstance(padding_size, str):
        assert padding_size in ['same', 'valid'], f"padding_size string must be 'same' or 'valid', got '{padding_size}'"
    else:
        padding_size = _check_and_normalize_1d_param(padding_size, 'padding_size')
    if isinstance(padding_size, str):
        if padding_size == 'valid':
            padding_size_int = 0
        elif padding_size == 'same':
            #The case where padding_needed is an odd number needs to be handled.
            if stride != 1:
                raise ValueError("padding_size='same' is only supported when stride=1")
            padding_needed = (L_in - 1) * stride + dilation * (K - 1) + 1 - L_in
            padding_size_int = padding_needed // 2
    else:
        padding_size_int = padding_size if padding_size is not None else 0

    assert len(input.shape) in [2, 3], f"input must be a 2D (C, L) or 3D (N, C, L) tensor, got {len(input.shape)}D"
    assert len(
        weight.shape) == 3, f"weight must be a 3D tensor (C_out, C_in // groups, kernel_size), got {len(weight.shape)}D"

    # Create output type
    C_in = input.shape[-2] if is_batched else input.shape[0]
    C_out = weight.shape[0]

    L_in_val = _unwrap_if_constexpr(input.shape[-1])
    K_val = _unwrap_if_constexpr(weight.shape[2])

    calculation_result = (L_in_val + 2 * padding_size_int - dilation * (K_val - 1) - 1) / stride + 1
    L_out_val = -int(-calculation_result)
    if is_batched:
        output_shape = [input.shape[0], C_out, L_out_val]
    else:
        output_shape = [C_out, L_out_val]

    return semantic.conv1d(input, weight, bias, stride, padding_size_int, dilation, groups, output_shape, _semantic)
