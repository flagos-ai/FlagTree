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
    "PIPE", "SYNC_HINT", "EVENT_ID", "IteratorType", "sub_vec_id", "sub_vec_num", "sync_block_all", "sync_block_set",
    "sync_block_wait", "SYNC_IN_VF", "conv1d", "dot", "conv2d"
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
SYNC_HINT = semantic.SYNC_HINT
EVENT_ID = semantic.EVENT_ID

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
    """Ascend hardware address space constants for buffer allocation.

    Provides named address space specifiers that map to Ascend NPU memory regions:

    - ``UB`` — Unified Buffer (on-chip shared memory)
    - ``L1`` — L1 cache buffer
    - ``L0A`` — L0 buffer A (Cube unit input)
    - ``L0B`` — L0 buffer B (Cube unit input)
    - ``L0C`` — L0 buffer C (Cube unit output)

    Usage with :func:`bl.alloc() <triton.extension.buffer.language.alloc>`::

        buf = bl.alloc(dtype, shape, al.ascend_address_space.UB)
    """

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
    """Sets a cross-core synchronization flag for producer-consumer sync between Cube and Vector cores.

    Pairs with :func:`sync_block_wait` to coordinate execution between different core types.
    Each call increments a per-event counter that the corresponding ``sync_block_wait`` will
    decrement (semaphore-like behavior).

    Must be used within an :func:`scope` context matching the sender's core type.

    :param sender: Sending core type. Must be ``"cube"`` or ``"vector"`` (must differ from ``receiver``).
    :type sender: str
    :param receiver: Receiving core type. Must be ``"cube"`` or ``"vector"``.
    :type receiver: str
    :param event_id: Sync flag identifier in range [0, 15]. Each ID maps to an independent counter.
    :type event_id: int
    :param sender_pipe: Sender-side pipeline type (e.g., ``PIPE_MTE1``, ``PIPE_V``).
        Defaults to ``PIPE_FIX`` if sender is cube, ``PIPE_MTE3`` if sender is vector.
    :type sender_pipe: PIPE
    :param receiver_pipe: Receiver-side pipeline type. Defaults to ``PIPE_MTE2``.
    :type receiver_pipe: PIPE
    """
    return create_sync_block(sender, receiver, event_id, True, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None, _semantic=None):
    """Waits on a cross-core synchronization flag set by :func:`sync_block_set`.

    Blocks execution until the corresponding event counter is positive (signaling the
    producer has completed), then decrements it by 1. Pairs with ``sync_block_set``
    for producer-consumer synchronization between Cube and Vector cores.

    Must be used within an :func:`scope` context matching the receiver's core type.

    :param sender: Sending core type. Must be ``"cube"`` or ``"vector"`` (must differ from ``receiver``).
    :type sender: str
    :param receiver: Receiving core type. Must be ``"cube"`` or ``"vector"``.
    :type receiver: str
    :param event_id: Sync flag identifier in range [0, 15]. Must match the ID used by the
        corresponding ``sync_block_set``.
    :type event_id: int
    :param sender_pipe: Sender-side pipeline type. Defaults to ``PIPE_FIX`` if sender is cube,
        ``PIPE_MTE3`` if sender is vector.
    :type sender_pipe: PIPE
    :param receiver_pipe: Receiver-side pipeline type. Defaults to ``PIPE_MTE2``.
    :type receiver_pipe: PIPE
    """
    return create_sync_block(sender, receiver, event_id, False, sender_pipe, receiver_pipe, _semantic)


@builtin
def sync_block_all(mode, event_id, _semantic=None):
    """Performs global synchronization across all cores of a specified type.

    Inserts a sync barrier to resolve RAW, WAR, and WAW data hazards on shared global
    memory across Cube cores, Vector cores, or both. Also supports sub-vector-level
    synchronization within Vector cores.

    :param mode: Synchronization scope. One of:
        ``"all_cube"`` — sync all Cube cores;
        ``"all_vector"`` — sync all Vector cores;
        ``"all"`` — sync all Cube and Vector cores;
        ``"all_sub_vector"`` — sync between Vector sub-blocks.
    :type mode: str
    :param event_id: Event marker ID in range [0, 15].
    :type event_id: int
    """
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
    """Synchronization barrier modes for fine-grained vector/scalar instruction ordering.

    Each value specifies which instruction types are blocked until previous
    instructions complete. The name follows the pattern ``{blocked}_{awaited}``:
    e.g., ``VST_VLD`` blocks vector stores (VST) until vector loads (VLD) complete.

    Intended for use within an :func:`scope` context.
    """

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
    """Inserts a synchronization barrier between vector/scalar load/store instructions.

    Provides fine-grained control over which instruction types are blocked until
    prior instructions complete. Intended for use within an :func:`scope` context.

    :param sync_mode: Barrier type specifying which instruction classes to synchronize.
    :type sync_mode: SYNC_IN_VF
    """
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


def _dot_to_nd_shape(shape, fractal, is_lhs):
    """A fractal operand's ND shape; lhs -> [M,K], rhs -> [K,N]. 2D is already ND.

    Both operands use the zN fractal (non-transpose): A [M,K] <-> [K1,M1,16,b]
    and B [K,N] <-> [N1,K1,16,b]. In each the two logical dims are
    [d1*d2, d0*d3] (block rows on d2, block cols on d3)."""
    if not (fractal and len(shape) == 4):
        return [shape[0], shape[1]]
    return [shape[1] * shape[2], shape[0] * shape[3]]


def _check_dot_fractal_block(name, shape, is_lhs, dtype):
    """Validate an nZ operand's trailing block [.., block_row, block_col].

    block_col must equal 32 / elem_bytes. block_row must be 16, except for int8
    where the cube fractal is per-operand: a non-transpose left operand is 16
    and a non-transpose right operand is 32.
    """
    elem_bytes = dtype.primitive_bitwidth // 8
    expected_col = 32 // elem_bytes
    block_row = _unwrap_if_constexpr(shape[-2])
    block_col = _unwrap_if_constexpr(shape[-1])

    if block_col != expected_col:
        raise ValueError(f"dot nZ operand `{name}`: block col dim is {block_col}, expected "
                         f"{expected_col} (= 32 / {elem_bytes} bytes per {dtype} element)")

    if elem_bytes != 1:
        if block_row != 16:
            raise ValueError(f"dot nZ operand `{name}`: block row dim is {block_row}, expected 16")
        return

    if block_row not in (16, 32):
        raise ValueError(f"dot int8 nZ operand `{name}`: block row dim is {block_row}, expected 16 or 32")
    if not is_lhs and block_row == 16:
        raise ValueError(f"dot int8 right operand `{name}`: block row dim is 16, but a non-transpose "
                         f"`{name}` needs 32 (block [32,32], i.e. zN[N/32, K/32, 32, 32]); a [16,32] "
                         f"block forces an unsupported layout conversion in the backend")


def _dot_operand_is_fractal(name, fmt):
    """Resolve operand ``name``'s layout from its ``format_*`` string:

      ``"fractal"``             -> fractal (zN)
      ``"nd"`` / ``""`` / unset -> non-fractal (ND)
      anything else             -> ValueError
    """
    fmt = _unwrap_if_constexpr(fmt)
    if fmt is None or fmt == "" or fmt == "nd":
        return False
    if fmt == "fractal":
        return True
    raise ValueError(f"dot `{name}`: format must be 'fractal', 'nd', or '' "
                     f"(empty/unset = non-fractal ND); got {fmt!r}")


@builtin
def dot(a: tl.tensor, b: tl.tensor, format_a="", format_b="", format_c="", _semantic=None) -> tl.tensor:
    """
    Matrix multiply ``D = A * B`` with per-operand layout format.

    ``format_a`` / ``format_b`` / ``format_c`` select each operand's layout:
      - ``"fractal"``: zN fractal (4D)
      - ``"nd"`` / ``""`` / unset: ND (non-fractal, 2D)
      - any other value raises ``ValueError``.

    zN convention (non-transpose, block [16, b], b = 32 / elem_bytes):
        A [M,K] <-> [K1, M1, 16, b]      (ND = [d1*d2, d0*d3])
        B [K,N] <-> [N1, K1, 16, b]      (ND = [d1*d2, d0*d3])
        C [M,N] <-> [N1, M1, 16, 16]     (L0C accumulator, block 16x16)

    :param a: left matrix A (2D ND, or 4D fractal when ``format_a="fractal"``).
    :param b: right matrix B (2D ND, or 4D fractal when ``format_b="fractal"``).
    :param format_a: layout of A: "fractal" | "nd" | "" (default ND).
    :param format_b: layout of B: "fractal" | "nd" | "" (default ND).
    :param format_c: layout of D: "fractal" | "nd" | "" (default ND).

    :return: D = A * B (fractal 4D if ``format_c="fractal"`` else 2D ND).
    :rtype: tensor
    """
    fractal_a = _dot_operand_is_fractal("a", format_a)
    fractal_b = _dot_operand_is_fractal("b", format_b)
    fractal_c = _dot_operand_is_fractal("c", format_c)

    for name, operand, is_frac, is_lhs in (("a", a, fractal_a, True), ("b", b, fractal_b, False)):
        if not isinstance(operand, tl.tensor):
            raise TypeError(f"dot operand `{name}` must be a tensor, got {type(operand)}")
        if len(operand.shape) not in (2, 4):
            raise ValueError(f"dot operand `{name}` must be 2D (ND) or 4D (fractal zN), "
                             f"got rank {len(operand.shape)}")
        if is_frac and len(operand.shape) == 4:
            _check_dot_fractal_block(name, operand.shape, is_lhs, operand.dtype)

    # cube mmad requires both inputs share the element dtype (the output takes
    # the accumulator dtype separately, see semantic.dot).
    if a.dtype != b.dtype:
        raise TypeError(f"dot operands must have the same dtype, got {a.dtype} and {b.dtype}")

    a_nd = _dot_to_nd_shape([_unwrap_if_constexpr(s) for s in a.shape], fractal_a, True)
    b_nd = _dot_to_nd_shape([_unwrap_if_constexpr(s) for s in b.shape], fractal_b, False)
    m, n = a_nd[0], b_nd[1]
    # The result carries the cube accumulator dtype (f32/i32, see semantic.dot);
    # fractal_c is the L0C accumulator fractal, whose block is 16x16.
    output_shape = [n // 16, m // 16, 16, 16] if fractal_c else [m, n]

    return semantic.dot(a, b, fractal_a, fractal_b, fractal_c, output_shape, _semantic=_semantic)


@builtin
def conv2d(input: tl.tensor, weight: tl.tensor, bias: tl.tensor = None, stride=None, padding=None, dilation=None,
           groups=None, _semantic=None) -> tl.tensor:
    """
    Applies a 2D convolution over an input signal.

    :param input: Input tensor of shape (N, C_in, H, W) or (C_in, H, W).
    :type input: tensor
    :param weight: Weight tensor of shape (C_out, C_in // groups, kH, kW).
    :type weight: tensor
    :param bias: Bias tensor of shape (C_out) or None. Default: None.
    :type bias: tensor or None
    :param stride: The stride of the convolution kernel. Can be an int or a 2-element tuple.
    :type stride: int or Tuple[int, int]
    :param padding: Padding added to the input. Can be an int (symmetric on all
        sides), a 2-element tuple (pad_h, pad_w) (symmetric per dimension),
        a 4-element tuple (pad_top, pad_bottom, pad_left, pad_right)
        (asymmetric), or a string.
        ``padding='valid'`` is the same as no padding.
        ``padding='same'`` pads the input so the output has the same shape as the input. However, this mode doesn't support any stride values other than 1.
    :type padding: int, Tuple[int, int], Tuple[int, int, int, int], or str
    :param dilation: The spacing between kernel elements. Can be an int or a 2-element tuple.
    :type dilation: int or Tuple[int, int]
    :param groups: Number of blocked connections from input to output channels.
    :type groups: int

    **Example:**

    .. code-block:: python

        @triton.jit
        def conv2d_kernel(input_ptr, weight_ptr, output_ptr, C, H, W, C_out, kH, kW):
            input_tile = tl.load(input_ptr + ...)     # shape (C, H, W)
            weight_tile = tl.load(weight_ptr + ...)   # shape (C_out, C, kH, kW)
            output = al.conv2d(input_tile, weight_tile,
                               stride=(1, 1), padding=(0, 0), dilation=(1, 1))
            tl.store(output_ptr + ..., output)

    :return: The output tensor of shape (N, C_out, H_out, W_out).
    :rtype: tensor
    """

    stride = _unwrap_if_constexpr(stride)
    padding = _unwrap_if_constexpr(padding)
    dilation = _unwrap_if_constexpr(dilation)
    groups = _unwrap_if_constexpr(groups)

    # Set default values
    stride = stride if stride is not None else 1
    padding = padding if padding is not None else 0
    dilation = dilation if dilation is not None else 1
    groups = groups if groups is not None else 1

    if type(bias).__name__ == 'constexpr':
        bias = getattr(bias, 'value', bias)
    if bias is not None:
        assert len(bias.shape) == 1, f"bias must be a 1D tensor (C_out), got {len(bias.shape)}D"
    assert isinstance(groups, int), f"groups must be an integer, got {groups}"

    def _check_and_normalize_2d_param(param, name):
        """Keep ints as ints and tuples as tuples; lists are accepted and
        converted to tuples."""
        if param is None:
            return None
        if isinstance(param, list):
            assert len(param) == 2, f"{name} must be an integer or a 2-element sequence, got {param}"
            return tuple(param)
        if isinstance(param, (tuple, tl.tuple)):
            assert len(param) == 2, f"{name} must be an integer or a 2-element sequence, got {param}"
            return param
        assert isinstance(param, int), f"{name} must be an integer or a 2-element sequence, got {type(param)}"
        return param

    stride = _check_and_normalize_2d_param(stride, 'stride')
    dilation = _check_and_normalize_2d_param(dilation, 'dilation')

    # Per-dimension accessors accepting either the int or tuple form.
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    is_batched = len(input.shape) == 4
    H_in = input.shape[-2]
    W_in = input.shape[-1]
    kH = weight.shape[2]
    kW = weight.shape[3]

    def _check_and_normalize_padding(param):
        """Keep ints as ints and tuples as tuples; lists are accepted and
        converted to tuples.

        Accepts an int (symmetric on all sides), a 2-element sequence
        [pad_h, pad_w] (symmetric per dimension), or a 4-element sequence
        [pad_top, pad_bottom, pad_left, pad_right] (fully asymmetric).
        """
        if param is None:
            return 0
        if isinstance(param, list):
            assert len(param) in [2, 4], \
                f"padding must be an int, a 2-element or 4-element sequence, got {param}"
            return tuple(param)
        if isinstance(param, (tuple, tl.tuple)):
            assert len(param) in [2, 4], \
                f"padding must be an int, a 2-element or 4-element sequence, got {param}"
            return param
        assert isinstance(param, int), \
            f"padding must be an int, a 2-element or 4-element sequence, got {type(param)}"
        return param

    if isinstance(padding, str):
        assert padding in ['same', 'valid'], f"padding string must be 'same' or 'valid', got '{padding}'"
        if padding == 'valid':
            padding_int = 0
        elif padding == 'same':
            if stride_h != 1 or stride_w != 1:
                raise ValueError("padding='same' is only supported when stride=1")
            # Total padding needed per dimension; extra element goes to bottom/right
            pad_h_total = (H_in - 1) * stride_h + dilation_h * (kH - 1) + 1 - H_in
            pad_w_total = (W_in - 1) * stride_w + dilation_w * (kW - 1) + 1 - W_in
            padding_int = (pad_h_total // 2, pad_h_total - pad_h_total // 2, pad_w_total // 2,
                           pad_w_total - pad_w_total // 2)
    else:
        padding_int = _check_and_normalize_padding(padding)

    assert len(input.shape) in [3, 4], f"input must be 3D (C,H,W) or 4D (N,C,H,W), got {len(input.shape)}D"
    assert len(weight.shape) == 4, f"weight must be 4D (C_out, C_in/groups, kH, kW), got {len(weight.shape)}D"

    C_in = input.shape[-3] if is_batched else input.shape[0]
    C_out = weight.shape[0]

    H_in_val = _unwrap_if_constexpr(input.shape[-2])
    W_in_val = _unwrap_if_constexpr(input.shape[-1])
    kH_val = _unwrap_if_constexpr(weight.shape[2])
    kW_val = _unwrap_if_constexpr(weight.shape[3])

    def _compute_output(in_size, pad_before, pad_after, dil, k, strd):
        return -int(-((in_size + pad_before + pad_after - dil * (k - 1) - 1) / strd + 1))

    # Expand padding to [pad_top, pad_bottom, pad_left, pad_right].
    if isinstance(padding_int, int):
        pad_top = pad_bottom = pad_left = pad_right = padding_int
    elif len(padding_int) == 2:
        pad_h, pad_w = padding_int
        pad_top = pad_bottom = pad_h
        pad_left = pad_right = pad_w
    else:
        pad_top, pad_bottom, pad_left, pad_right = padding_int

    H_out_val = _compute_output(H_in_val, pad_top, pad_bottom, dilation_h, kH_val, stride_h)
    W_out_val = _compute_output(W_in_val, pad_left, pad_right, dilation_w, kW_val, stride_w)

    if is_batched:
        output_shape = [input.shape[0], C_out, H_out_val, W_out_val]
    else:
        output_shape = [C_out, H_out_val, W_out_val]

    return semantic.conv2d(input, weight, bias, stride, padding_int, dilation, groups, output_shape, _semantic)
