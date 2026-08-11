# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import triton.language.core as tl

from .. import types as tle


def normalize_copy_shape(shape) -> tuple[int, ...]:
    return tuple(int(tl._unwrap_if_constexpr(dim)) for dim in shape)


def validate_copy_buffer(buffer: tle.buffered_tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(buffer, tle.buffered_tensor):
        raise ValueError(f"buffer must be a tle.gpu.buffered_tensor, but got {type(buffer)}")
    if buffer.type.storage != tle.smem:
        raise ValueError("MUSA TLE copy only supports tle.gpu.smem buffers")
    buffer_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in buffer.type.shape)
    if buffer_shape != shape:
        raise ValueError(f"copy shape {shape} must match buffer shape {buffer_shape}")


def tensor_shape(value: tl.tensor) -> tuple[int, ...]:
    if not value.type.is_block():
        return tuple()
    return tuple(int(tl._unwrap_if_constexpr(dim)) for dim in value.shape)


def tensor_pointer_element_ty(value: tl.tensor):
    scalar_ty = value.dtype
    if not scalar_ty.is_ptr():
        raise ValueError("tle.gpu.copy tensor operands must be pointer tensors")
    return scalar_ty.element_ty


def validate_normal_copy(src, dst, shape, direction) -> None:
    shape = normalize_copy_shape(shape)
    if direction.name == "GM_TO_LOCAL":
        global_tensor = src
        local_buffer = dst
    else:
        global_tensor = dst
        local_buffer = src

    validate_copy_buffer(local_buffer, shape)
    ptr_shape = tensor_shape(global_tensor)
    if ptr_shape != shape:
        raise ValueError(f"copy shape {shape} must match tensor pointer shape {ptr_shape}")
    elem_ty = tensor_pointer_element_ty(global_tensor)
    if elem_ty != local_buffer.dtype:
        raise ValueError(f"copy dtype mismatch: tensor points to {elem_ty}, buffer stores {local_buffer.dtype}")


def normalize_offsets(offsets, rank: int):
    offsets = tl._unwrap_if_constexpr(offsets)
    if offsets is None:
        raise ValueError("descriptor-based tle.gpu.copy requires offsets")
    if isinstance(offsets, tl.tuple):
        offsets_tuple = tuple(offsets.values)
    elif isinstance(offsets, (tuple, list)):
        offsets_tuple = tuple(offsets)
    elif hasattr(offsets, "__iter__"):
        offsets_tuple = tuple(offsets)
    else:
        raise ValueError(f"offsets must be a tuple or list, but got {type(offsets)}")
    if len(offsets_tuple) != rank:
        raise ValueError(f"offsets must provide {rank} values, got {len(offsets_tuple)}")
    return offsets_tuple


def tmacopy(src, dst, direction, shape, offsets, barrier=None, _semantic=None) -> None:
    shape = normalize_copy_shape(shape)
    desc = src if direction.name == "GM_TO_LOCAL" else dst
    buffer = dst if direction.name == "GM_TO_LOCAL" else src

    validate_copy_buffer(buffer, shape)
    desc_shape = tuple(int(tl._unwrap_if_constexpr(dim)) for dim in desc.block_shape)
    if desc_shape != shape:
        raise ValueError(f"copy shape {shape} must match tensor descriptor block shape {desc_shape}")
    if desc.dtype != buffer.dtype:
        raise ValueError(f"copy dtype mismatch: descriptor stores {desc.dtype}, buffer stores {buffer.dtype}")

    offset_values = normalize_offsets(offsets, len(desc_shape))
    offset_values = _semantic._convert_to_ir_values(offset_values, require_i64=False)
    if not hasattr(_semantic.builder, "create_tma_copy"):
        raise RuntimeError("TLE TMA copy builder binding is not available")
    if barrier is not None:
        if direction.name != "GM_TO_LOCAL":
            raise ValueError("TMA copy barrier is only supported for global-to-shared TMA copy")
        if not isinstance(barrier, tle.barrier) or not barrier.is_slot:
            raise ValueError("TMA copy barrier must be an indexed tle.gpu barrier slot")
        if barrier.expect_bytes is None or int(barrier.expect_bytes) <= 0:
            raise ValueError("TMA copy barrier must have positive expect_bytes")
        _semantic.builder.create_tma_copy(src.handle, dst.handle, offset_values, barrier.handle,
                                          int(barrier.expect_bytes))
        return
    _semantic.builder.create_tma_copy(src.handle, dst.handle, offset_values)
