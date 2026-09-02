# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._allocator import get_global_state_pointer
from triton._C.libtriton.gsan_testing import thread_state_address, SHADOW_GRANULARITY_BYTES, PER_DEVICE_STATE_STRIDE_BYTES, GLOBAL_STATE_SIZE_BYTES, shadow_cell_address, thread_state_stride_bytes, SHADOW_CELL_SIZE_BYTES
from ._testing import (decode_global_state_tensor, decode_shadow_cell_tensor, decode_thread_state_tensor)
from ._utils import uint8_cuda_tensor_from_ptr


@triton.jit
def nanosleep(duration):
    duration = tl.to_tensor(duration)
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.b32 $0, 0;", "=r, r", [duration], tl.int32, is_pure=False, pack=1)


@triton.jit
def atomic_poll(ptr, expect, sem: tl.constexpr = "relaxed", scope: tl.constexpr = "gpu"):
    while tl.atomic_add(ptr, 0, sem=sem, scope=scope) != expect:
        nanosleep(100)


def shadow_cell_tensor_from_address(real_address: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    shadow_ptr = shadow_cell_address(real_address)
    return uint8_cuda_tensor_from_ptr(shadow_ptr, SHADOW_CELL_SIZE_BYTES, device_index)


def shadow_cell_from_address(real_address: int, *, device_index: int | None = None):
    return decode_shadow_cell_tensor(shadow_cell_tensor_from_address(real_address, device_index=device_index))


def global_state_tensor(*, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    ptr = get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES
    return uint8_cuda_tensor_from_ptr(ptr, GLOBAL_STATE_SIZE_BYTES, device_index)


def global_state(*, device_index: int | None = None):
    return decode_global_state_tensor(global_state_tensor(device_index=device_index))


def thread_state_tensor(smid: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    ptr = thread_state_address(
        get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES,
        gs.num_threads,
        gs.clock_buffer_size,
        smid,
    )
    size = thread_state_stride_bytes(gs.num_threads, gs.clock_buffer_size)
    return uint8_cuda_tensor_from_ptr(ptr, size, device_index)


def thread_state_from_smid(smid: int, *, device_index: int | None = None):
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    return decode_thread_state_tensor(
        thread_state_tensor(smid, device_index=device_index),
        gs.num_threads,
        gs.clock_buffer_size,
    )


def shadow_tensor_for(real: torch.Tensor) -> torch.Tensor:
    shadow_ptr = shadow_cell_address(real.data_ptr())
    nbytes = real.untyped_storage().nbytes()
    num_cells = triton.cdiv(nbytes, SHADOW_GRANULARITY_BYTES)
    shadow_size = num_cells * SHADOW_CELL_SIZE_BYTES
    return uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, real.device.index)


@triton.jit
def store_one_i32(ptr):
    tl.store(ptr, 1)


@triton.jit
def load_one_i32(ptr, out_ptr):
    value = tl.load(ptr)
    tl.store(out_ptr, value)
