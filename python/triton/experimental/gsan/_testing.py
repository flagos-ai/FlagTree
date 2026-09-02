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

from typing import Any

from triton._C.libtriton import gsan_testing as _gsan_testing

from triton._C.libtriton.gsan_testing import (
    ShadowCell,
    GlobalState,
    ThreadState,
    shadow_cell_address,
    thread_state_stride_bytes,
    decode_shadow_cell,
    SHADOW_CELL_SIZE_BYTES,
    GLOBAL_STATE_SIZE_BYTES,
)


def shadow_cell(real_address: int) -> tuple[int, int]:
    return shadow_cell_address(real_address), SHADOW_CELL_SIZE_BYTES


def decode_global_state(data: bytes | bytearray | memoryview) -> GlobalState:
    return _gsan_testing.decode_global_state(bytes(data))


def decode_global_state_tensor(state_bytes: Any) -> GlobalState:
    if hasattr(state_bytes, "detach"):
        tensor = state_bytes.detach().cpu().contiguous().view(-1)
        return decode_global_state(bytes(tensor[:GLOBAL_STATE_SIZE_BYTES].tolist()))
    return decode_global_state(bytes(state_bytes))


def decode_shadow_cell_tensor(cell_bytes: Any) -> ShadowCell:
    # Accept a torch.Tensor without importing torch at module import time.
    if hasattr(cell_bytes, "detach"):
        tensor = cell_bytes.detach().cpu().contiguous().view(-1)
        return decode_shadow_cell(bytes(tensor[:SHADOW_CELL_SIZE_BYTES].tolist()))
    return decode_shadow_cell(bytes(cell_bytes))


def decode_thread_state(data: bytes | bytearray | memoryview, num_threads: int, clock_buffer_size: int) -> ThreadState:
    return _gsan_testing.decode_thread_state(bytes(data), int(num_threads), int(clock_buffer_size))


def decode_thread_state_tensor(state_bytes: Any, num_threads: int, clock_buffer_size: int) -> ThreadState:
    if hasattr(state_bytes, "detach"):
        tensor = state_bytes.detach().cpu().contiguous().view(-1)
        size = thread_state_stride_bytes(num_threads, clock_buffer_size)
        return decode_thread_state(bytes(tensor[:size].tolist()), num_threads, clock_buffer_size)
    return decode_thread_state(bytes(state_bytes), num_threads, clock_buffer_size)
