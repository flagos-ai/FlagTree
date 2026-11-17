# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

BLOCK = 16
INPUT_SIZE = 256


@triton.jit
def scalar_pointer_if_kernel(x, out, predicate_ptr):
    predicate = tl.load(predicate_ptr)
    if predicate != 0:
        pointer = x + 3
    else:
        pointer = x + 11
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_for_kernel(x, out, steps_ptr):
    steps = tl.load(steps_ptr)
    pointer = x + 2
    for _ in tl.range(0, steps):
        pointer = pointer + 3
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_while_kernel(x, out, steps_ptr):
    steps = tl.load(steps_ptr)
    pointer = x + 1
    iteration = 0
    while iteration < steps:
        pointer = pointer + 2
        iteration += 1
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_while_from_function_args_kernel(x, out, steps_ptr):
    """Keep function-argument bases outside the loop and carry only offsets."""
    steps = tl.load(steps_ptr)
    input_pointer = x
    output_pointer = out
    iteration = 0
    while iteration < steps:
        input_pointer = input_pointer + 2
        output_pointer = output_pointer + 3
        iteration += 1
    tl.store(output_pointer, tl.load(input_pointer))


@triton.jit
def scalar_pointer_while_if_backedge_kernel(x0, x1, out, steps_ptr, switch_at_ptr):
    """Select a new scalar-pointer base on one while-loop backedge."""
    steps = tl.load(steps_ptr)
    switch_at = tl.load(switch_at_ptr)
    pointer = x0 + 1
    iteration = 0
    while iteration < steps:
        if iteration == switch_at:
            pointer = x1 + 7
        else:
            pointer = pointer + 2
        iteration += 1
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_nested_if_for_kernel(x, out, predicate_ptr, steps_ptr):
    predicate = tl.load(predicate_ptr)
    steps = tl.load(steps_ptr)
    pointer = x
    for iteration in tl.range(0, steps):
        if (iteration & 1) == predicate:
            pointer = pointer + 2
        else:
            pointer = pointer + 1
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_select_loop_kernel(x0, x1, out, predicate_ptr, steps_ptr):
    predicate = tl.load(predicate_ptr)
    steps = tl.load(steps_ptr)
    pointer = tl.where(predicate != 0, x0 + 4, x1 + 7)
    for _ in tl.range(0, steps):
        pointer = pointer + 2
    tl.store(out, tl.load(pointer))


@triton.jit
def scalar_pointer_if_with_tensor_mask_kernel(x0, x1, out, predicate_ptr, limit_ptr, BLOCK: tl.constexpr):
    """Return a scalar pointer and an independently computed tensor mask."""
    predicate = tl.load(predicate_ptr)
    limit = tl.load(limit_ptr)
    lane = tl.arange(0, BLOCK)
    if predicate != 0:
        pointer = x0 + 3
        valid = lane < limit
    else:
        pointer = x1 + 7
        valid = lane < BLOCK - 2
    value = tl.load(pointer + lane, mask=valid, other=-1.0)
    tl.store(out + lane, value)


@triton.jit
def tensor_pointer_loop_kernel(tensor_source, tensor_out, steps_ptr, BLOCK: tl.constexpr):
    """Exercise one tensor-of-pointers loop carrier across several iterations."""
    steps = tl.load(steps_ptr)
    lane = tl.arange(0, BLOCK)
    tensor_pointer = tensor_source + lane + 6
    for _ in tl.range(0, steps):
        tensor_pointer += 2
    tl.store(tensor_out + lane, tl.load(tensor_pointer))


@triton.jit
def descriptor_boundary_scalar_kernel(x0, x1, out, steps_ptr, switch_at_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    steps = tl.load(steps_ptr)
    switch_at = tl.load(switch_at_ptr)
    pointer = tl.make_block_ptr(
        base=x0,
        shape=(N, ),
        strides=(1, ),
        offsets=(0, ),
        block_shape=(BLOCK, ),
        order=(0, ),
    )
    for iteration in tl.range(0, steps):
        if iteration == switch_at:
            pointer = tl.make_block_ptr(
                base=x1,
                shape=(N, ),
                strides=(1, ),
                offsets=(4, ),
                block_shape=(BLOCK, ),
                order=(0, ),
            )
        else:
            pointer = tl.advance(pointer, (1, ))
    value = tl.load(pointer, boundary_check=(0, ), padding_option="zero")
    tl.store(out + tl.arange(0, BLOCK), value)


def _inputs():
    x0_cpu = torch.arange(INPUT_SIZE, dtype=torch.float32)
    x1_cpu = 1000.0 + 2.0 * torch.arange(INPUT_SIZE, dtype=torch.float32)
    return x0_cpu, x1_cpu, x0_cpu.npu(), x1_cpu.npu()


def _device_i32(value):
    return torch.tensor([value], dtype=torch.int32, device="npu")


def _assert_output(actual, expected):
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("predicate", [0, 1])
def test_scalar_pointer_if_yield(predicate):
    x0_cpu, _, x0, _ = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_if_kernel[(1, )](x0, out, _device_i32(predicate))

    expected_index = 3 if predicate else 11
    _assert_output(out, x0_cpu[expected_index:expected_index + 1])


@pytest.mark.parametrize("steps", [0, 1, 4])
def test_scalar_pointer_for_carried(steps):
    x0_cpu, _, x0, _ = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_for_kernel[(1, )](x0, out, _device_i32(steps))

    expected_index = 2 + 3 * steps
    _assert_output(out, x0_cpu[expected_index:expected_index + 1])


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_scalar_pointer_while_carried(steps):
    x0_cpu, _, x0, _ = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_while_kernel[(1, )](x0, out, _device_i32(steps))

    expected_index = 1 + 2 * steps
    _assert_output(out, x0_cpu[expected_index:expected_index + 1])


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_scalar_pointer_while_from_function_args(steps):
    x0_cpu, _, x0, _ = _inputs()
    out_cpu = torch.zeros(INPUT_SIZE, dtype=torch.float32)
    out = out_cpu.npu()

    scalar_pointer_while_from_function_args_kernel[(1, )](x0, out, _device_i32(steps))

    expected_input_index = 2 * steps
    expected_output_index = 3 * steps
    expected = torch.zeros_like(out_cpu)
    expected[expected_output_index] = x0_cpu[expected_input_index]
    _assert_output(out, expected)


@pytest.mark.parametrize("steps,switch_at", [(0, 0), (1, 0), (4, 1), (4, 7)])
def test_scalar_pointer_while_if_backedge(steps, switch_at):
    x0_cpu, x1_cpu, x0, x1 = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_while_if_backedge_kernel[(1, )](
        x0,
        x1,
        out,
        _device_i32(steps),
        _device_i32(switch_at),
    )

    if switch_at < steps:
        expected = x1_cpu[7 + 2 * (steps - switch_at - 1):]
    else:
        expected = x0_cpu[1 + 2 * steps:]
    _assert_output(out, expected[:1])


@pytest.mark.parametrize("predicate", [0, 1])
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_scalar_pointer_nested_if_for(predicate, steps):
    x0_cpu, _, x0, _ = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_nested_if_for_kernel[(1, )](x0, out, _device_i32(predicate), _device_i32(steps))

    expected_index = sum(2 if (iteration & 1) == predicate else 1 for iteration in range(steps))
    _assert_output(out, x0_cpu[expected_index:expected_index + 1])


@pytest.mark.parametrize("predicate", [0, 1])
@pytest.mark.parametrize("steps", [0, 3])
def test_scalar_pointer_select_into_loop(predicate, steps):
    x0_cpu, x1_cpu, x0, x1 = _inputs()
    out = torch.empty(1, dtype=torch.float32, device="npu")

    scalar_pointer_select_loop_kernel[(1, )](x0, x1, out, _device_i32(predicate), _device_i32(steps))

    source = x0_cpu if predicate else x1_cpu
    expected_index = (4 if predicate else 7) + 2 * steps
    _assert_output(out, source[expected_index:expected_index + 1])


@pytest.mark.parametrize("predicate", [0, 1])
def test_scalar_pointer_if_with_tensor_mask(predicate):
    """Check mixed scalar-pointer/tensor-mask results across one scf.if."""
    block = 8
    limit = 5
    x0_cpu, x1_cpu, x0, x1 = _inputs()
    out = torch.empty(block, dtype=torch.float32, device="npu")

    scalar_pointer_if_with_tensor_mask_kernel[(1, )](
        x0,
        x1,
        out,
        _device_i32(predicate),
        _device_i32(limit),
        BLOCK=block,
    )

    source = x0_cpu if predicate else x1_cpu
    offset = 3 if predicate else 7
    valid_count = limit if predicate else block - 2
    expected = torch.full((block, ), -1.0, dtype=torch.float32)
    expected[:valid_count] = source[offset:offset + valid_count]
    _assert_output(out, expected)


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_tensor_pointer_loop(steps):
    """Check a tensor-of-pointers backedge through a real NPU load/store."""
    block = 8
    length = 64
    source_cpu = torch.arange(length, dtype=torch.float32)
    source = source_cpu.npu()
    tensor_output = torch.empty(block, dtype=torch.float32, device="npu")
    steps_device = _device_i32(steps)

    tensor_pointer_loop_kernel[(1, )](source, tensor_output, steps_device, BLOCK=block)

    tensor_start = 6 + 2 * steps
    _assert_output(tensor_output, source_cpu[tensor_start:tensor_start + block])


@pytest.mark.parametrize("steps,switch_at", [(0, -1), (3, -1), (3, 1)])
def test_descriptor_boundary_scalar_regression(steps, switch_at):
    x0_cpu, x1_cpu, x0, x1 = _inputs()
    out = torch.empty(BLOCK, dtype=torch.float32, device="npu")

    descriptor_boundary_scalar_kernel[(1, )](
        x0,
        x1,
        out,
        _device_i32(steps),
        _device_i32(switch_at),
        N=INPUT_SIZE,
        BLOCK=BLOCK,
    )

    source = x0_cpu
    offset = 0
    for iteration in range(steps):
        if iteration == switch_at:
            source = x1_cpu
            offset = 4
        else:
            offset += 1
    _assert_output(out, source[offset:offset + BLOCK])
