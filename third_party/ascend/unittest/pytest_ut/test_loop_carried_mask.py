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
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def loop_carried_mask(x, out, written, final_mask, limits, steps_ptr, initial_ptr, OBSERVE: tl.constexpr,
                      READ_BEFORE: tl.constexpr, REVERSE_AND: tl.constexpr, DISCRETE: tl.constexpr,
                      SUFFIX: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    steps = tl.load(steps_ptr)
    initial = tl.load(initial_ptr)
    mask = lane < initial
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.range(0, steps):
        if READ_BEFORE:
            acc += tl.load(x + lane, mask=mask, other=0.0)
        limit = tl.load(limits + i)
        if SUFFIX:
            current = lane >= limit
        else:
            current = lane < limit
        if DISCRETE:
            current = current & ((lane & 1) == 0)
        if REVERSE_AND:
            mask = current & mask
        else:
            mask = mask & current
        acc += tl.load(x + lane, mask=mask, other=0.0)
        tl.store(written + lane, acc, mask=mask)
    tl.store(out + lane, acc)
    if OBSERVE:
        tl.store(final_mask + lane, mask.to(tl.int32))


def check_loop_mask(limits, initial, observe=False, read_before=False, reverse_and=False, discrete=False, suffix=False,
                    compile_mode="simd_simt_template", block=16):
    lane = torch.arange(block)
    host_x = (lane.float() + 1) / 4
    x = host_x.npu()
    out = torch.empty_like(x)
    written = torch.full_like(x, -777)
    final_mask = torch.empty(block, dtype=torch.int32, device="npu")
    device_limits = torch.tensor(limits or [0], dtype=torch.int32, device="npu")
    steps_ptr = torch.tensor([len(limits)], dtype=torch.int32, device="npu")
    initial_ptr = torch.tensor([initial], dtype=torch.int32, device="npu")
    loop_carried_mask[(1, )](x, out, written, final_mask, device_limits, steps_ptr, initial_ptr, OBSERVE=observe,
                             READ_BEFORE=read_before, REVERSE_AND=reverse_and, DISCRETE=discrete, SUFFIX=suffix,
                             BLOCK=block, compile_mode=compile_mode)
    torch.npu.synchronize()

    # Keep the original boolean recurrence as the independent reference.
    mask = lane < initial
    expected = torch.zeros(block, dtype=torch.float32)
    expected_written = torch.full_like(expected, -777)
    for limit in limits:
        if read_before:
            expected += torch.where(mask, host_x, 0.0)
        current = lane >= limit if suffix else lane < limit
        if discrete:
            current &= (lane & 1) == 0
        mask &= current
        expected += torch.where(mask, host_x, 0.0)
        expected_written[mask] = expected[mask]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(written.cpu(), expected_written, rtol=0, atol=0)
    if observe:
        torch.testing.assert_close(final_mask.cpu(), mask.int(), rtol=0, atol=0)


@pytest.mark.parametrize("compile_mode", ["simd", "simd_simt_template"])
@pytest.mark.parametrize("limits,initial,observe,read_before,reverse_and", [
    pytest.param([], 7, True, False, False, id="zero-iterations"),
    pytest.param([15], 16, False, False, False, id="one-iteration"),
    pytest.param([15, 14, 13], 16, False, False, False, id="shrinking"),
    pytest.param([4, 12], 16, False, False, False, id="shrink-then-expand"),
    pytest.param([4, 12, 4], 16, False, False, False, id="alternating"),
    pytest.param([-4, 12], 16, False, False, False, id="negative-bound"),
    pytest.param([20, 4], 16, False, False, False, id="above-block"),
    pytest.param([12, 8], 3, False, False, False, id="narrow-initial"),
    pytest.param([12], 0, True, False, False, id="empty-initial"),
    pytest.param(list(range(15, -2, -1)), 16, False, False, False, id="past-empty"),
    pytest.param([4, 12], 16, True, False, False, id="final-mask"),
    pytest.param([4, 12], 7, True, True, False, id="use-before-update"),
    pytest.param([4, 12], 16, False, False, True, id="commuted-and"),
])
def test_loop_prefix_mask(limits, initial, observe, read_before, reverse_and, compile_mode):
    check_loop_mask(limits, initial, observe, read_before, reverse_and, compile_mode=compile_mode)


@pytest.mark.parametrize("observe", [False, True])
@pytest.mark.parametrize("suffix", [False, True])
def test_loop_runtime_mask(observe, suffix):
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("runtime loop masks require masked indirect access on Ascend910_95/950")

    limits = [12, 4] if suffix else [4, 12]
    check_loop_mask(limits, 16, observe=observe, discrete=not suffix, suffix=suffix)


@pytest.mark.parametrize("compile_mode", ["simd", "simd_simt_template"])
def test_loop_singleton_mask(compile_mode):
    check_loop_mask([0, 2, 0], 1, observe=True, suffix=True, compile_mode=compile_mode, block=1)
