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
"""Inactive lanes must not retain a predicated reduction load's old register."""

import re

import pytest
import torch
import triton
import triton.language as tl


@triton.jit(noinline=True)
def _softmax_row(source, output):
    offsets = tl.arange(0, 256)
    values = tl.load(source + offsets)
    numerator = tl.exp(values - tl.max(values, 0))
    tl.store(output + offsets, numerator / tl.sum(numerator, 0))


@triton.jit
def _repeated_softmax(source, output, ROWS: tl.constexpr):
    for row in range(ROWS):
        _softmax_row(source + row * 256, output + row * 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skip_if_tle_unsupported("gpu.set_layout")
@pytest.mark.parametrize("warps", (4, 8))
def test_predicated_reduction_load_has_no_live_in_register(warps):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    values = torch.randn((3, 256), device="cuda")
    output = torch.empty_like(values)
    kernel = _repeated_softmax[(1, )](values, output, 3, num_warps=warps)
    torch.testing.assert_close(output, values.softmax(1), rtol=2e-6, atol=2e-7)
    # A predicated inline-asm output without initialization is a PTX live-in
    # on inactive lanes. In a device function ptxas saves it to local memory,
    # even though those lanes' reduction values are discarded.
    loads = list(re.finditer(r"@%p\d+ ld\.shared\.b32 (%r\d+),", kernel.asm["ptx"]))
    assert loads, "The reproducer must exercise inter-warp shared reduction"
    for load in loads:
        preceding = kernel.asm["ptx"][:load.start()].rstrip()
        initialized = r"mov\.(?:u|b)32\s+" + re.escape(load.group(1)) + r",\s*(?:0|0x0);$"
        assert re.search(initialized, preceding), "Predicated shared load retains an undefined register"
    assert kernel.metadata.ptxas_spill_store_bytes == 0
    assert kernel.metadata.ptxas_spill_load_bytes == 0
