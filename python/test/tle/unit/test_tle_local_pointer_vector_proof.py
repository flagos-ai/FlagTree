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
"""Register layout contiguity is not a proof of pointer contiguity."""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_layout = tl.constexpr(tle.gpu.BlockEncoding([4], [32], [4], [0]))


@triton.jit
def _strided_shared_load(source, output, STEP: tl.constexpr):
    shared = tle.gpu.alloc(
        [512],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    indices = tl.arange(0, 512)
    pointer = tle.gpu.local_ptr(shared, (indices, ))
    tl.store(pointer, tl.load(source + indices))
    gather = tle.gpu.set_layout(tl.arange(0, 128), _layout)
    gathered = tle.gpu.local_ptr(shared, (gather * STEP, ))
    values = tle.gpu.set_layout(tl.load(gathered), _layout)
    tl.store(output + gather, values)


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.skip_if_tle_unsupported("gpu.alloc", "gpu.local_ptr", "gpu.set_layout")
def test_shared_vector_load_requires_pointer_contiguity(step):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    source = torch.arange(512, dtype=torch.float32, device="cuda")
    result = torch.empty(128, dtype=torch.float32, device="cuda")
    _strided_shared_load[(1, )](source, result, step)
    torch.testing.assert_close(result, source[:128 * step:step], rtol=0, atol=0)
