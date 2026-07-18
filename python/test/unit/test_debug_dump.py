# Copyright 2026 FlagOS Contributors
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

import os
from contextlib import contextmanager

import torch
import triton
import triton.language as tl


@contextmanager
def enable_dump_context(pass_name="1"):
    try:
        os.environ["MLIR_ENABLE_DUMP"] = pass_name
        yield
    finally:
        os.environ["MLIR_ENABLE_DUMP"] = "0"


def test_fn_dump(capfd, device, fresh_triton_cache):
    N = 1024
    src = torch.zeros(N, device=device)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]), )

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    with enable_dump_context():
        BLOCK_SIZE = 16
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    print(captured.err)
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    with enable_dump_context("_kernel"):
        BLOCK_SIZE = 32
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    with enable_dump_context("_kernel2"):
        BLOCK_SIZE = 64
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" not in captured.err
