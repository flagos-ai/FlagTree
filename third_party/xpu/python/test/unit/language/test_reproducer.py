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
import shutil

import pytest

import torch
import triton
import re


@triton.jit
def triton_():
    return


@pytest.mark.skipif(True, reason="no ir dump support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_reproducer():
    tmpdir = ".tmp"
    reproducer = 'triton-reproducer.mlir'
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    if os.path.exists(reproducer):
        os.remove(reproducer)
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    os.environ["TRITON_REPRODUCER_PATH"] = reproducer
    triton_[(1, )]()
    foundPipeline = ""
    with open(reproducer, 'r') as f:
        line = f.read()
        if 'pipeline:' in line:
            foundPipeline = line
    if 0 == len(foundPipeline):
        raise Exception("Failed to find pipeline info in reproducer file.")

    ttgir_to_llvm_pass = re.compile("convert-triton-{{.*}}gpu-to-llvm")
    if ttgir_to_llvm_pass.search(foundPipeline):
        raise Exception("Failed to find triton passes in pipeline")
    # cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    if os.path.exists(reproducer):
        os.remove(reproducer)
