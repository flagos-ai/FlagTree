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

import pytest
import triton

from triton._filecheck import run_filecheck_test


@triton.jit
def anchor(v):
    pass


# Smoke test to make sure filecheck is working correctly.
def test_filecheck_positive():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 42
        # CHECK: %c42_i32 = arith.constant 42 : i32
        # CHECK-NEXT: call @{{.*}}anchor{{.*}}(%c42_i32) : (i32) -> ()
        anchor(scalar)

    run_filecheck_test(test_kernel)


def test_filecheck_negative():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: test_kernel
        scalar = 11
        # CHECK: %c42_i32
        anchor(scalar)

    with pytest.raises(ValueError, match="expected string not found in input\n # CHECK: %c42_i32"):
        run_filecheck_test(test_kernel)
