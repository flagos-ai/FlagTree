import os
import pytest

import triton
import triton.language as tl


@triton.jit
def _empty_kernel():
    return


@pytest.mark.interpreter
def test_launcher_empty_signature():
    grid = (1,)
    _empty_kernel[grid]()
    assert True
