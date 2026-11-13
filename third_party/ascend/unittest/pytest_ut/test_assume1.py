import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

from triton._internal_testing import (is_interpreter)


@triton.jit
def assume(out_ptr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    current_size = N - tl.program_id(0) * BLOCK_N
    tl.assume(current_size >= BLOCK_N)
    if current_size >= BLOCK_N:
        tl.store(out_ptr + tl.program_id(0), current_size)
    else:
        tl.store(out_ptr + tl.program_id(0), current_size + 101024)


@pytest.mark.parametrize('dtype', ["float32"])
def test_assume(dtype):
    NBLOCKS = 1024 // 128
    BLOCK_N = 128
    N = 1024
    output = torch.zeros(NBLOCKS, device='npu')
    pgm = assume[(NBLOCKS, )](output, N=N, BLOCK_N=BLOCK_N)

    if is_interpreter():
        return

    assert 'llvm.intr.assume' in pgm.asm['ttadapter']
