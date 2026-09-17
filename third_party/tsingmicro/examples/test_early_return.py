import torch

import triton
import triton.language as tl

from triton.backends.ztc.driver import CPUDriver


@triton.jit
def early_return(in0, out0):
    pid = tl.program_id(0)
    id = tl.load(in0 + pid)
    if id == -1:
        return
    offs = 1 + tl.arange(0, 4)
    out_offs = tl.arange(0, 4)
    tl.store(out0 + out_offs, offs)


def compile(device):
    src = triton.compiler.ASTSource(
        fn=early_return,
        signature="*fp32,*fp32",
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])


def test_return_case(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 8
    input = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    early_return[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int32), output)


def test_normal_case(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 8
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    early_return[grid](input, output)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([1, 2, 3, 4, -1, -1, -1, -1], dtype=torch.int32), output)
