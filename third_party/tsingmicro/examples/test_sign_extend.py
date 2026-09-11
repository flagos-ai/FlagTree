import torch

import triton

import triton.language as tl

from triton.backends.ztc.driver import CPUDriver


@triton.jit
def sign_extend(off, in0, out0, in0_size):
    offset = tl.load(off).to(tl.int64)
    offsets = offset + tl.arange(0, 4)
    a = tl.load(in0 + offsets, mask=offsets < in0_size, other=11)
    tl.store(out0 + tl.arange(0, 4), a)


def compile():
    src = triton.compiler.ASTSource(
        fn=sign_extend,
        signature="*i32,*fp32,*fp32,i32",
    )
    ret = triton.compile(src, )
    print(ret.asm["ttir"])


def test_sign_extend(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    SIZE = 4
    offsets = torch.full((1, ), 1, device=device, dtype=torch.int32)
    input = torch.arange(0, SIZE, device=device, dtype=torch.int32)
    output = torch.full((SIZE, ), -1, device=device, dtype=torch.int32)
    grid = lambda meta: (1, )
    print(output)
    sign_extend[grid](offsets, input, output, SIZE)
    print(input)
    print(output)
    torch.testing.assert_close(torch.tensor([1, 2, 3, 11], device=device, dtype=torch.int32), output)
