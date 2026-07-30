import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def test_scalar_store(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    base_ptr = output_ptr + pid0
    for i in range(0, BLOCK_SIZE // 2):
        output = i * 2
        for j in range(0, BLOCK_SIZE // 4):
            output += j
            tl.store(base_ptr, output)
            base_ptr += 1


def compile():
    src = triton.compiler.ASTSource(fn=test_scalar_store, signature="*fp32", constexprs={"BLOCK_SIZE": 8})
    ret = triton.compile(src)
    print(ret.asm["ttir"])


def test(device):
    if device == 'cpu':
        triton.runtime.driver.set_active(CPUDriver())

    BLOCK_SIZE = 8
    x = torch.full([BLOCK_SIZE], -1, device=device, dtype=torch.float32)
    output = torch.full((BLOCK_SIZE, ), -99, device=device, dtype=x.dtype)
    grid = lambda meta: (1, )

    print(x)
    print(output)

    test_scalar_store[grid](output, BLOCK_SIZE=BLOCK_SIZE)
    print('---')
    print(output)
    ans = torch.arange(BLOCK_SIZE, device=device, dtype=torch.float32)
    torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)
