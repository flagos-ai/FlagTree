import torch

import triton
from triton.backends.compiler import GPUTarget
import triton.language as tl


@triton.jit
def reduce_kernel_2d(
    x_ptr,
    output_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    x = tl.load(
        tl.make_block_ptr(
            base=x_ptr,
            shape=[n_elements * tl.num_programs(0)],
            strides=[1],
            offsets=[stride * pid0],
            block_shape=[BLOCK_SIZE],
            order=[0],
        ),
        boundary_check=[0],
    )
    output = triton.language.sum(x, axis=0).to(dtype=x.dtype)
    tl.store(output_ptr + pid0, output)


def test(device):
    n_rows = 16
    n_cols = 32
    x = torch.rand([n_cols, n_rows], device=device, dtype=torch.float32)
    output = torch.empty([n_cols], device=device, dtype=x.dtype)
    BLOCK_SIZE = n_rows
    grid = lambda meta: (n_cols,)

    reduce_kernel_2d[grid](x, output, x.stride(0), n_rows, BLOCK_SIZE=BLOCK_SIZE)
    ans = torch.sum(x, dim=1)
    torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)

    # TODO: need to check some conditions otherwise the code below does not make any difference for the test
    src = triton.compiler.ASTSource(
        fn=reduce_kernel_2d,
        signature="*fp32,*fp32,i32,i32",
        constants={"BLOCK_SIZE": 32}
    )
    ret = triton.compile(
        src,
        target=GPUTarget(device, 0, 0)
    )
    print(ret.asm["ttir"])
    print(ret.asm["ttsharedir"])
    print(ret.asm["llir"])
    print(ret.asm["cpuasm"])
