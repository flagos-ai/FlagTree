import torch
import pytest
import math
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
    grid = lambda meta: (n_cols, )

    reduce_kernel_2d[grid](x, output, x.stride(0), n_rows, BLOCK_SIZE=BLOCK_SIZE)
    ans = torch.sum(x, dim=1)
    torch.testing.assert_close(output, ans, rtol=0.001, atol=1e-5)

    # TODO: need to check some conditions otherwise the code below does not make any difference for the test
    src = triton.compiler.ASTSource(
        fn=reduce_kernel_2d, signature={
            "x_ptr": "*fp32", "output_ptr": "*fp32", "stride": "i32", "n_elements": "i32", "BLOCK_SIZE": "constexpr"
        }, constexprs={"BLOCK_SIZE": 32})
    ret = triton.compile(src, target=GPUTarget(device, 0, 0))
    print(ret.asm["ttir"])
    print(ret.asm["ttsharedir"])
    print(ret.asm["llir"])
    print(ret.asm["obj"])


@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "float32"])
@pytest.mark.parametrize("shape", [(128, 2, 4), (64, 2, 4), (32, 2, 4), (2, 4, 32), (2, 4, 2)])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_reduce_max(dtype_str, shape, axis, device):

    @triton.jit
    def kernel(
        In,
        Out,
        in_shape1: tl.constexpr,
        in_shape2: tl.constexpr,
        in_shape3: tl.constexpr,
        ou_shape1: tl.constexpr,
        ou_shape2: tl.constexpr,
        axis: tl.constexpr,
    ):
        in_desc = tl.make_tensor_descriptor(
            base=In,
            shape=[in_shape1 * in_shape2 * in_shape3],
            strides=[1],
            block_shape=[in_shape1 * in_shape2 * in_shape3],
        )
        out_desc = tl.make_tensor_descriptor(
            base=Out,
            shape=[ou_shape1 * ou_shape2],
            strides=[1],
            block_shape=[ou_shape1 * ou_shape2],
        )
        val = in_desc.load([0]).reshape(in_shape1, in_shape2, in_shape3)
        output = tl.max(val, axis=axis)
        out_desc.store([0], output.reshape(out_desc.block_shape))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str),
                         device="cpu").reshape(shape).to(device=device)
    expected, indices = torch.max(input, dim=axis)
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](input, actual, *shape, *expected.shape, axis=axis)

    assert torch.equal(expected, actual)
