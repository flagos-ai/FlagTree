import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def triton_if_advance_kernel(in_ptr0, in_ptr1, out_ptr, xnumel, ynumel, k_loops, XBLOCK: tl.constexpr,
                             YBLOCK: tl.constexpr):

    K_block_ptr = tl.make_block_ptr(base=in_ptr0, shape=(xnumel, ynumel), strides=(ynumel, 1), offsets=(0, 0),
                                    block_shape=(XBLOCK, YBLOCK), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=in_ptr1, shape=(ynumel, xnumel), strides=(xnumel, 1), offsets=(0, 0),
                                    block_shape=(YBLOCK, XBLOCK), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(base=out_ptr, shape=(xnumel, xnumel), strides=(xnumel, 1), offsets=(0, 0),
                                    block_shape=(XBLOCK, XBLOCK), order=(1, 0))
    res = tl.zeros([XBLOCK, XBLOCK], tl.float32)
    for i in range(0, k_loops):
        if i > 0:
            K_block_ptr = tl.advance(K_block_ptr, (0, YBLOCK))
            V_block_ptr = tl.advance(V_block_ptr, (YBLOCK, 0))
        a = tl.load(K_block_ptr)
        b = tl.load(V_block_ptr)
        res = tl.dot(a, b, acc=res)
    tl.store(O_block_ptr, res)


def test_if_advance():
    x = torch.randn((64, 256), dtype=torch.float32, device="npu")
    y = torch.randn((256, 64), dtype=torch.float32, device="npu")
    out_tri = torch.empty((64, 64), dtype=torch.float32, device="npu")
    out_std = torch.empty((64, 64), dtype=torch.float32, device="npu")
    torch.matmul(x, y, out=out_std)
    triton_if_advance_kernel[1, 1, 1](x, y, out_tri, 64, 256, 4, 64, 64)
    torch.testing.assert_close(out_std, out_tri, atol=1e-2, rtol=1e-2)


test_if_advance()
