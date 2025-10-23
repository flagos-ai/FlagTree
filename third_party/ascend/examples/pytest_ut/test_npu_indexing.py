import torch
import torch_npu
import triton
import triton.language as tl
import time


def foo(a, b, c):
    Z, Y, X, R = (1, 1, 64, 64)
    y = a + b
    y = y.sum(-1)
    y = y.unsqueeze(3)
    y = y.broadcast_to(Z, Y, X, R) + b
    y = c + y.permute(0, 1, 3, 2)
    return y


@triton.jit
def triton_foo(in_ptr0, in_ptr1, in_ptr2, out_ptr0, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
                BLOCK2: tl.constexpr,
                Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr, R: tl.constexpr,
                Z_STRIDE: tl.constexpr, Y_STRIDE: tl.constexpr, X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
                Z_STRIDE1: tl.constexpr, Y_STRIDE1: tl.constexpr, X_STRIDE1: tl.constexpr, R_STRIDE1: tl.constexpr,
                ):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    base1 = tl.arange(0, BLOCK1_SUB)
    base2 = tl.arange(0, BLOCK2)
    nsub: tl.constexpr = BLOCK1 // BLOCK1_SUB
    # loops1 : tl.constexpr =  nsub * Y * Z
    loops1: tl.constexpr = nsub
    loops2: tl.constexpr = R // BLOCK2

    for z in range(Z):
        for y in range(Y):
            for loop1 in range(loops1):
                # y =  (loop1 // nsub) % Y
                # z = loop1 // nsub // Y
                # off1 = (loop1 % nsub)
                off1 = loop1
                x = offset + (off1 * BLOCK1_SUB) + base1[:, None]
                x1 = offset + (off1 * BLOCK1_SUB) + base1[None, :]
                _tmp4 = tl.full([BLOCK1_SUB, BLOCK2], 0, tl.float32)
                for loop2 in range(loops2):
                    r = loop2 * BLOCK2 + base2[None, :]
                    tmp0 = tl.load(in_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp1 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp2 = tmp0 + tmp1
                    _tmp4 = _tmp4 + tmp2
                tmp4 = tl.sum(_tmp4, 1)[:, None]
                tmp5 = tmp4.reshape(BLOCK1_SUB, 1).broadcast_to(BLOCK1_SUB, BLOCK2)

                for loop2 in range(loops2):
                    r = loop2 * BLOCK2 + base2[None, :]
                    tmp6 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp7 = tmp6 + tmp5
                    r1 = loop2 * BLOCK2 + base2[:, None]
                    tmp8 = tl.load(in_ptr2 + (R_STRIDE1 * x1 + (X_STRIDE1 * r1) + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)),
                                   None)

                    tmp9 = tmp8.reshape(BLOCK2, BLOCK1_SUB) + tmp7.reshape(BLOCK1_SUB, BLOCK2).permute(1, 0)
                    tl.store(out_ptr0 + (R_STRIDE1 * x1 + (X_STRIDE1 * r1) + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)), tmp9,
                             None)


def foo_triton_wrapper(a, b, c):
    NBLOCKS = 1
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = 64
    BLOCK2 = 64

    value = torch.empty_strided((c.shape[0], c.shape[1], c.shape[2], c.shape[3]),
                                (c.stride()[0], c.stride()[1], c.stride()[2], c.stride()[3]), dtype=torch.float32).npu()

    triton_foo[NBLOCKS, 1, 1](a, b, c, value, BLOCK1, BLOCK1_SUB, BLOCK2,
                   a.shape[0], a.shape[1], a.shape[2], a.shape[3],
                   a.stride()[0], a.stride()[1], a.stride()[2], a.stride()[3],
                   c.stride()[0], c.stride()[1], c.stride()[2], c.stride()[3],)
    return value

def test_npu_indexing():
    Z, Y, X, R = (1, 1, 64, 64)
    a = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    b = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    c = torch.randn((Z, Y, R, X), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[0, 0, 0:8, 0:8])
    print(r1[0, 0, 0:8, 0:8])
    torch.testing.assert_close(r, r1)
