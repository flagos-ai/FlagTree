import torch

import triton
import triton.language as tl


def test_wrap_stacked(device):

    @triton.jit
    def wrap_stacked(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm,
                     stride_cn, BLOCK_SIZE_K: tl.constexpr):
        offs_am = (2 + tl.arange(0, 4)) % M
        offs_an = tl.arange(0, 4)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_an[None, :] * stride_an)

        offs_cm = tl.arange(0, 4)
        offs_cn = tl.arange(0, 4)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]

        for k in range(0, 2):
            a = tl.load(a_ptrs)
            tl.store(c_ptrs, a)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an

    M = 4
    N = 8
    A = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(
        (M, N))
    out = torch.full((M, N), 88888, device=device, dtype=torch.float32)
    grid = lambda meta: (1, )

    wrap_stacked[grid](A,
                       out,
                       M,
                       N,
                       A.stride(0),
                       A.stride(1),
                       out.stride(0),
                       out.stride(1),
                       BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor(
        [[16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31],
         [0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        device=device)

    assert torch.equal(expected_out.int(), out.int())


def test_1d(device):

    @triton.jit
    def mod_1d(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn,
               BLOCK_SIZE_K: tl.constexpr):
        row = 7
        offs_an = (6 + tl.arange(0, 4)) % N
        a_ptrs = a_ptr + (row * stride_am) + offs_an[None, :] * stride_an

        offs_cn = tl.arange(0, 4)
        c_ptrs = c_ptr + stride_cn * offs_cn[None, :]

        a = tl.load(a_ptrs)
        tl.store(c_ptrs, a)

    M = 8
    N = 8
    A = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(
        (M, N))
    out = torch.full((M, N), 88888, device=device, dtype=torch.float32)
    grid = lambda meta: (1, )

    mod_1d[grid](A,
                 out,
                 M,
                 N,
                 A.stride(0),
                 A.stride(1),
                 out.stride(0),
                 out.stride(1),
                 BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor(
        [[62, 63, 56, 57, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888]],
        device=device)

    assert torch.equal(expected_out.int(), out.int())


def test_2d(device):

    @triton.jit
    def mod_2d(a_ptr, c_ptr, M, N, stride_am, stride_an, stride_cm, stride_cn,
               BLOCK_SIZE_K: tl.constexpr):
        offs_am = 2 + tl.arange(0, 4)
        offs_an = (6 + tl.arange(0, 4)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_an[None, :] * stride_an)

        offs_cm = tl.arange(0, 4)
        offs_cn = tl.arange(0, 4)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]

        a = tl.load(a_ptrs)
        tl.store(c_ptrs, a)

    M = 8
    N = 8
    A = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(
        (M, N))
    out = torch.full((M, N), 88888, device=device, dtype=torch.float32)
    grid = lambda meta: (1, )

    mod_2d[grid](A,
                 out,
                 M,
                 N,
                 A.stride(0),
                 A.stride(1),
                 out.stride(0),
                 out.stride(1),
                 BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor(
        [[22, 23, 16, 17, 88888, 88888, 88888, 88888],
         [30, 31, 24, 25, 88888, 88888, 88888, 88888],
         [38, 39, 32, 33, 88888, 88888, 88888, 88888],
         [46, 47, 40, 41, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888]],
        device=device)

    assert torch.equal(expected_out.int(), out.int())


def test_side_by_side_masked_loop(device):

    @triton.jit
    def wrap_side_by_side_masked_loop(a_ptr, c_ptr, M, N, stride_am, stride_an,
                                      stride_cm, stride_cn,
                                      BLOCK_SIZE_K: tl.constexpr):
        offs_am = 2 + tl.arange(0, BLOCK_SIZE_K)
        offs_an = (6 + tl.arange(0, BLOCK_SIZE_K)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_an[None, :] * stride_an)

        offs_k = tl.arange(0, BLOCK_SIZE_K)

        offs_cm = tl.arange(0, BLOCK_SIZE_K)
        offs_cn = tl.arange(0, BLOCK_SIZE_K)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]

        for k in range(0, 2):
            a = tl.load(a_ptrs, mask=offs_k[:, None] < 2, other=-99)
            tl.store(c_ptrs, a)
            a_ptrs += BLOCK_SIZE_K * stride_am
            c_ptrs += BLOCK_SIZE_K * stride_an

    M = 12
    N = 8
    A = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(
        (M, N))
    out = torch.full((M, N), 88888, device=device, dtype=torch.float32)
    print(out)
    grid = lambda meta: (1, )

    wrap_side_by_side_masked_loop[grid](A,
                                        out,
                                        M,
                                        N,
                                        A.stride(0),
                                        A.stride(1),
                                        out.stride(0),
                                        out.stride(1),
                                        BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor(
        [[22, 23, 16, 17, 54, 55, 48, 49], [30, 31, 24, 25, 62, 63, 56, 57],
         [-99, -99, -99, -99, -99, -99, -99, -99],
         [-99, -99, -99, -99, -99, -99, -99, -99],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888],
         [88888, 88888, 88888, 88888, 88888, 88888, 88888, 88888]],
        dtype=torch.int32)

    assert torch.equal(expected_out.int(), out.int())


def test_stacked_masked_loop(device):

    @triton.jit
    def wrap_stacked_masked_loop(a_ptr, c_ptr, M, N, stride_am, stride_an,
                                 stride_cm, stride_cn,
                                 BLOCK_SIZE_K: tl.constexpr):
        offs_am = (2 + tl.arange(0, BLOCK_SIZE_K)) % M
        offs_an = 3 + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                          offs_an[None, :] * stride_an)

        offs_cm = tl.arange(0, BLOCK_SIZE_K)
        offs_cn = tl.arange(0, BLOCK_SIZE_K)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[
            None, :]

        offs_k = tl.arange(0, BLOCK_SIZE_K)

        for k in range(0, 2):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < 3, other=-99)
            tl.store(c_ptrs, a)
            a_ptrs += BLOCK_SIZE_K * stride_an
            c_ptrs += BLOCK_SIZE_K * stride_an

    M = 4
    N = 12
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 4
    A = torch.arange(0, M * N, device=device, dtype=torch.float32).reshape(
        (M, N))
    out = torch.full((BLOCK_SIZE_M, N),
                     88888,
                     device=device,
                     dtype=torch.float32)
    print(out)
    grid = lambda meta: (1, )

    wrap_stacked_masked_loop[grid](A,
                                   out,
                                   M,
                                   N,
                                   A.stride(0),
                                   A.stride(1),
                                   out.stride(0),
                                   out.stride(1),
                                   BLOCK_SIZE_K=4)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor([
        [
            27.0,
            28.0,
            29.0,
            -99.0,
            31.0,
            32.0,
            33.0,
            -99.0,
            88888,
            88888,
            88888,
            88888,
        ],
        [
            39.0,
            40.0,
            41.0,
            -99.0,
            43.0,
            44.0,
            45.0,
            -99.0,
            88888,
            88888,
            88888,
            88888,
        ],
        [
            3.0,
            4.0,
            5.0,
            -99.0,
            7.0,
            8.0,
            9.0,
            -99.0,
            88888,
            88888,
            88888,
            88888,
        ],
        [
            15.0,
            16.0,
            17.0,
            -99.0,
            19.0,
            20.0,
            21.0,
            -99.0,
            88888,
            88888,
            88888,
            88888,
        ],
    ], )

    assert torch.equal(expected_out.int(), out.int())


def test_torch_inductor_pattern():

    @triton.jit
    def triton_(in_ptr2, out_ptr2, rnumel, XBLOCK: tl.constexpr,
                RBLOCK: tl.constexpr):
        xnumel = 128
        rnumel = 32
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        rbase = tl.arange(0, RBLOCK)[None, :]
        x0 = xindex % 7
        x0 = xindex
        roffset = 0
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr2 + (r2 + (xnumel * x0)), rmask, other=77)
        tl.store(
            out_ptr2 + (XBLOCK * tl.arange(0, RBLOCK)[None, :] +
                        tl.arange(0, XBLOCK)[:, None]), tmp3)

    device = "cpu"
    xnumel = 128
    rnumel = 32

    XBLOCK = 4
    RBLOCK = 64
    A = torch.arange(0, xnumel * rnumel, device=device,
                     dtype=torch.int32).reshape((xnumel, rnumel))
    out = torch.full((XBLOCK, RBLOCK), 88888, device=device, dtype=torch.int32)
    grid = lambda meta: (1, )

    triton_[grid](A, out, rnumel, XBLOCK=XBLOCK, RBLOCK=RBLOCK)

    # Expected output copied from running triton on NVDIA gpu
    expected_out = torch.tensor(
        [[
            0, 128, 256, 384, 1, 129, 257, 385, 2, 130, 258, 386, 3, 131, 259,
            387, 4, 132, 260, 388, 5, 133, 261, 389, 6, 134, 262, 390, 7, 135,
            263, 391, 8, 136, 264, 392, 9, 137, 265, 393, 10, 138, 266, 394,
            11, 139, 267, 395, 12, 140, 268, 396, 13, 141, 269, 397, 14, 142,
            270, 398, 15, 143, 271, 399
        ],
         [
             16, 144, 272, 400, 17, 145, 273, 401, 18, 146, 274, 402, 19, 147,
             275, 403, 20, 148, 276, 404, 21, 149, 277, 405, 22, 150, 278, 406,
             23, 151, 279, 407, 24, 152, 280, 408, 25, 153, 281, 409, 26, 154,
             282, 410, 27, 155, 283, 411, 28, 156, 284, 412, 29, 157, 285, 413,
             30, 158, 286, 414, 31, 159, 287, 415
         ],
         [
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77
         ],
         [
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
             77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77
         ]],
        device=device,
        dtype=torch.int32)

    assert torch.equal(expected_out.int(), out.int())
