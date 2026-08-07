import pytest
import torch

import triton
import triton.language as tl
from torch.testing import assert_close

torch.manual_seed(0)


@pytest.mark.parametrize('M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType',
                         [(M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType)
                          for M in [32, 64, 128]
                          for N in [32, 64]
                          for K in [32, 64]
                          for AT in [False, True]
                          for BT in [False, True]
                          for ACol in [False, True]
                          for BCol in [False, True]
                          for num_warps in [1, 2, 4]
                          for disable_sme in ["0", "1"]
                          for dataType in ["float16", "bfloat16", "float32", "int8"]])
def test_sme_and_swizzle_layout_trans(M, N, K, AT, BT, ACol, BCol, num_warps, disable_sme, dataType, monkeypatch,
                                      device='cuda'):

    @triton.jit
    def kernel(
        A,
        B,
        C,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_T: tl.constexpr,
        B_T: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_mk = tl.arange(0, BLOCK_K)
        if A_T:
            off_m = tl.arange(0, BLOCK_K)
            off_mk = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_nk = tl.arange(0, BLOCK_K)
        if B_T:
            off_n = tl.arange(0, BLOCK_K)
            off_nk = tl.arange(0, BLOCK_N)
        off_cm = tl.arange(0, BLOCK_M)
        off_cn = tl.arange(0, BLOCK_N)
        a = A + off_m[:, None] * stride_am + off_mk[None, :] * stride_ak
        b = B + off_nk[:, None] * stride_bk + off_n[None, :] * stride_bn
        C = C + off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn
        x = tl.load(a)
        y = tl.load(b)
        if A_T:
            x = tl.trans(x)
        if B_T:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(C, z)

    monkeypatch.setenv('TRITON_DISABLE_SME', disable_sme)  # disable_sme=1 tests swizzle transpose
    #run test
    dataType = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "int8":
                torch.int8}[dataType]
    if dataType is torch.int8:
        a = torch.randint(-8, 8, (K, M) if (AT ^ ACol) else (M, K), device='cuda', dtype=dataType)
        b = torch.randint(-8, 8, (N, K) if (BT ^ BCol) else (K, N), device='cuda', dtype=dataType)
        tt_c = torch.empty((M, N), device='cuda', dtype=torch.int32)
    else:
        a = .1 * torch.randn((K, M) if (AT ^ ACol) else (M, K), device='cuda', dtype=dataType)
        b = .1 * torch.randn((N, K) if (BT ^ BCol) else (K, N), device='cuda', dtype=dataType)
        tt_c = .1 * torch.randn((M, N), device='cuda', dtype=dataType)
    tt_a = a
    tt_b = b

    if ACol:
        tt_a = a.t()
    if BCol:
        tt_b = b.t()
    # triton result
    kernel[(1, 1)](tt_a, tt_b, tt_c, tt_a.stride(0), tt_a.stride(1), tt_b.stride(0), tt_b.stride(1), tt_c.stride(0),
                   tt_c.stride(1), BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, A_T=AT, B_T=BT, num_warps=num_warps)

    th_a = a.t() if (AT ^ ACol) else a
    th_b = b.t() if (BT ^ BCol) else b
    #torch result
    if dataType is torch.int8:
        th_c = torch.matmul(th_a.to(torch.float32), th_b.to(torch.float32))
        assert_close(tt_c.to(torch.float32), th_c, atol=0, rtol=0)
    else:
        th_c = torch.matmul(th_a, th_b)
        assert_close(tt_c, th_c, atol=1e-2, rtol=0)


@pytest.mark.parametrize('AT, BT, ACol, BCol, disable_sme, dataType', [(AT, BT, ACol, BCol, disable_sme, dataType)
                                                                       for AT in [False, True]
                                                                       for BT in [False, True]
                                                                       for ACol in [False, True]
                                                                       for BCol in [False, True]
                                                                       for disable_sme in ["0", "1"]
                                                                       for dataType in ["float16", "int8"]])
def test_sme_layout_trans_pipeline(AT, BT, ACol, BCol, disable_sme, dataType, monkeypatch, device='cuda'):

    @triton.jit
    def kernel(A, B, C, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_K_TOTAL: tl.constexpr, A_T: tl.constexpr,
               B_T: tl.constexpr, IS_INT8: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32 if IS_INT8 else tl.float32)
        for k_base in range(0, BLOCK_K_TOTAL, BLOCK_K):
            off_k = k_base + tl.arange(0, BLOCK_K)
            a_m = off_k if A_T else off_m
            a_k = off_m if A_T else off_k
            b_k = off_k if not B_T else off_n
            b_n = off_n if not B_T else off_k
            a = A + a_m[:, None] * stride_am + a_k[None, :] * stride_ak
            b = B + b_k[:, None] * stride_bk + b_n[None, :] * stride_bn
            x = tl.load(a)
            y = tl.load(b)
            if A_T:
                x = tl.trans(x)
            if B_T:
                y = tl.trans(y)
            acc += tl.dot(x, y)
        C = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
        tl.store(C, acc)

    monkeypatch.setenv('TRITON_DISABLE_SME', disable_sme)
    M, N, K, BK = 64, 64, 128, 64
    dataType = {"float16": torch.float16, "int8": torch.int8}[dataType]
    if dataType is torch.int8:
        a = torch.randint(-8, 8, (K, M) if (AT ^ ACol) else (M, K), device=device, dtype=dataType)
        b = torch.randint(-8, 8, (N, K) if (BT ^ BCol) else (K, N), device=device, dtype=dataType)
        tt_c = torch.empty((M, N), device=device, dtype=torch.int32)
    else:
        a = .1 * torch.randn((K, M) if (AT ^ ACol) else (M, K), device=device, dtype=dataType)
        b = .1 * torch.randn((N, K) if (BT ^ BCol) else (K, N), device=device, dtype=dataType)
        tt_c = torch.empty((M, N), device=device, dtype=dataType)
    tt_a = a.t() if ACol else a
    tt_b = b.t() if BCol else b

    kernel[(1, 1)](tt_a, tt_b, tt_c, tt_a.stride(0), tt_a.stride(1), tt_b.stride(0), tt_b.stride(1), tt_c.stride(0),
                   tt_c.stride(1), BLOCK_M=M, BLOCK_N=N, BLOCK_K=BK, BLOCK_K_TOTAL=K, A_T=AT, B_T=BT,
                   IS_INT8=(dataType is torch.int8), num_warps=4, num_stages=2)

    th_a = a.t() if (AT ^ ACol) else a
    th_b = b.t() if (BT ^ BCol) else b
    if dataType is torch.int8:
        th_c = torch.matmul(th_a.to(torch.float32), th_b.to(torch.float32))
        assert_close(tt_c.to(torch.float32), th_c, atol=0, rtol=0)
    else:
        th_c = torch.matmul(th_a, th_b)
        assert_close(tt_c, th_c, atol=1e-2, rtol=0)


def test_pipeline_mixed_direct_and_trans_dot_not_physicalized(monkeypatch, device='cuda'):

    @triton.jit
    def kernel(A, B, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               BLOCK_K_TOTAL: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k_base in range(0, BLOCK_K_TOTAL, BLOCK_K):
            off_k = k_base + tl.arange(0, BLOCK_K)
            x = tl.load(A + off_m[:, None] * BLOCK_K_TOTAL + off_k[None, :])
            y = tl.load(B + off_k[:, None] * BLOCK_N + off_n[None, :])
            acc += tl.dot(x, y)
            acc += tl.dot(x, tl.trans(y))
        tl.store(C + off_m[:, None] * BLOCK_N + off_n[None, :], acc)

    monkeypatch.setenv('TRITON_DISABLE_SME', "0")
    M, N, K_TOTAL, BK = 64, 64, 128, 64
    a = torch.randint(-8, 8, (M, K_TOTAL), device=device, dtype=torch.int8)
    b = torch.randint(-8, 8, (K_TOTAL, N), device=device, dtype=torch.int8)
    c = torch.empty((M, N), device=device, dtype=torch.int32)

    handler = kernel[(1, 1)](a, b, c, BLOCK_M=M, BLOCK_N=N, BLOCK_K=BK, BLOCK_K_TOTAL=K_TOTAL, num_warps=4,
                             num_stages=2)

    # This mixed direct/trans consumer pattern must not be physicalized. The
    # existing fallback lowering does not provide a reliable correctness oracle
    # for this shape, so this test verifies only the attr-marking contract, not
    # mixed fallback numerical correctness.
    assert "tt.iluvatar.physical_trans_dot" not in handler.asm["ttgir"]


@pytest.mark.parametrize('M, N, K, AT, BT, CT, num_warps, dataType', [(M, N, K, AT, BT, CT, num_warps, dataType)
                                                                      for M in [32, 64, 128]
                                                                      for N in [32, 64]
                                                                      for K in [32, 64]
                                                                      for AT in [False, True]
                                                                      for BT in [False, True]
                                                                      for CT in [False, True]
                                                                      for num_warps in [1, 2, 4]
                                                                      for dataType in ["float16", "bfloat16"]])
def test_multi_dot_trans(M, N, K, AT, BT, CT, num_warps, dataType, monkeypatch, device='cuda'):
    monkeypatch.setenv('TRITON_DISABLE_SME', "0")

    @triton.jit
    def kernel(
        A,
        B,
        C,
        D,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        A_T: tl.constexpr,
        B_T: tl.constexpr,
        C_T: tl.constexpr,
    ):
        off_m = tl.arange(0, BLOCK_M)
        off_mk = tl.arange(0, BLOCK_K)
        if A_T:
            off_m = tl.arange(0, BLOCK_K)
            off_mk = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_nk = tl.arange(0, BLOCK_K)
        if B_T:
            off_n = tl.arange(0, BLOCK_K)
            off_nk = tl.arange(0, BLOCK_N)
        off_cm = tl.arange(0, BLOCK_M)
        off_cn = tl.arange(0, BLOCK_N)
        if C_T:
            off_cm = tl.arange(0, BLOCK_N)
            off_cn = tl.arange(0, BLOCK_M)
        off_dn = tl.arange(0, BLOCK_N)
        a = A + off_m[:, None] * stride_am + off_mk[None, :] * stride_ak
        b = B + off_nk[:, None] * stride_bk + off_n[None, :] * stride_bn
        c = C + off_cm[:, None] * stride_cm + off_cn[None, :] * stride_cn
        x = tl.load(a)
        y = tl.load(b)
        w = tl.load(c)
        if A_T:
            x = tl.trans(x)
        if B_T:
            y = tl.trans(y)
        if C_T:
            w = tl.trans(w)
        z = tl.dot(x, y)
        z = z.to(C.dtype.element_ty)
        p = tl.dot(tl.trans(z), w)
        D = D + off_dn[:, None] * stride_dm + off_dn[None, :] * stride_dn
        tl.store(D, p)

    #run test
    dataType = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dataType]
    a = .1 * torch.randn((K, M) if AT else (M, K), device='cuda', dtype=dataType)
    b = .1 * torch.randn((N, K) if BT else (K, N), device='cuda', dtype=dataType)
    c = .1 * torch.randn((N, M) if CT else (M, N), device='cuda', dtype=dataType)
    d = .1 * torch.randn((N, N), device='cuda', dtype=dataType)
    # triton result
    kernel[(1, 1)](a, b, c,
                   d, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), d.stride(0),
                   d.stride(1), BLOCK_M=M, BLOCK_N=N, BLOCK_K=K, A_T=AT, B_T=BT, C_T=CT, num_warps=num_warps)
    ta = a.t() if AT else a
    tb = b.t() if BT else b
    tc = c.t() if CT else c
    #torch result
    th_c = torch.matmul(torch.matmul(ta, tb).t(), tc)
    assert_close(d, th_c, atol=1e-2, rtol=0)
