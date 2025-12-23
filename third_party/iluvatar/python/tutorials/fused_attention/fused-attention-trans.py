"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl

version = tuple(map(int, triton.__version__.split('.')[:2]))

if version >= (2, 0) and version < (3, 0):
    from triton.language.math import fast_dividef
elif version >= (3, 0):
    from triton.language.extra.cuda.libdevice import fast_dividef
else:
    raise ValueError(f"不支持的 Triton 版本: {triton.__version__}")


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    H,
    KV_H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_SN: tl.constexpr,
    BLOCK_OK: tl.constexpr,
    BLOCK_ON: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    sm_scale *= 1.44269504  # 1/log(2)
    start_m = tl.program_id(0)
    start_z = tl.program_id(1)
    start_h = tl.program_id(2)

    # initialize offsets
    offs_zh = ((start_z * H) + start_h)
    if GROUP_SIZE != 1:
        start_h_kv = start_h // GROUP_SIZE
    else:
        start_h_kv = start_h
    kv_offs_zh = ((start_z * KV_H) + start_h_kv)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_SK)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_SN)
    off_q = offs_zh * stride_qh + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    off_k = kv_offs_zh * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    off_v = kv_offs_zh * stride_vh + offs_n[None, :] * stride_vk + offs_d[:, None] * stride_vn

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)

    # loop over k, v and update accumulator
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_SN):
        qk = tl.zeros([BLOCK_SN, BLOCK_M], dtype=tl.float32)
        q_ptrs_loop = q_ptrs
        k_ptrs_loop = k_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            q = tl.load(q_ptrs_loop)
            k = tl.load(k_ptrs_loop)
            qk += tl.dot(k, q)
            q_ptrs_loop += BLOCK_SK * stride_qk
            k_ptrs_loop += BLOCK_SK * stride_kk
        # compute scaling constant
        qk *= sm_scale
        if IS_CAUSAL:
            if start_n >= start_m * BLOCK_M:
                qk = tl.where(offs_m[None, :] >= (start_n + offs_n[:, None]), qk, float("-inf"))

        v = tl.load(v_ptrs)
        m_curr = tl.maximum(tl.max(qk, 0), m_prev)
        alpha = tl.math.exp2(m_prev - m_curr)
        p = tl.math.exp2(qk - m_curr[None, :])
        # update l_prev
        l_prev = l_prev * alpha + tl.sum(p, 0)

        acc *= alpha[None, :]
        acc += tl.dot(v, p.to(Q.dtype.element_ty))
        # update m_prev
        m_prev = m_curr

        # update pointers
        k_ptrs += BLOCK_SN * stride_kn
        v_ptrs += BLOCK_SN * stride_vk

    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l
    acc = fast_dividef(acc, l_prev[None, :])
    l_ptrs = L + offs_zh * N_CTX + offs_m
    tl.store(l_ptrs, m_prev + tl.math.log2(l_prev))
    # write back O
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = offs_zh * stride_oh + offs_m[None, :] * stride_om + offs_n[:, None] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = ((tl.program_id(1) * H) + tl.program_id(2))
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(Out + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    N_CTX,
    BLOCK_QM: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_KN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    if GROUP_SIZE != 1:
        off_h_kv = off_h // GROUP_SIZE
    else:
        off_h_kv = off_h

    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h_kv * stride_kh
    V += off_z * stride_vz + off_h_kv * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    L += off_z * stride_deltaz + off_h * stride_deltah
    D += off_z * stride_deltaz + off_h * stride_deltah

    # initialize offsets
    start_n = tl.program_id(0)
    begin_m = 0 if not IS_CAUSAL else (start_n * BLOCK_KN)
    offs_qm = begin_m + tl.arange(0, BLOCK_QM)
    offs_m = tl.arange(0, BLOCK_QM)
    offs_sk = tl.arange(0, BLOCK_SK)
    offs_kn = start_n * BLOCK_KN + tl.arange(0, BLOCK_KN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to Q, K, V
    off_q = offs_qm[:, None] * stride_qm + offs_sk[None, :] * stride_qk
    off_qt = offs_qm[None, :] * stride_qm + offs_d[:, None] * stride_qk
    off_kt = offs_kn[None, :] * stride_kn + offs_sk[:, None] * stride_kk
    off_vt = offs_kn[None, :] * stride_vn + offs_sk[:, None] * stride_vk
    off_dot = offs_qm[None, :] * stride_qm + offs_d[:, None] * stride_qk

    q_ptrs = Q + off_q
    qt_ptrs = Q + off_qt
    kt_ptrs = K + off_kt
    vt_ptrs = V + off_vt
    do_ptrs = DO + off_q
    dot_ptrs = DO + off_dot

    # initialize dvT and dkT
    dvt = tl.zeros([BLOCK_DMODEL, BLOCK_KN], dtype=tl.float32)
    dkt = tl.zeros([BLOCK_DMODEL, BLOCK_KN], dtype=tl.float32)

    qk_scale = sm_scale * 1.4426950408889634  # = 1.0 / ln(2)

    for start_m in range(begin_m, N_CTX, BLOCK_QM):
        offs_qm_curr = start_m + offs_m

        l_i = tl.load(L + offs_qm_curr)
        # step1: S = Q * (K^T)
        qkt = tl.zeros([BLOCK_QM, BLOCK_KN], dtype=tl.float32)
        q_ptrs_loop = q_ptrs
        kt_ptrs_loop = kt_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            q = tl.load(q_ptrs_loop)
            kt = tl.load(kt_ptrs_loop)
            qkt += tl.dot(q, kt)
            q_ptrs_loop += BLOCK_SK * stride_qk
            kt_ptrs_loop += BLOCK_SK * stride_kk

        Di = tl.load(D + offs_qm_curr)
        # step2: dW = dO * (V^T)
        dp = tl.zeros([BLOCK_QM, BLOCK_KN], dtype=tl.float32)
        do_ptrs_loop = do_ptrs
        vt_ptrs_loop = vt_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            do = tl.load(do_ptrs_loop)
            vt = tl.load(vt_ptrs_loop)
            dp += tl.dot(do, vt)
            do_ptrs_loop += BLOCK_SK * stride_qk
            vt_ptrs_loop += BLOCK_SK * stride_vk

        dot = tl.load(dot_ptrs)
        # step3: softmax(S)
        qkt *= qk_scale
        if IS_CAUSAL:
            if start_m <= start_n * BLOCK_KN:
                qkt = tl.where(offs_qm_curr[:, None] >= (offs_kn[None, :]), qkt, float("-inf"))
        p = tl.math.exp2(qkt - l_i[:, None])

        qt = tl.load(qt_ptrs)
        # step4: dW = softmaxBwd(dW)
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)

        # step5: dV^T = dO^T * S
        dvt += tl.dot(dot, p.to(Q.dtype.element_ty))

        # step6: dK^T = Q^T * dW
        dkt += tl.dot(qt, ds)

        q_ptrs += BLOCK_QM * stride_qm
        qt_ptrs += BLOCK_QM * stride_qm
        do_ptrs += BLOCK_QM * stride_qm
        dot_ptrs += BLOCK_QM * stride_qm

    # step7: store dK
    off_dkt = offs_kn[None, :] * stride_kn + offs_d[:, None] * stride_kk
    dkt_ptrs = DK + off_dkt

    # step8: store dV
    off_dvt = offs_kn[None, :] * stride_vn + offs_d[:, None] * stride_vk
    dvt_ptrs = DV + off_dvt

    tl.store(dkt_ptrs, dkt)
    tl.store(dvt_ptrs, dvt)


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    L,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    N_CTX,
    BLOCK_QM: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_KN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    if GROUP_SIZE != 1:
        off_h_kv = off_h // GROUP_SIZE
    else:
        off_h_kv = off_h
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h_kv * stride_kh
    V += off_z * stride_vz + off_h_kv * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    L += off_z * stride_deltaz + off_h * stride_deltah
    D += off_z * stride_deltaz + off_h * stride_deltah

    # initialize offsets
    start_m = tl.program_id(0)
    offs_qm = start_m * BLOCK_QM + tl.arange(0, BLOCK_QM)
    offs_sk = tl.arange(0, BLOCK_SK)
    offs_kn = tl.arange(0, BLOCK_KN)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to Q, K, V
    off_qt = offs_qm[None, :] * stride_qm + offs_sk[:, None] * stride_qk
    off_k = offs_kn[:, None] * stride_kn + offs_sk[None, :] * stride_kk
    off_kt = offs_kn[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = offs_kn[:, None] * stride_vn + offs_sk[None, :] * stride_vk

    qt_ptrs = Q + off_qt
    k_ptrs = K + off_k
    dot_ptrs = DO + off_qt
    v_ptrs = V + off_v
    kt_ptrs = K + off_kt

    qk_scale = sm_scale * 1.4426950408889634  # = 1.0 / ln(2)

    l_i = tl.load(L + offs_qm)
    Di = tl.load(D + offs_qm)

    dqt = tl.zeros([BLOCK_DMODEL, BLOCK_QM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_QM if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_KN):
        # step1 : S = K @ Q^T
        qkt = tl.zeros([BLOCK_KN, BLOCK_QM], dtype=tl.float32)
        qt_ptrs_loop = qt_ptrs
        k_ptrs_loop = k_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            k = tl.load(k_ptrs_loop)
            qt = tl.load(qt_ptrs_loop)
            qkt += tl.dot(k, qt)
            qt_ptrs_loop += BLOCK_SK * stride_qk
            k_ptrs_loop += BLOCK_SK * stride_kk

        # step2: dP^T = V @ dO^T
        dp = tl.zeros([BLOCK_KN, BLOCK_QM], dtype=tl.float32)
        v_ptrs_loop = v_ptrs
        dot_ptrs_loop = dot_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            v = tl.load(v_ptrs_loop)
            dot = tl.load(dot_ptrs_loop)
            dp += tl.dot(v, dot)
            v_ptrs_loop += BLOCK_SK * stride_vk
            dot_ptrs_loop += BLOCK_SK * stride_qk

        kt = tl.load(kt_ptrs)
        qkt *= qk_scale
        if IS_CAUSAL:
            if start_n >= start_m * BLOCK_QM:
                qkt = tl.where(offs_qm[None, :] >= (start_n + offs_kn[:, None]), qkt, float("-inf"))

        # step3: softmax(S)
        p = tl.math.exp2(qkt - l_i[None, :])

        # step4: dW = softmaxBwd(dW)
        ds = (p * (dp - Di[None, :]) * sm_scale).to(Q.dtype.element_ty)

        # step5: dQ^T = K^T @ dS^T
        dqt += tl.dot(kt, ds)

        k_ptrs += BLOCK_KN * stride_kn
        v_ptrs += BLOCK_KN * stride_vn
        kt_ptrs += BLOCK_KN * stride_kn

    # step6: Store dQ
    off_dqt = offs_qm[None, :] * stride_qm + offs_d[:, None] * stride_qk
    dqt_ptrs = DQ + off_dqt
    tl.store(dqt_ptrs, dqt)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, is_causal, sm_scale):
        BLOCK_M = 256
        BLOCK_SK = 32
        BLOCK_SN = 128
        BLOCK_OK = 128
        BLOCK_ON = 128

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)

        N, H, N_CTX, HEAD_DIM = q.shape
        KV_H, KV_N_CTX = k.shape[1], k.shape[2]
        group_size = H // KV_H

        grid = (triton.cdiv(N_CTX, BLOCK_M), N, H)
        L = torch.empty((N, H, N_CTX), device=q.device, dtype=torch.float32)
        num_warps = 16

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            L,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            H,
            KV_H,
            KV_N_CTX,
            BLOCK_M=BLOCK_M,
            BLOCK_SK=BLOCK_SK,
            BLOCK_SN=BLOCK_SN,
            BLOCK_OK=BLOCK_OK,
            BLOCK_ON=BLOCK_ON,
            BLOCK_DMODEL=Lk,
            IS_CAUSAL=is_causal,
            GROUP_SIZE=group_size,
            num_warps=num_warps,
            num_stages=2,
            maxnreg=128,
        )

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.is_causal = is_causal
        ctx.group_size = group_size
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        Z, H, N_CTX = q.shape[:3]
        KV_H = k.shape[1]

        do = do.contiguous()
        delta = torch.empty_like(L)
        if H != KV_H:  # gqa
            dk = torch.zeros_like(q)
            dv = torch.zeros_like(q)
        else:  # mha
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
        dq = torch.empty_like(q)

        # rowsum kernel
        BLOCK_M, BLOCK_N = 128, 128
        num_warps = 8
        _bwd_preprocess[(triton.cdiv(N_CTX, BLOCK_M), Z, H)](o, do, delta, Z, H, N_CTX, BLOCK_M=BLOCK_M,
                                                             HEAD_DIM=ctx.BLOCK_DMODEL, num_warps=num_warps)

        # dkdv kernel
        if ctx.BLOCK_DMODEL == 128:
            BLOCK_QM, BLOCK_SK, BLOCK_KN = 128, 64, 128
        elif ctx.BLOCK_DMODEL == 64:
            BLOCK_QM, BLOCK_SK, BLOCK_KN = 128, 32, 128
        num_warps = 8
        grid_dkdv = (triton.cdiv(N_CTX, BLOCK_KN), Z, H)
        _bwd_dkdv_kernel[grid_dkdv](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dk,
            dv,
            L,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            delta.stride(0),
            delta.stride(1),
            N_CTX,
            BLOCK_QM=BLOCK_QM,
            BLOCK_SK=BLOCK_SK,
            BLOCK_KN=BLOCK_KN,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            IS_CAUSAL=ctx.is_causal,
            GROUP_SIZE=ctx.group_size,
            num_warps=num_warps,
            num_stages=2,
        )

        if H != KV_H:
            dk = dk.reshape(Z, KV_H, ctx.group_size, N_CTX, ctx.BLOCK_DMODEL)
            dk = torch.sum(dk, dim=2)
            dv = dv.reshape(Z, KV_H, ctx.group_size, N_CTX, ctx.BLOCK_DMODEL)
            dv = torch.sum(dv, dim=2)

        # dq kernel
        BLOCK_QM = 256
        BLOCK_SK = 32
        num_warps = 16
        grid_dq = (triton.cdiv(N_CTX, BLOCK_QM), Z, H)
        _bwd_dq_kernel[grid_dq](q, k, v, ctx.sm_scale, do, dq, L, delta, q.stride(0), q.stride(1), q.stride(2),
                                q.stride(3), k.stride(0), k.stride(1), k.stride(2),
                                k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), delta.stride(0),
                                delta.stride(1), N_CTX, BLOCK_QM=BLOCK_QM, BLOCK_SK=BLOCK_SK, BLOCK_KN=BLOCK_KN,
                                BLOCK_DMODEL=ctx.BLOCK_DMODEL, IS_CAUSAL=ctx.is_causal, GROUP_SIZE=ctx.group_size,
                                num_warps=num_warps, num_stages=2, maxnreg=128)

        return dq, dk, dv, None, None


attention = _attention.apply

DTYPES = [torch.float16, torch.bfloat16]
BATCHS = [1]
N_HEADS = [32, 64]
KV_HEADS = [8, 32]
N_CTXS = [1024 * 2**i for i in range(0, 4)]
HEAD_DIMS = [64, 128]
CAUSAL_MASK = [True, False]
# TEST_BACKWARDS = [False]
TEST_BACKWARDS = [True]


@pytest.mark.parametrize("test_backward", TEST_BACKWARDS)
@pytest.mark.parametrize("is_causal", CAUSAL_MASK)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("HEAD_DIM", HEAD_DIMS)
@pytest.mark.parametrize("N_CTX", N_CTXS)
@pytest.mark.parametrize("KV_H", KV_HEADS)
@pytest.mark.parametrize("H", N_HEADS)
@pytest.mark.parametrize("Z", BATCHS)
def test_attention(Z, H, KV_H, N_CTX, HEAD_DIM, dtype, is_causal, test_backward):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, KV_H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()

    sm_scale = 0.2
    dout = torch.randn_like(q)

    ref_out = torch.nn.functional.scaled_dot_product_attention(q.clone(), k.clone(), v.clone(), is_causal=is_causal,
                                                               scale=sm_scale)

    if test_backward:
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    tri_out = attention(q, k, v, is_causal, sm_scale)
    if test_backward:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None

    print("= " * 30)
    print("CHECK RESULTS: TORCH vs TRITON")
    print("o max diff = ", (ref_out - tri_out).to(torch.float32).abs().max())
    if test_backward:
        print("dv max diff = ", (ref_dv - tri_dv).to(torch.float32).abs().max())
        print("dk max diff = ", (ref_dk - tri_dk).to(torch.float32).abs().max())
        print("dq max diff = ", (ref_dq - tri_dq).to(torch.float32).abs().max())
    print("= " * 30)

    atol = 1e-2
    rtol = 1e-2
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=rtol, equal_nan=True)
    if test_backward:
        torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol, equal_nan=True)


BATCH, H, KV_H, N_CTX, D_HEAD, IS_CAUSAL = 1, 64, 64, 4096, 128, True
configs = [
    triton.testing.Benchmark(
        x_names=['BATCH', 'H', 'KV_H', 'N_CTX', 'D_HEAD', 'IS_CAUSAL',
                 'mode'],  # Argument names to use as an x-axis for the plot
        x_vals=[(1, 64, 64, N_CTX, 128, True, mode)
                for mode in ['fwd', 'bwd']
                for N_CTX in [1024 * 2**i for i in range(0, 6)]],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch-sdpa'],  # Possible values for `line_arg`
        line_names=['Triton(tflops)', 'Torch-sdpa(tflops)'],  # Label name for the lines
        styles=[('red', '-'), ('blue', '-')],  # Line styles
        xlabel="N_CTX",
        ylabel='tflops',
        plot_name=f'fused-attention-bf16',
        args={},
    )
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, KV_H, N_CTX, D_HEAD, IS_CAUSAL, mode, provider, dtype=torch.bfloat16,
                          device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 1000
    rep = 1000
    sm_scale = 1.3
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: attention(q, k, v, IS_CAUSAL, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if provider == 'torch-sdpa':
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=IS_CAUSAL, scale=sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # return ms
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if IS_CAUSAL:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    Z, H, KV_H, N_CTX, HEAD_DIM, dtype, is_causal, test_backward = 1, 32, 32, 4096, 128, torch.float16, True, True  # mha
    # Z, H, KV_H, N_CTX, HEAD_DIM, dtype, is_causal, test_backward = 1, 32, 8, 4096, 128, torch.float16, True, True # gqa
    test_attention(Z, H, KV_H, N_CTX, HEAD_DIM, dtype, is_causal, test_backward)
    # bench_flash_attention.run(save_path='.', print_data=True)
