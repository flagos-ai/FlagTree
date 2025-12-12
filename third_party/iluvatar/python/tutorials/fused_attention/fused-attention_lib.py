"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

lib version:
1) Only implemented fwd, no bwd yet. (# TODO)
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
    M,
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
    off_q = offs_zh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    off_k = kv_offs_zh * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    off_v = kv_offs_zh * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_SN):

        # 1st Gemm using tiling K = BLOCK_K
        qk = tl.zeros([BLOCK_M, BLOCK_SN], dtype=tl.float32)
        q_ptrs_loop = q_ptrs
        k_ptrs_loop = k_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_SK):
            # -- compute qk ----
            q = tl.load(q_ptrs_loop)
            k = tl.load(k_ptrs_loop)
            qk += tl.dot(q, k)
            q_ptrs_loop += BLOCK_SK * stride_qk
            k_ptrs_loop += BLOCK_SK * stride_kk
        # compute scaling constant
        qk *= sm_scale
        if IS_CAUSAL:
            if start_n >= start_m * BLOCK_M:
                qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        v = tl.load(v_ptrs)
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        alpha = tl.math.exp2(m_prev - m_curr)
        p = tl.math.exp2(qk - m_curr[:, None])
        l_prev = l_prev * alpha + tl.sum(p, 1)

        # scale and update acc
        acc *= alpha[:, None]
        acc += tl.dot(p.to(Q.dtype.element_ty), v)
        # update m_prev and l_prev
        m_prev = m_curr

        # update pointers (k_ptrs is already contiguous)
        k_ptrs += BLOCK_SN * stride_kn
        v_ptrs += BLOCK_SN * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    acc = fast_dividef(acc, l_prev[:, None])
    l_ptrs = L + offs_zh * N_CTX + offs_m
    tl.store(l_ptrs, m_prev + tl.math.log2(l_prev))

    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = offs_zh * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    L,
    NewDO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
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
    stride_vk,
    stride_vn,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            # compute dq
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dq_ptrs, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        capability = torch.cuda.get_device_capability()
        if capability[0] == 7:
            BLOCK_M = 256
            BLOCK_SK = 32
            BLOCK_SN = 128
            BLOCK_OK = 128
            BLOCK_ON = 128
            num_warps = 16
        elif capability[0] == 8:
            BLOCK_M = 128
            BLOCK_SK = 64
            BLOCK_SN = 128
            BLOCK_OK = 128
            BLOCK_ON = 128
            num_warps = 8
        else:
            assert False, "Iluvatar device not supported yet!"

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0], q.shape[1])
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        N, H, N_CTX, HEAD_DIM = q.shape
        KV_H, KV_N_CTX = k.shape[1], k.shape[2]
        group_size = H // KV_H

        _fwd_kernel[grid](q, k, v, sm_scale, L, m, o, q.stride(0), q.stride(1), q.stride(2), q.stride(3), k.stride(0),
                          k.stride(1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                          o.stride(0), o.stride(1), o.stride(2), o.stride(3), H, KV_H, KV_N_CTX, BLOCK_M=BLOCK_M,
                          BLOCK_SK=BLOCK_SK, BLOCK_SN=BLOCK_SN, BLOCK_OK=BLOCK_OK, BLOCK_ON=BLOCK_ON, BLOCK_DMODEL=Lk,
                          IS_CAUSAL=causal, GROUP_SIZE=group_size, num_warps=num_warps, num_stages=2, maxnreg=128)

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        # TODO: Not optimized yet.
        if torch.version.hip is not None:
            BLOCK = 64
        else:
            BLOCK = 128
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o,
            do,
            l,
            do_scaled,
            delta,
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1], )](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            l,
            m,
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
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=8,
            num_stages=1,
        )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None


attention = _attention.apply


@pytest.mark.parametrize('Z, H, KV_H, N_CTX, D_HEAD', [(1, 4, 1, 1024, 128)])
@pytest.mark.parametrize('causal', [False, True])
def test_op(Z, H, KV_H, N_CTX, D_HEAD, dtype=torch.float16, causal=False, test_backward=True, sdpa=True):
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2).requires_grad_()
    k = torch.empty((Z, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.4, std=0.2).requires_grad_()
    v = torch.empty((Z, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2).requires_grad_()
    sm_scale = 0.2
    dout = torch.randn_like(q)
    # reference implementation
    if sdpa:
        # call torch sdpa API
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=sm_scale,
            is_causal=causal,
        )
    else:
        # naive version:
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        # p = torch.exp(p)
        ref_out = torch.matmul(p, v)
    if test_backward:
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale)
    # print(ref_out)
    # print(tri_out)
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
    torch.testing.assert_close(ref_out, tri_out, atol=atol, rtol=0, equal_nan=True)
    if test_backward:
        torch.testing.assert_close(ref_dv, tri_dv, atol=atol, rtol=0, equal_nan=True)
        torch.testing.assert_close(ref_dk, tri_dk, atol=atol, rtol=0, equal_nan=True)
        torch.testing.assert_close(ref_dq, tri_dq, atol=atol, rtol=0, equal_nan=True)


try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None
HAS_FLASH = False

# BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
BATCH, H, KV_H, N_CTX, D_HEAD = 1, 4, 1, 4096, 128
# vary seq length for fixed head and batch=4
configs = [
    triton.testing.Benchmark(
        x_names=['N_CTX'], x_vals=[2**i for i in range(10, 16)], line_arg='provider',
        line_vals=['triton', 'torch'] + (['flash'] if HAS_FLASH else []),
        line_names=['Triton', 'torch'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []), styles=[('red', '-'),
                                                                                                ('blue', '-'),
                                                                                                ('green', '-')],
        ylabel='ms', plot_name=f'fused-attention-batch{BATCH}-head{H}-kvhead{KV_H}-d{D_HEAD}-{mode}--mask{causal}',
        args={
            'H': H,
            'KV_H': KV_H,
            'BATCH': BATCH,
            'D_HEAD': D_HEAD,
            'dtype': torch.float16,
            'mode': mode,
            'causal': causal,
        }) for mode in [
            'fwd',
        ] for causal in [
            True,
        ]
]


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, KV_H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 1000
    rep = 1000
    sm_scale = 1.3
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: attention(q, k, v, causal, sm_scale)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((3, BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        qkv = qkv.permute(1, 3, 0, 2, 4)  # (BATCH, N_CTX, 3, H, D_HEAD)
        if FLASH_VER == 1:
            lengths = torch.full((BATCH, ), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1, ), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = qkv.reshape(BATCH * N_CTX, 3, H, D_HEAD)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=causal)
        elif FLASH_VER == 2:
            fn = lambda: flash_attn_func(qkv, softmax_scale=sm_scale, causal=causal)
        else:
            raise ValueError(f'unknown {FLASH_VER = }')
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == 'torch':
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, KV_H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=causal)
        if mode == 'bwd':
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == 'bwd':
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    # return total_flops / ms * 1e-9
    return ms


if __name__ == "__main__":
    Z, H, KV_H, N_CTX, D_HEAD = 1, 16, 4, 4096, 128
    test_op(Z, H, KV_H, N_CTX, D_HEAD, dtype=torch.float16, causal=False, test_backward=False)
    test_op(Z, H, KV_H, N_CTX, D_HEAD, dtype=torch.float16, causal=True, test_backward=False)
    bench_flash_attention.run(save_path='.', print_data=True)
