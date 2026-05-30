import triton
import triton.language as tl


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(k, v, beta, w, u, A, g, cu_seqlens, chunk_indices, T, H: tl.constexpr, Hg: tl.constexpr,
                             K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                             IS_VARLEN: tl.constexpr):
    T_max = T
    i_t_o = tl.program_id(0)

    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(chunk_indices + i_t_o * 2 + 1).to(
                tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = (A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1)
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_g = g + bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)

            ptr_v = (v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1)
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = (b_v * b_beta[:, None])
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)

            ptr_u = (u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1)
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            ptr_k = (k + (bos * Hg + i_h // (H // Hg)) * K + offs_t_2d * (Hg * K) + offs_k * 1)
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = (b_k * b_beta[:, None] * b_g[:, None])
            b_w = tl.dot(b_A, b_kb)

            ptr_w = (w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k * 1)
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)
