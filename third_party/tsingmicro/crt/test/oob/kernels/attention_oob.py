"""Fused attention kernel with OOB error injection parameters.

Uses block pointers for Q/K/V/O loads and stores.
OOB is triggered by inflating N_CTX (block pointer shape) or corrupting strides.
"""
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_oob(Q, K, V, sm_scale, M, Out,  #
                  stride_qz, stride_qh, stride_qm, stride_qk,  #
                  stride_kz, stride_kh, stride_kn, stride_kk,  #
                  stride_vz, stride_vh, stride_vk, stride_vn,  #
                  stride_oz, stride_oh, stride_om, stride_on,  #
                  Z, H, N_CTX,  #
                  inject_N_CTX_scale: tl.constexpr,  # 1=normal, >1=inflate N_CTX in block pointer shapes
                  inject_stride_scale: tl.constexpr,  # 1=normal, 0=replace all strides with 1
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # --- OOB injection: inflate N_CTX ---
    eff_N_CTX = N_CTX * inject_N_CTX_scale

    # --- OOB injection: inflate strides to cause OOB access ---
    sqm = stride_qm * inject_stride_scale
    sqk = stride_qk * inject_stride_scale
    skn = stride_kn * inject_stride_scale
    skk = stride_kk * inject_stride_scale
    svk = stride_vk * inject_stride_scale
    svn = stride_vn * inject_stride_scale
    som = stride_om * inject_stride_scale
    son = stride_on * inject_stride_scale

    # block pointers (shapes may be inflated, strides may be corrupted)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(eff_N_CTX, HEAD_DIM),
        strides=(sqm, sqk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(svk, svn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, eff_N_CTX),
        strides=(skk, skn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(eff_N_CTX, HEAD_DIM),
        strides=(som, son),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # load Q — may OOB if eff_N_CTX is inflated
    q = tl.load(Q_block_ptr)

    # full non-causal loop over K/V blocks
    for start_n in range(0, eff_N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # load K — may OOB if stride is wrong
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        # load V — may OOB if stride is wrong
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    # store O — may OOB if eff_N_CTX is inflated
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
