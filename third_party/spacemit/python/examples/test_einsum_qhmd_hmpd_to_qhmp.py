import torch

import triton
import triton.language as tl


# This implements einsum(qhmd,hmpd->qhmp).
@triton.jit
def einsum_qhmd_hmpd_to_qhmp_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    Q,
    H,
    M,
    P,
    D,
    strideAq,
    strideAh,
    strideAm,
    strideAd,
    strideBh,
    strideBm,
    strideBp,
    strideBd,
    strideCq,
    strideCh,
    strideCm,
    strideCp,
    BLOCK_QHM: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """
    Triton kernel computing:
       C[q,h,m,p] = sum_{d=0..D-1} A[q,h,m,d] * B[h,m,p,d].

    We tile over Q*H*M in one dimension (BLOCK_QHM), and P in another (BLOCK_P).
    Then unroll a for-loop over d in [0..D-1], with a mask to guard against invalid indices loads.
    """

    pid_qhm = tl.program_id(axis=0)  # which block for Q*H*M
    pid_p = tl.program_id(axis=1)  # which block for P

    # Indices for qhm, p
    qhm_off = pid_qhm * BLOCK_QHM
    qhm_idx = qhm_off + tl.arange(0, BLOCK_QHM)  # [BLOCK_QHM]

    p_off = pid_p * BLOCK_P
    p_idx = p_off + tl.arange(0, BLOCK_P)  # [BLOCK_P]

    # Valid ranges
    valid_qhm = qhm_idx < (Q * H * M)
    valid_p = p_idx < P

    # Expand to 2D for final store
    qhm_mask_2d = tl.broadcast_to(valid_qhm[:, None], [BLOCK_QHM, BLOCK_P])
    p_mask_2d = tl.broadcast_to(valid_p[None, :], [BLOCK_QHM, BLOCK_P])
    full_mask = qhm_mask_2d & p_mask_2d

    # Decompose qhm = q * (H * M) + h * M + m -> (q, h, m)
    q_ = qhm_idx // (H * M)
    hm_ = qhm_idx % (H * M)
    h_ = hm_ // M
    m_ = hm_ % M

    # Accumulator
    c_acc = tl.zeros((BLOCK_QHM, BLOCK_P), dtype=tl.float32)

    for d_idx in range(D):
        # offset in A => q_*strideAq + h_*strideAh + m_*strideAm + d_idx*strideAd
        baseA = (q_ * strideAq) + (h_ * strideAh) + (m_ * strideAm)
        A_offset = baseA + d_idx * strideAd
        # shape => [BLOCK_QHM]
        # Expand to 2D for masked load
        A_mask = valid_qhm  # shape [BLOCK_QHM]
        A_vals = tl.load(A_ptr + A_offset, mask=A_mask, other=0.0)

        # offset in B => h_*strideBh + m_*strideBm + d_idx*strideBd + p_idx*strideBp
        baseB = (h_ * strideBh) + (m_ * strideBm) + (d_idx * strideBd)
        B_offset = baseB[:, None] + (p_idx[None, :] * strideBp)
        B_mask_2d = qhm_mask_2d & p_mask_2d
        B_vals = tl.load(B_ptr + B_offset, mask=B_mask_2d, other=0.0)

        # multiply-accumulate
        # Expand A_vals to [BLOCK_QHM, 1]
        # A_vals shape [BLOCK_QHM], B_vals shape [BLOCK_QHM,BLOCK_P]
        a_col_2d = A_vals[:, None]
        c_acc += a_col_2d * B_vals

    # Store result
    baseC = (q_ * strideCq) + (h_ * strideCh) + (m_ * strideCm)
    c_offset = baseC[:, None] + (p_idx[None, :] * strideCp)
    tl.store(C_ptr + c_offset, c_acc, mask=full_mask)


def einsum_qhmd_hmpd_to_qhmp(A, B, BLOCK_QHM=2, BLOCK_P=2):
    """
    A: [Q,H,M,D], B: [H,M,P,D]
    => C: [Q,H,M,P] with sum_{d=0..D-1} A[q,h,m,d]*B[h,m,p,d].
    """

    ## Assertions to make sure we got shapes compatible with the fixed einsum implementation
    Q, H, M, D = A.shape
    assert B.shape[0] == H, f"B's H={B.shape[0]} != {H}"
    assert B.shape[1] == M, f"B's M={B.shape[1]} != {M}"
    P = B.shape[2]
    assert B.shape[3] == D, f"B's D={B.shape[3]} != {D}"

    C = torch.empty((Q, H, M, P), device=A.device, dtype=A.dtype)

    grid = (triton.cdiv(Q * H * M, BLOCK_QHM), triton.cdiv(P, BLOCK_P))

    einsum_qhmd_hmpd_to_qhmp_kernel[grid](
        A,
        B,
        C,
        Q,
        H,
        M,
        P,
        D,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        A.stride(3),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        BLOCK_QHM,
        BLOCK_P,
    )

    return C


def test(device):
    Q, H, M, D = (1, 2, 2, 2)
    P = 4

    A = torch.randn((Q, H, M, D), dtype=torch.float32, device=device)
    B = torch.randn((H, M, P, D), dtype=torch.float32, device=device)

    # Reference
    C_ref = torch.einsum("qhmd,hmpd->qhmp", A, B)

    C_triton = einsum_qhmd_hmpd_to_qhmp(A, B)
    torch.testing.assert_close(C_triton, C_ref)
