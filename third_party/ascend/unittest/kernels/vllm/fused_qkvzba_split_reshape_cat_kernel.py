import triton
import triton.language as tl


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T + tl.arange(0, q_end))
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T + tl.arange(q_end, k_end))
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T + tl.arange(k_end, v_end))
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (mixed_qkvz + i_bs * NUM_HEADS_QK * QKVZ_DIM_T + i_qk * QKVZ_DIM_T + tl.arange(v_end, z_end))
    blk_q_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK))
    blk_k_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK + i_qk * HEAD_QK +
                    tl.arange(0, HEAD_QK))
    blk_v_st_ptr = (mixed_qkv + i_bs * NUM_HEADS_QK * QKV_DIM_T + NUM_HEADS_QK * HEAD_QK * 2 +
                    i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK))
    blk_z_st_ptr = (z + i_bs * NUM_HEADS_V * HEAD_V + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK +
                    tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK))
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end))
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))
