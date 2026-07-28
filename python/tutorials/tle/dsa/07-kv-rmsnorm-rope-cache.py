# Copyright 2026- Xcoresigma Technology Co., Ltd

import logging

import torch
import torch_npu
import triton
import triton.language as tl
import triton.experimental.tle as tle

from triton.backends.ascend.testing import do_bench_npu

logger = logging.getLogger(__name__)

# set device ID
torch.manual_seed(0)
DEVICE = "npu"
DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

# set mode type
CACHE_MODE_NORM = tl.constexpr(0)
CACHE_MODE_PA = tl.constexpr(1)
CACHE_MODE_PA_BNSD = tl.constexpr(1)
CACHE_MODE_PA_NZ = tl.constexpr(2)
CACHE_MODE_PA_BLK_BNSD = tl.constexpr(3)
CACHE_MODE_PA_BLK_NZ = tl.constexpr(4)
cache_mode_map = {
    "Norm": 0,
    "PA_BNSD": 1,
    "PA": 1,
    "PA_NZ": 2,
    "PA_BLK_BNSD": 3,
    "PA_BLK_NZ": 4,
}


# @triton.autotune(configs=[
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 8}),
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 16}),
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 32}),
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 64}),
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 128}),
#     triton.Config(kwargs={'BLOCK_SIZE_TOKEN': 256}),
#     ],
#   key=['token_num','cache_mode_const'] # the two above configs will be evaluated anytime
# )
@triton.jit
def apply_rotary_pos_emb_kernel(
    q_embed_ptr,
    k_cache_ptr,
    index,  # [batch_size*seq_len]
    q_ptr,  # (batch,1, squence_len, head_dim)
    cos_ptr,  # (batch,1, squence_len, head_dim)
    sin_ptr,  # (batch,1, squence_len, head_dim)
    q_last_dim,
    SQUNECE_LEN,
    HEAD_DIM: tl.constexpr,
    k_cache_stride_0,
    k_cache_stride_1,
    k_cache_stride_2,
    k_cache_stride_3,
    HALF_PADDED_HEAD_DIM: tl.constexpr,
    MAX_TOKEN_NUM: tl.constexpr,
    CACHE_MODE: tl.constexpr,
    SQ_PER_BLOCK,
    BLOCK_SIZE: tl.constexpr,
    token_num,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    IS_ALIGNED: tl.constexpr,
):
    s_id_batch = tl.program_id(0)

    s_id = s_id_batch * BLOCK_SIZE_TOKEN + tl.arange(0, BLOCK_SIZE_TOKEN)

    mask_batch = s_id[:, None] < token_num
    cos_or_sin_offset = s_id[:, None] * HEAD_DIM
    cos_ptr += cos_or_sin_offset
    sin_ptr += cos_or_sin_offset

    offsets_all = tl.arange(0, HEAD_DIM)[None, :]

    x_offset = s_id[:, None] * q_last_dim + 512

    # 加载x1和x2
    x = tl.load(q_ptr + x_offset + offsets_all, mask=mask_batch, other=0.0)
    x1 = tle.dsa.extract_slice(x, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 2))
    x2 = tle.dsa.extract_slice(x, (0, 1), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 2))

    # cos/sin layout: [BLOCK_SIZE_TOKEN, HEAD_DIM], stride [HEAD_DIM, 1]
    cos_tile = tl.load(cos_ptr + offsets_all, mask=mask_batch, other=0.0).to(tl.float32)
    sin_tile = tl.load(sin_ptr + offsets_all, mask=mask_batch, other=0.0).to(tl.float32)

    # cos_1/sin_1: 前半部分 [0:half_dim]，顺序访问
    cos_1 = tle.dsa.extract_slice(cos_tile, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1))
    sin_1 = tle.dsa.extract_slice(sin_tile, (0, 0), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1))

    # cos_2/sin_2: 后半部分 [half_dim:HEAD_DIM]，顺序访问
    cos_2 = tle.dsa.extract_slice(cos_tile, (0, HALF_PADDED_HEAD_DIM), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1))
    sin_2 = tle.dsa.extract_slice(sin_tile, (0, HALF_PADDED_HEAD_DIM), (BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM), (1, 1))

    # 计算前半部分输出: x1*cos_1 - x2*sin_1
    first_half = (x1 * cos_1 - x2 * sin_1).to(q_embed_ptr.dtype.element_ty)

    # 计算后半部分输出: x2*cos_2 + x1*sin_2
    second_half = (x2 * cos_2 + x1 * sin_2).to(q_embed_ptr.dtype.element_ty)

    result = tl.zeros((BLOCK_SIZE_TOKEN, HEAD_DIM), dtype=q_embed_ptr.dtype.element_ty)
    result = tle.dsa.insert_slice(
        result,
        first_half,
        offsets=(0, 0),
        sizes=(BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        strides=(1, 1),
    )
    result = tle.dsa.insert_slice(
        result,
        second_half,
        offsets=(0, HALF_PADDED_HEAD_DIM),
        sizes=(BLOCK_SIZE_TOKEN, HALF_PADDED_HEAD_DIM),
        strides=(1, 1),
    )

    # 写入结果
    tl.store(q_embed_ptr + cos_or_sin_offset + offsets_all, result)

    index_value = tl.load(index + s_id)

    row_ids = tl.arange(0, HEAD_DIM)
    if CACHE_MODE == CACHE_MODE_PA:
        for i in tle.dsa.parallel(BLOCK_SIZE_TOKEN):
            offset_i = s_id_batch * BLOCK_SIZE_TOKEN + i
            if IS_ALIGNED:
                reload_result = tle.dsa.extract_slice(result, (i, 0), (1, HEAD_DIM), (1, 1))
                reload_result = tl.reshape(reload_result, (HEAD_DIM))

                k_cache_offset = tle.dsa.extract_element(index_value, (i, )) * HEAD_DIM

                tl.store(k_cache_ptr + k_cache_offset + row_ids, reload_result)
            else:
                if offset_i < token_num:
                    reload_result = tle.dsa.extract_slice(result, (i, 0), (1, HEAD_DIM), (1, 1))
                    reload_result = tl.reshape(reload_result, (HEAD_DIM))

                    k_cache_offset = tle.dsa.extract_element(index_value, (i, )) * HEAD_DIM

                    tl.store(k_cache_ptr + k_cache_offset + row_ids, reload_result)


def apply_rotary_pos_emb(q, cos, sin, cache_mode, index, k_cache):
    """
    Apply rotary position embedding to kv

    Args:
        q: [batch_size, 1, seq_len,  rope_size]
        cos:[batch_size, 1, seq_len, rope_size]
        sin:[batch_size, 1, seq_len, rope_size]
        mode: PA/PA_BNSD
        q_embedding: [batch_size, 1, seq_len,  rope_size]
        kv_cache: (*, head_dim)  diff from mode

    Returns:
        q_embedding:[batch_size, 1, seq_len,  rope_size]
        kv_cache: (*, k_heads, head_dim)
    """
    logger.debug("GEMS ROTARY_POS_EMBEDDING")

    assert (cos.shape[-1] == sin.shape[-1]
            ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"

    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    batch, _, squence_len, head_dim = cos.shape

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    q_embed = torch.empty_like(cos)

    n_tokens = batch * squence_len

    # grid = lambda META: (
    #     triton.cdiv(n_tokens, META["BLOCK_SIZE_TOKEN"]),
    # )

    BLOCK_SIZE_TOKEN = 64
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE_TOKEN), )

    IS_ALIGNED = (n_tokens % BLOCK_SIZE_TOKEN == 0)

    block_num, block_size, _, _ = k_cache.shape
    max_tokens = block_num * block_size
    sq_per_block = (squence_len + block_size - 1) // block_size
    cache_mode_const = cache_mode_map.get(cache_mode, None)

    apply_rotary_pos_emb_kernel[grid](
        q_embed,
        k_cache,
        index,
        q,
        cos,
        sin,
        q.shape[-1],
        squence_len,
        head_dim,
        *k_cache.stride(),
        padded_head_dim // 2,
        max_tokens,
        cache_mode_const,
        sq_per_block,
        block_size,
        n_tokens,
        BLOCK_SIZE_TOKEN,
        IS_ALIGNED,
    )
    return q_embed


# @triton.autotune(configs=[
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':4, 'BLOCK_SIZE_TOKEN_PER': 8,'BLOCK_SIZE_TOKEN': 32,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':4, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 64,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':8, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 128,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':16, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 256,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':32, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 512,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':64, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 1024,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':128, 'BLOCK_SIZE_TOKEN_PER': 16,'BLOCK_SIZE_TOKEN': 2048,}),


#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':4, 'BLOCK_SIZE_TOKEN_PER': 32,'BLOCK_SIZE_TOKEN': 128,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':8, 'BLOCK_SIZE_TOKEN_PER': 32,'BLOCK_SIZE_TOKEN': 256,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':16, 'BLOCK_SIZE_TOKEN_PER': 32,'BLOCK_SIZE_TOKEN': 512,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':32, 'BLOCK_SIZE_TOKEN_PER': 32,'BLOCK_SIZE_TOKEN': 1024,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':2, 'BLOCK_SIZE_TOKEN_PER': 64,'BLOCK_SIZE_TOKEN': 128,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':4, 'BLOCK_SIZE_TOKEN_PER': 64,'BLOCK_SIZE_TOKEN': 256,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':8, 'BLOCK_SIZE_TOKEN_PER': 64,'BLOCK_SIZE_TOKEN': 512,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':16, 'BLOCK_SIZE_TOKEN_PER': 64,'BLOCK_SIZE_TOKEN': 1024,}),
#     triton.Config(kwargs={'BLOCK_LOOP_SIZE':32, 'BLOCK_SIZE_TOKEN_PER': 64,'BLOCK_SIZE_TOKEN': 2048,}),
#   ],
#   key=['token_num','CACHE_MODE'] # the two above configs will be evaluated anytime
#                  # the value of x_size changes
# )
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    out_ptr,  # pointer to the output
    in_ptr,  # pointer to the input
    w_ptr,  # pointer to the weights
    index_ptr,
    kv_cache_ptr,
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    kv_cache_stride_0,
    kv_cache_stride_1,
    kv_cache_stride_2,
    kv_cache_stride_3,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    SQUNECE_LEN,
    BLOCK_SIZE: tl.constexpr,
    SQ_PER_BLOCK,
    MAX_TOKEN_NUM,
    CACHE_MODE: tl.constexpr,
    token_num,
    BLOCK_SIZE_TOKEN: tl.constexpr,
    BLOCK_SIZE_TOKEN_PER: tl.constexpr,
    BLOCK_LOOP_SIZE: tl.constexpr,
    IS_ALIGNED: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(in_ptr.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)

    pid_offset_ori = pid * BLOCK_SIZE_TOKEN + tl.arange(0, BLOCK_SIZE_TOKEN_PER)
    for j in range(BLOCK_LOOP_SIZE):
        pid_offset_1 = pid_offset_ori + j * BLOCK_SIZE_TOKEN_PER
        pid_offset = pid_offset_1[:, None]
        out_ptr_1 = out_ptr + pid_offset * y_stride_r
        # in_ptr = pid_offset * x_stride_r

        mask = pid_offset < token_num
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + pid_offset * x_stride_r + cols * x_stride_c, mask, other=0.0).to(cdtype)

        var = tl.sum(x * x, axis=1) * (1 / N)
        rrms = tl.sqrt(var + eps)

        w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE)[None, :])
        y = (x / rrms[:, None] * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr_1 + cols * y_stride_c, y)

        index_value = tl.load(index_ptr + pid_offset_1)

        if CACHE_MODE == CACHE_MODE_PA:
            for i in tle.dsa.parallel(BLOCK_SIZE_TOKEN_PER):
                offset_i = pid * BLOCK_SIZE_TOKEN + j * BLOCK_SIZE_TOKEN_PER + i
                if IS_ALIGNED:
                    # Get global token position in KV cache
                    value_reload = tle.dsa.extract_slice(y, (i, 0), (1, BLOCK_SIZE), (1, 1))
                    tl.compile_hint(value_reload, "disable_bubble_up")
                    value_reload = tl.reshape(value_reload, (BLOCK_SIZE))

                    offset_kv_cache = tle.dsa.extract_element(index_value, (i, ))

                    k_cache_offset = offset_kv_cache * N + cols
                    tl.store(kv_cache_ptr + k_cache_offset, value_reload)
                else:
                    if offset_i < token_num:
                        value_reload = tle.dsa.extract_slice(y, (i, 0), (1, BLOCK_SIZE), (1, 1))
                        tl.compile_hint(value_reload, "disable_bubble_up")
                        value_reload = tl.reshape(value_reload, (BLOCK_SIZE))

                        offset_kv_cache = tle.dsa.extract_element(index_value, (i, ))

                        k_cache_offset = offset_kv_cache * N + cols
                        tl.store(kv_cache_ptr + k_cache_offset, value_reload)


def rms_norm(x, weight, index, ckv_cache, cache_mode, eps=1e-5):
    N = 512
    BLOCK_SIZE = N

    batch, _, squence_len, x_last_dim = x.shape
    y = torch.zeros([batch, 1, squence_len, N], dtype=x.dtype).to(x.device)

    block_num, block_size, _, _ = ckv_cache.shape
    max_tokens = block_num * block_size
    cache_mode_const = cache_mode_map.get(cache_mode, None)

    sq_per_block = (squence_len + block_size - 1) // block_size

    token_num = batch * squence_len
    # grid = lambda META: (
    #          triton.cdiv(token_num, META["BLOCK_SIZE_TOKEN_PER"]* META["BLOCK_LOOP_SIZE"]),
    # )
    BLOCK_SIZE_TOKEN = 128
    BLOCK_SIZE_TOKEN_PER = 16
    BLOCK_LOOP_SIZE = 8

    IS_ALIGNED = (token_num % BLOCK_SIZE_TOKEN == 0)

    grid = (triton.cdiv(token_num, BLOCK_SIZE_TOKEN), )

    rms_norm_kernel[grid](
        y,
        x,
        weight,
        index,
        ckv_cache,
        N,
        1,
        x_last_dim,
        1,
        *ckv_cache.stride(),
        N,
        eps,
        squence_len,
        BLOCK_SIZE,
        sq_per_block,
        max_tokens,
        cache_mode_const,
        token_num,
        BLOCK_SIZE_TOKEN,
        BLOCK_SIZE_TOKEN_PER,
        BLOCK_LOOP_SIZE,
        IS_ALIGNED,
    )

    return y


def kv_rmsnorm_rope_cache(
    kv,
    gamma,
    cos,
    sin,
    index,
    k_cache,
    ckv_cache,
    k_rope_scale,
    c_kv_scale,
    k_rope_offset,
    c_kv_offset,
    epsilon,
    cache_mode,
    is_output_kv,
):
    """
    kv: Tensor shape (batch_size, 1, seq_len, rope_size + rms_size)  rope_size=64 rms_size=512 rope when 512-576 rms when 0-512
    gamma:Tensor shape (rms_size) weight of rmsnorm
    cos: Tensor shape (batch_size, 1, seq_len, rope_size)
    sin: Tensor shape (batch_size, 1, seq_len, rope_size)
    index: index for k_cache and ckv_cache, shape depend on cache_mode  total_length batch_size*seq_len(most likely)
    k_cache: Tensor shepe depend on cache_mode, usually (page_num, page_size, 1, rope_size)
    ckv_cache: Tensor shepe depend on cache_mode, usually (page_num, page_size, 1, rope_size)
    k_rope_scale: Tensor shape [rope_size] or None,used for rope scaling
    c_kv_scale: Tensor shape [rms_size] or None,used for rms scaling
    k_rope_offset: Tensor  [rope_size], None default, used for rope offset
    c_kv_offset: Tensor [rms_size], None default, used for rms offset
    epsilon: default 1e-5, float, for rmsnorm
    cache_mode:
    is_output_kv: whether output kv after rmsnorm and rope embedding
    """
    rope_embedding_value = apply_rotary_pos_emb(kv, cos, sin, cache_mode, index, k_cache)

    rms_norm_out = rms_norm(kv, gamma, index, ckv_cache, cache_mode, epsilon)
    if is_output_kv:
        return k_cache, ckv_cache, rope_embedding_value, rms_norm_out
    else:
        return k_cache, ckv_cache


def test_op(
    batch_size,
    seq_len,
    rms_size=512,
    rope_size=64,
    cache_length=10,
    cache_mode="PA_BNSD",
    page_num=493,
    page_size=128,
    epsilon=1e-5,
    is_output_kv=True,
    dtype=torch.float16,
):
    assert rope_size == 64, "only support rope_size 64 for test now"
    assert rms_size == 512, "only support rms_size 512 for test now"
    assert cache_mode in ["PA_BNSD"], "only support PA_BNSD for test now"

    # 初始化张量
    kv = torch.randn(batch_size, 1, seq_len, rms_size + rope_size, dtype=dtype).npu()
    gamma = torch.ones(rms_size, dtype=dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, rope_size, dtype=dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, rope_size, dtype=dtype).npu()
    k_cache = torch.zeros(page_num, page_size, 1, 64, dtype=dtype).npu()
    ckv_cache = torch.zeros(page_num, page_size, 1, 512, dtype=dtype).npu()

    k_cache1 = torch.zeros(page_num, page_size, 1, 64, dtype=dtype).npu()
    ckv_cache1 = torch.zeros(page_num, page_size, 1, 512, dtype=dtype).npu()

    index_shape = (batch_size * seq_len, )
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()

    # 参数部分不支持
    k_rope_scale = None
    c_kv_scale = None
    k_rope_offset = None
    c_kv_offset = None

    k_cache, ckv_cache, k_rope, c_kv = kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache,
        ckv_cache,
        k_rope_scale=k_rope_scale,
        c_kv_scale=c_kv_scale,
        k_rope_offset=k_rope_offset,
        c_kv_offset=c_kv_offset,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
    )

    k_cache_ref, v_cache_ref, k_rope_ref, c_kv_ref = (torch_npu.npu_kv_rmsnorm_rope_cache(
        kv,
        gamma,
        cos,
        sin,
        index,
        k_cache1,
        ckv_cache1,
        k_rope_scale=k_rope_scale,
        c_kv_scale=c_kv_scale,
        k_rope_offset=k_rope_offset,
        c_kv_offset=c_kv_offset,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
    ))

    triton_time = do_bench_npu(
        lambda: kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache,
            ckv_cache,
            k_rope_scale=k_rope_scale,
            c_kv_scale=c_kv_scale,
            k_rope_offset=k_rope_offset,
            c_kv_offset=c_kv_offset,
            epsilon=epsilon,
            cache_mode=cache_mode,
            is_output_kv=is_output_kv,
        ), clear_l2_cache=True, collect_prof=False)
    print(f"[Triton kv_rmsnorm_rope_cache] Time: {triton_time:.4f} us")

    npu_time = do_bench_npu(
        lambda: torch_npu.npu_kv_rmsnorm_rope_cache(
            kv,
            gamma,
            cos,
            sin,
            index,
            k_cache1,
            ckv_cache1,
            k_rope_scale=k_rope_scale,
            c_kv_scale=c_kv_scale,
            k_rope_offset=k_rope_offset,
            c_kv_offset=c_kv_offset,
            epsilon=epsilon,
            cache_mode=cache_mode,
            is_output_kv=is_output_kv,
        ), clear_l2_cache=True, collect_prof=False)
    print(f"[Torch-NPU kv_rmsnorm_rope_cache] Time: {npu_time:.4f} us")

    eps = 1e-2
    print("==== check k_rope start =====")
    torch.testing.assert_close(k_rope_ref, k_rope, atol=eps, rtol=1e-3)
    print("k_rope check [Passed]")

    print("==== check k_cache start =====")
    torch.testing.assert_close(k_cache_ref, k_cache, atol=eps, rtol=1e-3)
    print("k_cache check [Passed]")

    print("==== check v_cache start =====")
    torch.testing.assert_close(v_cache_ref, ckv_cache, atol=eps, rtol=1e-3)
    print("v_cache check [Passed]")

    print("==== check c_kv start =====")
    torch.testing.assert_close(c_kv_ref, c_kv, atol=eps, rtol=1e-3)
    print("c_kv check [Passed]")
    print("Accuracy compliance check [All Passed]")


if __name__ == "__main__":
    print("====kv_rmsnorm_rope_cache test start ====")
    test_op(
        batch_size=16384,
        seq_len=1,
        rms_size=512,
        rope_size=64,
        cache_length=10,
        cache_mode="PA_BNSD",
        page_num=493,
        page_size=128,
        epsilon=1e-5,
        is_output_kv=True,
        dtype=torch.float16,
    )
