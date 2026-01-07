import sys
import pytest
import triton
import torch
import triton.language as tl

sys.path.append("..")
import test_common

#source: python\sglang\srt\layers\moe\router.py


@triton.jit
def fused_moe_router_large_bs_kernel(
    a_ptr,  # input (bs, hidden_dim)
    b_ptr,  # input (num_experts, hidden_dim)
    topk_weights_ptr,  # output (bs, topk)
    topk_ids_ptr,  # output (bs, topk)
    bs,
    acc_ptr,
    num_experts: tl.constexpr,
    topk: tl.constexpr,  # only support topk == 1
    moe_softcapping: tl.constexpr,
    moe_renormalize: tl.constexpr,  # not supported
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_acc: tl.constexpr
):

    # 1. get block id
    pid = tl.program_id(axis=0)

    # 2. create pointers for the first block of A and B
    # 2.1. setup a_ptrs with offsets in m and k
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]
    bs_mask = offs_m < bs
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    a_ptrs = a_ptr + (offs_m * stride_am + offs_k)

    # 2.2. setup b_ptrs with offsets in k and n.
    #      Note: b matrix is k-major.
    offs_k = tl.arange(0, BLOCK_SIZE_K)[None, :]
    offs_n = tl.arange(0, BLOCK_SIZE_N)[:, None]
    expert_mask = offs_n < num_experts
    b_ptrs = b_ptr + (offs_n * stride_bn + offs_k)

    # 3. Create an accumulator of float32 of size [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #    3.1. iterate in K dimension
    #    3.2. transpose tile B
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K // BLOCK_SIZE_K):  # hidden_dim % BLOCK_SIZE_K == 0
        a = tl.load(
            a_ptrs,
            mask=bs_mask,
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=expert_mask, other=0.0).to(tl.float32).T
        acc += tl.dot(a, b)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # 4. logit softcap
    logits_scaled = acc / moe_softcapping
    exped = tl.exp(2 * logits_scaled)
    logits_softcapped = (exped - 1) / (exped + 1) * moe_softcapping

    # 5. top1
    cond = tl.arange(0, BLOCK_SIZE_N)[None, :] < num_experts
    top1 = tl.argmax(tl.where(cond, logits_softcapped, float("-inf")), axis=1)
    top1_v = tl.max(
        tl.where(cond, logits_softcapped, float("-inf")), axis=1, keep_dims=True
    )
    invsumexp = 1.0 / tl.sum(
        tl.where(cond, tl.exp(logits_softcapped - top1_v), 0.0), axis=1
    )

    # 6. store to output
    offs_topk = pid * topk * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    topk_mask = offs_topk < bs
    tl.store(topk_ids_ptr + offs_topk, top1, mask=topk_mask)
    tl.store(
        topk_weights_ptr + offs_topk,
        invsumexp,
        mask=topk_mask,
    )

    # Debug output: Intermediate matrix storage (added for debugging, not original)
    offs_nn = tl.arange(0, BLOCK_SIZE_N)[None, : ]
    offset_acc = offs_m * stride_acc + offs_nn
    out_ptrs = acc_ptr + offset_acc
    tl.store(
        out_ptrs, logits_softcapped
    )


def test_fused_moe_router_large_bs_kernel(ptfile_path):
    try:
        data = torch.load(ptfile_path, map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        pytest.fail(f"load file {ptfile_path} failed: {str(e)}")

    # ptfile format:
    # [input_data] (dict):
    #     key : value
    # [gpu_output] (dict):
    #     key : value
    # [grid] :
    #     (1,)
    input_data = test_common.convert_tensor_with_device_type(data["input_data"], device_type='npu')

    fused_moe_router_large_bs_kernel[data["grid"]](**input_data)

    # compare the results of GPU and NPU.
    try:
        test_common.compare_data_precision(data["gpu_output"], input_data, device_type='cpu')
    except ValueError as e:
        pytest.fail(f"The testcase failed")