import torch
import triton
import triton.language as tl
import pytest


@triton.jit
def rope_like_load_kernel(
    Kv_cache,
    Req_to_tokens,
    output_ptr,
    stride_kv: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):

    offs_d_kpe = tl.arange(HEAD_DIM_V, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    kv_page_number = tl.load(Req_to_tokens + offs_n // PAGE_SIZE)
    kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE

    offs_k_pe = kv_loc[None, :] * stride_kv + offs_d_kpe[:, None]

    k_pe = tl.load(Kv_cache + offs_k_pe)

    offs_out = offs_n[:, None] * ROPE_DIM + tl.arange(0, ROPE_DIM)[None, :]
    tl.store(output_ptr + offs_out, tl.trans(k_pe))


def test_bubbleup_extract_nonzero_offset():
    device = "npu"

    PAGE_SIZE = 2
    BLOCK_N = 4
    head_dim = 32
    head_dim_v = 24
    rope_dim = head_dim - head_dim_v
    num_pages = BLOCK_N // PAGE_SIZE

    req_to_tokens = torch.arange(num_pages, dtype=torch.int32, device=device)
    total_tokens = num_pages * PAGE_SIZE
    kv_cache = torch.zeros(total_tokens, head_dim, dtype=torch.float32, device=device)
    for token_id in range(total_tokens):
        kv_cache[token_id, :head_dim_v] = (torch.arange(head_dim_v, dtype=torch.float32) + token_id * 100)
        kv_cache[token_id, head_dim_v:] = (torch.arange(head_dim_v, head_dim, dtype=torch.float32) + token_id * 1000)
    output = torch.zeros(BLOCK_N, rope_dim, dtype=torch.float32, device=device)

    rope_like_load_kernel[(1, )](
        kv_cache.flatten(),
        req_to_tokens,
        output.flatten(),
        stride_kv=head_dim,
        HEAD_DIM_V=head_dim_v,
        HEAD_DIM=head_dim,
        PAGE_SIZE=PAGE_SIZE,
        BLOCK_N=BLOCK_N,
        ROPE_DIM=rope_dim,
    )

    expected = torch.zeros(BLOCK_N, rope_dim, dtype=torch.float32, device=device)
    for token_id in range(BLOCK_N):
        expected[token_id] = (torch.arange(head_dim_v, head_dim, dtype=torch.float32) + token_id * 1000)

    buggy = torch.zeros(BLOCK_N, rope_dim, dtype=torch.float32, device=device)
    for token_id in range(BLOCK_N):
        buggy[token_id] = (torch.arange(rope_dim, dtype=torch.float32) + token_id * 100)

    assert torch.allclose(output, expected, atol=1e-5)
