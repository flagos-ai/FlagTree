"""concat_and_cache_mla: concatenate each token's compressed KV and rope parts into one
cache row.

Both source tensors are read through PartitionViews. The cache row is written through two
GatherScatterViews over the same cache: one addresses the leading kv_c columns, the other the
trailing k_pe columns, and the sparse dimension of each takes the slot mapping. The views
enable bounds_check, so unmapped tokens (slot -1) are dropped by the view.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_TOKENS = 4096
KV_LORA_RANK = 512
ROPE_DIM = 64
BLOCK_SIZE = 16
BLOCK_T = 16


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def concat_and_cache_mla_kernel(kv_c, k_pe, kv_cache, slot_mapping, num_tokens, num_slots, kv_c_stride, k_pe_stride,
                                RANK: tl.constexpr, ROPE: tl.constexpr, BLOCK_T: tl.constexpr):
    token_block = tl.program_id(0)
    tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    slots = tl.load(slot_mapping + tokens, mask=tokens < num_tokens, other=-1).to(tl.int32)

    kv_c_view = tle.make_partition_view(kv_c, shape=[num_tokens, RANK], strides=[kv_c_stride, 1], tile=[BLOCK_T, RANK])
    k_pe_view = tle.make_partition_view(k_pe, shape=[num_tokens, ROPE], strides=[k_pe_stride, 1], tile=[BLOCK_T, ROPE])
    kv_c_tile = tle.tile_load(kv_c_view, index=(token_block, 0), dst_space=tle.language.dsa.ascend.UB)
    k_pe_tile = tle.tile_load(k_pe_view, index=(token_block, 0), dst_space=tle.language.dsa.ascend.UB)

    # The cache row is RANK + ROPE wide. A tile of RANK columns at dense index 0 covers the
    # kv_c half; a tile of ROPE columns at dense index RANK // ROPE covers the rope half.
    kv_c_cache = tle.make_gather_scatter_view(kv_cache, shape=[num_slots, RANK + ROPE], strides=[RANK + ROPE, 1],
                                              tile=[BLOCK_T, RANK], sparse_dim=[0], bounds_check=True)
    k_pe_cache = tle.make_gather_scatter_view(kv_cache, shape=[num_slots, RANK + ROPE], strides=[RANK + ROPE, 1],
                                              tile=[BLOCK_T, ROPE], sparse_dim=[0], bounds_check=True)
    tle.tile_store(kv_c_cache, value=kv_c_tile, index=(slots, 0), src_space=tle.language.dsa.ascend.UB)
    tle.tile_store(k_pe_cache, value=k_pe_tile, index=(slots, RANK // ROPE), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, block_t=BLOCK_T):
    """kv_c is [num_tokens, kv_lora_rank], k_pe is [num_tokens, rope_dim], and kv_cache is
    [num_blocks, block_size, kv_lora_rank + rope_dim]. The rope width must divide the rank so
    the rope half starts on a tile boundary of the narrower tile."""
    num_tokens = kv_c.shape[0]
    rank = kv_c.shape[1]
    rope = k_pe.shape[1]
    num_slots = kv_cache.shape[0] * kv_cache.shape[1]
    assert rank % rope == 0, "kv_lora_rank must be a multiple of rope_dim"
    grid = (triton.cdiv(num_tokens, block_t), )
    concat_and_cache_mla_kernel[grid](kv_c, k_pe, kv_cache, slot_mapping, num_tokens, num_slots, kv_c.stride(0),
                                      k_pe.stride(0), RANK=rank, ROPE=rope, BLOCK_T=block_t)
    return kv_cache


def run_case(num_tokens, rank, rope, block_size, device, sentinel=-7.0):
    num_blocks = triton.cdiv(num_tokens, block_size) + 2
    num_slots = num_blocks * block_size
    kv_c = torch.randn((num_tokens, rank), dtype=torch.float16, device=device)
    k_pe = torch.randn((num_tokens, rope), dtype=torch.float16, device=device)
    kv_cache = torch.full((num_blocks, block_size, rank + rope), sentinel, dtype=torch.float16, device=device)

    slot_mapping = torch.randperm(num_slots, device=device)[:num_tokens].to(torch.int64)
    slot_mapping[::5] = -1

    concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping)

    live = slot_mapping >= 0
    expected = torch.full((num_slots, rank + rope), sentinel, dtype=torch.float16, device=device)
    expected[slot_mapping[live]] = torch.cat((kv_c[live], k_pe[live]), dim=1)

    torch.testing.assert_close(kv_cache.reshape(num_slots, rank + rope), expected)
    print(f"  tokens={num_tokens} rank={rank} rope={rope}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="concat_and_cache_mla through TensorView")
    parser.add_argument("--tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--rank", type=int, default=KV_LORA_RANK)
    parser.add_argument("--rope", type=int, default=ROPE_DIM)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.tokens, args.rank, args.rope, args.block_size, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block.
    run_case(args.tokens - 3, args.rank, args.rope, args.block_size, device)
    run_case(512, 256, 32, args.block_size, device)
    print("Test Passed!")
