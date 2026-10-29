"""reshape_and_cache_flash: write each token's key/value row into its paged KV cache slot.

The token tile is read through a PartitionView. The cache rows are written through a
GatherScatterView whose sparse dimension takes the slot mapping. The view enables
bounds_check, so a token whose slot is negative, or whose index falls outside the batch, is
dropped by the view instead of being masked off by the kernel.
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
NUM_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 16
BLOCK_T = 32


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def reshape_and_cache_flash_kernel(key, value, key_cache, value_cache, slot_mapping, num_tokens, num_slots, key_stride,
                                   value_stride, ROW: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_ROW: tl.constexpr):
    token_block = tl.program_id(0)
    tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    slots = tl.load(slot_mapping + tokens, mask=tokens < num_tokens, other=-1).to(tl.int32)

    key_view = tle.make_partition_view(key, shape=[num_tokens, ROW], strides=[key_stride, 1], tile=[BLOCK_T, BLOCK_ROW])
    value_view = tle.make_partition_view(value, shape=[num_tokens, ROW], strides=[value_stride, 1],
                                         tile=[BLOCK_T, BLOCK_ROW])
    key_tile = tle.tile_load(key_view, index=(token_block, 0), dst_space=tle.language.dsa.ascend.UB)
    value_tile = tle.tile_load(value_view, index=(token_block, 0), dst_space=tle.language.dsa.ascend.UB)

    key_cache_view = tle.make_gather_scatter_view(key_cache, shape=[num_slots, ROW], strides=[ROW, 1],
                                                  tile=[BLOCK_T, BLOCK_ROW], sparse_dim=[0], bounds_check=True)
    value_cache_view = tle.make_gather_scatter_view(value_cache, shape=[num_slots, ROW], strides=[ROW, 1],
                                                    tile=[BLOCK_T, BLOCK_ROW], sparse_dim=[0], bounds_check=True)
    tle.tile_store(key_cache_view, value=key_tile, index=(slots, 0), src_space=tle.language.dsa.ascend.UB)
    tle.tile_store(value_cache_view, value=value_tile, index=(slots, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, block_t=BLOCK_T):
    """key/value are [num_tokens, num_heads, head_size]; the caches are
    [num_blocks, block_size, num_heads, head_size]. Both cache layouts flatten to one row
    per slot, which is the row the scatter view addresses."""
    num_tokens = key.shape[0]
    row = key.shape[1] * key.shape[2]
    num_slots = key_cache.shape[0] * key_cache.shape[1]
    grid = (triton.cdiv(num_tokens, block_t), )
    reshape_and_cache_flash_kernel[grid](key, value, key_cache, value_cache, slot_mapping, num_tokens, num_slots,
                                         key.stride(0), value.stride(0), ROW=row, BLOCK_T=block_t,
                                         BLOCK_ROW=triton.next_power_of_2(row))
    return key_cache, value_cache


def run_case(num_tokens, num_heads, head_size, block_size, device, sentinel=-7.0):
    row = num_heads * head_size
    num_blocks = triton.cdiv(num_tokens, block_size) + 2
    num_slots = num_blocks * block_size
    key = torch.randn((num_tokens, num_heads, head_size), dtype=torch.float16, device=device)
    value = torch.randn((num_tokens, num_heads, head_size), dtype=torch.float16, device=device)
    key_cache = torch.full((num_blocks, block_size, num_heads, head_size), sentinel, dtype=torch.float16, device=device)
    value_cache = torch.full((num_blocks, block_size, num_heads, head_size), sentinel, dtype=torch.float16,
                             device=device)

    # Every third token is unmapped, so the dropped-row path runs in the same launch as
    # the written rows.
    slot_mapping = torch.randperm(num_slots, device=device)[:num_tokens].to(torch.int64)
    slot_mapping[::3] = -1

    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)

    live = slot_mapping >= 0
    expected_key = torch.full((num_slots, row), sentinel, dtype=torch.float16, device=device)
    expected_value = torch.full((num_slots, row), sentinel, dtype=torch.float16, device=device)
    expected_key[slot_mapping[live]] = key.reshape(-1, row)[live]
    expected_value[slot_mapping[live]] = value.reshape(-1, row)[live]

    torch.testing.assert_close(key_cache.reshape(num_slots, row), expected_key)
    torch.testing.assert_close(value_cache.reshape(num_slots, row), expected_value)
    print(f"  tokens={num_tokens} heads={num_heads} head_size={head_size} row={row}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reshape_and_cache_flash through TensorView")
    parser.add_argument("--tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--heads", type=int, default=NUM_HEADS)
    parser.add_argument("--head-size", type=int, default=HEAD_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.tokens, args.heads, args.head_size, args.block_size, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block;
    # a row width that is not a power of two makes the dense tile wider than the view.
    run_case(args.tokens - 7, args.heads, args.head_size, args.block_size, device)
    run_case(512, 3, args.head_size, args.block_size, device)
    print("Test Passed!")
