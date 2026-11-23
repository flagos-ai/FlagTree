"""compute_global_topk_indices_and_lens: map block-local top-k indices to global cache slots.

Token t belongs to request token_to_req[t]. A non-negative local index i selects the slot
block_table[req, i // block_size] * block_size + i % block_size, and a negative index stays -1.
The second output counts the non-negative indices of every token and is 0 for tokens whose
validity flag is 0.

Each program handles BLOCK_T tokens. The top-k rows, the request ids and the validity flags are
read through PartitionViews and both outputs are written through PartitionViews, so tail tokens
and the columns past the top-k width fall outside the views and are dropped on store. The block
table lookup is the only data-dependent access and stays a masked load.
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
TOPK = 128
NUM_REQS = 4
BLOCKS_PER_REQ = 640
BLOCK_SIZE = 64
TILE_ELEMENTS = 8192
MAX_BLOCK_T = 32
MIN_PROGRAMS = 32
# A one-row tile does not compile, so the tile height is at least 2.
MIN_BLOCK_T = 2


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def compute_global_topk_kernel(global_indices, lens, local_indices, token_to_req, block_table, is_valid, num_tokens,
                               block_table_stride, block_size, TOPK: tl.constexpr, BLOCK_T: tl.constexpr,
                               BLOCK_K: tl.constexpr):
    tile = tl.program_id(0)
    rows = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_K)

    local_view = tle.make_partition_view(local_indices, shape=[num_tokens, TOPK], strides=[TOPK, 1],
                                         tile=[BLOCK_T, BLOCK_K])
    req_view = tle.make_partition_view(token_to_req, shape=[num_tokens], strides=[1], tile=[BLOCK_T])
    flag_view = tle.make_partition_view(is_valid, shape=[num_tokens], strides=[1], tile=[BLOCK_T])
    local = tle.tile_load(local_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB)
    req = tle.tile_load(req_view, index=(tile, ), dst_space=tle.language.dsa.ascend.UB)
    token_valid = tle.tile_load(flag_view, index=(tile, ), dst_space=tle.language.dsa.ascend.UB)

    inside = (rows < num_tokens)[:, None] & (cols < TOPK)[None, :]
    valid = inside & (local >= 0)
    block_idx = local // block_size
    block_off = local - block_idx * block_size
    block_no = tl.load(block_table + req[:, None] * block_table_stride + block_idx, mask=valid, other=0)
    slot = tl.where(valid, block_no * block_size + block_off, -1)

    out_view = tle.make_partition_view(global_indices, shape=[num_tokens, TOPK], strides=[TOPK, 1],
                                       tile=[BLOCK_T, BLOCK_K])
    tle.tile_store(out_view, value=slot, index=(tile, 0), src_space=tle.language.dsa.ascend.UB)

    count = tl.sum(valid.to(tl.int32), axis=1)
    lens_view = tle.make_partition_view(lens, shape=[num_tokens], strides=[1], tile=[BLOCK_T])
    tle.tile_store(lens_view, value=tl.where(token_valid != 0, count, 0), index=(tile, ),
                   src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def compute_global_topk_indices_and_lens(topk_indices, token_to_req_indices, block_table, block_size,
                                         is_valid_token=None, block_t=None):
    """Returns (global_indices, lens). global_indices has the shape of topk_indices and holds
    the cache slot of every non-negative index, -1 elsewhere. lens counts the non-negative
    indices per token and is 0 for tokens flagged invalid."""
    assert topk_indices.dim() == 2 and block_table.dim() == 2
    topk_indices = topk_indices.contiguous()
    num_tokens, topk = topk_indices.shape
    if is_valid_token is None:
        is_valid_token = torch.ones((num_tokens, ), dtype=torch.int32, device=topk_indices.device)
    block_k = triton.next_power_of_2(topk)
    if block_t is None:
        block_t = max(MIN_BLOCK_T, min(MAX_BLOCK_T, TILE_ELEMENTS // block_k))
        while block_t > MIN_BLOCK_T and triton.cdiv(num_tokens, block_t) < MIN_PROGRAMS:
            block_t //= 2
    global_indices = torch.empty_like(topk_indices)
    lens = torch.empty((num_tokens, ), dtype=torch.int32, device=topk_indices.device)
    grid = (triton.cdiv(num_tokens, block_t), )
    compute_global_topk_kernel[grid](global_indices, lens,
                                     topk_indices, token_to_req_indices, block_table, is_valid_token, num_tokens,
                                     block_table.stride(0), block_size, TOPK=topk, BLOCK_T=block_t, BLOCK_K=block_k)
    return global_indices, lens


def run_case(num_tokens, topk, num_reqs, blocks_per_req, block_size, device, invalid_ratio=0.25):
    topk_indices = torch.randint(-1, blocks_per_req * block_size, (num_tokens, topk), dtype=torch.int32, device=device)
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device) % num_reqs
    block_table = torch.randperm(num_reqs * blocks_per_req,
                                 device=device).to(torch.int32).view(num_reqs, blocks_per_req)
    is_valid = (torch.rand(num_tokens, device=device) >= invalid_ratio).to(torch.int32)

    slots, lens = compute_global_topk_indices_and_lens(topk_indices, token_to_req, block_table, block_size, is_valid)

    local = topk_indices.long()
    valid = local >= 0
    block_idx = torch.where(valid, local // block_size, torch.zeros_like(local))
    table = block_table.long()[token_to_req.long()[:, None], block_idx]
    expected = torch.where(valid, table * block_size + local % block_size, torch.full_like(local, -1)).int()
    expected_lens = torch.where(is_valid != 0, valid.sum(1), torch.zeros_like(valid.sum(1))).int()
    assert torch.equal(slots, expected) and torch.equal(lens, expected_lens)
    print(f"  tokens={num_tokens} topk={topk} reqs={num_reqs} block_size={block_size}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute_global_topk_indices_and_lens "
                                     "through TensorView")
    parser.add_argument("--tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--reqs", type=int, default=NUM_REQS)
    parser.add_argument("--blocks-per-req", type=int, default=BLOCKS_PER_REQ)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.tokens, args.topk, args.reqs, args.blocks_per_req, args.block_size, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block;
    # a top-k width that is not a power of two makes the dense tile wider than the view.
    run_case(args.tokens - 5, args.topk, args.reqs, args.blocks_per_req, args.block_size, device)
    run_case(13, 33, 3, 5, 16, device)
    run_case(100, 100, 2, 7, 64, device)
    run_case(512, 64, 2, 128, 64, device)
    run_case(5, 4, 2, 4, 64, device, invalid_ratio=0.0)
    print("Test Passed!")
