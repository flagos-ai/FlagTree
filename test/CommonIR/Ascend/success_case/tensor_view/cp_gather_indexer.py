"""cp_gather_indexer: gather one KV cache row per token, following a paged block table.

Each token's physical row is derived from the cumulative sequence lengths and the block
table, then the cache rows are read through a GatherScatterView over the flattened cache
(bounds_check enabled, so an invalid token reads as padding) and the result is written
through a PartitionView. The per-token quantisation scales keep their
own layout and are not part of the view path.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
BATCH_SIZE = 4
TOKENS_PER_SEQ = 1024
HEAD_DIM = 128
QUANT_BLOCKS = 1
CACHE_BLOCK_SIZE = 64
BLOCK_T = 32


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def cp_gather_indexer_kernel(kv_cache, kv_scale, out, scale_out, block_table, cu_seqlen, num_tokens, cache_rows,
                             block_size, table_stride, scale_stride, batch, HEAD_DIM: tl.constexpr,
                             QUANT_BLOCKS: tl.constexpr, BATCH_SLOTS: tl.constexpr, BLOCK_T: tl.constexpr):
    token_block = tl.program_id(0)
    tokens = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    in_batch = tokens < num_tokens

    # Locate each token's sequence by the only interval of cu_seqlen that contains it.
    seqs = tl.arange(0, BATCH_SLOTS)
    in_batch_seq = seqs < batch
    starts = tl.load(cu_seqlen + seqs, mask=in_batch_seq, other=0)
    ends = tl.load(cu_seqlen + seqs + 1, mask=in_batch_seq, other=0)
    inside = ((tokens[:, None] >= starts[None, :]) & (tokens[:, None] < ends[None, :]) & in_batch_seq[None, :])
    seq = tl.max(tl.where(inside, seqs[None, :], -1), axis=1)
    valid = in_batch & (seq >= 0)

    offset = tokens - tl.load(cu_seqlen + seq, mask=valid, other=0)
    block = tl.load(block_table + seq * table_stride + offset // block_size, mask=valid, other=0)
    slot = offset % block_size
    rows = tl.where(valid, block * block_size + slot, -1).to(tl.int32)

    cache_view = tle.make_gather_scatter_view(kv_cache, shape=[cache_rows, HEAD_DIM], strides=[HEAD_DIM, 1],
                                              tile=[BLOCK_T, HEAD_DIM], sparse_dim=[0], bounds_check=True)
    gathered = tle.tile_load(cache_view, index=(rows, 0), dst_space=tle.language.dsa.ascend.UB)
    out_view = tle.make_partition_view(out, shape=[num_tokens, HEAD_DIM], strides=[HEAD_DIM, 1],
                                       tile=[BLOCK_T, HEAD_DIM])
    tle.tile_store(out_view, value=gathered, index=(token_block, 0), src_space=tle.language.dsa.ascend.UB)

    quant = tl.arange(0, QUANT_BLOCKS)
    scale = tl.load(kv_scale + block[:, None] * scale_stride + slot[:, None] * QUANT_BLOCKS + quant[None, :],
                    mask=valid[:, None], other=0.0)
    tl.store(scale_out + tokens[:, None] * QUANT_BLOCKS + quant[None, :], scale, mask=valid[:, None])


# =============================================================================
#  Operator entry
# =============================================================================
def cp_gather_indexer(kv_cache, kv_scale, block_table, cu_seqlen, num_tokens, block_t=BLOCK_T):
    """kv_cache is [num_blocks, block_size, head_dim] and kv_scale is
    [num_blocks, block_size, quant_blocks]; cu_seqlen holds batch + 1 prefix sums."""
    num_blocks, block_size, head_dim = kv_cache.shape
    quant_blocks = kv_scale.shape[2]
    batch = block_table.shape[0]
    out = torch.zeros((num_tokens, head_dim), dtype=kv_cache.dtype, device=kv_cache.device)
    scale_out = torch.zeros((num_tokens, quant_blocks), dtype=kv_scale.dtype, device=kv_scale.device)
    grid = (triton.cdiv(num_tokens, block_t), )
    cp_gather_indexer_kernel[grid](kv_cache, kv_scale, out, scale_out, block_table, cu_seqlen,
                                   num_tokens, num_blocks * block_size, block_size, block_table.stride(0),
                                   kv_scale.stride(0), batch, HEAD_DIM=head_dim, QUANT_BLOCKS=quant_blocks,
                                   BATCH_SLOTS=triton.next_power_of_2(batch), BLOCK_T=block_t)
    return out, scale_out


def run_case(batch, tokens_per_seq, head_dim, quant_blocks, block_size, device):
    num_tokens = batch * tokens_per_seq
    blocks_per_seq = triton.cdiv(tokens_per_seq, block_size)
    num_blocks = batch * blocks_per_seq
    kv_cache = torch.randn((num_blocks, block_size, head_dim), dtype=torch.float16, device=device)
    kv_scale = torch.rand((num_blocks, block_size, quant_blocks), dtype=torch.float32, device=device)
    block_table = torch.randperm(num_blocks, device=device)[:num_blocks].reshape(batch, blocks_per_seq).to(torch.int32)
    cu_seqlen = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * tokens_per_seq

    out, scale_out = cp_gather_indexer(kv_cache, kv_scale, block_table, cu_seqlen, num_tokens)

    tokens = torch.arange(num_tokens, device=device)
    seq = tokens // tokens_per_seq
    offset = tokens % tokens_per_seq
    block = block_table[seq, offset // block_size].long()
    slot = offset % block_size
    torch.testing.assert_close(out, kv_cache[block, slot])
    torch.testing.assert_close(scale_out, kv_scale[block, slot])
    print(f"  batch={batch} tokens={num_tokens} head_dim={head_dim} "
          f"quant_blocks={quant_blocks}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cp_gather_indexer through TensorView")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--tokens-per-seq", type=int, default=TOKENS_PER_SEQ)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--quant-blocks", type=int, default=QUANT_BLOCKS)
    parser.add_argument("--block-size", type=int, default=CACHE_BLOCK_SIZE)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.batch, args.tokens_per_seq, args.head_dim, args.quant_blocks, args.block_size, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block.
    run_case(args.batch, args.tokens_per_seq - 5, args.head_dim, args.quant_blocks, args.block_size, device)
    run_case(2, 256, 64, 4, 32, device)
    # A batch size that is not a power of two pads the sequence axis of the lookup.
    run_case(3, 256, 64, 1, 32, device)
    print("Test Passed!")
