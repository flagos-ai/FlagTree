"""paged_kv_block_gather: collect one sequence's KV blocks out of a paged cache.

The cache is viewed as [num_blocks, block_size, head_dim] with the block axis sparse, so the
block table drives the gather while each block's own two dimensions stay dense and move as
one contiguous transfer. The destination is written through a PartitionView.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_BLOCKS = 1024
BLOCK_SIZE = 32
HEAD_DIM = 128
BLOCKS_PER_SEQ = 8


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def paged_kv_block_gather_kernel(cache, block_table, out, num_blocks, num_seqs, table_stride, BLOCK_SIZE: tl.constexpr,
                                 HEAD_DIM: tl.constexpr, BLOCKS_PER_SEQ: tl.constexpr):
    seq = tl.program_id(0)
    slots = tl.arange(0, BLOCKS_PER_SEQ)
    physical = tl.load(block_table + seq * table_stride + slots).to(tl.int32)

    cache_view = tle.make_gather_scatter_view(cache, shape=[num_blocks, BLOCK_SIZE,
                                                            HEAD_DIM], strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
                                              tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM], sparse_dim=[0])
    gathered = tle.tile_load(cache_view, index=(physical, 0, 0), dst_space=tle.language.dsa.ascend.UB)

    # The destination holds one group of BLOCKS_PER_SEQ blocks per sequence, and a
    # partition view takes tile indices, so the sequence id selects the group directly.
    out_view = tle.make_partition_view(out, shape=[num_seqs * BLOCKS_PER_SEQ, BLOCK_SIZE,
                                                   HEAD_DIM], strides=[BLOCK_SIZE * HEAD_DIM, HEAD_DIM, 1],
                                       tile=[BLOCKS_PER_SEQ, BLOCK_SIZE, HEAD_DIM])
    tle.tile_store(out_view, value=gathered, index=(seq, 0, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def paged_kv_block_gather(cache, block_table):
    """cache is [num_blocks, block_size, head_dim] and block_table is
    [num_seqs, blocks_per_seq]. The result is [num_seqs, blocks_per_seq, block_size,
    head_dim]."""
    num_blocks, block_size, head_dim = cache.shape
    num_seqs, blocks_per_seq = block_table.shape
    out = torch.empty((num_seqs, blocks_per_seq, block_size, head_dim), dtype=cache.dtype, device=cache.device)
    paged_kv_block_gather_kernel[(num_seqs, )](cache, block_table, out, num_blocks, num_seqs, block_table.stride(0),
                                               BLOCK_SIZE=block_size, HEAD_DIM=head_dim, BLOCKS_PER_SEQ=blocks_per_seq)
    return out


def run_case(num_blocks, block_size, head_dim, num_seqs, blocks_per_seq, device):
    cache = torch.randn((num_blocks, block_size, head_dim), dtype=torch.float16, device=device)
    block_table = torch.randint(0, num_blocks, (num_seqs, blocks_per_seq), dtype=torch.int32, device=device)

    out = paged_kv_block_gather(cache, block_table)

    torch.testing.assert_close(out, cache[block_table.long()])
    print(f"  blocks={num_blocks} block_size={block_size} head_dim={head_dim} "
          f"seqs={num_seqs} blocks_per_seq={blocks_per_seq}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="paged KV block gather through TensorView")
    parser.add_argument("--blocks", type=int, default=NUM_BLOCKS)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--blocks-per-seq", type=int, default=BLOCKS_PER_SEQ)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.blocks, args.block_size, args.head_dim, 4, args.blocks_per_seq, device)
    # A single sequence and a single block per sequence reduce the sparse axis to one lane.
    run_case(args.blocks, args.block_size, args.head_dim, 1, 1, device)
    run_case(256, 16, 96, 2, 4, device)
    print("Test Passed!")
