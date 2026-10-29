"""embedding: look up one weight row per token index.

The weight rows are read through a GatherScatterView whose sparse dimension takes the token
indices, and the output tile is written through a PartitionView. An index that falls outside
the batch is turned into an out-of-range row, which the view pads instead of reading.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
VOCAB_SIZE = 32000
NUM_TOKENS = 4096
EMBED_DIM = 256
BLOCK_M = 32


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def embedding_kernel(out, indices, weight, num_tokens, vocab_size, DIM: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_DIM: tl.constexpr):
    token_block = tl.program_id(0)
    tokens = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(indices + tokens, mask=tokens < num_tokens, other=-1).to(tl.int32)

    weight_view = tle.make_gather_scatter_view(weight, shape=[vocab_size, DIM], strides=[DIM, 1],
                                               tile=[BLOCK_M, BLOCK_DIM], sparse_dim=[0], bounds_check=True)
    gathered = tle.tile_load(weight_view, index=(rows, 0), dst_space=tle.language.dsa.ascend.UB)

    out_view = tle.make_partition_view(out, shape=[num_tokens, DIM], strides=[DIM, 1], tile=[BLOCK_M, BLOCK_DIM])
    tle.tile_store(out_view, value=gathered, index=(token_block, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def embedding(indices, weight, block_m=BLOCK_M):
    """indices is [num_tokens] and weight is [vocab_size, embed_dim]."""
    vocab_size, dim = weight.shape
    num_tokens = indices.numel()
    out = torch.empty((num_tokens, dim), dtype=weight.dtype, device=weight.device)
    grid = (triton.cdiv(num_tokens, block_m), )
    embedding_kernel[grid](out, indices, weight, num_tokens, vocab_size, DIM=dim, BLOCK_M=block_m,
                           BLOCK_DIM=triton.next_power_of_2(dim))
    return out


def run_case(vocab_size, num_tokens, dim, device):
    weight = torch.randn((vocab_size, dim), dtype=torch.float16, device=device)
    indices = torch.randint(0, vocab_size, (num_tokens, ), dtype=torch.int32, device=device)

    out = embedding(indices, weight)

    torch.testing.assert_close(out, weight[indices.long()])
    print(f"  vocab={vocab_size} tokens={num_tokens} dim={dim}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="embedding through TensorView")
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument("--tokens", type=int, default=NUM_TOKENS)
    parser.add_argument("--dim", type=int, default=EMBED_DIM)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.vocab, args.tokens, args.dim, device)
    # A token count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case(args.vocab, args.tokens - 9, args.dim, device)
    run_case(4096, 1024, 768, device)
    print("Test Passed!")
