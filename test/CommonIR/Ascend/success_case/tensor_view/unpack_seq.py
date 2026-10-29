"""unpack_seq: gather a padded [B, Lmax, D] tensor back into a ragged [N, D] tensor.

Sequence b owns the lengths[b] rows that follow the rows of sequences 0..b-1 in the output.
Each program handles one tile of one sequence. The padded input is read through a StridedView
and the output is written through a StridedView whose shape ends at the last valid row of the
sequence, so the rows of a tile that lie past the sequence length are dropped by the store and
never reach the next sequence.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_SEQS = 8
MAX_LEN = 512
FEATURES = 128
TILE_ELEMENTS = 32768
MAX_BLOCK_T = 128


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def unpack_seq_kernel(out, packed, lengths, max_len, FEATURES: tl.constexpr, BLOCK_T: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    seq = tl.program_id(0)
    tile = tl.program_id(1)

    start = 0
    for i in range(seq):
        start += tl.load(lengths + i)
    length = tl.minimum(tl.load(lengths + seq), max_len)

    packed_view = tle.make_strided_view(packed, shape=[(seq + 1) * max_len, FEATURES], strides=[FEATURES, 1],
                                        tile=[BLOCK_T, BLOCK_D], traversal_strides=[1, BLOCK_D])
    rows = tle.tile_load(packed_view, index=(seq * max_len + tile * BLOCK_T, 0), dst_space=tle.language.dsa.ascend.UB)

    out_view = tle.make_strided_view(out, shape=[start + length, FEATURES], strides=[FEATURES, 1],
                                     tile=[BLOCK_T, BLOCK_D], traversal_strides=[1, BLOCK_D])
    tle.tile_store(out_view, value=rows, index=(start + tile * BLOCK_T, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def unpack_seq(packed, lengths, block_t=None):
    """packed is [B, Lmax, ...]. Returns [sum(lengths), ...] holding the first lengths[b] rows
    of every sequence in order."""
    original_shape = packed.shape
    num_seqs, max_len = original_shape[:2]
    packed = packed.reshape(num_seqs, max_len, -1).contiguous()
    features = packed.shape[2]
    total = int(lengths.sum().item())
    lengths = lengths.to(torch.int32).contiguous()

    block_d = triton.next_power_of_2(features)
    if block_t is None:
        block_t = min(MAX_BLOCK_T, max(1, TILE_ELEMENTS // block_d), triton.next_power_of_2(max_len))
    out = torch.empty((total, features), dtype=packed.dtype, device=packed.device)
    grid = (num_seqs, triton.cdiv(max_len, block_t))
    unpack_seq_kernel[grid](out, packed, lengths, max_len, FEATURES=features, BLOCK_T=block_t, BLOCK_D=block_d)
    return out.reshape((total, ) + tuple(original_shape[2:]))


def run_case(num_seqs, max_len, features, device, dtype=torch.float16, min_len=None):
    min_len = max_len // 2 if min_len is None else min_len
    lengths = torch.randint(min_len, max_len + 1, (num_seqs, ), dtype=torch.int32)
    lengths[0] = max_len
    if dtype == torch.uint8:
        packed = torch.randint(0, 255, (num_seqs, max_len, features), dtype=dtype, device=device)
    else:
        packed = torch.randn((num_seqs, max_len, features), dtype=dtype, device=device)

    out = unpack_seq(packed, lengths.to(device))

    expected = torch.cat([packed[b, :n] for b, n in enumerate(lengths.tolist())])
    assert torch.equal(out, expected)
    print(f"  seqs={num_seqs} max_len={max_len} features={features} {dtype}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unpack_seq through TensorView")
    parser.add_argument("--seqs", type=int, default=NUM_SEQS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--features", type=int, default=FEATURES)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.seqs, args.max_len, args.features, device)
    # A maximum length that is not a multiple of the tile height leaves a partial tail tile;
    # a feature width that is not a power of two makes the dense tile wider than the view.
    run_case(8, 500, 128, device)
    run_case(4, 1024, 512, device)
    run_case(16, 256, 100, device)
    run_case(5, 77, 130, device)
    run_case(4, 200, 64, device, dtype=torch.uint8)
    # Short and empty sequences leave whole tiles past the sequence length.
    run_case(8, 300, 128, device, min_len=0)
    run_case(6, 40, 96, device, min_len=1)
    print("Test Passed!")
