"""pack_seq: scatter a ragged [N, D] tensor into a padded [B, Lmax, D] tensor.

Sequence b owns the lengths[b] rows that follow the rows of sequences 0..b-1. Each program
handles one tile of one sequence. The input is read through a StridedView and the output is
written through a StridedView whose shape ends at the last row of the sequence, so a tail tile
never spills into the next sequence.

The load tile always starts inside the input: a tile that begins past the end of the sequence is
moved back to the sequence start, and the rows beyond the sequence length are replaced by the pad
value after the load. This keeps every load on the whole-tile DMA path instead of the
element-by-element fill that a tile with no in-bounds row would take.
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
def pack_seq_kernel(out, x, lengths, num_rows, max_len, PAD_VALUE: tl.constexpr, FEATURES: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    seq = tl.program_id(0)
    tile = tl.program_id(1)

    start = 0
    for i in range(seq):
        start += tl.load(lengths + i)
    length = tl.load(lengths + seq)

    x_view = tle.make_strided_view(x, shape=[num_rows, FEATURES], strides=[FEATURES, 1], tile=[BLOCK_T, BLOCK_D],
                                   traversal_strides=[1, BLOCK_D])
    first = tl.where(tile * BLOCK_T < length, tile * BLOCK_T, 0)
    rows = tle.tile_load(x_view, index=(start + first, 0), dst_space=tle.language.dsa.ascend.UB)

    valid = (tile * BLOCK_T + tl.arange(0, BLOCK_T)) < length
    pad = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, rows.dtype)
    rows = tl.where(valid[:, None], rows, pad)

    out_view = tle.make_strided_view(out, shape=[(seq + 1) * max_len, FEATURES], strides=[FEATURES, 1],
                                     tile=[BLOCK_T, BLOCK_D], traversal_strides=[1, BLOCK_D])
    tle.tile_store(out_view, value=rows, index=(seq * max_len + tile * BLOCK_T, 0),
                   src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def pack_seq(x, lengths, pad_value=float("-inf"), block_t=None):
    """x is [N, ...] with N = sum(lengths). Returns [B, max(lengths), ...] with every row past
    a sequence's length set to pad_value."""
    is_uint8 = x.dtype == torch.uint8
    if is_uint8:
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255
    original_shape = x.shape
    x = x.reshape(original_shape[0], -1).contiguous()
    num_rows, features = x.shape
    num_seqs = lengths.numel()
    max_len = int(lengths.max().item())
    lengths = lengths.to(torch.int32).contiguous()

    block_d = triton.next_power_of_2(features)
    if block_t is None:
        block_t = min(MAX_BLOCK_T, max(1, TILE_ELEMENTS // block_d), triton.next_power_of_2(max_len))
    out = torch.empty((num_seqs, max_len, features), dtype=x.dtype, device=x.device)
    grid = (num_seqs, triton.cdiv(max_len, block_t))
    pack_seq_kernel[grid](out, x, lengths, num_rows, max_len,
                          PAD_VALUE=int(pad_value) if is_uint8 else float(pad_value), FEATURES=features,
                          BLOCK_T=block_t, BLOCK_D=block_d)
    return out.reshape((num_seqs, max_len) + tuple(original_shape[1:]))


def run_case(num_seqs, max_len, features, device, dtype=torch.float16, pad_value=float("-inf"), min_len=None):
    min_len = max_len // 2 if min_len is None else min_len
    lengths = torch.randint(min_len, max_len + 1, (num_seqs, ), dtype=torch.int32)
    lengths[0] = max_len
    total = int(lengths.sum())
    if dtype == torch.uint8:
        x = torch.randint(0, 255, (total, features), dtype=dtype, device=device)
    else:
        x = torch.randn((total, features), dtype=dtype, device=device)

    out = pack_seq(x, lengths.to(device), pad_value)

    expected = torch.full((num_seqs, max_len, features), pad_value, dtype=dtype, device=device)
    offset = 0
    for b, n in enumerate(lengths.tolist()):
        expected[b, :n] = x[offset:offset + n]
        offset += n
    assert torch.equal(out, expected)
    print(f"  seqs={num_seqs} max_len={max_len} features={features} {dtype}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pack_seq through TensorView")
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
    run_case(5, 77, 130, device, pad_value=0.0)
    run_case(4, 200, 64, device, dtype=torch.uint8, pad_value=7)
    # Short and empty sequences leave whole tiles past the sequence length.
    run_case(8, 300, 128, device, min_len=0)
    run_case(6, 40, 96, device, min_len=1)
    print("Test Passed!")
