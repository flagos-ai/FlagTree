"""combine_topk_swa_indices: build the per-token index row that joins the compressed top-k
entries with the sliding-window entries.

Token t of request b sits at position pos in its sequence. Its row holds, in order, the first
topk_len = min((pos + 1) // compress_ratio, topk) top-k indices shifted by M * b, then
swa_len = min(pos + 1, window_size) consecutive window indices, then -1 up to the padded row
width. The second output is topk_len + swa_len.

Each program handles one tile of BLOCK_T consecutive tokens of one request. The top-k rows are
read through a StridedView whose row index is the token and the combined rows are written
through a StridedView whose shape ends at the last token of the request, so the tokens of a
tile that lie past the request are dropped on store. A tile that starts past the end of its
request loads from the request start instead, so every load keeps a row inside the view and takes
the whole-tile DMA path; all rows of such a tile are dropped on store. The whole padded row,
-1 filler included, is computed in the tile, so the output needs no separate fill pass. A padded
row wider than the top-k width makes the dense load tile wider than the view; the extra columns
read as zero and are overwritten by the window and filler values.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_REQS = 2
QUERY_LEN = 512
TOPK = 128
WINDOW_SIZE = 256
COMPRESS_RATIO = 4
OFFSET_M = 45056
OFFSET_N = 40960
ROW_ALIGNMENT = 128
TILE_ELEMENTS = 8192
MAX_BLOCK_T = 32
# A one-row tile does not compile, so the tile height is at least 2.
MIN_BLOCK_T = 2


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def combine_topk_swa_kernel(combined, lens, topk_indices, query_start_loc, seq_lens, gather_lens, num_tokens, offset_m,
                            offset_n, TOPK: tl.constexpr, COMBINED_TOPK: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
                            WINDOW_SIZE: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)

    base = tl.load(query_start_loc)
    query_start = tl.load(query_start_loc + batch) - base
    query_end = tl.load(query_start_loc + batch + 1) - base
    seq_len = tl.load(seq_lens + batch)
    gather_len = tl.load(gather_lens + batch)
    start_pos = seq_len - (query_end - query_start)
    gather_start = seq_len - gather_len
    first_token = query_start + tile * BLOCK_T

    topk_view = tle.make_strided_view(topk_indices, shape=[num_tokens, TOPK], strides=[TOPK, 1],
                                      tile=[BLOCK_T, BLOCK_C], traversal_strides=[1, BLOCK_C])
    load_token = tl.where(first_token < query_end, first_token, query_start)
    topk_vals = tle.tile_load(topk_view, index=(load_token, 0), dst_space=tle.language.dsa.ascend.UB)

    pos = start_pos + tile * BLOCK_T + tl.arange(0, BLOCK_T)
    topk_count = tl.minimum((pos + 1) // COMPRESS_RATIO, TOPK)
    swa_count = tl.minimum(pos + 1, WINDOW_SIZE)
    topk_len = topk_count[:, None]
    swa_len = swa_count[:, None]
    col = tl.arange(0, BLOCK_C)[None, :]

    window_vals = (batch * offset_m + offset_n + (col - topk_len) + pos[:, None] - swa_len + 1 - gather_start)
    row = tl.where(col < topk_len, topk_vals + batch * offset_m, tl.where(col < topk_len + swa_len, window_vals, -1))

    combined_view = tle.make_strided_view(combined, shape=[query_end, COMBINED_TOPK], strides=[COMBINED_TOPK, 1],
                                          tile=[BLOCK_T, BLOCK_C], traversal_strides=[1, BLOCK_C])
    tle.tile_store(combined_view, value=row, index=(first_token, 0), src_space=tle.language.dsa.ascend.UB)

    lens_view = tle.make_strided_view(lens, shape=[query_end], strides=[1], tile=[BLOCK_T], traversal_strides=[1])
    tle.tile_store(lens_view, value=topk_count + swa_count, index=(first_token, ), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def combine_topk_swa_indices(topk_indices, query_start_loc, seq_lens, gather_lens, window_size, compress_ratio, topk, m,
                             n, block_t=None):
    """Returns (combined, lens). The requests described by query_start_loc cover every row of
    topk_indices. combined is [num_tokens, padded width] with the padded width the top-k plus
    window size rounded up to a multiple of ROW_ALIGNMENT."""
    assert topk_indices.dim() == 2 and topk_indices.shape[1] >= topk
    topk_indices = topk_indices.contiguous()
    num_tokens, topk_width = topk_indices.shape
    num_reqs = seq_lens.numel()
    combined_topk = triton.cdiv(topk + window_size, ROW_ALIGNMENT) * ROW_ALIGNMENT
    block_c = triton.next_power_of_2(combined_topk)
    max_query_len = int((query_start_loc[1:] - query_start_loc[:-1]).max().item())
    if block_t is None:
        block_t = max(MIN_BLOCK_T, min(MAX_BLOCK_T, TILE_ELEMENTS // block_c, triton.next_power_of_2(max_query_len)))
    combined = torch.empty((num_tokens, combined_topk), dtype=torch.int32, device=topk_indices.device)
    lens = torch.empty((num_tokens, ), dtype=torch.int32, device=topk_indices.device)
    grid = (num_reqs, triton.cdiv(max_query_len, block_t))
    combine_topk_swa_kernel[grid](combined, lens, topk_indices, query_start_loc, seq_lens, gather_lens, num_tokens, m,
                                  n, TOPK=topk_width, COMBINED_TOPK=combined_topk, COMPRESS_RATIO=compress_ratio,
                                  WINDOW_SIZE=window_size, BLOCK_T=block_t, BLOCK_C=block_c)
    return combined, lens


def reference(topk_indices, query_start_loc, seq_lens, gather_lens, window_size, compress_ratio, topk, m, n):
    combined_topk = triton.cdiv(topk + window_size, ROW_ALIGNMENT) * ROW_ALIGNMENT
    combined = torch.full((topk_indices.shape[0], combined_topk), -1, dtype=torch.int32, device=topk_indices.device)
    lens = torch.empty((topk_indices.shape[0], ), dtype=torch.int32, device=topk_indices.device)
    starts = query_start_loc.tolist()
    for batch in range(seq_lens.numel()):
        start, end = starts[batch] - starts[0], starts[batch + 1] - starts[0]
        seq_len = int(seq_lens[batch])
        start_pos = seq_len - (end - start)
        gather_start = seq_len - int(gather_lens[batch])
        for token in range(start, end):
            pos = start_pos + token - start
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            combined[token, :topk_len] = topk_indices[token, :topk_len] + m * batch
            window = torch.arange(swa_len, dtype=torch.int32, device=topk_indices.device)
            combined[token, topk_len:topk_len + swa_len] = (m * batch + n + window + pos - swa_len + 1 - gather_start)
            lens[token] = topk_len + swa_len
    return combined, lens


def run_case(query_lens, seq_lens, gather_lens, window_size, compress_ratio, topk, m, n, device):
    num_tokens = sum(query_lens)
    topk_indices = torch.randint(-1, n, (num_tokens, topk), dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    gather_lens = torch.tensor(gather_lens, dtype=torch.int32, device=device)
    args = (topk_indices, query_start_loc, seq_lens, gather_lens, window_size, compress_ratio, topk, m, n)

    combined, lens = combine_topk_swa_indices(*args)

    expected, expected_lens = reference(*args)
    assert torch.equal(combined, expected) and torch.equal(lens, expected_lens)
    print(f"  queries={query_lens} topk={topk} window={window_size} "
          f"ratio={compress_ratio}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="combine_topk_swa_indices through TensorView")
    parser.add_argument("--reqs", type=int, default=NUM_REQS)
    parser.add_argument("--query-len", type=int, default=QUERY_LEN)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--ratio", type=int, default=COMPRESS_RATIO)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    queries = [args.query_len] * args.reqs
    seqs = [args.query_len * 2] * args.reqs
    run_case(queries, seqs, seqs, args.window, args.ratio, args.topk, OFFSET_M, OFFSET_N, device)
    # Query lengths that are not multiples of the tile height leave partial tail tiles;
    # a padded row width that is not a power of two makes the dense tile wider than the view.
    run_case([37, 21], [90, 60], [90, 40], 24, 4, 20, 1000, 500, device)
    run_case([3, 2], [6, 4], [4, 3], 4, 2, 4, 20, 8, device)
    run_case([128], [512], [256], 32, 4, 128, 42240, 40960, device)
    run_case([200, 55, 1], [300, 55, 9], [300, 50, 9], 64, 8, 100, 5000, 4000, device)
    print("Test Passed!")
