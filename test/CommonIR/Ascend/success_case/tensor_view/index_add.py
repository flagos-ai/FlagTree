"""index_add on dim 0: add each source row into the row of the input named by the index.

One GatherScatterView serves both directions: the target rows are read through it, the source
tile comes in through a PartitionView, and the sum goes back out through the same scatter
view. The indices must be distinct, since two tiles that name the same row would both read it
before either writes.
"""
import argparse
import math

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_ROWS = 32000
NUM_INDICES = 4096
ROW_WIDTH = 256
# The kernel holds the target tile, the source tile and their sum at once, so the tile is
# shorter than for operators that keep a single tile live.
BLOCK_M = 8


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def index_add_kernel(out, src, index, num_indices, num_rows, alpha, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(tl.int32)

    out_view = tle.make_gather_scatter_view(out, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_M, BLOCK_WIDTH], sparse_dim=[0], bounds_check=True)
    src_view = tle.make_partition_view(src, shape=[num_indices, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    current = tle.tile_load(out_view, index=(rows, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    update = tle.tile_load(src_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)

    tle.tile_store(out_view, value=(current + alpha * update).to(out.dtype.element_ty), index=(rows, 0),
                   src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def index_add(inp, dim, index, source, alpha=1, block_m=BLOCK_M):
    """dim must be 0. The trailing dimensions collapse into one contiguous row."""
    assert dim == 0, "this path covers dim 0, where an index names a contiguous row"
    out = inp.contiguous().clone()
    source = source.contiguous()
    num_rows = out.shape[0]
    width = math.prod(out.shape[1:]) if out.dim() > 1 else 1
    num_indices = index.numel()
    grid = (triton.cdiv(num_indices, block_m), )
    index_add_kernel[grid](out, source, index, num_indices, num_rows, alpha, WIDTH=width, BLOCK_M=block_m,
                           BLOCK_WIDTH=triton.next_power_of_2(width))
    return out


def run_case(shape, num_indices, alpha, device):
    inp = torch.randn(shape, dtype=torch.float16, device=device)
    source = torch.randn((num_indices, *shape[1:]), dtype=torch.float16, device=device)
    index = torch.randperm(shape[0], device=device)[:num_indices].to(torch.int64)

    out = index_add(inp, 0, index.to(torch.int32), source, alpha)

    expected = inp.float().index_add(0, index, source.float(), alpha=alpha).to(torch.float16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    print(f"  shape={tuple(shape)} indices={num_indices} alpha={alpha}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index_add dim 0 through TensorView")
    parser.add_argument("--rows", type=int, default=NUM_ROWS)
    parser.add_argument("--indices", type=int, default=NUM_INDICES)
    parser.add_argument("--width", type=int, default=ROW_WIDTH)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case((args.rows, args.width), args.indices, 1, device)
    # An index count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case((args.rows, args.width), args.indices - 7, 1, device)
    run_case((4096, 768), 1024, 0.5, device)
    print("Test Passed!")
