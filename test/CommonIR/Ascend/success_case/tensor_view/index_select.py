"""index_select on dim 0: build an output whose rows are the input rows named by the index.

The input rows are read through a GatherScatterView whose sparse dimension takes the index,
and the output tile is written through a PartitionView. Selecting along dim 0 keeps every
selected row contiguous, so no transpose of the input is needed.
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
BLOCK_M = 32


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def index_select_kernel(out, inp, index, num_indices, num_rows, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr,
                        BLOCK_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(tl.int32)

    inp_view = tle.make_gather_scatter_view(inp, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_M, BLOCK_WIDTH], sparse_dim=[0], bounds_check=True)
    selected = tle.tile_load(inp_view, index=(rows, 0), dst_space=tle.language.dsa.ascend.UB)

    out_view = tle.make_partition_view(out, shape=[num_indices, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    tle.tile_store(out_view, value=selected, index=(tile, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def index_select(inp, dim, index, block_m=BLOCK_M):
    """dim must be 0. The trailing dimensions collapse into one contiguous row."""
    assert dim == 0, "this path covers dim 0, where an index names a contiguous row"
    inp = inp.contiguous()
    num_rows = inp.shape[0]
    width = math.prod(inp.shape[1:]) if inp.dim() > 1 else 1
    num_indices = index.numel()
    out = torch.empty((num_indices, *inp.shape[1:]), dtype=inp.dtype, device=inp.device)
    grid = (triton.cdiv(num_indices, block_m), )
    index_select_kernel[grid](out, inp, index, num_indices, num_rows, WIDTH=width, BLOCK_M=block_m,
                              BLOCK_WIDTH=triton.next_power_of_2(width))
    return out


def run_case(shape, num_indices, device):
    inp = torch.randn(shape, dtype=torch.float16, device=device)
    index = torch.randint(0, shape[0], (num_indices, ), dtype=torch.int32, device=device)

    out = index_select(inp, 0, index)

    torch.testing.assert_close(out, inp[index.long()])
    print(f"  shape={tuple(shape)} indices={num_indices}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index_select dim 0 through TensorView")
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

    run_case((args.rows, args.width), args.indices, device)
    # An index count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case((args.rows, args.width), args.indices - 7, device)
    run_case((4096, 768), 1024, device)
    # Trailing dimensions beyond the first collapse into the row width.
    run_case((2048, 8, 32), 512, device)
    print("Test Passed!")
