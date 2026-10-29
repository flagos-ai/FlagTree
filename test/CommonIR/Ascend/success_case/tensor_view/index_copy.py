"""index_copy_ on dim 0: copy each row of the source into the row of self named by index.

The source tile is read through a PartitionView and the destination rows are written through
a GatherScatterView whose sparse dimension takes the index tensor. Rows of the destination
that no index names are left untouched, which is what the view's scatter semantics give for
an out-of-range row.
"""
import argparse

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
BLOCK_T = 32


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def index_copy_kernel(dst, src, index, num_indices, num_rows, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr,
                      BLOCK_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(tl.int32)

    src_view = tle.make_partition_view(src, shape=[num_indices, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_T, BLOCK_WIDTH])
    values = tle.tile_load(src_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB)

    dst_view = tle.make_gather_scatter_view(dst, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_T, BLOCK_WIDTH], sparse_dim=[0], bounds_check=True)
    tle.tile_store(dst_view, value=values, index=(rows, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def index_copy_(self, dim, index, source, block_t=BLOCK_T):
    """dim must be 0: the index names whole rows, each of which is contiguous."""
    assert dim == 0, "this path covers dim 0, where an index names a contiguous row"
    num_rows, width = self.shape
    num_indices = index.numel()
    grid = (triton.cdiv(num_indices, block_t), )
    index_copy_kernel[grid](self, source, index, num_indices, num_rows, WIDTH=width, BLOCK_T=block_t,
                            BLOCK_WIDTH=triton.next_power_of_2(width))
    return self


def run_case(num_rows, num_indices, width, device, sentinel=-7.0):
    dst = torch.full((num_rows, width), sentinel, dtype=torch.float16, device=device)
    source = torch.randn((num_indices, width), dtype=torch.float16, device=device)
    index = torch.randperm(num_rows, device=device)[:num_indices].to(torch.int64)

    index_copy_(dst, 0, index, source)

    expected = torch.full((num_rows, width), sentinel, dtype=torch.float16, device=device)
    expected[index] = source
    torch.testing.assert_close(dst, expected)
    print(f"  rows={num_rows} indices={num_indices} width={width}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index_copy_ dim 0 through TensorView")
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

    run_case(args.rows, args.indices, args.width, device)
    # An index count that is not a multiple of the tile height leaves a partial tail block;
    # a row width that is not a power of two makes the dense tile wider than the view.
    run_case(args.rows, args.indices - 5, args.width, device)
    run_case(4096, 512, 768, device)
    print("Test Passed!")
