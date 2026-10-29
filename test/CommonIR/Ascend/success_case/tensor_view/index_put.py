"""index_put_ with a single dim-0 index: write each row of values into the row of self named
by the index.

Same access pattern as index_copy_ on dim 0, a PartitionView over the values and a
GatherScatterView over the destination rows, with the accumulating form added, where the
destination rows are also read back through the same view before the add.
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
#  Kernels
# =============================================================================
@triton.jit
def index_put_kernel(dst, values, index, num_indices, num_rows, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr,
                     BLOCK_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(tl.int32)

    values_view = tle.make_partition_view(values, shape=[num_indices, WIDTH], strides=[WIDTH, 1],
                                          tile=[BLOCK_T, BLOCK_WIDTH])
    tile_values = tle.tile_load(values_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB)

    dst_view = tle.make_gather_scatter_view(dst, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_T, BLOCK_WIDTH], sparse_dim=[0], bounds_check=True)
    tle.tile_store(dst_view, value=tile_values, index=(rows, 0), src_space=tle.language.dsa.ascend.UB)


@triton.jit
def index_put_accumulate_kernel(dst, values, index, num_indices, num_rows, WIDTH: tl.constexpr, BLOCK_T: tl.constexpr,
                                BLOCK_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    rows = tl.load(index + offsets, mask=offsets < num_indices, other=-1).to(tl.int32)

    values_view = tle.make_partition_view(values, shape=[num_indices, WIDTH], strides=[WIDTH, 1],
                                          tile=[BLOCK_T, BLOCK_WIDTH])
    tile_values = tle.tile_load(values_view, index=(tile, 0), dst_space=tle.language.dsa.ascend.UB)

    dst_view = tle.make_gather_scatter_view(dst, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_T, BLOCK_WIDTH], sparse_dim=[0], bounds_check=True)
    current = tle.tile_load(dst_view, index=(rows, 0), dst_space=tle.language.dsa.ascend.UB)
    tle.tile_store(dst_view, value=current + tile_values, index=(rows, 0), src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def index_put_(self, indices, values, accumulate=False, block_t=BLOCK_T):
    """indices is a one-element list holding a dim-0 index tensor, so each index names a
    contiguous row. The accumulating form requires distinct indices: concurrent tiles would
    otherwise read the same row before either writes it."""
    assert len(indices) == 1, "this path covers a single dim-0 index"
    index = indices[0]
    num_rows, width = self.shape
    num_indices = index.numel()
    grid = (triton.cdiv(num_indices, block_t), )
    kernel = index_put_accumulate_kernel if accumulate else index_put_kernel
    kernel[grid](self, values, index, num_indices, num_rows, WIDTH=width, BLOCK_T=block_t,
                 BLOCK_WIDTH=triton.next_power_of_2(width))
    return self


def run_case(num_rows, num_indices, width, device, accumulate, sentinel=-7.0):
    dst = torch.full((num_rows, width), sentinel, dtype=torch.float16, device=device)
    values = torch.randn((num_indices, width), dtype=torch.float16, device=device)
    index = torch.randperm(num_rows, device=device)[:num_indices].to(torch.int64)

    expected = dst.clone()
    if accumulate:
        expected[index] += values
    else:
        expected[index] = values

    index_put_(dst, [index], values, accumulate=accumulate)

    torch.testing.assert_close(dst, expected)
    print(f"  rows={num_rows} indices={num_indices} width={width} "
          f"accumulate={accumulate}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index_put_ dim 0 through TensorView")
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

    for accumulate in (False, True):
        run_case(args.rows, args.indices, args.width, device, accumulate)
        # An index count that is not a multiple of the tile height leaves a partial tail
        # block; a row width that is not a power of two makes the tile wider than the view.
        run_case(args.rows, args.indices - 5, args.width, device, accumulate)
        run_case(4096, 512, 768, device, accumulate)
    print("Test Passed!")
