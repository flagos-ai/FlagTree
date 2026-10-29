"""skip_layernorm: add the residual to the input, then layer-normalise each row.

The input, the residual and the output are PartitionViews, and the weight and bias rows are
PartitionViews over a single row. One tile covers a full row, so the mean and variance are plain
reductions along the tile width. Rows past the end of the tensor fall outside the view and are
dropped on store, which covers the tail block without a mask. A row width that is not a power of
two makes the dense tile wider than the view; the padded columns read as zero and are never
stored. The supported row width is bounded by the tile the UB can hold.
"""
import argparse

import torch
import triton
import triton.language as tl

import triton.experimental.tle as tle

# =============================================================================
#  Compile-time configuration
# =============================================================================
NUM_ROWS = 4096
ROW_WIDTH = 512
EPS = 1e-5
TILE_ELEMENTS = 16384


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def skip_layer_norm_kernel(y, x, residual, weight, bias, num_rows, eps, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_WIDTH: tl.constexpr):
    row_block = tl.program_id(0)

    x_view = tle.make_partition_view(x, shape=[num_rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    residual_view = tle.make_partition_view(residual, shape=[num_rows, WIDTH], strides=[WIDTH, 1],
                                            tile=[BLOCK_M, BLOCK_WIDTH])
    weight_view = tle.make_partition_view(weight, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH])
    bias_view = tle.make_partition_view(bias, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH])
    summed = tle.tile_load(x_view, index=(row_block, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    summed += tle.tile_load(residual_view, index=(row_block, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    scale = tle.tile_load(weight_view, index=(0, ), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    shift = tle.tile_load(bias_view, index=(0, ), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)

    mean = tl.sum(summed, axis=1) / WIDTH
    var = tl.sum(summed * summed, axis=1) / WIDTH - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = (summed - mean[:, None]) * rstd[:, None] * scale[None, :] + shift[None, :]

    y_view = tle.make_partition_view(y, shape=[num_rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    tle.tile_store(y_view, value=normalized.to(y.dtype.element_ty), index=(row_block, 0),
                   src_space=tle.language.dsa.ascend.UB)


# =============================================================================
#  Operator entry
# =============================================================================
def skip_layer_norm(x, residual, normalized_shape, weight, bias, eps=EPS, block_m=None):
    """Returns layer_norm(x + residual) over the trailing normalized_shape. x and residual
    share one shape and the trailing dimensions collapse into a row."""
    width = 1
    for size in normalized_shape:
        width *= size
    assert x.shape == residual.shape and x.numel() % width == 0
    assert weight.numel() == width and bias.numel() == width
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    rows = x.numel() // width
    block_width = triton.next_power_of_2(width)
    if block_m is None:
        block_m = max(1, TILE_ELEMENTS // block_width)
    y = torch.empty_like(x)
    grid = (triton.cdiv(rows, block_m), )
    skip_layer_norm_kernel[grid](y, x, residual, weight, bias, rows, eps, WIDTH=width, BLOCK_M=block_m,
                                 BLOCK_WIDTH=block_width)
    return y


def run_case(shape, width, device):
    x = torch.randn(shape, dtype=torch.float16, device=device)
    residual = torch.randn(shape, dtype=torch.float16, device=device)
    weight = torch.randn((width, ), dtype=torch.float16, device=device)
    bias = torch.randn((width, ), dtype=torch.float16, device=device)

    y = skip_layer_norm(x, residual, (width, ), weight, bias)

    summed = x.float() + residual.float()
    expected = torch.nn.functional.layer_norm(summed, (width, ), weight.float(), bias.float(), EPS).to(torch.float16)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)
    print(f"  shape={tuple(shape)} width={width}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="skip_layernorm through TensorView")
    parser.add_argument("--rows", type=int, default=NUM_ROWS)
    parser.add_argument("--width", type=int, default=ROW_WIDTH)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case((args.rows, args.width), args.width, device)
    # A row count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case((args.rows - 3, args.width), args.width, device)
    run_case((4096, 768), 768, device)
    run_case((2048, 1024), 1024, device)
    run_case((16, 33, 1000), 1000, device)
    print("Test Passed!")
