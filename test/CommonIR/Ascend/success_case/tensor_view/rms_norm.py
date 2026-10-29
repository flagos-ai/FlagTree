"""rms_norm: normalise each row by its root mean square and scale by the weight.

The input tile, the weight row and the result are accessed through PartitionViews, so the
whole operator stays on the tile.load / tile.store path. Rows past the end of the tensor fall
outside the view and are clipped by it, which covers the tail block without a mask. One tile
covers a full row, which bounds the supported row width by the tile the UB can hold.
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
BLOCK_M = 8
EPS = 1e-5


# =============================================================================
#  Kernel
# =============================================================================
@triton.jit
def rms_norm_kernel(y, inv_rms, x, weight, num_rows, eps, WIDTH: tl.constexpr, BLOCK_M: tl.constexpr,
                    BLOCK_WIDTH: tl.constexpr):
    row_block = tl.program_id(0)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)

    x_view = tle.make_partition_view(x, shape=[num_rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    weight_view = tle.make_partition_view(weight, shape=[WIDTH], strides=[1], tile=[BLOCK_WIDTH])
    tile = tle.tile_load(x_view, index=(row_block, 0), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)
    scale = tle.tile_load(weight_view, index=(0, ), dst_space=tle.language.dsa.ascend.UB).to(tl.float32)

    rrms = 1.0 / tl.sqrt(tl.sum(tile * tile, axis=1) / WIDTH + eps)
    normalized = tile * rrms[:, None] * scale[None, :]

    y_view = tle.make_partition_view(y, shape=[num_rows, WIDTH], strides=[WIDTH, 1], tile=[BLOCK_M, BLOCK_WIDTH])
    tle.tile_store(y_view, value=normalized.to(y.dtype.element_ty), index=(row_block, 0),
                   src_space=tle.language.dsa.ascend.UB)
    tl.store(inv_rms + rows, rrms, mask=rows < num_rows)


# =============================================================================
#  Operator entry
# =============================================================================
def rms_norm_forward(x, normalized_shape, weight, eps=EPS, block_m=BLOCK_M):
    """Returns the normalised tensor and the inverse RMS of each row, which is what a
    backward pass needs. x is [..., width] and weight is [width]; the leading dimensions
    collapse into rows."""
    width = normalized_shape[-1]
    assert x.shape[-1] == width and weight.numel() == width
    rows = x.numel() // width
    y = torch.empty_like(x)
    inv_rms = torch.empty((rows, ), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(rows, block_m), )
    rms_norm_kernel[grid](y, inv_rms, x, weight, rows, eps, WIDTH=width, BLOCK_M=block_m,
                          BLOCK_WIDTH=triton.next_power_of_2(width))
    return y, inv_rms


def rms_norm(x, normalized_shape, weight, eps=EPS):
    """Forward entry: the inverse RMS stays internal."""
    return rms_norm_forward(x, normalized_shape, weight, eps)[0]


def run_case(num_rows, width, device):
    x = torch.randn((num_rows, width), dtype=torch.float16, device=device)
    weight = torch.randn((width, ), dtype=torch.float16, device=device)

    y, inv_rms = rms_norm_forward(x, (width, ), weight)

    xf = x.float()
    expected_inv = 1.0 / torch.sqrt((xf * xf).mean(-1) + EPS)
    expected = (xf * expected_inv[:, None] * weight.float()[None, :]).to(torch.float16)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(inv_rms, expected_inv, rtol=1e-3, atol=1e-3)
    print(f"  rows={num_rows} width={width}: ok")


# =============================================================================
#  CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rms_norm through TensorView")
    parser.add_argument("--rows", type=int, default=NUM_ROWS)
    parser.add_argument("--width", type=int, default=ROW_WIDTH)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
    torch.manual_seed(0)

    if args.no_check:
        print("Reference check skipped.")
        raise SystemExit(0)

    run_case(args.rows, args.width, device)
    # A row count that is not a multiple of the tile height leaves a partial tail block;
    # a width that is not a power of two makes the dense tile wider than the view.
    run_case(args.rows - 3, args.width, device)
    run_case(1024, 768, device)
    print("Test Passed!")
