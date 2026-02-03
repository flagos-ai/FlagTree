import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def builtin_index_select_kernel(src_ptr, index_ptr, out_ptr):
    # Define 2x2 tile indices for output tensor
    r = tl.arange(0, 2)[:, None]  # Row indices: shape [2, 1]
    c = tl.arange(0, 2)[None, :]  # Column indices: shape [1, 2]

    # Load index tensor (shape [2]) from GM to UB
    idx = tl.load(index_ptr + tl.arange(0, 2))
    # Initialize empty 2x2 output tile in UB (default value: 0)
    dst = tl.full((2, 2), 0, dtype=tl.float32)

    # Invoke __builtin_index_select custom op to gather elements
    out_tile = al.custom("__builtin_index_select", src_ptr,  # Pointer to source tensor in GM
                         idx,  # Index tensor (in UB) for gathering
                         dim=0,  # Dimension to gather along
                         bound=4,  # Upper bound for valid index values (out-of-bound check)
                         end_offset=(2, 2),  # End offsets of each dimension for the index tensor
                         start_offset=(0, 0),  # Start offsets of each dimension for the source tensor
                         src_stride=(4, 1),  # Stride of each dimension for the source tensor in GM
                         out=dst  # Output tensor (in UB) to store gathered elements
                         )

    # Store the gathered tile from UB to output tensor in GM
    tl.store(out_ptr + r * 2 + c, out_tile)


if __name__ == "__main__":
    src = torch.tensor(
        [[10., 11., 12., 13.], [20., 21., 22., 23.], [30., 31., 32., 33.], [40., 41., 42., 43.]],
        device="npu",
        dtype=torch.float32,
    )
    index = torch.tensor([2, 0], device="npu", dtype=torch.int32)
    out = torch.empty((2, 2), device="npu", dtype=torch.float32)
    ref = torch.index_select(src, 0, index.to(torch.int64))[:, :2]
    builtin_index_select_kernel[(1, )](src, index, out)
    torch.testing.assert_close(out, ref)  # ref: [[30., 31.], [10., 11.]]
