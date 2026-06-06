#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def test_gather_load_kernel(src_ptr, index_ptr, out_ptr):
    # index tile shape: (2, 2)
    cols = tl.arange(0, 2)[None, :]  # [[0, 1]]
    rows = tl.arange(0, 2)[:, None]  # [[0],[1]]
    mask = (rows < 2) & (cols < 2)

    # load index tile to UB
    index = tl.load(index_ptr + rows * 2 + cols, mask)

    # gather load from GM to UB
    dst = tl.full(index.shape, 0, tl.float32)
    gathered = al.custom("__builtin_gather_load", src_ptr, index, bound=4, dim=0, src_stride=(2, 1), index_shape=(2, 2),
                         offsets=(0, 0), out=dst)

    # store result to GM
    tl.store(out_ptr + rows * 2 + cols, gathered, mask)


if __name__ == "__main__":
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device='npu')
    index = torch.tensor([[0, 1], [2, 3]], device='npu')
    out = torch.empty((2, 2), device='npu', dtype=torch.float32)
    test_gather_load_kernel[(1, )](src, index, out)
    print("result: ", out)  # [[1., 4.], [5., 8.]]
