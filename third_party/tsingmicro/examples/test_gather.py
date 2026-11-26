import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def gather_test_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
                       src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr,
                       idx_dim1: tl.constexpr, idx_stride0: tl.constexpr, idx_stride1: tl.constexpr,
                       out_dim0: tl.constexpr, out_dim1: tl.constexpr, out_stride0: tl.constexpr,
                       out_stride1: tl.constexpr):
    src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
    src = tl.load(src_ptr + src_offs)

    idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
    tl.store(out_ptr + out_offs, out)


@pytest.mark.interpreter
@pytest.mark.parametrize("src_shape, indices_shape, axis", [
    ([4, 4], [8, 4], 0),
    # ([128, 64], [256, 64], 0),
    # ([128, 64], [128, 128], 1),
])
def test_gather(src_shape, indices_shape, axis, device):

    def triton_gather(src: torch.Tensor, axis: int, indices: torch.Tensor):
        output = torch.empty(indices.shape, dtype=src.dtype, device=src.device)

        gather_test_kernel[(1, )](src, indices, output, axis, src.shape[0],
                                  src.shape[1], src.stride(0), src.stride(1), indices.shape[0], indices.shape[1],
                                  indices.stride(0), indices.stride(1), output.shape[0], output.shape[1],
                                  output.stride(0), output.stride(1))

        return output

    src = torch.randn(src_shape, device=device)
    indices = torch.randint(0, src.shape[axis], indices_shape, device=device)
    ref = torch.gather(src, axis, indices)
    result = triton_gather(src, axis, indices)
    torch.testing.assert_close(result, ref, rtol=0, atol=0)
