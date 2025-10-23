import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice
import numpy as np
import test_common
import pytest

@pytest.mark.skip(reason="waiting for the compiler to support.")
@pytest.mark.parametrize("src_shape, indices_shape, axis", [
    ([4, 4], [8, 4], 0),
    ([4, 64], [4, 32], 1),
    ([128, 64], [128, 128], 1),
])
def test_gather(src_shape, indices_shape, axis):

    @triton.jit
    def gather_kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
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

    def triton_gather(src: torch.Tensor, axis: int, indices: torch.Tensor):
        output = torch.empty(indices.shape, dtype=src.dtype, device=src.device)
        gather_kernel[(1, )](src, indices, output, axis,
                             src.shape[0], src.shape[1],
                             src.stride(0), src.stride(1),
                             indices.shape[0], indices.shape[1],
                             indices.stride(0), indices.stride(1),
                             output.shape[0], output.shape[1],
                             output.stride(0), output.stride(1))
        return output

    DEV = "npu"
    src = torch.randn(src_shape, device=DEV)
    indices = torch.randint(0, src.shape[axis], indices_shape, device=DEV)
    ref = torch.gather(src, axis, indices)
    result = triton_gather(src, axis, indices)
    torch.testing.assert_close(result, ref, rtol=0, atol=0)

@pytest.mark.parametrize('param_list',
                         [
                             ['float16', (11, 12, 256, 512), 48],
                         ])
def test_gather_flip(param_list):

    def torch_func(inp, idx):
        return torch.gather(input=inp, dim=-1, index=idx)

    @triton.jit
    def triton_kernel(dst_ptr, src_ptr, idx_ptr,
                    XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                    R0_BLOCK: tl.constexpr, R1_BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        poff = pid * XBLOCK
        x0_idx_base = 0
        r1_idx = tl.arange(0, R1_BLOCK)
        loop0 = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
        for xsub_id in tl.range(loop0):
            x0_idx = poff + xsub_id * XBLOCK_SUB + x0_idx_base
            idx_idx = idx_ptr + x0_idx * R1_BLOCK + r1_idx
            idx_blk = tl.load(idx_idx)
            idx_min = tl.min(idx_blk, axis=0)
            src_idx = src_ptr + x0_idx * R0_BLOCK + idx_min + r1_idx
            src_blk = tl.load(src_idx)
            fliped_blk = libdevice.flip(src_blk, 0)
            dst_idx = dst_ptr + x0_idx * R1_BLOCK + r1_idx
            tl.store(dst_idx, fliped_blk)

    def triton_func(p2c_out, p2c_att, p2c_pos, ncore):
        nrows = p2c_att.shape[0] * p2c_att.shape[1] * p2c_att.shape[2]
        xs = nrows // ncore
        assert(xs * ncore == nrows)
        xss = 1 # must be 1
        r0s = p2c_att.shape[3]
        r1s = p2c_att.shape[2]
        triton_kernel[ncore, 1, 1](p2c_out, p2c_att, p2c_pos,
                    XBLOCK=xs, XBLOCK_SUB=xss,
                    R0_BLOCK=r0s, R1_BLOCK=r1s)
        return p2c_out

    dtype, shape, ncore = param_list
    M0, M1, N0, N1 = shape
    r0 = torch.arange(N0)
    c0 = torch.arange(N0)
    p2c_pos = r0[:, None] - c0[None, :] + N0-1
    p2c_pos = p2c_pos.broadcast_to((M0, M1, N0, N0))
    p2c_pos = p2c_pos.npu()
    if (p2c_pos.dtype == torch.int64):
        p2c_pos = p2c_pos.to(torch.int32)
    assert(np.all(np.diff(p2c_pos.cpu()) == -1))
    p2c_att = test_common.generate_tensor(shape, dtype).npu()
    p2c_out = test_common.generate_tensor(p2c_pos.shape, dtype).npu()

    p2c_ref = torch_func(p2c_att, p2c_pos)
    triton_func(p2c_out, p2c_att, p2c_pos, ncore)
    test_common.validate_cmp(dtype, p2c_out, p2c_ref)

if __name__ == "__main__":
    param_list = ['float16', (11, 12, 256, 512), 48]
    test_gather_flip(param_list)
    print("success: test_gather_flip")
    test_gather([4, 64], [4, 32], 1)
    print("success: test_gather")