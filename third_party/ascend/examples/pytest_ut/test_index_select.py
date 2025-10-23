import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common

# shape： same as load, not scalar
# dim [0,len(shape)-1]
# indice_shape: 1D tensor, 
# dtype： same as load
# profiling: 0.6x AscendC

# known issue: runtime error when src_tensor.stride(dim) is not 32B aligned
@pytest.mark.parametrize("src_shape, dim, indice_shape, dtype", [
    ((500000, 240), 0, (375144,), "float32"), # standard 1984us +- 5us fp32 48core
    # ((500000, 37), 0, (324344,), "float32"), # standard 266us +- 10us fp32 48core 
    ((3200, 16), 0, (1971940,), "float32"), # standard 1104us +- fp32
    # ((1971940, 2), 1, (1,), "int32"), #  4678us +- 129us INT32
    # ((3200, 1, 37), 0, (1022226,), "float32"), # 1965us +- 65us fp32 
    # ((323357, 37), 0, (1022226,), "float32"), # 3500us +- 50us fp32 
    ((480000, 16), 0, (3943880,), "float32"), # 3636 +- 42us fp32
    ((480000, 32), 0, (3943880,), "float32"), # 3678 +- 5us fp32
    ((480000, 64), 0, (3943880,), "float32"), # 5392 +- 20us fp32
    ((378103, 240), 0, (1992337,), "float32"), # 10000us fp32
    ((374035, 240), 0, (1971940,), "float32"), # 10100us fp32
    ((2000000, 32), 0, (270244,), "float32"), # 155us fp32s
    ((1000000, 8), 0, (21329,), "float32"), # 44us fp32
    ((270098, 32), 0, (512,), "float32"), # 40us fp32
    ((20669, 8), 0, (2048,), "float32"), # 31us fp32
    ((10, 16), 0, (1024,), "float32"), # 27us fp32
    ((270610, 32), 0, (2928000,), "float32"), # 553us fp32 
    ((22717, 8), 0, (278400,), "float32"), # 76us fp32
    ((1034, 16), 0, (48000,), "float32"), # 38us fp32
])
def test_index_select(src_shape, dim, indice_shape, dtype):

    def torch_func(x0, dim, indices):
        res = torch.index_select(x0, dim, indices)
        return res

    @triton.jit
    def basic_index_select(in_ptr, indices_ptr, out_ptr, dim,
        g_stride: tl.constexpr, indice_length: tl.constexpr, 
        g_block : tl.constexpr, g_block_sub: tl.constexpr, other_block:tl.constexpr):
        g_begin=tl.program_id(0) * g_block
        for goffs in range(0, g_block, g_block_sub):
            g_idx=tl.arange(0, g_block_sub) + g_begin + goffs
            g_mask = g_idx < indice_length
            indices = tl.load(indices_ptr + g_idx, g_mask, other=0)
            for other_offset in range(0, g_stride, other_block): 
                tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
                other_idx = tl.arange(0, other_block) + other_offset
                other_mask = other_idx < g_stride
                for i in range(0, g_block_sub):
                    gather_offset = tl.get_element(indices, (i,)) * g_stride
                    val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, other_block), strides=(g_stride, 1))
                tl.store(out_ptr + g_idx[:,None] * g_stride + other_idx[None,:], tmp_buf, g_mask[:,None] & other_mask[None,:])

    def triton_func(x0, dim, indices):
        sz = list(x0.shape)
        sz[dim]=len(indices)
        out = torch.empty(tuple(sz), dtype=x0.dtype).npu()
        g_stride = x0.stride(dim)
        indice_length=indices.numel()
        num_vec_core=40
        g_block = (indice_length - 1) // num_vec_core + 1
        enable_multi_buffer=True
        available_ub_space = (125 * 1024) // (x0.element_size() * (2 if enable_multi_buffer else 1))
        if g_stride * 2 < available_ub_space:
            other_block = g_stride
            g_block_sub = available_ub_space // other_block
        else:
            other_block = available_ub_space
            g_block_sub = 1
        basic_index_select[num_vec_core, 1, 1](x0, indices, out, dim, g_stride = g_stride, indice_length=indice_length, 
        g_block = g_block, g_block_sub = g_block_sub, other_block = other_block, multibuffer=False)
        return out

    x0 = test_common.generate_tensor(shape=src_shape, dtype=dtype).npu()
    indices = torch.randint(0, src_shape[dim], size=indice_shape, dtype=torch.int32).npu()

    torch_ref = torch_func(x0, dim, indices)
    triton_cal = triton_func(x0, dim, indices)
    assert torch.equal(torch_ref, triton_cal)
