from collections import namedtuple

import torch
import triton
import triton.language as tl

# active driver
driver = triton.runtime.driver.active
# torch.cuda, torch.aipu, torch.npu
torch_device_fn = triton.runtime.driver.active.get_device_interface()
# device
if hasattr(driver, "get_active_torch_device"):
    device = triton.runtime.driver.active.get_active_torch_device()
else:
    device = triton.runtime.driver.active.get_current_device()


@triton.jit
def get_dtype_max(dtype: tl.constexpr):
    """get a value which is greater that all other values of that dtype"""
    # extract the tl.dtype from tl.constexpr so as to use its methods
    dtype_ = dtype.value
    if dtype_.is_floating():
        value: tl.constexpr = float("inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**(width - 1) - 1
        return value
    if dtype_.is_int_unsigned():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = 2**width - 1
        return value


@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M,  # 1
    N,  # 32
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    # 1. prepare offset
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dtype = inp.type.element_ty
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)

    # 2. for
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N + n_offset[None, :]
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)  #@hint: dma
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    # 3. store
    offset_index = m_offset
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def min_dim(inp, dim=None, keepdim=False):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N

    out_value = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), )
    with torch_device_fn.device(inp.device):
        min_kernel[grid](inp, out_value, out_index, M, N)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out


if __name__ == "__main__":
    # param
    shape = (1, 32)
    dim = 1
    keepdim = True

    # inp
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    ref_inp = inp.cpu()

    # op
    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = min_dim(inp, dim=dim, keepdim=keepdim)

    # check
    res_out_value = res_out_value.cpu()
    print(f'The maximum difference out value between torch and triton is '
          f'{torch.max(torch.abs(ref_out_value - res_out_value))}')
    assert torch.allclose(res_out_value, ref_out_value), (res_out_value, ref_out_value)
    res_out_index = res_out_index.cpu()
    print(f'The maximum difference out index between torch and triton is '
          f'{torch.max(torch.abs(ref_out_index - res_out_index))}')
    assert torch.allclose(res_out_index, ref_out_index), (res_out_index, ref_out_index)
