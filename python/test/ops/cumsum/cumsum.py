import math

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


@triton.jit(do_not_specialize=["part_num"])
def scan_part_sum_abc_kernel(
    inp,
    out,
    partial_sum,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    offset = a_idx * B * C + b_idx * C + c_idx
    base_part_offset = a_idx * part_num * C + c_idx
    part_offset = base_part_offset + pid_b * C

    mask = b_idx < B
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask)
    if (tl.constexpr(inp_vals.dtype.is_int64()) or tl.constexpr(inp_vals.dtype.is_uint64())) or tl.constexpr(
            inp_vals.dtype.is_fp64()):
        inp_vals = inp_vals
    elif tl.constexpr(inp_vals.dtype.is_int()):
        inp_vals = inp_vals.to(tl.int32)
    else:
        inp_vals = inp_vals.to(tl.float32)
    result = tl.cumsum(inp_vals, axis=0)

    part_sum_via_sum = tl.sum(inp_vals)

    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)

    partial_sum_ptrs = partial_sum + part_offset
    tl.store(partial_sum_ptrs, part_sum_via_sum)


@triton.jit(do_not_specialize=["part_num"])
def add_base_sum_abc_kernel(
    out,
    partial_sum,
    B,
    C,
    part_num,
    BLOCK_SIZE: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)

    a_idx = pid_a
    b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c_idx = pid_c

    base_offset = a_idx * B * C + c_idx
    offset = base_offset + b_idx * C
    base_part_offset = a_idx * part_num * C + c_idx
    last_part_offset = base_part_offset + (pid_b - 1) * C

    mask = b_idx < B
    out_ptrs = out + offset
    out_vals = tl.load(out_ptrs, mask=mask)

    if pid_b > 0:
        partial_sum_ptrs = partial_sum + last_part_offset
        last_part_sum_via_sum = tl.load(partial_sum_ptrs)

        final_vals = out_vals + last_part_sum_via_sum
        tl.store(out_ptrs, final_vals.to(out_vals.dtype), mask=mask)


def scan_then_fan(inp, out, A, B, C, dtype):
    # TODO(all): tune on target board
    BLOCK_SIZE = 1024
    if B <= 1024 * 4:
        BLOCK_SIZE = triton.next_power_of_2(B)
    part_num = math.ceil(B / BLOCK_SIZE)
    partial_sum = torch.empty(A, part_num, C, dtype=dtype, device=inp.device)

    grid = (A, part_num, C)
    with torch_device_fn.device(inp.device):
        scan_part_sum_abc_kernel[grid](inp, out, partial_sum, B, C, part_num, BLOCK_SIZE)

    if part_num >= 2:
        scan_then_fan(partial_sum, partial_sum, A, part_num, C, dtype)
        with torch_device_fn.device(inp.device):
            add_base_sum_abc_kernel[grid](out, partial_sum, B, C, part_num, BLOCK_SIZE)


def cumsum_wrapper(inp, dim=1, dtype=None, out=None):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = out.dtype
    if inp.dtype == torch.float16 or inp.dtype == torch.bfloat16:
        compute_dtype = torch.float32

    if M == 1 and K == 1:
        pass
    else:
        scan_then_fan(inp, out, M, N, K, compute_dtype)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    return cumsum_wrapper(inp, dim, dtype)


def check(name, ref, res, equal_nan=False, reduce_dim=1, atol=1e-4):
    RESOLUTION = {
        torch.bool: 0,
        torch.uint8: 0,
        torch.int8: 0,
        torch.int16: 0,
        torch.int32: 0,
        torch.int64: 0,
        torch.float8_e4m3fn: 1e-3,
        torch.float8_e5m2: 1e-3,
        torch.float8_e4m3fnuz: 1e-3,
        torch.float8_e5m2fnuz: 1e-3,
        torch.float16: 1e-3,
        torch.float32: 1.3e-6,
        torch.bfloat16: 0.016,
        torch.float64: 1e-7,
        torch.complex32: 1e-3,
        torch.complex64: 1.3e-6,
    }
    res = res.cpu()
    print(f"The maximum difference out {name} between torch and triton is "
          f"{torch.max(torch.abs(ref - res))}")
    rtol = RESOLUTION[ref.dtype]
    assert torch.allclose(res, ref, atol=atol * reduce_dim, rtol=rtol), (res, ref)


if __name__ == "__main__":
    # param
    shape = (2, 4096)
    dtype = torch.float32
    dim = 1

    # inp
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.cpu()

    # op
    ref_out = torch.cumsum(ref_inp, dim=dim)
    res_out = cumsum(inp, dim=dim)
    check("value", ref_out, res_out, reduce_dim=shape[dim])
