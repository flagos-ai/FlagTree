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
def get_dtype_min(dtype):
    """get a value which is less that all other values of that dtype"""
    dtype_ = dtype.value  # tl.dtype
    if dtype_.is_floating():
        value: tl.constexpr = float("-inf")
        return value
    if dtype_.is_int_signed():
        width: tl.constexpr = dtype_.int_bitwidth
        value: tl.constexpr = -1 * 2**(width - 1)
        return value
    if dtype_.is_int_unsigned():
        value: tl.constexpr = 0
        return value


@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    # Map the program id to the row of inp it should compute.
    workers = tl.num_programs(0)
    pid = tl.program_id(0)

    total_workloads = tl.cdiv(M, BLOCK_M)
    workloads = tl.cdiv(total_workloads, workers)

    for w in range(workloads):
        work_id = pid + w * workers
        rows = work_id * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        ninp = inp + rows * N
        nout = out + rows
        row_mask = rows < M

        acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
        _all = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(ninp + cols, mask, other=min_value)
            _all = tl.maximum(_all, a)
        all = tl.max(_all, axis=1)[:, None]
        tl.store(nout, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    if dim is not None:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), )
        with torch_device_fn.device(inp.device):
            amax_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


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
    keepdim = True
    dim = 1
    shape = (1, 32)
    dtype = torch.float32

    # inp
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = inp.cpu()

    # op
    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    res_out = amax(inp, dim=dim, keepdim=keepdim)
    check("value", ref_out, res_out)
