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
def sum_dim_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    if tl.constexpr(inp.dtype.element_ty == tl.float16) or tl.constexpr(inp.dtype.element_ty == tl.bfloat16):
        cdtype = tl.float32
    else:
        cdtype = inp.dtype.element_ty

    # 1. prepare offset
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + pid * N
    out = out + pid
    row_mask = pid < M
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)

    # 2. for
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=0).to(cdtype)
        _sum += a

    # 3. store
    sum = tl.sum(_sum, axis=1)[:, None]
    tl.store(out, sum, row_mask)


def sum_dim_comm(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64

    if dim == []:
        pass

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1 or len(dim) > 1:
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N
        if out is None:
            out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), )
        with torch_device_fn.device(inp.device):
            sum_dim_kernel[grid](inp, out, M, N)
        if not keepdim:
            out = out.squeeze(dim=dim)
        return out


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)


if __name__ == "__main__":
    # param
    shape = (1, 32)
    dim = [1]
    keepdim = True

    # inp
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    ref_inp = inp.cpu()

    # op
    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    res_out = sum_dim(inp, dim=dim, keepdim=keepdim)

    # check
    res_out = res_out.cpu()
    print(f"The maximum difference out value between torch and triton is "
          f"{torch.max(torch.abs(ref_out - res_out))}")
    assert torch.allclose(res_out, ref_out), (res_out, ref_out)
