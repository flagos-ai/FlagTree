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


@triton.jit(do_not_specialize=["correction"])
def var_mean_welford_kernel(
    X,
    Var,
    Mean,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr = 4,
    BLOCK_N: tl.constexpr = 64,
):
    # Map the program id to the row of X it should compute.
    pid = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    Mean = Mean + pid
    row_mask = pid < M

    _mean = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    _count = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)

        count = _count + mask
        cnt = tl.maximum(count, 1)
        cur_mean = (_mean * _count + x) / cnt
        _acc += (x - cur_mean) * (x - _mean) * mask
        _mean = cur_mean
        _count = count

    # 手动实现 tl.reduce 的功能，沿着 axis=1 进行归约
    # 使用 tl.sum 来进行归约，这等价于 welford 算法在这种情况下的行为

    # 计算每行的总计数
    total_count = tl.sum(_count, axis=1)  # shape: (BLOCK_M,)

    # 计算加权平均值
    weighted_sum = tl.sum(_mean * _count, axis=1)  # shape: (BLOCK_M,)
    mean = weighted_sum / tl.maximum(total_count, 1)  # shape: (BLOCK_M,)

    # 计算方差累积值
    # 对于每个元素，计算其对总体方差的贡献
    mean_expanded = mean[:, None]  # shape: (BLOCK_M, 1)

    # 计算每个局部统计量对总体方差的贡献
    # 这是 Welford 算法的并行化版本
    local_var_contrib = _acc + _count * (_mean - mean_expanded) * (_mean - mean_expanded)
    acc = tl.sum(local_var_contrib, axis=1)  # shape: (BLOCK_M,)

    var = acc / (N - correction)
    mean = mean[:, None]
    var = var[:, None]

    # Write mean / var
    tl.store(Mean, mean, row_mask)
    tl.store(Var, var, row_mask)


def var_mean(x, dim=None, *, correction=None, keepdim=False):
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        assert False
    else:
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        mean = torch.empty(shape, dtype=x.dtype, device=x.device)

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), )
        with torch_device_fn.device(x.device):
            var_mean_welford_kernel[grid](x, var, mean, M, N, correction)

    if not keepdim:
        var = var.squeeze(dim=dim)
        mean = mean.squeeze(dim=dim)
    return var, mean


if __name__ == "__main__":
    # param
    shape = (2, 32)
    dim = [1]
    correction = 1
    keepdim = True

    # inp
    inp = torch.randn(shape, dtype=torch.float32, device=device)
    ref_inp = inp.cpu()

    # op
    ref_var, ref_mean = torch.var_mean(ref_inp, dim, correction=correction, keepdim=keepdim)
    res_var, res_mean = var_mean(inp, dim, correction=correction, keepdim=keepdim)

    # check
    res_var = res_var.cpu()
    print(f"The maximum difference var between torch and triton is "
          f"{torch.max(torch.abs(ref_var - res_var))}")
    assert torch.allclose(res_var, ref_var), (res_var, ref_var)
    res_mean = res_mean.cpu()
    print(f"The maximum difference mean between torch and triton is "
          f"{torch.max(torch.abs(ref_mean - res_mean))}")
    assert torch.allclose(res_mean, ref_mean), (res_mean, ref_mean)
