# this is a benchmark which multiplies square matrices with maximum block size
# to check the performance of tl.dot operation
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import triton
import triton.language as tl
import benchmark

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def prune_invalid_configs(configs, named_args, **kwargs):
    """过滤掉会导致 tensor 过大的配置"""
    MAX_NUMEL = 1048576
    m = named_args.get("M")
    n = named_args.get("N")
    k = named_args.get("K")
    valid_configs = []
    for config in configs:
        block_size = config.kwargs.get("BLOCK_SIZE", 64)
        if block_size * block_size > MAX_NUMEL:
            continue
        # Kernel has no boundary mask; BLOCK_SIZE must not exceed the tensor.
        if m is not None and block_size > m:
            continue
        if n is not None and block_size > n:
            continue
        if k is not None and block_size > k:
            continue
        valid_configs.append(config)
    return valid_configs if valid_configs else [configs[0]]


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 64}, num_stages=2, num_warps=1),
        triton.Config(kwargs={"BLOCK_SIZE": 64}, num_stages=2, num_warps=1),
    ],
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
    warmup=1,
    rep=1,
)
@triton.jit
def bare_matmul(X, Y, Z, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_x = tl.program_id(0)  # block row id
    pid_y = tl.program_id(1)  # block column id

    offs_x = pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_y = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + offs_x[:, None] * K + offs_y[None, :])
    y = tl.load(Y + offs_x[:, None] * N + offs_y[None, :])

    z = tl.dot(x, y)

    tl.store(Z + offs_x[:, None] * N + offs_y[None, :], z)


# @benchmark.measure()
def bench_matmul(M, N, K):
    dtype = torch.float32
    a = torch.randint(0, 10, (M, K), dtype=torch.int32).to(dtype)
    b = torch.randint(0, 10, (K, N), dtype=torch.int32).to(dtype)

    # CPU: a, b stay on host
    c_ref = torch.matmul(a, b)

    # TX81: H2D, launch, sync, D2H
    a_dev = a.to(DEVICE)
    b_dev = b.to(DEVICE)
    c_dev = torch.empty((M, N), device=DEVICE, dtype=dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    bare_matmul[grid](a_dev, b_dev, c_dev, M, N, K)
    torch.txda.synchronize()
    c_act = c_dev.cpu()

    print(bare_matmul.best_config)
    print(f"dev sample[:4,:4]=\n{c_act[:4, :4]}")
    print(f"cpu sample[:4,:4]=\n{c_ref[:4, :4]}")
    print(f"max_abs_diff={(c_act - c_ref).abs().max().item()}")
    torch.testing.assert_close(c_act, c_ref, atol=1e-2, rtol=1e-2)
    print("allclose PASS")


if __name__ == "__main__":
    bench_matmul(64, 64, 64)
