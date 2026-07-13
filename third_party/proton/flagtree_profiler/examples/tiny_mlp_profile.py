#!/usr/bin/env python3
"""多算子 profiling 示例：一个两层 Tiny MLP。

这个例子用于演示 Proton scope 如何把多个 Triton kernel 组织成一个简单神经
网络调用树。它比 GPT 示例小，适合先解释多算子 profile 的基本结构。

运行：

    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/tiny_mlp_profile.py

输出文件：

    /tmp/flagtree_profiler_examples/tiny_mlp/profile.hatchet
    /tmp/flagtree_profiler_examples/tiny_mlp/profile.meta.json
    /tmp/flagtree_profiler_examples/tiny_mlp/profile.timeline.json
    /tmp/flagtree_profiler_examples/tiny_mlp/profile.vendor.json

查看结果：

    proton-viewer -m time/us,cann.task_duration_us,cann.aicore_time_us,cann.bandwidth_gb_s,cann.memory_access_bytes \
      /tmp/flagtree_profiler_examples/tiny_mlp/profile.hatchet

预期 Hatchet 树形结构大致如下，具体数值会随机器和 CANN 版本变化：

    ROOT
    └─ tiny_mlp
       ├─ 1_layer1
       │  └─ linear
       │     └─ _linear_kernel mix
       ├─ 2_activation
       │  └─ relu
       │     └─ _relu_kernel aiv
       └─ 3_layer2
          └─ linear
             └─ _linear_kernel mix

其中 `time` 和 `cann.task_duration_us` 会在父节点聚合；
`cann.bandwidth_gb_s` 是 CANN 原始带宽字段，不能跨父子节点直接相加，
所以父节点通常显示 NaN。
"""

from __future__ import annotations

import os
import pathlib
import shutil

import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def _linear_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + k[:, None] * N + offs_n[None, :],
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=mask)


@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def _load_torch_npu():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def _linear(x, weight, bias, out, m, n, k):
    block_m = 32
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )
    _linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )


def _relu(x, out, n):
    block = 1024
    grid = (triton.cdiv(n, block), )
    _relu_kernel[grid](x, out, n, BLOCK_SIZE=block)


def _tiny_mlp(x, weights, hidden, output, *, batch, input_dim, hidden_dim, output_dim):
    with proton.scope("tiny_mlp"):
        with proton.scope("1_layer1"):
            with proton.scope("linear"):
                _linear(x, weights["w1"], weights["b1"], hidden, batch, hidden_dim, input_dim)

        with proton.scope("2_activation"):
            with proton.scope("relu"):
                _relu(hidden, hidden, batch * hidden_dim)

        with proton.scope("3_layer2"):
            with proton.scope("linear"):
                _linear(hidden, weights["w2"], weights["b2"], output, batch, output_dim, hidden_dim)


def main() -> int:
    out_dir = pathlib.Path("/tmp/flagtree_profiler_examples/tiny_mlp")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_base = out_dir / "profile"

    torch = _load_torch_npu()
    device_id = 1
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    batch = 1024
    input_dim = 1024
    hidden_dim = 2048
    output_dim = 1024
    x = torch.randn((batch, input_dim), device=device, dtype=torch.float16)
    hidden = torch.empty((batch, hidden_dim), device=device, dtype=torch.float16)
    output = torch.empty((batch, output_dim), device=device, dtype=torch.float16)
    weights = {
        "w1": torch.randn((input_dim, hidden_dim), device=device, dtype=torch.float16),
        "b1": torch.randn((hidden_dim, ), device=device, dtype=torch.float16),
        "w2": torch.randn((hidden_dim, output_dim), device=device, dtype=torch.float16),
        "b2": torch.randn((output_dim, ), device=device, dtype=torch.float16),
    }

    _tiny_mlp(x, weights, hidden, output, batch=batch, input_dim=input_dim, hidden_dim=hidden_dim,
              output_dim=output_dim)
    torch.npu.synchronize()

    sid = proton.start(
        name=str(profile_base),
        context="shadow",
        data="tree",
        backend="cann",
        hook="triton",
        mode=("runtime_base:"
              f"device_id={device_id}:"
              "vendor_metrics=aicore,bandwidth:"
              "mstx_enabled=true:"
              "mstx_domain=proton"),
    )
    try:
        _tiny_mlp(x, weights, hidden, output, batch=batch, input_dim=input_dim, hidden_dim=hidden_dim,
                  output_dim=output_dim)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
