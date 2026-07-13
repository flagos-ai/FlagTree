#!/usr/bin/env python3
"""Timeline 展示示例：多 block 的小型 MLP workload。

这个例子专门用于演示 `profile.timeline.json`。它会直接运行，不需要传参：

    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/timeline_showcase_profile.py

输出文件：

    /tmp/flagtree_profiler_examples/timeline_showcase/profile.hatchet
    /tmp/flagtree_profiler_examples/timeline_showcase/profile.meta.json
    /tmp/flagtree_profiler_examples/timeline_showcase/profile.timeline.json
    /tmp/flagtree_profiler_examples/timeline_showcase/profile.vendor.json

这个 workload 有 4 个 block，每个 block 包含 5 个 Triton kernel：

    linear_up -> relu_square -> linear_down -> residual_add -> layer_norm

预期 timeline:

    - traceEvents 中有 20 个左右的 kernel 事件；
    - 每个事件的 args.call_stack 包含 block 和 phase，例如
      ["ROOT", "timeline_showcase", "block_00", "1_linear_up", "_linear_kernel mix"]；
    - 每个事件 args 顶层包含 CANN 指标，例如 cann.task_duration_us、
      cann.op_summary_task_duration_us、cann.task_wait_time_us、
      cann.bandwidth_gb_s、cann.memory_access_bytes；
    - 适合用 Perfetto UI 或 Chrome trace viewer 打开查看连续算子时间线。

查看 Hatchet：

    proton-viewer -m time/us,cann.task_duration_us,cann.task_wait_time_us,cann.bandwidth_gb_s,cann.memory_access_bytes \
      /tmp/flagtree_profiler_examples/timeline_showcase/profile.hatchet

查看 timeline 基本信息：

    python3 - <<'PY'
    import json
    p = "/tmp/flagtree_profiler_examples/timeline_showcase/profile.timeline.json"
    trace = json.load(open(p))
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
    print("kernel events:", len(events))
    print(events[0]["name"])
    print(events[0]["args"].keys())
    print(events[0]["args"].get("call_stack"))
    PY
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
def _relu_square_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    relu = tl.maximum(x, 0.0)
    y = relu * relu
    tl.store(y_ptr + offsets, y.to(tl.float16), mask=mask)


@triton.jit
def _residual_add_kernel(x_ptr, residual_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + residual, mask=mask)


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < COLS
    x = tl.load(x_ptr + row * COLS + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / COLS
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / COLS
    inv_std = tl.rsqrt(variance + 1.0e-5)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * scale + bias
    tl.store(y_ptr + row * COLS + offsets, y.to(tl.float16), mask=mask)


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


def _elementwise(kernel, x, y, n):
    block = 1024
    grid = (triton.cdiv(n, block), )
    kernel[grid](x, y, n, BLOCK_SIZE=block)


def _residual_add(x, residual, out, n):
    block = 1024
    grid = (triton.cdiv(n, block), )
    _residual_add_kernel[grid](x, residual, out, n, BLOCK_SIZE=block)


def _layer_norm(x, scale, bias, out, rows, cols):
    block = triton.next_power_of_2(cols)
    _layer_norm_kernel[(rows, )](x, scale, bias, out, rows, cols, BLOCK_SIZE=block)


def _run_block(x, weights, buffers, block_index, *, batch, hidden_dim, intermediate_dim):
    block_name = f"block_{block_index:02d}"
    with proton.scope(block_name):
        with proton.scope("1_linear_up"):
            _linear(
                x,
                weights["up_w"],
                weights["up_b"],
                buffers["up"],
                batch,
                intermediate_dim,
                hidden_dim,
            )

        with proton.scope("2_relu_square"):
            _elementwise(_relu_square_kernel, buffers["up"], buffers["act"], batch * intermediate_dim)

        with proton.scope("3_linear_down"):
            _linear(
                buffers["act"],
                weights["down_w"],
                weights["down_b"],
                buffers["down"],
                batch,
                hidden_dim,
                intermediate_dim,
            )

        with proton.scope("4_residual_add"):
            _residual_add(buffers["down"], x, buffers["residual"], batch * hidden_dim)

        with proton.scope("5_layer_norm"):
            _layer_norm(
                buffers["residual"],
                weights["ln_w"],
                weights["ln_b"],
                buffers["norm"],
                batch,
                hidden_dim,
            )

    return buffers["norm"]


def _run_model(x, weights, buffers, *, blocks, batch, hidden_dim, intermediate_dim):
    with proton.scope("timeline_showcase"):
        for block_index in range(blocks):
            x = _run_block(
                x,
                weights,
                buffers,
                block_index,
                batch=batch,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
            )
    return x


def main() -> int:
    out_dir = pathlib.Path("/tmp/flagtree_profiler_examples/timeline_showcase")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_base = out_dir / "profile"

    torch = _load_torch_npu()
    device_id = 1
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")

    blocks = 4
    batch = 512
    hidden_dim = 512
    intermediate_dim = 1024
    dtype = torch.float16

    torch.manual_seed(2026)
    x = torch.randn((batch, hidden_dim), device=device, dtype=dtype)
    weights = {
        "up_w": torch.randn((hidden_dim, intermediate_dim), device=device, dtype=dtype),
        "up_b": torch.randn((intermediate_dim, ), device=device, dtype=dtype),
        "down_w": torch.randn((intermediate_dim, hidden_dim), device=device, dtype=dtype),
        "down_b": torch.randn((hidden_dim, ), device=device, dtype=dtype),
        "ln_w": torch.ones((hidden_dim, ), device=device, dtype=dtype),
        "ln_b": torch.zeros((hidden_dim, ), device=device, dtype=dtype),
    }
    buffers = {
        "up": torch.empty((batch, intermediate_dim), device=device, dtype=dtype),
        "act": torch.empty((batch, intermediate_dim), device=device, dtype=dtype),
        "down": torch.empty((batch, hidden_dim), device=device, dtype=dtype),
        "residual": torch.empty((batch, hidden_dim), device=device, dtype=dtype),
        "norm": torch.empty((batch, hidden_dim), device=device, dtype=dtype),
    }

    _run_model(
        x,
        weights,
        buffers,
        blocks=blocks,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
    )
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
        _run_model(
            x,
            weights,
            buffers,
            blocks=blocks,
            batch=batch,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
