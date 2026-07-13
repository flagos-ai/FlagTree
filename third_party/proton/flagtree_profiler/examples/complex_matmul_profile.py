#!/usr/bin/env python3
"""复杂 Triton 算子的 profiling 示例：tiled fp16 matmul。

这个示例 profile 一个小型 `tl.dot` 矩阵乘。它覆盖 K 维 tile 循环、
block pointer arithmetic、多次 load、矩阵累加和 fp16 输出 store。

在 FlagTree 昇腾容器内运行：

    source /usr/local/Ascend/cann-8.5.0/set_env.sh
    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/complex_matmul_profile.py

预期输出文件：

    /tmp/flagtree_profiler_examples/complex_matmul/profile.hatchet
    /tmp/flagtree_profiler_examples/complex_matmul/profile.meta.json
    /tmp/flagtree_profiler_examples/complex_matmul/profile.timeline.json
    /tmp/flagtree_profiler_examples/complex_matmul/profile.vendor.json

说明：默认不保留 CANN/msprof 原始目录。Profiler 会使用内部临时目录收集原始
CANN 数据，导入 profile.vendor.json 后自动清理。如果需要调试原始 CSV，可以设置
环境变量 PROTON_CANN_PROFILE_OUTPUT=/path/to/msprof。

预期数据：

    - vendor.json: 包含 CANN op summary 中的 matmul/DSA kernel 行
    - vendor.json: CANN 导出 op_summary/task_time 时包含 AICore 和 timing 指标
    - vendor.json: CANN 导出 hbm/llc/mem CSV 时包含 bandwidth association
    - timeline.json: traceEvents 包含 hook="triton" 自动生成的 Triton launch scope
    - hatchet: 包含 CANN 原生导入的 task duration、AICore、cycles、bandwidth
      和 memory access 等字段

说明：`_matmul_kernel mix` 不是第二个用户 kernel。它是 Ascend/CANN 对 Triton
kernel 的 mix_mode 标记，表示该 kernel 走 mix 执行模式。CANN op_summary 中
的 `_matmul_kernel` AI_CORE task 会按名字和时间戳合并到同一个 Hatchet 节点，
避免用户看到两个 kernel。

输出文件内容节选：

    profile.meta.json:
    {
      "backend": "cann",
      "context": "shadow",
      "data": "tree",
      "device": {"id": 1, "name": "Ascend910B4-1", "type": "ASCEND"},
      "hook": "triton",
      "profiler_name": "cann",
      "runtime_base_enabled": true,
      "vendor_metrics_enabled": ["aicore", "bandwidth"],
      "config": {"aclprof_output_cleanup": "removed", "aclprof_output_retained": "false"},
      "degrade_reasons": [
        "task_time runtime events unavailable; using host runtime events for op_summary correlation..."
      ]
    }

    profile.vendor.json:
    {
      "backend": "cann",
      "raw_inputs": [{"kind": "csv", "path": ".../mindstudio_profiler_output/op_summary_*.csv"}],
      "associations": [
        {
          "source": "msprof_mstx",
          "state": "collected",
          "metrics": {
            "domain": "proton",
            "message": "_matmul_kernel mix",
            "task_duration_us": 66.816
          },
          "runtime_event": {"device_id": 1, "op_name": "_matmul_kernel mix"}
        },
        {
          "source": "aclprof_op_summary",
          "state": "collected",
          "metrics": {"op_type": "_matmul_kernel"}
        }
      ]
    }

    profile.timeline.json:
    {
      "traceEvents": [
        {
          "cat": "kernel",
          "dur": 66.816,
          "name": "_matmul_kernel mix",
          "ph": "X",
          "args": {"call_stack": ["ROOT", "_matmul_kernel mix"]}
        }
      ]
    }

    profile.hatchet:
    [
      {
        "children": [
          {
            "frame": {"name": "_matmul_kernel mix", "type": "function"},
            "metrics": {
              "cann.aicore_time_us": 2.458,
              "cann.bandwidth_gb_s": 333.526,
              "cann.op_summary_task_duration_us": 3.46,
              "cann.task_duration_us": 66.816,
              "count": 10,
              "device_id": 1,
              "runtime.duration_us": 66.816,
              "time (ns)": 39119
            }
          }
        ],
        "frame": {"name": "ROOT", "type": "function"}
      }
    ]

说明：具体耗时、run_id、PROF_* 目录名、association 数量会随机器、CANN 版本
和迭代次数变化。完整 vendor.json 和 CANN CSV 通常较大，这里只展示关键字段。
"""

from __future__ import annotations

import os
import pathlib
import shutil

import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
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
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=mask)


def _load_torch_npu():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def main() -> int:
    out = pathlib.Path("/tmp/flagtree_profiler_examples/complex_matmul")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch = _load_torch_npu()
    torch.npu.set_device(1)
    device = torch.device("npu:1")

    m = 256
    n = 256
    k = 256
    block_m = 64
    block_n = 64
    block_k = 64
    warmup = 2
    iters = 10
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    result = torch.empty((m, n), device=device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), )

    for _ in range(warmup):
        _matmul_kernel[grid](a, b, result, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    torch.npu.synchronize()

    sid = proton.start(
        name="/tmp/flagtree_profiler_examples/complex_matmul/profile",
        context="shadow",
        data="tree",
        backend="cann",
        hook="triton",
        mode=("runtime_base:"
              "device_id=1:"
              "vendor_metrics=aicore,bandwidth:"
              "mstx_enabled=true:"
              "mstx_domain=proton"),
    )
    try:
        for _ in range(iters):
            _matmul_kernel[grid](a, b, result, m, n, k, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
