#!/usr/bin/env python3
"""中等复杂度 Triton 算子的 profiling 示例：row-wise softmax。

这个示例 profile 一个融合的逐行 softmax kernel。它比 vector add 更接近
真实算子，因为它包含行内 reduce、exp、sum、除法和 masked store。

在 FlagTree 昇腾容器内运行：

    source /usr/local/Ascend/cann-8.5.0/set_env.sh
    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/medium_softmax_profile.py

预期输出文件：

    /tmp/flagtree_profiler_examples/medium_softmax/profile.hatchet
    /tmp/flagtree_profiler_examples/medium_softmax/profile.meta.json
    /tmp/flagtree_profiler_examples/medium_softmax/profile.timeline.json
    /tmp/flagtree_profiler_examples/medium_softmax/profile.vendor.json

说明：默认不保留 CANN/msprof 原始目录。Profiler 会使用内部临时目录收集原始
CANN 数据，导入 profile.vendor.json 后自动清理。如果需要调试原始 CSV，可以设置
环境变量 PROTON_CANN_PROFILE_OUTPUT=/path/to/msprof。

预期数据：

    - vendor.json: association_sources 通常包含 aclprof_op_summary
    - vendor.json: 如果 CANN 导出 hbm/llc/mem CSV，会出现 bandwidth 指标
    - timeline.json: traceEvents 包含由 hook="triton" 生成的 softmax launch 范围
    - meta.json: 如果 task_time runtime event 不完整，degrade_reasons 会记录 host fallback

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
            "message": "_softmax_kernel aiv",
            "task_duration_us": 65.28
          },
          "runtime_event": {"device_id": 1, "op_name": "_softmax_kernel aiv"}
        },
        {
          "source": "msprof_bandwidth",
          "state": "collected",
          "metrics": {"bandwidth_gb_s": 0.0}
        }
      ]
    }

    profile.timeline.json:
    {
      "traceEvents": [
        {
          "cat": "kernel",
          "dur": 65.28,
          "name": "_softmax_kernel aiv",
          "ph": "X",
          "args": {"call_stack": ["ROOT", "_softmax_kernel aiv"]}
        }
      ]
    }

    profile.hatchet:
    [
      {
        "children": [
          {
            "frame": {"name": "_softmax_kernel", "type": "function"},
            "metrics": {
              "cann.task_duration_us": 73.608,
              "count": 10,
              "device_id": 1,
              "runtime.duration_us": 73.608,
              "time (ns)": 368040
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
def _softmax_kernel(x_ptr, out_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


def _load_torch_npu():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def main() -> int:
    out = pathlib.Path("/tmp/flagtree_profiler_examples/medium_softmax")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch = _load_torch_npu()
    torch.npu.set_device(1)
    device = torch.device("npu:1")

    rows = 1024
    cols = 512
    block = triton.next_power_of_2(cols)
    warmup = 2
    iters = 10
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    result = torch.empty_like(x)
    grid = (rows, )

    for _ in range(warmup):
        _softmax_kernel[grid](x, result, cols, BLOCK_SIZE=block)
    torch.npu.synchronize()

    sid = proton.start(
        name="/tmp/flagtree_profiler_examples/medium_softmax/profile",
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
            _softmax_kernel[grid](x, result, cols, BLOCK_SIZE=block)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
