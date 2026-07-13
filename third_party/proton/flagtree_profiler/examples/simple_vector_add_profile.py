#!/usr/bin/env python3
"""简单 Triton 算子的 profiling 示例：vector add。

在 FlagTree 昇腾容器内运行：

    source /usr/local/Ascend/cann-8.5.0/set_env.sh
    cd /workspace/FlagTree
    python3 third_party/proton/flagtree_profiler/examples/simple_vector_add_profile.py

预期输出文件：

    /tmp/flagtree_profiler_examples/simple_vector_add/profile.hatchet
    /tmp/flagtree_profiler_examples/simple_vector_add/profile.meta.json
    /tmp/flagtree_profiler_examples/simple_vector_add/profile.timeline.json
    /tmp/flagtree_profiler_examples/simple_vector_add/profile.vendor.json

说明：默认不保留 CANN/msprof 原始目录。Profiler 会使用内部临时目录收集原始
CANN 数据，导入 profile.vendor.json 后自动清理。如果需要调试原始 CSV，可以设置
环境变量 PROTON_CANN_PROFILE_OUTPUT=/path/to/msprof。

预期数据：

    - meta.json: backend == "cann"，hook == "triton"，runtime_base_enabled == true
    - vendor.json: raw_inputs 包含 op_summary/task_time/msprof_tx 等 CANN CSV
    - vendor.json: association_sources 通常包含 aclprof_op_summary 和 msprof_mstx
    - timeline.json: traceEvents 包含 Proton/CANN timeline events

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
            "message": "_vector_add_kernel aiv",
            "task_duration_us": 71.424
          },
          "runtime_event": {"device_id": 1, "op_name": "_vector_add_kernel aiv"}
        }
      ]
    }

    profile.timeline.json:
    {
      "traceEvents": [
        {
          "cat": "kernel",
          "dur": 71.424,
          "name": "_vector_add_kernel aiv",
          "ph": "X",
          "args": {"call_stack": ["ROOT", "_vector_add_kernel aiv"]}
        }
      ]
    }

    profile.hatchet:
    [
      {
        "children": [
          {
            "frame": {"name": "_vector_add_kernel", "type": "function"},
            "metrics": {
              "cann.task_duration_us": 71.424,
              "count": 10,
              "device_id": 1,
              "runtime.duration_us": 71.424,
              "time (ns)": 368500
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
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _load_torch_npu():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is false")
    return torch


def main() -> int:
    out = pathlib.Path("/tmp/flagtree_profiler_examples/simple_vector_add")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    torch = _load_torch_npu()
    torch.npu.set_device(1)
    device = torch.device("npu:1")

    n = 1_048_576
    block = 1024
    warmup = 2
    iters = 10
    x = torch.randn((n, ), device=device, dtype=torch.float32)
    y = torch.randn((n, ), device=device, dtype=torch.float32)
    result = torch.empty_like(x)
    grid = (triton.cdiv(n, block), )

    for _ in range(warmup):
        _vector_add_kernel[grid](x, y, result, n, BLOCK_SIZE=block)
    torch.npu.synchronize()

    sid = proton.start(
        name="/tmp/flagtree_profiler_examples/simple_vector_add/profile",
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
            _vector_add_kernel[grid](x, y, result, n, BLOCK_SIZE=block)
        torch.npu.synchronize()
    finally:
        proton.finalize(sid)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
