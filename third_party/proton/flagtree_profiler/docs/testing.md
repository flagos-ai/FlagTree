# FlagTree Profiler 测试指南

本文说明如何测试 FlagTree Profiler 的 CANN 后端。当前后端只接入昇腾 CANN。

## 环境准备

在容器或机器中先加载 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_VISIBLE_DEVICES=0
```

确认基础依赖存在。默认 IR instrumentation 路径不要求 `msprof`；如果要跑 CANN
legacy 或 CSV import 验证，再检查 `which msprof`。

```bash
python3 - <<'PY'
import torch
import torch_npu
import triton
print("torch:", torch.__version__)
print("npu available:", torch.npu.is_available())
print("triton:", triton.__version__)
PY
which msprof  # only required by legacy CANN/msprof tests
```

## 1. 默认 smoke 测试

这是默认自动化测试，包含 CSV import 单元测试和真实 NPU direct-finalize 测试。

```bash
python3 -m pytest -q third_party/proton/flagtree_profiler/test/test_cann_smoke.py -s
```

预期结果：

```text
12 passed
```

这个测试验证：

- `backend="cann"` 可以通过 `proton.start()` / `proton.finalize()` 正常工作。
- CANN CSV import 能解析 op summary、MSTX、bandwidth。
- bandwidth 可以从 op summary byte counters 或 supplemental CSV 中获得。
- 真实 Triton kernel 可以通过 `hook="triton"` 被 profile。
- `finalize()` 后直接生成 `vendor.json`、`timeline.json`、`meta.json`。

兼容旧路径的命令仍可收集同一组测试：

```bash
python3 -m pytest -q third_party/proton/test/test_cann_smoke.py -s
```

## 2. 统一 profiler suite

`scripts/cann_profile_test_suite.py` 是测试的唯一推荐入口。默认只运行项目内自定义的 12 个 Triton kernel，覆盖 elementwise、activation、math、memory、cast、reduction、softmax、transpose、matmul、masking。

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests \
  --clean
```

如果只想跑其中一个算子：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests_one \
  --clean \
  --custom-operator triton_vector_add_fp32
```

输出：

```text
/tmp/proton_cann_tests/summary.json
/tmp/proton_cann_tests/custom/summary.json
/tmp/proton_cann_tests/custom/post_import.vendor.json
/tmp/proton_cann_tests/custom/post_import.timeline.json
```

`summary.json` 同时包含每个算子的耗时和 profiler overhead：

- `results[].baseline_elapsed_s`：未启动 Proton session 时，单独运行该算子的总耗时。
- `results[].profiled_elapsed_s`：启动 `proton.start(... backend="cann" ...)` 后，运行同一算子的总耗时。
- `results[].overhead_s`：`profiled_elapsed_s - baseline_elapsed_s`。
- `results[].overhead_percent`：该算子的相对 overhead。
- `timing.average_overhead_percent`：所有成功算子的 per-operator overhead 算术平均。
- `timing.weighted_overhead_percent`：按总耗时加权的整体 overhead，等价于 `(profiled_total_s - baseline_total_s) / baseline_total_s`。
- `overhead_method`：overhead 统计口径。默认 driver 使用 `separate_process_no_profiler_baseline`，即先单独运行一个无 profiler baseline 进程，再运行 profiled 进程。

顶层 `summary.json` 汇总每个子 suite 的状态；具体 overhead 和 CANN artifact 仍在子目录的 `summary.json` 中。专项 runner 仍保留在 `scripts/` 下，但作为统一入口内部实现，通常不直接调用。

## 3. 加入 Liger-Kernel

第二级测试用真实开源 Triton 算子库验证 profiler。Liger-Kernel 主要覆盖 LLM 训练/推理相关 low-level Triton kernel，规模比自定义 12 算子更接近实际库封装，但仍可控，适合作为日常扩展回归。

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests_liger \
  --clean \
  --with-liger \
  --warmup 1 \
  --iters 3
```

未传 `--liger-source` 时，脚本会自动 clone Liger-Kernel 到 `<out>/liger/Liger-Kernel`；已有 checkout 时可以传 `--liger-source /path/to/Liger-Kernel` 复用源码。只跑一个 Liger case 时加 `--liger-case liger_rms_norm`。

Liger suite 当前覆盖 19 个已选定、可在 Ascend 环境稳定运行的 low-level Liger case。`liger/summary.json` 字段和 12 算子 suite 保持一致，也包含 overhead、CANN association sources、MSTX ranges、bandwidth association count 和 top op types。

重点检查：

- `ok_count` 是否等于 `case_count`。
- `failed_count` 是否为 0。
- `association_sources` 是否包含 `aclprof_op_summary`、`msprof_mstx`、`msprof_bandwidth`。
- `mstx_ranges` 是否包含 `proton_cann_liger::...`。
- `top_op_types` 是否出现 Liger Triton kernel 名称，例如 `_rms_norm_forward_kernel_no_tiling`。

## 4. 加入 FlagGems

第三级测试用 [FlagGems](https://github.com/flagos-ai/FlagGems) 的公开 Triton 算子库做评估。FlagGems 自带 Ascend backend，benchmark 文件按 PyTorch API/算子组织，覆盖点算、矩阵乘、归约、softmax、构造、索引等多类真实库算子。

快速验证时只加 `--with-flaggems`，此时运行默认代表性集合：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests_flaggems \
  --clean \
  --with-flaggems
```

跑全量 FlagGems op-level benchmark：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests_flaggems_all \
  --clean \
  --with-flaggems \
  --flaggems-all
```

未传 `--flaggems-source` 时，脚本会自动 clone FlagGems 到 `<out>/flaggems/FlagGems`；已有 checkout 时可以传 `--flaggems-source /path/to/FlagGems` 复用源码。只统计 op-level case、不实际运行时加 `--list-flaggems-ops --flaggems-all`。只跑一个 op 时加 `--flaggems-op add`。

当前 FlagGems checkout 静态识别结果：

```text
op-level benchmark case: 645
unique op marker: 596
```

预期 `summary.json` 中应看到：

```json
{
  "backend": "cann",
  "failed_count": 0,
  "association_sources": {
    "aclprof_op_summary": "...",
    "msprof_bandwidth": "...",
    "msprof_mstx": "..."
  }
}
```

`flaggems/summary.json` 会汇总每个 FlagGems op-level case 的 profiler overhead，字段含义和 12 算子 suite 一致：

- `results[].baseline_elapsed_s`
- `results[].profiled_elapsed_s`
- `results[].overhead_s`
- `results[].overhead_percent`
- `timing.average_overhead_percent`
- `timing.weighted_overhead_percent`
- `overhead_method`：FlagGems suite 使用 `separate_process_no_profiler_baseline`，每个 op-level case 分别跑 baseline worker 和 profiled worker。

重点检查：

- 全量 FlagGems op-level benchmark 中如果出现失败，需要区分是 FlagGems/Ascend 算子兼容性问题，还是 profiler 崩溃。
- `association_sources` 是否包含 `aclprof_op_summary`、`msprof_mstx`、`msprof_bandwidth`。
- `mstx_ranges` 是否包含 `proton_cann_flaggems::...`。
- `top_op_types` 是否出现 FlagGems kernel 名称或 CANN op 类型，例如 `add_func_kernel_rank_1`、`addmm_kernel`、`amax_kernel`。

FlagGems runner 不修改、不安装 FlagGems；它只在 worker 进程中把 `<FlagGems>/src` 和仓库根目录追加到 `sys.path`。注意不要覆盖 CANN `set_env.sh` 已经设置的 `PYTHONPATH`，否则会导致 CANN Python 组件如 `tbe` 不可见。

FlagGems runner 通过 `@pytest.mark.<op>` 识别算子，并把 pytest target 精确到 `benchmark/test_x.py::test_func`。同一个 marker 出现在多个 test function 中时会作为多个 op-level benchmark case 运行，避免结果互相覆盖。FlagGems benchmark 自身也会记录 kernel latency，但 profiler overhead 使用外层 wall time 统计。少量 case 下这个值会受编译缓存、shape 顺序和系统抖动影响，可能出现负数；正式数据建议增加 case 数、固定缓存状态并重复多轮。

## 5. 最小 direct-finalize 手工测试

如果只想验证用户 API 是否可用，可以写一个最小 Triton workload，外层只包：

```python
sid = proton.start(
    name="/tmp/my_triton_profile/profile",
    context="shadow",
    data="tree",
    backend="cann",
    hook="triton",
    mode=(
        "runtime_base:"
        "vendor_metrics=aicore,bandwidth:"
        "mstx_enabled=true:"
        "mstx_domain=proton"
    ),
)

# run Triton kernels

proton.finalize(sid)
```

检查输出：

```bash
ls /tmp/my_triton_profile
python3 - <<'PY'
import json
base = "/tmp/my_triton_profile/profile"
vendor = json.load(open(base + ".vendor.json"))
meta = json.load(open(base + ".meta.json"))
print("backend:", meta.get("backend"))
print("raw inputs:", len(vendor.get("raw_inputs", [])))
print("associations:", len(vendor.get("associations", [])))
print("sources:", sorted({a.get("source") for a in vendor.get("associations", []) if a.get("source")}))
PY
```

当前昇腾默认行为下，`hook="triton"` 会启用 IR instrumentation，并关闭 CANN
legacy `aclprof/msprof`。预期仍然只生成标准四个输出文件：

```text
/tmp/my_triton_profile/profile.hatchet
/tmp/my_triton_profile/profile.meta.json
/tmp/my_triton_profile/profile.timeline.json
/tmp/my_triton_profile/profile.vendor.json
```

内部 IR op timeline 会合并进 `profile.timeline.json`，Hatchet 中会增加
`flagtree.ir.*` / `flagtree.internal.*` 指标。最小检查：

```bash
python3 - <<'PY'
import json
base = "/tmp/my_triton_profile/profile"
trace = json.load(open(base + ".timeline.json"))
hatchet = json.load(open(base + ".hatchet"))
events = [
    event for event in trace["traceEvents"]
    if event.get("cat") == "flagtree.kernel_internal"
]
metrics = set()
def walk(node):
    metrics.update(node.get("metrics", {}).keys())
    for child in node.get("children", []):
        walk(child)
walk(hatchet[0])
print("internal_timeline_events:", len(events))
print("first_internal_event:", events[0] if events else None)
print("ir_metrics:", sorted(m for m in metrics if m.startswith("flagtree.ir.")))
PY
```

预期 `internal_timeline_events > 0`，并能看到 `flagtree.ir.duration_cycle`
等指标。这些 records 来自设备端 debug ring buffer，不是静态 IR metadata。
默认 IR 路径没有 CANN kernel event 时，会在 `profile.timeline.json` 中生成
`flagtree.ir_kernel` synthetic kernel event，并把内部 timeline 事件的 `ts` /
`dur` 映射到该窗口内；原始设备 `SYS_CNT` cycle 会保留在
`args.start_cycle`、`args.end_cycle`、`args.duration_cycle` 中。Timeline 默认覆盖
collect region 内非 constant tracked IR op，轻量 op 可能因为计数器分辨率显示为
`duration_cycle=0`。

自动插桩的 record buffer 默认是 32 MiB（`524288` 条、64 bytes/条），每次
kernel launch 导出后释放。设置 `PROTON_IR_RECORD_BUFFER_MB=<MiB>` 可以覆盖默认
值；该值会进入编译 cache key，因此修改后会自动重新编译插桩 kernel。更大的
buffer 能覆盖更多 program instance，但也会增加设备侧插桩扰动、导出时间和
`timeline.json` 体积。

如果要验证旧 CANN legacy 路径，运行前设置：

```bash
export PROTON_CANN_TRITON_HOOK_LEGACY=1
```

此时 `hook="triton"` 会恢复旧行为，不写入 `flagtree.kernel_internal` 事件或
`flagtree.ir.*` 指标。正常情况下，`sources` 至少应包含部分以下来源：

```text
aclprof_op_summary
msprof_mstx
msprof_bandwidth
msprof_api_statistic
msprof_op_statistic
```

## 6. 结果文件怎么看

主要输出文件：

- `*.meta.json`：backend、mode 配置、启用的 vendor metrics、degrade reasons。
- `*.vendor.json`：CANN CSV 导入结果、metric association、bandwidth、MSTX range。
- `*.timeline.json`：Chrome trace 格式时间线。
- `*.hatchet`：Proton/Hatchet 聚合视图。
- `summary.json`：suite 脚本的汇总结果。

常见字段：

- `degrade_reasons`：非致命降级原因。例如 task_time 不完整时使用 host timing fallback。
- `association_sources`：vendor 数据来源统计。
- `bandwidth_association_count`：包含 `bandwidth_gb_s` 的 association 数量。
- `mstx_ranges`：Proton scope / Triton hook 进入 CANN profiler 后导出的 range。
- `top_op_types`：CANN op summary 中出现次数最多的 op 类型。

## 7. 常见问题

### `msprof` 找不到

先确认 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
which msprof
```

### NPU 不可用

确认：

```bash
python3 - <<'PY'
import torch
import torch_npu
print(torch.npu.is_available())
PY
```

### 输出目录权限问题

CANN/msprof 对输出目录权限较敏感。建议使用独立目录，并确保不是 group/other writable：

```bash
mkdir -p /tmp/my_triton_profile/msprof
chmod 700 /tmp/my_triton_profile /tmp/my_triton_profile/msprof
```

### `degrade_reasons` 不为空

不一定表示测试失败。常见情况是 CANN runtime event 不完整，Profiler 默认使用 host timing fallback 保留基础关联。需要结合 `association_sources`、`bandwidth_association_count` 和 `mstx_ranges` 判断是否采到了核心数据。
