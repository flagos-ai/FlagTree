# FlagTree Profiler

FlagTree Profiler 是 FlagTree 在 Proton 上扩展出的 Triton profiling 工具。它复用 Proton 的 session、scope、Triton hook、数据导出能力，并增加 IR instrumentation 路径，用来在不依赖特定厂商 profiler 的情况下采集 Triton kernel 内部 timeline。昇腾 CANN 的 legacy `aclprof/msprof` 路径仍然保留，可通过环境变量显式打开。

当前用户接口仍然是 Proton API：

```python
import triton.profiler as proton

sid = proton.start(
    name="/tmp/my_triton_profile/profile",
    context="shadow",
    data="tree",
    backend="cann",
    hook="triton",
    mode=(
        "runtime_base:"
        "device_id=0:"
        "vendor_metrics=aicore,bandwidth:"
        "mstx_enabled=true:"
        "mstx_domain=proton"
    ),
)

# 运行 Triton kernel

proton.finalize(sid)
```

输出文件包括：

```text
/tmp/my_triton_profile/profile.hatchet
/tmp/my_triton_profile/profile.meta.json
/tmp/my_triton_profile/profile.timeline.json
/tmp/my_triton_profile/profile.vendor.json
```

默认情况下，`backend="cann", hook="triton"` 会使用 FlagTree IR instrumentation，不会启动 CANN `aclprof/msprof`。`profile.hatchet` 可以直接用 `proton-viewer` 查看调用树和数值指标，例如：

```bash
proton-viewer -m flagtree.ir.duration_cycle,flagtree.ir.memory_access_bytes,flagtree.ir.estimated_bandwidth_bytes_per_cycle \
  /tmp/my_triton_profile/profile.hatchet
```

如果需要恢复旧的 CANN legacy 路径，在运行程序前设置：

```bash
export PROTON_CANN_TRITON_HOOK_LEGACY=1
```

此时 `hook="triton"` 会恢复旧行为：启动 CANN `aclprof/msprof`、导入 CANN CSV，并输出 `cann.*` 指标，例如：

```bash
PROTON_CANN_TRITON_HOOK_LEGACY=1 python3 your_program.py
proton-viewer -m time/us,cann.task_duration_us,cann.bandwidth_gb_s \
  /tmp/my_triton_profile/profile.hatchet
```

`hook="instrumentation"` 仍然可以显式启用 IR 自动插桩：

```python
sid = proton.start(
    name="/tmp/my_triton_profile/profile",
    context="shadow",
    data="tree",
    backend="cann",
    hook="instrumentation",
    mode=(
        "runtime_base:"
        "device_id=0:"
        "vendor_metrics=aicore,bandwidth:"
        "mstx_enabled=true:"
        "mstx_domain=proton"
    ),
)
```

`hook="instrumentation"` 会在 Triton 编译阶段自动给当前 kernel 插入 FlagTree
debugger collect region。输出文件仍然只有原来的 4 个：

```text
/tmp/my_triton_profile/profile.hatchet
/tmp/my_triton_profile/profile.meta.json
/tmp/my_triton_profile/profile.timeline.json
/tmp/my_triton_profile/profile.vendor.json
```

内部 IR op 的信息会合并进 `profile.timeline.json` 和 `profile.hatchet`。
`profile.timeline.json` 中会出现 `flagtree.kernel_internal` 事件，表示 Triton
kernel 内部非 constant tracked IR op 的设备侧时间戳窗口，例如
`tt.get_program_id`、地址/mask 计算、`tt.load`、`arith.addf`、`tt.store`、
`tt.dot` 等。显示用时间轴会映射到对应 kernel event 的窗口内；事件 `args`
保留原始昇腾 `SYS_CNT` cycle，包括 `op_id`、`logical_instance_id`、
`mlir_op`、`source_loc`、`triton_statement`、`start_cycle`、`end_cycle`、
`duration_cycle`。`profile.hatchet` 中会在对应 kernel 节点下增加 IR op 子节点，
并提供 `flagtree.internal.duration_cycle`、
`flagtree.internal.avg_duration_cycle`、`flagtree.internal.count` 等指标。

在当前版本中，昇腾上 `hook="triton"` 默认等价于“自动 Triton launch hook +
IR instrumentation + 不启动 CANN legacy profiler”。设置
`PROTON_CANN_TRITON_HOOK_LEGACY=1` 后，`hook="triton"` 才会恢复原来的 kernel
级 CANN profiling，并且不会增加 `flagtree.kernel_internal` 事件/节点。

`profile.timeline.json` 是 Chrome Trace Event 格式，可以用 Perfetto UI、
Chrome trace viewer 或 MindStudio 兼容 trace viewer 打开。Timeline 中会包含：

- `stream <id>` 线程：实际 kernel/CANN task 事件。Timeline 主视图只画真实
  设备侧 task，不再把 Proton scope/range 作为单独事件画出来。

kernel 事件的 `args` 中会把常用字段放在顶层，便于 trace viewer 点选查看，例如
`cann.task_duration_us`、`cann.op_summary_task_duration_us`、
`cann.task_wait_time_us`、`cann.aicore_time_us`、`cann.aiv_time_us`、
`cann.bandwidth_gb_s`、`cann.memory_access_bytes`。完整原始字段仍保留在
`args.metrics` 和 `profile.vendor.json` 中。Proton scope 信息保留在每个事件的
`args.call_stack` 中，用于说明该 kernel 归属于哪个逻辑 scope。

Hatchet 输出只保留可聚合的数值 metric；CANN 原始字符串字段、文件路径、op
类型等元数据保留在 `profile.vendor.json` 和 `profile.timeline.json` 中。如果
Triton kernel 通过 `launch_metadata` 提供 `flops16`/`bytes` 等 workload
hint，Hatchet 会保留这些字段，并在有 CANN `task_duration_us` 时派生
`cann.tflop16_s`、`cann.estimated_bandwidth_gb_s` 等指标。
同一个用户 Triton kernel 在 CANN 中可能同时出现 launch range 名称
（例如 `_matmul_kernel mix`）和 op_summary 名称（例如 `_matmul_kernel`）；
导出到 Hatchet 时会按规范化 kernel 名和时间戳合并到一个节点，避免展示成两个
用户 kernel。

### 可采集和展示的指标

`profile.hatchet` 保存适合树形聚合和 `proton-viewer` 展示的数值字段。
`proton-viewer -m` 可以传一个或多个字段名。时间、cycle、bytes、KB、
FLOPs、count 等可加字段会以 inclusive 方式聚合，输出名通常带 `(inc)`
后缀；带宽、吞吐、ratio/rate 等不能直接相加的字段不会强行聚合，父节点显示
`NaN`。少数能由基础量重新计算的字段，例如 `cann.tflop16_s` 和
`cann.estimated_bandwidth_gb_s`，会按子树内的 `sum(flops*) / sum(duration)`
或 `sum(bytes) / sum(duration)` 重新计算。例如：

```bash
proton-viewer -m time/ms,cann.task_duration_us,cann.aicore_time_us,cann.bandwidth_gb_s,cann.memory_access_bytes \
  /tmp/my_triton_profile/profile.hatchet
```

常用字段：

- `flagtree.ir.duration_cycle`：IR op 的设备侧 `SYS_CNT` cycle 持续时间。父节点按 inclusive 方式聚合。
- `flagtree.ir.kernel_elapsed_cycle`：一个 Triton kernel 内部 IR timeline 的最早 start 到最晚 end 的 cycle 窗口。
- `flagtree.ir.count`：IR op timeline record 数。
- `flagtree.ir.memory_access_bytes`、`flagtree.ir.memory_read_bytes`、`flagtree.ir.memory_write_bytes`：由 IR memory op 的静态 `accessBytes * vecWidth` 和 runtime record 数估算出的访问字节数。
- `flagtree.ir.estimated_bandwidth_bytes_per_cycle`：由 `flagtree.ir.memory_access_bytes / flagtree.ir.duration_cycle` 估算的 bytes/cycle。它不是硬件带宽计数器，不能等价于 CANN `cann.bandwidth_gb_s`。
- `time/ms`、`time/us`、`time/ns`：`proton-viewer` 展示的默认时间字段。有 CANN
  `op_summary` begin/finish 数据时，它表示设备侧 elapsed 时间：父节点取子树
  最早 task start 到最晚 task end，叶子节点表示该 task 的设备执行窗口，输出
  名会显示为 `time/us (cann elapsed)`。没有 CANN 数据时回退到 Proton 原生
  `time (ns)`。
- `time (ns)`：Hatchet 文件中保存的 Proton 原生时间，主要对应 host launch/range
  时间；CANN 后端分析设备耗时时通常不直接使用这个原始字段。
- `avg_time/ms`、`avg_time/us`、`avg_time/ns`：`time / count` 的平均耗时，由 `proton-viewer` 派生。
- `count`：当前节点累计的事件次数。
- `bytes`：Triton `launch_metadata` 提供的 workload hint，表示估算数据访问字节数。
- `flops`、`flops16`：Triton `launch_metadata` 提供的 workload hint，表示估算 FLOPs / fp16 FLOPs。
- `gbyte/s`、`tbyte/s`、`gflop/s`、`tflop/s`、`gflop16/s`、`tflop16/s`：`proton-viewer` 基于 `time` 和 `bytes` / `flops` / `flops16` 派生的吞吐指标。
- `cann.task_duration_us`：CANN/MSTX 关联到 Proton scope 后的任务持续时间，单位 us。通常用于看设备侧 kernel 耗时。
- `cann.op_summary_task_duration_us`：CANN `aclprof_op_summary` 原始 task duration，单位 us。用于保留 op summary 的原始耗时。
- `cann.task_wait_time_us`：CANN 任务等待时间，单位 us。
- `runtime.duration_us`：runtime 侧事件持续时间，单位 us。
- `cann.aicore_time_us`、`cann.aic_total_cycles`：AICore 执行时间和 cycle 数。
- `cann.aiv_time_us`、`cann.aiv_total_cycles`：AIV 执行时间和 cycle 数。
- `cann.tflop_s`、`cann.tflop8_s`、`cann.tflop16_s`、`cann.tflop32_s`、`cann.tflop64_s`：`proton-viewer` 结合 CANN `task_duration_us` 和 Triton workload hint 派生的有效计算吞吐。父节点会按 `sum(flops*) / sum(cann.task_duration_us)` 重新计算，不会把子节点吞吐率直接相加。
- `cann.bandwidth_gb_s`：CANN 导入的原始带宽，单位 GB/s。该字段不能跨父子节点直接相加；父节点通常显示 `NaN`。
- `cann.estimated_bandwidth_gb_s`：`proton-viewer` 结合 `bytes` 和 CANN `task_duration_us` 派生的有效带宽，单位 GB/s。父节点会按 `sum(bytes) / sum(cann.task_duration_us)` 重新计算。
- `cann.memory_access_bytes`、`cann.memory_read_bytes`、`cann.memory_write_bytes`：CANN 统计的总访存、读访存、写访存字节数。

CANN 还可能提供更细的分层存储流量字段，例如：

- `cann.aic_read_main_memory_datas_kb`、`cann.aic_write_main_memory_datas_kb`
- `cann.aic_gm_to_l1_datas_kb`、`cann.aic_gm_to_ub_datas_kb`
- `cann.aic_l0c_to_gm_datas_kb`、`cann.aic_l0c_to_l1_datas_kb`
- `cann.aic_ub_to_gm_datas_kb`
- `cann.aiv_read_main_memory_datas_kb`、`cann.aiv_write_main_memory_datas_kb`
- `cann.aiv_gm_to_l1_datas_kb`、`cann.aiv_gm_to_ub_datas_kb`
- `cann.aiv_l0c_to_gm_datas_kb`、`cann.aiv_l0c_to_l1_datas_kb`
- `cann.aiv_ub_to_gm_datas_kb`

这些字段单位为 KB，用于分析 GM、L1、L0C、UB 等层级之间的数据搬运。
不同 CANN 版本、芯片型号和算子类型导出的字段可能不同；实际可用字段以
`proton-viewer --list /tmp/my_triton_profile/profile.hatchet` 和
`profile.vendor.json` 为准。需要查看 CANN 原始字段和字符串元数据时，使用：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_vendor_raw_report.py \
  /tmp/my_triton_profile/profile.vendor.json
```

### 参数说明

`proton.start()` 的顶层参数：

- `name="/tmp/my_triton_profile/profile"`：输出文件前缀。最终会生成 `profile.hatchet`、`profile.meta.json`、`profile.timeline.json`、`profile.vendor.json`。
- `context="shadow"`：使用 Proton scope/shadow context 组织调用树。配合 Triton hook 和手动 scope 时，输出更适合按算子或代码区域聚合。
- `data="tree"`：使用树形数据格式输出 profile 结果。
- `backend="cann"`：选择昇腾 CANN vendor backend。目前 FlagTree Profiler 只接入了这个后端。
- `hook="triton"`：打开 Triton kernel launch hook。用户的 Triton kernel 启动时会自动进入/退出 Proton scope，不需要手动包每个 kernel。
- `hook="instrumentation"`：在 `hook="triton"` 的基础上启用 Triton IR 自动插桩，并把内部 IR op timeline/metrics 合并进原有 `profile.timeline.json` 和 `profile.hatchet`。用户不需要手动写 `tl.debug_collect_start/end`。
- `mode=...`：后端配置字符串，多个配置项用 `:` 连接。

`mode` 中的配置项：

- `runtime_base`：启用基础 runtime profiling。即使 vendor 数据不完整，也会保留 Proton 的基础运行时信息。
- `device_id=0`：指定 CANN profiler 采集的 NPU 设备号。这个值必须和程序实际使用的 `torch.npu.set_device(n)` / `npu:n` 保持一致；设备不一致时，CANN 可能采不到目标 kernel，严重时会在 `finalize()` 的 `aclprofStop` 阶段等待错误设备而卡住。
- `vendor_metrics=aicore,bandwidth`：请求 CANN vendor metrics。`aicore` 表示采集/导入 AICore 相关指标；`bandwidth` 表示采集/导入内存访问和带宽相关指标。
- `mstx_enabled=true`：启用 MSTX range 标注。开启后，Proton scope/Triton hook 产生的范围可以进入 CANN/MindStudio profiler 数据，并出现在 `vendor.json` 和 `timeline.json` 中。
- `mstx_domain=proton`：MSTX domain 名称。默认建议使用 `proton`，方便在导出的 CANN 数据中识别 Proton 产生的 range。

默认行为和调试开关：

- 昇腾上 `hook="triton"` 默认启用 IR instrumentation，不启动 CANN `aclprof/msprof`。需要恢复旧 CANN 路径时设置环境变量 `PROTON_CANN_TRITON_HOOK_LEGACY=1`。
- `runtime_host_timing_fallback` 默认开启。当 CANN runtime event 不完整时，Profiler 会用 host timing 辅助关联 CANN op summary 和 Proton scope；普通用户不需要配置。需要关闭时设置环境变量 `PROTON_CANN_RUNTIME_HOST_FALLBACK=0`。
- IR 自动插桩默认为每次 kernel launch 分配 32 MiB record buffer（`524288` 条、64 bytes/条），kernel 结束导出后释放。可通过 `PROTON_IR_RECORD_BUFFER_MB=<MiB>` 调整；复杂 kernel 每个 program instance 包含更多 IR op，需要更大的 buffer 才能覆盖相同数量的 instance。
- `aclprof_output_path` 默认不需要设置。Profiler 会使用内部临时目录收集 CANN 原始数据，导入 `profile.vendor.json` 后自动清理。需要保留 CANN 原始 `PROF_*` 和 CSV 时，设置环境变量 `PROTON_CANN_PROFILE_OUTPUT=/tmp/my_triton_profile/msprof`。
- `aclprof_runtime_enabled=false`：关闭 Proton 内部启动 CANN aclprof。适合只做已有 CSV 的 post-import 或外部 `msprof` 包裹验证。
- `aclprof_auto_export=false`：关闭 `finalize()` 内部自动 `msprof --export=on`。适合调试 CANN 原始 profiler 目录或外部导出流程。
- `mstx_enabled=false`：关闭 MSTX range 标注。

## 已接入后端

目前只接入了 **昇腾 CANN**：

- `backend="cann"`
- 默认 `hook="triton"` 使用 IR instrumentation，采集 Triton kernel 内部 `flagtree.ir.*` cycle、op count、估算 memory bytes/bytes-per-cycle，并输出到 `hatchet`、`timeline.json`、`vendor.json`。
- 旧 CANN legacy 路径通过 `PROTON_CANN_TRITON_HOOK_LEGACY=1` 启用。该路径支持基础 runtime profiling 和 CANN vendor metrics 导入，包括 AICore、MSTX range、bandwidth 相关指标。
- legacy 路径的 `finalize()` 内部会自动调用 CANN 的 `msprof --export=on` 把 profiler 原始目录导出成 CSV，然后导入到 Proton artifact。用户不需要手动用 `msprof python ...` 包住程序。
- Triton kernel 始终通过 `hook="triton"` 自动进入 Proton scope。
- IR 自动插桩当前已在 CANN/Ascend 路径打通 kernel 内部 TIMELINE record 导出；自动 profiler 模式默认只采内部 timeline，避免额外 value summary 插桩影响复杂 kernel lowering。

## 示例

`examples/` 目录提供几个可直接运行的示例，先覆盖简单/中等/复杂单算子，再提供
一个带 Proton scope 的多算子 Tiny MLP，适合在介绍 GPT 示例前解释多 kernel
调用树和父子节点聚合：

- `examples/simple_vector_add_profile.py`：vector add，验证最基础的 Triton launch hook、runtime_base 和 CANN CSV 导入。
- `examples/medium_softmax_profile.py`：row-wise softmax，验证包含 reduce/exp/div/store 的融合算子。
- `examples/complex_matmul_profile.py`：tiled fp16 matmul，验证 `tl.dot`、tile 循环和更复杂的 AICore 指标关联。
- `examples/tiny_mlp_profile.py`：两层 Tiny MLP，包含 linear、ReLU、linear 三个 Triton kernel，并用 `proton.scope` 组织 `tiny_mlp/1_layer1/2_activation/3_layer2` 调用树。
- `examples/timeline_showcase_profile.py`：多 block 小型 MLP，包含 20 个左右 Triton kernel 事件，并用 `timeline_showcase/block_xx/phase` scope 组织 timeline，适合演示 `profile.timeline.json`。
- `examples/internal_timeline_profile.py`：单个 vector add Triton kernel，使用 `hook="instrumentation"` 演示合并在 `profile.timeline.json` 和 `profile.hatchet` 中的内部 IR op 信息，可看到 `tt.get_program_id`、地址/mask 计算、`tt.load`、`arith.addf`、`tt.store` 等 IR op 的设备 cycle 时间窗口。

每个示例文件顶部都写明了运行命令、预期输出文件，以及 `hatchet`、`meta.json`、`vendor.json`、`timeline.json` 的内容样例或查看命令。

## 测试套件

Profiler 目录下提供了几类测试脚本：

- `scripts/cann_profile_test_suite.py`：统一测试入口。默认运行项目内 12 个自定义 Triton 算子；加 `--with-liger` 后运行 Liger-Kernel 真实 LLM Triton 算子库；加 `--with-flaggems` 后运行公开 Triton 算子库 FlagGems 的 benchmark。
- `scripts/cann_operator_profile_suite.py`、`scripts/cann_liger_profile_suite.py`、`scripts/cann_flaggems_profile_suite.py`：统一入口内部调用的专项 runner，通常不需要直接使用。

默认 12 算子测试：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests \
  --clean
```

加入 Liger 和 FlagGems：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_tests_full \
  --clean \
  --with-liger \
  --with-flaggems \
  --flaggems-all
```

未传 `--liger-source` / `--flaggems-source` 时，统一入口会让对应 runner 自动 clone 到 `<out>/liger/Liger-Kernel` 或 `<out>/flaggems/FlagGems`；已有 checkout 时可以传源码路径复用。FlagGems 的 `--flaggems-all` 会运行全量 op-level case；不加时只运行默认代表性集合，适合快速验证。

## 设计简述

FlagTree Profiler 的实现分为几层，目标是在不改变 Proton 使用方式的前提下，优先用 IR instrumentation 提供跨后端基础 profiler；如果某个后端有成熟厂商 profiler，再把厂商数据作为增强指标接进 `proton.start()` / `proton.finalize()` 生命周期。

1. **用户入口层**：用户仍调用 `triton.profiler`。`proton.start(..., backend="cann", hook="triton", mode="...")` 进入 Proton 原有 Python API。昇腾上 `hook="triton"` 默认打开 FlagTree debugger 自动插桩，并把 CANN legacy `aclprof/msprof` mode 改成关闭；设置 `PROTON_CANN_TRITON_HOOK_LEGACY=1` 后恢复旧 CANN 路径。相关文件：`third_party/proton/proton/proton.py`、`third_party/proton/proton/profile.py`。

2. **Triton hook 层**：`hook="triton"` 在 Triton kernel launch 前后自动进入/退出 Proton scope，使用户不需要手动包每个 kernel。相关文件：`third_party/proton/proton/hook.py`。

3. **IR 插桩层**：`triton.compiler.flagtree_debug` 在无用户 marker 时自动插入默认 collect region，运行现有 FlagTree debugger metadata/instrumentation pass，并通过 Ascend launch hidden arg 把 debug control buffer 传给 kernel。kernel 执行结束后 runtime 导出 ring buffer。关键文件：`python/triton/compiler/flagtree_debug.py`、`third_party/Debugger/lib/Metadata/Passes.cpp`、`third_party/Debugger/lib/Instrumentation/Passes.cpp`、`python/triton/runtime/debugger.py`、`python/triton/runtime/debug_collect_runtime.py`、`third_party/ascend/backend/spec/triton/runtime/jit.py`。

4. **Artifact 合成层**：Python `finalize()` 在 Proton C++ session 写完基础文件后，把 IR runtime records 合并进原有 `profile.timeline.json` 和 `profile.hatchet`，同时在 `profile.meta.json`、`profile.vendor.json` 中标注 IR 数据源、默认/legacy 模式、可用指标和不可用的 CANN-only 指标。关键文件：`third_party/proton/proton/profile.py`。

5. **Proton 生命周期层**：`Session` 在 `start()` 时创建 profiler，在 `finalize()` 时停止采集、触发导出和基础 artifact 写入。CANN legacy 路径仍然通过这里创建 vendor profiler。相关文件：`third_party/proton/csrc/lib/Session/Session.cpp`。

6. **Vendor adapter 层**：`Adapter` 根据 `backend` 创建具体厂商后端，并解析通用 `mode` 配置。它是可选增强层，不是默认 IR profiler 的必要条件。相关文件：`flagtree_profiler/csrc/include/Profiler/Vendor/Adapter.h`、`flagtree_profiler/csrc/lib/Profiler/Vendor/Adapter.cpp`、`Mode.cpp`。

7. **CANN legacy 后端层**：`CannProfiler` 负责调用/控制 CANN profiling，处理 MSTX range、自动 `msprof --export=on`、CSV 导入，以及把 AICore、bandwidth、runtime/API 等数据关联到 Proton scope。相关文件：`flagtree_profiler/csrc/lib/Profiler/Vendor/CannProfiler.cpp`、`CannAdapter.cpp`、`Driver/Ascend/AscendApi.cpp`。

能够独立归属 FlagTree Profiler 的代码集中放在本目录；少量 Proton 公共 API、生命周期和数据模型接入点仍保留在 Proton 原目录，详见 [目录结构](docs/directory_structure.md)。

## 代码位置

Profiler 自身代码集中在 `third_party/proton/flagtree_profiler/`：

- `csrc/include/Profiler/Vendor/Adapter.h`：vendor backend 的核心接口，定义 `VendorAdapter`、`VendorMetricsImporter` 和 backend registry。
- `csrc/include/Profiler/Vendor/Mode.h`、`csrc/lib/Profiler/Vendor/Mode.cpp`：解析 `mode="runtime_base:vendor_metrics=..."`，输出 `VendorProfileOptions`。
- `csrc/lib/Profiler/Vendor/Adapter.cpp`：根据 `backend` 名称选择具体 backend。目前注册了 `cann`。
- `csrc/include/Profiler/Vendor/CannAdapter.h`、`csrc/lib/Profiler/Vendor/CannAdapter.cpp`：CANN backend 的能力声明、默认参数、metric plan 和 importer 创建。
- `csrc/include/Profiler/Vendor/CannProfiler.h`、`csrc/lib/Profiler/Vendor/CannProfiler.cpp`：CANN runtime profiling、MSTX range、aclprof/msprof 导出、CSV 导入、CANN metric 与 Proton scope 关联。
- `csrc/include/Driver/Ascend/AscendApi.h`、`csrc/lib/Driver/Ascend/AscendApi.cpp`：Ascend runtime/device discovery shim。
- `examples/`：可直接运行的单算子、多算子示例。
- `scripts/`：测试入口、开源算子库验证、CANN 原始数据报告脚本。
- `docs/`：Profiler 专项文档。

Proton 公共接入点仍在 `third_party/proton/` 原目录：

- `proton/proton.py`、`proton/profile.py`：Python API，把 `backend`、`hook`、`mode` 传到 C++ session。
- `proton/hook.py`：`hook="triton"` 的 Python launch hook。
- `proton/viewer.py`：`proton-viewer`，负责 Hatchet 树展示和字段聚合/派生规则。
- `csrc/include/Profiler/Profiler.h`：Proton profiler 生命周期基类，定义 `doStart()`、`doStop()`、`doSetMode()`。
- `csrc/include/Session/Session.h`、`csrc/lib/Session/Session.cpp`：`proton.start()` / `proton.finalize()` 生命周期，创建 vendor profiler，停止采集并写出 artifact。
- `csrc/include/Data/Artifacts.h`：vendor artifact、runtime event、metric association 的统一数据结构。
- `csrc/lib/Data/TreeData.cpp`：输出 `profile.hatchet`。
- `csrc/lib/Data/TraceData.cpp`：输出 `profile.timeline.json`。
- `CMakeLists.txt`：把 `flagtree_profiler/csrc` 编进 Proton。

IR 插桩相关代码在 FlagTree 主体和 Ascend backend 中：

- `python/triton/compiler/flagtree_debug.py`：决定是否自动插入 collect marker，并调度 debugger pass。
- `third_party/Debugger/include/Debugger/Metadata/Passes.h`、`third_party/Debugger/lib/Metadata/Passes.cpp`：默认 collect marker 插入和 metadata pass。
- `third_party/Debugger/lib/Instrumentation/Passes.cpp`：把 debugger record op lowering 成设备侧 ring buffer 写入。
- `python/triton/runtime/debugger.py`、`python/triton/runtime/debug_collect_runtime.py`：准备 hidden arg、导出和 decode runtime records。
- `third_party/ascend/backend/spec/triton/compiler/compiler.py`、`third_party/ascend/backend/spec/triton/runtime/jit.py`：Ascend 编译/JIT cache key 和 hidden arg launch 接入。

## 新增后端

新增芯片后端有两条路径。默认建议先接 IR profiler，因为它能复用 Proton API 和 artifact，不要求目标后端提供 `msprof` 这类厂商工具。

### IR profiler 最小适配面

1. **设备 timestamp/cycle**：在 instrumentation lowering 中提供目标后端可用的设备侧时间戳或 cycle 读取方式。
2. **trace buffer 写入**：支持 debugger record op 写入设备侧 ring buffer。
3. **hidden arg / launch 接入**：在对应 backend 的 compiler/JIT launch 路径中传入 debug control buffer，并在 kernel 结束后导出 runtime records。
4. **artifact 合成复用**：只要导出的 runtime records 符合 FlagTree debugger decode 结构，`third_party/proton/proton/profile.py` 中的 artifact 合成逻辑可以直接复用，生成 `flagtree.ir.*` 指标和四个输出文件。

### 厂商 profiler 增强适配面

如果目标芯片也有厂商 profiler，希望补充硬件 counter、runtime queue、真实带宽等指标，再实现三组接口：

1. **`VendorAdapter`**，定义后端叫什么、支持哪些 metric、怎么把用户请求转成执行计划。

   需要实现的接口在 `csrc/include/Profiler/Vendor/Adapter.h`：

   ```cpp
   std::string getName() const;
   DeviceType getDeviceType() const;
   std::vector<std::string> getSupportedVendorMetrics() const;
   VendorProfilePlan makePlan(const VendorProfileOptions &options) const;
   Profiler *getRuntimeProfiler() const;
   std::unique_ptr<VendorMetricsImporter> createImporter() const;
   ```

   参考实现：`CannAdapter`。

2. **`Profiler`**，接入 Proton 的 start/finalize 生命周期，负责启动和停止厂商 runtime profiler。

   需要实现的接口在 `third_party/proton/csrc/include/Profiler/Profiler.h`：

   ```cpp
   void doStart() override;
   void doStop() override;
   void doSetMode(const std::vector<std::string> &modeAndOptions) override;
   ```

   如果后端需要把 Triton launch scope 映射到厂商 range，还需要实现类似 CANN 的 `ThreadLocalOpInterface::startOp()` / `stopOp()` 逻辑。参考实现：`CannProfiler`。

3. **`VendorMetricsImporter`**，把厂商 profiler 的导出文件或内存结果转换成 Proton 统一 artifact。

   需要实现的接口在 `csrc/include/Profiler/Vendor/Adapter.h`：

   ```cpp
   std::string getName() const;
   VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                                const VendorProfilePlan &plan) const;
   ```

   import 的结果应填充 `VendorProfileArtifact`，核心是 `associations`：每条 association 把一个厂商 metric row 关联到一个 `RuntimeTraceEventKey` 或标记为 unmatched。后续 `Session.cpp` 会把这些数据写入 `profile.vendor.json`，并 overlay 到 `hatchet` / `timeline`。

实际落地时，建议按下面的顺序做：

1. 在 `csrc/include/Profiler/Vendor/` 增加 `<Vendor>Adapter.h` 和 `<Vendor>Profiler.h`。
2. 在 `csrc/lib/Profiler/Vendor/` 增加 `<Vendor>Adapter.cpp` 和 `<Vendor>Profiler.cpp`。
3. 在 `Adapter.cpp` 的 backend factory 中注册新的 `backend` 名称。
4. 在 `Mode` 解析中定义该后端需要的 `mode` 参数和默认值。
5. 在 `<Vendor>Profiler` 中实现 `doStart()`、`doStop()`、`doSetMode()`，并输出统一的 vendor association 数据。
6. 如果后端需要设备发现或 runtime shim，在 `csrc/include/Driver/<Vendor>/` 和 `csrc/lib/Driver/<Vendor>/` 下补充。
7. 增加对应的 `scripts/` 验证脚本和 `test/` 自动化测试。
8. 在 `docs/` 中补充后端使用方式、依赖、采集指标和验收标准。

新增后端应尽量保持用户接口一致：

```python
proton.start(..., backend="<vendor>", hook="triton", mode="...")
proton.finalize(sid)
```

## 相关文档

- [目录结构](docs/directory_structure.md)
- [测试指南](docs/testing.md)
- [CANN minimal adapter patch](docs/proton_vendor_adapter_minimal_patch.md)
- [CANN acceptance status](docs/proton_cann_acceptance_status.md)
- [FlagGems full suite](docs/proton_cann_flaggems_suite.md)
- [Liger-Kernel full suite](docs/proton_cann_liger_full_suite.md)
