# Proton CANN FlagGems Suite

FlagGems suite 是第三级测试，用公开 Triton 算子库 [FlagGems](https://github.com/flagos-ai/FlagGems) 全量验证 FlagTree Profiler。它比手写 smoke kernel 和 Liger 子集覆盖更广：FlagGems 自带 Ascend backend，benchmark 按算子文件组织，并通过 PyTorch API 调到 Triton kernel。

## 运行方式

运行完整 benchmark 集合：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_flaggems_full \
  --clean \
  --with-flaggems \
  --flaggems-all
```

未传 `--flaggems-source` 时，脚本会自动 clone FlagGems 到 `<out>/flaggems/FlagGems`；已有 checkout 时可以传 `--flaggems-source /path/to/FlagGems` 复用源码。

只跑一个文件：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_flaggems_add \
  --clean \
  --with-flaggems \
  --flaggems-op add
```

快速验证 runner 时，可以不加 `--all`，此时只跑默认 12 个代表性 benchmark 文件：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_flaggems_default \
  --clean \
  --with-flaggems
```

只统计 op-level case，不实际运行：

```bash
python3 third_party/proton/flagtree_profiler/scripts/cann_profile_test_suite.py \
  --out /tmp/proton_cann_flaggems_ops_list \
  --clean \
  --with-flaggems \
  --list-flaggems-ops \
  --flaggems-all
```

## 默认快速覆盖

默认集合包含：

```text
test_add.py
test_addmm.py
test_bmm.py
test_mm.py
test_softmax.py
test_log_softmax.py
test_amax.py
test_argmax.py
test_cumsum.py
test_where_self_out.py
test_arange.py
test_zeros.py
```

这组 case 覆盖点算、矩阵乘、batch matmul、softmax/log-softmax、归约、scan、where、构造类算子。正式第三级评估使用 `--op-level --all` 跑完整 benchmark 集合。

当前 `/tmp/FlagGems` checkout 静态识别到：

```text
op-level benchmark case: 645
unique op marker: 596
```

## 输出

每个 benchmark 文件都会生成独立目录：

```text
<out>/cases/<case>/baseline.json
<out>/cases/<case>/baseline.pytest.json
<out>/cases/<case>/profiled.json
<out>/cases/<case>/profiled.pytest.json
<out>/cases/<case>/profile.hatchet
<out>/cases/<case>/profile.meta.json
<out>/cases/<case>/profile.timeline.json
<out>/cases/<case>/profile.vendor.json
```

总汇总写入：

```text
<out>/summary.json
```

重点字段：

- `ok_count` / `failed_count`：成功和失败的 benchmark 文件数。
- `association_sources`：CANN 数据来源，通常应包含 `aclprof_op_summary`、`msprof_bandwidth`、`msprof_mstx`。
- `mstx_ranges`：应包含 `proton_cann_flaggems::<case>` 和部分 Triton/CANN kernel range。
- `top_op_types`：从 CANN 数据中解析到的 op/kernel 类型。
- `timing.*overhead*`：baseline worker 与 profiled worker 的 wall time 差值。

## 已验证结果

在 `flagtree-dev-chijin` 容器中，源码方式运行 `/tmp/FlagGems` 已验证：

- `test_add.py` 单 case：`ok_count=1`、`failed_count=0`。
- 3-case smoke：`test_add.py`、`test_addmm.py`、`test_amax.py` 全部通过。
- CANN association sources 包含 `aclprof_op_summary`、`msprof_api_statistic`、`msprof_bandwidth`、`msprof_mstx`、`msprof_op_statistic`。
- `mstx_ranges` 包含 `proton_cann_flaggems::test_add`、`proton_cann_flaggems::test_addmm`、`proton_cann_flaggems::test_amax`。
- `top_op_types` 出现 `add_func_kernel_rank_1`、`addmm_kernel`、`amax_kernel`、`MatMulV2`、`ReduceMax` 等。

## 注意事项

- runner 不修改、不安装 FlagGems；它只在 worker 中把 `<FlagGems>/src` 和仓库根目录追加进 `sys.path`。
- 加载 CANN 后不要覆盖 `PYTHONPATH`，否则 CANN Python 组件可能不可见。需要加入 FlagGems 路径时使用追加方式。
- FlagGems benchmark 自身会运行多个 shape。少量 case 的外层 wall time overhead 会受编译缓存和系统抖动影响，可能为负数；正式统计建议用更多 case、多轮重复和固定缓存状态。
- 全量 `--all` 中如果出现失败，先看 `failures` 判断是 FlagGems/Ascend 算子兼容性问题还是 profiler 问题。
