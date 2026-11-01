# 工作区总结（2026-11-01）

本次保存累计的 TLE/FlagMega 开发状态、回归用例及 H800 教程结果。完整 GLM decode 的性能结论已有四卡验证；这不代表所有后端、全量测试及原始 IR fresh 重建均已通过。

更新：合并后 `53843322b` 的完整 TP4 重新验收失败，四个 rank 的第 2 个生成 token 均为 `646`，参考为 `284`。下文 6.4–6.8ms 表格对应合并前的历史源码和 Core；本轮没有新 TPOT。新 Core 构建、26 个入口/228 次调用的编译与资源检查通过。仅在隔离实验中开启 FP fusion 后，两次完整 8-token 请求及状态比较通过，数值问题已缩小到严格浮点合同与编译行为的差异，具体运算根因仍待定位；正式配置保持，隔离实验不计作验收。见[重新验收记录](reacceptance-20261101-zh.md)。

## 主要改动

| 范围 | 改动与用途 |
| --- | --- |
| TLE 编译器 | 修复共享参数及物理别名范围的同步分析、返回值原子操作的 scratch WAR、mbarrier 初始化与生命周期、WGMMA 跨循环携带及旧 MMA accumulator 转换、显式 layout/view 边界、while/coalesce 依赖与 warp scope；新增动态 device barrier 合同和对应 MLIR/Python 回归。原生 Triton 文件中的 TLE 改动受 `__TLE__` 保护。 |
| FlagMega IR 与规划 | 新增 typed execution mesh、program/task mapping、普通与 persistent program、program DAG、循环携带与共享存储所有权；重新计算候选、布局、ABI、生命周期、同步和资源约束。 |
| 代码生成与运行时 | 为不同 program 分别生成源码和 launch 配置，共享重复函数的 GPU 入口；保留每次调用的参数、descriptor、offset 和 scratch 身份。支持 DAG stream/Graph、联合驻留检查、readonly 物理 owner 与复合共享装载。 |
| 数值与内核 | 补充 NVIDIA FP8 WGMMA、KDA/MLA、DSA 完整 pool 评分、精确排序与 query 并行等实现及参考边界，保持原 FP32/BF16 舍入和完整任务覆盖。 |
| 教程与规范 | 更新 Qwen3.5 H800 教程，整理 GLM 入口及 `local_optimizations/`，发布 SVG 和正式样本；更新 FlagMega 开发规范中的完整 IR、partition 与 producer/consumer 验收合同。 |

编译器根因对应的可运行回归位于 `third_party/tle/test/Analysis/gather-axis-info.mlir`、`third_party/tle/test/GPU/membar-atomic-return-scratch-reuse.mlir`、`membar-owned-alias-byte-ranges.mlir`、`test_tle_mbarrier_lifetime*.mlir`、`wgmma-async-forwarding-loop.mlir`、`wgmma-async-loop-carry*.mlir` 和 `wgmma-legacy-mma-accumulator.mlir`。具体因果说明和历史 fail-before/pass-after 记录保留在本教程 `.local/reports/`，中文优化与性能记录保留在 `.local/iteration-log-zh.md`。

## GLM 完整四卡 decode 结果

45 层、E288/top8、H4096、cache4096、TP4、batch1、生成 8 token；原四张 H800、1980MHz、CUDA Graph。NFFN 与 FNNF 共 384 原始请求，扣除预先规定的 64 warmup 后保留全部 320 正式请求，没有删除 outlier。FlagMega 每请求取四 rank 最慢 wall TPOT，vLLM 使用官方 decode TPOT；该指标排除 prefill 和首个生成 token。

| 上下文 | FlagMega TPOT | vLLM TPOT | 延迟降低 |
| --- | ---: | ---: | ---: |
| 32 | 6.399618ms | 6.753940ms | 5.246% |
| 128 | 6.496676ms | 6.789592ms | 4.314% |
| 1024 | 6.650775ms | 6.827689ms | 2.591% |
| 2048 | 6.843258ms | 6.892736ms | 0.718% |

每个 Flag 段均快于本顺序两个 Native 段，四个 Flag 段也均快于四个 Native 段。2048 的最小段间余量为 21.512µs。完整 Gold、状态比较、26 编译 key、228 调用零 spill 及独占/无 thermal 检查通过。正式对照耗时约 40.5 分钟。

结果、全部正式样本和图表见 [readme.md](readme.md) 与 [summary-20261101.json](figures/summary-20261101.json)。当前完整模型采用 Base>=1024 的 96/32 score/query CTA 分配，短路径保持 64/64；后续 GPU5 上的 104/24、108/20 局部候选未接入正式模型，也不计入整模收益。

## 提交前检查与未完成项

- 705 份改动 Python 文件通过语法检查；原生 C++ 新增行的 TLE guard 检查通过。已验证模型相关文件及固定 SDK patch 保留原字节，whitespace 检查排除合法 patch 上下文并允许原文件末尾空行。
- program/DAG/函数入口及 shared readonly owner 的 CPU 回归：51 passed、15 skipped。
- GLM 本地 CPU 回归：58 passed、3 skipped。此次发现旧 column-overlay 测试输入遗漏正常调用必有的 `parameters` 字段，实际 2 failed/56 passed/3 skipped；仅补齐测试输入后全组通过，模型实现和原断言保持。
- 改动的 FlagMega 测试抽取了 266 个文件；达到预定 5 个失败停止点时为 654 passed、157 skipped、5 failed。失败涉及三个 C++ artifact 测试、一个 elementwise family 期望和一个 FP8 workspace 合同，未将其记录为全量通过。
- 全量 FlagMega 检查仍有 collection/API 及架构归属断言失败。重复测试文件名用 pytest importlib 模式处理；`pipeline_source` 的旧私有 slot/drain API 在本次起点已不存在，保留实际 collection 失败。其他失败仍需逐项定位。
- 含 prefill 的完整请求吞吐仍低于 vLLM。原始 IR fresh 重建在 `compile_model.py::measured_physical_tir` 仍因两个历史固定 kernel 名未匹配而违反 6 个更新的合同，实际产生 7 个更新；没有添加 hash 白名单、放宽合同或绕过规划。直接脚本的三个包导入问题已有回归修复。
- 当前性能验收对应正常 API 载入的已生成 artifact；合并新的上游代码后不能沿用旧验收直接宣称新源码也已完成整模 GPU 验证。

`reports/flagmega` 禁止保留。本次及上游同目录的任务记录统一归入教程 `.local/reports/`；`.local/` 的实验、原始回执、构建产物和临时张量文件不纳入提交，`third_party/triton_shared` 的 gitlink 不纳入提交。
