# 工作区总结（2026-11-02）

本次保存累计的 TLE/FlagMega 开发状态、回归用例及 H800 教程结果。完整 GLM decode 的性能结论已有四卡验证；这不代表所有后端、全量测试及原始 IR fresh 重建均已通过。

最新：N8 原生 FP8 累加能力与 scan explicit-result conversion 已因果修复，20 个 GPU 回归、6 个相关 lit 测试通过。已验收 recipe 接入 up N8/stage8、down N8、join 4 warps 和 DSA 单 writer 整数可见性发布；仓库入口正常重新生成的完整 manifest/26 个源码与验收候选相等，23 个 helper 测试通过、1 个跳过。原四卡 Gold100、792 次后续状态比较、228 调用零 spill、双 stream/CUDA Graph 均通过。本轮八段全部 320 正式请求的严格性能门槛通过：四上下文 TPOT 6.424/6.515/6.686/6.856 ms。2048 最小余量约 0.019 ms；历史更快的合法 Native 轮次继续保留，完整请求含 prefill 仍慢，原始输入 fresh 重建仍有已知失败。详细数据、因果说明及范围见[本轮验收](performance-acceptance-20261102-zh.md)。下文保留此前历史结果。

合并后 strict 数值失配的 KDA 原生 FMA/归约布局和 DSA 指数/normalizer 修复仍保留，见[严格浮点修复记录](strict-fp-fix-20261101-zh.md)。mode0 的 1830 MHz 旧诊断没有计作 1980 MHz 结果。

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

## 上游同步与合并验证

在独立工作树合并原本落后的 42 个上游提交（`67ab9fa39`），逐处解决 18 个文件冲突。保留多行 prefill、readonly layout specialization、HIP host registration、严格浮点编译合同，以及本地的 program/DAG、每程序参数与资源、prepared rendering、共享 owner 生命周期。renderer 版本升至 v23，避免把合并后的生成合同误认为 v21/v22 缓存。

合并检查检出并处理了两项因果问题：

- `targets/nvidia/workspaces.py::attach_workspace_requirements` 的统计 workspace 不能按单行输出分配。输入有 4 行而输出仅 1 行时，错误大小 `(128,)` 的实际回归失败；保留上游按输入 local rows 分配 `(128,4)` 的合同，并保留 finalize 对输出 token 的寻址。没有改变该回归断言。
- `targets/iluvatar_bi_v150.py::corex_triton_implementation_model` 原先仅检查是否能找到模板；通用模板可找到时，`mma_v3` 候选进入不支持 MMA3 的 COREX 目录，原默认候选与教程 CTA 两个回归实际失败。注册时同时检查候选声明的 instruction requirements，保持受支持的指针 pipeline 和 NVIDIA MMA3 实现；原两项测试及新增整个目录的要求验证通过。此修复针对能力匹配入口，没有增加候选 ID 白名单。

最终合并检查：

- GLM、program/DAG、函数入口、共享 owner：109 passed、18 skipped。
- 目标、prefill/RMS、QKV、embedding、函数 ABI、source-rdata 与 launch 邻域：172 passed、72 skipped。当前 NVIDIA 安装不提供 HIP JIT，四项 AMD 乘法编译检查的实际失败保留，不计入这组可执行检查。
- paged attention 的 online BF16、row softmax、提前 value load、clamped tiles、block-ID prefetch 组合：32 份生成源码通过 Python 编译；原正向/反向 softmax 的 step width、掩码与 MMA 路径保持。
- 合并后 2673 份 Python 文件语法检查通过，原生编译器合并新增行的 TLE guard 检查通过。两个旧测试预期同步到实际 v11 distribution 合同及上游 COREX 专属 QKV 模板，未改变运行实现来迁就旧预期。

这些检查没有重新运行合并后完整四卡性能，也没有重新构建全部后端编译器。前述全量测试和 fresh 重建限制仍保留；本次推送是包含已验证 decode 结果的开发状态提交。
