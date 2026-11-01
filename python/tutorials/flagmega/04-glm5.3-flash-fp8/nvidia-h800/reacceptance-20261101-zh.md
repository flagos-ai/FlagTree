# 合并后重新验收（2026-11-01）

**验收未通过。** 对已推送的 `53843322b370413ced3e86ff3f2eb25a2464b14d` 正常重建 NVIDIA/TLE 编译器，修正本轮发现的 GEMV 模板生成问题后，完整 TP4 模型在第 2 个生成 token 上失败：四个 rank 均得到 `646`，参考为 `284`。没有新的正式性能结果；README 的 6.4–6.8ms 表格属于合并前 Source6881d6c8… 的历史验收。

## 验收对象与实际结果

模型保持 45 层、E288/top8、H4096、cache4096、TP4、batch1、8 个生成 token。使用原四张 H800、固定 1980MHz 的配置、真实双 stream DAG 和 prefill/decode CUDA Graph。编译器通过正常 CMake 增量构建生成独立 Core；没有替换历史验收 Core。

本轮正常 artifact API 从已有完整 final IR 重新生成源码并经正常 runtime 装载。当前源码 SHA256 为 `2dbf9492d1fbbf85ee0a3e0fb77552e4b52ae85345f8c9bd8addb89422f1b1a7`，语义为 `7a073b51c9d72f7383aa0a5763697ee53917de427451b6f57841a4468c43a6c1`。该实验没有绕过或替代原始 IR fresh 重建：后者仍失败。

| 检查 | 实际结果 |
| --- | --- |
| 当前编译器正常构建 | 退出 0，118.999s |
| 完整 final IR 正常生成与 CPU runtime 装载 | 退出 0，118.265s |
| 当前 ABI 和编译选项的 CPU 编译 | 8 个 worker 均退出 0；26 个入口、228 次调用，零 spill；89.485s |
| GLM/program/共享 owner CPU 邻域 | 116 passed、19 skipped |
| GEMV 模板回归 | 修正前 1 failed/2 passed，修正后 3 passed |
| TLE 共享别名及 mbarrier 生命周期 GPU 回归 | 8 passed |
| TLE TMA/WGMMA ring 与显式 layout GPU 邻域 | 15 passed |
| TLE MLIR 全组 | FileCheck 范围修正后 143 passed、1 failed |
| 全量 FlagMega CPU 收集 | 4 个 collection error，退出 2 |
| 原始 IR fresh 重建 | 物理替换 7 项，合同要求 6 项，退出 1 |
| 完整 TP4 Gold | 退出 1，340.694s；四 rank 在 step=1 选错 token |
| 仅开启 FP fusion 的隔离诊断 | 退出 0，196.474s；两次完整 8-token 请求和原状态断言通过；不计作完整 Gold |
| 本轮正式 Native/Flag 性能比较 | 未执行，暂无新 TPOT |

Gold 使用当前 Core、当前 runtime 的正常编译选项；CPU/GPU 的实际 26 个编译 key 相等，228 次调用的资源报告均为零 spill。首个生成 token `3847` 及该步原有状态断言通过，随后第一个 decode step 在 `gold_worker.py::main` 的原 token 断言失败：`AssertionError: (rank, 1, 646, 284)`。完整 8-token、多请求和四种上下文验证没有完成。控制器实际等待子进程退出，源输入冻结检查通过，不能将此失败记录解释为验收成功。

Gold 初始化约 321.5s，其中正常 CPU 构建约 39s、device load 约 268s，后者含约 255–258s 的正常权重装载；这些阶段互相嵌套。当前 26 个入口从新缓存编译约 89.5s，耗时最长的 `program_011` 为 66.9s。拆分 program 已减少重编译范围，但权重装载和大入口冷编译仍占主要时间。

## 本轮因果修正及风险范围

`local_optimizations/gemv.py::prepare` 只识别 `{{ entry.kernel_decorator(call) }}`。上游 `codegen/triton/kernels/dense_matmul/_function.py.jinja` 增加 `prefer_inline=true` 后匹配数变为 0，正常模型发布失败。现在识别并保留完整装饰器实参，在其前面插入 warp-row 布局；仍检查唯一装饰器。GEMV 的乘加、归约、布局转换和上游内联选择保留。真实模板 fail-before/pass-after 回归同时覆盖旧装饰器、新装饰器和非 module 模板，并检查去掉两处预期插入后整份模板原文相等。

`third_party/tle/test/GPU/coalesce-unreachable-memory.mlir` 在多个 `CHECK-LABEL` 之间使用局部 FileCheck 变量 `VEC`。TLE lit 默认开启 `--enable-var-scope`，标签会清除局部变量；实际编译输出中的可达路径已经正确向量化为 sizePerThread=4。将此共享匹配变量改为全局 `$VEC` 后原测试通过，编译器和向量化断言保持。全组剩余失败 `test_tle_inline_argument_hints.mlir` 仍然保留，没有降低其提示保留断言。

这些修正解决生成入口和测试范围问题，没有解决当前完整模型的错误 token。没有修改原生 C++ 源码、模型数学、program 的 grid/warps 或 DAG 来压过数值失败。

## 数值诊断与未完成项

旧 Source6881d6c8… 与当前生成源码的完整模块 AST 相等；26 个函数、模块常量、签名及装饰器语义都一致。renderer 版本和注释/排版引起文本 SHA 变化。228 次调用的 grid、warps、依赖、stream、输出写入和同步合同也相等。

本轮发现编译选项发生变化：`runtime/module.py::_prepare_external` 将 IR 的 `floating_point_contract="strict"` 传给 `runtime/nvidia_launch.py::compilation_options`，后者设置 `enable_fp_fusion=False`；合并前验收启用了 FP fusion。

隔离实验保持相同的新 Core、完整模型、生成源码、ABI、grid/warps、DAG 和 CUDA Graph，逐入口核对唯一编译选项差异为 `enable_fp_fusion=False→True`。26 个入口的隔离预编译为 93.680s、零 spill；四卡实验为 196.474s，正常装载权重约 89–99s，初始化约 166.5s。权重装载时间变化单独保留，不记作模型优化收益。

开启 fusion 后，四个 rank 的两次完整 8-token 请求都得到 `[3847, 284, 1373, 8282, 4218, 3755, 947, 906]`，每 rank 16 行 token/状态记录与原参考一致，实际 GPU key 与对应 CPU 编译证据相等、228 次调用零 spill。独立核验退出码、实际 Core 和状态记录通过。隔离结果将排查范围缩小到严格浮点合同与编译行为的差异；具体哪处运算的舍入或编译结果出错仍未定位。

该实验仅检查两次单 token prompt 的完整请求，没有完成正式 100 请求、四种上下文 Gold 或性能比较。正式 runtime 的 `strict` 处理保持原样，没有提交恢复旧选项、放宽数值断言或降规模的“修复”。当前源码验收仍为失败；下一步必须定位首个发生差异的 decode 运算及其正确舍入合同，再重新验收。

全量 CPU 的四个收集失败分别涉及未限定的 `test_sum`、已不存在的 `_configure_inter_stage_buffer_slots`、未限定的 `test_form_qkv_rope_with_cache` 和未限定的 `test_flagos_results` 导入。完整物理 catalog 的额外核验仍未通过；原始 IR 的历史固定 kernel 名排除规则也未修正。以上限制均独立于正常 final-IR 装载实验保留。

## 本地证据与复现输入

证据集中在 [.local/reaccept-20261101-53843322b](.local/reaccept-20261101-53843322b/)。正常编译/API/runtime 驱动与环境、完整 checkpoint/cache 路径保存在该目录 `continuation-state.json`；需要原模型输入、固定 FlagCX SDK 和可用原四卡。驱动采用新目录的独占创建，保留已有结果后才能使用新输出目录重跑。

- 构建：`build_current_compiler.py`、`compiler/receipt.json`。
- 生成：`publish_current.py`、`trial-v4/exit.json`、`current-codegen-launch-contract.json`、`generated-module-AST-identity.json`、`old-new-generated-function-AST.json`。
- 编译：`run_prewarm.py`、`prewarm-v5/controller-exit.json`、`prewarm-v5/input-proof.json`。
- 失败复现：`run_gold.py`、`gold_worker.py`；输入 prompt=`[11]`，第 2 个生成 token 的原断言。实际结果为 `gold/exit.json`、`gold/run/current/rank-{0,1,2,3}.json`、`failed-Gold-independent-result.json`。
- 因果回归：`GEMV-decorator-fail-before.log`、`GEMV-decorator-pass-after.log`、`TLE-lit-global-check-variable-fixed.log`。
- 隔离诊断：`run_fusion_diagnostic.py`、`fusion_diagnostic_worker.py`、`prewarm-fusion-on-diagnostic/input-proof.json`、`fusion-diagnostic-isolation.json`、`fusion-on-diagnostic/exit.json`、`fusion-diagnostic-independent-result.json`。
- 原始 IR 和全量收集失败：`fresh-before.log`、`FlagMega-full-CPU-package-fixed.log`。

优化、正确性及性能变化另记入 [.local/iteration-log-zh.md](.local/iteration-log-zh.md)。本轮没有创建 `reports/flagmega`，实验和构建产物不纳入提交。
