# Main rebase 与 H800 重新验收

基底：main 9cc00c5188bb711d1a6c409849c13062bef2ea5c。保留分支 merge commit 的最终修改，原生 TLE C++ 修改仍受 __TLE__ 保护；reports/flagmega 已删除。验收使用当前 Python 源码和正常 CMake 构建的独立 Core，不覆盖公开或历史 Core。

## 修复与最小复现

| 根因位置与触发 | 因果修复与回归证据 |
| --- | --- |
| TLE LowerTokenCalls.h：带 full_count=2 的多 TMA writer 经 helper-call lowering 被初始化为 1 | 统一保留显式 full_count；test_tle_multi_tma_writers.mlir 失败前/修复后保持 2 producer、8192 字节 TMA 和 128 reader 的断言 |
| TLE Ops.cpp::verifyPipeAttrs：错误要求控制 pipe 有 payload | 恢复 fieldless 协议，保留所有已有字段/tiled-SMEM 校验；noinline 控制边及三/五 stage 的循环关闭 GPU 回归 |
| TLE gpu/core.py::alloc：物理线性 arena 被当作非二次幂逻辑矩阵 | 将物理 MemDesc extent 与自动逻辑矩阵规划区分；精确 arena/alias/masked copy GPU 回归及 main 80×256 logical TMA/initializer 控制 |
| TLE gpu/types.py::swizzled_shared_layout：JIT deepcopy 后结构不变却比较不等 | 对完整 encoding 字段作结构比较与 hash；同一 constexpr layout 的循环 pipe 从全局量变化错误恢复通过 |
| TLE distributed.py / runtime/_distributed.py：单通道 intra ABI 与 unified mesh ABI、显式多 context 的生命周期合并错误 | 两种契约各自验证；枚举在 ABI 边界显式转换；显式 contexts 独立，teardown 使全部旧句柄失效；静态/动态 slot 与 host ABI 回归 |
| TritonInlinerInterface::handleArgument：inline 丢失 callee 参数的轴提示 | 在 __TLE__ 中为该 call 的值保留提示，避免污染其他 caller 用途；pointer/integer/tensor 三项 MLIR 回归 |
| local_optimizations/compile_model.py::measured_physical_tir：按含摘要的 kernel 名识别 dense FFN，down 名更新后误选第七项 | 根据原调用的 dense MLP 语义及 lowered_from 保留两项原调度；六项覆盖合同不变，最小改名回归 fail-before/pass-after；完整原始输入正常重建退出 0 |

同时修正重放冲突造成的 barrier 常量、条件编译、旧枚举、重复 mask 参数和 close-tag 类型声明；更新旧 MLIR 的 pipe identity、operand segments、enum 属性与物理 barrier 顺序，保持原语义断言。

## 验证范围

当前独立 Core 构建退出 0；157 项 TLE MLIR 全部通过（含 AMD pipe lowering、逻辑 SMEM、多 TMA writer、FP8 N8 和 scan）。CPU 130 passed / 51 skipped，GPU 65 passed；CPU 跳过项不计入验证。Iluvatar 仅有 target/codegen 的 CPU 检查，没有该硬件验收。

原始输入 fresh 流水真实退出 0，约 140.79 秒，正常生成 bufferized_tir，semantic hash b0d11885153361af9c38246b9be979a65c0a21e5456a4686784f05a919779f97。正式整模候选使用原已验证 final IR；fresh 输出尚未作为正式性能候选，两个结论分开记录。

正式候选完整 45 层/E288/top8/H4096/cache4096/TP4/b1/out8；Source ebc471b31f7df9adab032f6e759578ac44b1ad7bb7701f4eb2037ea7efd52c00，228 调用/26 独立 program。正常生成约 120.52 秒，当前 Core 冷预编译约 94.83 秒，全部零 spill；host layout 修复后的正常缓存复用约 16 秒。首次 Gold 初始化为修复 layout 主动结束，真实退出 1，不能计作数值通过或失败。

完整 Gold 与八段 NFFN/FNNF 正式 vLLM 对照进行中。历史性能数值不作为此次 rebase 的验收结果。

本地证据：.local/rebase-main-reaccept-20261102-ff363313e。各 phase 保留真实顶层 OS wait/退出码；构建、失败前/后、CPU/GPU/lit、fresh、生成与预编译均有独立日志。中文优化记录：.local/iteration-log-zh.md。
