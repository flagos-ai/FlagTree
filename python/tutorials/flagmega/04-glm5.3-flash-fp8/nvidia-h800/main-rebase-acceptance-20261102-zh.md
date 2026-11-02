# Main rebase 与 H800 重新验收

当前基底：main 7f25177d2af4eb49dd8193fec31e03070ea13c38；首次完整性能对照基底为 9cc00c5188bb711d1a6c409849c13062bef2ea5c，main 新增提交仅涉及 CI 记录格式。保留分支 merge commit 的最终修改，原生 TLE C++ 修改仍受 __TLE__ 保护；reports/flagmega 已删除。验收使用当前 Python 源码和正常 CMake 构建的独立 Core，不覆盖公开或历史 Core。

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

第一版独立 Core（v7）的完整 Gold 四 rank 各 100 次请求通过，八段 NFFN/FNNF 对照真实退出 0，独立审计确认数值和硬件条件通过、性能失败。384 原始请求（64 warmup、320 formal）全部保留，无异常值剔除。

| 上下文 | FlagMega TPOT (ms) | vLLM TPOT (ms) | FlagMega 变慢 |
| --- | ---: | ---: | ---: |
| 32 | 6.417071 | 6.330361 | 1.37% |
| 128 | 6.515189 | 6.349773 | 2.61% |
| 1024 | 6.687358 | 6.426791 | 4.05% |
| 2048 | 6.907239 | 6.463474 | 6.87% |

这是四个 Flag leg 和四个 Native leg 各上下文所有 40 个正式请求的中位数，不能覆盖更严格的逐段门槛；四种上下文的逐段门槛全部失败。2048 完整请求中位数约 13.635 s，vLLM 约 0.520 s；TPOT 排除了 prefill 与首个输出 token，完整请求优势尚未达到。CUDA Graph 和双流启用，原 GPU1–4 独占、实际 1980 MHz、无热降频。正式对照真实用时 2501.94 s。

## 异步填充契约的根因修复

目标是在保留 cp.async 搬运能力时保持 mask 排除位置的全部位模式。main 新增的零值条件在经过 ttg.convert_layout 后漏掉合法字节零；其数值零比较也将浮点负零错误视为 cp.async 的字节零。TLEOptimizeLocalPointerAsyncStores.cpp::matchAsyncStoreCandidate 改用 TLE-owned CopyFill.h::hasZeroFillBits，穿过布局转换并检查实际位模式。program_011 的 cp.async 数量恢复 410→462；24/26 个 program 的全部 PTX 指令与 rebase 前一致，两个仍有 main 布局/点积变化，不能声称全体代码等价。

TLEFrontendSemantic::scalar_constant 的继承路径在 value == 0 时生成 null，先于 lowering 丢失 -0.0 的符号。TLE-owned frontend 改为按实际浮点精度生成负零，不修改 native semantic.py。四种浮点位模式回归在修复前 4 failed / 12 passed，修复后通过。

NVIDIA LoadStoreOpToLLVM.cpp::AsyncCopyGlobalToLocalOpConversion::matchAndRewrite 原先忽略 other，显式异步 copy 的负零、1、负无穷在四种浮点类型上全部被写成零（12 failed / 20 passed）。修复保留有效位置的异步 copy，仅在无效位置使用 st.shared 写入 other，两类操作不会覆盖对方。所有原生新增行受 __TLE__ 保护。冗余线程谓词的空值表示全体线程参与，谓词组合统一使用 maybeAnd；直接对空值做 AND 的诊断版本产生 6 项编译失败，最终实现消除该缺陷。

最小复现与回归：third_party/tle/test/GPU/local-pointer-async-copy-fill.mlir 使用 triton-opt --triton-tle-optimize-local-pointer-async-stores | FileCheck；python/test/tle/unit/test_async_copy_fill_bits.py 使用 pytest -q，覆盖 FP16/BF16/FP32/FP64 的 ±0/1/-inf，8/16/32/64 位整数的逐元素运行时填充，以及混合、全开、全关 mask。最终独立 Core（v15）44 项填充 GPU 回归通过，同时断言有效位置仍有 cp.async；包含原有 pipe/TMA、FP8 N8、scan 的扩展套件 109 项 GPU 全部通过，CPU 130 passed / 51 skipped。整模 Source、全部调度和计时循环不变。进一步检查发现，将非零填充直接放在 opaque PTX 的谓词之后，会保留 LLVM 范围分析已经证明永远不执行的 shared store。原实现使专家 program_012/018 各增加约 20 行 PTX 和 4 个谓词寄存器；最小全真 mask 回归四项失败，包含只有 LLVM 才能证明全真的表达式后共八项失败。最终将填充置于 LLVM 条件分支，使无效分支可被优化删除；转换后的 LLVM 标量位模式负责区分零字节与负零，并识别全真常量 mask。v18 的 48 项回归全部通过，涵盖 ±0/1/-inf、运行时逐元素整数填充、空/全/混合 mask 与后端才能证明全真的 mask，同时保持异步复制。

最终 Core 为正常 CMake 构建的独立 v18；公开和历史 Core 未改变。正常候选生成及 fresh 原始输入重建分别实际用时 128.44/140.21 秒，fresh semantic hash 仍为 b0d11885153361af9c38246b9be979a65c0a21e5456a4686784f05a919779f97，输出未采用作性能候选。后续仅修改 NVIDIA 低层 copy lowering，1293 个生成相关 Python 源文件逐一核对不变，因此保留既有正常生成产物，通过正常缓存接口用当前 Core 重新预编译全部 program；没有复制 Core、cubin 或 checkpoint payload。最终四卡 Gold 与正式性能对照待完成。

范围检查：TLE MLIR 158 项通过（比 v7 新增一项）。原生 SM100 TCGen5 邻接 membar 测试 wait_after_mma 出现额外 CTA barrier，根因是 lib/Analysis/Membar.cpp::addEffectIntervals 对无 allocation ID 的 borrowed shared 参数作保守 allShared 区间，产生与等待 barrier 的潜在 WAR；H800 不执行该 TCGen5 路径，未删断言或禁用 pass。本任务没有 SM100/Iluvatar 硬件结论；原生邻接三项通过、一项失败的事实保留。

本地证据：.local/rebase-main-reaccept-20261102-ff363313e。各 phase 保留真实顶层 OS wait/退出码；构建、失败前/后、CPU/GPU/lit、fresh、生成与预编译均有独立日志。中文优化记录：.local/iteration-log-zh.md。
