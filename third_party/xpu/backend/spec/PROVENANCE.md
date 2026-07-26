# FlagTree XPU — main-tree source overrides (whole-file vendored replacement)

> 机制：顶层 CMake 调用 `cmake/FlagTreeBackendSpec.cmake` 中的
> `flagtree_apply_backend_source_overrides`，自动扫描本目录下 `lib/` 中的
> 可编译源码，关闭镜像路径对应的主树源码，并把 XPU spec 副本加入原 target。
> 无需在 XPU CMake 中逐文件维护 target 映射。
> 该机制用于保持共享主树源码不含 XPU 专属语义，同时允许后端维护完整的源文件替换。

## 为什么用整文件替换而非就地改主树
主树为所有后端共享，禁止就地改（Q0b）。这些副本**以 FlagTree 主树文件为底**（保留 `flagtree_hints` 等本地适配）+ 叠加 XPU 专属改动；**不以 internal 版为底**（实测 `Ops.cpp` 与 internal 有 162 行冲突 gap，如 `LoadOp::build` 的 `flagtree_hints` vs `offsetState/syncMode`）。

## Provenance（派生基线，用于反 drift）
- 派生自 FlagTree `main` 提交：`69fca32372ef94541579aed090a0fd582a094f89`
- 迁移目标 internal Triton：`triton_3.6` @ `d3c64dd65e401239608164d6be4d893b261b4869`
- 生成时间：2026-07-10；2026-07-24 对 PR 当前 base 重新同步

4 个 `.cpp` 副本已重新同步当前 base 的 FlagOS license header，并保留下表所列
XPU 差异。

## 覆盖文件清单（副本 = 上述 FlagTree 文件 + 下列改动；括号为相对 pristine 的变动行）
| 覆盖路径 | 叠加的 XPU 改动 | 变动行 |
|---|---|---|
| `lib/Dialect/Triton/IR/Ops.cpp` | DotOp::verify 放宽 i8×i4(w4a8) · ReshapeOp::fold 去 `!getAllowReorder()` · BitcastOp::verify vector↔vector | 46 |
| `lib/Dialect/Triton/IR/Traits.cpp` | verifyTensorSize：允许非 pow2 元素数；verifyTensorLayouts：SliceEncoding 递归解析到 triton_xpu parent | 38 |
| `lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp` | ArithConstantSplatOpConversion splat guard | 2 |
| `lib/Dialect/TritonGPU/IR/Dialect.cpp` | ceil-offset：`getTotalElemsPerThread`/`getElemsPerThread(Attribute,shape)` 对 XPU 层（ClusterLayoutAttr / XPU-backed SliceEncoding）走 ceil-based 派发（新增 `isXPUBackedLayout` + `#include TritonXPU/IR/Dialect.h` + 文件内前置声明 `getElemsPerThread(Attribute,shape)`）；绕开泛型 LinearEncoding 的 pow2 断言 | 59 |
| `include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h` | `maybeDeduplicate` 遇到非 2 的幂 constancy 时保守地跳过 dedup，避免 XPU layout 触发主树 pow2 断言 | 3 |
| `include/triton/Dialect/Triton/IR/TritonTypes.td` | 将 `TT_Vector`/`TT_VectorTensor` 组成的 `TT_VectorLike` 加入 `TT_Type`，允许 XPU vectorize 后的 `tt.extern_elementwise` 等主 Triton op 接受 vector-like operand/result | 6 |
| `include/triton/Dialect/Triton/IR/Traits.h` | XPU legalize 支持大 tensor，将共享的 1M element verifier 上限放宽为 `INT_MAX` | 1 |
| `python/triton/runtime/jit.py` | 将 XPU `xpubin` launch grid 注入编译 options，保持编译时 grid 与实际 launch 一致 | 5 |
| `python/triton/language/semantic.py` | masked load 默认对齐 internal Triton；仅在 `TRITONXPU_OTHER_SIM` 下显式模拟 `other`/zero 语义 | 10 |
| `language/xpu/libdevice.py` | 198 个 XPU extern 全部从 legacy `_builder` 迁移到 Triton 3.6 `_semantic` 注入 | mechanical |
| `backend/compiler.py` | XPU float modulo 直接使用 `libdevice.fmod`，与 internal 的 LLVM frem 语义对齐 | 8 |
| `language/xpu/libdevice.py` | 修正 `rsqrt(fp64)` 使用 fp64 ABI symbol `_ZN3xpu6rsqrtfEd`；避免错误调用 fp32 symbol | 1 |
| `python/triton/tools/build_extern.py` | 生成器模板从 legacy `_builder` 改为 Triton 3.6 `_semantic` | 4 |

> `Dialect.cpp` 关键点：internal 在主树 header `TritonGPU/IR/Dialect.h` 加了 `getElemsPerThread(Attribute,ArrayRef)` 声明；FlagTree 该 header **无**此声明（Q0b 不改共享 header），故 `getTotalElemsPerThread` 里 line~112 的 `getElemsPerThread(layout,shape)` 会误配到 `getElemsPerThread(Type)` 报 `Attribute→Type` 转换错。修法：在本 vendored 副本内 `namespace mlir::triton::gpu` 前置声明该 overload（不动主树 header）。已经 XTDK clang22 实测编译+链接通过。

> 注：`Dialect.cpp` 副本保留了 pristine 顶部的 `flagtree_spec.h` 原生守卫（`#if __has_include("flagtree_spec.h")` / `#ifndef FLAGTREE_SPEC_Dialect_TritonGPU_IR_Dialect`）。XPU 未提供 `third_party/xpu/backend/spec/include/flagtree_spec.h`，故 `__has_include` 为假、宏未定义、整个 body 正常编译——守卫无副作用。新增的 `#include "triton/Dialect/TritonXPU/IR/Dialect.h"` 经 XPU 后端 include dir（`third_party/xpu/include`）解析，仅存在于本副本。

## 维护须知（drift）
FlagTree 主树每次升级上述任一文件，**必须**用新版主树文件重做副本（以新主树为底重叠 XPU 改动），并更新本文件的派生提交号。否则 XPU 编译的是过期主树逻辑。

## XPU 行为边界与对齐验证

- XPU launch grid 通过 cluster/core 映射执行，不能用 CUDA 风格的连续
  `program_id` 输出布局验证。internal Triton 与 FlagTree 均以编译 metadata
  中的 `grid` 为准；`grid=(5,)` 时两边都生成 `"grid": [5]`。
- 对 `tl.load(mask=..., other=...)` 的结果不做后续计算、直接无 mask 写回
  无效 lane，internal Triton 与 FlagTree 默认 lowering 均不会保证这些 lane
  等于 `other`。这是当前 XPU lowering 的既有边界，不应作为 migration 回归。
- 当前 masked-load 的默认契约与 internal Triton 对齐：不额外生成
  `where(mask, loaded, other)`；仅 `TRITONXPU_OTHER_SIM=1` 才启用显式
  `other`/zero 模拟。真实消费该值的 `layer_norm_backward` 仍作为行为验收。

## LLVM toolchain codegen boundary

本次移植验证中，FlagTree 与 internal Triton 的 XPU pipeline 保持一致，
包括 BF16 kernel 的 `add_tritonxpu_vectorize_pass`；不能通过关闭 vectorization
来规避结果差异。

对固定 seed 的 BF16 broadcast `mul`，两边在进入
`xpu.llvm.translate_to_asm` 前生成的最终优化 LLVM IR 在计算语义上完全一致，
仅有生成 Python 文件名/PID 的 debug metadata 差异。相同 LLVM IR 分别交给
internal `xtdk-llvm22` 与 FlagTree 使用的 `llvm_trust`：

- internal `xtdk-llvm22` `llc` 生成的 object 在设备上通过；
- `llvm_trust` `llc` 生成的 object 在设备上稳定出现 1,048,481 / 33,554,432
  （3.1%）BF16 mismatch；
- 交叉使用两套 `elfconv` 后，正确性仍跟随 object 的 `llc` 生产者，而不是
  `elfconv`，因此问题位于 `llvm_trust` 的 XPU LLVM codegen 阶段。

复现用例：
`tests/test_mul.py::test_mul_broadcast_shape[dtype2-shape_a2-shape_b2]`，
shape 为 `(1048576, 1)` 和 `(1, 32)`，dtype 为 BF16，random seed 为 0。

后续遇到同类问题时，按以下顺序验证：

1. 用相同 kernel、shape、dtype、launch metadata、环境和固定 seed 生成两边
   的最终优化 LLVM IR，并忽略非语义 debug metadata 后比较 executable IR。
2. 用两个 LLVM toolchain 分别对同一份 IR 执行相同 target 的 `llc`，保存
   assembly 和 object；记录 LLVM/XPU CodeGen 版本与 SHA256。
3. 分别用两套 `xpu3-elfconv-triton` 生成 XPUBIN，必要时交叉组合两套
   `llc`/`elfconv`，以区分 codegen 和 XPUBIN packaging。
4. 在相同 runtime 和设备上只注入最终 XPUBIN，运行相同固定 seed correctness
   test；只有当结果跟随 object producer 时，才能将问题归因到 LLVM codegen。

详细命令、artifact hashes、设备结果和交叉矩阵见
`../../docs/llvm-codegen-validation.md`。

本次 Triton 3.6 clean build、wheel 验证、150-run FlagGems 回归及 pristine
internal 对照结果见 `../../docs/triton-3.6-validation.md`。
