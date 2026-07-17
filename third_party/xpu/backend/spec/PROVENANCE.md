# FlagTree XPU — main-tree source overrides (whole-file vendored replacement)

> 机制：`add_triton_object`（top `CMakeLists.txt`）对每个源文件检查本目录下是否有同「项目相对路径」的覆盖文件；有则 XPU build 编本副本，非 XPU 编主树原版。EXISTS 门控，非 XPU / 无覆盖零影响。
> 该机制用于保持共享主树源码不含 XPU 专属语义，同时允许后端维护完整的源文件替换。

## 为什么用整文件替换而非就地改主树
主树为所有后端共享，禁止就地改（Q0b）。这些副本**以 FlagTree 主树文件为底**（保留 `flagtree_hints` 等本地适配）+ 叠加 XPU 专属改动；**不以 internal 版为底**（实测 `Ops.cpp` 与 internal 有 162 行冲突 gap，如 `LoadOp::build` 的 `flagtree_hints` vs `offsetState/syncMode`）。

## Provenance（派生基线，用于反 drift）
- 派生自 FlagTree `main` 提交：`40a57023ddbd075c732eefb6886d6d3a152a181e`
- 迁移目标 internal Triton：`triton_3.6` @ `d3c64dd65e401239608164d6be4d893b261b4869`
- 生成时间：2026-07-10；2026-07-16 对最新 `main` 复核

上述主树源文件在原始派生提交 `b26ec15c65bea71f4e011c7626f7de9f5146937e`
与当前 `main` 基线之间内容一致；本次更新提交号用于准确反映 PR 的实际基线。

## 覆盖文件清单（副本 = 上述 FlagTree 文件 + 下列改动；括号为相对 pristine 的变动行）
| 覆盖路径 | 叠加的 XPU 改动 | 变动行 |
|---|---|---|
| `lib/Dialect/Triton/IR/Ops.cpp` | DotOp::verify 放宽 i8×i4(w4a8) · ReshapeOp::fold 去 `!getAllowReorder()` · BitcastOp::verify vector↔vector | 46 |
| `lib/Dialect/Triton/IR/Traits.cpp` | verifyTensorSize：允许非 pow2 元素数；verifyTensorLayouts：SliceEncoding 递归解析到 triton_xpu parent | 38 |
| `lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp` | ArithConstantSplatOpConversion splat guard | 2 |
| `lib/Dialect/TritonGPU/IR/Dialect.cpp` | ceil-offset：`getTotalElemsPerThread`/`getElemsPerThread(Attribute,shape)` 对 XPU 层（ClusterLayoutAttr / XPU-backed SliceEncoding）走 ceil-based 派发（新增 `isXPUBackedLayout` + `#include TritonXPU/IR/Dialect.h` + 文件内前置声明 `getElemsPerThread(Attribute,shape)`）；绕开泛型 LinearEncoding 的 pow2 断言 | 59 |
| `include/triton/Dialect/Triton/IR/TritonTypes.td` | 将 `TT_Vector`/`TT_VectorTensor` 组成的 `TT_VectorLike` 加入 `TT_Type`，允许 XPU vectorize 后的 `tt.extern_elementwise` 等主 Triton op 接受 vector-like operand/result | 6 |
| `include/triton/Dialect/Triton/IR/Traits.h` | XPU legalize 支持大 tensor，将共享的 1M element verifier 上限放宽为 `INT_MAX` | 1 |
| `python/triton/runtime/jit.py` | 将 XPU `xpubin` launch grid 注入编译 options，保持编译时 grid 与实际 launch 一致 | 5 |
| `python/triton/language/semantic.py` | 在 frontend IR 显式实现 masked load 的 `other`/zero 语义，避免 GM2LM 丢失无效 lane 值 | 10 |
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
- 本次 masked-load 兼容修复的契约是在 frontend IR 中显式生成
  `where(mask, loaded, other)`，并以真实消费该值的 `layer_norm_backward`
  作为行为验收；internal Triton 与修复后的 FlagTree 均通过对应 backward 用例。
