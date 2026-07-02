# PR #717 — XPU 后端集成清理（将改动收敛到 overlay）

提交：`[XPU] Clean up XPU backend integration to confine changes to overlay`
分支：`xpu-cleanup-20250625`
基线：`62af3c006 [XPU] Add XPU backend support for Triton (triton 3.6.x)`

## 1. 背景与目标

社区不允许将 XPU 专属逻辑合入主树（main tree：`include/`、`lib/`、`python/triton/`）。
本次清理的核心目标：

- **几乎所有 XPU 专属代码都必须落在 `third_party/xpu/` overlay 内**；
- 主树（共享树）的改动收敛到不可避免的最小集合，且每一处都能作为**与 XPU 无关的上游 TLE 兼容性修复**向社区解释；
- 构建产物（`.so`、`FileCheck` 等）不得进入版本库。

约束来源：主树 `lib/Analysis` 会被编译进 `TritonAnalysis`；overlay 通过
`third_party/xpu/CMakeLists.txt` 里的 `include_directories(BEFORE)` 让
`third_party/xpu/include/` 在编译期遮蔽（shadow）主树同名头文件。XPU 的辅助实现
真正落在 `third_party/xpu/lib/Analysis/NewAnalysis/Utility.cpp`（编译进
`TritonXPUAnalysis` 静态库）。理解这一点是本次所有取舍的前提。

## 2. 改动总览

主树（共享树）只剩 4 个文件被改动，其中 2 个是 C++ TLE 兼容性修复，2 个是 XPU python overlay 接线：

- `include/triton/Dialect/Triton/IR/Dialect.h` — TLE bugfix
- `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` — TLE bugfix
- `setup.py` — XPU python overlay 接线（+1 行）
- `python/setup_tools/utils/xpu.py` — XPU python overlay 逻辑（新增）

其余全部落在 `third_party/xpu/`。

> 说明：本提交中 `python/triton/compiler/compiler.py`、`python/triton/language/semantic.py`、
> `python/triton/runtime/jit.py` 三个主树 python 文件的改动，是相对基线 commit 的**回退**
> （删除了之前散落在主树里的 XPU 特判，例如 `is_xpu_backend` 分支、`float_mod` codegen hook、
> `TensorWrapper.numel`），即把 XPU 痕迹从主树移除，符合“收敛到 overlay”的目标。

## 3. 主树 C++ 改动（需向社区解释为上游 TLE 修复）

### 3.1 `include/triton/Dialect/Triton/IR/Dialect.h` — `SharedMemory` 必须无条件声明

问题：上游把 `SharedMemory` 结构体放进了 `#ifdef __TLE__` 守卫，但
`TritonOps.td:25` 的 `def SharedMemory : Resource<"::mlir::triton::SharedMemory">`
以及 `AtomicRMWOp` / `AtomicCASOp` 上的 `MemRead/MemWrite<SharedMemory>` 副作用
都是 **TableGen 无条件生成**的（围绕它的 `// flagtree tle` 只是注释，不是预处理守卫）。
因此非 TLE 构建会报 `mlir::triton::SharedMemory has not been declared`。

修复：移除 `#ifdef __TLE__` / `#endif`，让 C++ 声明与 TableGen 生成保持一致，并补充
持久解释性注释（非一次性提交注释）：

```cpp
// NOTE: SharedMemory must be declared unconditionally. It is referenced by
// `def SharedMemory : Resource<"::mlir::triton::SharedMemory">` in TritonOps.td
// and by the MemRead/MemWrite<SharedMemory> side effects on AtomicRMWOp /
// AtomicCASOp, both of which TableGen emits unconditionally (the surrounding
// `// flagtree tle` markers are comments, not preprocessor guards). Guarding
// this struct behind `#ifdef __TLE__` therefore breaks every non-TLE build with
// "mlir::triton::SharedMemory has not been declared". Keep it unguarded so the
// C++ declaration matches what TableGen generates.
struct SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
  StringRef getName() final { return "<SharedMemory>"; }
};
```

为什么不能移入 overlay：`TritonOps.td` 是核心、与 XPU 无关的代码，且无条件引用
`mlir::triton::SharedMemory`，无法在 overlay 里补声明。

### 3.2 `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` — 模板参数列表逗号位置

问题：`addDynamicallyLegalDialect<...>` 的方言列表中，`LLVM::LLVMDialect` 被
`#ifdef __TLE__` 守卫，但分隔逗号留在守卫**外面**（紧跟 `ub::UBDialect`），导致
非 TLE 构建时模板参数列表以悬挂逗号结尾，编译失败。

修复：把分隔逗号移入 `#ifdef __TLE__` 块内，使列表在 TLE / 非 TLE 两种配置下都合法，
并补充持久注释解释逗号为何必须 TLE-gated。

为什么不能移入 overlay：这是核心 conversion 代码，非 XPU 专属。

## 4. Overlay 改动（全部位于 `third_party/xpu/`）

### 4.1 Analysis 头文件对齐主树

- `include/triton/Analysis/Alias.h`
- `include/triton/Analysis/Allocation.h`
- `include/triton/Analysis/AxisInfo.h`
- `include/triton/Analysis/Membar.h`

这 4 个 overlay 头文件改为与主树同名头文件**逐字节一致**。原因：

- overlay 旧版 `AxisInfo.h` 把 `ModuleAxisInfoAnalysis::initialize` 声明为单参数版本，
  而主树 `.cpp` 定义的是双参数版本（`initialize(FunctionOpInterface, CallbackType=nullptr)`），
  链接期报 `undefined symbol: ModuleAxisInfoAnalysis::initialize(FunctionOpInterface)`。
- 这些头文件不含任何 XPU 专属符号；`Allocation.h` 中的
  `getScratchConfigForCvtLayout` / `getRepShapeForCvtLayout` 经确认无任何活跃调用方。

由于它们经 `include_directories(BEFORE)` 遮蔽主树头文件，对齐为逐字节一致后行为与主树完全等价。

### 4.2 `include/triton/Analysis/Utility.h` — 保留 XPU 扩展 + 适配新 LLVM API

保留 overlay 版本（含 XPU 扩展：`ReduceOpHelper`/`ScanLoweringHelper` 的 XPU 构造、
`xpu_op`、SMOffset、各类 XPU getter）。仅适配 LLVM API 迁移：

- `resolveCallable()` → `resolveCallableInTable(&symbolTable)`（需 `SymbolTableCollection`）
- 配套引入 `SymbolTableCollection symbolTable;`

### 4.3 `include/triton/Tools/Sys/GetEnv.hpp` — 引入命名独立的 XPU env helper

问题（运行期 abort）：`getBoolEnv` / `getStrEnv` 是头文件内 `inline` 函数 → 弱符号（weak symbol）。
链接期 overlay 版本与主树版本去重（dedup），主树定义（不含 `TRITONXPU_` 白名单）胜出，
导致读取 `TRITONXPU_BF16_ROUND_MID` 等变量时触发 `assertIsRecognized` abort（smoke STEP 2）。
之前尝试调整 include 顺序无效，因为这是**链接期去重**而非编译期遮蔽。

修复（方案一）：新增命名独立、不会与主树去重的 helper：

```cpp
static std::mutex getenv_mutex_xpu;

inline std::string getStrEnvXPU(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex_xpu);
  const char *cstr = std::getenv(env.c_str());
  if (!cstr) return "";
  return std::string(cstr);
}

inline bool getBoolEnvXPU(const std::string &env) {
  std::lock_guard<std::mutex> lock(getenv_mutex_xpu);
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return str == "on" || str == "true" || str == "1";
}
```

并新增 `#include <mutex>`。`mutex` 与主树 `getStrEnv`/`getBoolEnv` 的线程安全语义保持一致
（review 发现 R001 已修复）。

### 4.4 调用点改名（6 处）

- `lib/Conversion/TritonXPUToLLVM/LoadStoreOpToLLVM.cpp`：
  `TRITONXPU_BF16_ROUND_MID` / `TRITONXPU_BF16_FAST` 改用 `getBoolEnvXPU(...)`；
  保留 `axisAnalysisPass.getContiguity(ptr)`（主树无 `getPtrContiguity`）。
- `lib/Dialect/TritonXPU/Transforms/DtypeConvert.cpp`：`TRITONXPU_FP16_FAST` → `getBoolEnvXPU`。
- `lib/Conversion/TritonXPUToLLVM/XPUPrintOpToLLVM.cpp`：`TRITON_PRINT_VERBOSE` → `getStrEnvXPU`。
- `python/src/llvm.cc`：`TRITON_TUNE_BUFFER_LM_SIZE` / `LLVM_ERROR_LM_SIZE` → `getStrEnvXPU`。

## 5. 构建系统 / Python overlay 接线

- `python/setup_tools/utils/xpu.py`（新增，mthreads 风格）：`XPU_PYTHON_ROOT="third_party/xpu/python"`，
  `_merge_xpu_packages()`、`_merge_xpu_package_dir()`（设置 `package_dir[""]=XPU_PYTHON_ROOT`）、
  `XpuBuildPy`、`_patch_setup_for_xpu_python_root()`（import 时自动执行）。
- `setup.py`：+1 行接入 XPU overlay 逻辑。

## 6. 构建产物排除

`.gitignore` 新增：

```
third_party/xpu/python/triton/_C/*.so
third_party/xpu/python/triton/FileCheck
third_party/xpu/python/triton/instrumentation/*.so
third_party/xpu/python/*.egg-info
```

确保以下产物不进入版本库：`libtriton.so`（约 1.9 GB）、`FileCheck`、
`instrumentation/libPrintLoadStoreMemSpaces.so`。提交前已确认三者均未被 staged。

## 7. 未改动（已验证等价）

- overlay `lib/Analysis/*.cpp`（死文件，不参与编译）
- overlay `lib/Analysis/CMakeLists.txt`（仅 `add_subdirectory(Analysis/NewAnalysis)`）
- `NewAnalysis/*`（与 internal 逻辑等价；`OpFoldResultUtils.h` 为超集）

## 8. 验证

构建：
```
cd python && FLAGTREE_BACKEND=xpu LLVM_SYSPATH=<llvm_trust> \
  ninja -C build/cmake.linux-x86_64-cpython-3.10 triton
```
`libtriton.so` 链接成功（187/187）。

Smoke 测试（`triton_xpu3_smoke.py`）全部通过：

- STEP 0：import ok
- STEP 1：alloc ok
- STEP 2：empty kernel launch ok（`assertIsRecognized` abort 已消除）
- STEP 3：add_one kernel，result_correct=True
- STEP 4：fp64 sqrt kernel，result_correct=True
- ALL DONE

## 9. Review 结论

- C001–C005（getAnchor / visitNonControlFlowArguments / SymbolTable include / BufferT / package_dir）
  均为误报：4 个 Analysis 头文件已与主树逐字节一致（主树本身可编译、链接、通过 smoke）。
- R001（`getStrEnvXPU`/`getBoolEnvXPU` 缺少主树同款 mutex）为有效发现，已修复（见 4.3）。
