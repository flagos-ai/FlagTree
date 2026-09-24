# Iluvatar C++ 特化第一步审计清单

## 审计范围

- 分支：`refactor/iluvatar-cpp-specialization`
- 基线：`a246cf05a`，与 `origin/main` 一致
- 范围：
  - `third_party/iluvatar/include/triton/`
  - `third_party/iluvatar/lib/`
- 本阶段不修改源码、不修改 CMake、不执行格式化。
- `third_party/iluvatar/backend/`、`third_party/iluvatar/python/test/`、插件和 TLE 独有目录不在本次主干镜像裁剪范围内。

## 比较方法

不能只比较当前主干和厂商副本。厂商副本来自历史提交，直接比较会把主干后续演进误判为 Iluvatar 特化。

本次使用以下三层结果：

1. 当前主干与厂商副本的逐文件比较。
2. 厂商目录引入提交 `35451e81e` 作为三方基线。
3. 对相对引入基线发生变化的文件逐个查看语义差异。

候选文件共 274 个，分类总数如下：

| 分类 | 数量 | 后续处理 |
| --- | ---: | --- |
| 可直接复用主干 | 232 | 第二步删除厂商副本 |
| Iluvatar/SME/TCU/TLE 核心特化 | 34 | 第二步使用 `git mv` 移到 `spec_cpp` |
| 构建配置差异 | 7 | 第三步单独处理 CMake |
| Iluvatar 独有文件 | 1 | 第二步移动到 `spec_cpp`，第三步确认引用 |

其中，232 个可复用主干文件包括：当前比较完全相同的 37 个文件、仅版权声明不同的 3 个文件、相对引入基线没有厂商新增逻辑的 169 个文件，以及下面“非特化差异”中的 23 个文件。

前一版清单遗漏的两个完全相同文件如下，第二步同样删除厂商副本并复用主干：

```text
third_party/iluvatar/include/triton/Dialect/Gluon/CMakeCache.txt
third_party/iluvatar/include/triton/Dialect/TritonInstrument/IR/TritonInstrument.md
```

## 一、保留并迁移的核心特化文件

这些文件包含实际的 Iluvatar、SME、TCU 或 Iluvatar TLE 逻辑。第二步迁移时保持文件内容不变，使用 `git mv`。

### 头文件和 TableGen

```text
third_party/iluvatar/include/triton/Conversion/TritonGPUToLLVM/Utility.h
third_party/iluvatar/include/triton/Dialect/Triton/IR/TritonOps.td
third_party/iluvatar/include/triton/Dialect/Triton/IR/Utility.h
third_party/iluvatar/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h
third_party/iluvatar/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
third_party/iluvatar/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td
third_party/iluvatar/include/triton/Dialect/TritonGPU/Transforms/Passes.td
third_party/iluvatar/include/triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h
third_party/iluvatar/include/triton/Tools/Sys/GetEnv.hpp
```

主要逻辑包括：

- `tt.load`/`async_copy_global_to_local` 的 `inputStride` 和 SME builder。
- `BlockedEncodingAttr`、`SwizzledSharedEncodingAttr` 的 SME/TCU 属性。
- Iluvatar MMA 编码和 SME layout。
- Iluvatar linker 查找和 cache invalidation 环境变量。
- Iluvatar TLE dialect 的动态合法性和类型转换接口。

### 分析和转换

```text
third_party/iluvatar/lib/Analysis/Alias.cpp
third_party/iluvatar/lib/Analysis/Allocation.cpp
third_party/iluvatar/lib/Analysis/Membar.cpp
third_party/iluvatar/lib/Analysis/Utility.cpp
third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/MakeRangeOpToLLVM.cpp
third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp
third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/Utility.cpp
third_party/iluvatar/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp
third_party/iluvatar/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp
```

主要逻辑包括：

- SME global-to-shared load/store、waitcnt、mask fixup 和 rowxfb8 修正。
- TCU MMA 支持、layout 转换和 warp 配置。
- Iluvatar TLE 的 alias、scratch allocation、dialect conversion 和 pattern。
- Iluvatar barrier 和异步复制依赖。

### Triton IR 和 TritonGPU IR/Transforms

```text
third_party/iluvatar/lib/Dialect/Triton/IR/Ops.cpp
third_party/iluvatar/lib/Dialect/Triton/IR/Utility.cpp
third_party/iluvatar/lib/Dialect/Triton/Transforms/Combine.cpp
third_party/iluvatar/lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/IR/Dialect.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/IR/Ops.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/CoalesceAsyncCopy.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Prefetch.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Utility.cpp
```

这些文件中包含 `inputStride` 传播、SME layout island、SME pipelining、TCU dot chain、shared-memory 强制路径和 Iluvatar TLE 属性传播。

## 二、只复用当前主干的非特化差异

这些文件虽然与厂商副本不同，但差异只是删除通用 TLE/其他厂商逻辑、API 迁移、主干同步或版权头变化，不属于 Iluvatar 特化。第二步删除厂商副本，不迁移。

```text
third_party/iluvatar/include/triton/Analysis/AxisInfo.h
third_party/iluvatar/include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h
third_party/iluvatar/include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h
third_party/iluvatar/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td
third_party/iluvatar/include/triton/Dialect/TritonGPU/Transforms/Passes.h
third_party/iluvatar/include/triton/Dialect/TritonGPU/Transforms/Utility.h
third_party/iluvatar/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td
third_party/iluvatar/lib/Analysis/AxisInfo.cpp
third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/ControlFlowOpToLLVM.cpp
third_party/iluvatar/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp
third_party/iluvatar/lib/Dialect/Gluon/Transforms/Canonicalize.cpp
third_party/iluvatar/lib/Dialect/Gluon/Transforms/SimplifyControlFlow.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/Schedule.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp
third_party/iluvatar/lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp
third_party/iluvatar/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp
third_party/iluvatar/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp
third_party/iluvatar/lib/Dialect/TritonNvidiaGPU/Transforms/ProxFenceInsertion.cpp
third_party/iluvatar/lib/Instrumentation/PrintLoadStoreMemSpaces.cpp
```

特别注意：不能因为这些文件曾经出现在厂商提交中，就把它们当成特化文件迁移。

## 三、留到第三步的 CMake 文件

这些文件只记录构建入口、额外链接库或 TableGen 配置，暂不在第一步和第二步修改。

```text
third_party/iluvatar/include/triton/Dialect/TritonGPU/IR/CMakeLists.txt
third_party/iluvatar/lib/Conversion/CMakeLists.txt
third_party/iluvatar/lib/Conversion/TritonToTritonGPU/CMakeLists.txt
third_party/iluvatar/lib/Dialect/Triton/IR/CMakeLists.txt
third_party/iluvatar/lib/Dialect/Triton/Transforms/CMakeLists.txt
third_party/iluvatar/lib/Dialect/TritonGPU/IR/CMakeLists.txt
third_party/iluvatar/lib/Dialect/TritonGPU/Transforms/CMakeLists.txt
```

第三步需要重新确认：

- 共享 Triton target 的唯一 owner。
- `spec_cpp` 源文件的 source override 是否能命中现有 target。
- Iluvatar 独有 `.cpp` 是否需要显式 `target_sources()`。
- 特化 `.td` 是否同时被 TableGen 和文档生成 target 使用。
- Iluvatar TLE/插件库是否只在需要的 target 上链接。

## 四、Iluvatar 独有文件

```text
third_party/iluvatar/include/triton/Tools/LLVMWarningFilter.h
```

该头文件被 `third_party/iluvatar/triton_iluvatar.cc` 使用。第二步移动为：

```text
third_party/iluvatar/spec_cpp/include/triton/Tools/LLVMWarningFilter.h
```

第三步必须确认 include 搜索路径和插件 target 的实际编译命令。

## 五、第一步结论

第一步审计已完成。后续第二步只允许使用本清单：

1. 对“核心特化文件”执行 `git mv` 到 `third_party/iluvatar/spec_cpp/`。
2. 对“非特化差异”和其余可复用文件删除厂商副本。
3. 不改文件内容，不格式化，不重排 include。
4. 不处理 `backend/`、插件、`python/test/`、语言扩展和本阶段列出的 CMake。
5. 完成第二步 staged diff 检查后再提交并等待 Iluvatar CI。
