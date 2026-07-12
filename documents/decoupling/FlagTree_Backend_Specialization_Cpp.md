# FlagTree Backend Specialization 统一设计（C++）

FlagTree 设计的后端统一特化，目的是整合后端接入范式，对后端的特化实现清晰化管理，为后端适配 Triton 版本升级迁移提供工程基础。具体实施方案是将各后端对 Triton 的特化，从以往的 fork 仓库直接修改并单独维护，标准化为定义接口并在后端目录下给出差异化实现。

## 1. 原则与规范

主干代码在保证缺省逻辑不变的基础上，允许调用接口，然后在后端目录中（third_party/backendxxx/）实现特化。主干代码原则上不允许直接出现某后端的特化实现，也不允许对后端做选择判断后特化实现。<br>
与 Python 特化不同，C++ 特化接口方法由后端按需添加。当多后端对同一段主干代码有特化需求时，应协调保障多方特化功能。<br>

## 2. 基本设计

FlagTree 为 C++ 代码的后端特化提供的实现方案：使用宏判断在工程编译时选择是否特化。宏定义在后端特化目录 spec 目录下的头文件，统一通过 backend/spec/include/flagtree_spec.h 最先被包含，保证同名文件以特化为优先。特化实现的目标保证最先生成，使得主干链接目标时能正确选择特化实现生成的目标。

- CMakeLists.txt
```shell
set(FLAGTREE_BACKEND_DIR ${PROJECT_SOURCE_DIR}/third_party/${FLAGTREE_BACKEND})
## flagtree spec include dir
set(BACKEND_SPEC_INCLUDE_DIR ${FLAGTREE_BACKEND_DIR}/backend/spec/include)
if(FLAGTREE_BACKEND AND EXISTS ${BACKEND_SPEC_INCLUDE_DIR})
  include_directories(${BACKEND_SPEC_INCLUDE_DIR})
endif()

...

if(TRITON_BUILD_PYTHON_MODULE)
  ...
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()
  ...
endif()
```

- third_party/iluvatar/CMakeLists.txt（本文以 iluvatar 后端为例）

```shell
# 忽略 LLVM Compile Warning
add_compile_options("-Wno-deprecated-declarations")
add_compile_options("-Wno-error=deprecated-declarations")

# 对 editable 模式的处理
...

# 进入代码目录
add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(bin)
```

## 3. td 文件特化

### 3.1 td 文件整体特化

td 文件如果需要特化，可整体复制到对应的后端 spec 目录下进行后端特化实现。例如将 include/triton/Dialect/Triton/IR/TritonAttrDefs.td 复制到 <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Dialect/Triton/IR/TritonAttrDefs.td 进行特化修改，注意不需要修改 td 文件头部的 #ifndef 和 #define 宏，因为 CMakeLists.txt 中通过 set_flagtree_backend_td 方法只选择其中一个进行代码生成。

- include/triton/Dialect/Triton/IR/CMakeLists.txt
```shell
# set(LLVM_TARGET_DEFINITIONS TritonOps.td)  # 原实现
set_flagtree_backend_td(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)
```

### 3.2 EncodingAttr 使用特化

#### 3.2.1 主干代码的特化接入

- include/triton/Conversion/TritonGPUToLLVM/Utility.h
```c++
#ifdef FLAGTREE_SPEC_BackendMmaEncodingAttr
using FLAGTREE_SPEC_BackendMmaEncodingAttr;
#endif
```

#### 3.2.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Dialect/TritonGPU/IR/iluvatar_Dialect.h
```c++
#define FLAGTREE_SPEC_BackendMmaEncodingAttr                                   \
  ::mlir::triton::gpu::IluvatarMmaEncodingAttr
```

- </strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Dialect/TritonGPU/IR/iluvatar_Dialect.h"
```

## 4. h 头文件特化

### 4.1 情形一：函数声明修改返回类型或参数类型

#### 4.1.1 主干代码的缺省实现与特化接入

- include/triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h
```c++
#ifdef FLAGTREE_SPEC_TargetInfoBase_function
  virtual Value storeShared(ConversionPatternRewriter &rewriter, Location loc,
                            Value ptr, Value val, Value pred) const = 0;
  virtual Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Type elemTy, Value pred) const = 0;
#else
  virtual void storeShared(ConversionPatternRewriter &rewriter, Location loc,
                           Value ptr, Value val, Value pred) const = 0;
  virtual Value loadShared(ConversionPatternRewriter &rewriter, Location loc,
                           const TypeConverter *converter, Value ptr,
                           Type elemTy, Value pred) const = 0;
#endif
```

#### 4.1.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Conversion/TritonGPUToLLVM/iluvatar_TargetInfoBase.h
```c++
#define FLAGTREE_SPEC_TargetInfoBase_function
```

- <strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Conversion/TritonGPUToLLVM/iluvatar_TargetInfoBase.h"
```

#### 4.1.3 后端目录的特化实现

这两个函数在 iluvatar 后端的特化实现不在本仓库，在厂商仓库的 third_party/iluvatar/plugin。

### 4.2 情形二：函数声明添加特化参数

#### 4.2.1 主干代码的缺省实现与特化接入

- include/triton/Analysis/Utility.h
```c++
SetVector<Operation *> multiRootGetSlice(
    Operation *op, TransitiveFilter backwardFilter = nullptr,
#ifndef FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG
    TransitiveFilter forwardFilter = nullptr);
#else
    TransitiveFilter forwardFilter = nullptr,
    FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG omitBlockArguments = true);
#endif
```

- lib/Analysis/Utility.cpp
```c++
#ifndef FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG
SetVector<Operation *> multiRootGetSlice(
    Operation *op, TransitiveFilter backwardFilter,
    TransitiveFilter forwardFilter) {
  ...
}
```

#### 4.2.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Analysis/iluvatar_Utility.h
```c++
#define FLAGTREE_SPEC_Utility_multiRootGetSlice_ARG bool
```

- <strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Analysis/iluvatar_Utility.h"
```

#### 4.2.3 后端目录的特化实现

- <strong>third_party/iluvatar/backend/spec/</strong>lib/Analysis/Utility.cpp
```c++
SetVector<Operation *> multiRootGetSlice(
    Operation *op, TransitiveFilter backwardFilter,
    TransitiveFilter forwardFilter,
    bool omitBlockArguments) {
  ...
}
```

- <strong>third_party/iluvatar/backend/spec/</strong>lib/Analysis/CMakeLists.txt
```shell
add_triton_library(FlagTree_iluvatar_TritonAnalysis
  Utility.cpp
  ...

  DEPENDS
  ...
)
```

## 5. cpp 文件特化

### 5.1 情形一：cpp 文件中添加一段特化逻辑

#### 5.1.1 主干代码的特化接入

- lib/Analysis/Allocation.cpp
```c++
  void getScratchValueSize(Operation *op) {
    ...
      auto smemShape = getScratchConfigForCvtLayout(cvtLayout, inVec, outVec);
      unsigned elems = std::accumulate(smemShape.begin(), smemShape.end(), 1,
                                       std::multiplies{});
#ifdef FLAGTREE_SPEC_Analysis_Allocation_AllocationAnalysis_getScratchValueSizeElems
      elems = getScratchValueSizeElems(smemShape);
#endif
    ...
  }
```

- include/triton/Analysis/Allocation.h
```c++
#ifdef FLAGTREE_SPEC_Analysis_Allocation_AllocationAnalysis_getScratchValueSizeElems
unsigned getScratchValueSizeElems(const SmallVector<unsigned> &smemShape);
#endif
```

#### 5.1.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Analysis/iluvatar_Allocation.h
```c++
#define FLAGTREE_SPEC_Analysis_Allocation_AllocationAnalysis_getScratchValueSizeElems
```

- <strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Analysis/iluvatar_Allocation.h"
```

#### 5.1.3 后端目录的特化实现

- <strong>third_party/iluvatar/backend/spec/</strong>lib/Analysis/Allocation.cpp
```c++
unsigned getScratchValueSizeElems(const SmallVector<unsigned> &smemShape) {
  ...
}
```

- <strong>third_party/iluvatar/backend/spec/</strong>lib/Analysis/CMakeLists.txt
```c++
add_triton_library(FlagTree_iluvatar_TritonAnalysis
  Allocation.cpp
  ...

  DEPENDS
  ...
)
```

### 5.2 情形二：cpp 文件中定义的 static 函数特化

仅在一个 cpp 文件中定义和使用的静态函数如果需要被其他特化函数调用，需要改为非静态函数，在头文件中添加声明。

#### 5.2.1 主干代码的缺省实现与特化接入

- lib/Analysis/Allocation.cpp
```c++
#ifdef FLAGTREE_SPEC_Analysis_Allocation_getCvtOrder
std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
#else
static std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
#endif
getCvtOrder(Attribute srcLayout, Attribute dstLayout) {
  ...
}
```

- include/triton/Analysis/Allocation.h
```c++
#ifdef FLAGTREE_SPEC_Analysis_Allocation_getCvtOrder
std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
getCvtOrder(Attribute srcLayout, Attribute dstLayout);
#endif
```

#### 5.2.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Analysis/iluvatar_Allocation.h
```c++
#define FLAGTREE_SPEC_Analysis_Allocation_getCvtOrder
```

- <strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Analysis/iluvatar_Allocation.h"
```

#### 5.2.3 后端目录的特化实现

本函数在 iluvatar 后端无特化实现。如果有，则定义在对应特化路径的 cpp 中并生成对应目标。

### 5.3 情形三：整个 cpp 文件特化

调用关系耦合太多时，可退化为整个文件特化。常用于 cpp 内定义多个 class/struct 并交叉调用的情形。

#### 5.3.1 主干代码的缺省实现

- lib/Dialect/Triton/IR/Ops.cpp
```c++
#if __has_include("flagtree_spec.h")
#include "flagtree_spec.h"
#endif

#ifndef FLAGTREE_SPEC_Dialect_Triton_IR_Ops_cpp
...
#endif
```

#### 5.3.2 宏定义及头文件包含（注意修改文件名及头部宏）

- <strong>third_party/iluvatar/backend/spec/</strong>include/triton/Dialect/Triton/IR/iluvatar_Ops.h
```c++
#define FLAGTREE_SPEC_Dialect_Triton_IR_Ops_cpp
```

- <strong>third_party/iluvatar/backend/spec/</strong>include/flagtree_spec.h
```c++
#include "triton/Dialect/Triton/IR/iluvatar_Ops.h"
```

#### 5.3.3 后端目录的特化实现

- <strong>third_party/iluvatar/backend/spec/</strong>lib/Dialect/Triton/IR/Ops.cpp
- <strong>third_party/iluvatar/backend/spec/</strong>lib/Dialect/Triton/IR/CMakeLists.txt
```shell
add_triton_library(FlagTree_iluvatar_TritonIR
  Ops.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen
)
```

### 5.4 特化目标链接

CMakeLists.txt 中通过 get_flagtree_backend_lib 方法将 spec 目录中的特化实现目标链接进来。注意 spec 目录中，特化实现 cpp 生成的目标名规范，是给原名（本例中为 TritonIR）加上前缀 ```FlagTree_${FLAGTREE_BACKEND}_```。
- lib/Dialect/Triton/IR/CMakeLists.txt
```shell
get_flagtree_backend_lib("TritonIR" _EXTRA_LINK_LIBS)

add_triton_library(TritonIR
  ....cpp

  DEPENDS
  ...

  LINK_LIBS PUBLIC
  ...
  ${_EXTRA_LINK_LIBS}
)
```
