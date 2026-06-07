/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_CFG_ALIAS_ANALYSIS_H
#define TRITON_TO_CFG_ALIAS_ANALYSIS_H

#include "TritonToGraph/tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace triton {
namespace cfg {

class ControlFlowGraph;

// AliasAnalysis - Alias分析和指针跟踪
// 用于分析pointer的alias关系，跟踪全局内存对象（如gm_obj）的别名链
class AliasAnalysis {
public:
  // 为整个CFG分析pointer别名
  void analyzePointerAliases(ControlFlowGraph &cfg);

  // 获取Value的base pointer（递归查找真实的基础指针）
  Value getBasePointer(Value ptr) const;

  // 获取Value对应的TensorObject
  TensorObject *getTensorObject(Value value) const {
    auto it = baseTensorMap.find(value);
    return (it != baseTensorMap.end()) ? it->second : nullptr;
  }

  // 判断两个指针是否可能指向同一tensor
  bool mayAlias(Value ptr1, Value ptr2) const {
    return getBasePointer(ptr1) == getBasePointer(ptr2);
  }

  // 添加alias关系
  void addAlias(Value ptr, Value base, TensorObject *tensor) {
    aliasMap[ptr] = base;
    baseTensorMap[ptr] = tensor;
  }

  // 判断是否是指针类型
  static bool isPointerType(Type type) {
    if (auto ptrType =
            mlir::dyn_cast<triton::PointerType>(getElementTypeOrSelf(type))) {
      return true;
    }
    return false;
  }

  // 判断是否是tensor pointer类型（指向tensor的指针）
  static bool isTensorPointerType(Type type) {
    if (auto ptrType = mlir::dyn_cast<triton::PointerType>(type)) {
      Type pointeeType = ptrType.getPointeeType();
      return mlir::isa<RankedTensorType>(pointeeType);
    }
    return false;
  }

  // 判断是否是标量指针类型（指向标量的指针）
  static bool isScalarPointerType(Type type) {
    if (auto ptrType = mlir::dyn_cast<triton::PointerType>(type)) {
      Type pointeeType = ptrType.getPointeeType();
      return !mlir::isa<RankedTensorType>(pointeeType);
    }
    return false;
  }

  // 判断是否是全局内存类型
  static bool isGlobalMemoryType(Type type) {
    // 在Triton中，全局内存通常由!tt.ptr类型表示
    // 可以进一步根据地址空间等信息判断
    if (auto ptrType = mlir::dyn_cast<triton::PointerType>(type)) {
      // 地址空间1通常是全局内存
      // 需要根据具体硬件进行调整
      return ptrType.getAddressSpace() == 1 || ptrType.getAddressSpace() == 0;
    }
    return false;
  }

  // 获取所有tracked的base pointer
  const DenseMap<Value, Value> &getAliasMap() const { return aliasMap; }

  // 获取所有tracked的tensor对象
  const DenseMap<Value, TensorObject *> &getBaseTensorMap() const {
    return baseTensorMap;
  }

  // 打印所有alias信息
  void print(llvm::raw_ostream &os) const {
    os << "=== Alias Analysis Result ===\n";
    os << "Total tracked aliases: " << aliasMap.size() << "\n";

    for (const auto &entry : aliasMap) {
      Value ptr = entry.first;
      Value base = entry.second;
      TensorObject *tensor = baseTensorMap.lookup(ptr);

      os << "  " << ptr << " -> " << base;
      if (tensor) {
        os << " [" << tensor->getName() << "]";
      }
      os << "\n";
    }
  }

private:
  // 分析addptr操作
  void analyzeAddPtrOp(mlir::triton::AddPtrOp addptrOp);

  // 分析make_tensor_ptr操作
  void analyzeMakeTensorPtrOp(mlir::triton::MakeTensorPtrOp op);

  // 分析load操作
  void analyzeLoadOp(mlir::triton::LoadOp loadOp);

  // 分析store操作
  void analyzeStoreOp(mlir::triton::StoreOp storeOp);

  // 分析broadcast操作
  void analyzeBroadcastOp(mlir::triton::BroadcastOp broadcastOp);

  // 分析splat操作
  void analyzeSplatOp(mlir::triton::SplatOp splatOp);

  DenseMap<Value, Value> aliasMap;               // ptr -> base ptr
  DenseMap<Value, TensorObject *> baseTensorMap; // value -> tensor
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_ALIAS_ANALYSIS_H
