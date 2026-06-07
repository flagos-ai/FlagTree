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

#ifndef TRITON_TO_CFG_DATAFLOW_GRAPH_H
#define TRITON_TO_CFG_DATAFLOW_GRAPH_H

#include "TritonToGraph/AliasAnalysis.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/MemorySSA.h"
#include "TritonToGraph/MemorySsaBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

// 前向声明结果类
class DataFlowResult;
class MemorySSAResult;
class SSAResult;

// DataFlowInfo - 统一的数据流信息
class DataFlowInfo {
public:
  // 为函数入口创建参数定义
  void createParameterDefinitions(triton::FuncOp func);

  // Memory SSA接口
  MemorySSADef *getMemoryDefinition(Value value) const;
  void addMemoryDefinition(Value value, MemorySSADef *def);

  SmallVector<MemorySSAUse> getMemoryUses(Value value) const;
  void addMemoryUse(Value value, const MemorySSAUse &use);

  void removeMemoryDefinition(Value value);
  void clearMemoryUses(Value value);

  // 传统SSA接口（复用MLIR原生功能）
  Operation *getSSADefinition(Value value) const {
    return value.getDefiningOp();
  }

  SmallVector<OpOperand *> getSSAUses(Value value) const {
    SmallVector<OpOperand *> result;
    for (OpOperand &use : value.getUses()) {
      result.push_back(&use);
    }
    return result;
  }

  // 循环Phi接口
  void addPhi(Value value, const PhiInfo &phiInfo) { Phis[value] = phiInfo; }

  PhiInfo &getPhi(Value value) { return Phis[value]; }

  bool hasPhi(Value value) const { return Phis.count(value) > 0; }

  // 统一查询接口 - 返回unique_ptr，使用LLVM RTTI进行类型判断
  std::unique_ptr<DataFlowResult> queryDataFlow(Value value) const;

  // 查询某个定义的所有使用
  SmallVector<MemorySSAUse> getUses(MemorySSADef *def) const;

  // 查询某个操作的memory使用
  SmallVector<MemorySSAUse> getUsesByUserOp(Operation *userOp) const;

  // 遍历接口
  void
  forEachDefinition(llvm::function_ref<void(Value, MemorySSADef *)> func) const;
  void forEachUse(llvm::function_ref<void(const MemorySSAUse &)> func) const;

  // 获取所有Memory SSA definitions
  const DenseMap<Value, MemorySSADef *> &getMemoryDefinitions() const {
    return memoryDefinitions;
  }

  // 获取所有循环phi信息
  const DenseMap<Value, PhiInfo> &getPhis() const { return Phis; }

  // 构建def-use缓存
  void buildDefUseCache() const;

  // 打印信息（调试用）
  void print(llvm::raw_ostream &os) const;

  // 导出到JSON
  void exportToJSON(llvm::raw_ostream &os) const;

private:
  // Memory SSA映射
  DenseMap<Value, MemorySSADef *> memoryDefinitions;
  DenseMap<Value, SmallVector<MemorySSAUse>> memoryUses;

  // Loop Phi映射
  DenseMap<Value, PhiInfo> Phis;

  // Use-Def映射缓存（def -> uses）
  mutable DenseMap<MemorySSADef *, SmallVector<MemorySSAUse>> defUseCache;
  mutable bool defUseCacheValid = false;

  void invalidateDefUseCache() {
    defUseCacheValid = false;
    defUseCache.clear();
  }
};

// DataFlowResult - 数据流查询结果的基类
// 使用LLVM RTTI系统，支持isa<>和dyn_cast<>进行类型判断
class DataFlowResult {
public:
  enum class Kind {
    MemorySSA, // Memory SSA结果（tensor/pointer）
    SSA,       // 传统SSA结果（标量）
    NONE       // 无数据流信息
  };

  DataFlowResult(Kind kind, Operation *originOp)
      : kind(kind), originOp(originOp) {}
  virtual ~DataFlowResult() = default;

  Kind getKind() const { return kind; }
  Operation *getOriginOp() const { return originOp; }

  SmallVector<OpOperand *> &getUses() { return uses; }
  const SmallVector<OpOperand *> &getUses() const { return uses; }

  std::optional<PhiInfo> &getPhi() { return Phi; }
  const std::optional<PhiInfo> &getPhi() const { return Phi; }

  // LLVM RTTI支持
  static bool classof(const DataFlowResult *) { return true; }

protected:
  Kind kind;
  Operation *originOp;
  SmallVector<OpOperand *> uses; // 所有uses
  std::optional<PhiInfo> Phi;    // Phi信息（如果有）
};

// MemorySSAResult - Memory SSA的结果
class MemorySSAResult : public DataFlowResult {
public:
  MemorySSAResult(Operation *originOp, MemorySSADef *definition)
      : DataFlowResult(Kind::MemorySSA, originOp), definition(definition) {}

  MemorySSADef *getDefinition() const { return definition; }

  // LLVM RTTI支持
  static bool classof(const DataFlowResult *result) {
    return result->getKind() == Kind::MemorySSA;
  }

private:
  MemorySSADef *definition; // MEMORY_SSA时使用
};

// SSAResult - 传统SSA的结果
class SSAResult : public DataFlowResult {
public:
  SSAResult(Operation *originOp, Operation *ssaDefinition)
      : DataFlowResult(Kind::SSA, originOp), ssaDefinition(ssaDefinition) {}

  Operation *getSSADefinition() const { return ssaDefinition; }

  // LLVM RTTI支持
  static bool classof(const DataFlowResult *result) {
    return result->getKind() == Kind::SSA;
  }

private:
  Operation *ssaDefinition; // SSA时使用
};

// NoneResult - 无数据流信息的结果
class NoneResult : public DataFlowResult {
public:
  NoneResult() : DataFlowResult(Kind::NONE, nullptr) {}

  // LLVM RTTI支持
  static bool classof(const DataFlowResult *result) {
    return result->getKind() == Kind::NONE;
  }
};

// DataFlowGraph - 数据流图
class DataFlowGraph {
public:
  explicit DataFlowGraph(ControlFlowGraph &cfg) : cfg(cfg) {}

  ~DataFlowGraph() = default;

  // 构建完整的数据流信息
  void build();

  // 查询Value的数据流信息（使用LLVM RTTI判断具体类型）
  std::unique_ptr<DataFlowResult> queryDataFlow(Value value) const {
    return dataFlowInfo.queryDataFlow(value);
  }

  // 获取所有Memory SSA definitions
  SmallVector<MemorySSADef *> getAllDefinitions() const {
    SmallVector<MemorySSADef *> result;
    for (const auto &entry : dataFlowInfo.getMemoryDefinitions()) {
      result.push_back(entry.second);
    }
    return result;
  }

  // 获取definition的所有uses
  SmallVector<MemorySSAUse> getUses(MemorySSADef *def) const {
    return dataFlowInfo.getUses(def);
  }

  // 获取操作的所有uses
  SmallVector<MemorySSAUse> getUsesByUserOp(Operation *userOp) const {
    return dataFlowInfo.getUsesByUserOp(userOp);
  }

  // 获取CFG
  ControlFlowGraph &getCFG() { return cfg; }
  const ControlFlowGraph &getCFG() const { return cfg; }

  // 获取DataFlowInfo
  DataFlowInfo &getDataFlowInfo() { return dataFlowInfo; }
  const DataFlowInfo &getDataFlowInfo() const { return dataFlowInfo; }

  // 导出数据流信息到JSON
  void exportToJSON(llvm::raw_ostream &os) const;

  // 导出def-use链到DOT格式
  void exportDefUseToDOT(llvm::raw_ostream &os) const;

  // 打印所有数据流信息（调试用）
  void print(llvm::raw_ostream &os) const;
  void dump() const;

private:
  ControlFlowGraph &cfg;

  // 组件
  std::unique_ptr<AliasAnalysis> aliasAnalysis;
  std::unique_ptr<MemorySSABuilder> memorySSABuilder;

  // 数据流信息
  DataFlowInfo dataFlowInfo;

  // 构建def-use图
  void buildDefUseGraph();
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_DATAFLOW_GRAPH_H
