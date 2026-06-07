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

#ifndef TRITON_TO_CFG_MEMORY_SSA_BUILDER_H
#define TRITON_TO_CFG_MEMORY_SSA_BUILDER_H

#include "TritonToGraph/AliasAnalysis.h"
#include "TritonToGraph/MemorySSA.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace cfg {

// 前向声明
class DataFlowInfo;
class ControlFlowGraph;
class Instruction;

// MemorySSABuilder - Memory SSA构建器
// 构建整个CFG的Memory SSA信息，包括创建definitions、uses、处理控制流phi节点
class MemorySSABuilder {
public:
  MemorySSABuilder(ControlFlowGraph &cfg, AliasAnalysis &aliasAnalysis,
                   DataFlowInfo &dataFlowInfo)
      : cfg(cfg), aliasAnalysis(aliasAnalysis), dataFlowInfo(dataFlowInfo) {}

  ~MemorySSABuilder();

  // 构建整个CFG的Memory SSA
  void build();

private:
  // 处理单个BasicBlock
  void processBasicBlock(BasicBlock *bb);

  // 处理单个指令
  void processInstruction(Instruction *inst);

  // 处理scf.if的phi节点
  void processIfOp(scf::IfOp ifOp, Instruction *inst, BasicBlock *thenEntryBB,
                   BasicBlock *elseEntryBB);

  // 处理scf.for的iter_args
  void processForOp(scf::ForOp forOp, Instruction *inst,
                    BasicBlock *loopBodyEntryBB);

  // 处理scf.while的args
  void processWhileOp(scf::WhileOp whileOp, Instruction *inst,
                      BasicBlock *beforeEntryBB, BasicBlock *afterEntryBB);

  // 判断是否是tensor类型
  bool isTensorType(Type type) const {
    return mlir::isa<RankedTensorType>(type) ||
           mlir::isa<triton::PointerType>(type);
  }

  // 根据操作创建tensor对象
  TensorObject *createTensorObject(Operation *op);

  // 创建tensor definition
  MemorySSADef *createDefinition(TensorObject *tensor, Operation *op);

  // 创建use
  MemorySSAUse createUse(MemorySSADef *def, Operation *userOp,
                         unsigned operandIdx);

  // 判断是否是入参
  bool isParameter(Operation *op) const {
    return op == nullptr; // 入参的defOp为nullptr
  }

  // 判断是否是返回新Tensor的操作（根据返回值类型判断，排除load）
  bool isTensorWriter(Operation *op) const;

  // 判断是否是修改内存的操作（有副作用）
  bool isMemoryWriter(Operation *op) const {
    // 只有：tt.store（写入内存）
    return isa<triton::StoreOp>(op);
  }

  // 判断是否是修改读取的操作（有副作用）
  bool isMemoryReader(Operation *op) const {
    // 只有：tt.store（写入内存）
    return isa<triton::LoadOp>(op);
  }

  // 判断是否是创建指针的操作
  bool isPointerOp(Operation *op) const {
    // 返回指针类型：addptr（偏移指针）、make_tensor_ptr（创建张量指针）
    if (isa<triton::AddPtrOp, triton::MakeTensorPtrOp>(op))
      return true;

    if (isPointerBroadcastOrSplat(op))
      return true;

    return false;
  }

  bool isPointerBroadcastOrSplat(mlir::Operation *op) const;

  // 创建函数的参数定义
  void createParameterDefinitions();

  // 获取或创建tensor对象
  TensorObject *getOrCreateTensorObject(Value value);

  // 获取操作的字符串名称（用于生成tensor名称）
  std::string getOpName(Operation *op);

  // 成员变量
  ControlFlowGraph &cfg;
  AliasAnalysis &aliasAnalysis;
  DataFlowInfo &dataFlowInfo;
  // size_t nextVersionId;  // 下一个可用的版本号（用于MemorySSADef）
  // size_t nextTensorId;   // 下一个可用的tensor ID（用于TensorObject命名）

  std::map<TensorObject *, size_t> nextVersion;
  std::map<std::string, size_t> nextTensor;

  // 所有创建的definitions（用于内存管理）
  SmallVector<MemorySSADef *> allDefinitions;

  // tensor对象缓存
  DenseMap<Value, TensorObject *> tensorObjectCache;
};

// MemorySSABuilderHelper - 辅助函数
namespace MemorySSABuilderHelper {
// 获取操作的结果类型
Type getResultType(Operation *op, unsigned resultIdx);

// 获取Value的形状信息
SmallVector<int64_t> getShapeFromValue(Value value);

// 判断两个shape是否相同
bool shapesEqual(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2);

// 获取scf.if的region yield操作
Operation *getYieldOp(Region &region);

// 为tensor创建唯一名称
std::string createUniqueTensorName(StringRef prefix, size_t id);

// 判断是否需要为操作创建新版本
bool shouldCreateNewVersion(Operation *op, MemorySSADef *currentDef);
} // namespace MemorySSABuilderHelper

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_MEMORY_SSA_BUILDER_H
