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

#include "TritonToGraph/MemorySsaBuilder.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "TritonToGraph/DataflowGraph.h"
#include "TritonToGraph/tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "memory-ssa-builder"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// MemorySSABuilder
//===----------------------------------------------------------------------===//

MemorySSABuilder::~MemorySSABuilder() {
  // 清理所有创建的tensor objects（如果有动态分配的）
  // 这里假设TensorObject由外部管理生命周期
}

void MemorySSABuilder::build() {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Memory SSA Build ===\n");

  // 步骤1: 拓扑排序，确定处理顺序
  std::vector<BasicBlock *> topoOrder;
  {
    // 简单的拓扑排序实现
    DenseSet<BasicBlock *> visited;
    std::function<void(BasicBlock *)> dfs = [&](BasicBlock *bb) {
      if (visited.contains(bb))
        return;
      visited.insert(bb);
      for (BasicBlock *succ : bb->getSuccessors()) {
        dfs(succ);
      }
      topoOrder.push_back(bb);
    };

    // 从入口块开始DFS
    dfs(cfg.getEntryBlock());
    std::reverse(topoOrder.begin(), topoOrder.end());
  }

  LLVM_DEBUG(llvm::dbgs() << "Topological order: " << topoOrder.size()
                          << " blocks\n");

  // 步骤2: 初始化函数的参数
  createParameterDefinitions();

  LLVM_DEBUG(llvm::dbgs() << "Created parameter definitions\n");

  // 步骤3: 按拓扑序遍历每个BasicBlock
  for (BasicBlock *bb : topoOrder) {
    LLVM_DEBUG(llvm::dbgs() << "Processing BB" << bb->getId() << "\n");

    // 处理block
    processBasicBlock(bb);
  }

  LLVM_DEBUG(llvm::dbgs() << "=== Memory SSA Build Complete ===\n"
                          << "Processed blocks: " << topoOrder.size() << "\n"
                          << "Total definitions: " << allDefinitions.size()
                          << "\n");
}

void MemorySSABuilder::createParameterDefinitions() {
  triton::FuncOp func = cfg.getFunction();

  LLVM_DEBUG(llvm::dbgs() << "Creating parameter definitions for "
                          << func.getName() << "\n");

  // 遍历函数参数
  for (BlockArgument arg : func.getArguments()) {
    Type argType = arg.getType();

    // 检查是否是我们关心的类型
    if (isTensorType(argType)) {
      // 创建参数名称
      std::string paramName = "param_" + std::to_string(arg.getArgNumber());

      // 如果是指针类型，从aliasAnalysis获取tensor对象
      TensorObject *tensor = nullptr;
      if (aliasAnalysis.isPointerType(argType)) {
        tensor = aliasAnalysis.getTensorObject(arg);
      }

      // 如果没有找到tensor对象，创建一个新的
      if (!tensor) {
        // 从类型推断shape和element type
        SmallVector<int64_t> shape;
        Type elementType;

        extractShapeAndElementType(argType, shape, elementType);

        tensor = new TensorObject(paramName, shape, argType, elementType,
                                  TensorObject::TensorKind::GLOBAL_MEMORY);
      }

      tensor->print(llvm::outs());
      llvm::outs() << "\n";

      // 为入参创建definition
      MemorySSADef *def = createDefinition(tensor, nullptr);

      // 记录到dataFlowInfo
      dataFlowInfo.addMemoryDefinition(arg, def);

      LLVM_DEBUG(llvm::dbgs()
                 << "  Created parameter definition: " << def->getId() << "\n");
    }
  }
}

void MemorySSABuilder::processBasicBlock(BasicBlock *bb) {
  if (!bb)
    return;

  // 处理block内的所有指令
  for (auto &instPtr : bb->getInstructions()) {
    Instruction *inst = instPtr.get();
    inst->print(llvm::outs());
    llvm::outs() << "\n";
    processInstruction(inst);
    MemorySSAInfo &ssaInfo = inst->getMemorySSAInfo();
    ssaInfo.print(llvm::outs());
    llvm::outs() << "\n";
  }
}

void MemorySSABuilder::processInstruction(Instruction *inst) {
  if (!inst)
    return;

  Operation *op = inst->getOperation();
  if (!op)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Processing: " << op->getName() << "\n");

  MemorySSAInfo &ssaInfo = inst->getMemorySSAInfo();

  // 1. 处理operands：创建uses
  LLVM_DEBUG(llvm::dbgs() << "  Processing operands...\n");

  for (OpOperand &operand : op->getOpOperands()) {
    Value operandValue = operand.get();
    unsigned operandIdx = operand.getOperandNumber();

    // 检查是否是tensor或pointer类型
    if (isTensorType(operandValue.getType()) ||
        aliasAnalysis.isPointerType(operandValue.getType())) {

      // 查找operand的definition
      MemorySSADef *def = dataFlowInfo.getMemoryDefinition(operandValue);

      if (def) {
        // 创建use
        MemorySSAUse use = createUse(def, op, operandIdx);
        ssaInfo.uses.push_back(use);

        // 记录到全局map
        dataFlowInfo.addMemoryUse(operandValue, use);

        LLVM_DEBUG(llvm::dbgs()
                   << "    MemorySSAUse: " << def->getId() << " in "
                   << op->getName() << " [operand #" << operandIdx << "]\n");
      }
    }
  }

  // 2. 按操作类型处理：store、load、tensor writer、pointer op
  LLVM_DEBUG(llvm::dbgs() << "  Processing by operation type...\n");

  // 对store操作（内存写入）- 为第一个operand指向的tensor创建新definition
  if (isMemoryWriter(op)) {
    LLVM_DEBUG(llvm::dbgs() << "    Memory writer: " << op->getName() << "\n");

    // store的第一个operand是ptr（第二个是value）
    // 针对Triton算子场景做了简化，此处不是流敏感分析
    if (op->getNumOperands() >= 2) {
      Value ptr = op->getOperand(0);
      if (isTensorType(ptr.getType()) ||
          aliasAnalysis.isPointerType(ptr.getType())) {
        // 获取ptr当前的definition（修改前的状态）
        MemorySSADef *oldDef = dataFlowInfo.getMemoryDefinition(ptr);
        if (oldDef) {
          // store会修改内存，为同一个tensor创建新的definition
          TensorObject *tensor = oldDef->getTensor();
          MemorySSADef *newDef = createDefinition(tensor, op);

          // store之后，ptr指向的内存状态改变，更新definition
          dataFlowInfo.addMemoryDefinition(ptr, newDef);

          LLVM_DEBUG(llvm::dbgs() << "    Store from: " << oldDef->getId()
                                  << " to: " << newDef->getId() << "\n");
        }
      }
    }
  }
  // 对load操作（内存读取）- 第一个operand的definition作为result的definition
  else if (isMemoryReader(op)) {
    LLVM_DEBUG(llvm::dbgs() << "    Memory reader: " << op->getName() << "\n");

    if (op->getNumOperands() > 0 && op->getNumResults() > 0) {
      Value ptr = op->getOperand(0);
      Value result = op->getResult(0);

      // 获取ptr的definition
      MemorySSADef *ptrDef = dataFlowInfo.getMemoryDefinition(ptr);
      if (ptrDef) {
        ssaInfo.definitions.push_back(ptrDef);
        dataFlowInfo.addMemoryDefinition(result, ptrDef);

        LLVM_DEBUG(llvm::dbgs()
                   << "    Load creates new def: " << ptrDef->getId()
                   << " from " << ptrDef->getId() << "\n");
      }
    }
  }
  // 对返回新Tensor的操作（tensor writer）
  else if (isTensorWriter(op)) {
    LLVM_DEBUG(llvm::dbgs() << "    Tensor writer: " << op->getName() << "\n");

    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // 检查是否是tensor类型
      if (isTensorType(resultType)) {
        // 创建tensor对象
        TensorObject *tensor = createTensorObject(op);

        // Tensor writer：创建新definition
        MemorySSADef *newDef = createDefinition(tensor, op);
        ssaInfo.definitions.push_back(newDef);
        dataFlowInfo.addMemoryDefinition(result, newDef);

        LLVM_DEBUG(llvm::dbgs() << "    Tensor definition: " << newDef->getId()
                                << " for " << result << "\n");
      }
    }
  }
  // 对pointer操作（addptr, make_tensor_ptr, [broadcast, splat]）
  else if (isPointerOp(op)) {
    LLVM_DEBUG(llvm::dbgs()
               << "    Pointer operation: " << op->getName() << "\n");

    for (Value result : op->getResults()) {
      Type resultType = result.getType();

      // 检查是否是pointer类型
      if (aliasAnalysis.isPointerType(resultType)) {
        // Pointer op：复用base pointer的definition（alias）
        Value basePtr = aliasAnalysis.getBasePointer(result);
        MemorySSADef *baseDef = dataFlowInfo.getMemoryDefinition(basePtr);

        if (baseDef) {
          ssaInfo.definitions.push_back(baseDef);
          dataFlowInfo.addMemoryDefinition(result, baseDef);

          LLVM_DEBUG(llvm::dbgs() << "    Pointer alias: " << baseDef->getId()
                                  << " for " << result << "\n");
        }
      }
    }
  }

  // 3. 特殊处理控制流操作
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    processIfOp(ifOp, inst, nullptr, nullptr);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    processForOp(forOp, inst, nullptr);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    // processWhileOp(whileOp, inst, nullptr, nullptr);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Done\n");
}

void MemorySSABuilder::processIfOp(scf::IfOp ifOp, Instruction *inst,
                                   BasicBlock *thenEntryBB,
                                   BasicBlock *elseEntryBB) {
  // 实现scf.if的phi节点处理
  // 从then和else区域收集yield的values，创建phi definitions

  LLVM_DEBUG(llvm::dbgs() << "Processing IfOp: " << ifOp << "\n");

  // 获取if指令
  // scf::IfOp ifOp =
  // cast<scf::IfOp>(ifCondBB->getInstruction(0)->getOperation());

  // 为每个result创建phi节点
  for (size_t i = 0; i < ifOp.getNumResults(); ++i) {
    Value ifResult = ifOp.getResult(i);
    Type resultType = ifResult.getType();

    // 只处理tensor/pointer类型
    if (!isTensorType(resultType) && !aliasAnalysis.isPointerType(resultType)) {
      continue;
    }

    // 从then区域获取yield的value
    Operation *thenYield =
        MemorySSABuilderHelper::getYieldOp(ifOp.getThenRegion());
    Value thenValue = thenYield ? thenYield->getOperand(i) : Value();
    MemorySSADef *thenDef =
        thenValue ? dataFlowInfo.getMemoryDefinition(thenValue) : nullptr;

    // 从else区域获取yield的value
    Operation *elseYield =
        MemorySSABuilderHelper::getYieldOp(ifOp.getElseRegion());
    Value elseValue = elseYield ? elseYield->getOperand(i) : Value();
    MemorySSADef *elseDef =
        elseValue ? dataFlowInfo.getMemoryDefinition(elseValue) : nullptr;

    // 如果then和else都返回相同的definition，可以直接使用
    if (thenDef && elseDef && thenDef == elseDef) {
      dataFlowInfo.addMemoryDefinition(ifResult, thenDef);

      LLVM_DEBUG(llvm::dbgs()
                 << "  If result #" << i
                 << " uses same definition: " << thenDef->getId() << "\n");
      continue;
    }

    // 创建phi definition
    if (thenDef || elseDef) {
      TensorObject *tensor = thenDef   ? thenDef->getTensor()
                             : elseDef ? elseDef->getTensor()
                                       : nullptr;

      if (tensor) {
        std::string phiName = "phi_" + std::to_string(ifOp.getNumResults()) +
                              "_" + std::to_string(i);
        TensorObject *phiTensor =
            new TensorObject(phiName, tensor->getShape(), resultType,
                             tensor->getElementType(), tensor->getKind());

        MemorySSADef *phiDef = createDefinition(phiTensor, ifOp.getOperation());

        // 记录if result的definition
        dataFlowInfo.addMemoryDefinition(ifResult, phiDef);

        // 为phi节点的operands创建uses
        if (thenDef && thenValue) {
          MemorySSAUse thenUse(thenDef, ifOp.getOperation(), /*operandIdx=*/i);
          dataFlowInfo.addMemoryUse(thenValue, thenUse);
        }
        if (elseDef && elseValue) {
          MemorySSAUse elseUse(elseDef, ifOp.getOperation(), /*operandIdx=*/i);
          dataFlowInfo.addMemoryUse(elseValue, elseUse);
        }

        // 创建phi信息
        PhiInfo phiInfo;
        phiInfo.type = PhiInfo::IF_RESULT;
        phiInfo.loopHeader = nullptr;
        phiInfo.comingFrom.initialValue = thenDef;
        phiInfo.comingFrom.yieldValue = elseDef;

        dataFlowInfo.addPhi(ifResult, phiInfo);

        LLVM_DEBUG(llvm::dbgs() << "  Phi: " << phiDef->getId()
                                << " for if result #" << i << "\n");
      }
    }
  }
}

void MemorySSABuilder::processForOp(scf::ForOp forOp, Instruction *inst,
                                    BasicBlock *loopBodyEntryBB) {
  // 实现scf.for的iter_args处理
  // iter_args在所有迭代中共享同一个definition

  LLVM_DEBUG(llvm::dbgs() << "Processing ForOp: " << forOp << "\n");

  unsigned numIterArgs = forOp.getInitArgs().size();

  for (unsigned i = 0; i < numIterArgs; i++) {
    Value iterArg = forOp.getRegionIterArg(i);
    Value initValue = forOp.getInitArgs()[i];

    // 查找initValue的definition
    MemorySSADef *initDef = dataFlowInfo.getMemoryDefinition(initValue);

    if (initDef) {
      // iter_arg使用同一个definition，不创建新版本
      dataFlowInfo.addMemoryDefinition(iterArg, initDef);

      // 创建PhiInfo（用于跟踪循环依赖）
      PhiInfo phiInfo;
      phiInfo.type = PhiInfo::ITER_ARG;
      phiInfo.loopHeader = nullptr; // 需要根据CFG确定
      phiInfo.comingFrom.initialValue = initDef;
      phiInfo.comingFrom.yieldValue = nullptr; // 将在处理yield时更新

      dataFlowInfo.addPhi(iterArg, phiInfo);

      // 为initValue创建use
      MemorySSAUse initUse(initDef, forOp.getOperation(), /*operandIdx=*/i);
      dataFlowInfo.addMemoryUse(initValue, initUse);

      LLVM_DEBUG(llvm::dbgs() << "  IterArg #" << i << ": " << iterArg << " -> "
                              << initDef->getId() << "\n");
    }
  }

  // 处理yield操作（在循环体中）
  Operation *yieldOp = MemorySSABuilderHelper::getYieldOp(forOp.getRegion());
  if (yieldOp) {
    for (unsigned i = 0; i < yieldOp->getNumOperands(); i++) {
      Value yieldedValue = yieldOp->getOperand(i);
      Value iterArg = forOp.getRegionIterArg(i);

      MemorySSADef *yieldedDef = dataFlowInfo.getMemoryDefinition(yieldedValue);

      if (yieldedDef) {
        // 更新PhiInfo
        if (PhiInfo *phiInfo = &dataFlowInfo.getPhi(iterArg)) {
          phiInfo->comingFrom.yieldValue = yieldedDef;

          // 为yieldedValue创建use
          MemorySSAUse yieldUse(yieldedDef, yieldOp, /*operandIdx=*/i);
          dataFlowInfo.addMemoryUse(yieldedValue, yieldUse);

          LLVM_DEBUG(llvm::dbgs() << "    Yield #" << i << ": " << yieldedValue
                                  << " updates iter_arg\n");
        }
      }
    }
  }
}

MemorySSADef *MemorySSABuilder::createDefinition(TensorObject *tensor,
                                                 Operation *op) {
  unsigned version = isParameter(op) ? 0 : ++nextVersion[tensor];
  auto *def = new MemorySSADef(tensor, op, version);
  allDefinitions.push_back(def);
  return def;
}

MemorySSAUse MemorySSABuilder::createUse(MemorySSADef *def, Operation *userOp,
                                         unsigned operandIdx) {
  return MemorySSAUse(def, userOp, operandIdx);
}

TensorObject *MemorySSABuilder::createTensorObject(Operation *op) {
  if (!op)
    return nullptr;

  // 根据操作创建tensor对象，使用独立的Tensor ID确保唯一性
  std::string name = getOpName(op);
  Type resultType = op->getResultTypes().front();

  SmallVector<int64_t> shape;
  Type elementType;

  // 使用Tensor.h中的辅助函数提取shape和element type
  extractShapeAndElementType(resultType, shape, elementType);

  // 设置默认的kind（可以根据操作类型推断）
  TensorObject::TensorKind kind = TensorObject::TensorKind::GLOBAL_MEMORY;

  auto *tensor = new TensorObject(name, shape, resultType, elementType, kind);

  // 缓存tensor对象
  if (!op->getResults().empty()) {
    tensorObjectCache[op->getResult(0)] = tensor;
  }

  return tensor;
}

std::string MemorySSABuilder::getOpName(Operation *op) {
  if (!op)
    return "unknown";

  // 基于操作类型和独立的Tensor ID生成名称
  std::string opName = op->getName().getStringRef().str();
  std::replace(opName.begin(), opName.end(), '.', '_');

  // 使用独立的Tensor ID生成器，确保唯一性
  return opName + "_tensor_" + std::to_string(++nextTensor[opName]);
}

//===----------------------------------------------------------------------===//
// MemorySSABuilderHelper
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace cfg {
namespace MemorySSABuilderHelper {

Type getResultType(Operation *op, unsigned resultIdx) {
  if (!op || resultIdx >= op->getNumResults())
    return Type();
  return op->getResultTypes()[resultIdx];
}

SmallVector<int64_t> getShapeFromValue(Value value) {
  Type type = value.getType();
  SmallVector<int64_t> shape;

  if (auto rankedType = mlir::dyn_cast<RankedTensorType>(type)) {
    shape.append(rankedType.getShape().begin(), rankedType.getShape().end());
  }

  return shape;
}

bool shapesEqual(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2) {
  if (shape1.size() != shape2.size())
    return false;
  return std::equal(shape1.begin(), shape1.end(), shape2.begin());
}

Operation *getYieldOp(Region &region) {
  if (region.empty())
    return nullptr;

  Block &block = region.back();
  if (block.empty())
    return nullptr;

  Operation &lastOp = block.back();
  if (isa<scf::YieldOp>(&lastOp)) {
    return &lastOp;
  }

  return nullptr;
}

std::string createUniqueTensorName(StringRef prefix, size_t id) {
  return prefix.str() + "_" + std::to_string(id);
}

bool shouldCreateNewVersion(Operation *op, MemorySSADef *currentDef) {
  if (!op || !currentDef)
    return true;

  // 如果操作会修改tensor内容，则创建新版本
  // 例如：tt.store、tt.trans等
  if (isa<triton::StoreOp>(op))
    return true;
  if (isa<triton::TransOp>(op))
    return true;

  // 其他操作可能复用当前definition
  return false;
}

} // namespace MemorySSABuilderHelper
} // namespace cfg
} // namespace triton
} // namespace mlir

bool MemorySSABuilder::isPointerBroadcastOrSplat(mlir::Operation *op) const {
  if (auto broadcastOp = mlir::dyn_cast<triton::BroadcastOp>(op)) {
    Type elemType = getElementTypeOrSelf(broadcastOp.getResult().getType());
    return mlir::isa<triton::PointerType>(elemType);
  }

  // 处理 SplatOp: 检查第一个 operand (src) 是否是指针
  if (auto splatOp = mlir::dyn_cast<triton::SplatOp>(op)) {
    return mlir::isa<triton::PointerType>(splatOp.getSrc().getType());
  }

  return false;
}

// 判断是否是返回新Tensor的操作（根据返回值类型判断，排除load）
bool MemorySSABuilder::isTensorWriter(Operation *op) const {
  if (!op || op->getNumResults() == 0)
    return false;

  // 检查返回值类型
  for (Value result : op->getResults()) {
    Type resultType = result.getType();
    // 如果是RankedTensorType（不是指针），则是TensorWriter
    if (mlir::isa<RankedTensorType>(resultType)) {
      // 但排除load（虽然load返回tensor，但它是从内存读取，不是"写入"或"创建"）
      if (mlir::isa<triton::LoadOp>(op) || mlir::isa<triton::StoreOp>(op))
        return false;

      if (mlir::isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op))
        return false;

      if (isPointerBroadcastOrSplat(op))
        return false;

      if (auto ptrType = mlir::dyn_cast<triton::PointerType>(
              getElementTypeOrSelf(resultType))) {
        return false;
      }
      return true;
    }
  }
  return false;
}
