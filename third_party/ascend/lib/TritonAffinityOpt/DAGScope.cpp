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

#include "TritonAffinityOpt/Passes.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Block.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include "Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

#include "TritonAffinityOpt/DAG.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DAGSCOPE
#include "ascend/include/TritonAffinityOpt/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hivm;

namespace {
struct DAGScopePass
    : public mlir::triton::impl::DAGScopeBase<
          DAGScopePass> {
  void runOnOperation() override;
};
} // namespace


static std::pair<Operation*, Operation*> encapsulateWithScope(triton::FuncOp funcOp) {
    Block &entryBlock = funcOp.getBody().front();
    Block &lastBlock = funcOp.getBody().back();
    Operation *terminator = lastBlock.getTerminator();


    // 辅助函数：判断操作是否应该被跳过
    auto shouldSkipOp = [](Operation *op) -> bool {
        return isa<arith::ConstantOp>(op) || isa<triton::GetProgramIdOp>(op) || isa<memref::AllocOp>(op);
    };

    // 第三步：准备要移动的操作列表（按顺序）
    SmallVector<Operation*> opsToMove;
    DenseMap<Operation*, int> opOrder;
    int order = 0;

    // 记录原始顺序并收集需要移动的操作
    for (Operation &op : lastBlock.without_terminator()) {
        opOrder[&op] = order++;
        if (!shouldSkipOp(&op)) {
            opsToMove.push_back(&op);
        }
    }

    // 按原始顺序排序
    std::sort(opsToMove.begin(), opsToMove.end(),
              [&](Operation *a, Operation *b) {
                  return opOrder[a] < opOrder[b];
              });

    if (opsToMove.empty()) {
        return std::make_pair(nullptr, nullptr);
    }

    // 第四步：创建scope操作并移动操作
    Operation *lastOpToMove = opsToMove.back();
    OpBuilder builder(&lastBlock, ++lastOpToMove->getIterator());

    // 创建第一个scope
    auto scopeOp = builder.create<scope::ScopeOp>(builder.getUnknownLoc(), llvm::ArrayRef<mlir::Type>{});
    scopeOp.getBodyRegion().emplaceBlock();
    Block *scopeBody = &scopeOp.getBodyRegion().front();

    // 移动操作到scope中
    OpBuilder scopeBuilder(scopeBody, scopeBody->end());
    DenseMap<Value, Value> valueMapping;

    for (Operation *op : opsToMove) {
        SmallVector<Value> originalResults = op->getResults();
        op->remove();
        scopeBuilder.insert(op);

        // 更新值的映射
        for (size_t i = 0; i < originalResults.size(); ++i) {
            valueMapping[originalResults[i]] = op->getResult(i);
        }
    }

    // 添加return操作
    scopeBuilder.create<scope::ReturnOp>(builder.getUnknownLoc());

    // 创建第二个scope（如果需要）
    scopeBuilder.setInsertionPointAfter(scopeOp);
    auto newScopeOp = scopeBuilder.create<scope::ScopeOp>(builder.getUnknownLoc(), llvm::ArrayRef<mlir::Type>{});
    newScopeOp.getRegion().emplaceBlock();

    OpBuilder newScopeBuilder(&newScopeOp.getRegion().front(),
                             newScopeOp.getRegion().front().begin());
    newScopeBuilder.create<scope::ReturnOp>(scopeOp->getLoc());

    // 设置属性
    auto vecAttr = hivm::TCoreTypeAttr::get(
        builder.getContext(),
        hivm::TCoreType::VECTOR);
    auto aicAttr = hivm::TCoreTypeAttr::get(
        builder.getContext(),
        hivm::TCoreType::CUBE);

    scopeOp->setAttr(hivm::TCoreTypeAttr::name, vecAttr);
    newScopeOp->setAttr(hivm::TCoreTypeAttr::name, aicAttr);

    return std::make_pair(scopeOp, newScopeOp);
}

struct OpMoveInfo {
    Operation* op;
    Operation* targetParent;  // 目标父操作（nullptr表示aicScope本身）
  };

// 递归遍历函数 - 优化版本
void collectOpsToMove(Operation* op, AffinityDAG::Graph& graph,
                     Operation* parentFor, llvm::SmallVector<OpMoveInfo>& aivToMove, llvm::SmallVector<OpMoveInfo>& cubeToMove) {
  // 检查当前操作是否需要移动
  bool needsMoveAiv = false;
  bool needsMoveCube = false;
  auto& valueTypes = graph.getValueTypes();
  // 检查结果类型
  int i = 0;
  for (auto res : op->getResults()) {
    i++;
    if (AffinityDAG::intersects(valueTypes[res], AffinityDAG::CoreType::VECTOR_ONLY)) {
      needsMoveAiv = true;
    }
    if (AffinityDAG::intersects(valueTypes[res], AffinityDAG::CoreType::CUBE_ONLY)) {
      needsMoveCube = true;
    }
  }

  if (isa<annotation::MarkOp>(op)) {
    auto res = op->getOperand(0);
    if (AffinityDAG::intersects(valueTypes[res], AffinityDAG::CoreType::VECTOR_ONLY)) {
      needsMoveAiv = true;
    }
    if (AffinityDAG::intersects(valueTypes[res], AffinityDAG::CoreType::CUBE_ONLY)) {
      needsMoveCube = true;
    }
  }
  // 检查特定操作类型
  if (isa<hivm::CopyOp>(op)) {
    needsMoveAiv = true;
  }

  // 检查特定操作类型
  if (isa<hivm::FixpipeOp>(op)) {
    needsMoveCube = true;
  }

  // 检查特定操作类型
  if (isa<scf::YieldOp>(op) || isa<scope::ScopeOp>(op) || isa<scf::ForOp>(op)) {
    needsMoveAiv = true;
    needsMoveCube = true;
  }

  if (isa<triton::StoreOp>(op)) {
      if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
        // 获取所有操作数列表
        auto operands = storeOp.getOperands();
        bool typeMatched = false;

        // 按顺序检查第1个、第0个、第2个操作数
        std::vector<size_t> checkOrder = {1, 0, 2};
        for (size_t idx : checkOrder) {
          // 先判断操作数索引是否有效，避免越界访问
          if (idx >= operands.size()) {
            continue;
          }
          auto operand = operands[idx];
          auto coreType = valueTypes[operand];

          if (coreType == AffinityDAG::CoreType::VECTOR_ONLY) {
            needsMoveAiv = true;
            typeMatched = true;
          } else if (coreType == AffinityDAG::CoreType::CUBE_ONLY) {
            needsMoveCube = true;
            typeMatched = true;
          }
        }
        // 所有指定操作数都不匹配时，执行原else逻辑
        if (!typeMatched) {
          needsMoveAiv = true;
          needsMoveCube = true;
        }
      }
  }

  if (isa<triton::AssertOp>(op)) {
      if (auto assertOp = dyn_cast<triton::AssertOp>(op)) {
        // 获取所有操作数列表
        auto operand = assertOp.getCondition();

        auto coreType = valueTypes[operand];
        if (coreType == AffinityDAG::CoreType::VECTOR_ONLY) {
          needsMoveAiv = true;
        } else if (coreType == AffinityDAG::CoreType::CUBE_ONLY) {
          needsMoveCube = true;
        } else {
          needsMoveAiv = true;
          needsMoveCube = true;
        }
      }
  }

  // 检查 Sync 操作的 tcore_type 属性
  if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
    mlir::OpBuilder builder(op);
    auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
    if (op->getAttr("tcore_type") == coreAttr) {
      needsMoveCube = true;
    }
    else {
      needsMoveAiv = true;
    }
  }

  // 如果不需要移动，直接返回
  if (!needsMoveAiv && !needsMoveCube) {
    llvm::outs()<<"Unsupport Op: "<< *op<<" \n";
  }

  // 处理 for 循环
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // 确定父级 for 循环
    Operation* targetParent = parentFor != nullptr ? parentFor : nullptr;
    aivToMove.push_back({op, targetParent});
    cubeToMove.push_back({op, targetParent});

    // 递归处理循环体
    for (auto &block : forOp.getRegion()) {
      for (auto &innerOp : block) {
        collectOpsToMove(&innerOp, graph, forOp, aivToMove, cubeToMove);
      }
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    // 确定父级 for 循环
    Operation* targetParent = parentFor != nullptr ? parentFor : nullptr;
    aivToMove.push_back({op, targetParent});
    cubeToMove.push_back({op, targetParent});

    // 递归处理循环体
    for (auto &block : ifOp.getThenRegion()) {
      for (auto &innerOp : block) {
        collectOpsToMove(&innerOp, graph, ifOp, aivToMove, cubeToMove);
      }
    }

    // 检查并遍历IfOp的else分支（如果存在）
    for (auto &block : ifOp.getElseRegion()) {
      for (auto &innerOp : block) {
        collectOpsToMove(&innerOp, graph, ifOp, aivToMove, cubeToMove);
      }
    }
  } else {
    if (needsMoveAiv) {
      // 处理其他操作
      aivToMove.push_back({op, parentFor});
    }
    if (needsMoveCube) {
      cubeToMove.push_back({op, parentFor});
    }

  }
}

mlir::Block* getBlockByIndex(mlir::Region& region, int blockIndex) {
  // 边界校验：索引非法时返回nullptr
  if (blockIndex < 0) return nullptr;

  int currentIdx = 0;
  for (auto& block : region) {
    if (currentIdx == blockIndex) {
      return &block; // 找到对应索引的Block，直接返回
    }
    currentIdx++;
  }
  // 索引越界时返回nullptr
  return nullptr;
}

void processOperationToMove(const OpMoveInfo& info,
                            llvm::DenseMap<mlir::Operation*, mlir::Operation*>& parentMap,
                            mlir::OpBuilder& builder,
                            mlir::IRMapping& mapper,
                            mlir::Block* aivBlock,
                            mlir::Operation* terminator,
                            AffinityDAG::Graph& graph,
                            int MoveType) {
  // llvm::outs()<<*info.op<<"  ssss\n\n\n";
  // llvm::outs().flush();
  // 获取原始Block信息并计算索引
  mlir::Block* originalBlock = info.op->getBlock();
  int originalRegionIndex = -1;
  int originalBlockIndex = -1;
  int blockCounter = 0;
  auto& valueTypes = graph.getValueTypes();
  if (originalBlock) {
    mlir::Operation* parentOp = info.op->getParentOp(); // 原始父操作
    if (parentOp) { // 确保父操作存在
      // 老版本MLIR用 getParent() 替代 getParentRegion()，返回值就是Region*
      mlir::Region* blockBelongsToRegion = originalBlock->getParent();
      int regionCounter = 0;
      for (auto& region : parentOp->getRegions()) { // 遍历父操作的所有region
        // 直接对比指针，判断当前region是否是block所属的region
        if (&region == blockBelongsToRegion) {
          originalRegionIndex = regionCounter;
          break;
        }
        regionCounter++;
      }
    }
  }

  if (originalBlock) {
    for (auto& block : originalBlock->getParent()->getBlocks()) {
      if (&block == originalBlock) {
        originalBlockIndex = blockCounter;
        break;
      }
      blockCounter++;
    }
  }

  if (originalBlockIndex == -1) {
    originalBlockIndex = 0;
  }
  if (originalRegionIndex == -1) {
    originalRegionIndex = 0;
  }

  // 处理 scf::ForOp 类型操作
  if (mlir::isa<mlir::scf::ForOp>(info.op)) {
    auto forOp = mlir::cast<mlir::scf::ForOp>(info.op);

    auto getMapped = [&](mlir::Value v) { return mapper.lookupOrDefault(v); };
    auto inputs = forOp.getInitArgs();
    auto outputs = forOp.getResults();

    // 分离需要移动到aivScope的参数
    llvm::SmallVector<mlir::Value> aivInputs;
    llvm::DenseMap<int, int> aivInputsMap;
    int aivIndex = 1;

    for (int i = 0; i < inputs.size(); ++i) {
      if (valueTypes[outputs[i]] != MoveType) {
        aivInputs.push_back(inputs[i]);
        aivInputsMap[i + 1] = aivIndex;
        aivIndex++;
      }
    }

    // 创建新的for循环
    auto aivForOp = builder.create<mlir::scf::ForOp>(
        forOp.getLoc(),
        getMapped(forOp.getLowerBound()),
        getMapped(forOp.getUpperBound()),
        getMapped(forOp.getStep()),
        llvm::to_vector(llvm::map_range(aivInputs, getMapped))
    );

    // 清空循环体
    if (!aivForOp.getBody()->empty()) {
      aivForOp.getBody()->getTerminator()->erase();
    }

    // 处理原始循环的yield操作
    auto oldBody = forOp.getBody();
    auto oldYield = mlir::dyn_cast<mlir::scf::YieldOp>(oldBody->getTerminator());
    assert(oldYield && "scf::ForOp must have a yield terminator");

    llvm::SmallVector<mlir::Value> aivYieldOperands;
    for (int i = 0; i < inputs.size(); ++i) {
      if (valueTypes[outputs[i]] != MoveType) {
        aivYieldOperands.push_back(oldYield.getOperand(i));
      }
    }

    // 映射循环参数
    auto oldBodyArgs = forOp.getBody()->getArguments();
    auto aivBodyArgs = aivForOp.getBody()->getArguments();

    for (auto it = aivInputsMap.begin(); it != aivInputsMap.end(); ++it) {
      int oldInputIndex = it->first;
      int mappedNewIndex = it->second;
      mapper.map(oldBodyArgs[oldInputIndex], aivBodyArgs[mappedNewIndex]);
      mapper.map((*info.op).getResults()[oldInputIndex - 1], aivForOp->getResults()[mappedNewIndex - 1]);
    }
    mapper.map(oldBodyArgs[0], aivBodyArgs[0]);

    // 将新循环移动到目标位置
    if (info.targetParent == nullptr) {
      mlir::Block* targetBlock = aivBlock;
      if (terminator) {
        aivForOp->moveBefore(terminator);
      } else {
        aivForOp->moveBefore(targetBlock, targetBlock->end());
      }
      parentMap[forOp] = aivForOp;
    } else {
      auto targetParent = parentMap[info.targetParent];
      auto& region = targetParent->getRegion(originalRegionIndex);

      if (region.empty()) {
        region.push_back(new mlir::Block());
      }

      mlir::Block* targetBlock = getBlockByIndex(region, originalBlockIndex);
      if (targetBlock) {
        aivForOp->moveBefore(targetBlock, targetBlock->end());
        parentMap[forOp] = aivForOp;
      } else {
        llvm::outs()<<"Can't find block by index\n";
      }
      
    }
  }

  // 处理 scf::YieldOp 类型操作
  else if (mlir::isa<mlir::scf::YieldOp>(info.op)) {
    auto yieldOp = mlir::cast<mlir::scf::YieldOp>(info.op);

    // 处理父节点为 scf::ForOp 的情况
    if (auto parentForOp = mlir::dyn_cast<mlir::scf::ForOp>(info.targetParent)) {
      auto it = parentMap.find(parentForOp);
      if (it == parentMap.end()) {
        return;
      }
      auto targetOp = it->second;
      auto newForOp = mlir::cast<mlir::scf::ForOp>(targetOp);

      auto oldInputs = parentForOp.getInitArgs();
      auto oldOutputs = parentForOp.getResults();
      auto oldYieldOperands = yieldOp.getOperands();

      llvm::SmallVector<mlir::Value> newYieldOperands;
      for (int i = 0; i < oldInputs.size(); ++i) {
        if (valueTypes[oldOutputs[i]] != MoveType) {
          mlir::Value oldOperand = oldYieldOperands[i];
          mlir::Value newOperand = mapper.lookupOrDefault(oldOperand);
          newYieldOperands.push_back(newOperand);
        }
      }

      auto newYieldOp = builder.create<mlir::scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      auto& region = newForOp->getRegion(0);
      mlir::Block* targetBlock = &region.front();
      newYieldOp->moveBefore(targetBlock, targetBlock->end());
    }
    // 处理父节点为 scf::IfOp 的情况
    else if (auto parentIfOp = mlir::dyn_cast<mlir::scf::IfOp>(info.targetParent)) {
      auto it = parentMap.find(parentIfOp);
      if (it == parentMap.end()) {
        return;
      }
      auto targetOp = it->second;
      auto newIfOp = mlir::cast<mlir::scf::IfOp>(targetOp);

      auto oldInputs = parentIfOp.getResults();
      auto oldOutputs = parentIfOp.getResults();
      auto oldYieldOperands = yieldOp.getOperands();

      llvm::SmallVector<mlir::Value> newYieldOperands;
      for (int i = 0; i < oldInputs.size(); ++i) {
        if (valueTypes[oldOutputs[i]] != MoveType) {
          mlir::Value oldOperand = oldYieldOperands[i];
          mlir::Value newOperand = mapper.lookupOrDefault(oldOperand);
          newYieldOperands.push_back(newOperand);
        }
      }

      auto& region = newIfOp->getRegion(originalRegionIndex);
      auto newYieldOp = builder.create<mlir::scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      mlir::Block* targetBlock = getBlockByIndex(region, originalBlockIndex);
      if (targetBlock) {
        newYieldOp->moveBefore(targetBlock, targetBlock->end());
      } else {
        llvm::outs()<<"Can't find block by index\n";
      }
    }
  }

  // 处理 scf::IfOp 类型操作
  else if (mlir::isa<mlir::scf::IfOp>(info.op)) {
    auto ifOp = mlir::cast<mlir::scf::IfOp>(info.op);

    auto getMapped = [&](mlir::Value v) { return mapper.lookupOrDefault(v); };
    mlir::Value condition = ifOp.getCondition();

    // 分离需要移动到aivScope的结果
    llvm::SmallVector<mlir::Value> aivResults;
    llvm::SmallVector<Type> aivResultTypes;
    llvm::DenseMap<int, int> aivResultMap;
    int aivResultIndex = 0;

    for (int i = 0; i < ifOp.getNumResults(); ++i) {
      mlir::Value result = ifOp.getResult(i);
      if (valueTypes[result] != MoveType) {
        aivResults.push_back(result);
        aivResultTypes.push_back(result.getType());
        aivResultMap[i] = aivResultIndex;
        aivResultIndex++;
      }
    }

    // 创建新的if操作
    auto aivIfOp = builder.create<mlir::scf::IfOp>(
        ifOp.getLoc(),
        aivResultTypes,
        getMapped(condition)
    );

    // 映射if操作结果
    for (auto& [oldIdx, newIdx] : aivResultMap) {
      mapper.map(ifOp.getResult(oldIdx), aivIfOp.getResult(newIdx));
    }

    // 初始化then和else区域
    mlir::Region& thenRegion = aivIfOp.getThenRegion();
    mlir::Block* thenBlock = new mlir::Block();
    thenRegion.push_back(thenBlock);

    mlir::Region& elseRegion = ifOp.getElseRegion();
    if (!elseRegion.empty()) {
        mlir::Region& elseRegion = aivIfOp.getElseRegion();
        mlir::Block* elseBlock = new mlir::Block();
        elseRegion.push_back(elseBlock);
    }

    // 将新if操作移动到目标位置
    if (info.targetParent == nullptr) {
      mlir::Block* targetBlock = aivBlock;
      if (terminator) {
        aivIfOp->moveBefore(terminator);
      } else {
        aivIfOp->moveBefore(targetBlock, targetBlock->end());
      }
      parentMap[ifOp] = aivIfOp;
    } else {
      auto& region = parentMap[info.targetParent]->getRegion(originalRegionIndex);
      if (region.empty()) {
        region.push_back(new mlir::Block());
      }

      mlir::Block* targetBlock = getBlockByIndex(region, originalBlockIndex);
      if (targetBlock) {
        aivIfOp->moveBefore(targetBlock, targetBlock->end());
        parentMap[ifOp] = aivIfOp;
      } else {
        llvm::outs()<<"Can't find block by index\n";
      }
    }
  }

  // 处理其他类型操作（克隆）
  else {
    auto clonedOp = builder.clone(*info.op, mapper);
    auto numberRes = clonedOp->getNumResults();
    for (auto i = 0; i < numberRes; i++) {
      mapper.map((*info.op).getResults()[i], clonedOp->getResults()[i]);
    }

    if (info.targetParent == nullptr) {
      mlir::Block* targetBlock = aivBlock;
      clonedOp->moveBefore(terminator);
      parentMap[info.op] = clonedOp;
    } else {
      auto parentIt = parentMap.find(info.targetParent);
      auto mappedParentOp = parentIt->second;
      auto& region = mappedParentOp->getRegion(originalRegionIndex);

      if (region.empty()) {
        region.push_back(new mlir::Block());
      }

      mlir::Block* targetBlock = getBlockByIndex(region, originalBlockIndex);
      if (targetBlock) {
        clonedOp->moveBefore(targetBlock, targetBlock->end());
      } else {
        llvm::outs()<<"Can't find block by index\n";
      }
      
    }
  }
}

static void SplitScope(triton::FuncOp funcOp, AffinityDAG::Graph& graph, Operation* aivScope, Operation* aicScope, ModuleOp module) {
  llvm::SmallVector<OpMoveInfo> aivToMove;
  llvm::SmallVector<OpMoveInfo> cubeToMove;
  for (auto &block : aivScope->getRegion(0)) {
    for (auto &op : block) {
      collectOpsToMove(&op, graph, nullptr, aivToMove, cubeToMove);
    }
  }
  mlir::IRMapping aivmapper;
  mlir::OpBuilder builder(aivScope);
  llvm::DenseMap<Operation*, Operation*> aivparentMap;

  // 第二遍：实际移动操作
  // 先移动for循环
  mlir::Block* aivBlock = &aivScope->getRegion(0).front(); // 或者使用合适的block
  SmallVector<Operation*> deleteOp;
  auto* terminator = aivBlock->getTerminator();
    // 如果操作已被使用，直接跳过
  llvm::SmallVector<mlir::Operation*> aivUsedOp; // 改为函数内静态，保持原有逻辑
  for (const auto& info : aivToMove) {
    if (std::find(aivUsedOp.begin(), aivUsedOp.end(), info.op) != aivUsedOp.end()) {
      return;
    }
    aivUsedOp.push_back(info.op);
    processOperationToMove(info, aivparentMap, builder, aivmapper, aivBlock, terminator, graph, AffinityDAG::CoreType::CUBE_ONLY);
  }

  llvm::DenseMap<Operation*, Operation*> aicparentMap;
  mlir::IRMapping aicmapper;
  mlir::Block* aicBlock = &aicScope->getRegion(0).front(); // 或者使用合适的block
  terminator = aicBlock->getTerminator();
  llvm::SmallVector<mlir::Operation*> aicUsedOp; // 改为函数内静态，保持原有逻辑
  for (const auto& info : cubeToMove) {
    if (std::find(aicUsedOp.begin(), aicUsedOp.end(), info.op) != aicUsedOp.end()) {
      return;
    }
    aicUsedOp.push_back(info.op);
    processOperationToMove(info, aicparentMap, builder, aicmapper, aicBlock, terminator, graph, AffinityDAG::CoreType::VECTOR_ONLY);
  }

  for (const auto& info : aivToMove) {
    if (std::find(deleteOp.begin(), deleteOp.end(), info.op) == deleteOp.end()) {
        deleteOp.push_back(info.op);
    }
  }
  for (const auto& info : cubeToMove) {
    if (std::find(deleteOp.begin(), deleteOp.end(), info.op) == deleteOp.end()) {
        deleteOp.push_back(info.op);
    }
  }

  // llvm::outs() << "\n" << module<<" ====== ddd ====== \n\n\n";
  // llvm::outs().flush();
  for (auto it = deleteOp.rbegin(); it != deleteOp.rend(); ++it) {
    (*it)->erase();  // 解引用反向迭代器，调用 erase 方法
  }
  return;

}

 /// 创建setop
 	 static hivm::SyncBlockSetOp createSyncBlockSetOp(
 	     OpBuilder &builder,
 	     Location loc,
 	     hivm::TCoreType coreType,
 	     hivm::PIPE setPipeEnum,
 	     hivm::PIPE waitPipeEnum,
 	     int64_t flag) {
 	   MLIRContext *ctx = builder.getContext();
 	   auto coreAttr = hivm::TCoreTypeAttr::get(ctx, coreType);
 	   auto setPipe = hivm::PipeAttr::get(ctx, setPipeEnum);
 	   auto waitPipe = hivm::PipeAttr::get(ctx, waitPipeEnum);
 	   auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);
 	   return builder.create<hivm::SyncBlockSetOp>(loc, coreAttr, setPipe, waitPipe, flagId);
 	 }

 	 /// 创建waitop
 	 static hivm::SyncBlockWaitOp createSyncBlockWaitOp(
 	     OpBuilder &builder,
 	     Location loc,
 	     hivm::TCoreType coreType,
 	     hivm::PIPE setPipeEnum,
 	     hivm::PIPE waitPipeEnum,
 	     int64_t flag) {
 	   MLIRContext *ctx = builder.getContext();
 	   auto coreAttr = hivm::TCoreTypeAttr::get(ctx, coreType);
 	   auto setPipe = hivm::PipeAttr::get(ctx, setPipeEnum);
 	   auto waitPipe = hivm::PipeAttr::get(ctx, waitPipeEnum);
 	   auto flagId = builder.getIntegerAttr(builder.getI64Type(), flag);
 	   return builder.create<hivm::SyncBlockWaitOp>(loc, coreAttr, setPipe, waitPipe, flagId);
 	 }

 	 // 在scope return前插入wait
 	 static void insertWaitBeforeFinalReturn(Region *region, OpBuilder &builder, int64_t flag, bool coretypebool) {
 	   for (Block &block : *region) {
 	     if (auto returnOp = dyn_cast_or_null<scope::ReturnOp>(block.getTerminator())) {
 	       builder.setInsertionPoint(returnOp);
 	       if (coretypebool) {
 	         createSyncBlockWaitOp(
 	           builder,
 	           returnOp->getLoc(),
 	           hivm::TCoreType::CUBE,
 	           hivm::PIPE::PIPE_V,
 	           hivm::PIPE::PIPE_FIX,
 	           flag
 	         );
 	         return;
 	       }
 	       else {
 	         createSyncBlockWaitOp(
 	           builder,
 	           returnOp->getLoc(),
 	           hivm::TCoreType::VECTOR,
 	           hivm::PIPE::PIPE_M,
 	           hivm::PIPE::PIPE_MTE3,
 	           flag
 	         );
 	         return;
 	       }
 	     }
 	   }
 	 }

 	 /// 在scope内起始位置加上set
 	 static void insertSetAtRegionStart(Region *region, OpBuilder &builder, int64_t flag, bool coretypebool) {
 	   if (!region->empty()) {
 	     Block &entry = region->front();
 	     Location loc = entry.empty() ? region->getParentOp()->getLoc() : entry.front().getLoc();
 	     builder.setInsertionPointToStart(&entry);
 	     if (coretypebool) {
 	       createSyncBlockSetOp(
 	         builder,
 	         loc,
 	         hivm::TCoreType::VECTOR,
 	         hivm::PIPE::PIPE_V,
 	         hivm::PIPE::PIPE_FIX,
 	         flag
 	       );
 	     }
 	     else {
 	       createSyncBlockSetOp(
 	         builder,
 	         loc,
 	         hivm::TCoreType::CUBE,
 	         hivm::PIPE::PIPE_M,
 	         hivm::PIPE::PIPE_MTE3,
 	         flag
 	       );
 	     }
 	   }
 	 }

 	 static Operation *findNextSyncBlockSetAfter(Operation *startOp) {
 	   Block *block = startOp->getBlock();
 	   auto it = ++startOp->getIterator();
 	   for (; it != block->end(); ++it) {
 	     if (isa<hivm::SyncBlockSetOp>(*it))
 	       return &*it;
 	   }
 	   return nullptr;
 	 }

 	 static hivm::SyncBlockWaitOp findWaitOpInRegionWithFlag(Region *region, int64_t flag) {
 	   hivm::SyncBlockWaitOp result;
 	   region->walk([&](hivm::SyncBlockWaitOp op) {
 	     auto flagAttr = op->getAttrOfType<IntegerAttr>("static_flag_id");
 	     if (flagAttr && flagAttr.getInt() == flag) {
 	       result = op;
 	       return WalkResult::interrupt();
 	     }
 	     return WalkResult::advance();
 	   });
 	   return result;
 	 }

 	 static Operation *findInsertionPointAfterWaitForAIV(Operation *waitOp) {
 	   Block *block = waitOp->getBlock();
      auto it = ++waitOp->getIterator();
      
      for (; it != block->end(); ++it) {
        if (isa<bufferization::ToBufferOp>(*it) || isa<scf::YieldOp>(*it)) {
          break;
        }
      }

      while (it != block->begin()) {
        auto prevIt = std::prev(it);
        if (isa<triton::AdvanceOp>(*prevIt)) {
          it = prevIt;
        } else {
          break;
        }
      }

      return &*it;
 	 }

 	 static Operation *findInsertionPointAfterWaitForAIC(Operation *waitOp) {
 	   Block *block = waitOp->getBlock();
 	   auto it = ++waitOp->getIterator();
 	   for (; it != block->end(); ++it) {
 	     if (auto fixpipe = dyn_cast<hivm::FixpipeOp>(*it)) {
 	       if (it != block->begin()) {
 	         auto prev = std::prev(it);
 	         if (isa<hivm::SyncBlockWaitOp>(*prev))
 	           return &*prev;
 	       }
 	       return &*it;
 	     }
 	     if (isa<scf::YieldOp>(*it))
 	       return &*it;
 	   }
 	   return nullptr;
 	 }

 	 // 查找 FixpipeOp 下一行的 sync_block_set 操作的 flag 值
    static int findFixPipeFlagSafe(hivm::FixpipeOp fixpipeOp) {
        mlir::Operation *fixpipeOperation = fixpipeOp.getOperation();
        if (!fixpipeOperation || !fixpipeOperation->getBlock()) {
          return -1;
      }

      // 获取 FixpipeOp 的迭代器
      auto it = ++fixpipeOperation->getIterator();
      
      // 遍历后续操作直到找到 sync_block_set
      while (it != fixpipeOperation->getBlock()->end()) {
          mlir::Operation &op = *it++;
          
          if (op.getName().getStringRef() == "hivm.hir.sync_block_set") {
              auto staticFlagAttr = op.getAttrOfType<mlir::IntegerAttr>("static_flag_id");
              return staticFlagAttr.getInt();
              break;
          }
      }

      return -1;

    }

    /// cube处理逻辑
    static void processFixpipeOpsInAIC(
        Region *aicRegion,
        Region *aivRegion) {

      MLIRContext *ctx = aicRegion->getContext();
      OpBuilder builder(ctx);
      SmallVector<hivm::FixpipeOp> fixpipes;
      aicRegion->walk([&](hivm::FixpipeOp op) {
        fixpipes.push_back(op);
      });


      for (auto fixpipeOp : fixpipes) {
      
      auto newflag = findFixPipeFlagSafe(fixpipeOp);
        // 1. 在 FixpipeOp 前插 Wait
        builder.setInsertionPoint(fixpipeOp);
        createSyncBlockWaitOp(
          builder,
          fixpipeOp->getLoc(),
          hivm::TCoreType::CUBE,
          hivm::PIPE::PIPE_V,
          hivm::PIPE::PIPE_FIX,
          newflag);
        bool coretypebool = true;

        // 2. 在 aicRegion 末尾 Return 前插 Wait
        insertWaitBeforeFinalReturn(aicRegion, builder, newflag, coretypebool);

        // 3. 在 aivRegion 开头插 Set
        insertSetAtRegionStart(aivRegion, builder, newflag, coretypebool);

        // 4. 在 aicRegion 向后找 SyncBlockSetOp
        if (auto *nextSetOp = findNextSyncBlockSetAfter(fixpipeOp)) {
          auto setFlagAttr = nextSetOp->getAttrOfType<IntegerAttr>("static_flag_id");
          // 调试：打印set
          // llvm::dbgs() << "aicnextSetOp:";
          // nextSetOp->dump();
          if (!setFlagAttr) {
            llvm::dbgs() << "AIC can not find setop in aic\n";
            continue;
          }
          int64_t setflag = setFlagAttr.getInt();

          // 5. 在 aivRegion 中找 flag=setflag 的 WaitOp
          auto targetWait = findWaitOpInRegionWithFlag(aivRegion, setflag);
          if (!targetWait) {
            llvm::dbgs() << "AIC can not find waitop in aiv\n";
            continue;
          }

          // 调试：打印wait
          // llvm::dbgs() << "aictargetWait:";
          // llvm::dbgs() << targetWait << "\n";

          // 6. 从该 Wait 向下找 ToMemrefOp 或 Yield，插 Set(newflag)
          if (auto *insertPt = findInsertionPointAfterWaitForAIV(targetWait)) {
            builder.setInsertionPoint(insertPt);
            createSyncBlockSetOp(
              builder,
              fixpipeOp->getLoc(),
              hivm::TCoreType::VECTOR,
              hivm::PIPE::PIPE_V,
              hivm::PIPE::PIPE_FIX,
              newflag);
          }
        }
      }
    }

    // 查找 copyOp 下一行的 sync_block_set 操作的 flag 值
    static int findCopyFlagSafe(bufferization::ToBufferOp toMemrefOp) {
      mlir::Operation *toMemrefOperation = toMemrefOp.getOperation();
      if (!toMemrefOperation || !toMemrefOperation->getBlock()) {
          return -1;
      }

      // 获取 copyOp 的迭代器
      auto it = ++toMemrefOperation->getIterator();
      
      // 遍历后续操作直到找到 sync_block_set
      while (it != toMemrefOperation->getBlock()->end()) {
          mlir::Operation &op = *it++;
          
          if (op.getName().getStringRef() == "hivm.hir.sync_block_set") {
              auto staticFlagAttr = op.getAttrOfType<mlir::IntegerAttr>("static_flag_id");
              return staticFlagAttr.getInt();
              break;
          }
      }

      return -1;

    }
    /// vector处理逻辑
    static void processToMemrefOpsInAIV(
    Region *aivRegion,
    Region *aicRegion) {

      MLIRContext *ctx = aivRegion->getContext();
      OpBuilder builder(ctx);
      SmallVector<bufferization::ToBufferOp> toMemrefs;
      aivRegion->walk([&](bufferization::ToBufferOp op) {
        toMemrefs.push_back(op);
      });

      for (auto toMemrefOp : toMemrefs) {
      auto newflag = findCopyFlagSafe(toMemrefOp);

        // 1. 在 ToMemrefOp 前插 Wait
        builder.setInsertionPoint(toMemrefOp);
        createSyncBlockWaitOp(
          builder,
          toMemrefOp->getLoc(),
          hivm::TCoreType::VECTOR,
          hivm::PIPE::PIPE_M,
          hivm::PIPE::PIPE_MTE3,
          newflag);
        bool coretypebool = false;

        // 2. 在 aivRegion 末尾 Return 前插 Wait
        insertWaitBeforeFinalReturn(aivRegion, builder, newflag, coretypebool);

        // 3. 在 aicRegion 开头插 Set
        insertSetAtRegionStart(aicRegion, builder, newflag, coretypebool);

        // 4. 在 aivRegion 向后找 SyncBlockSetOp
        if (auto *nextSetOp = findNextSyncBlockSetAfter(toMemrefOp)) {
          auto setFlagAttr = nextSetOp->getAttrOfType<IntegerAttr>("static_flag_id");
          // 调试：打印set及其所有attribute
          // llvm::dbgs() << "aivnextSetOp:";
          // nextSetOp->dump();
          // llvm::dbgs() << "Attributes:\n";
          // for (auto namedAttr : nextSetOp->getAttrs()) {
          //   llvm::dbgs() << "  " << namedAttr.getName() << " = ";
          //   namedAttr.getValue().print(llvm::dbgs());
          //   llvm::dbgs() << "\n";
          // }
          if (!setFlagAttr) {
            llvm::dbgs() << "AIV can not find setop in aiv\n";
            continue;
          }
          int64_t setflag = setFlagAttr.getInt();

          // 5. 在 aicRegion 中找 flag=setflag 的 WaitOp
          auto targetWait = findWaitOpInRegionWithFlag(aicRegion, setflag);

          if (!targetWait) {
            llvm::dbgs() << "AIV can not find waitop in aic\n";
            continue;
          }

          // 调试：打印wait
          // llvm::dbgs() << "aivtargetWait:";
          // llvm::dbgs() << targetWait << "\n";

          // 6. 从该 Wait 向下找 Fixpipe 前 Wait 或 Yield，插 Set(newflag)
          if (auto *insertPt = findInsertionPointAfterWaitForAIC(targetWait)) {
            builder.setInsertionPoint(insertPt);
            createSyncBlockSetOp(
              builder,
              toMemrefOp->getLoc(),
              hivm::TCoreType::CUBE,
              hivm::PIPE::PIPE_M,
              hivm::PIPE::PIPE_MTE3,
              newflag);
          }
        }
      }
    }

 	 /// 同步点增强
    void addSyncOpsForBufferWait(ModuleOp module) {
      for (auto funcOp : llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
        if (funcOp.getBody().empty()) {
          continue;
        }

        Region *aicRegion = nullptr;
        Region *aivRegion = nullptr;

        funcOp.walk([&](scope::ScopeOp scopeOp) {
          auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
              hivm::TCoreTypeAttr::name);
          if (!coreTypeAttr) return;

          if (coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE) {
            aicRegion = &scopeOp.getRegion();
          }
          if (coreTypeAttr.getTcoretype() == hivm::TCoreType::VECTOR) {
            aivRegion = &scopeOp.getRegion();
          }
        });

        if (!aicRegion || !aivRegion) {
          continue;
        }

        processFixpipeOpsInAIC(aicRegion, aivRegion);
        processToMemrefOpsInAIV(aivRegion, aicRegion);
      }
    }


void DAGScopePass::runOnOperation() {
  auto module = getOperation();
  // llvm::outs()<<module<<"  before dag scope\n\n\n";

  mlir::OpBuilder builder(&getContext());

  for (auto funcOp : llvm::make_early_inc_range(module.getOps<triton::FuncOp>())) {
    // skip invalid function
    if (funcOp.getBody().empty()) {
      continue;
    }

    // 收集所有 memref.alloc 操作
    llvm::SmallVector<mlir::Operation*> allocOps;

    // 遍历函数中的所有操作（包括嵌套区域中的操作）
    funcOp.walk([&](mlir::Operation *op) {
        if (mlir::isa<memref::AllocOp>(op)) {
          allocOps.push_back(op);
        }
    });

    mlir::Block& entryBlock = funcOp.getBody().front();
    mlir::Block::iterator insertPos = entryBlock.begin();

    // 将 alloc 操作移动到函数的最前面
    for (mlir::Operation* allocOp : allocOps) {
        // 如果 alloc 操作已经是最前面的操作，跳过
        if (allocOp->getBlock() == &entryBlock &&
            allocOp->isBeforeInBlock(&*insertPos)) {
            continue;
        }

        // 将 alloc 操作移动到指定位置
        allocOp->moveBefore(&entryBlock, insertPos);
    }

    auto funcName = funcOp.getName();
    auto* graph_ptr = AffinityDAG::GraphManager::getInstance().getGraph(funcName);
    if (!graph_ptr) {
      continue;
    }
    auto& main_graph = *graph_ptr;


    auto ScopeList = encapsulateWithScope(funcOp);
    auto aivScope = ScopeList.first;    // 第一个元素
    auto aicScope = ScopeList.second; // 第二个元素

    SplitScope(funcOp, main_graph, aivScope, aicScope, module);
  }

    addSyncOpsForBufferWait(module);
    // llvm::outs()<<module<<"  after dag scope\n\n\n";
    // llvm::outs()<<module<<"  after dag scope\n\n\n";
  return;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createDAGScopePass() {
  return std::make_unique<DAGScopePass>();
}

