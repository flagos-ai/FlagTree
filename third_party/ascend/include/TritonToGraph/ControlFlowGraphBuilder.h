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

#ifndef TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H
#define TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H

#include "TritonToGraph/ControlFlowGraph.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <optional>
#include <stack>

namespace mlir {
namespace triton {
namespace cfg {

// 构建控制流图的 Pass
class BuildCFGPass
    : public PassWrapper<BuildCFGPass, OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BuildCFGPass)

  BuildCFGPass() = default;

  StringRef getArgument() const override { return "build-cfg"; }
  StringRef getDescription() const override {
    return "Build Control Flow Graph from TTIR";
  }

  void runOnOperation() override;

  // Pass 选项
  std::string outputDir = ".";

  // 获取依赖的方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, scf::SCFDialect,
                    cf::ControlFlowDialect>();
  }

protected:
  // 构建单个函数的 CFG
  std::unique_ptr<cfg::ControlFlowGraph> buildForFunction(triton::FuncOp func);
};

// 添加数据结构定义
// IF_COND 的 yield value 和 result value 的对应关系
struct IfYieldResultMapping {
  // then 分支的 yield values（来自 scf.yield 操作的操作数）
  SmallVector<Value> thenYieldValues;
  // else 分支的 yield values（如果有 else 分支）
  SmallVector<Value> elseYieldValues;
  // if 操作的 result values
  SmallVector<Value> resultValues;
  // 对应关系: resultValues[i] 对应 thenYieldValues[i] 或 elseYieldValues[i]
};

// FOR_COND 的 yield value 和 iter args value 的对应关系
struct ForYieldIterArgMapping {
  // yield 操作的 values（来自循环体末尾的 scf.yield）
  SmallVector<Value> yieldValues;
  // iter_args（循环初始参数，对应 for 操作的 iter_args）
  SmallVector<Value> iterArgValues;
  // for 操作的 result values
  SmallVector<Value> resultValues;
  // 对应关系: iterArgValues[i] 在循环体中使用时被更新，yieldValues[i] 是新的值
  // resultValues[i] 对应最后一次迭代的 yieldValues[i]
};

// COND_BR 的 true/false 分支信息
struct CondBranchMapping {
  // true 分支的目标块参数值（来自 cf.cond_br 的 trueOperands）
  SmallVector<Value> trueOperands;
  // false 分支的目标块参数值（来自 cf.cond_br 的 falseOperands）
  SmallVector<Value> falseOperands;
  // 条件值
  Value condition;
  // true 分支目标块
  Block *trueDest;
  // false 分支目标块
  Block *falseDest;
};

// BR 的分支信息
struct BranchMapping {
  // 目标块参数值（来自 cf.br 的 destOperands）
  SmallVector<Value> destOperands;
  // 目标块
  Block *dest;
};

// 独立的 CFG 构建器类（用于非 Pass 场景）
class ControlFlowGraphBuilder {
public:
  // 为函数构建 CFG
  std::unique_ptr<cfg::ControlFlowGraph> build(triton::FuncOp func);

  // 为模块构建所有函数的 CFG
  std::vector<std::unique_ptr<cfg::ControlFlowGraph>>
  buildForModule(ModuleOp module);

  // 处理一个 region，返回该 region 的入口块和出口块
  struct RegionBlocks {
    cfg::BasicBlock *entryBlock;
    cfg::BasicBlock *exitBlock;
  };

  RegionBlocks buildForRegion(Region &region, cfg::ControlFlowGraph &cfg,
                              cfg::BasicBlock *entryBlock,
                              cfg::BasicBlock *parentStructure = nullptr);

  // 处理 block 中的操作，返回最后处理的基本块
  cfg::BasicBlock *processBlock(Block &block, cfg::ControlFlowGraph &cfg,
                                cfg::BasicBlock *currentBB,
                                cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.if 操作，返回 if 后面的基本块
  cfg::BasicBlock *handleIfOp(scf::IfOp ifOp, cfg::ControlFlowGraph &cfg,
                              cfg::BasicBlock *currentBB,
                              cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.for 操作，返回 for 后面的基本块
  cfg::BasicBlock *handleForOp(scf::ForOp forOp, cfg::ControlFlowGraph &cfg,
                               cfg::BasicBlock *currentBB,
                               cfg::BasicBlock *parentStructure = nullptr);

  // 处理 scf.while 操作，返回 while 后面的基本块
  cfg::BasicBlock *handleWhileOp(scf::WhileOp whileOp,
                                 cfg::ControlFlowGraph &cfg,
                                 cfg::BasicBlock *currentBB,
                                 cfg::BasicBlock *parentStructure = nullptr);

  // 处理 cf.cond_br 操作，返回条件分支后面的基本块
  cfg::BasicBlock *
  handleCondBranchOp(cf::CondBranchOp condBrOp, cfg::ControlFlowGraph &cfg,
                     cfg::BasicBlock *currentBB,
                     cfg::BasicBlock *parentStructure = nullptr);

  // 处理 cf.br 操作，返回无条件跳转后面的基本块
  cfg::BasicBlock *handleBranchOp(cf::BranchOp brOp, cfg::ControlFlowGraph &cfg,
                                  cfg::BasicBlock *currentBB,
                                  cfg::BasicBlock *parentStructure = nullptr);

  // 创建一个新的指令并添加到 basic block
  cfg::Instruction *createInstruction(Operation *op,
                                      cfg::BasicBlock *parentBlock,
                                      cfg::ControlFlowGraph &cfg);

  // 1. 快速收集所有的 IF_COND 基本块
  // 遍历 CFG 中所有基本块，返回类型为 IF_COND 的基本块列表
  SmallVector<cfg::BasicBlock *>
  collectIfCondBlocks(cfg::ControlFlowGraph &cfg);

  // 2. 快速收集所有的 FOR_COND 基本块
  // 遍历 CFG 中所有基本块，返回类型为 FOR_COND 的基本块列表
  SmallVector<cfg::BasicBlock *>
  collectForCondBlocks(cfg::ControlFlowGraph &cfg);

  // 3. 获取 IF_COND 对应的 yield value 和 result value 的对应关系
  // 参数: IF_COND 类型的基本块
  // 返回: IfYieldResultMapping 结构体，包含 then/else 的 yield values 和 result
  // values
  std::optional<IfYieldResultMapping>
  getIfYieldResultMapping(cfg::BasicBlock *ifCondBB);

  // 4. 获取 FOR_COND 对应的 yield value 和 iter args value 的对应关系
  // 参数: FOR_COND 类型的基本块
  // 返回: ForYieldIterArgMapping 结构体，包含 yield values、iter_args 和 result
  // values
  std::optional<ForYieldIterArgMapping>
  getForYieldIterArgMapping(cfg::BasicBlock *forCondBB);

  // 5. 快速收集所有的 COND_BR 基本块
  // 遍历 CFG 中所有基本块，返回类型为 COND_BR 的基本块列表
  SmallVector<cfg::BasicBlock *>
  collectCondBrBlocks(cfg::ControlFlowGraph &cfg);

  // 6. 获取 COND_BR 对应的条件分支信息
  // 参数: COND_BR 类型的基本块
  // 返回: CondBranchMapping 结构体，包含条件、目标块和参数信息
  std::optional<CondBranchMapping>
  getCondBranchMapping(cfg::BasicBlock *condBrBB);

  // 7. 快速收集所有的 BR 基本块
  // 遍历 CFG 中所有基本块，返回类型为 BR 的基本块列表
  SmallVector<cfg::BasicBlock *> collectBrBlocks(cfg::ControlFlowGraph &cfg);

  // 8. 获取 BR 对应的分支信息
  // 参数: BR 类型的基本块
  // 返回: BranchMapping 结构体，包含目标块和参数信息
  std::optional<BranchMapping> getBranchMapping(cfg::BasicBlock *brBB);

  // 获取下一个指令 ID
  size_t getNextInstructionId() { return nextInstructionId++; }

  // MLIR Block 到 CFG BasicBlock 的映射（用于处理 cf.cond_br 等跳转指令）
  DenseMap<Block *, cfg::BasicBlock *> blockToBasicBlockMap;

  // 获取或创建 Block 对应的 BasicBlock
  cfg::BasicBlock *
  getOrCreateBasicBlockForBlock(Block *block, cfg::ControlFlowGraph &cfg,
                                cfg::BasicBlock *parentStructure = nullptr);

  // 注册 Block 到 BasicBlock 的映射
  void registerBlockMapping(Block *mlirBlock, cfg::BasicBlock *cfgBlock);

private:
  size_t nextInstructionId = 0; // 下一个指令 ID
};

// 创建 Pass 的工厂函数
std::unique_ptr<OperationPass<mlir::ModuleOp>> createBuildCFGPass();

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_CONTROL_FLOW_GRAPH_BUILDER_H
