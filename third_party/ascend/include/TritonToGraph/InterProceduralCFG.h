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

#ifndef TRITON_TO_CFG_INTER_PROCEDURAL_CFG_H
#define TRITON_TO_CFG_INTER_PROCEDURAL_CFG_H

#include "TritonToGraph/ControlFlowGraph.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

namespace mlir {
namespace triton {
namespace cfg {

// 过程间控制流图 (ICFG)
class InterProceduralCFG {
public:
  explicit InterProceduralCFG(ModuleOp module);
  ~InterProceduralCFG();

  // 为每个函数构建CFG
  void build();

  // 获取函数的CFG
  ControlFlowGraph *getFunctionCFG(triton::FuncOp func);
  const ControlFlowGraph *getFunctionCFG(triton::FuncOp func) const;
  ControlFlowGraph *getFunctionCFG(StringRef funcName);

  // 在调用点连接调用者和被调用者的CFG
  struct CallSite {
    Operation *callOp;     // 调用操作
    triton::FuncOp caller; // 调用者函数
    triton::FuncOp callee; // 被调用者函数
    BasicBlock *callBlock; // 调用点的基本块
    Instruction *callInst; // 调用指令
  };

  // 获取所有调用点
  const SmallVector<CallSite> &getCallSites() const { return callSites; }

  // 连接ICFG中的调用边
  void connectCallGraph();

  // 查询函数调用关系
  SmallVector<triton::FuncOp> getCallees(triton::FuncOp caller) const;
  SmallVector<triton::FuncOp> getCallers(triton::FuncOp callee) const;

  // 全局可达性分析
  void computeReachability();
  bool isReachable(triton::FuncOp from, triton::FuncOp to) const;

  // 可视化
  void dumpToDot(const std::string &filename) const;
  void print(raw_ostream &os) const;

  // 导出到 HTML（包含所有函数）
  llvm::Error exportToHTML(const std::string &filename) const;

private:
  ModuleOp module;

  // 函数到CFG的映射
  DenseMap<triton::FuncOp, std::unique_ptr<ControlFlowGraph>> functionCFGs;

  // 调用点列表
  SmallVector<CallSite> callSites;

  // 调用图 (函数级别)
  DenseMap<triton::FuncOp, SmallVector<triton::FuncOp>> callGraph;
  DenseMap<triton::FuncOp, SmallVector<triton::FuncOp>> reverseCallGraph;

  // 可达性矩阵
  DenseMap<triton::FuncOp, DenseSet<triton::FuncOp>> reachability;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_INTER_PROCEDURAL_CFG_H
