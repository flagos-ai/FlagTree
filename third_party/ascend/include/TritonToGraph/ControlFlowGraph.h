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

#ifndef TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H
#define TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H

#include "TritonToGraph/MemorySSA.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlir {
namespace triton {
namespace cfg {

// 前向声明
class BasicBlock;
class ControlFlowGraph;

// Instruction 结构体：表示一个指令
class Instruction {
public:
  Instruction(size_t id, Operation *op, BasicBlock *parentBlock)
      : id(id), operation(op), parentBlock(parentBlock), memorySSAInfo() {}

  // 获取基本信息
  size_t getId() const { return id; }
  Operation *getOperation() const { return operation; }
  BasicBlock *getParentBlock() const { return parentBlock; }

  // 检查是否有子图（内部区域）
  bool hasSubGraph() const { return subGraph != nullptr; }
  ControlFlowGraph *getSubGraph() const { return subGraph.get(); }
  void setSubGraph(std::unique_ptr<ControlFlowGraph> graph) {
    subGraph = std::move(graph);
  }

  // 获取指令的字符串表示
  std::string getAsString() const;

  // 打印
  void print(raw_ostream &os, unsigned indent = 0) const;
  void dump() const;

  // Memory SSA信息
  MemorySSAInfo &getMemorySSAInfo() { return memorySSAInfo; }
  const MemorySSAInfo &getMemorySSAInfo() const { return memorySSAInfo; }

private:
  size_t id;               // 唯一ID
  Operation *operation;    // 对应的 MLIR Operation
  BasicBlock *parentBlock; // 所属的 BasicBlock
  std::unique_ptr<ControlFlowGraph>
      subGraph;                // 子图（用于 reduce 等有内部区域的操作）
  MemorySSAInfo memorySSAInfo; // Memory SSA信息（用于tensor/pointer分析）
};

// 基本块类型
enum class BlockType {
  NORMAL,     // 普通块（包含多个指令）
  ENTRY,      // 函数入口块
  EXIT,       // 函数出口块
  IF_COND,    // if 条件判断块（包含单个 scf.if 指令）
  FOR_COND,   // for 循环头块（包含单个 scf.for 指令）
  WHILE_COND, // while 条件块（包含单个 scf.while 指令）
  COND_BR,    // cf.cond_br 条件分支块（包含单个 cf.cond_br 指令）
  BR,         // cf.br 无条件跳转块（包含单个 cf.br 指令）
  LOOP_BODY,  // 循环体块
  LOOP_EXIT,  // 循环出口块
};

// 基本块节点
class BasicBlock {
public:
  BasicBlock(size_t id, BlockType type, BasicBlock *parentStructure = nullptr)
      : id(id), type(type), parentStructure(parentStructure) {}

  // 获取基本信息
  size_t getId() const { return id; }
  BlockType getType() const { return type; }
  void setType(BlockType t) { type = t; }

  // 外层结构（如果是嵌套在 loop/if 中）
  BasicBlock *getParentStructure() const { return parentStructure; }
  void setParentStructure(BasicBlock *parent) { parentStructure = parent; }

  // 对应控制流结构的出口块（用于 IF_COND/FOR_COND/WHILE_COND）
  // 指向 if/for/while 结束后到达的基本块
  BasicBlock *getExitBlock() const { return exitBlock; }
  void setExitBlock(BasicBlock *exit) { exitBlock = exit; }

  // Instruction 操作
  void addInstruction(std::unique_ptr<Instruction> inst);
  Instruction *getInstruction(size_t idx) const;
  size_t getNumInstructions() const { return instructions.size(); }
  const SmallVector<std::unique_ptr<Instruction>> &getInstructions() const {
    return instructions;
  }

  // 检查最后一条指令是否为 ReturnOp
  bool endsWithReturnOp() const;

  // 获取后继和前驱
  ArrayRef<BasicBlock *> getSuccessors() const { return successors; }
  ArrayRef<BasicBlock *> getPredecessors() const { return predecessors; }

  // 边的操作
  void addSuccessor(BasicBlock *succ);
  void addPredecessor(BasicBlock *pred);

  size_t getNumSuccessors() const { return successors.size(); }
  size_t getNumPredecessors() const { return predecessors.size(); }

  // 获取名称
  std::string getName() const;

  // 获取类型字符串
  StringRef getTypeString() const;

  // 打印
  void print(raw_ostream &os) const;
  void dump() const;

  // 导出为 JSON（用于网页可视化）
  void exportToJSON(raw_ostream &os, unsigned indent = 0) const;

private:
  size_t id;                   // 唯一ID
  BlockType type;              // 块类型
  BasicBlock *parentStructure; // 外层结构（loop/if）的 basic block
  BasicBlock *exitBlock =
      nullptr; // 对应控制流结构的出口块（用于 IF_COND/FOR_COND/WHILE_COND）
  SmallVector<std::unique_ptr<Instruction>> instructions; // 指令列表
  SmallVector<BasicBlock *> successors;                   // 后继块指针列表
  SmallVector<BasicBlock *> predecessors;                 // 前驱块指针列表
};

// 控制流图
class ControlFlowGraph {
public:
  explicit ControlFlowGraph(triton::FuncOp func);
  ~ControlFlowGraph();

  // 获取函数
  triton::FuncOp getFunction() const { return function; }

  // 基本块操作
  BasicBlock *createBasicBlock(BlockType type,
                               BasicBlock *parentStructure = nullptr);

  BasicBlock *getBasicBlock(size_t id) {
    if (id < basicBlocks.size())
      return basicBlocks[id].get();
    return nullptr;
  }
  const BasicBlock *getBasicBlock(size_t id) const {
    if (id < basicBlocks.size())
      return basicBlocks[id].get();
    return nullptr;
  }

  size_t getNumBlocks() const { return basicBlocks.size(); }

  // 获取入口和出口块
  BasicBlock *getEntryBlock() { return entryBlock; }
  const BasicBlock *getEntryBlock() const { return entryBlock; }
  BasicBlock *getExitBlock() { return exitBlock; }
  const BasicBlock *getExitBlock() const { return exitBlock; }

  void setEntryBlock(BasicBlock *bb) { entryBlock = bb; }
  void setExitBlock(BasicBlock *bb) { exitBlock = bb; }

  // 添加边
  void addEdge(BasicBlock *from, BasicBlock *to);

  // 遍历
  using BlockVisitor = llvm::function_ref<void(BasicBlock &)>;
  void traverse(BlockVisitor visitor);

  // Operation 到 Instruction 的查询
  Instruction *getInstruction(Operation *op) const {
    auto it = opToInstructionMap.find(op);
    return (it != opToInstructionMap.end()) ? it->second : nullptr;
  }

  // 添加 Operation 到 Instruction 的映射
  void addOpToInstruction(Operation *op, Instruction *inst) {
    opToInstructionMap[op] = inst;
  }

  bool isBackEdge(BasicBlock *from, BasicBlock *to) const;

  // 结构化搜索 API
  // 从起始块开始，沿着 successor 顺序向下结构化搜索
  // NORMAL 块：遍历每个 Instruction，调用 callback 传入 Operation*
  // IF_COND/FOR_COND/WHILE_COND 块：递归访问每个 successor
  // 遇到 exitBlock 时停止
  using OperationVisitor = llvm::function_ref<void(Operation *)>;
  void searchNormalBlock(BasicBlock *block, OperationVisitor callback) const;
  void searchCondBlock(BasicBlock *block, OperationVisitor callback) const;
  void searchBlock(BasicBlock *block, OperationVisitor callback) const;

  // 打印
  void print(raw_ostream &os) const;
  void dump() const;

  // 导出为 DOT 格式
  void exportDOT(raw_ostream &os) const;

  // 导出到文件
  llvm::Error exportToFile(StringRef filename) const;

  // 导出为 HTML（网页可视化）
  llvm::Error exportToHTML(StringRef filename) const;

  // 导出为 JSON
  void exportToJSON(raw_ostream &os) const;

private:
  triton::FuncOp function;                              // 所属函数
  SmallVector<std::unique_ptr<BasicBlock>> basicBlocks; // 基本块节点
  BasicBlock *entryBlock = nullptr;                     // 入口块
  BasicBlock *exitBlock = nullptr;                      // 出口块
  size_t nextBlockId = 0;                               // 下一个块ID

  // Operation 到 Instruction 的映射（支持快速查询）
  std::unordered_map<Operation *, Instruction *> opToInstructionMap;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_CONTROL_FLOW_GRAPH_H
