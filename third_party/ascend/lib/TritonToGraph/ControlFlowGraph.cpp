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

#include "TritonToGraph/ControlFlowGraph.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// Instruction
//===----------------------------------------------------------------------===//

std::string Instruction::getAsString() const {
  if (!operation)
    return "<null operation>";
  std::string str;
  llvm::raw_string_ostream os(str);
  operation->print(os);
  return str;
}

void Instruction::print(raw_ostream &os, unsigned indent) const {
  std::string instStr = getAsString();
  os << "Inst[" << id << "]: ";

  if (operation) {
    BlockType parentType = parentBlock->getType();
    if (parentType == BlockType::IF_COND || parentType == BlockType::FOR_COND ||
        parentType == BlockType::WHILE_COND ||
        parentType == BlockType::COND_BR || parentType == BlockType::BR) {
      size_t newlinePos = instStr.find('\n');
      if (newlinePos != std::string::npos) {
        instStr = instStr.substr(0, newlinePos) + " ...";
      }
    }
    os.indent(indent) << instStr;
  } else {
    os << "<null>\n";
  }
}

void Instruction::dump() const { print(llvm::errs()); }

//===----------------------------------------------------------------------===//
// BasicBlock
//===----------------------------------------------------------------------===//

void BasicBlock::addInstruction(std::unique_ptr<Instruction> inst) {
  instructions.push_back(std::move(inst));
}

Instruction *BasicBlock::getInstruction(size_t idx) const {
  if (idx < instructions.size()) {
    return instructions[idx].get();
  }
  return nullptr;
}

void BasicBlock::addSuccessor(BasicBlock *succ) {
  if (!succ)
    return;
  // 避免重复添加
  for (auto *s : successors) {
    if (s == succ)
      return;
  }
  successors.push_back(succ);
  succ->addPredecessor(this);
}

void BasicBlock::addPredecessor(BasicBlock *pred) {
  if (!pred)
    return;
  // 避免重复添加
  for (auto *p : predecessors) {
    if (p == pred)
      return;
  }
  predecessors.push_back(pred);
}

std::string BasicBlock::getName() const {
  std::string name = "BB";
  name += std::to_string(id);
  return name;
}

bool BasicBlock::endsWithReturnOp() const {
  // 空块检查
  if (instructions.empty()) {
    return false;
  }

  // 获取最后一条指令
  const Instruction *lastInst = instructions.back().get();
  if (!lastInst) {
    return false;
  }

  // 获取对应的 Operation
  Operation *op = lastInst->getOperation();
  if (!op) {
    return false;
  }

  // 检查是否为 triton::ReturnOp
  return isa<triton::ReturnOp>(op);
}

StringRef BasicBlock::getTypeString() const {
  switch (type) {
  case BlockType::NORMAL:
    return "NORMAL";
  case BlockType::ENTRY:
    return "ENTRY";
  case BlockType::EXIT:
    return "EXIT";
  case BlockType::IF_COND:
    return "IF_COND";
  case BlockType::FOR_COND:
    return "FOR_COND";
  case BlockType::WHILE_COND:
    return "WHILE_COND";
  case BlockType::COND_BR:
    return "COND_BR";
  case BlockType::BR:
    return "BR";
  case BlockType::LOOP_BODY:
    return "LOOP_BODY";
  case BlockType::LOOP_EXIT:
    return "LOOP_EXIT";
  }
  return "UNKNOWN";
}

void BasicBlock::print(raw_ostream &os) const {
  os << "============================================================\n";
  os << "BasicBlock " << getName() << " [" << getTypeString() << "]\n";
  if (parentStructure) {
    os << "  Parent Structure: " << parentStructure->getName() << "\n";
  }

  // 打印前驱
  if (!predecessors.empty()) {
    os << "  Predecessors: [";
    for (size_t i = 0; i < predecessors.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << predecessors[i]->getName();
    }
    os << "]\n";
  }

  // 打印后继
  if (!successors.empty()) {
    os << "  Successors: [";
    for (size_t i = 0; i < successors.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << successors[i]->getName();
    }
    os << "]\n";
  }

  // 打印指令 - COND节点只打印第一行
  os << "  Instructions (" << instructions.size() << "):\n";
  for (const auto &inst : instructions) {
    std::string instStr = inst->getAsString();

    // 对COND节点截断：只保留第一行
    if (type == BlockType::IF_COND || type == BlockType::FOR_COND ||
        type == BlockType::WHILE_COND) {
      size_t newlinePos = instStr.find('\n');
      if (newlinePos != std::string::npos) {
        instStr = instStr.substr(0, newlinePos) + " ...";
      }
    }

    os.indent(4) << "Inst[" << inst->getId() << "]: " << instStr << "\n";
  }
  os << "\n";
}

void BasicBlock::dump() const { print(llvm::errs()); }

void BasicBlock::exportToJSON(raw_ostream &os, unsigned indent) const {
  std::string ind(indent, ' ');
  os << ind << "{\n";
  os << ind << "  \"id\": " << id << ",\n";
  os << ind << "  \"name\": \"" << getName() << "\",\n";
  os << ind << "  \"type\": \"" << getTypeString() << "\",\n";
  os << ind << "  \"parentStructure\": "
     << (parentStructure ? std::to_string(parentStructure->getId()) : "null")
     << ",\n";

  // 前驱
  os << ind << "  \"predecessors\": [";
  for (size_t i = 0; i < predecessors.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << predecessors[i]->getId();
  }
  os << "],\n";

  // 后继
  os << ind << "  \"successors\": [";
  for (size_t i = 0; i < successors.size(); ++i) {
    if (i > 0)
      os << ", ";
    os << successors[i]->getId();
  }
  os << "],\n";

  // 指令
  os << ind << "  \"instructions\": [\n";
  for (size_t i = 0; i < instructions.size(); ++i) {
    os << ind << "    {\n";
    os << ind << "      \"id\": " << instructions[i]->getId() << ",\n";
    // 转义字符串用于 JSON
    std::string instStr = instructions[i]->getAsString();

    // COND节点只取第一行
    if (type == BlockType::IF_COND || type == BlockType::FOR_COND ||
        type == BlockType::WHILE_COND) {
      size_t newlinePos = instStr.find('\n');
      if (newlinePos != std::string::npos) {
        instStr = instStr.substr(0, newlinePos) + " ...";
      }
    }

    // 简单的 JSON 字符串转义
    std::string escaped;
    for (char c : instStr) {
      if (c == '"')
        escaped += "\\\"";
      else if (c == '\\')
        escaped += "\\\\";
      else if (c == '\n')
        escaped += "\\n";
      else if (c == '\r')
        escaped += "\\r";
      else if (c == '\t')
        escaped += "\\t";
      else if ((unsigned char)c < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", c);
        escaped += buf;
      } else {
        escaped += c;
      }
    }
    os << ind << "      \"operation\": \"" << escaped << "\"\n";
    os << ind << "    }";
    if (i < instructions.size() - 1)
      os << ",";
    os << "\n";
  }
  os << ind << "  ]\n";
  os << ind << "}";
}

//===----------------------------------------------------------------------===//
// ControlFlowGraph
//===----------------------------------------------------------------------===//

ControlFlowGraph::ControlFlowGraph(triton::FuncOp func) : function(func) {}

ControlFlowGraph::~ControlFlowGraph() = default;

BasicBlock *ControlFlowGraph::createBasicBlock(BlockType type,
                                               BasicBlock *parentStructure) {
  auto bb = std::make_unique<BasicBlock>(nextBlockId++, type, parentStructure);
  BasicBlock *bbPtr = bb.get();
  basicBlocks.push_back(std::move(bb));

  // 自动设置入口/出口块
  if (type == BlockType::ENTRY)
    entryBlock = bbPtr;
  if (type == BlockType::EXIT)
    exitBlock = bbPtr;

  return bbPtr;
}

void ControlFlowGraph::addEdge(BasicBlock *from, BasicBlock *to) {
  if (!from || !to)
    return;
  from->addSuccessor(to);
}

// 判断是否为回边（从循环体回到循环头）
bool ControlFlowGraph::isBackEdge(BasicBlock *from, BasicBlock *to) const {
  // 回边定义：指向 FOR_COND/WHILE_COND 且 from 是该循环的后代
  if (to->getType() != BlockType::FOR_COND &&
      to->getType() != BlockType::WHILE_COND)
    return false;

  // 检查 from 是否属于以 to 为头的循环
  // 方法：检查 from 的 parentStructure 链是否包含 to
  BasicBlock *current = from;
  while (current) {
    if (current == to)
      return true;
    current = current->getParentStructure();
  }
  return false;
}

void ControlFlowGraph::searchNormalBlock(BasicBlock *block,
                                         OperationVisitor callback) const {
  for (const auto &inst : block->getInstructions()) {
    if (Operation *op = inst->getOperation()) {
      callback(op);
    }
  }
}

void ControlFlowGraph::searchCondBlock(BasicBlock *block,
                                       OperationVisitor callback) const {
  if (!block)
    return;

  // 获取该 Cond block 对应的 exit block（停止条件）
  BasicBlock *exitBlock = block->getExitBlock();

  // 遍历 Cond block 的每一个 successor
  for (BasicBlock *succ : block->getSuccessors()) {
    if (!succ)
      continue;

    // 从每个 successor 开始，沿着 block 链条向下遍历
    // 使用队列进行 BFS 遍历该分支
    std::deque<BasicBlock *> workList;
    workList.push_back(succ);

    while (!workList.empty()) {
      BasicBlock *current = workList.front();
      workList.pop_front();

      // 如果碰到 exit block，停止该分支的遍历
      if (current == exitBlock)
        break;

      // 对当前 block 调用 searchBlock
      searchBlock(current, callback);

      if (current->getType() == BlockType::NORMAL) {
        // 将后继加入队列（排除回边）
        for (BasicBlock *next : current->getSuccessors()) {
          // 跳过回边避免无限循环
          if (isBackEdge(current, next))
            continue;
          workList.push_back(next);
        }
      } else if (block->getType() == BlockType::IF_COND ||
                 block->getType() == BlockType::FOR_COND ||
                 block->getType() == BlockType::WHILE_COND) {
        BasicBlock *next = block->getExitBlock();
        if (isBackEdge(current, next))
          continue;
        workList.push_back(next);
      }
    }
  }
}

void ControlFlowGraph::searchBlock(BasicBlock *block,
                                   OperationVisitor callback) const {
  if (block->getType() == BlockType::NORMAL) {
    searchNormalBlock(block, callback);
  } else if (block->getType() == BlockType::IF_COND ||
             block->getType() == BlockType::FOR_COND ||
             block->getType() == BlockType::WHILE_COND) {
    searchCondBlock(block, callback);
  }
}

void ControlFlowGraph::traverse(BlockVisitor visitor) {
  // 从入口块开始进行拓扑排序
  std::vector<BasicBlock *> topoOrder;
  DenseSet<BasicBlock *> visited;

  // 使用DFS进行拓扑排序
  std::function<void(BasicBlock *)> dfs = [&](BasicBlock *bb) {
    if (!bb || visited.contains(bb))
      return;

    visited.insert(bb);

    // 递归访问所有后继块
    for (BasicBlock *succ : bb->getSuccessors()) {
      dfs(succ);
    }

    // 在后序位置添加，最后需要反转
    topoOrder.push_back(bb);
  };

  // 从入口块开始DFS
  dfs(entryBlock);

  // 反转得到拓扑序
  std::reverse(topoOrder.begin(), topoOrder.end());

  // 按照拓扑序遍历
  for (BasicBlock *bb : topoOrder) {
    visitor(*bb);
  }
}

void ControlFlowGraph::print(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp &>(function).getName();
  os << "=================================================================\n";
  os << "Control Flow Graph for function '"
     << (funcName.empty() ? "unnamed" : funcName) << "'\n";
  os << "Number of blocks: " << basicBlocks.size() << "\n";
  os << "=================================================================\n\n";

  for (const auto &bb : basicBlocks) {
    bb->print(os);
  }
}

void ControlFlowGraph::dump() const { print(llvm::errs()); }

void ControlFlowGraph::exportDOT(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp &>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  // 清理函数名用于 DOT 标识符
  std::string cleanFuncName;
  for (char c : funcNameStr) {
    if (isalnum(c) || c == '_')
      cleanFuncName += c;
    else
      cleanFuncName += '_';
  }

  os << "digraph CFG_" << cleanFuncName << " {\n";
  os << "  label=\"CFG for " << funcNameStr << "\";\n";
  os << "  labelloc=t;\n";
  os << "  rankdir=TB;\n";
  os << "  splines=true;\n";  // 使用曲线边
  os << "  overlap=false;\n"; // 防止节点重叠
  os << "  nodesep=0.6;\n";   // 节点水平间距
  os << "  ranksep=1.2;\n";   // 层间距
  os << "  fontsize=12;\n\n";

  // 设置节点样式
  for (const auto &bb : basicBlocks) {
    os << "  \"" << bb->getName() << "\" [";
    os << "label=\"" << bb->getName() << "\\n(" << bb->getTypeString() << ")";

    // 指令摘要：COND节点只取第一行，其他节点最多3条
    size_t numInsts = bb->getNumInstructions();
    if (numInsts > 0) {
      os << "\\n";

      // COND节点只显示第一条指令的第一行
      if (bb->getType() == BlockType::IF_COND ||
          bb->getType() == BlockType::FOR_COND ||
          bb->getType() == BlockType::WHILE_COND) {
        std::string instStr = bb->getInstruction(0)->getAsString();
        size_t newlinePos = instStr.find('\n');
        if (newlinePos != std::string::npos) {
          instStr = instStr.substr(0, newlinePos);
        }
        if (instStr.length() > 40)
          instStr = instStr.substr(0, 37) + "...";

        // 转义
        std::string escaped;
        for (char c : instStr) {
          if (c == '"')
            escaped += "\\\"";
          else if (c == '\\')
            escaped += "\\\\";
          else if (c == '\n')
            escaped += "\\n";
          else
            escaped += c;
        }
        os << escaped;
        if (numInsts > 1)
          os << "\\n... (" << (numInsts - 1) << " more)";
      } else {
        // 其他节点显示最多3条
        for (size_t i = 0; i < std::min(numInsts, (size_t)3); ++i) {
          std::string instStr = bb->getInstruction(i)->getAsString();
          // 只取第一行
          size_t newlinePos = instStr.find('\n');
          if (newlinePos != std::string::npos) {
            instStr = instStr.substr(0, newlinePos) + "...";
          }
          if (instStr.length() > 40)
            instStr = instStr.substr(0, 37) + "...";

          std::string escaped;
          for (char c : instStr) {
            if (c == '"')
              escaped += "\\\"";
            else if (c == '\\')
              escaped += "\\\\";
            else if (c == '\n')
              escaped += "\\n";
            else
              escaped += c;
          }
          os << escaped << "\\n";
        }
        if (numInsts > 3)
          os << "... (" << (numInsts - 3) << " more)\\n";
      }
    }

    os << "\", ";

    // 形状和颜色
    switch (bb->getType()) {
    case BlockType::ENTRY:
      os << "style=filled, fillcolor=lightgreen, shape=ellipse";
      break;
    case BlockType::EXIT:
      os << "style=filled, fillcolor=lightcoral, shape=ellipse";
      break;
    case BlockType::IF_COND:
      os << "style=filled, fillcolor=lightyellow, shape=diamond";
      break;
    case BlockType::FOR_COND:
    case BlockType::WHILE_COND:
      os << "style=filled, fillcolor=lightblue, shape=diamond";
      break;
    case BlockType::LOOP_BODY:
      os << "style=filled, fillcolor=lightcyan, shape=box";
      break;
    case BlockType::LOOP_EXIT:
      os << "style=filled, fillcolor=lightpink, shape=box";
      break;
    default:
      os << "shape=box";
      break;
    }
    os << ", fontsize=10];\n";
  }

  os << "\n";

  // 输出边：回边红色，其他绿色
  for (const auto &bb : basicBlocks) {
    for (auto *succ : bb->getSuccessors()) {
      bool isBack = isBackEdge(bb.get(), succ);
      os << "  \"" << bb->getName() << "\" -> \"" << succ->getName() << "\"";
      os << " [color=" << (isBack ? "red" : "green");
      os << ", penwidth=" << (isBack ? "2.5" : "1.5") << "";
      if (isBack) {
        os << ", style=dashed"; // 回边用虚线
      }
      os << "];\n";
    }
  }

  os << "}\n";
}

llvm::Error ControlFlowGraph::exportToFile(StringRef filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  // 判断文件扩展名
  if (filename.ends_with(".dot")) {
    exportDOT(os);
  } else if (filename.ends_with(".json")) {
    exportToJSON(os);
  } else {
    print(os);
  }

  os.close();
  return llvm::Error::success();
}

void ControlFlowGraph::exportToJSON(raw_ostream &os) const {
  auto funcName = const_cast<triton::FuncOp &>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << "{\n";
  os << "  \"functionName\": \"" << funcNameStr << "\",\n";
  os << "  \"numBlocks\": " << basicBlocks.size() << ",\n";
  os << "  \"blocks\": [\n";

  for (size_t i = 0; i < basicBlocks.size(); ++i) {
    basicBlocks[i]->exportToJSON(os, 4);
    if (i < basicBlocks.size() - 1)
      os << ",";
    os << "\n";
  }

  // 添加边信息（带类型标记）
  os << "  ],\n";
  os << "  \"edges\": [\n";

  bool first = true;
  for (const auto &bb : basicBlocks) {
    for (auto *succ : bb->getSuccessors()) {
      if (!first)
        os << ",\n";
      first = false;

      bool isBack = isBackEdge(bb.get(), succ);
      os << "    {\"from\": " << bb->getId() << ", \"to\": " << succ->getId()
         << ", \"fromName\": \"" << bb->getName() << "\""
         << ", \"toName\": \"" << succ->getName() << "\""
         << ", \"type\": \"" << (isBack ? "back" : "normal") << "\"}";
    }
  }
  os << "\n  ]\n";
  os << "}\n";
}

llvm::Error ControlFlowGraph::exportToHTML(StringRef filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  auto funcName = const_cast<triton::FuncOp &>(function).getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << R"html(<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CFG for )html"
     << funcNameStr << R"html(</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
    <script src="https://unpkg.com/dagre-d3@0.6.4/dist/dagre-d3.min.js"></script>
    <style type="text/css">
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            height: 100vh;
            background: #f0f2f5;  /* 浅色背景 */
            overflow: hidden;
        }
        #svg-container {
            flex: 1;
            overflow: auto;
            background: #ffffff;  /* 纯白画布 */
            cursor: grab;
            border-right: 1px solid #ddd;
        }
        #svg-container:active {
            cursor: grabbing;
        }
        #sidebar {
            width: 520px;
            background: #ffffff;
            padding: 24px;
            overflow-y: auto;
            box-shadow: -2px 0 8px rgba(0,0,0,0.08);
            z-index: 10;
        }
        h1 {
            margin-top: 0;
            color: #1a1a1a;
            font-size: 24px;
            border-bottom: 3px solid #1976d2;
            padding-bottom: 12px;
            font-weight: 600;
        }
        h2 {
            color: #444;
            font-size: 16px;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 8px;
            margin-top: 28px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .block-info {
            background: #f5f7fa;
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 16px;
            border-left: 4px solid #1976d2;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
        }
        .instruction {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-left: 3px solid #ff9800;
            border-radius: 6px;
            padding: 14px;
            margin: 10px 0;
            font-family: 'Consolas', 'Monaco', 'SF Mono', monospace;
            font-size: 15px;           /* 大字体 */
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-all;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .instruction-id {
            color: #555;
            font-size: 13px;
            background: #eceff1;
            padding: 3px 8px;
            border-radius: 4px;
            margin-right: 8px;
            font-weight: 600;
        }
        #default-msg {
            color: #888;
            text-align: center;
            margin-top: 120px;
            font-style: italic;
            font-size: 16px;
        }
        .legend {
            position: absolute;
            top: 16px;
            left: 16px;
            background: rgba(255,255,255,0.96);
            padding: 14px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 13px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 100;
            color: #333;
        }
        .legend-item {
            margin: 8px 0;
            display: flex;
            align-items: center;
        }
        .legend-line {
            width: 30px;
            height: 3px;
            margin-right: 12px;
            border-radius: 2px;
        }

        /* SVG 样式 - 适配浅色背景 */
        .node rect {
            stroke-width: 2px;
            rx: 6;
            ry: 6;
        }
        .node text {
            font-family: 'Consolas', 'Monaco', 'SF Mono', monospace;
            font-size: 13px;          /* 节点内字体 */
            font-weight: 500;
            pointer-events: none;
            fill: #212121;
        }
        .edgePath path {
            stroke-width: 2.5px;
            stroke-linecap: square;
            fill: none;
        }
        .arrowhead {
            fill-opacity: 1;
        }
    </style>
</head>
<body>
    <div id="svg-container">
        <svg id="svg-canvas" width="2400" height="1800"></svg>
    </div>
    <div class="legend">
        <div class="legend-item"><div class="legend-line" style="background: #2e7d32;"></div>正常控制流 (向下)</div>
        <div class="legend-item"><div class="legend-line" style="background: #d32f2f; border-top: 2px dashed #d32f2f; height: 0;"></div>循环回边 (向上)</div>
    </div>
    <div id="sidebar">
        <h1>CFG: )html"
     << funcNameStr << R"html(</h1>
        <div id="default-msg">
            <p>Click on a node to view block details</p>
        </div>
        <div id="block-details" style="display:none;">
            <div class="block-info">
                <strong>Block:</strong> <span id="block-name"></span><br>
                <strong>Type:</strong> <span id="block-type"></span><br>
                <strong>ID:</strong> <span id="block-id"></span>
            </div>
            <h2>Predecessors</h2>
            <div id="preds"></div>
            <h2>Successors</h2>
            <div id="succs"></div>
            <h2>Instructions (<span id="inst-count"></span>)</h2>
            <div id="instructions"></div>
        </div>
    </div>

<script type="text/javascript">
    var cfgData = )html";

  exportToJSON(os);

  os << R"html(;

    // 使用 dagre 计算布局
    const g = new dagre.graphlib.Graph({ compound: true });

    g.setGraph({
        rankdir: 'TB',           // 自上而下
        ranksep: 120,            // 层间距
        nodesep: 100,            // 节点水平间距
        edgesep: 25,
        marginx: 60,
        marginy: 60,
        acyclicer: 'greedy',
        ranker: 'longest-path'
    });

    g.setDefaultEdgeLabel(function() { return {}; });

    // 颜色配置 - 适配浅色背景
    const colorMap = {
        'ENTRY': '#c8e6c9',      // 柔和绿
        'EXIT': '#ffcdd2',       // 柔和红
        'IF_COND': '#fff9c4',    // 柔和黄
        'FOR_COND': '#b3e5fc',   // 柔和蓝
        'WHILE_COND': '#b3e5fc', // 柔和蓝
        'LOOP_BODY': '#e1f5fe',  // 浅蓝
        'LOOP_EXIT': '#f8bbd0',   // 浅粉
        'NORMAL': '#f5f5f5'      // 浅灰
    };

    const borderMap = {
        'ENTRY': '#2e7d32',
        'EXIT': '#c62828',
        'IF_COND': '#f57c00',
        'FOR_COND': '#0288d1',
        'WHILE_COND': '#0288d1',
        'LOOP_BODY': '#0288d1',
        'LOOP_EXIT': '#c2185b',
        'NORMAL': '#616161'
    };

    const nodeWidth = 320;       // 节点加宽以容纳更多文本
    const lineHeight = 18;       // 每行高度

    // 辅助函数：截断指令文本
    function truncateLine(text, maxLen) {
        if (text.length <= maxLen) return text;
        return text.substring(0, maxLen-3) + '...';
    }

    // 添加节点
    cfgData.blocks.forEach(block => {
        const isCond = ['IF_COND', 'FOR_COND', 'WHILE_COND'].includes(block.type);
        const isEntry = block.type === 'ENTRY';
        const isExit = block.type === 'EXIT';

        // 构建多行标签
        let lines = [];
        lines.push(block.name + ' (' + block.type + ')');

        if (isCond && block.instructions.length > 0) {
            // COND节点：只显示第一行指令
            let firstLine = truncateLine(block.instructions[0].operation.split('\n')[0], 58);
            lines.push(firstLine);
            if (block.instructions.length > 1) {
                lines.push('  +' + (block.instructions.length - 1) + ' instructions');
            }
        } else if (block.instructions.length > 0) {
            // 非COND节点：最多显示3行指令
            const showCount = Math.min(block.instructions.length, 3);
            for (let i = 0; i < showCount; i++) {
                let line = truncateLine(block.instructions[i].operation.split('\n')[0], 58);
                lines.push(line);
            }
            if (block.instructions.length > 3) {
                lines.push('  +' + (block.instructions.length - 3) + ' more...');
            }
        }

        const label = lines.join('\n');
        const height = 50 + lines.length * lineHeight;  // 基础高度+行高

        const nodeConfig = {
            label: label,
            width: nodeWidth,
            height: height,
            customData: block,
            fillColor: colorMap[block.type] || '#ffffff',
            strokeColor: borderMap[block.type] || '#424242',
            isCond: isCond
        };

        // ENTRY固定最上层(rank: source)，EXIT固定最下层(rank: sink)
        if (isEntry) {
            nodeConfig.rank = 'source';
        } else if (isExit) {
            nodeConfig.rank = 'sink';
        }

        g.setNode(block.id.toString(), nodeConfig);
    });

    // 添加边
    if (cfgData.edges) {
        cfgData.edges.forEach(edge => {
            const isBack = edge.type === 'back';
            g.setEdge(edge.from.toString(), edge.to.toString(), {
                lineInterpolate: 'orthogonal',  // Manhattan routing
                isBack: isBack,
                strokeColor: isBack ? '#d32f2f' : '#2e7d32',
                strokeDasharray: isBack ? '6,4' : '0',
                strokeWidth: isBack ? 2.5 : 2
            });
        });
    }

    // 计算布局
    dagre.layout(g);

    // 使用 D3 绘制
    const svg = d3.select('#svg-canvas');
    const svgGroup = svg.append('g');

    // 缩放和平移
    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', function(event) {
            svgGroup.attr('transform', event.transform);
        });
    svg.call(zoom);

    // 绘制边 - 正交路径
    const edgeSelection = svgGroup.selectAll('.edgePath')
        .data(g.edges())
        .enter()
        .append('g')
        .attr('class', 'edgePath');

    edgeSelection.append('path')
        .attr('d', function(e) {
            const edge = g.edge(e);
            const points = edge.points;

            // Manhattan routing: 直角线段
            let d = d3.path();
            d.moveTo(points[0].x, points[0].y);

            for (let i = 1; i < points.length; i++) {
                d.lineTo(points[i].x, points[i].y);
            }
            return d.toString();
        })
        .attr('stroke', function(e) { return g.edge(e).strokeColor; })
        .attr('stroke-width', function(e) { return g.edge(e).strokeWidth; })
        .attr('stroke-dasharray', function(e) { return g.edge(e).strokeDasharray; })
        .attr('fill', 'none');

    // 箭头
    edgeSelection.append('polygon')
        .attr('class', 'arrowhead')
        .attr('points', function(e) {
            const edge = g.edge(e);
            const points = edge.points;
            const last = points[points.length - 1];
            const prev = points[points.length - 2] || points[0];

            const dx = last.x - prev.x;
            const dy = last.y - prev.y;
            const angle = Math.atan2(dy, dx);

            const size = 10;
            const angle1 = angle + Math.PI * 0.85;
            const angle2 = angle - Math.PI * 0.85;

            return [
                [last.x, last.y],
                [last.x + Math.cos(angle1) * size, last.y + Math.sin(angle1) * size],
                [last.x + Math.cos(angle2) * size, last.y + Math.sin(angle2) * size]
            ].join(' ');
        })
        .attr('fill', function(e) { return g.edge(e).strokeColor; });

    // 绘制节点
    const nodeSelection = svgGroup.selectAll('.node')
        .data(g.nodes())
        .enter()
        .append('g')
        .attr('class', 'node')
        .attr('transform', function(v) {
            const node = g.node(v);
            return 'translate(' + (node.x - node.width/2) + ',' + (node.y - node.height/2) + ')';
        })
        .style('cursor', 'pointer')
        .on('click', function(event, v) {
            const node = g.node(v);
            showBlockDetails(node.customData);
        });

    // 节点矩形
    nodeSelection.append('rect')
        .attr('width', function(v) { return g.node(v).width; })
        .attr('height', function(v) { return g.node(v).height; })
        .attr('fill', function(v) { return g.node(v).fillColor; })
        .attr('stroke', function(v) { return g.node(v).strokeColor; })
        .attr('stroke-width', 2)
        .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

    // 节点文本 - 多行
    nodeSelection.each(function(v) {
        const node = g.node(v);
        const lines = node.label.split('\n');
        const selection = d3.select(this);

        lines.forEach((line, i) => {
            selection.append('text')
                .attr('x', 14)
                .attr('y', 22 + i * lineHeight)
                .text(line)
                .style('font-weight', i === 0 ? 'bold' : (node.isCond && i === 1 ? '600' : 'normal'))
                .style('fill', i === 0 ? '#1565c0' : '#424242')  // 第一行标题蓝色，其余深灰
                .style('font-size', i === 0 ? '14px' : '13px');
        });
    });

    // 调整画布尺寸并居中
    const graphWidth = g.graph().width + 120;
    const graphHeight = g.graph().height + 120;
    svg.attr('width', graphWidth).attr('height', graphHeight);

    const container = document.getElementById('svg-container');
    const scale = Math.min(
        container.clientWidth / graphWidth,
        container.clientHeight / graphHeight,
        1.0
    ) * 0.9;

    const transform = d3.zoomIdentity
        .translate(container.clientWidth/2 - graphWidth*scale/2, 40)
        .scale(scale);
    svg.call(zoom.transform, transform);

    // 详情面板
    function showBlockDetails(block) {
        document.getElementById('default-msg').style.display = 'none';
        document.getElementById('block-details').style.display = 'block';

        document.getElementById('block-name').textContent = block.name;
        document.getElementById('block-type').textContent = block.type;
        document.getElementById('block-id').textContent = block.id;

        let predsHtml = '';
        if (block.predecessors.length === 0) {
            predsHtml = '<em style="color:#888;">None</em>';
        } else {
            block.predecessors.forEach(predId => {
                const pred = cfgData.blocks.find(b => b.id === predId);
                const isBack = cfgData.edges.find(e => e.from === predId && e.to === block.id && e.type === 'back');
                const style = isBack ? 'color: #d32f2f; font-weight: bold;' : 'color: #333;';
                predsHtml += `<div style="${style} margin: 5px 0; font-size: 14px;">← ${pred ? pred.name : 'BB'+predId}${isBack ? ' [回边]' : ''}</div>`;
            });
        }
        document.getElementById('preds').innerHTML = predsHtml;

        let succsHtml = '';
        if (block.successors.length === 0) {
            succsHtml = '<em style="color:#888;">None</em>';
        } else {
            block.successors.forEach(succId => {
                const succ = cfgData.blocks.find(b => b.id === succId);
                const isBack = cfgData.edges.find(e => e.from === block.id && e.to === succId && e.type === 'back');
                const style = isBack ? 'color: #d32f2f; font-weight: bold;' : 'color: #333;';
                succsHtml += `<div style="${style} margin: 5px 0; font-size: 14px;">→ ${succ ? succ.name : 'BB'+succId}${isBack ? ' [回边]' : ''}</div>`;
            });
        }
        document.getElementById('succs').innerHTML = succsHtml;

        document.getElementById('inst-count').textContent = block.instructions.length;
        let instHtml = '';
        block.instructions.forEach(inst => {
            instHtml += `<div class="instruction"><span class="instruction-id">[${inst.id}]</span>${escapeHtml(inst.operation)}</div>`;
        });
        document.getElementById('instructions').innerHTML = instHtml;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
</script>
</body>
</html>)html";

  os.close();
  return llvm::Error::success();
}
