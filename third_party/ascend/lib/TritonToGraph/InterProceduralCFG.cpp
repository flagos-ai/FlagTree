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

#include "TritonToGraph/InterProceduralCFG.h"
#include "TritonToGraph/ControlFlowGraphBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "icfg"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace cfg {

InterProceduralCFG::InterProceduralCFG(ModuleOp module) : module(module) {}

InterProceduralCFG::~InterProceduralCFG() = default;

void InterProceduralCFG::build() {
  LLVM_DEBUG(llvm::dbgs() << "Building ICFG for module\n");

  ControlFlowGraphBuilder builder;

  // 遍历模块中的所有函数
  for (triton::FuncOp func : module.getOps<triton::FuncOp>()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Building CFG for function: " << func.getName() << "\n");

    // 为每个函数构建CFG
    auto cfg = builder.build(func);
    functionCFGs[func] = std::move(cfg);
  }
}

ControlFlowGraph *InterProceduralCFG::getFunctionCFG(triton::FuncOp func) {
  auto it = functionCFGs.find(func);
  if (it != functionCFGs.end()) {
    return it->second.get();
  }
  return nullptr;
}

const ControlFlowGraph *
InterProceduralCFG::getFunctionCFG(triton::FuncOp func) const {
  auto it = functionCFGs.find(func);
  if (it != functionCFGs.end()) {
    return it->second.get();
  }
  return nullptr;
}

ControlFlowGraph *InterProceduralCFG::getFunctionCFG(StringRef funcName) {
  for (auto &[func, cfg] : functionCFGs) {
    if (func.getName() == funcName) {
      return cfg.get();
    }
  }
  return nullptr;
}

void InterProceduralCFG::connectCallGraph() {
  LLVM_DEBUG(llvm::dbgs() << "Connecting call graph\n");

  // 遍历所有操作，查找调用点
  module.walk([&](Operation *op) {
    // 检查是否是调用操作
    if (auto callInterface = dyn_cast<CallOpInterface>(op)) {
      // 尝试获取被调用函数
      auto callable = callInterface.getCallableForCallee();

      // 从符号引用获取函数名
      StringRef calleeName;
      if (auto symbolRef = callable.dyn_cast<SymbolRefAttr>()) {
        calleeName = symbolRef.getRootReference();
      } else if (auto flatSymbolRef =
                     op->getAttrOfType<FlatSymbolRefAttr>("callee")) {
        calleeName = flatSymbolRef.getValue();
      }

      if (!calleeName.empty()) {
        // 在模块中查找被调用函数
        if (auto calleeFunc = module.lookupSymbol<triton::FuncOp>(calleeName)) {
          // 获取调用者的函数
          Operation *parentOp = op->getParentOp();
          while (parentOp && !isa<triton::FuncOp>(parentOp)) {
            parentOp = parentOp->getParentOp();
          }

          if (auto callerFunc = dyn_cast<triton::FuncOp>(parentOp)) {
            auto callerCFG = getFunctionCFG(callerFunc);

            if (callerCFG) {
              // 找到调用点的基本块
              BasicBlock *callBlock = nullptr;
              Instruction *callInst = nullptr;

              // 遍历查找包含该操作的 basic block 和 instruction
              for (size_t i = 0; i < callerCFG->getNumBlocks(); ++i) {
                auto *bb = callerCFG->getBasicBlock(i);
                for (size_t j = 0; j < bb->getNumInstructions(); ++j) {
                  auto *inst = bb->getInstruction(j);
                  if (inst->getOperation() == op) {
                    callBlock = bb;
                    callInst = inst;
                    break;
                  }
                }
                if (callBlock)
                  break;
              }

              if (callBlock) {
                // 创建调用点记录
                CallSite callSite;
                callSite.callOp = op;
                callSite.caller = callerFunc;
                callSite.callee = calleeFunc;
                callSite.callBlock = callBlock;
                callSite.callInst = callInst;

                callSites.push_back(callSite);

                // 添加到调用图
                callGraph[callerFunc].push_back(calleeFunc);
                reverseCallGraph[calleeFunc].push_back(callerFunc);

                LLVM_DEBUG(llvm::dbgs()
                           << "Found call: " << callerFunc.getName() << " -> "
                           << calleeFunc.getName() << "\n");
              }
            }
          }
        }
      }
    }
  });
}

SmallVector<triton::FuncOp>
InterProceduralCFG::getCallees(triton::FuncOp caller) const {
  auto it = callGraph.find(caller);
  if (it != callGraph.end()) {
    return it->second;
  }
  return SmallVector<triton::FuncOp>();
}

SmallVector<triton::FuncOp>
InterProceduralCFG::getCallers(triton::FuncOp callee) const {
  auto it = reverseCallGraph.find(callee);
  if (it != reverseCallGraph.end()) {
    return it->second;
  }
  return SmallVector<triton::FuncOp>();
}

void InterProceduralCFG::computeReachability() {
  LLVM_DEBUG(llvm::dbgs() << "Computing reachability\n");

  // 使用简单的递归DFS计算可达性
  std::function<void(triton::FuncOp, DenseSet<triton::FuncOp> &)> dfs =
      [&](triton::FuncOp func, DenseSet<triton::FuncOp> &visited) {
        if (visited.contains(func)) {
          return;
        }
        visited.insert(func);

        for (auto callee : getCallees(func)) {
          reachability[func].insert(callee);
          dfs(callee, reachability[func]);
        }
      };

  // 对每个函数计算可达性
  for (auto &[func, cfg] : functionCFGs) {
    reachability[func].clear();
    dfs(func, reachability[func]);
  }
}

bool InterProceduralCFG::isReachable(triton::FuncOp from,
                                     triton::FuncOp to) const {
  auto it = reachability.find(from);
  if (it != reachability.end()) {
    return it->second.contains(to);
  }
  return false;
}

void InterProceduralCFG::dumpToDot(const std::string &filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);

  if (ec) {
    llvm::errs() << "Failed to open file: " << filename << "\n";
    return;
  }

  file << "digraph ICFG {\n";
  file << "  node [shape=box];\n\n";

  // 输出所有函数节点
  for (const auto &[func, cfg] : functionCFGs) {
    StringRef name = const_cast<triton::FuncOp &>(func).getName();
    file << "  \"" << name << "\" [label=\"" << name << "\"];\n";
  }

  file << "\n";

  // 输出调用边
  for (const auto &[caller, callees] : callGraph) {
    StringRef callerName = const_cast<triton::FuncOp &>(caller).getName();
    for (auto callee : callees) {
      StringRef calleeName = const_cast<triton::FuncOp &>(callee).getName();
      file << "  \"" << callerName << "\" -> \"" << calleeName
           << "\" [label=\"call\"];\n";
    }
  }

  file << "}\n";
}

void InterProceduralCFG::print(raw_ostream &os) const {
  os << "InterProcedural Control Flow Graph:\n";
  os << "Functions: " << functionCFGs.size() << "\n";
  os << "CallSites: " << callSites.size() << "\n";

  os << "\nCall Graph:\n";
  for (const auto &[caller, callees] : callGraph) {
    StringRef callerName = const_cast<triton::FuncOp &>(caller).getName();
    os << "  " << callerName << " -> ";
    for (auto callee : callees) {
      StringRef calleeName = const_cast<triton::FuncOp &>(callee).getName();
      os << calleeName << " ";
    }
    os << "\n";
  }
}

llvm::Error
InterProceduralCFG::exportToHTML(const std::string &filename) const {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);

  if (ec) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open file: " + filename);
  }

  os << R"html(<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Inter-Procedural CFG</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
        }
        #mynetwork {
            width: 100%;
            height: 100%;
            background: #fafafa;
        }
        h1 {
            position: absolute;
            top: 10px;
            left: 10px;
            margin: 0;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Inter-Procedural CFG</h1>
    <div id="mynetwork"></div>

<script type="text/javascript">
    var nodes = new vis.DataSet();
    var edges = new vis.DataSet();

)html";

  // 添加函数节点
  for (const auto &[func, cfg] : functionCFGs) {
    StringRef name = const_cast<triton::FuncOp &>(func).getName();
    os << "    nodes.add({id: '" << name << "', label: '" << name << "', ";
    os << "color: {background: '#90EE90', border: '#666'}});\n";
  }

  // 添加调用边
  for (const auto &[caller, callees] : callGraph) {
    StringRef callerName = const_cast<triton::FuncOp &>(caller).getName();
    for (auto callee : callees) {
      StringRef calleeName = const_cast<triton::FuncOp &>(callee).getName();
      os << "    edges.add({from: '" << callerName << "', to: '" << calleeName
         << "', ";
      os << "arrows: 'to', color: {color: '#666'}});\n";
    }
  }

  os << R"html(
    var container = document.getElementById('mynetwork');
    var data = {nodes: nodes, edges: edges};
    var options = {
        layout: {improvedLayout: true},
        physics: {stabilization: false},
        interaction: {hover: true}
    };
    var network = new vis.Network(container, data, options);
</script>
</body>
</html>)html";

  os.close();
  return llvm::Error::success();
}

} // namespace cfg
} // namespace triton
} // namespace mlir
