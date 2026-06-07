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

#include "TritonToGraph/DataflowGraph.h"
#include "TritonToGraph/AliasAnalysis.h"
#include "TritonToGraph/ControlFlowGraph.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "dataflow-graph"

using namespace mlir;
using namespace triton;
using namespace cfg;

//===----------------------------------------------------------------------===//
// DataFlowInfo
//===----------------------------------------------------------------------===//

// Memory SSA相关接口
MemorySSADef *DataFlowInfo::getMemoryDefinition(Value value) const {
  auto it = memoryDefinitions.find(value);
  return (it != memoryDefinitions.end()) ? it->second : nullptr;
}

void DataFlowInfo::addMemoryDefinition(Value value, MemorySSADef *def) {
  memoryDefinitions[value] = def;
  // 确保没有旧的uses冲突
  memoryUses.erase(value);
}

SmallVector<MemorySSAUse> DataFlowInfo::getMemoryUses(Value value) const {
  auto it = memoryUses.find(value);
  if (it != memoryUses.end()) {
    return it->second;
  }
  return SmallVector<MemorySSAUse>();
}

void DataFlowInfo::addMemoryUse(Value value, const MemorySSAUse &use) {
  memoryUses[value].push_back(use);
}

void DataFlowInfo::removeMemoryDefinition(Value value) {
  memoryDefinitions.erase(value);
  memoryUses.erase(value);
  invalidateDefUseCache();
}

void DataFlowInfo::clearMemoryUses(Value value) {
  memoryUses[value].clear();
  invalidateDefUseCache();
}

std::unique_ptr<DataFlowResult> DataFlowInfo::queryDataFlow(Value value) const {
  // 1. 优先查询Memory SSA
  if (MemorySSADef *def = getMemoryDefinition(value)) {
    auto result = std::make_unique<MemorySSAResult>(def->getDefOp(), def);
    result->getUses() = getSSAUses(value);
    return result;
  }

  // 2. 查询传统SSA
  if (Operation *defOp = value.getDefiningOp()) {
    auto result = std::make_unique<SSAResult>(defOp, defOp);
    result->getUses() = getSSAUses(value);
    return result;
  }

  // 3. 入参
  auto result = std::make_unique<SSAResult>(nullptr, nullptr);
  result->getUses() = getSSAUses(value);
  return result;
}

SmallVector<MemorySSAUse> DataFlowInfo::getUses(MemorySSADef *def) const {
  SmallVector<MemorySSAUse> result;

  // 遍历所有uses，查找使用该definition的
  for (const auto &entry : memoryUses) {
    for (const MemorySSAUse &use : entry.second) {
      if (use.getDefinition() == def) {
        result.push_back(use);
      }
    }
  }

  return result;
}

SmallVector<MemorySSAUse>
DataFlowInfo::getUsesByUserOp(Operation *userOp) const {
  SmallVector<MemorySSAUse> result;

  for (const auto &entry : memoryUses) {
    for (const MemorySSAUse &use : entry.second) {
      if (use.getUserOp() == userOp) {
        result.push_back(use);
      }
    }
  }

  return result;
}

void DataFlowInfo::buildDefUseCache() const {
  if (defUseCacheValid)
    return;

  defUseCache.clear();

  for (const auto &entry : memoryUses) {
    for (const MemorySSAUse &use : entry.second) {
      MemorySSADef *def = use.getDefinition();
      if (def) {
        defUseCache[def].push_back(use);
      }
    }
  }

  defUseCacheValid = true;
}

void DataFlowInfo::forEachDefinition(
    llvm::function_ref<void(Value, MemorySSADef *)> func) const {
  for (const auto &entry : memoryDefinitions) {
    func(entry.first, entry.second);
  }
}

void DataFlowInfo::forEachUse(
    llvm::function_ref<void(const MemorySSAUse &)> func) const {
  for (const auto &entry : memoryUses) {
    for (const MemorySSAUse &use : entry.second) {
      func(use);
    }
  }
}

void DataFlowInfo::print(llvm::raw_ostream &os) const {
  os << "=== Data Flow Information ===" << "\n";

  os << "Memory Definitions: " << memoryDefinitions.size() << "\n";
  for (const auto &entry : memoryDefinitions) {
    os << "  " << entry.first << " -> ";
    entry.second->print(os);
    os << "\n";
  }

  os << "Memory Uses: " << "\n";
  for (const auto &entry : memoryUses) {
    os << "  " << entry.first << ": ";
    for (const MemorySSAUse &use : entry.second) {
      os << "[" << use.getDefinition()->getId() << "] ";
    }
    os << "\n";
  }

  os << "Phis: " << Phis.size() << "\n";
  for (const auto &entry : Phis) {
    os << "  " << entry.first << ": ";
    switch (entry.second.type) {
    case PhiInfo::ITER_ARG:
      os << "ITER_ARG";
      break;
    case PhiInfo::IF_RESULT:
      os << "IF_RESULT";
      break;
    case PhiInfo::WHILE_ARG:
      os << "WHILE_ARG";
      break;
    }
    os << "\n";
  }
}

void DataFlowInfo::exportToJSON(llvm::raw_ostream &os) const {
  os << "{\n";
  os << "  \"memoryDefinitions\": {\n";
  bool first = true;
  for (const auto &entry : memoryDefinitions) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    \"" << entry.first << "\": {\n";
    os << "      \"id\": \"" << entry.second->getId() << "\",\n";
    os << "      \"tensor\": \"" << entry.second->getTensor()->getName()
       << "\",\n";
    os << "      \"version\": " << entry.second->getVersion();
    os << "\n    }";
  }
  os << "\n  },\n";

  os << "  \"Phis\": {\n";
  first = true;
  for (const auto &entry : Phis) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    \"" << entry.first << "\": {\n";
    os << "      \"type\": " << entry.second.type << "\n";
    os << "    }";
  }
  os << "\n  }\n";
  os << "}\n";
}

//===----------------------------------------------------------------------===//
// DataFlowGraph
//===----------------------------------------------------------------------===//

void DataFlowGraph::build() {
  LLVM_DEBUG(llvm::dbgs() << "=== Starting Data Flow Graph Build ===" << "\n");

  // 步骤1: 构建Alias分析
  aliasAnalysis = std::make_unique<AliasAnalysis>();
  aliasAnalysis->analyzePointerAliases(cfg);
  // aliasAnalysis->print(llvm::outs());

  LLVM_DEBUG(llvm::dbgs() << "Alias analysis complete" << "\n");

  // 步骤2: 构建Memory SSA
  memorySSABuilder =
      std::make_unique<MemorySSABuilder>(cfg, *aliasAnalysis, dataFlowInfo);
  memorySSABuilder->build();

  LLVM_DEBUG(llvm::dbgs() << "Memory SSA build complete" << "\n");

  // 步骤3: 构建def-use图
  buildDefUseGraph();

  LLVM_DEBUG(llvm::dbgs() << "=== Data Flow Graph Build Complete ===" << "\n");
}

void DataFlowGraph::buildDefUseGraph() {
  // 构建def-use图（在DataFlowInfo中实现）
  dataFlowInfo.buildDefUseCache();

  LLVM_DEBUG(llvm::dbgs() << "Def-use graph built" << "\n");
}

void DataFlowGraph::print(llvm::raw_ostream &os) const {
  os << "=== Data Flow Graph ===" << "\n";
  dataFlowInfo.print(os);
}

void DataFlowGraph::dump() const { print(llvm::errs()); }

void DataFlowGraph::exportToJSON(llvm::raw_ostream &os) const {
  auto funcOp = cfg.getFunction();
  auto funcName = funcOp.getName();
  std::string funcNameStr = funcName.empty() ? "unnamed" : funcName.str();

  os << "{\n";
  os << "  \"function\": \"" << funcNameStr << "\"," << "\n";
  os << "  \"dataFlow\": ";
  dataFlowInfo.exportToJSON(os);
  os << "}\n";
}

void DataFlowGraph::exportDefUseToDOT(llvm::raw_ostream &os) const {
  os << "digraph DefUseGraph {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=box];\n\n";

  // 遍历所有definitions并导出
  size_t nodeId = 0;
  DenseMap<const MemorySSADef *, size_t> defToNode;

  cfg.traverse([&](const BasicBlock &bb) {
    for (const auto &instPtr : bb.getInstructions()) {
      const Instruction *inst = instPtr.get();
      const MemorySSAInfo &ssaInfo = inst->getMemorySSAInfo();

      for (MemorySSADef *def : ssaInfo.definitions) {
        if (def && !defToNode.count(def)) {
          defToNode[def] = nodeId++;
          os << "  node_" << defToNode[def] << " [label=\"" << def->getId()
             << "\"];\n";
        }
      }
    }
  });

  os << "}\n";
}
