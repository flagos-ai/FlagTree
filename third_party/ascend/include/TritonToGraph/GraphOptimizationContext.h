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

#ifndef TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_CONTEXT_H
#define TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_CONTEXT_H

#include "TritonToGraph/ResourceCostModel.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cassert>
#include <cstdint>
#include <memory>

namespace mlir {
namespace triton {
namespace cfg {

class AliasAnalysis;
class ControlFlowGraph;
class DataFlowGraph;
class EntryArgPointerAliasAnalysis;
class ProgramAxisDependenceAnalysis;

enum class AnalysisRequirement : uint8_t {
  None = 0,
  CFG = 1u << 0,
  ControlFlowGraph = CFG,
  Alias = 1u << 1,
  AliasAnalysis = Alias,
  DataFlow = 1u << 2,
  DataFlowGraph = DataFlow,
  MemorySSA = 1u << 3,
  EntryArgPointerAlias = 1u << 4,
  ProgramAxisDependence = 1u << 5,
  ResourceCost = 1u << 6,
};

constexpr AnalysisRequirement operator|(AnalysisRequirement lhs,
                                        AnalysisRequirement rhs) {
  return static_cast<AnalysisRequirement>(static_cast<unsigned>(lhs) |
                                          static_cast<unsigned>(rhs));
}

constexpr AnalysisRequirement operator&(AnalysisRequirement lhs,
                                        AnalysisRequirement rhs) {
  return static_cast<AnalysisRequirement>(static_cast<unsigned>(lhs) &
                                          static_cast<unsigned>(rhs));
}

constexpr bool hasAnalysisRequirement(AnalysisRequirement requirements,
                                      AnalysisRequirement requirement) {
  return static_cast<unsigned>(requirements & requirement) != 0;
}

// Owns all graph analyses for one immutable function epoch. Any rewrite must
// call invalidate() before requesting analysis results again.
class GraphOptimizationContext {
public:
  explicit GraphOptimizationContext(triton::FuncOp function,
                                    ResourceSnapshot resources = {});
  ~GraphOptimizationContext();

  GraphOptimizationContext(const GraphOptimizationContext &) = delete;
  GraphOptimizationContext &
  operator=(const GraphOptimizationContext &) = delete;

  // Lazily constructs the requested analysis and all of its dependencies.
  LogicalResult ensure(AnalysisRequirement requirements);

  // Drops every analysis that may retain pointers into the old IR epoch.
  void invalidate();

  unsigned getEpoch() const { return epoch; }
  triton::FuncOp getFunction() const {
    assert(function.operator->() &&
           "GraphOptimizationContext requires a valid tt.func");
    return function;
  }

  ControlFlowGraph &getCFG() {
    assert(controlFlowGraph && "call ensure() before accessing the CFG");
    return *controlFlowGraph;
  }

  ControlFlowGraph &getControlFlowGraph() { return getCFG(); }

  AliasAnalysis &getAliasAnalysis() {
    assert(aliasAnalysis && "call ensure() before accessing alias analysis");
    return *aliasAnalysis;
  }

  EntryArgPointerAliasAnalysis &getEntryArgPointerAliasAnalysis() {
    assert(entryArgPointerAliasAnalysis &&
           "call ensure() before accessing entry-argument pointer alias "
           "analysis");
    return *entryArgPointerAliasAnalysis;
  }

  DataFlowGraph &getDataFlowGraph() {
    assert(dataFlowGraph &&
           "call ensure() before accessing the data-flow graph");
    return *dataFlowGraph;
  }

  ProgramAxisDependenceAnalysis &getProgramAxisDependenceAnalysis() {
    assert(programAxisDependenceAnalysis &&
           "call ensure() before accessing program-axis dependence analysis");
    return *programAxisDependenceAnalysis;
  }

  ResourceCostAnalysis &getResourceCostAnalysis() {
    assert(resourceCostAnalysis &&
           "call ensure() before accessing resource/cost analysis");
    return *resourceCostAnalysis;
  }

private:
  LogicalResult ensureControlFlowGraph();
  LogicalResult ensureAliasAnalysis();
  LogicalResult ensureEntryArgPointerAliasAnalysis();
  LogicalResult ensureDataFlowGraph();
  LogicalResult ensureProgramAxisDependenceAnalysis();
  LogicalResult ensureResourceCostAnalysis();

  triton::FuncOp function;
  ResourceSnapshot resources;
  unsigned epoch = 0;

  std::unique_ptr<ControlFlowGraph> controlFlowGraph;
  std::unique_ptr<AliasAnalysis> aliasAnalysis;
  std::unique_ptr<EntryArgPointerAliasAnalysis> entryArgPointerAliasAnalysis;
  std::unique_ptr<DataFlowGraph> dataFlowGraph;
  std::unique_ptr<ProgramAxisDependenceAnalysis> programAxisDependenceAnalysis;
  std::unique_ptr<ResourceCostAnalysis> resourceCostAnalysis;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_GRAPH_OPTIMIZATION_CONTEXT_H
