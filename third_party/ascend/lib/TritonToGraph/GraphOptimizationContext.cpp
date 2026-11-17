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

#include "TritonToGraph/GraphOptimizationContext.h"

#include "TritonToGraph/AliasAnalysis.h"
#include "TritonToGraph/ControlFlowGraphBuilder.h"
#include "TritonToGraph/DataflowGraph.h"
#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"

using namespace mlir;
using namespace triton;
using namespace cfg;

GraphOptimizationContext::GraphOptimizationContext(triton::FuncOp function,
                                                   ResourceSnapshot resources)
    : function(function), resources(resources) {
  assert(this->function.getOperation() &&
         "GraphOptimizationContext requires a valid tt.func");
}

GraphOptimizationContext::~GraphOptimizationContext() = default;

LogicalResult
GraphOptimizationContext::ensure(AnalysisRequirement requirements) {
  if (hasAnalysisRequirement(requirements, AnalysisRequirement::CFG) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::Alias) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::DataFlow) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::MemorySSA) ||
      hasAnalysisRequirement(requirements,
                             AnalysisRequirement::EntryArgPointerAlias)) {
    if (failed(ensureControlFlowGraph()))
      return failure();
  }

  if (hasAnalysisRequirement(requirements,
                             AnalysisRequirement::ProgramAxisDependence)) {
    if (failed(ensureProgramAxisDependenceAnalysis()))
      return failure();
  }

  if (hasAnalysisRequirement(requirements, AnalysisRequirement::Alias) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::DataFlow) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::MemorySSA) ||
      hasAnalysisRequirement(requirements,
                             AnalysisRequirement::EntryArgPointerAlias)) {
    if (failed(ensureAliasAnalysis()))
      return failure();
  }

  if (hasAnalysisRequirement(requirements,
                             AnalysisRequirement::EntryArgPointerAlias)) {
    if (failed(ensureEntryArgPointerAliasAnalysis()))
      return failure();
  }

  if (hasAnalysisRequirement(requirements, AnalysisRequirement::DataFlow) ||
      hasAnalysisRequirement(requirements, AnalysisRequirement::MemorySSA)) {
    if (failed(ensureDataFlowGraph()))
      return failure();
  }

  if (hasAnalysisRequirement(requirements, AnalysisRequirement::ResourceCost)) {
    if (failed(ensureResourceCostAnalysis()))
      return failure();
  }

  return success();
}

void GraphOptimizationContext::invalidate() {
  resourceCostAnalysis.reset();
  programAxisDependenceAnalysis.reset();
  dataFlowGraph.reset();
  entryArgPointerAliasAnalysis.reset();
  aliasAnalysis.reset();
  controlFlowGraph.reset();
  ++epoch;
}

LogicalResult GraphOptimizationContext::ensureControlFlowGraph() {
  if (controlFlowGraph)
    return success();
  if (!function.getOperation())
    return failure();

  ControlFlowGraphBuilder builder;
  controlFlowGraph = builder.build(function);
  return controlFlowGraph ? success() : failure();
}

LogicalResult GraphOptimizationContext::ensureAliasAnalysis() {
  if (failed(ensureControlFlowGraph()))
    return failure();
  if (aliasAnalysis)
    return success();

  aliasAnalysis = std::make_unique<AliasAnalysis>();
  aliasAnalysis->analyzePointerAliases(*controlFlowGraph);
  return success();
}

LogicalResult GraphOptimizationContext::ensureEntryArgPointerAliasAnalysis() {
  if (failed(ensureAliasAnalysis()))
    return failure();
  if (entryArgPointerAliasAnalysis)
    return success();

  entryArgPointerAliasAnalysis =
      std::make_unique<EntryArgPointerAliasAnalysis>(function, *aliasAnalysis);
  return success();
}

LogicalResult GraphOptimizationContext::ensureDataFlowGraph() {
  if (failed(ensureAliasAnalysis()))
    return failure();
  if (dataFlowGraph)
    return success();

  dataFlowGraph =
      std::make_unique<DataFlowGraph>(*controlFlowGraph, *aliasAnalysis);
  dataFlowGraph->build();
  return success();
}

LogicalResult GraphOptimizationContext::ensureProgramAxisDependenceAnalysis() {
  if (programAxisDependenceAnalysis)
    return success();
  if (!function.getOperation())
    return failure();
  programAxisDependenceAnalysis =
      std::make_unique<ProgramAxisDependenceAnalysis>(function);
  return success();
}

LogicalResult GraphOptimizationContext::ensureResourceCostAnalysis() {
  if (resourceCostAnalysis)
    return success();
  if (!function.getOperation())
    return failure();

  resourceCostAnalysis = std::make_unique<ResourceCostAnalysis>(
      function.getOperation(), resources);
  return success();
}
