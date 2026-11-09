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

#ifndef TRITON_ADAPTER_OP_CLASSIFIER_H
#define TRITON_ADAPTER_OP_CLASSIFIER_H

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

// Core type enumeration for operations
enum OpCoreType {
  OP_UNDETERMINED = 0,
  OP_CUBE_ONLY = 1,
  OP_VECTOR_ONLY = 2,
  OP_CUBE_AND_VECTOR = 3
};

// OpClassifierPass for categorizing operations as CUBE or VECTOR
class OpClassifierPass
    : public PassWrapper<OpClassifierPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpClassifierPass)

  // Constructor
  OpClassifierPass() = default;

  // Run the pass
  void runOnOperation() override;

  // Return the pass argument name
  static constexpr ::llvm::StringRef getArgumentName() {
    return "op-classifier";
  }
  ::llvm::StringRef getArgument() const override { return "op-classifier"; }
  ::llvm::StringRef getDescription() const override {
    return "Classify operations as CUBE or VECTOR for dynamic CV pipeline";
  }
  ::llvm::StringRef getName() const override { return "OpClassifierPass"; }

private:
  llvm::DenseMap<Operation *, Operation *> CloneOpMap;

  // Map from operation to its core type
  llvm::DenseMap<Operation *, OpCoreType> opCoreTypes;

  // Cache: value -> whether its defining chain reaches a VECTOR-only op.
  // Memoizes hasVectorOnlyProducer during CUBE upstream propagation.
  llvm::DenseMap<mlir::Value, bool> vectorOnlyProducerCache;

  // All operations in the module
  llvm::SmallVector<Operation *> allOps;

  // Seed operations for CUBE upstream propagation
  llvm::SmallVector<Operation *> cubeSeeds;
  // if A*B+C's C from broadcast chain, they need to keep for Normalize.
  llvm::DenseSet<Operation *> inBroadcastChain;

  std::shared_ptr<AliasAnalysis> aliasAnalysis;
  std::shared_ptr<CVPipeline::MemoryDependenceGraph> memDepGraph;

  // Mark an operation as CUBE
  void markCube(Operation *op);

  // Mark synchronization Op
  llvm::LogicalResult markSynchronizationOp();

  // Pattern matching for CUBE operations
  int patternMatchCUBE();

  // Upstream pattern matching helpers
  void matchToTensorPattern(Operation *def);
  void matchTransposePattern(Operation *def);
  void matchFillPattern(Operation *def);
  void matchEmptyPattern(Operation *def);
  void matchBroadcastPattern(Operation *def);
  Value extractMmadBiasFromPotentialUnitDimExpand(Value bias);

  // Downstream pattern matching helpers
  void matchStorePattern(Operation *user);
  void matchExtractSlicePattern(Operation *user);
  void matchMaterializePattern(Operation *user);

  // Propagate CUBE core type upstream
  int propagateCubeUpstream();

  // Propagate CUBE upstream for a specific operation
  void propagateCubeUpstreamForOp(Operation *startOp);

  // Shared skip predicates for both CUBE upstream BFS paths.
  bool shouldSkipCubeUpstream(Operation *op);

  // Helper: Handle fill op in scf.if - if all ops in scf.if are CUBE, mark
  // scf.if and propagate upstream
  void handleFillInScfIf(Operation *fillOp);

  // Get upstream operations based on both SSA and memory dependencies
  void
  getUpstreamOpsWithMemoryDeps(Operation *cur,
                               llvm::SmallVectorImpl<Operation *> &upstreamOps);

  // Step 3: Mark remaining operations as VECTOR
  int markRemainingAsVector();

  void markUpstreamsOfImplicitTranspose();

  // Step 4: Propagate VECTOR core type upstream
  int propagateVectorUpstream();

  // Step 4.5: Penetrate CUBE coloring into pure loader scf.for loops whose
  // results are consumed exclusively by CUBE ops.
  llvm::LogicalResult penetrateCubeIntoForLoops();

  // Helper: decide whether an scf.for or scf.while is a pure cube-loader loop
  bool isCubeLoaderForOp(scf::ForOp forOp);
  bool isCubeLoaderForWhileOp(scf::WhileOp whileOp);

  // Initialize the pass
  void initializePass(ModuleOp module);

  // Get the core type of an operation
  OpCoreType getCoreType(Operation *op) const;
  OpCoreType getForInitCoreType(OpOperand *operand) const;

  // Set the core type of an operation
  void setCoreType(Operation *op, OpCoreType coreType);

  // Helper: join core types into comma-separated string
  std::string joinCoreTypes(const std::vector<OpCoreType> &coreTypes);

  // Helper: check if propagation should stop at this operation for yield
  bool shouldStopPropagationForYield(Operation *op, OpCoreType targetCoreType);

  // Helper: propagate core_type upward for yield operand
  void propagateCoreTypeUpwardForYield(Operation *startOp,
                                       OpCoreType targetCoreType);

  // Step 5: Handle else region yield of scf.if by extracting core_type from
  // then region yield
  bool handleYieldFromElseRegion(std::vector<OpCoreType> &coreTypes,
                                 unsigned operandIndex,
                                 Operation *thenYieldForElse, Value &operand,
                                 Operation *elseYieldOp);

  // Step 6: Handle CUBE_AND_VECTOR operations
  int handleCubeAndVector();

  // Helper: split CUBE_AND_VECTOR operation into CUBE and VECTOR versions
  void splitOperationForCubeAndVector(
      Operation *op, llvm::DenseSet<Operation *> &processedOps,
      llvm::DenseMap<Operation *, Operation *> &opToVectorClone);

  // Step 7: Process SCF yield results
  int handleSCFYield();

  // Helper: process a single yield operation
  void processYieldOperation(Operation *op, Operation *thenYieldForElse);

  // Helper: Mark fill operations as CUBE when their output buffer is CUBE
  void markFillOpsAsCube();

  // Step 8: Stamp core type info to IR
  int stampToIR();

  // Helper: describe operation for logging
  std::string describeOp(Operation *op) const;
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass();
void registerAddOpClassifierPasses();
} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_OP_CLASSIFIER_H
