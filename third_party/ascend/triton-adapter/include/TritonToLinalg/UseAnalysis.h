//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_USEANALYSIS_H
#define TRITON_ANALYSIS_USEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

enum class UseType {
  Undefined, // Initial state
  DataUse,   // value used for tensor computation only
  MetaUse,   // value used for metadata only
  MixUse     // value used for both tensor computation and metadata
};

struct UseInfo : public dataflow::AbstractSparseLattice {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UseInfo)
  using AbstractSparseLattice::AbstractSparseLattice;

  // Lattice state transfer function
  ChangeResult meetUseType(const UseType &other) {
    if (other == UseType::Undefined) {
      return ChangeResult::NoChange;
    }

    switch (type) {
    case UseType::Undefined:
      type = other;
      return ChangeResult::Change;
    case UseType::DataUse:
    case UseType::MetaUse:
      if (type == other) {
        return ChangeResult::NoChange;
      } else {
        type = UseType::MixUse;
        return ChangeResult::Change;
      }
    case UseType::MixUse:
      return ChangeResult::NoChange;
    default:
      llvm_unreachable("bad type");
    }
  }

  ChangeResult meet(const AbstractSparseLattice &other) override {
    auto rhs = reinterpret_cast<const UseInfo *>(&other);
    return meetUseType(rhs->type);
  }

  void print(raw_ostream &os) const override {
    switch (type) {
    case UseType::DataUse:
      os << "DataUse";
      break;
    case UseType::MetaUse:
      os << "MetaUse";
      break;
    case UseType::MixUse:
      os << "MixUse";
      break;
    default:
      os << "Undefined";
    }
  }

  UseType type = UseType::Undefined;
};

class UseAnalysis : public dataflow::SparseBackwardDataFlowAnalysis<UseInfo> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

#if LLVM_VERSION_MAJOR >= 20
  LogicalResult visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                               ArrayRef<const UseInfo *> results) override;
#else
  void visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                      ArrayRef<const UseInfo *> results) override;
#endif

  void visitBranchOperand(OpOperand &operand) override { return; }

  void visitCallOperand(OpOperand &operand) override { return; }

  void setToExitState(UseInfo *lattice) override {
    lattice->type = UseType::Undefined;
  }

private:
  void propagateUse(UseInfo *lattice, const UseType &type) {
    auto changed = lattice->meetUseType(type);
    propagateIfChanged(lattice, changed);
  }

  void propagateResults(UseInfo *lattice, ArrayRef<const UseInfo *> results) {
    auto changed = ChangeResult::NoChange;
    for (auto result : results) {
      changed |= lattice->meet(*result);
    }
    propagateIfChanged(lattice, changed);
  }
};

class MetaUseEraser : public RewritePattern {
public:
  MetaUseEraser(MLIRContext *context);

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult runUseAnalysis(triton::FuncOp &funcOp);

} // namespace triton

} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOAFFINE_TRITONUSEANALYSIS_H
