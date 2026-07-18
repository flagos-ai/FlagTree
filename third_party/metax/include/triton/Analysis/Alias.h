// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_ANALYSIS_ALIAS_H
#define TRITON_ANALYSIS_ALIAS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {

class AliasInfo {
public:
  AliasInfo() = default;
  AliasInfo(Value value) { insert(value); }

  void insert(Value value) { allocs.insert(value); }

  const DenseSet<Value> &getAllocs() const { return allocs; }

  bool operator==(const AliasInfo &other) const {
    return allocs == other.allocs;
  }

  /// The pessimistic value state of a value without alias
  static AliasInfo getPessimisticValueState(MLIRContext *context = nullptr) {
    return AliasInfo();
  }
  static AliasInfo getPessimisticValueState(Value value) { return AliasInfo(); }

  /// The union of both arguments
  static AliasInfo join(const AliasInfo &lhs, const AliasInfo &rhs);

  void print(raw_ostream &os) const {
    llvm::interleaveComma(allocs, os, [&](Value alloc) { alloc.print(os); });
  }

private:
  /// The set of allocated values that are aliased by this lattice.
  /// For now, we only consider aliased value produced by the following
  /// situations:
  /// 1. values returned by scf.yield
  /// 2. block arguments in scf.for
  /// Example:
  ///    alloc v1                  alloc v2
  ///       |                         |
  ///    |--------------|   |------------|
  ///  scf.for v3     scf.for v4       scf.for v5
  ///    |
  /// scf.yield v6
  ///
  /// v1's alloc [v1]
  /// v2's alloc [v2]
  /// v3's alloc [v1]
  /// v4's alloc [v1, v2]
  /// v5's alloc [v2]
  /// v6's alloc [v1]
  ///
  /// Therefore, v1's liveness range is the union of v3, v4, and v6
  /// v2's liveness range is the union of v4 and v5.
  DenseSet<Value> allocs;
};

//===----------------------------------------------------------------------===//
// Shared Memory Alias Analysis
//===----------------------------------------------------------------------===//
class SharedMemoryAliasAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<AliasInfo>> {
public:
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::SparseForwardDataFlowAnalysis;
  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<AliasInfo>>::getLatticeElement;

  /// XXX(Keren): Compatible interface with MLIR AliasAnalysis for future use.
  /// Given two values, returns their aliasing behavior.
  AliasResult alias(Value lhs, Value rhs);

  /// Returns the modify-reference behavior of `op` on `location`.
  ModRefResult getModRef(Operation *op, Value location);

  void setToEntryState(dataflow::Lattice<AliasInfo> *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(AliasInfo::getPessimisticValueState(
                           lattice->getAnchor())));
  }

  /// Computes if the alloc set of the results are changed.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
                 ArrayRef<dataflow::Lattice<AliasInfo> *> results) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<dataflow::Lattice<AliasInfo> *> argLattices,
      unsigned firstIndex) override;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_ALIAS_H
