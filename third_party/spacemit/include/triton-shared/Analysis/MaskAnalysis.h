//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANALYSIS_MASKANALYSIS_H
#define TRITON_ANALYSIS_MASKANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/LogicalResult.h"

#include <utility>

namespace mlir {

class OpBuilder;

namespace triton {
// Data structure used to decode the pattern in a mask used for load and store.
// start and end field represent the start and end index of a range (produced
// by make_range, addi, etc.). While multi-dimensional data is possible, we
// assume range comparison can only be done on 1 dimension at a time (and
// results of range comparions across dimensions can be combined), hence start
// and end are not vectors. dims represents the real access size for ld/st
// (instead of the tensor/memref size specified by the IR). scalar is a shortcut
// used when the entire state contains a single scalar value.
//
// The general lifetime of this data structure is roughly:
// 1. A range is created by make_range and optionally operated on by addi w/
// result of splat, expand_dims, etc. During this phase, either (1) both start
// and end are populated, or (2) scalar is populated. Only one of the dimensions
// (that contains the range) can have dim > 1.
// 2. Result from step 1 is compared with a another MaskState that represents a
// scalar value. The resulting state only has dims populated.
// 3. Optionally, result from step 2 can be broadcasted and anded with other
// results from step 2. The resulting state only has dims populated.
//
// Example of creating 2D mask:
//  mask = (rows[:, None] < M) & (cols[None, :] < N)
struct MaskState {
  OpFoldResult start;
  OpFoldResult end;
  SmallVector<OpFoldResult> dims;
  OpFoldResult scalar;
  const bool useUnsafeMask;

  void dump() const;

  MaskState(bool useUnsafeMask = false) : useUnsafeMask(useUnsafeMask) {}

  int64_t getRank() const { return dims.size(); }

  bool isEmpty() const { return getRank() == 0 && !scalar && !start && !end; }

  bool isMask() const { return !start && !end && !scalar && dims.size() != 0; }

  // Recursively parse a Value; call the coresponding function based on the
  // defining operation and Value type
  LogicalResult parse(Value operand, const Location loc, OpBuilder &builder);

  tensor::ExtractSliceOp getExtractSlice(Value source, const Location loc,
                                         OpBuilder &builder) const;

  memref::SubViewOp getSubview(Value source, const Location loc,
                               OpBuilder &builder) const;

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getSideBySideSubviews(Value block1, Value block2, const Location loc,
                        OpBuilder &builder) const;

  std::pair<memref::SubViewOp, memref::SubViewOp>
  getStackedSubviews(Value block1, Value block2, const Location loc,
                     OpBuilder &builder) const;

private:
  // -------
  // Utility functions to operate on MaskState
  // -------
  LogicalResult addStateScalar(const MaskState &state,
                               const OpFoldResult scalar, Location loc,
                               OpBuilder &builder);

  LogicalResult addStates(const MaskState &lhsState, const MaskState &rhsState,
                          Location loc, OpBuilder &builder);

  LogicalResult minStateScalar(const MaskState &lhsState,
                               const MaskState &rhsState, Location loc,
                               OpBuilder &builder);

  LogicalResult minStates(const MaskState &lhsState, const MaskState &rhsState,
                          Location loc, OpBuilder &builder);
  // -------
  // Helper functions to parse values to populate MaskState
  // -------

  LogicalResult parseExtSI(arith::ExtSIOp op, const Location loc,
                           OpBuilder &builder);

  // Operand is the result of a constant
  // Get the value of the constant and assign it to scalar.
  LogicalResult parseConstant(arith::ConstantOp constOp, const Location loc,
                              OpBuilder &builder);

  // Operand is an integer scalar
  LogicalResult parseIntScalar(Value scalar, const Location loc,
                               OpBuilder &builder);

  // Operand is the result of addi
  // One and only one of the operands should be a scalar. Increment both start
  // and end, dims remains unchanged, and scalar is empty.
  LogicalResult parseAdd(arith::AddIOp addOp, const Location loc,
                         OpBuilder &builder);
  // Operand is the result of andi
  // Each of the result state dims is smaller of the two operands' dims.
  // Insert instruction if needed to get new dims.
  LogicalResult parseAnd(arith::AndIOp andOp, const Location loc,
                         OpBuilder &builder);

  // Operand is the result of cmpi
  // Assume only one of the dimensions has size > 1. Only support slt/ult, and
  // sge against 0 for now. For that dimension, we have three cases:
  //  1. Constant comparison with both left and right-hand sides being scalars.
  //     Calculate this new dim as a compare and select.
  //      I.e. dim = lhs < rhs ? end : 0
  //  2. Left-hand side is not a scalar, and the right-hand side is.
  //      2.a. Predicate is slt/ult. Calculate this new dim as:
  //            dim = max(min(end, value), start) - start
  //      2.b. Predicate is sge against 0. Mask analysis already has an
  //            assumption that the mask starts at 0, so evaluate this to true
  //            and calculate this new dim as: dim = end
  LogicalResult parseCmp(arith::CmpIOp cmpOp, const Location loc,
                         OpBuilder &builder);
  // Operand is the result of make_range
  // Set start and end accordingly; step size must be 1.
  LogicalResult parseMakeRange(triton::MakeRangeOp rangeOp, const Location loc,
                               OpBuilder &builder);
  // Operand is the result of broadcast
  // Change dims only; assume only applies to tensors.
  LogicalResult parseBroadcast(triton::BroadcastOp broadcastOp,
                               const Location loc, OpBuilder &builder);
  // Operand is the result of splat
  // Assume only applies to scalar. start and end are left empty; scalar will
  // be assigned, and dims will be updated.
  LogicalResult parseSplat(triton::SplatOp splatOp, const Location loc,
                           OpBuilder &builder);
  // Operand is the result of expand_dims
  // Insert additional dims; start and end do not change and correspond to the
  // dimension that contains the range.
  LogicalResult parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                const Location loc, OpBuilder &builder);

  LogicalResult parseLoopIterArg(Value v, const Location loc,
                                 OpBuilder &builder);
};

} // namespace triton

} // namespace mlir

#endif
