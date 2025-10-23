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
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <utility>

namespace mlir {

// this class helps build Operations
class OpBuilder;

namespace triton {
// use to decode the pattern in a mask used for load and store

class MaskState {
public:
  OpFoldResult start;
  OpFoldResult end;
  SmallVector<OpFoldResult> dims;
  SmallVector<OpFoldResult> offsets;
  OpFoldResult scalar;

  int64_t getRank() const {
    assert(dims.size() == offsets.size() && "dims and offsets rank mismatch!");
    return dims.size();
  }

  bool isEmpty() const { return getRank() == 0 && !scalar && !start && !end; }

  bool isMask() const {
    return !start && !end && !scalar && dims.size() != 0 && offsets.size() != 0;
  }

  // parse value recursively
  LogicalResult parse(Value operand, const Location &loc, OpBuilder &builder);

  tensor::ExtractSliceOp getExtractSlice(Value source, const Location &loc,
                                         OpBuilder &builder) const;

  tensor::InsertSliceOp getInsertSlice(Value source, Value dest,
                                       const Location &loc,
                                       OpBuilder &builder) const;

  memref::SubViewOp getSubview(Value source, const Location &loc,
                               OpBuilder &builder) const;

  void eraseInsertedOps(Operation *rawOp, PatternRewriter &rewriter);

private:
  LogicalResult addStateScalar(const MaskState &state,
                               const OpFoldResult scalar, const Location &loc,
                               OpBuilder &builder);

  LogicalResult addStates(const MaskState &lhsState, const MaskState &rhsState,
                          const Location &loc, OpBuilder &builder);

  LogicalResult divStateScalar(const MaskState &state,
                               const OpFoldResult scalar, const Location &loc,
                               OpBuilder &builder);

  LogicalResult divStates(const MaskState &lhsState, const MaskState &rhsState,
                          const Location &loc, OpBuilder &builder);

  // Helper function to handle operator `and` both mask state
  LogicalResult minStates(const MaskState &lhsState, const MaskState &rhsState,
                          const Location &loc, OpBuilder &builder);

  // Helper functions to parse values to populate MaskState

  LogicalResult parseConstant(arith::ConstantOp constOp, const Location &loc,
                              OpBuilder &builder);

  // Operand is an integer scalar
  LogicalResult parseIntScalar(Value scalar, const Location &loc,
                               OpBuilder &builder);

  // TODO
  LogicalResult parseAdd(arith::AddIOp addOp, const Location &loc,
                         OpBuilder &builder);

  // operand is the result of divsi
  LogicalResult parseDiv(arith::DivSIOp divOp, const Location &loc,
                         OpBuilder &builder);

  // Operand is the result of andi
  LogicalResult parseAnd(arith::AndIOp andOp, const Location &loc,
                         OpBuilder &builder);

  // Operand is the result of cmpi, necessary method to fuse scalar, start and
  // end into dims and offset
  LogicalResult parseCmp(arith::CmpIOp cmpOp, const Location &loc,
                         OpBuilder &builder);

  // Operand is the result of make_range
  LogicalResult parseMakeRange(triton::MakeRangeOp rangeOp, const Location &loc,
                               OpBuilder &builder);

  // Operand is the result of broadcast
  LogicalResult parseBroadcast(triton::BroadcastOp broadcastOp,
                               const Location &loc, OpBuilder &builder);

  // Operand is the result of splat
  LogicalResult parseSplat(triton::SplatOp splatOp, const Location &loc,
                           OpBuilder &builder);

  // Operand is the result of expand_dims
  LogicalResult parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                const Location &loc, OpBuilder &builder);
};

} // namespace triton

} // namespace mlir

#endif
