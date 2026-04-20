/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCU_ANALYSIS_PTRANALYSIS_H
#define GCU_ANALYSIS_PTRANALYSIS_H

#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

//===--------------------------------------------------------------------===//
// The main code is modified from triton-to-linalg branch in triton repo.
//===--------------------------------------------------------------------===//

namespace mlir {

class ModuleOp;
class PatternRewriter;

namespace triton {
namespace gcu {

using llvm::SmallDenseMap;
using llvm::SmallVector;

struct MaskState;

struct PtrInfo {
  Value base;
  llvm::SmallVector<Value, 4> shape;
  llvm::SmallVector<Value, 4> strides;
  llvm::SmallVector<Value, 4> offsets;
  llvm::DenseSet<int32_t> broadcastDims;
};

// Data structure used to decode pointer arithmetic and potentially to be
// translate it into memref. offsets, sizes, and strides are in unit of elements
// in a linearly laid-out memory, which is the same as pointer arithmetic
// operations in Triton language. scalar is a shortcut used when the entire
// state describes a single scalar value. source is the base pointer.
struct PtrState {
  llvm::SmallVector<OpFoldResult> offsets;
  llvm::SmallVector<OpFoldResult> sizes;
  llvm::SmallVector<OpFoldResult> strides;

  Value source;
  Value scalar;

  int64_t getRank() const;
  bool isEmpty() const;

  // Process addition of two PtrStates
  void addState(OpBuilder &builder, Location loc, const PtrState &lhsState,
                const PtrState &rhsState);

  // Process multiplication of two PtrStates
  void mulState(OpBuilder &builder, Location loc, const PtrState &lhsState,
                const PtrState &rhsState);

  // Process division remainder of two PtrStates
  void remState(OpBuilder &builder, Location loc, const PtrState &lhsState,
                const PtrState &rhsState);

  // Process division of two PtrStates
  void divState(OpBuilder &builder, Location loc, const PtrState &lhsState,
                const PtrState &rhsState);

  // set state for srcState
  void setState(OpBuilder &builder, Location loc, const PtrState &srcState);

  PtrInfo getPtrInfo(OpBuilder &builder, Location loc, const MaskState &mstate);
};

class PtrAnalysis {
public:
  // Recursively parse a Value; call the corresponding function based on the
  // defining operation and Value type
  static void visitOperand(PatternRewriter &rewriter, Location loc,
                           Value operand, PtrState &state,
                           llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is a block argument
  static void
  visitBlockArgument(PatternRewriter &rewriter, Location loc, BlockArgument arg,
                     PtrState &state,
                     llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.constant that is a splat
  // Main assumptions:
  //  Source is a constant op that produces a constant dense tensor where all
  //  elements are the same (i.e.: a constant that is splatted)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] =
  //  splat value if i == 0, otherwise 0
  static void
  visitOperandConstSplat(PatternRewriter &rewriter, Location loc,
                         arith::ConstantOp op, PtrState &state,
                         llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.addi. Process both arguments and insert any
  // arith.addi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = lhsState.source ? lhsState.source : rhsState.source
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] + rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] + rhsState.strides[i]
  static void visitOperandAdd(PatternRewriter &rewriter, Location loc,
                              arith::AddIOp addOp, PtrState &state,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.muli. Process both arguments and insert any
  // arith.muli instruction as needed.
  // Main assumptions:
  //  Neither lhsState nor rhsState has source field set
  //  Current PtrState should be empty
  //  Currently only support one of the operand is a scalar index
  // Expected result (scalar and tensorState represent the two operands):
  //  source = null
  //  sizes[i] = tensorState.sizes[i]
  //  offsets[i] = tensorState.offsets[i] * scalar
  //  strides[i] = tensorState.strides[i] * scalar
  static void visitOperandMul(PatternRewriter &rewriter, Location loc,
                              arith::MulIOp mulOp, PtrState &state,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.remsi. Process both arguments and insert any
  // arith.remsi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = null
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] % rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] (which should match rhsState.strides[i])
  static void visitOperandRem(PatternRewriter &rewriter, Location loc,
                              arith::RemSIOp remOp, PtrState &state,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.divsi. Process both arguments and insert any
  // arith.divsi instruction as needed.
  // Main assumptions:
  //  Only one of lhsState and rhsState has source field set
  //  Current PtrState should be empty
  // Expected result:
  //  source = null
  //  sizes[i] = lhsState.sizes[i] (which should match rhsState.sizes[i])
  //  offsets[i] = lhsState.offsets[i] / rhsState.offsets[i]
  //  strides[i] = lhsState.strides[i] (which should match rhsState.strides[i])
  static void visitOperandDiv(PatternRewriter &rewriter, Location loc,
                              arith::DivSIOp divOp, PtrState &state,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of arith.select. Process both arguments and insert
  // any arith.select instruction as needed.
  // Main assumptions:
  // Select lhsState or rhsState
  //  Current PtrState should be empty
  // Expected result:
  //  The resulting state is lhsState or rhsState
  static void
  visitOperandSelect(PatternRewriter &rewriter, Location loc,
                     arith::SelectOp selectOp, PtrState &state,
                     llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of make_range.
  // Main assumptions:
  //  start, end, and shape are all statically known
  //  The output of make_range is 1-dimensional
  //  Does not check validity of inputs (e.g., stride > 0)
  // Expected result:
  //  source = null
  //  sizes[0] = shape[0]
  //  offset[0] = start
  //  strides[0] = ceiling( (end - start) / shape[0] )
  static void
  visitOperandMakeRange(PatternRewriter &rewriter, Location loc,
                        triton::MakeRangeOp rangeOp, PtrState &state,
                        llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of expand_dims
  // Main assumptions:
  //  Only 1 dimension changes for each invocation of reshape
  //  The changed dimension must have size of 1
  // Expected result:
  //  Insert a dimension of size 1, stride 0, and offset 0
  static void
  visitOperandExpandDims(PatternRewriter &rewriter, Location loc,
                         triton::ExpandDimsOp expandDimsOp, PtrState &state,
                         llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of broadcast
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  Update sizes[i] only, no changes to other fields
  static void
  visitOperandBroadcast(PatternRewriter &rewriter, Location loc,
                        triton::BroadcastOp broadcastOp, PtrState &state,
                        llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of splat
  // Main assumptions:
  //  Source is a scalar value (i.e., an integer or a pointer, not a tensor)
  // Expected result:
  //  sizes[i] reflect the shape of the result, strides[i] = 0,  offsets[i] = 0
  //  if source is an integer, offset[0] = scalar = source
  static void
  visitOperandSplat(PatternRewriter &rewriter, Location loc,
                    triton::SplatOp splatOp, PtrState &state,
                    llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of addptr.
  // Main assumptions:
  //  The ptr field should populate the source field
  //  ptr and offset fields should result in same rank
  // Expected result:
  //  The resulting state for ptr and offset will be added
  static void
  visitOperandAddptr(PatternRewriter &rewriter, Location loc,
                     triton::AddPtrOp addptrOp, PtrState &state,
                     llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of bitcast
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be added
  static void
  visitOperandBitcast(PatternRewriter &rewriter, Location loc,
                      triton::BitcastOp bitcastOp, PtrState &state,
                      llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of trans
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be trans
  static void
  visitOperandTrans(PatternRewriter &rewriter, Location loc,
                    triton::TransOp transOp, PtrState &state,
                    llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of dots
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be same
  static void visitOperandDot(PatternRewriter &rewriter, Location loc,
                              triton::DotOp dotOp, PtrState &state,
                              llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of reduce
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be same
  static void
  visitOperandReduce(PatternRewriter &rewriter, Location loc,
                     triton::ReduceOp reduceOp, PtrState &state,
                     llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of load
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be same
  static void visitOperandLoad(PatternRewriter &rewriter, Location loc,
                               triton::LoadOp loadOp, PtrState &state,
                               llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of extsi
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be same
  static void
  visitOperandExtsi(PatternRewriter &rewriter, Location loc,
                    arith::ExtSIOp extsiOp, PtrState &state,
                    llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Operand is the result of extui
  // Main assumptions:
  //  Rank of source and result is the same
  // Expected result:
  //  The resulting state for all will be same
  static void
  visitOperandExtui(PatternRewriter &rewriter, Location loc,
                    arith::ExtUIOp extuiOp, PtrState &state,
                    llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // bypass ForOp not include ld/st.
  static bool byPassForOp(PatternRewriter &rewriter, scf::ForOp op,
                          const SmallVector<Operation *, 8> &candidateOps);
  // Parse the state of ForOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  static LogicalResult rewriteForOp(
      PatternRewriter &rewriter, scf::ForOp op,
      SmallDenseMap<Value, PtrState> &knownPtrs,
      SmallDenseMap<Value, MaskState> &knownMasks,
      SmallVector<Operation *, 8> &candidateOps,
      SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateHints);

  // Parse the state of YieldOp, insert any instruction needed to calculate
  // strides and offsets, build PtrState for this operand, and record PtrState
  // in knownPtrs.
  static void rewriteYieldOp(PatternRewriter &rewriter, scf::YieldOp op,
                             llvm::SmallDenseMap<Value, PtrState> &knownPtrs,
                             llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Parse the iter arg of ForOp, fold away unused ones.
  static void foldAwayForOp(PatternRewriter &rewriter, scf::ForOp op,
                            llvm::SmallDenseMap<Value, PtrState> &knownPtrs);

  // Collect candidate load/store op which could be converted to dma.
  static void collectCandidateLoadStoreOps(
      ModuleOp &moduleOp, llvm::SmallVector<Operation *, 8> &candidates,
      llvm::SmallDenseMap<Operation *, SmallVector<int32_t>> &candidateOrders);
};

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // GCU_ANALYSIS_PTRANALYSIS_H
