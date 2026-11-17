
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
#ifndef GCU_ANALYSIS_MASKANALYSIS_H
#define GCU_ANALYSIS_MASKANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

//===--------------------------------------------------------------------===//
// The main code is modified from triton-to-linalg branch in triton repo.
//===--------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace gcu {
// Data structure used to decode the pattern in a mask used for load and store.
// starts and ends are per-dimension: starts[i] / ends[i] delimit the valid
// region [starts[i], ends[i]) within the tile along dimension i, in units of
// the corresponding make_range index (whose stride is always 1). dims[i] is
// the valid size used as the load/store shape and equals ends[i] - starts[i].
// start/end are per-dimension (not single) because a multi-dim mask can carry
// a range on more than one dimension at a time, e.g. (rows >= r0) & (cols >=
// c0). scalar is a shortcut used when the entire state contains a single
// scalar value.
//
// The general lifetime of this data structure is roughly:
// 1. A range is created by make_range and optionally operated on by addi w/
// result of splat, expand_dims, etc. During this phase, starts/ends hold the
// (possibly shifted) range bounds and dims the range size. Only one of the
// dimensions (that contains the range) can have dim > 1.
// 2. Result from step 1 is compared with a another MaskState that represents a
// scalar value. For a `<`-style comparison the valid region is clipped from
// the end (starts stays 0); for a `>=`-style comparison it is clipped from
// the start (starts becomes > 0). dims is shrunk accordingly.
// 3. Optionally, result from step 2 can be broadcasted and anded with other
// results from step 2; each dimension's valid region is intersected.
//
// Example of creating 2D mask:
//  mask = (rows[:, None] < M) & (cols[None, :] >= half)
struct MaskState {
  SmallVector<OpFoldResult> starts;
  SmallVector<OpFoldResult> ends;
  SmallVector<OpFoldResult> dims;
  OpFoldResult scalar;

  int64_t getRank() const { return dims.size(); }

  bool isEmpty() const {
    return getRank() == 0 && !scalar && starts.empty() && ends.empty();
  }

  bool isMask() const { return !scalar && dims.size() != 0; }

  void addStateScalar(OpBuilder &builder, Location loc, const MaskState &state,
                      const OpFoldResult scalar);

  void addStates(OpBuilder &builder, Location loc, const MaskState &lhsState,
                 const MaskState &rhsState);

  void minStates(OpBuilder &builder, Location loc, const MaskState &lhsState,
                 const MaskState &rhsState);

  void setStates(OpBuilder &builder, Location loc, const MaskState &srcState);
};

class MaskAnalysis {
public:
  // Recursively parse a Value; call the corresponding function based on the
  // defining operation and Value type
  static bool parse(OpBuilder &builder, Location loc, Value operand,
                    MaskState &state,
                    llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  static void
  parseBlockArgument(OpBuilder &builder, Location loc, BlockArgument blockArg,
                     MaskState &state,
                     llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  static void parseIntScalar(OpBuilder &builder, Location loc, Value scalar,
                             MaskState &state,
                             llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of a constant
  // Get the value of the constant and assign it to scalar.
  static void parseConstant(OpBuilder &builder, Location loc,
                            arith::ConstantOp constOp, MaskState &state,
                            llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of addi
  // One and only one of the operands should be a scalar. Increment both start
  // and end, dims remains unchanged, and scalar is empty.
  static void parseAdd(OpBuilder &builder, Location loc, arith::AddIOp addOp,
                       MaskState &state,
                       llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of andi
  // Each of the result state dims is smaller of the two operands' dims.
  // Insert instruction if needed to get new dims.
  static void parseAnd(OpBuilder &builder, Location loc, arith::AndIOp andOp,
                       MaskState &state,
                       llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of cmpi
  // Assume only of the dimensions have size > 1. Only support slt for now.
  // For that dimension, calculate this new dim as: dim = min(end, value) -
  // start
  static void parseCmp(OpBuilder &builder, Location loc, arith::CmpIOp cmpOp,
                       MaskState &state,
                       llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of make_range
  // Set start and end accordingly; step size must be 1.
  static void parseMakeRange(OpBuilder &builder, Location loc,
                             triton::MakeRangeOp rangeOp, MaskState &state,
                             llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of broadcast
  // Change dims only; assume only applies to tensors.
  static void parseBroadcast(OpBuilder &builder, Location loc,
                             triton::BroadcastOp broadcastOp, MaskState &state,
                             llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of splat
  // Assume only applies to scalar. start and end are left empty; scalar will
  // be assigned, and dims will be updated.
  static void parseSplat(OpBuilder &builder, Location loc,
                         triton::SplatOp splatOp, MaskState &state,
                         llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of expand_dims
  // Insert additional dims; start and end do not change and correspond to the
  // dimension that contains the range.
  static void
  parseExpandDims(OpBuilder &builder, Location loc,
                  triton::ExpandDimsOp expandDimsOp, MaskState &state,
                  llvm::SmallDenseMap<Value, MaskState> &knownMasks);

  // Operand is the result of DotC
  static void parseDot(OpBuilder &builder, Location loc, triton::DotOp dotOp,
                       MaskState &state,
                       llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of remsi
  // One and only one of the operands should be a scalar. Increment both start
  // and end, dims remains unchanged, and scalar is empty.
  static void parseRemsi(OpBuilder &builder, Location loc,
                         arith::RemSIOp RemSIOp, MaskState &state,
                         llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of SelectOp
  // only for bypass
  static void parseSelect(OpBuilder &builder, Location loc,
                          arith::SelectOp SelectOp, MaskState &state,
                          llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of ReduceOp
  // only for bypass
  static void parseReduce(OpBuilder &builder, Location loc,
                          triton::ReduceOp ReduceOp, MaskState &state,
                          llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of LoadOp
  // only for bypass
  static void parseLoad(OpBuilder &builder, Location loc, triton::LoadOp LoadOp,
                        MaskState &state,
                        llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of ExtSIOp
  // only for bypass
  static void parseExtsi(OpBuilder &builder, Location loc,
                         arith::ExtSIOp ExtSIOp, MaskState &state,
                         llvm::SmallDenseMap<Value, MaskState> &knownMasks);
  // Operand is the result of ExtUIOp
  // only for bypass
  static void parseExtui(OpBuilder &builder, Location loc,
                         arith::ExtUIOp ExtUIOp, MaskState &state,
                         llvm::SmallDenseMap<Value, MaskState> &knownMasks);
};

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // GCU_ANALYSIS_MASKANALYSIS_H
