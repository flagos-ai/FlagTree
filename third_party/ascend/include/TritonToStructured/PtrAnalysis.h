/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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
#ifndef TRITON_TO_STRUCTURED_PTRANALYSIS_H
#define TRITON_TO_STRUCTURED_PTRANALYSIS_H

#include <cstddef>
#include <set>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace TritonToStructured {
using namespace mlir;
using namespace triton;

struct StateInfo {
  OpFoldResult stride;
  OpFoldResult shape; // rem value
  size_t dimIndex;

  StateInfo() : dimIndex(0) {}
  StateInfo(OpFoldResult stride, OpFoldResult shape, size_t dimIndex = 0)
      : stride(stride), shape(shape), dimIndex(dimIndex) {}
  void dump() const;
};

struct PtrState {
  SmallVector<StateInfo>
      stateInfo; // shape info when load, maintained with visitOps
  SmallVector<OpFoldResult> sizes; // original shape, maintained with visitOps
  SmallVector<size_t> permuteIds;
  SmallVector<size_t>
      order; // the order of the original data format, only used for block_ptr
  SmallVector<OpFoldResult>
      dimOffsets; // the offsets per dimension, only used for block_ptr

  Value source;        // base address (ptr), maintained with visitOps
  OpFoldResult offset; // scalar offset (int), maintained with visitOps

  // whether the record needs to be processed in the current pass, when ignore
  // is true, it indicates that this scenario should not be processed within the
  // current pass
  bool shouldLinearize = false;
  bool isPermuted = false;

  void dump() const;
  bool isEmpty() const;
  bool isScalar() const;
  bool hasSource() const;
  bool isBlockPtr() const;
  bool isSameSizeAs(const PtrState &x) const;

  void generateOriginPermuteIds();

  // Formula of "contiguous axes"
  // - axis i is contiguous if stride[i] == product(shape[0..i-1]).
  size_t countContiguousAxes(SmallVector<StateInfo> stateInfo) const;
  void analyzePermute();

  void updatePtrState(SmallVector<StateInfo> stateInfo,
                      SmallVector<OpFoldResult> sizes, Value source,
                      OpFoldResult offset, const Location loc,
                      OpBuilder &builder, bool shouldLinearize = false);

  void normalizeState(const Location loc, OpBuilder &builder);

  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder);

  LogicalResult subState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder);

  LogicalResult addState(PtrState &lhsState, PtrState &rhsState, Operation *op,
                         OpBuilder &builder);

  triton::AddPtrOp createAddPtrOp(OpBuilder &builder, Location loc);

  triton::MakeTensorPtrOp createMakeTensorPtrOp(OpBuilder &builder,
                                                Location loc);
};

class PtrAnalysis {
public:
  // AddptrOp result -> PtrState
  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  IRMapping ptrMap;

  bool operandIsScalar(Value operand);

  bool optimizeDynamicOffset;

  PtrAnalysis(bool optimizeDynamicOffset = false)
      : optimizeDynamicOffset(optimizeDynamicOffset) {}

  LogicalResult initStateByScalar(Value operand, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult initStateByPointer(Value operand, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMul(arith::MulIOp mulOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandSub(arith::SubIOp subOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder);

  LogicalResult visitOperandBroadcast(triton::BroadcastOp broadcastOp,
                                      PtrState &state, const Location loc,
                                      OpBuilder &builder);

  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExpandDims(triton::ExpandDimsOp expandDimsOp,
                                       PtrState &state, const Location loc,
                                       OpBuilder &builder);

  LogicalResult visitOperandConstSplat(arith::ConstantOp op, PtrState &state,
                                       const Location loc, OpBuilder &builder);

  LogicalResult visitOperandExtSI(arith::ExtSIOp extOp, PtrState &state,
                                  const Location loc, OpBuilder &builder);

  LogicalResult visitOperandRem(arith::RemSIOp remOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandDiv(arith::DivSIOp divOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  LogicalResult visitOperandAdd(arith::AddIOp addOp, PtrState &state,
                                const Location loc, OpBuilder &builder);

  // Recursively parse a Value; call the corresponding
  // function based on the defining operation and argument type.
  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc,
                             OpBuilder &builder);

  // Operand is the result of addptr.
  // Main assumptions:
  // - The ptr field should populate the source field
  // - ptr and offset fields should result in same rank
  // Expected result:
  // - The resulting state for ptr and offset wil be added
  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   const Location loc, OpBuilder &builder);

  // Operand is the result of tt.make_tensor_ptr.
  // Expected result:
  //  Parse source pointer and grab results
  LogicalResult
  visitOperandMakeTensorPtr(triton::MakeTensorPtrOp makeTensorPtrOp,
                            PtrState &state, const Location loc,
                            OpBuilder &builder);

  // Parse the state of AddPtrOp, insert any instruction needed to
  // calculate strides and offsets, build PtrState for this operand, and record
  // PtrState for knownPtrs.
  LogicalResult rewriteAddptrOp(triton::AddPtrOp op);
};

bool isMultiple(const OpFoldResult &dividend, const OpFoldResult &divisor);
bool isEqual(const OpFoldResult &ofr1, const OpFoldResult &ofr2);
bool isLess(const OpFoldResult &ofs1, const OpFoldResult &ofs2);
bool isGreater(const OpFoldResult &ofs1, const OpFoldResult &ofs2);
std::optional<int32_t>
extractDivisibilityFromOpFoldResult(mlir::OpFoldResult ofr);

} // namespace TritonToStructured

#endif
