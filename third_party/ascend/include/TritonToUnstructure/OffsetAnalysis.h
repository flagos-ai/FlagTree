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

#ifndef TRITON_ANALYSIS_OFFSETANALYSIS_H
#define TRITON_ANALYSIS_OFFSETANALYSIS_H
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {

/// Temporary T2U handoff marker for loop slots rewritten from a scalar
/// pointer to a relative integer offset.  OffsetAnalysis uses this marker to
/// keep the region argument's dynamic backedge value instead of replaying the
/// original pointer-init provenance on every iteration.
inline constexpr llvm::StringLiteral kScalarPointerOffsetBoundaryAttr =
    "ScalarPointerOffsetBoundary";

struct PtrOffsetInfo {
  /**
  Possible status of the ptr offset:
   - ScalarLike:
      - Tensor's elements are all the same such as [[2.0,2.0,2.0],[2.0,2.0,2.0]]
      - Constant integer or floating-point such as 2, 2.0, and `load
  tensor<1xptr>`
   - Unstructured:
      - Not a `ScalarLike` ptr offset
      - Or satisfy any below conditions:
        - Incontinuous stride such as
          - `muli [0,1,2,3] [0,1,2,3]` => [0,1,4,9]
          - `divsi [9,8,7] [3,2,1]` => [3,4,7]
          - `minsi [3,4,5] [5,4,3]` => [3,4,3]
        - From non-`scalarLike` floating point element type such as
          - `fptosi [1.0,2.0,3.0]` => [1,2,3]
        - Compilation time unknown value
          - `load %ptr, %offset` => %value
    - Structured:
      - orthongonal to `Unstructured`
        - if PtrOffsetInfo isn't `Unstructured`, it is `Structured`

  In short:
  ScalarLike ⊆ Structured
  Unstructured = {x| x ∉ Structured}

  Example:
  ```
  %y = sitofp %x
  %z = fptosi %y
  ```
  If %x is scalarLike (structured), %z will be scalar (structured) as well.
  If %x is non-scalarLike structured, %z will be unstructured.
  */

public:
  enum class AxisInfo { unstructured, structured, scalarlike, scalar };

  explicit PtrOffsetInfo();
  PtrOffsetInfo(const PtrOffsetInfo &other);

  explicit PtrOffsetInfo(const Value &ptr);
  explicit PtrOffsetInfo(ArrayRef<AxisInfo> structured);
  explicit PtrOffsetInfo(const Value &ptr, AxisInfo structured);
  explicit PtrOffsetInfo(const Value &ptr, ArrayRef<AxisInfo> structured);
  explicit PtrOffsetInfo(const Value &ptr, const Value &offset,
                         AxisInfo structured);
  explicit PtrOffsetInfo(const Value &ptr, const Value &offset,
                         ArrayRef<AxisInfo> structured);

  PtrOffsetInfo &operator=(const PtrOffsetInfo &other);

  Value getPtr() const;
  Value getOffset() const;
  SmallVector<Value> getOffsets() const;
  SmallVector<Value> &getOffsetsRef();
  bool isScalarLike() const;
  bool isPointerDescriptorOwned() const;
  SmallVector<AxisInfo> &getStructuredRef();
  const SmallVector<AxisInfo> &getStructured() const;
  int getRank() const;

  void setPtr(const Value &ptr);
  void setOffset(const Value &offset);
  void setOffsets(ValueRange offsets);
  void setStructured();
  void setStructured(int rank);
  void setStructured(int rank, AxisInfo info);
  // These setters update only per-axis structure. They intentionally preserve
  // the independent scalarLike property; callers that invalidate scalar
  // equivalence must clear it explicitly.
  void setUnstructured();
  void setUnstructured(int rank);
  void setStructured(ArrayRef<AxisInfo> structured);
  void setStructured(const PtrOffsetInfo &other);
  void setScalarLike(bool scalarLike);
  void setPointerDescriptorOwned(bool pointerDescriptorOwned);

  bool isStructured(int dim) const;
  bool isStructured() const;
  bool isUnstructured() const;
  bool isUnstructuredOrScalarlike() const;

  void setZeroOffset();

private:
  Value ptr;
  Value offset;
  SmallVector<Value> tptOffsets;

  bool scalarLike = false;

  // True only when the pointer originates from a CFO descriptor rebuild and
  // every pointer-preserving operation since that rebuild was modeled by this
  // analysis. Consumers may then use the analyzed scalar base without relying
  // on the load operand's direct producer.
  bool pointerDescriptorOwned = false;

  SmallVector<AxisInfo> structured;
};

PtrOffsetInfo combineInfo(const PtrOffsetInfo &lhs, const PtrOffsetInfo &rhs);

void parse(Value operand, const Location &loc, RewriterBase &rewriter,
           llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseLoopRegionIterArg(LoopLikeOpInterface loopOp, const Location &loc,
                            RewriterBase &rewriter,
                            llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                            BlockArgument regionIterArg);

void parseArithOp(Operation *arithOp, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseTritonOp(Operation *tritonOp, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAddPtr(triton::AddPtrOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSplat(triton::SplatOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

template <typename BinOpTy>
void parseBinaryOp(BinOpTy op, const Location &loc, RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAddI(arith::AddIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSubI(arith::SubIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

template <typename ConstOpTy>
void parseConstantOp(ConstOpTy dst, const Location &loc, RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMakeRange(triton::MakeRangeOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExtSI(arith::ExtSIOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseBitcast(triton::BitcastOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseLoad(triton::LoadOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMulI(arith::MulIOp op, const Location &loc, RewriterBase &rewriter,
               llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseBroadcast(triton::BroadcastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExpandDims(triton::ExpandDimsOp op, const Location &loc,
                     RewriterBase &rewriter,
                     llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseReshape(triton::ReshapeOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseClampF(triton::ClampFOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSelect(arith::SelectOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseFPToSI(arith::FPToSIOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseSIToFP(arith::SIToFPOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

// FIXME:Z|wait triton version upgrade to 3.4
// void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
//                          RewriterBase &rewriter,
//                          llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseMakeTensorPtr(triton::MakeTensorPtrOp op, const Location &loc,
                        RewriterBase &rewriter,
                        llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseAdvance(triton::AdvanceOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseReduce(triton::ReduceOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseReduceReturn(triton::ReduceReturnOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIf(scf::IfOp op, const Location &loc, RewriterBase &rewriter,
             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseYield(scf::YieldOp op, const Location &loc, RewriterBase &rewriter,
                llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseLoopOp(LoopLikeOpInterface op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseInsertSlice(tensor::InsertSliceOp op, const Location &loc,
                      RewriterBase &rewriter,
                      llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExtract(tensor::ExtractOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseInsert(tensor::InsertOp op, const Location &loc,
                 RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseStructuredCustomOp(Operation *op, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
                             unsigned resultIdx);
} // namespace triton

} // namespace mlir

#endif
