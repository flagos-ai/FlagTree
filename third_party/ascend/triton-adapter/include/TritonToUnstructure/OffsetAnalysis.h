#ifndef TRITON_ANALYSIS_OFFSETANALYSIS_H
#define TRITON_ANALYSIS_OFFSETANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

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
  explicit PtrOffsetInfo();
  PtrOffsetInfo(const PtrOffsetInfo &other);

  explicit PtrOffsetInfo(const Value &ptr);
  explicit PtrOffsetInfo(ArrayRef<bool> structured);
  explicit PtrOffsetInfo(const Value &ptr, bool structured);
  explicit PtrOffsetInfo(const Value &ptr, ArrayRef<bool> structured);
  explicit PtrOffsetInfo(const Value &ptr, const Value &offset, bool structured);
  explicit PtrOffsetInfo(const Value &ptr, const Value &offset, ArrayRef<bool> structured);

  PtrOffsetInfo &operator=(const PtrOffsetInfo &other);

  Value getPtr() const;
  Value getOffset() const;
  bool isScalarLike() const;
  bool isNegativeFlag() const;
  SmallVector<bool> &getStructuredRef();
  const SmallVector<bool> &getStructured() const;
  int getRank() const;

  void setPtr(const Value &ptr);
  void setOffset(const Value &offset);
  void setStructured();
  void setStructured(int rank);
  void setUnstructured();
  void setUnstructured(int rank);
  void setStructured(ArrayRef<bool> structured);
  void setStructured(const PtrOffsetInfo &other);
  void setScalarLike(bool scalarLike);
  void setNegativeFlag(bool negativeFlag);
  bool isStructured(int dim) const;
  bool isStructured() const;
  bool isUnstructured() const;

  void setZeroOffset();
private:
  Value ptr;
  Value offset;

  bool scalarLike = false;
  bool negativeFlag = false;
  SmallVector<bool> structured;
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

void parseIndexCast(arith::IndexCastOp op, const Location &loc,
                    RewriterBase &rewriter,
                    llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

template <typename ConstOpTy>
void parseConstantOp(ConstOpTy dst, const Location &loc,
                     RewriterBase &rewriter,
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

void parseMakeTensorDesc(triton::MakeTensorDescOp op, const Location &loc,
                         RewriterBase &rewriter,
                         llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

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

void parseLoopOp(LoopLikeOpInterface op, const Location &loc, RewriterBase &rewriter,
                 llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap, Value dst);

void parseExtractSlice(tensor::ExtractSliceOp op, const Location &loc,
                       RewriterBase &rewriter,
                       llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseExtract(tensor::ExtractOp op, const Location &loc,
                  RewriterBase &rewriter,
                  llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);

void parseIntToPtr(triton::IntToPtrOp op, const Location &loc,
                   RewriterBase &rewriter,
                   llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap);
} // namespace triton

} // namespace mlir

#endif