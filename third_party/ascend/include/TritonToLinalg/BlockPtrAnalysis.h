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

#ifndef TRITON_ANALYSIS_BLOCKPTRANALYSIS_H
#define TRITON_ANALYSIS_BLOCKPTRANALYSIS_H
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <set>
namespace mlir {

class ConversionPatternRewriter;

namespace triton {

/// Temporary provenance for memref carriers materialized from complete scalar
/// pointer addresses. Address round-trip folding is valid only for casts with
/// this marker; ordinary HIVM PointerCast operations retain their baseline
/// descriptor semantics. TritonToLinalg removes the marker after conversion.
inline constexpr llvm::StringLiteral kScalarPointerCarrierAttr =
    "ScalarPointerCarrier";

/// Returns true when a loop still carries Triton pointer state, or when a
/// loop-carried integer tensor participates in legacy address computation.
/// Ordinary memrefs, including fixed-layout memrefs created by
/// memref.reinterpret_cast, are complete SSA values and do not require the
/// legacy BlockData loop expansion solely because of their producer.
bool needsLegacyBlockDataLoopRewrite(LoopLikeOpInterface loopOp);

enum class MemAccVal { Undefined = 0, StrucMemAcc = 1, UnstrucMemAcc = 2 };

/// Creates a verifier-valid HIVM pointer cast for a scalar Triton pointer
/// represented by an integer address. Triton scalar pointers do not carry an
/// extent, while their downstream carrier is normally `memref<?xT>` and every
/// dynamic memref dimension requires a size operand. Use one element as the
/// conservative carrier extent; reinterpret-cast lowering replaces it with a
/// precise access range when a larger descriptor is materialized.
hivm::PointerCastOp createScalarPointerCast(OpBuilder &builder, Location loc,
                                            MemRefType resultType,
                                            Value address);

struct MemAccType {

  MemAccVal value;

  explicit constexpr MemAccType(MemAccVal v = MemAccVal::Undefined)
      : value(v) {}

  constexpr operator MemAccVal() const { return value; }
  explicit operator bool() = delete;

  constexpr bool isUndefined() const { return value == MemAccVal::Undefined; }
  constexpr bool isStructured() const {
    return value == MemAccVal::StrucMemAcc;
  }
  constexpr bool isUnstructured() const {
    return value == MemAccVal::UnstrucMemAcc;
  }

  void merge(MemAccType &other) {
    this->value = (this->value > other.value) ? this->value : other.value;
  }

  std::string_view toString() const {
    static constexpr std::string_view names[] = {"Undefined", "StrucMemAcc",
                                                 "UnstrucMemAcc"};
    return names[static_cast<int>(value)];
  }
};

class BlockData {
public:
  SmallVector<OpFoldResult> &getOffsetsRef();
  SmallVector<OpFoldResult> &getSizesRef();
  SmallVector<OpFoldResult> &getStridesRef();
  Value &getSourceRef();
  OpFoldResult &getScalarRef();
  Type &getResElemTyRef();
  MemAccType &getMemAccTypeRef();

  SmallVector<OpFoldResult> getOffsets() const;
  SmallVector<OpFoldResult> getSizes() const;
  SmallVector<OpFoldResult> getStrides() const;
  Type getResElemTy() const;
  OpFoldResult getOffset(int) const;
  OpFoldResult getSize(int) const;
  OpFoldResult getStride(int) const;
  OpFoldResult getScalar() const;
  Value getSource() const;
  MemAccType getMemAccType() const;

  bool isScalar() const;
  bool isEmpty() const;
  bool hasSource() const;
  bool hasResElemTy() const;
  void removeSource();

  int64_t getRank() const;
  FailureOr<MemRefType>
  getResultMemrefType(int64_t offset, ArrayRef<int64_t> resultShape) const;

  void addBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);
  void subBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);
  void mulBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);
  void divBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);

  FailureOr<memref::ReinterpretCastOp>
  createCastOp(ArrayRef<int64_t> resultShape, const Location &loc,
               OpBuilder &builder) const;

  void setResElemTy(const Type &);
  void setSource(const Value &);
  void setScalar(const OpFoldResult &);
  void setOffsets(const SmallVector<OpFoldResult> &);
  void setStrides(const SmallVector<OpFoldResult> &);
  void setSizes(const SmallVector<OpFoldResult> &);
  void setMemAccTy(const MemAccType &);
  void setMemAccVal(const MemAccVal);

  void dump() const;

private:
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  Value source;
  // `Scalar` is a shortcut used when the entire blockdata describes a single
  // scalar value
  OpFoldResult scalar;
  Type resElemTy;
  MemAccType memAccTy;

  // Accumulate offsets of each dimension in BlockData to get a total offset
  // from source ptr, which is used in memref::ReinterpretCastOp
  OpFoldResult inferBlockOffset(const Location &loc, OpBuilder &builder) const;
};

class BlockDataParser {
public:
  /// Materialize the rank-one memref used by a scalar load only when the
  /// pointer producer has already been converted to a compatible memref.
  /// Returning failure lets a consumer reject an unconverted producer without
  /// asserting or creating partial IR.
  static FailureOr<Value> getScalarMemRef(Value ptr, Value memref,
                                          const Location &loc,
                                          ConversionPatternRewriter &rewriter);

  static LogicalResult
  parse(Value operand, BlockData &data, const Location &loc,
        ConversionPatternRewriter &rewriter,
        const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseAdd(arith::AddIOp op, BlockData &data, const Location &loc,
           ConversionPatternRewriter &rewriter,
           const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseSub(arith::SubIOp op, BlockData &data, const Location &loc,
           ConversionPatternRewriter &rewriter,
           const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseMul(arith::MulIOp op, BlockData &data, const Location &loc,
           ConversionPatternRewriter &rewriter,
           const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseDiv(arith::DivSIOp op, BlockData &data, const Location &loc,
           ConversionPatternRewriter &rewriter,
           const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseRem(arith::RemSIOp op, BlockData &data, const Location &loc,
           ConversionPatternRewriter &rewriter,
           const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseUnrealizedCast(UnrealizedConversionCastOp op, BlockData &data,
                      const Location &loc, ConversionPatternRewriter &rewriter,
                      const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseMakeRange(triton::MakeRangeOp op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseLinalgGenericFromMakeRange(
      linalg::GenericOp op, BlockData &data, const Location &loc,
      ConversionPatternRewriter &rewriter,
      const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseExpandDims(triton::ExpandDimsOp op, BlockData &data, const Location &loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseBitcast(triton::BitcastOp op, BlockData &data, const Location &loc,
               ConversionPatternRewriter &rewriter,
               const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseExtSI(arith::ExtSIOp op, BlockData &data, const Location &loc,
             ConversionPatternRewriter &rewriter,
             const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseBroadcast(triton::BroadcastOp op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseSplat(triton::SplatOp op, BlockData &data, const Location &loc,
             ConversionPatternRewriter &rewriter,
             const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseConstSplat(arith::ConstantOp op, BlockData &data, const Location &loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, BlockData> &known);

  template <typename T>
  static std::enable_if_t<std::is_same_v<T, triton::MakeTensorPtrOp> ||
                              std::is_same_v<T, triton::AdvanceOp>,
                          LogicalResult>
  parseTensorPtr(T op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseAddPtr(triton::AddPtrOp op, BlockData &data, const Location &loc,
              ConversionPatternRewriter &rewriter,
              const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseExtractSlice(tensor::ExtractSliceOp op, BlockData &data,
                    const Location &loc, ConversionPatternRewriter &rewriter,
                    const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseReinterpretCast(memref::ReinterpretCastOp op, BlockData &data,
                       const Location &loc, ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseReduce(triton::ReduceOp op, BlockData &data, const Location &loc,
              ConversionPatternRewriter &rewriter,
              const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseAtomicRmw(triton::AtomicRMWOp op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseFill(linalg::FillOp op, BlockData &data, const Location &loc,
            ConversionPatternRewriter &rewriter,
            const llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  parseSelect(arith::SelectOp op, BlockData &data, const Location &loc,
              ConversionPatternRewriter &rewriter,
              const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseCustomOp(hivm::CustomOp op, BlockData &data,
                            const Location &loc,
                            ConversionPatternRewriter &rewriter,
                            const llvm::SmallDenseMap<Value, BlockData> &known,
                            unsigned resultIdx);

  static LogicalResult
  parseStructuredCustomOp(Operation *op, BlockData &data, const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known,
                          unsigned resultIdx);

  static void rewriteStructuredCustomOp(hivm::CustomOp op,
                                        hivm::CustomOp::Adaptor &adaptor,
                                        ConversionPatternRewriter &rewriter);

  static void rewriteStructuredCustomOp(hivm::CustomMacroOp op,
                                        hivm::CustomMacroOp::Adaptor &adaptor,
                                        ConversionPatternRewriter &rewriter);

  static void rewriteStructuredCustomOp(Operation *op,
                                        ConversionPatternRewriter &rewriter);

  static LogicalResult
  rewriteAddPtr(triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
                ConversionPatternRewriter &rewriter,
                llvm::SmallDenseMap<Value, BlockData> &known);

  static FailureOr<Value>
  materializePointer(Value ptr, ConversionPatternRewriter &rewriter,
                     llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op, Value convertedBase,
                         ConversionPatternRewriter &rewriter,
                         llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  rewriteAdvanceOp(triton::AdvanceOp op, ConversionPatternRewriter &rewriter,
                   llvm::SmallDenseMap<Value, BlockData> &known);

  static LogicalResult
  rewriteCustomOp(hivm::CustomOp op, hivm::CustomOp::Adaptor &adaptor,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, BlockData> &known);

  template <typename T>
  static std::enable_if_t<std::is_same_v<T, scf::YieldOp> ||
                              std::is_same_v<T, scf::ConditionOp>,
                          LogicalResult>
  rewriteTerminator(T op, ConversionPatternRewriter &rewriter,
                    const llvm::SmallDenseSet<size_t> &blockArgIdxSet,
                    ArrayRef<int64_t> iterArgIdxMap,
                    const llvm::SmallDenseMap<Value, BlockData> &known);

  /// @param known is mainly designed for `rewriteLoop`, and is just non-const
  /// in `rewriteLoop`, `rewriteAddPtr` and `rewriteAdvance`
  static LogicalResult
  rewriteLoopOp(LoopLikeOpInterface op, ConversionPatternRewriter &rewriter,
                llvm::SmallDenseMap<Value, BlockData> &known,
                ArrayRef<unsigned> onlyIndexTensorSlots = {});

  static LogicalResult rewriteAddPtrToUnstrucMemAcc(
      triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
      ConversionPatternRewriter &rewriter, BlockData &data);
};

template <typename OpTy>
void parseIndirectLoad(OpTy op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known,
                       unsigned resultIdx = 0);

} // namespace triton

} // namespace mlir

#endif
