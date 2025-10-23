#ifndef TRITON_ANALYSIS_BLOCKPTRANALYSIS_H
#define TRITON_ANALYSIS_BLOCKPTRANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <set>
namespace mlir {

class ConversionPatternRewriter;

namespace triton {

enum class MemAccVal { Undefined = 0, StrucMemAcc = 1, UnstrucMemAcc = 2 };

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
  MemRefType getResultMemrefType(int64_t offset,
                                 ArrayRef<int64_t> resultShape) const;

  void addBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);
  void mulBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);
  void divBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                ConversionPatternRewriter &rewriter);

  memref::ReinterpretCastOp createCastOp(ArrayRef<int64_t> resultShape,
                                         const Location &loc,
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
  static Value getScalarMemRef(Value ptr, Value memref, const Location &loc,
                               ConversionPatternRewriter &rewriter);

  static void parse(Value operand, BlockData &data, const Location &loc,
                    ConversionPatternRewriter &rewriter,
                    const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseAdd(arith::AddIOp op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseMul(arith::MulIOp op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseDiv(arith::DivSIOp op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseRem(arith::RemSIOp op, BlockData &data, const Location &loc,
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

  static void
  parseExpandDims(triton::ExpandDimsOp op, BlockData &data, const Location &loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseBitcast(triton::BitcastOp op, BlockData &data,
                           const Location &loc,
                           ConversionPatternRewriter &rewriter,
                           const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseExtSI(arith::ExtSIOp op, BlockData &data,
                         const Location &loc,
                         ConversionPatternRewriter &rewriter,
                         const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseBroadcast(triton::BroadcastOp op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseSplat(triton::SplatOp op, BlockData &data,
                         const Location &loc,
                         ConversionPatternRewriter &rewriter,
                         const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseConstSplat(arith::ConstantOp op, BlockData &data, const Location &loc,
                  ConversionPatternRewriter &rewriter,
                  const llvm::SmallDenseMap<Value, BlockData> &known);

  template <typename T>
  static std::enable_if_t<std::is_same_v<T, triton::MakeTensorPtrOp> ||
                          std::is_same_v<T, triton::AdvanceOp>>
  parseTensorPtr(T op, BlockData &data, const Location &loc,
                 ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseAddPtr(triton::AddPtrOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseExtractSlice(tensor::ExtractSliceOp op, BlockData &data,
                            const Location &loc,
                            ConversionPatternRewriter &rewriter,
                            const llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  parseReinterpretCast(memref::ReinterpretCastOp op, BlockData &data,
                       const Location &loc, ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

  static void parseReduce(triton::ReduceOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known);

  static void rewriteAddPtr(triton::AddPtrOp op,
                            triton::AddPtrOp::Adaptor &adaptor,
                            ConversionPatternRewriter &rewriter,
                            llvm::SmallDenseMap<Value, BlockData> &known);

  static void rewriteMakeTensorPtrOp(triton::MakeTensorPtrOp op,
                                     Value base,
                                     ConversionPatternRewriter &rewriter,
                                     llvm::SmallDenseMap<Value, BlockData> &known);

  static void rewriteAdvanceOp(triton::AdvanceOp op,
                               ConversionPatternRewriter &rewriter,
                               llvm::SmallDenseMap<Value, BlockData> &known);

  static void
  rewriteYieldOp(scf::YieldOp op, ConversionPatternRewriter &rewriter,
                 const llvm::SmallDenseSet<size_t> &blockArgIdxSet, ArrayRef<int64_t> iterArgIdxMap,
                 const llvm::SmallDenseMap<Value, BlockData> &known);

  /// @param known is mainly designed for `rewriteLoop`, and is just non-const in
  /// `rewriteLoop`, `rewriteAddPtr` and `rewriteAdvance`
  static void rewriteLoopOp(LoopLikeOpInterface op, ConversionPatternRewriter &rewriter,
                            llvm::SmallDenseMap<Value, BlockData> &known);

  static void rewriteAddPtrToUnstrucMemAcc(triton::AddPtrOp op,
                                           triton::AddPtrOp::Adaptor &adaptor,
                                           ConversionPatternRewriter &rewriter,
                                           BlockData &data);
};

template <typename OpTy>
void parseIndirectLoad(OpTy op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known);

} // namespace triton

} // namespace mlir

#endif