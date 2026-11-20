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

#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/TritonControlFlowOpt/ControlFlowRewrite.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Utils/DebugUtils.h"
#include "ascend/include/Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>
#include <limits>
#include <set>

#define DEBUG_TYPE "triton-block-ptr-analysis"
namespace mlir {
namespace triton {

hivm::PointerCastOp createScalarPointerCast(OpBuilder &builder, Location loc,
                                            MemRefType resultType,
                                            Value address) {
  SmallVector<Value> dynamicSizes;
  if (resultType.getNumDynamicDims() != 0) {
    Value defaultSize = builder.create<arith::ConstantIndexOp>(loc, 1);
    dynamicSizes.assign(resultType.getNumDynamicDims(), defaultSize);
  }
  auto pointerCast = builder.create<hivm::PointerCastOp>(
      loc, resultType, ValueRange{address}, ValueRange{dynamicSizes});
  pointerCast->setAttr(kScalarPointerCarrierAttr,
                       UnitAttr::get(builder.getContext()));
  return pointerCast;
}

// Recognize original scalar-pointer producers whose converted result may act
// as a memref carrier. The parse site still requires BaseMemRefType, so merely
// appearing in this list never makes an unconverted pointer an opaque source.
static bool isScalarPointerTransport(Operation *op) {
  return op && isa<scf::IfOp, scf::ForOp, scf::WhileOp, arith::SelectOp>(op);
}

// Returns true only for sources known to carry a complete scalar integer
// address. Before dialect conversion the source is tt.int_to_ptr; after
// IntToPtrConverter it is a memref-producing HIVM PointerCast with explicit
// ScalarPointerCarrier provenance. Ordinary PointerCast descriptors must keep
// using the existing unstructured fallback.
static bool isScalarPointerCarrierSource(Value source) {
  if (!source)
    return false;
  if (source.getDefiningOp<triton::IntToPtrOp>())
    return true;
  auto pointerCast = source.getDefiningOp<hivm::PointerCastOp>();
  return pointerCast && pointerCast->hasAttr(kScalarPointerCarrierAttr);
}

// Dialect conversion may temporarily materialize a scalar Triton pointer from
// a memref carrier so that an operation which has not been rewritten yet can
// keep its original type.  Scalar pointer transport must consume the carrier
// directly; otherwise the conversion driver is forced to leave a live
// memref-to-!tt.ptr cast next to tt.load/tt.store.  Only peel the narrow,
// one-to-one UCC form whose input is already a BaseMemRefType.  Unknown casts,
// multi-value casts, and non-memref inputs remain unsupported and therefore
// continue through the existing diagnostics.
static Value unwrapScalarPointerMemRefCarrier(Value value) {
  while (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() != 1 || castOp.getOutputs().size() != 1)
      break;
    Value input = castOp.getInputs().front();
    Value output = castOp.getOutputs().front();
    if (!isa<BaseMemRefType>(input.getType()) ||
        !isa<triton::PointerType, BaseMemRefType>(output.getType()))
      break;
    value = input;
  }
  return value;
}

// MemAccType selectMaxMemAccTy(const MemAccType &v1, const MemAccType &v2) {
//   return (v1 > v2) ? v1 : v2;
// }

SmallVector<OpFoldResult> &BlockData::getOffsetsRef() { return this->offsets; }

SmallVector<OpFoldResult> &BlockData::getSizesRef() { return this->sizes; }

SmallVector<OpFoldResult> &BlockData::getStridesRef() { return this->strides; }

Value &BlockData::getSourceRef() { return this->source; }

OpFoldResult &BlockData::getScalarRef() { return this->scalar; }

SmallVector<OpFoldResult> BlockData::getOffsets() const {
  return this->offsets;
}

SmallVector<OpFoldResult> BlockData::getSizes() const { return this->sizes; }

SmallVector<OpFoldResult> BlockData::getStrides() const {
  return this->strides;
}

OpFoldResult BlockData::getOffset(int index) const {
  return this->offsets[index];
}

OpFoldResult BlockData::getSize(int index) const { return this->sizes[index]; }

OpFoldResult BlockData::getStride(int index) const {
  return this->strides[index];
}

OpFoldResult BlockData::getScalar() const { return this->scalar; }

Value BlockData::getSource() const { return this->source; }

MemAccType BlockData::getMemAccType() const { return this->memAccTy; };

MemAccType &BlockData::getMemAccTypeRef() { return this->memAccTy; };

bool BlockData::isScalar() const { return !(this->scalar).isNull(); }

bool BlockData::isEmpty() const {
  return !(this->getRank() || this->source || !(this->scalar).isNull());
}

bool BlockData::hasSource() const { return this->source != nullptr; }

void BlockData::removeSource() { this->source = nullptr; };

bool BlockData::hasResElemTy() const { return this->resElemTy != nullptr; }

Type &BlockData::getResElemTyRef() { return this->resElemTy; }

Type BlockData::getResElemTy() const { return this->resElemTy; }

int64_t BlockData::getRank() const {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size());
  return this->offsets.size();
}

void BlockData::setResElemTy(const Type &Ty) { this->resElemTy = Ty; }

void BlockData::setScalar(const OpFoldResult &scalar) { this->scalar = scalar; }

void BlockData::setSource(const Value &src) { this->source = src; }

void BlockData::setOffsets(const SmallVector<OpFoldResult> &offsets) {
  this->offsets = offsets;
}

void BlockData::setStrides(const SmallVector<OpFoldResult> &strides) {
  this->strides = strides;
}

void BlockData::setSizes(const SmallVector<OpFoldResult> &szs) {
  this->sizes = szs;
}

void BlockData::setMemAccTy(const MemAccType &v) { this->memAccTy = v; }

void BlockData::setMemAccVal(const MemAccVal v) { this->memAccTy.value = v; }

OpFoldResult BlockData::inferBlockOffset(const Location &loc,
                                         OpBuilder &builder) const {
  OpFoldResult retOffset = builder.getIndexAttr(0);
  for (auto ofr : offsets) {
    retOffset =
        addOpFoldResult(retOffset, ofr, loc, builder, builder.getIndexType());
  }
  return retOffset;
}

FailureOr<MemRefType>
BlockData::getResultMemrefType(int64_t offset,
                               ArrayRef<int64_t> resultShape) const {
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  if (!this->source)
    return failure();
  auto baseMemrefType = dyn_cast<BaseMemRefType>(this->source.getType());
  if (!baseMemrefType)
    return failure();
  auto elementType = baseMemrefType.getElementType();
  auto layout =
      StridedLayoutAttr::get(this->source.getContext(), offset, staticStrides);
  return MemRefType::get(resultShape, elementType, layout);
}

void BlockData::addBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());
  // When both left block and right block have source, it is indirect load.
  assert(!(lBlock.hasSource() && rBlock.hasSource()) &&
         "Don't support each BlockData has own base source pointer");
  this->source =
      lBlock.hasSource() ? lBlock.getSourceRef() : rBlock.getSourceRef();

  assert(!(lBlock.hasResElemTy() && rBlock.hasResElemTy()));
  if (lBlock.hasResElemTy()) {
    assert(lBlock.hasSource());
    this->resElemTy = lBlock.getResElemTyRef();
  } else if (rBlock.hasResElemTy()) {
    assert(rBlock.hasSource());
    this->resElemTy = rBlock.getResElemTyRef();
  }

  // Acctually `scalar` should be accumulated into `offset` and `stride` finally
  // In addBlock, just pass `scalar` when:
  // 1. both lhs and rhs have `scalar`
  // 2. otherwise, both lhs and rhs are scalar type with rank 0
  // Except above, original `scalar` has been fused into `offset` under add.
  if (lBlock.isScalar() && rBlock.isScalar()) {
    auto addScalar =
        addOpFoldResult(lBlock.getScalarRef(), rBlock.getScalarRef(), loc,
                        rewriter, rewriter.getIndexType());
    this->scalar = addScalar;
  } else if (lBlock.getRank() == 0) {
    // When both lhs and rhs are scalar type with rank 0, just try passing
    // potential `scalar`
    this->scalar =
        lBlock.isScalar() ? lBlock.getScalarRef() : rBlock.getScalarRef();
  }

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(addOpFoldResult(lOffset, rOffset, loc, rewriter,
                                            rewriter.getIndexType()));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(addOpFoldResult(lStride, rStride, loc, rewriter,
                                            rewriter.getIndexType()));
  }

  // Both sizes are same implicitly under `add`
  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

void BlockData::subBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());

  if (lBlock.isScalar() && rBlock.isScalar()) {
    auto subScalar = subOpFoldResult(lBlock.getScalarRef(),
                                     rBlock.getScalarRef(), loc, rewriter);
    this->scalar = subScalar;
  } else if (lBlock.getRank() == 0) {
    // When both lhs and rhs are scalar type with rank 0, just try passing
    // potential `scalar`
    this->scalar =
        lBlock.isScalar() ? lBlock.getScalarRef() : rBlock.getScalarRef();
  }

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(subOpFoldResult(lOffset, rOffset, loc, rewriter));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(subOpFoldResult(lStride, rStride, loc, rewriter));
  }

  // Both sizes are same implicitly under `sub`
  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

void BlockData::mulBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());

  assert(!(lBlock.hasSource() && rBlock.hasSource()));

  if (lBlock.isScalar() && rBlock.isScalar()) {
    LLVM_DEBUG({
      llvm::dbgs() << "lBlock.scalar:" << lBlock.getScalar()
                   << " rBlbock.scalar:" << rBlock.getScalar() << "\n";
    });

    auto scalar = mulOpFoldResult(lBlock.getScalar(), rBlock.getScalar(), loc,
                                  rewriter, rewriter.getIndexType());
    this->scalar = scalar;
  }

  // assert(
  //     (lBlock.isScalar() ^ rBlock.isScalar()) &&
  //     "Currently only support one and only one scalar in function
  //     mulBlock()");

  BlockData *lb = &lBlock;
  BlockData *rb = &rBlock;
  if (lb->isScalar()) {
    std::swap(lb, rb);
  }

  // In mulBlock, `scalar` will be accumulated into `offset` and `stride`
  OpFoldResult rScalar = rb->getScalarRef();
  for (const auto &lOffset : lb->getOffsetsRef()) {
    this->offsets.push_back(mulOpFoldResult(lOffset, rScalar, loc, rewriter,
                                            rewriter.getIndexType()));
  }

  for (const auto &lStride : lb->getStridesRef()) {
    this->strides.push_back(mulOpFoldResult(lStride, rScalar, loc, rewriter,
                                            rewriter.getIndexType()));
  }

  this->sizes = lb->getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

void BlockData::divBlock(BlockData &lBlock, BlockData &rBlock, Location loc,
                         ConversionPatternRewriter &rewriter) {
  assert(this->isEmpty() && lBlock.getRank() == rBlock.getRank());

  assert(!(lBlock.hasSource() && rBlock.hasSource()));
  assert(lBlock.isScalar() && rBlock.isScalar());

  auto rScalar = rBlock.getScalar();
  this->scalar = divOpFoldResult(lBlock.getScalar(), rScalar, loc, rewriter);

  for (auto lOffset : lBlock.getOffsetsRef()) {
    this->offsets.push_back(divOpFoldResult(lOffset, rScalar, loc, rewriter));
  }

  for (auto lStride : lBlock.getStridesRef()) {
    this->strides.push_back(divOpFoldResult(lStride, rScalar, loc, rewriter));
  }

  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

FailureOr<memref::ReinterpretCastOp>
BlockData::createCastOp(ArrayRef<int64_t> resultShape, const Location &loc,
                        OpBuilder &builder) const {
  OpFoldResult resOffset = this->inferBlockOffset(loc, builder);
  int64_t staticOffset = ShapedType::kDynamic;
  if (isa<Attribute>(resOffset)) {
    auto constantOffset = getConstantIntValue(resOffset);
    if (!constantOffset)
      return failure();
    staticOffset = *constantOffset;
  }
  FailureOr<MemRefType> resultType =
      this->getResultMemrefType(staticOffset, resultShape);
  if (failed(resultType))
    return failure();

  SmallVector<OpFoldResult> strides(this->strides);
  for (size_t i = 0; i < strides.size(); i++) {
    if (resultShape[i] == 1) {
      if (auto strideValue = dyn_cast<Value>(strides[i])) {
        auto oneIdx =
            builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
        strides[i] = builder.create<arith::MaxSIOp>(loc, strideValue, oneIdx)
                         .getResult();
      }
    }
  }

  return builder.create<memref::ReinterpretCastOp>(
      loc, *resultType, this->source, resOffset, this->sizes, strides);
}

void BlockData::dump() const {
  llvm::outs() << "[INFO][BEG] BlockData info\n";
  llvm::outs() << "offsets has " << offsets.size() << " items\n";
  int cnt = 0;
  for (auto it = offsets.begin(); it != offsets.end(); ++it) {
    llvm::outs() << "offsets[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "sizes has " << sizes.size() << " items\n";
  cnt = 0;
  for (auto it = sizes.begin(); it != sizes.end(); ++it) {
    llvm::outs() << "sizes[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "strides has " << strides.size() << " items\n";
  cnt = 0;
  for (auto it = strides.begin(); it != strides.end(); ++it) {
    llvm::outs() << "strides[" << cnt++ << "] = " << *it << "\n";
  }
  llvm::outs() << "source = " << source << "\n";
  llvm::outs() << "scalar = " << scalar << "\n";
  llvm::outs() << "resElemTy = " << resElemTy << "\n";
  llvm::outs() << "memAccTy = " << memAccTy.toString() << "\n";
  llvm::outs() << "[INFO][END] BlockData info\n";
}

FailureOr<Value>
BlockDataParser::getScalarMemRef(Value ptr, Value memref, const Location &loc,
                                 ConversionPatternRewriter &rewriter) {
  if (!ptr || !memref)
    return failure();
  auto pointerType = dyn_cast<triton::PointerType>(ptr.getType());
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return failure();

  memref = unwrapScalarPointerMemRefCarrier(memref);

  // A complete scalar address reconstructed after an SCF boundary is exposed
  // as tt.int_to_ptr in the Triton IR.  Its converted operand is already the
  // canonical ScalarPointerCarrier memref; handle it before the legacy
  // block-argument path so a direct load/store does not request a reverse
  // memref-to-pointer materialization.
  if (ptr.getDefiningOp<triton::IntToPtrOp>()) {
    if (!isa<BaseMemRefType>(memref.getType()))
      return failure();
    // IntToPtrConverter's rank-1 carrier is still a dynamic base memref, not
    // the canonical scalar view consumed by indirect loads.  Normalize every
    // int_to_ptr carrier to the one-element identity view below.
    BlockData data;
    data.setSource(memref);
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    data.getSizesRef().push_back(rewriter.getIndexAttr(1));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
    auto castOp = data.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
    if (failed(castOp))
      return failure();
    return (*castOp).getResult();
  }

  if (ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>())
      return castOp.getResult();
    if (auto pointerCast = memref.getDefiningOp<hivm::PointerCastOp>();
        pointerCast && pointerCast->hasAttr(kScalarPointerCarrierAttr))
      return memref;
    return failure();
  }

  // A scalar pointer produced by structured control flow or select is already
  // represented by a converted memref. Give it the same one-element view used
  // for a scalar block argument so direct tt.load/tt.store users can consume
  // the transport result without attempting memref-to-pointer materialization.
  if (auto definingOp = ptr.getDefiningOp();
      definingOp && isScalarPointerTransport(definingOp)) {
    if (!isa<BaseMemRefType>(memref.getType()))
      return failure();
    if (auto memrefType = dyn_cast<MemRefType>(memref.getType());
        memrefType && memrefType.getRank() == 1)
      return memref;
    BlockData data;
    data.setSource(memref);
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    data.getSizesRef().push_back(rewriter.getIndexAttr(1));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
    auto castOp = data.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
    if (failed(castOp))
      return failure();
    return (*castOp).getResult();
  }

  if (!isa<BlockArgument>(ptr) || !isa<BaseMemRefType>(memref.getType()))
    return failure();

  BlockData data;
  data.setSource(memref);
  data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
  data.getSizesRef().push_back(rewriter.getIndexAttr(1));
  data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  auto castOp = data.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
  if (failed(castOp))
    return failure();
  return (*castOp).getResult();
}

LogicalResult
BlockDataParser::parse(Value operand, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known) {
  if (known.find(operand) != known.end()) {
    data = known.lookup(operand);
    return success();
  }

  if (isa<IntegerType>(operand.getType())) {
    data.setScalar(getOpFoldResultOfLayoutInfo(operand, rewriter));
    return success();
  }

  //
  if (isa<triton::PointerType>(operand.getType())) {
    // Just consider two state: ptr<scalar> and ptr<tensor<scalar>>
    Value remappedPtr = rewriter.getRemappedValue(operand);
    if (!remappedPtr) {
      if (Operation *definingOp = operand.getDefiningOp())
        return definingOp->emitError("scalar pointer has no converted value");
      emitError(loc) << "scalar pointer block argument has no converted value: "
                     << operand;
      return failure();
    }
    // A not-yet-rewritten pointer user may be represented by a one-to-one
    // UCC from a converted memref.  Consume that memref as the transport
    // source instead of asking the driver to materialize the pointer back.
    remappedPtr = unwrapScalarPointerMemRefCarrier(remappedPtr);
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        return parseAddPtr(addPtrOp, data, loc, rewriter, known);
      } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(op)) {
        return parseBitcast(bitcastOp, data, loc, rewriter, known);
      } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        return parseTensorPtr(makeTensorPtrOp, data, loc, rewriter, known);
      } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
        // To support
        // ptr_0 = tl.advance(ptr)
        // ptr_1 = tl.advance(ptr_0)
        return parseTensorPtr(advanceOp, data, loc, rewriter, known);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        if (!isa<BaseMemRefType>(remappedPtr.getType())) {
          return op->emitError(
              "int_to_ptr did not convert to a memref carrier");
        }
        data.setSource(remappedPtr);
        // An address reconstructed from an i64 is a complete scalar pointer,
        // but AddPtr still needs the ordinary one-element BlockData schema to
        // form a memref.reinterpret_cast for a later load/store.  Without
        // this identity view the generic path sees an empty rank and may
        // attempt to materialize a pointer from a non-memref source.
        data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
        data.getSizesRef().push_back(rewriter.getIndexAttr(1));
        data.getStridesRef().push_back(rewriter.getIndexAttr(1));
      } else if (isDistributedTypeCustomOp(op)) {
        data.setSource(remappedPtr);
      } else if (isScalarPointerTransport(op)) {
        if (!isa<BaseMemRefType>(remappedPtr.getType())) {
          return op->emitError(
              "scalar pointer transport did not convert to a memref");
        }
        data.setSource(remappedPtr);
        // Transport producers carry a complete scalar address but do not
        // expose BlockData dimensions. Model the address as a one-element
        // identity view so a following addptr can add a scalar offset without
        // falling into the rank-mismatch/materialization failure path.
        data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
        data.getSizesRef().push_back(rewriter.getIndexAttr(1));
        data.getStridesRef().push_back(rewriter.getIndexAttr(1));
      } else {
        return op->emitError()
               << "unsupported scalar pointer producer '" << op->getName()
               << "' with original type " << operand.getType()
               << " and converted type " << remappedPtr.getType();
      }
    } else {
      data.setSource(remappedPtr);
    }
    return success();
  }

  // not a scalar pointer
  if (auto addOp = operand.getDefiningOp<arith::AddIOp>()) {
    return parseAdd(addOp, data, loc, rewriter, known);
  } else if (auto subOp = operand.getDefiningOp<arith::SubIOp>()) {
    return parseSub(subOp, data, loc, rewriter, known);
  } else if (auto mulOp = operand.getDefiningOp<arith::MulIOp>()) {
    return parseMul(mulOp, data, loc, rewriter, known);
  } else if (auto addPtrOp = operand.getDefiningOp<triton::AddPtrOp>()) {
    return parseAddPtr(addPtrOp, data, loc, rewriter, known);
  } else if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
    parseConstSplat(constOp, data, loc, rewriter, known);
    return success();
  } else if (auto broadcastOp = operand.getDefiningOp<triton::BroadcastOp>()) {
    return parseBroadcast(broadcastOp, data, loc, rewriter, known);
  } else if (auto splatOp = operand.getDefiningOp<triton::SplatOp>()) {
    return parseSplat(splatOp, data, loc, rewriter, known);
  } else if (auto expandDimsOp =
                 operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return parseExpandDims(expandDimsOp, data, loc, rewriter, known);
  } else if (auto remOp = operand.getDefiningOp<arith::RemSIOp>()) {
    return parseRem(remOp, data, loc, rewriter, known);
  } else if (auto bitcastOp = operand.getDefiningOp<triton::BitcastOp>()) {
    return parseBitcast(bitcastOp, data, loc, rewriter, known);
  } else if (auto extsiOp = operand.getDefiningOp<arith::ExtSIOp>()) {
    return parseExtSI(extsiOp, data, loc, rewriter, known);
  } else if (auto divOp = operand.getDefiningOp<arith::DivSIOp>()) {
    return parseDiv(divOp, data, loc, rewriter, known);
  } else if (auto makeRangeOp = operand.getDefiningOp<triton::MakeRangeOp>()) {
    parseMakeRange(makeRangeOp, data, loc, rewriter, known);
    return success();
  } else if (auto reduceOp = operand.getDefiningOp<triton::ReduceOp>()) {
    return parseReduce(reduceOp, data, loc, rewriter, known);
  } else if (auto loadOp = operand.getDefiningOp<triton::LoadOp>()) {
    parseIndirectLoad<triton::LoadOp>(loadOp, data, loc, rewriter, known);
    return success();
  } else if (auto castOp = operand.getDefiningOp<arith::FPToSIOp>()) {
    parseIndirectLoad<arith::FPToSIOp>(castOp, data, loc, rewriter, known);
    return success();
  } else if (auto extractSliceOp =
                 operand.getDefiningOp<tensor::ExtractSliceOp>()) {
    return parseExtractSlice(extractSliceOp, data, loc, rewriter, known);
  } else if (auto forOp = operand.getDefiningOp<scf::ForOp>()) {
    auto opResult = dyn_cast<OpResult>(operand);
    if (!opResult)
      return forOp.emitOpError("expected an OpResult while parsing its result");
    unsigned resultIdx = opResult.getResultNumber();
    parseIndirectLoad<scf::ForOp>(forOp, data, loc, rewriter, known, resultIdx);
    return success();
  } else if (auto tensorCastOp = operand.getDefiningOp<tensor::CastOp>()) {
    // Used for identity operation.
    return parse(tensorCastOp.getSource(), data, loc, rewriter, known);
  } else if (auto fillOp = operand.getDefiningOp<linalg::FillOp>()) {
    return parseFill(fillOp, data, loc, rewriter, known);
  } else if (auto selectOp = operand.getDefiningOp<arith::SelectOp>()) {
    if (auto resultType = dyn_cast<ShapedType>(selectOp.getType());
        resultType && isa<triton::PointerType>(resultType.getElementType()))
      return selectOp.emitOpError(
          "tensor-of-pointers select must be lowered before BlockData parsing");
    return parseSelect(selectOp, data, loc, rewriter, known);
  } else if (isDistributedTypeCustomOp(operand.getDefiningOp())) {
    auto opResult = dyn_cast<OpResult>(operand);
    if (!opResult)
      return emitError(loc, "expected a custom operation result");
    return parseStructuredCustomOp(operand.getDefiningOp(), data, loc, rewriter,
                                   known, opResult.getResultNumber());
  } else if (auto genericOp = operand.getDefiningOp<linalg::GenericOp>()) {
    if (genericOp->hasAttr("tt.from_make_range")) {
      parseLinalgGenericFromMakeRange(genericOp, data, loc, rewriter, known);
      return success();
    }
    return genericOp.emitOpError(
        "cannot parse a generic operation without tt.from_make_range");
  } else if (auto atomicRMWOp = operand.getDefiningOp<triton::AtomicRMWOp>()) {
    parseAtomicRmw(atomicRMWOp, data, loc, rewriter, known);
    return success();
  }

  if (Operation *producer = operand.getDefiningOp())
    return producer->emitError()
           << "unsupported BlockData producer '" << producer->getName()
           << "' with result type " << operand.getType();
  emitError(loc) << "unsupported BlockData block argument of type "
                 << operand.getType();
  return failure();
}

void BlockDataParser::parseAtomicRmw(
    triton::AtomicRMWOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  auto opRes = op->getResult(0);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    resShape = shapedResTy.getShape().vec();
    if (resShape.size() == 1 && resShape[0] == 1) {
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value extracted =
          rewriter.create<tensor::ExtractOp>(loc, opRes, ValueRange{zeroIdx});
      Value scalarIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), extracted);
      data.setMemAccVal(MemAccVal::StrucMemAcc);
      data.setScalar(scalarIdx);
      data.getSizesRef().push_back(rewriter.getIndexAttr(1));
      data.getStridesRef().push_back(rewriter.getIndexAttr(0));
      data.getOffsetsRef().push_back(scalarIdx);
      return;
    }
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    data.setMemAccVal(MemAccVal::UnstrucMemAcc);
  } else {
    data.setMemAccVal(MemAccVal::StrucMemAcc);
    resShape.push_back(1);
  }
  for (auto &s : resShape) {
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    data.getSizesRef().push_back(rewriter.getIndexAttr(s));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  }
  data.setSource(opRes);
}

LogicalResult
BlockDataParser::parseAdd(arith::AddIOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  if (failed(parse(op.getLhs(), lBlock, loc, rewriter, known)) ||
      failed(parse(op.getRhs(), rBlock, loc, rewriter, known)))
    return failure();
  data.addBlock(lBlock, rBlock, loc, rewriter);
  return success();
}

LogicalResult
BlockDataParser::parseSub(arith::SubIOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  if (failed(parse(op.getLhs(), lBlock, loc, rewriter, known)) ||
      failed(parse(op.getRhs(), rBlock, loc, rewriter, known)))
    return failure();
  data.subBlock(lBlock, rBlock, loc, rewriter);
  return success();
}

LogicalResult
BlockDataParser::parseMul(arith::MulIOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  if (failed(parse(op.getLhs(), lBlock, loc, rewriter, known)) ||
      failed(parse(op.getRhs(), rBlock, loc, rewriter, known)))
    return failure();

  data.mulBlock(lBlock, rBlock, loc, rewriter);
  return success();
}

LogicalResult
BlockDataParser::parseDiv(arith::DivSIOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  if (failed(parse(op.getLhs(), lBlock, loc, rewriter, known)) ||
      failed(parse(op.getRhs(), rBlock, loc, rewriter, known)))
    return failure();
  data.divBlock(lBlock, rBlock, loc, rewriter);
  return success();
}

// TODO : support modulos
LogicalResult
BlockDataParser::parseRem(arith::RemSIOp op, BlockData &data,
                          const Location &loc,
                          ConversionPatternRewriter &rewriter,
                          const llvm::SmallDenseMap<Value, BlockData> &known) {
  return op.emitOpError(
      "address expressions with modulo are not supported by BlockDataParser");
}

void BlockDataParser::parseMakeRange(
    triton::MakeRangeOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  insertDebugNop(op.getLoc(), rewriter);
  assert(data.isEmpty());
  auto shape = dyn_cast<ShapedType>(op.getType()).getShape();

  auto start = op.getStart();
  auto end = op.getEnd();
  auto stride = (end >= start) && (end - start <= shape[0]);
  assert(stride == 1 &&
         "make_range op should always return a tensor of stride 1");

  data.getOffsetsRef().push_back(rewriter.getIndexAttr(start));
  data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
  data.getStridesRef().push_back(rewriter.getIndexAttr(stride));
}

void BlockDataParser::parseLinalgGenericFromMakeRange(
    linalg::GenericOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  assert(op->hasAttr("tt.from_make_range") &&
         "expected tt.from_make_range attribute");

  auto offsetAttr = op->getAttr("tt.make_range_offset");
  auto sizeAttr = op->getAttr("tt.make_range_size");
  assert(offsetAttr && sizeAttr &&
         "tt.make_range_offset and tt.make_range_size required");

  int64_t offset = cast<IntegerAttr>(offsetAttr).getInt();
  int64_t size = cast<IntegerAttr>(sizeAttr).getInt();

  data.getOffsetsRef().push_back(rewriter.getIndexAttr(offset));
  data.getSizesRef().push_back(rewriter.getIndexAttr(size));
  data.getStridesRef().push_back(rewriter.getIndexAttr(1));
}

LogicalResult BlockDataParser::parseExpandDims(
    triton::ExpandDimsOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  if (failed(parse(op.getSrcMutable().get(), data, loc, rewriter, known)))
    return failure();
  auto resShape = dyn_cast<ShapedType>(op.getResult().getType()).getShape();
  auto axis = op.getAxis();

  assert(resShape[axis] == 1 &&
         "The destiny shape of changed dimension should be 1");

  data.getOffsetsRef().insert(data.getOffsetsRef().begin() + axis,
                              rewriter.getIndexAttr(0));
  data.getSizesRef().insert(data.getSizesRef().begin() + axis,
                            rewriter.getIndexAttr(1));
  data.getStridesRef().insert(data.getStridesRef().begin() + axis,
                              rewriter.getIndexAttr(0));
  return success();
}

LogicalResult BlockDataParser::parseExtractSlice(
    tensor::ExtractSliceOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  const std::string scenarioMessages =
      "PtsAnalysis supports indirectly block load in the "
      "following scenario\n"
      "B = tl.load(Aptr + Aoffset) # B is 1D tensor\n"
      "s = tl.extract_slice(indices, offsets= (i,), sizes= "
      "(1,), strides= (1,)) # s is a tensor<1x$dtype>\n"
      "D = tl.load(Cptr + s + Coffset) # s is used as the "
      "scalar offset\n"; // tensor<2x$dtype> will be support
                         // soon

  auto extract_src = op->getOperand(0);
  BlockData srcBlock;
  if (failed(parse(extract_src, srcBlock, loc, rewriter, known)))
    return failure();
  if (!srcBlock.hasSource())
    return op.emitOpError(scenarioMessages);
  // Use isa_and_nonnull for LLVM 21 compatibility
  if (!isa_and_nonnull<triton::LoadOp>(srcBlock.getSource().getDefiningOp()))
    return op.emitOpError(scenarioMessages);

  auto extract_result = op->getResult(0);
  auto shaped_ty = dyn_cast<RankedTensorType>(extract_result.getType());
  auto shape = shaped_ty.getShape();
  if (shape.size() > 1 || shape[0] > 1)
    return op.emitOpError(scenarioMessages);
  auto castOp = rewriter.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(shape, rewriter.getIndexType()),
      extract_result);
  auto offset = castOp.getResult();
  if (data.isEmpty()) {
    data.getOffsetsRef().push_back(offset);
    data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  } else {
    return op.emitOpError(
        "extract_slice parsing with a pre-populated offset is unsupported");
  }
  return success();
}

LogicalResult BlockDataParser::parseBitcast(
    triton::BitcastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  if (failed(parse(op.getSrc(), data, loc, rewriter, known)))
    return failure();

  auto resType = op.getResult().getType();
  Type resElemPointeeTy = nullptr;
  if (auto resShapedTy = dyn_cast<ShapedType>(resType)) {
    auto resElemTy = resShapedTy.getElementType();
    resElemPointeeTy =
        dyn_cast<triton::PointerType>(resElemTy).getPointeeType();
  } else {
    auto srcPointeeType =
        cast<triton::PointerType>(op.getSrc().getType()).getPointeeType();
    auto resPointeeType = cast<triton::PointerType>(resType).getPointeeType();

    // Handling special case
    // If Op is MetaUse or src is i1 block argument and dst is i8,
    // it should be converted to UnrealizedConversionCast
    if (op->hasAttr("MetaUse") ||
        (isa<BlockArgument>(op.getSrc()) &&
         srcPointeeType == rewriter.getIntegerType(1) &&
         resPointeeType == rewriter.getIntegerType(8))) {
      resElemPointeeTy = resPointeeType;
    } else {
      auto remappedValue = rewriter.getRemappedValue(op);
      if (!remappedValue)
        return op.emitOpError("bitcast result has no converted value");
      data.setSource(remappedValue);
      LLVM_DEBUG({
        llvm::dbgs() << "Remapping bitcastOp:\n";
        llvm::dbgs() << op << "\nto \n";
        llvm::dbgs() << remappedValue << "\n";
      });
    }
  }
  data.setResElemTy(resElemPointeeTy);
  return success();
}

LogicalResult BlockDataParser::parseExtSI(
    arith::ExtSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  return parse(op.getIn(), data, loc, rewriter, known);
}

LogicalResult BlockDataParser::parseBroadcast(
    triton::BroadcastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  auto src = op.getSrcMutable().get();
  auto dst = op.getResult();
  assert(isa<ShapedType>(src.getType()) &&
         "tt.broadcast's input should be a tensor");

  auto srcShape = dyn_cast<ShapedType>(src.getType()).getShape();
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();
  assert(srcShape.size() == dstShape.size() &&
         "rank of source shoule be equal to destnation");

  if (failed(parse(src, data, loc, rewriter, known)))
    return failure();

  for (const auto &[idx, src_dst] :
       llvm::enumerate(llvm::zip(srcShape, dstShape))) {
    const auto &[srcAxis, dstAxis] = src_dst;
    if (srcAxis == dstAxis) {
      continue;
    }
    assert(srcAxis < dstAxis &&
           "srcShape of broadcastOp must be less than dstShape.");
    data.getSizesRef()[idx] = rewriter.getIndexAttr(dstAxis);
  }
  return success();
}

LogicalResult BlockDataParser::parseSplat(
    triton::SplatOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  auto src = op.getSrc();
  auto dst = op.getResult();
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();

  if (failed(parse(src, data, loc, rewriter, known)))
    return failure();

  // A ScalarPointerCarrier stores the complete integer address passed to
  // tt.int_to_ptr, so an addptr displacement exists only in BlockData and must
  // survive the following splat. Ordinary scalar pointers already carry their
  // displacement in the converted memref descriptor; retain the established
  // behavior of resetting those offsets while constructing the tensor layout.
  OpFoldResult splatOffset;
  if (isa<triton::PointerType>(src.getType()) &&
      isScalarPointerCarrierSource(data.getSource())) {
    SmallVector<OpFoldResult> pointerOffsets = data.getOffsets();
    if (pointerOffsets.size() > 1)
      return op.emitOpError(
          "scalar pointer carrier splat requires at most one BlockData offset");
    splatOffset = pointerOffsets.empty()
                      ? OpFoldResult(rewriter.getIndexAttr(0))
                      : pointerOffsets.front();
  } else if (data.isScalar()) {
    splatOffset = data.getScalarRef();
  }

  if (isa<IntegerType>(src.getType()) ||
      isa<triton::PointerType>(src.getType())) {
    if (!data.isEmpty()) {
      data.getOffsetsRef().clear();
      data.getSizesRef().clear();
      data.getStridesRef().clear();
    }
    for (auto dstAxis : dstShape) {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
      data.getSizesRef().push_back(rewriter.getIndexAttr(dstAxis));
      data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    }
  } else {
    return op.emitOpError("BlockDataParser does not support this splat source");
  }
  if (!splatOffset.isNull())
    data.getOffsetsRef()[0] = splatOffset;
  return success();
}

void BlockDataParser::parseConstSplat(
    arith::ConstantOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  DenseElementsAttr denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
  assert(denseAttr && denseAttr.isSplat() &&
         isa<IntegerType>(denseAttr.getElementType()));

  auto innerVal = denseAttr.getValues<IntegerAttr>()[0].getValue();
  auto innerValIndexAttr = rewriter.getIndexAttr(innerVal.getSExtValue());

  // for mul state
  data.setScalar(innerValIndexAttr);

  auto resType = dyn_cast<ShapedType>(op.getResult().getType());
  size_t loopLimit = resType.getShape().size();
  for (auto i = 0; i < loopLimit; i++) {
    // Add original dense val to first dim offset for add state
    if (i == 0) {
      data.getOffsetsRef().push_back(innerValIndexAttr);
    } else {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    }
    data.getSizesRef().push_back(rewriter.getIndexAttr(resType.getShape()[i]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
  }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, triton::MakeTensorPtrOp> ||
                     std::is_same_v<T, triton::AdvanceOp>,
                 LogicalResult>
BlockDataParser::parseTensorPtr(
    T op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  Value remappedValue = rewriter.getRemappedValue(op);
  if (!remappedValue)
    return op.emitOpError("tensor pointer has no converted value");
  if (auto castOp = remappedValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    parseReinterpretCast(castOp, data, loc, rewriter, known);
    return success();
  }
  return op.emitOpError(
      "expected the converted tensor pointer to be a memref.reinterpret_cast");
}

LogicalResult BlockDataParser::parseAddPtr(
    triton::AddPtrOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  BlockData ptrBlock, offsetBlock;
  if (failed(parse(op.getPtr(), ptrBlock, op.getLoc(), rewriter, known)) ||
      failed(parse(op.getOffset(), offsetBlock, op.getLoc(), rewriter, known)))
    return failure();

  if (!ptrBlock.hasSource())
    return op.emitOpError(
        "could not resolve a source/base pointer for the addptr operand");
  // offset has source means offset is from tl.load and other ops(TODO)
  if (offsetBlock.hasSource()) {
    ptrBlock.setMemAccTy(offsetBlock.getMemAccType());
    offsetBlock.removeSource();
  }

  // handle for loop & scalar
  if (ptrBlock.getRank() == 1 && offsetBlock.getRank() == 0) {
    offsetBlock.getSizesRef().push_back(rewriter.getIndexAttr(1));
    offsetBlock.getOffsetsRef().push_back(offsetBlock.getScalarRef());
    offsetBlock.getStridesRef().push_back(rewriter.getIndexAttr(0));
  }

  if (ptrBlock.getRank() != offsetBlock.getRank())
    return op.emitOpError("pointer and offset BlockData ranks do not match");
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "[parseAddPtr][BEG] =========================\n";
    os << "[parseAddPtr] op is " << op << "\n";
    for (int i = 0; i < ptrBlock.getRank(); i++) {
      os << "ptrBlock.getOffsetsRef()[" << i
         << "] = " << ptrBlock.getOffsetsRef()[i] << "\n";
      os << "ptrBlock.getSizesRef()[" << i
         << "] = " << ptrBlock.getSizesRef()[i] << "\n";
      os << "ptrBlock.getStridesRef()[" << i
         << "] = " << ptrBlock.getStridesRef()[i] << "\n";
      os << "offsetBlock.getOffsetsRef()[" << i
         << "] = " << offsetBlock.getOffsetsRef()[i] << "\n";
      os << "offsetBlock.getSizesRef()[" << i
         << "] = " << offsetBlock.getSizesRef()[i] << "\n";
      os << "offsetBlock.getStridesRef()[" << i
         << "] = " << offsetBlock.getStridesRef()[i] << "\n";
    }
    os << "[parseAddPtr][END] -------------------------\n";
  });
  data.addBlock(ptrBlock, offsetBlock, op.getLoc(), rewriter);
  return success();
}

void BlockDataParser::parseReinterpretCast(
    memref::ReinterpretCastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  data.setOffsets(op.getMixedOffsets());
  data.setSizes(op.getMixedSizes());
  data.setStrides(op.getMixedStrides());
  data.setSource(op.getSource());

  // In memref::ReinterpretCastOp, offset means the total of collapsing multiple
  // dimensions, which corresponds to first dim offset in block data.
  // Here populate the rest of the dimensions with zeroes.
  assert(data.getOffsetsRef().size() == 1);
  size_t loopLimit = data.getSizesRef().size();
  for (size_t i = 1; i < loopLimit; i++) {
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
  }
}

LogicalResult BlockDataParser::parseReduce(
    triton::ReduceOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {

  const std::string scenarioMessages =
      "PtsAnalysis supports indirectly block load in the following scenario\n"
      "B = tl.load(Aptr + Aoffset) # B is 1D tensor\n"
      "s = tl.min(B) # s is a scalar\n"
      "D = tl.load(Cptr + s + Coffset) # s is used as the scalar offset\n";

  auto reduce_src = op->getOperand(0);
  BlockData srcBlock;
  if (failed(parse(reduce_src, srcBlock, loc, rewriter, known)))
    return failure();
  if (!srcBlock.hasSource())
    return op.emitOpError(scenarioMessages);
  // Use isa_and_nonnull for LLVM 21 compatibility
  if (!isa_and_nonnull<triton::LoadOp>(srcBlock.getSource().getDefiningOp()))
    return op.emitOpError(scenarioMessages);

  auto reduce_result = op->getResult(0);
  auto shaped_ty = dyn_cast<RankedTensorType>(reduce_result.getType());
  auto shape = shaped_ty.getShape();
  auto ops = llvm::map_to_vector(op.getBody()->without_terminator(),
                                 [](Operation &op) { return &op; });
  // Support only the case: scalar = tl.load(1D tensor)
  if (shape.size() != 1 || op.getAxis() != 0 || ops.size() != 1 ||
      !isa<arith::MinSIOp>(ops.front()))
    return op.emitOpError(scenarioMessages);

  auto castOp = rewriter.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(shape, rewriter.getIndexType()),
      reduce_result);
  auto offset = castOp.getResult();
  if (data.isEmpty()) {
    data.getOffsetsRef().push_back(offset);
    data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  } else {
    return op.emitOpError(
        "reduce parsing with a pre-populated offset is unsupported");
  }
  return success();
}

template <typename OpTy>
void parseIndirectLoad(OpTy op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known,
                       unsigned resultIdx) {
  assert(resultIdx < op->getNumResults() &&
         "resultIdx out of range for parseIndirectLoad");
  auto opRes = op->getResult(resultIdx);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    resShape = shapedResTy.getShape().vec();
    auto numOperands = 3;
    if (resShape.size() == 1 && resShape[0] == 1 &&
        op->getNumOperands() == numOperands) {
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value extracted =
          rewriter.create<tensor::ExtractOp>(loc, opRes, ValueRange{zeroIdx});
      Value scalarIdx = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), extracted);
      data.setMemAccVal(MemAccVal::StrucMemAcc);
      data.setScalar(scalarIdx);
      data.getSizesRef().push_back(rewriter.getIndexAttr(1));
      data.getStridesRef().push_back(rewriter.getIndexAttr(0));
      data.getOffsetsRef().push_back(scalarIdx);
      return;
    }
    data.setMemAccVal(MemAccVal::UnstrucMemAcc);
  } else {
    // scalar load means this is used as offset. It is StrucMemAcc.
    data.setMemAccVal(MemAccVal::StrucMemAcc);
    resShape.push_back(1);
  }
  for (auto &s : resShape) {
    data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    data.getSizesRef().push_back(rewriter.getIndexAttr(s));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  }
  // set the source in BlockData so that we know an indirect-load op exists in
  // the chain.
  data.setSource(opRes);
}

namespace {
template <typename CustomOpT>
LogicalResult
parseStructuredCustomOpImpl(CustomOpT op, BlockData &data, const Location &loc,
                            ConversionPatternRewriter &rewriter,
                            const llvm::SmallDenseMap<Value, BlockData> &known,
                            unsigned resultIdx) {
  auto srcValArrayAttr = op->template getAttrOfType<DenseI32ArrayAttr>(
      ConverterUtils::customSrcPtrIndexAttrName);
  assert(srcValArrayAttr &&
         "structure hivm custom op should present src tensor<tt.ptr>");
  auto srcValArray = srcValArrayAttr.asArrayRef();
  assert(srcValArray[resultIdx] != -1 &&
         "tensor<tt.ptr> result should map to src tensor<tt.ptr>");
  if (failed(BlockDataParser::parse(op->getOperand(srcValArray[resultIdx]),
                                    data, loc, rewriter, known)))
    return failure();
  Value remappedResult = rewriter.getRemappedValue(op->getResult(resultIdx));
  if (!remappedResult)
    return op.emitOpError("custom operation result has no converted value");
  data.setSource(remappedResult);
  return success();
}

template <typename CustomOpT>
CustomOpT createRewrittenStructuredCustomOp(
    ConversionPatternRewriter &rewriter, Location loc,
    llvm::ArrayRef<Type> resultTypes, CustomOpT op,
    typename CustomOpT::Adaptor &adaptor, ValueRange newOutputs) {
  if constexpr (std::is_same_v<CustomOpT, hivm::CustomMacroOp>) {
    return rewriter.create<hivm::CustomMacroOp>(
        loc, resultTypes, op.getName(), adaptor.getInputs(), newOutputs,
        adaptor.getTempBuffers(), adaptor.getSyncRelatedArgs());
  } else {
    return rewriter.create<hivm::CustomOp>(loc, resultTypes, op.getName(),
                                           adaptor.getInputs(), newOutputs,
                                           adaptor.getTempBuffers());
  }
}

template <typename CustomOpT>
void rewriteStructuredCustomOpImpl(CustomOpT op,
                                   typename CustomOpT::Adaptor &adaptor,
                                   ConversionPatternRewriter &rewriter) {
  if (isDistributedTypeCustomOp(op)) {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    auto loc = op.getLoc();
    llvm::SmallVector<Value> newOutputs;
    for (auto out : op.getOutputs()) {
      auto tensorTy = llvm::cast<RankedTensorType>(out.getType());
      if (llvm::isa<triton::PointerType>(tensorTy.getElementType())) {
        continue;
      }
      newOutputs.emplace_back(rewriter.getRemappedValue(out));
    }
    llvm::SmallVector<Type> resultTypes;
    for (auto ty : op->getResultTypes()) {
      if (auto ptrTy = llvm::dyn_cast<triton::PointerType>(ty)) {
        resultTypes.emplace_back(
            MemRefType::get({ShapedType::kDynamic}, ptrTy.getPointeeType()));
        continue;
      }
      if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(ty)) {
        if (auto ptrTy = llvm::dyn_cast<triton::PointerType>(
                tensorTy.getElementType())) {
          resultTypes.emplace_back(
              MemRefType::get(tensorTy.getShape(), ptrTy.getPointeeType()));
          continue;
        }
      }
      resultTypes.emplace_back(ty);
    }
    auto newOp = createRewrittenStructuredCustomOp(rewriter, loc, resultTypes,
                                                   op, adaptor, newOutputs);
    auto operandSegmentSizesAttr = newOp->getAttr("operandSegmentSizes");
    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("operandSegmentSizes", operandSegmentSizesAttr);
    rewriter.replaceOp(op, newOp.getResults());
    rewriter.restoreInsertionPoint(ip);
  } else {
    SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                  op->getResultTypes().end());
    auto newOp = createRewrittenStructuredCustomOp(
        rewriter, op.getLoc(), resultTypes, op, adaptor, adaptor.getOutputs());
    auto operandSegmentSizesAttr = newOp->getAttr("operandSegmentSizes");
    newOp->setAttrs(op->getAttrs());
    newOp->setAttr("operandSegmentSizes", operandSegmentSizesAttr);
    rewriter.replaceOp(op, newOp);
  }
}
} // namespace

void BlockDataParser::rewriteStructuredCustomOp(
    hivm::CustomOp op, hivm::CustomOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter) {
  rewriteStructuredCustomOpImpl(op, adaptor, rewriter);
}

void BlockDataParser::rewriteStructuredCustomOp(
    hivm::CustomMacroOp op, hivm::CustomMacroOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter) {
  rewriteStructuredCustomOpImpl(op, adaptor, rewriter);
}

LogicalResult BlockDataParser::parseStructuredCustomOp(
    Operation *op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known, unsigned resultIdx) {
  if (auto customOp = dyn_cast<hivm::CustomOp>(op)) {
    return parseStructuredCustomOpImpl(customOp, data, loc, rewriter, known,
                                       resultIdx);
  } else if (auto macroOp = dyn_cast<hivm::CustomMacroOp>(op)) {
    return parseStructuredCustomOpImpl(macroOp, data, loc, rewriter, known,
                                       resultIdx);
  }
  return op->emitError("expected a structured hivm custom operation");
}

void BlockDataParser::rewriteStructuredCustomOp(
    Operation *op, ConversionPatternRewriter &rewriter) {
  if (auto customOp = dyn_cast<hivm::CustomOp>(op)) {
    hivm::CustomOp::Adaptor adaptor(customOp);
    rewriteStructuredCustomOpImpl(customOp, adaptor, rewriter);
  } else if (auto macroOp = dyn_cast<hivm::CustomMacroOp>(op)) {
    hivm::CustomMacroOp::Adaptor adaptor(macroOp);
    rewriteStructuredCustomOpImpl(macroOp, adaptor, rewriter);
  } else {
    llvm_unreachable("expected hivm custom op");
  }
}

LogicalResult
BlockDataParser::parseFill(linalg::FillOp op, BlockData &data,
                           const Location &loc,
                           ConversionPatternRewriter &rewriter,
                           const llvm::SmallDenseMap<Value, BlockData> &known) {
  auto src = op.getInputs()[0];
  auto dst = op.getResult(0);
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();

  if (failed(parse(src, data, loc, rewriter, known)))
    return failure();

  if (isa<IntegerType>(src.getType())) {
    if (!data.isEmpty()) {
      data.getOffsetsRef().clear();
      data.getSizesRef().clear();
      data.getStridesRef().clear();
    }
    for (auto dstAxis : dstShape) {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
      data.getSizesRef().push_back(rewriter.getIndexAttr(dstAxis));
      data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    }
  } else {
    return op.emitOpError("BlockDataParser does not support this fill pattern");
  }
  if (data.isScalar()) {
    data.getOffsetsRef()[0] = data.getScalarRef();
  }
  return success();
}

LogicalResult BlockDataParser::parseSelect(
    arith::SelectOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  if (!data.isEmpty())
    return op.emitOpError(
        "select parsing requires an empty BlockData destination");

  auto res = op.getResult();
  auto resType = dyn_cast<ShapedType>(res.getType());
  if (!resType || (!isa<IntegerType>(resType.getElementType()) &&
                   !isa<IndexType>(resType.getElementType())))
    return op.emitOpError(
        "BlockData select requires a shaped integer or index result");

  OpFoldResult indexOfr;
  size_t loopLimit = resType.getShape().size();

  Value cond = op.getCondition();
  bool condIsScalarI1 = isa<IntegerType>(cond.getType()) &&
                        cast<IntegerType>(cond.getType()).getWidth() == 1 &&
                        !isa<ShapedType>(cond.getType());

  auto trueConst =
      dyn_cast_or_null<arith::ConstantOp>(op.getTrueValue().getDefiningOp());
  auto falseConst =
      dyn_cast_or_null<arith::ConstantOp>(op.getFalseValue().getDefiningOp());
  auto trueDense = trueConst ? dyn_cast<DenseElementsAttr>(trueConst.getValue())
                             : DenseElementsAttr();
  auto falseDense = falseConst
                        ? dyn_cast<DenseElementsAttr>(falseConst.getValue())
                        : DenseElementsAttr();

  bool denseConstCase = condIsScalarI1 && trueDense && falseDense &&
                        trueDense.isSplat() && falseDense.isSplat();

  if (denseConstCase) {
    // if cond is scalar i1 and both true and false value are splat dense const,
    // we can directly use the value of the dense const to create scalar select
    // op.
    Attribute trueFirst = *trueDense.value_begin<Attribute>();
    Attribute falseFirst = *falseDense.value_begin<Attribute>();

    Value trueScalar = nullptr;
    Value falseScalar = nullptr;
    if (auto tInt = dyn_cast<IntegerAttr>(trueFirst)) {
      trueScalar = rewriter.create<arith::ConstantOp>(loc, tInt).getResult();
    } else {
      return op.emitOpError(
          "BlockData select requires integer dense true values");
    }

    if (auto fInt = dyn_cast<IntegerAttr>(falseFirst)) {
      falseScalar = rewriter.create<arith::ConstantOp>(loc, fInt).getResult();
    } else {
      return op.emitOpError(
          "BlockData select requires integer dense false values");
    }

    if (trueScalar.getType() != falseScalar.getType())
      return op.emitOpError(
          "scalarized BlockData select values must have the same type");

    auto scalarSelect = rewriter.create<arith::SelectOp>(
        loc, trueScalar.getType(), cond, trueScalar, falseScalar);

    indexOfr = getOpFoldResultOfLayoutInfo(scalarSelect.getResult(), rewriter);
  } else {
    if (!llvm::all_of(resType.getShape(), [](int64_t dim) { return dim == 1; }))
      return op.emitOpError(
          "BlockData select supports non-splat values only when every result "
          "dimension is one");

    SmallVector<Value> indices;
    indices.reserve(loopLimit);
    for (size_t i = 0; i < loopLimit; ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    auto extractOp = rewriter.create<tensor::ExtractOp>(loc, res, indices);
    indexOfr = extractOp.getResult();
    if (isa<IntegerType>(extractOp.getType())) {
      indexOfr = getOpFoldResultOfLayoutInfo(extractOp.getResult(), rewriter);
    }
  }

  // Set scalar for mul state
  data.setScalar(indexOfr);

  for (size_t i = 0; i < loopLimit; ++i) {
    // Add scalar to first dim offset for add state
    if (i == 0) {
      data.getOffsetsRef().push_back(indexOfr);
    } else {
      data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
    }
    data.getSizesRef().push_back(rewriter.getIndexAttr(resType.getShape()[i]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
  }
  return success();
}

LogicalResult
BlockDataParser::rewriteAddPtr(triton::AddPtrOp op,
                               triton::AddPtrOp::Adaptor &adaptor,
                               ConversionPatternRewriter &rewriter,
                               llvm::SmallDenseMap<Value, BlockData> &known) {
  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  Location offLoc = op.getLoc();
  if (Value off = op.getOffset())
    if (Operation *defOp = off.getDefiningOp())
      offLoc = defOp->getLoc();
  insertDebugNop(offLoc, rewriter);
  BlockData data;
  if (failed(parseAddPtr(op, data, op.getLoc(), rewriter, known)))
    return failure();

  if (data.getMemAccTypeRef().isUnstructured() &&
      !isScalarPointerCarrierSource(data.getSource())) {
    // TODO: Based on more info, try to create a performant IR
    if (failed(rewriteAddPtrToUnstrucMemAcc(op, adaptor, rewriter, data)))
      return failure();
    LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(op) << "\n"; });
    return success();
  }

  if (data.getSizesRef().size() == 0) {
    data.getSizesRef().push_back(rewriter.getIndexAttr(1));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    data.getOffsetsRef().push_back(data.getScalarRef());
  }

  ArrayRef<int64_t> resultShape;
  // shape {1,} is stub for single ptr
  SmallVector<int64_t> stubScalarTypeShape(1, 1);
  if (auto shapedType = dyn_cast<ShapedType>(op.getResult().getType())) {
    resultShape = shapedType.getShape();
  } else {
    assert(data.getRank() == 1);
    resultShape = stubScalarTypeShape;
  }

  known[op.getResult()] = data;

  // If there are dimensions with size 1 and stride 0, replace 0 stride with the
  // product of sizes of all lower dimensions. This avoids creating memref with
  // zero stride.
  // And here store the unmodified state into known ptrs, since any following
  // pointer arithmetic operations should still use the original 0 stride.
  auto inferedSize = 1;
  auto hoistDim = op->getAttrOfType<IntegerAttr>("hoist_dim");
  for (int i = data.getSizesRef().size() - 1; i >= 0; i--) {
    auto strideConst = getConstantIntValue(data.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(data.getSizesRef()[i]);
    assert(sizeConst.has_value());
    bool shouldReplaceStride =
        (sizeConst.value() == 1) || (hoistDim && hoistDim.getValue() == i);
    if (shouldReplaceStride && strideConst && strideConst.value() == 0) {
      data.getStridesRef()[i] = rewriter.getIndexAttr(inferedSize);
    }
    inferedSize *= sizeConst.value();
  }

  auto &offsets = data.getOffsetsRef();
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (auto constVal = getConstantIntValue(offsets[i])) {
      if (constVal.value() < 0) {
        LLVM_DEBUG({
          llvm::dbgs() << "[NegOffsetElim] Detected negative offset: "
                       << constVal.value() << " at dim " << i << "\n";
        });

        Value negOffsetVal = rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), constVal.value());
        offsets[i] = negOffsetVal;
      }
    }
  }

  if (auto intToPtrOp = dyn_cast_or_null<triton::IntToPtrOp>(
          data.getSourceRef().getDefiningOp())) {
    auto rtype = cast<triton::PointerType>(intToPtrOp.getResult().getType());
    auto memrefType =
        MemRefType::get({ShapedType::kDynamic}, rtype.getPointeeType());
    auto hivmPointCastOp = createScalarPointerCast(
        rewriter, intToPtrOp.getLoc(), memrefType, intToPtrOp.getSrc());
    data.setSource(hivmPointCastOp.getResult());
  }

  if (data.hasResElemTy()) {
    // Handle bitcast scenario
    auto memrefType = dyn_cast<BaseMemRefType>(data.getSourceRef().getType())
                          .cloneWith(std::nullopt, data.getResElemTyRef());
    UnrealizedConversionCastOp castOp =
        rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), memrefType, data.getSourceRef());
    data.setSource(castOp.getOutputs()[0]);
  }

  // ToDo: need to handle module scenario

  FailureOr<memref::ReinterpretCastOp> castOp =
      data.createCastOp(resultShape, op.getLoc(), rewriter);
  if (failed(castOp))
    return op.emitOpError(
        "could not materialize a memref from the source type");
  Value src = (*castOp).getResult();
  LLVM_DEBUG({
    llvm::dbgs() << "cast MemRefType:\n";
    (*castOp).getOperation()->print(llvm::dbgs(),
                                    OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  rewriter.replaceOp(op, src);
  return success();
}

FailureOr<Value> BlockDataParser::materializePointer(
    Value ptr, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  SmallVector<int64_t> resultShape;
  if (auto resultTy = dyn_cast<RankedTensorType>(ptr.getType())) {
    if (!isa<triton::PointerType>(resultTy.getElementType()))
      return failure();
    resultShape.append(resultTy.getShape().begin(), resultTy.getShape().end());
  } else if (auto pointerTy = dyn_cast<triton::PointerType>(ptr.getType())) {
    // A pointer to a shaped value is a block pointer, not a scalar element
    // pointer. It is materialized by the make_tensor_ptr path instead.
    if (isa<ShapedType>(pointerTy.getPointeeType()))
      return failure();
    resultShape.push_back(1);
  } else {
    return failure();
  }

  Operation *defOp = ptr.getDefiningOp();
  if (!defOp)
    return failure();

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(defOp);

  BlockData data;
  if (failed(parse(ptr, data, ptr.getLoc(), rewriter, known)) ||
      !data.hasSource() || data.getMemAccType().isUnstructured())
    return failure();

  if (data.getSizesRef().empty()) {
    data.getSizesRef().push_back(rewriter.getIndexAttr(1));
    data.getStridesRef().push_back(rewriter.getIndexAttr(0));
    data.getOffsetsRef().push_back(data.getScalarRef().isNull()
                                       ? OpFoldResult(rewriter.getIndexAttr(0))
                                       : data.getScalarRef());
  }

  if (data.getRank() != static_cast<int64_t>(resultShape.size()))
    return failure();

  known[ptr] = data;

  // A unit dimension with zero stride is represented as a normal contiguous
  // unit dimension. Non-unit zero strides are intentional (for example a
  // splatted base pointer) and must be preserved.
  int64_t inferredSize = 1;
  for (int64_t i = data.getRank() - 1; i >= 0; --i) {
    auto strideConst = getConstantIntValue(data.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(data.getSizesRef()[i]);
    if (!sizeConst)
      return failure();
    if (sizeConst.value() == 1 && strideConst && strideConst.value() == 0)
      data.getStridesRef()[i] = rewriter.getIndexAttr(inferredSize);
    inferredSize *= sizeConst.value();
  }

  // Keep negative static offsets as SSA values. This mirrors rewriteAddPtr and
  // avoids unsigned attribute handling in later reinterpret_cast lowering.
  auto &offsets = data.getOffsetsRef();
  for (OpFoldResult &offset : offsets) {
    if (auto constVal = getConstantIntValue(offset);
        constVal && constVal.value() < 0) {
      offset =
          rewriter
              .create<arith::ConstantIndexOp>(ptr.getLoc(), constVal.value())
              .getResult();
    }
  }

  if (auto intToPtrOp = dyn_cast_or_null<triton::IntToPtrOp>(
          data.getSourceRef().getDefiningOp())) {
    auto pointerTy =
        cast<triton::PointerType>(intToPtrOp.getResult().getType());
    auto memrefTy =
        MemRefType::get({ShapedType::kDynamic}, pointerTy.getPointeeType());
    auto pointerCast = createScalarPointerCast(rewriter, intToPtrOp.getLoc(),
                                               memrefTy, intToPtrOp.getSrc());
    data.setSource(pointerCast.getResult());
  }

  if (data.hasResElemTy()) {
    auto sourceTy = dyn_cast<BaseMemRefType>(data.getSourceRef().getType());
    if (!sourceTy)
      return failure();
    auto castTy = sourceTy.cloneWith(std::nullopt, data.getResElemTyRef());
    if (sourceTy != castTy) {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          ptr.getLoc(), castTy, data.getSourceRef());
      data.setSource(cast.getResult(0));
    }
  }

  FailureOr<memref::ReinterpretCastOp> castOp =
      data.createCastOp(resultShape, ptr.getLoc(), rewriter);
  if (failed(castOp))
    return failure();
  return (*castOp).getResult();
}

static FailureOr<OpFoldResult>
getBaseMemRefOffset(Value convertedBase, ConversionPatternRewriter &rewriter) {
  auto memrefType = dyn_cast<MemRefType>(convertedBase.getType());
  if (!memrefType)
    return failure();
  // Preserve the existing foldable path for a directly converted tt.addptr.
  // Reinterpret-casting a reinterpret cast does not compose offsets, so the
  // first cast's absolute offset must be carried into the new descriptor.
  if (auto baseRecast =
          convertedBase.getDefiningOp<memref::ReinterpretCastOp>())
    return baseRecast.getConstifiedMixedOffset();

  auto stridedLayout = memrefType.getStridesAndOffset();
  int64_t staticOffset = stridedLayout.second;
  if (!ShapedType::isDynamic(staticOffset))
    return OpFoldResult(rewriter.getIndexAttr(staticOffset));

  // A control-flow-carried BlockPtr base uses the canonical identity-layout
  // memref and therefore has static offset zero. Dynamic hidden offsets are not
  // valid BlockPtr bases; their displacement belongs in descriptor offsets.
  return failure();
}

LogicalResult BlockDataParser::rewriteCustomOp(
    hivm::CustomOp op, hivm::CustomOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();
  llvm::SmallVector<Value> newInputs;
  llvm::SmallVector<Value> newOutputs;
  auto convertIntToPtr = [&rewriter](BlockData &data) {
    if (auto intToPtrOp = dyn_cast_or_null<triton::IntToPtrOp>(
            data.getSourceRef().getDefiningOp())) {
      auto rtype = cast<triton::PointerType>(intToPtrOp.getResult().getType());
      auto memrefType =
          MemRefType::get({ShapedType::kDynamic}, rtype.getPointeeType());
      auto hivmPointCastOp = createScalarPointerCast(
          rewriter, intToPtrOp.getLoc(), memrefType, intToPtrOp.getSrc());
      if (data.getSizesRef().size() == 0) {
        data.getSizesRef().push_back(rewriter.getIndexAttr(1));
        if (data.getScalarRef().isNull()) {
          data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
        } else {
          data.getOffsetsRef().push_back(data.getScalarRef());
        }
        data.getStridesRef().push_back(rewriter.getIndexAttr(1));
      }
      data.setSource(hivmPointCastOp.getResult());
    }
  };
  for (auto in : op.getInputs()) {
    in = rewriter.getRemappedValue(in);
    if (!in)
      return op.emitOpError("custom operation input has no converted value");
    BlockData blockData;
    auto curInput = in;
    if (llvm::isa<triton::PointerType>(in.getType())) {
      if (failed(parse(in, blockData, loc, rewriter, known)))
        return failure();
      convertIntToPtr(blockData);
      FailureOr<memref::ReinterpretCastOp> castOp =
          blockData.createCastOp({ShapedType::kDynamic}, loc, rewriter);
      if (failed(castOp))
        return failure();
      curInput = (*castOp).getResult();
    } else if (auto tensor = llvm::dyn_cast<RankedTensorType>(in.getType())) {
      if (llvm::isa<triton::PointerType>(tensor.getElementType())) {
        if (failed(parse(in, blockData, loc, rewriter, known)))
          return failure();
        convertIntToPtr(blockData);
        FailureOr<memref::ReinterpretCastOp> castOp =
            blockData.createCastOp(tensor.getShape(), loc, rewriter);
        if (failed(castOp))
          return failure();
        curInput = (*castOp).getResult();
      }
    }
    newInputs.emplace_back(curInput);
  }
  for (auto out : op.getOutputs()) {
    auto tensorTy = llvm::cast<RankedTensorType>(out.getType());
    if (llvm::isa<triton::PointerType>(tensorTy.getElementType())) {
      // simd library shouldn't output tensor<tt.ptr>
      // after rewrite, delete the tensor<tt.ptr> output value
      continue;
    }
    newOutputs.emplace_back(rewriter.getRemappedValue(out));
  }
  llvm::SmallVector<Type> resultTypes;
  for (auto ty : op->getResultTypes()) {
    if (auto ptrTy = llvm::dyn_cast<triton::PointerType>(ty)) {
      resultTypes.emplace_back(
          MemRefType::get({ShapedType::kDynamic}, ptrTy.getPointeeType()));
      continue;
    }
    if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(ty)) {
      if (auto ptrTy =
              llvm::dyn_cast<triton::PointerType>(tensorTy.getElementType())) {
        resultTypes.emplace_back(
            MemRefType::get(tensorTy.getShape(), ptrTy.getPointeeType()));
        continue;
      }
    }
    resultTypes.emplace_back(ty);
  }
  auto newCustomOp =
      rewriter.create<hivm::CustomOp>(loc, resultTypes, op.getName(), newInputs,
                                      newOutputs, adaptor.getTempBuffers());
  auto operandSegmentSizesAttr = newCustomOp->getAttr("operandSegmentSizes");
  newCustomOp->setAttrs(op->getAttrs());
  newCustomOp->setAttr("operandSegmentSizes", operandSegmentSizesAttr);
  rewriter.replaceOp(op, newCustomOp.getResults());
  return success();
}

// Design for load/store boundary_check.
FailureOr<memref::ReinterpretCastOp>
createRedundantOp(triton::MakeTensorPtrOp op, OpFoldResult sourceBaseOffset,
                  ConversionPatternRewriter &rewriter, BlockData &data) {
  auto loc = op.getLoc();
  // to do boundary_check in tt.load, we need to keep the parent tensor's
  // shape info in the IR.
  // use the parent tensor's shape to create a cast
  auto resultSizes = data.getSizes();
  auto resultOffsets = data.getOffsets();
  data.getSizesRef().clear();
  data.getOffsetsRef().clear();
  data.getSizesRef() =
      std::move(llvm::map_to_vector(op.getShape(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));

  // This redundant ReinterpretCastOp is to describe full tensor_ptr, so each
  // dim offset from base is initialized as zero.
  SmallVector<OpFoldResult> curOffsets(op.getOffsets().size(),
                                       rewriter.getIndexAttr(0));
  // Both the full-shape descriptor and the final block descriptor use the
  // same absolute source offset. Reusing this value avoids both dropping it
  // across SCF and accidentally composing it twice.
  curOffsets.front() = sourceBaseOffset;

  for (auto offset : curOffsets) {
    data.getOffsetsRef().push_back(offset);
  }

  SmallVector<int64_t> staticShapes;
  SmallVector<Value> dynamicShapes;
  dispatchIndexOpFoldResults(data.getSizesRef(), dynamicShapes, staticShapes);
  auto castOp = data.createCastOp(staticShapes, loc, rewriter);
  // restore sizes and offsets
  data.getSizesRef().clear();
  for (auto &s : resultSizes) {
    data.getSizesRef().push_back(s);
  }
  data.getOffsetsRef().clear();
  for (auto &offset : resultOffsets) {
    data.getOffsetsRef().push_back(offset);
  }
  return castOp;
}

// Carries the converted runtime descriptor and records whether the resolver
// has already proven the complete ptr<i1>-to-ptr<i8> normalization contract.
// That exact fallback can initialize BlockData without consulting conversion
// state attached to the original bitcast result.
struct ResolvedMakeTensorPtrBase {
  Value value;
  bool normalizedI1ToI8 = false;
};

static FailureOr<ResolvedMakeTensorPtrBase>
resolveMakeTensorPtrBase(triton::MakeTensorPtrOp op, Value adaptorBase,
                         ConversionPatternRewriter &rewriter) {
  if (adaptorBase && isa<BaseMemRefType>(adaptorBase.getType()))
    return ResolvedMakeTensorPtrBase{adaptorBase};

  auto bitcast = op.getBase().getDefiningOp<triton::BitcastOp>();
  if (!bitcast)
    return failure();

  auto sourceArgument = dyn_cast<BlockArgument>(bitcast.getSrc());
  if (!sourceArgument)
    return failure();

  auto sourcePointer =
      dyn_cast<triton::PointerType>(bitcast.getSrc().getType());
  auto resultPointer =
      dyn_cast<triton::PointerType>(bitcast.getResult().getType());
  if (!sourcePointer || !resultPointer ||
      !sourcePointer.getPointeeType().isInteger(1) ||
      !resultPointer.getPointeeType().isInteger(8))
    return failure();

  Value convertedSource = rewriter.getRemappedValue(bitcast.getSrc());
  if (!convertedSource)
    return failure();
  auto sourceMemRef = dyn_cast<MemRefType>(convertedSource.getType());
  auto expectedType =
      MemRefType::get({ShapedType::kDynamic}, rewriter.getIntegerType(8));
  if (!sourceMemRef || sourceMemRef != expectedType)
    return failure();

  return ResolvedMakeTensorPtrBase{convertedSource,
                                   /*normalizedI1ToI8=*/true};
}

LogicalResult BlockDataParser::rewriteMakeTensorPtrOp(
    triton::MakeTensorPtrOp op, Value convertedBase,
    ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  FailureOr<ResolvedMakeTensorPtrBase> resolvedBase =
      resolveMakeTensorPtrBase(op, convertedBase, rewriter);
  if (failed(resolvedBase)) {
    op.emitOpError("expected the converted base to be a memref descriptor");
    return failure();
  }
  convertedBase = resolvedBase->value;
  Location loc = op.getLoc();
  BlockData data;

  if (resolvedBase->normalizedI1ToI8) {
    // The resolver has already established the complete normalized mask-base
    // contract. Avoid querying the bitcast result mapping while dialect
    // conversion may still be rewriting its owner block.
    data.setSource(convertedBase);
    data.setResElemTy(rewriter.getIntegerType(8));
  } else {
    // Parse the original producer only for semantic information such as a
    // bitcast element type. The runtime source always comes from the resolved
    // converted base so SCF-selected memref descriptors are not bypassed.
    if (failed(
            BlockDataParser::parse(op.getBase(), data, loc, rewriter, known)))
      return failure();
  }
  if (!data.hasSource()) {
    op.emitOpError("failed to resolve the converted scalar base");
    return failure();
  }
  if (data.hasResElemTy()) {
    auto sourceType = dyn_cast<BaseMemRefType>(convertedBase.getType());
    if (!sourceType) {
      op.emitOpError("bitcast base did not resolve to a memref descriptor");
      return failure();
    }
    auto memrefType =
        sourceType.cloneWith(std::nullopt, data.getResElemTyRef());
    if (sourceType == memrefType) {
      data.setSource(convertedBase);
    } else {
      UnrealizedConversionCastOp castOp =
          rewriter.create<mlir::UnrealizedConversionCastOp>(loc, memrefType,
                                                            convertedBase);
      data.setSource(castOp.getOutputs()[0]);
    }
  } else {
    data.setSource(convertedBase);
  }

  data.getOffsetsRef() =
      std::move(llvm::map_to_vector(op.getOffsets(), [&](Value v) {
        auto zeroVal = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(0));
        v = rewriter.create<arith::MaxSIOp>(loc, v, zeroVal);
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));
  data.getStridesRef() =
      std::move(llvm::map_to_vector(op.getStrides(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));

  SmallVector<OpFoldResult> newOffsets;
  for (auto [offset, stride] :
       llvm::zip(data.getOffsetsRef(), data.getStridesRef()))
    newOffsets.push_back(mulOpFoldResult(offset, stride, loc, rewriter,
                                         rewriter.getIndexType()));

  if (newOffsets.empty()) {
    op.emitOpError("expected at least one block pointer dimension");
    return failure();
  }

  FailureOr<OpFoldResult> sourceBaseOffset =
      getBaseMemRefOffset(convertedBase, rewriter);
  if (failed(sourceBaseOffset)) {
    op.emitOpError("could not extract the converted base offset");
    return failure();
  }
  newOffsets.front() = addOpFoldResult(newOffsets.front(), *sourceBaseOffset,
                                       loc, rewriter, rewriter.getIndexType());

  data.getOffsetsRef().clear();

  for (auto offset : newOffsets) {
    data.getOffsetsRef().push_back(offset);
  }

  ArrayRef<int64_t> resultShape;
  auto pointerType = cast<mlir::triton::PointerType>(op.getResult().getType());
  if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
    resultShape = shapedType.getShape();
    data.getSizesRef().clear();
    for (auto dim_size : resultShape) {
      data.getSizesRef().push_back(
          IntegerAttr::get(IntegerType::get(op.getContext(), 64), dim_size));
    }
  } else {
    // scalar pointer, should produce a one dimensional memref
    SmallVector<int64_t> scalarShape(1, 1);
    resultShape = scalarShape;
    assert(data.getRank() == 1);
  }

  // special handling for davinci
  // create redundant reinterpret_cast op for record shape info
  FailureOr<memref::ReinterpretCastOp> redundantOp =
      createRedundantOp(op, *sourceBaseOffset, rewriter, data);
  if (failed(redundantOp))
    return op.emitOpError("could not materialize the full tensor pointer");
  (*redundantOp)->setAttr("tensor_ptr_full_shape", rewriter.getUnitAttr());

  // create reinterpret_cast op for the target block
  data.setSource((*redundantOp).getResult());
  known[op.getResult()] = data;
  FailureOr<memref::ReinterpretCastOp> castOp =
      data.createCastOp(resultShape, loc, rewriter);
  if (failed(castOp))
    return op.emitOpError("could not materialize the block pointer");
  rewriter.replaceOp(op, (*castOp).getResult());

  if (nd2nzFlag) {
    auto basePtr = (*castOp).getResult();
    int original_rank = op.getShape().size() + 1;
    std::string shapeStr;

    auto baseMemrefType = mlir::dyn_cast<MemRefType>(basePtr.getType());
    assert(baseMemrefType && "basePtr is not a memref type");
    auto shape = baseMemrefType.getShape();

    if (auto memrefType = mlir::dyn_cast<MemRefType>(basePtr.getType())) {
      for (auto dim : memrefType.getShape()) {
        shapeStr += llvm::formatv("_{0}", dim);
      }
    }
    std::string elemTypeName;
    Type elemType = baseMemrefType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
      elemTypeName = llvm::formatv("i{0}", intType.getWidth());
    } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType)) {
      std::string floatTypeName;
      llvm::raw_string_ostream os(floatTypeName);
      floatType.print(os);
      os.flush();
      elemTypeName = floatTypeName;
    } else {
      std::string typeName;
      llvm::raw_string_ostream os(typeName);
      elemType.print(os);
      os.flush();
      elemTypeName = typeName;
    }

    std::string memrefTypeStr;
    llvm::raw_string_ostream os(memrefTypeStr);
    baseMemrefType.print(os);
    os.flush();

    std::string laydbgsuffix;
    for (char c : memrefTypeStr) {
      if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
          (c >= 'A' && c <= 'Z') || c == '_' || c == ',' || c == '[' ||
          c == ']') {
        laydbgsuffix += c;
      }
    }
    auto funcName = rewriter.getStringAttr(
        llvm::formatv("__hmf_original_shape{0}d{1}_{2}_{3}", original_rank,
                      shapeStr, elemTypeName, laydbgsuffix));
    MemRefType targetMemrefType = MemRefType::get(
        baseMemrefType.getShape(), baseMemrefType.getElementType(),
        baseMemrefType.getLayout());
    const int vectorSize = 4;
    SmallVector<Type, vectorSize> srcElemTys;
    for (auto sz : op.getShape()) {
      srcElemTys.push_back(sz.getType());
    }
    srcElemTys.push_back(targetMemrefType);
    Type dstElemTy = rewriter.getNoneType();
    FunctionType hintFuncType =
        FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});

    auto mod = SymbolTable::getNearestSymbolTable(op);
    auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(mod, funcName));
    SmallVector<Value, vectorSize> args;
    for (auto sz : op.getShape()) {
      args.push_back(sz);
    }
    args.push_back(basePtr);
    if (!extFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
      extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                              funcName, hintFuncType);
      extFunc.setPrivate();
      extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                       UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPoint(op);
    }
    rewriter.create<func::CallOp>(loc, funcName, dstElemTy, args);
  }

  return success();
}

LogicalResult BlockDataParser::rewriteAdvanceOp(
    triton::AdvanceOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();

  BlockData blockData;
  if (failed(parse(op.getOperand(0), blockData, loc, rewriter, known)))
    return failure();

  // region [BUGFIX] Add the code block below following the same logic as
  // 'BlockDataParser::rewriteAddPtr' function.
  known[op.getResult()] = blockData;
  auto inferedSize = 1;
  for (int i = blockData.getSizesRef().size() - 1; i >= 0; i--) {
    auto strideConst = getConstantIntValue(blockData.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(blockData.getSizesRef()[i]);
    assert(sizeConst.has_value());
    if (sizeConst.value() == 1 && strideConst && strideConst.value() == 0) {
      blockData.getStridesRef()[i] = rewriter.getIndexAttr(inferedSize);
    }
    inferedSize *= sizeConst.value();
  }
  // endregion

  SmallVector<OpFoldResult> incrementOffsets =
      llvm::map_to_vector(op.getOffsets(), [&](Value offset) {
        return getOpFoldResultOfLayoutInfo(offset, rewriter);
      });

  SmallVector<OpFoldResult> newOffsets;
  for (const auto [increment, originalOffset, stride] :
       llvm::zip(incrementOffsets, blockData.getOffsetsRef(),
                 blockData.getStridesRef())) {
    auto curDimOffset =
        addOpFoldResult(mulOpFoldResult(increment, stride, loc, rewriter,
                                        rewriter.getIndexType()),
                        originalOffset, loc, rewriter, rewriter.getIndexType());

    newOffsets.push_back(curDimOffset);
  }

  blockData.getOffsetsRef().clear();

  for (auto offset : newOffsets)
    blockData.getOffsetsRef().push_back(offset);

  SmallVector<int64_t> scalarShape(1, 1); // Stub shape
  ArrayRef<int64_t> resultShape;
  auto pointerType = cast<mlir::triton::PointerType>(op.getResult().getType());

  if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
    resultShape = shapedType.getShape();
  } else {
    // scalar pointer, should produce a one dimensional memref
    resultShape = scalarShape;
    assert(blockData.getRank() == 1);
  }

  FailureOr<memref::ReinterpretCastOp> newOp =
      blockData.createCastOp(resultShape, loc, rewriter);
  if (failed(newOp))
    return op.emitOpError("could not materialize the advanced pointer");
  rewriter.replaceOp(op, (*newOp).getResult());

  known[(*newOp).getResult()] = blockData;
  return success();
}

static bool isIntegerTensorBlockDataValue(Value value) {
  auto tensorType = dyn_cast<TensorType>(value.getType());
  return tensorType && isa<IntegerType>(tensorType.getElementType());
}

static bool containsLegacyTritonPointer(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && isa<triton::PointerType>(shapedType.getElementType());
}

// Before expanding a loop signature, reject backedges that are already known
// to be opaque to BlockData. Triton pointer values may still receive a legal
// converted mapping later, while complete memrefs require an explicit
// reinterpret-cast producer. In particular, this deliberately rejects an
// arbitrary fixed-layout memref returned by func.call.
static bool canAnalyzeLegacyBlockDataBackedge(LoopLikeOpInterface loopOp,
                                              unsigned slot, Value value) {
  if (containsLegacyTritonPointer(value.getType()) ||
      isIntegerTensorBlockDataValue(value) ||
      value.getDefiningOp<memref::ReinterpretCastOp>())
    return true;

  // Forwarding the same loop-carried block argument preserves the descriptor
  // state already parsed from the init. A captured function/block argument is
  // not equivalent and remains opaque to this analysis.
  if (slot < loopOp.getRegionIterArgs().size() &&
      value == loopOp.getRegionIterArgs()[slot])
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    return slot < whileOp.getAfterArguments().size() &&
           value == whileOp.getAfterArguments()[slot];
  }
  return false;
}

// Resolve a selected legacy BlockData state after dialect-conversion remapping.
// The legacy representation is intentionally limited to reinterpret-cast
// descriptors and integer tensors. An opaque memref returned by func.call is a
// complete value, not an analyzable descriptor producer, and must never be
// guessed across the call boundary.
static FailureOr<Value>
resolveBlockDataStateValue(Value originalValue,
                           ConversionPatternRewriter &rewriter) {
  Value value = originalValue;
  if (Value mappedValue = rewriter.getRemappedValue(originalValue)) {
    if (originalValue.getDefiningOp<triton::AddPtrOp>() ||
        originalValue.getDefiningOp<triton::AdvanceOp>() ||
        originalValue.getDefiningOp<triton::MakeTensorPtrOp>()) {
      if (!mappedValue.getDefiningOp<memref::ReinterpretCastOp>())
        return failure();
    } else if (auto tensorType = dyn_cast<TensorType>(mappedValue.getType());
               tensorType &&
               isa<triton::PointerType>(tensorType.getElementType())) {
      return failure();
    }
    value = mappedValue;
  }

  if (value.getDefiningOp<memref::ReinterpretCastOp>() ||
      isIntegerTensorBlockDataValue(value))
    return value;
  return failure();
}

template <typename T>
std::enable_if_t<std::is_same_v<T, scf::YieldOp> ||
                     std::is_same_v<T, scf::ConditionOp>,
                 LogicalResult>
BlockDataParser::rewriteTerminator(
    T op, ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseSet<size_t> &blockArgIdxSet,
    ArrayRef<int64_t> iterArgIdxMap,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  // Any inserted instruction should be before this yield
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);

  auto adaptor = typename T::Adaptor(op);
  ValueRange args;
  if constexpr (std::is_same_v<T, scf::YieldOp>) {
    args = adaptor.getOperands();
  } else {
    args = adaptor.getArgs();
  }

  SmallVector<BlockData, 5> initArgState;
  SmallVector<Value> operands;

  operands.reserve(op->getNumOperands());
  for (const auto &[oper, newIterArgIdx] :
       llvm::zip_equal(args, iterArgIdxMap)) {
    if (newIterArgIdx != -1)
      operands.push_back(oper);
  }

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // BlockData for those values.
  for (auto [i, originalValue] : llvm::enumerate(args)) {
    if (blockArgIdxSet.find(i) == blockArgIdxSet.end())
      continue;

    Value knownKey = rewriter.getRemappedValue(originalValue);
    if (!knownKey)
      knownKey = originalValue;
    if (auto knownState = known.find(knownKey); knownState != known.end()) {
      initArgState.push_back(knownState->second);
      continue;
    }

    FailureOr<Value> stateValue =
        resolveBlockDataStateValue(originalValue, rewriter);
    if (failed(stateValue)) {
      InFlightDiagnostic diagnostic = op.emitError(
          "legacy BlockData loop rewrite cannot analyze carried slot ");
      diagnostic << i << " produced by ";
      if (Operation *producer = originalValue.getDefiningOp())
        diagnostic << producer->getName();
      else
        diagnostic << "a block argument";
      return failure();
    }

    Value v = *stateValue;
    auto reintCastOp = v.getDefiningOp<memref::ReinterpretCastOp>();

    BlockData state;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, state, op.getLoc(), rewriter, known);
    } else {
      if (failed(parse(v, state, op.getLoc(), rewriter, known)))
        return failure();
    }
    initArgState.push_back(state);
  }

  // For each of the BlockData recorded in the last step, extract value
  // that correspond to offset and stride for each dimension and append
  // them to yield operands.
  for (auto state : initArgState) {
    for (auto offset : state.getOffsetsRef()) {
      // offsets can be IntAttr zeroes, since reinterpret_cast collapses
      // them for the input memref, and the for loop may not update
      // offsets other than offsets[0]. Create constants Values for those
      // zeroes.
      if (isa<Attribute>(offset)) {
        auto constOffset = cast<Attribute>(offset);
        assert(isa<IntegerAttr>(constOffset) &&
               dyn_cast<IntegerAttr>(constOffset).getInt() == 0 &&
               "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(cast<Value>(offset));
      }
    }

    auto sizesRef = state.getSizesRef();
    size_t dimIdx = 0;
    for (OpFoldResult stride : state.getStridesRef()) {
      if (isa<Attribute>(stride)) {
        auto constStride = cast<Attribute>(stride);
        assert(isa<IntegerAttr>(constStride) &&
               "attribute strides should be IntegerAttr");

        auto strideVal = dyn_cast<IntegerAttr>(constStride).getInt();
        bool isSizeOne =
            (dimIdx < sizesRef.size() && isa<Attribute>(sizesRef[dimIdx]) &&
             cast<IntegerAttr>(cast<Attribute>(sizesRef[dimIdx])).getInt() ==
                 1);
        assert((strideVal == 1 || (strideVal == 0 && isSizeOne)) &&
               "attribute strides should be ones");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(1));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(cast<Value>(stride));
      }
      dimIdx++;
    }
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  Operation *newOp;
  if constexpr (std::is_same_v<T, scf::YieldOp>) {
    newOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  } else {
    newOp = rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                          operands);
  }

  assert(op->getNumResults() == 0);

  LLVM_DEBUG({
    llvm::dbgs() << "new terminator: ";
    newOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
  return success();
}

// This function is util function for rewriteLoopOp that
// check if given regionIterArg is used with given condition
bool isUsedWithCondition(Value v, std::function<bool(OpOperand *)> cond,
                         int depth = 0,
                         llvm::SmallSetVector<Value, 8> *visited = nullptr) {
  llvm::SmallSetVector<Value, 8> localVisited;
  if (!visited) {
    visited = &localVisited;
  }

  if (visited->contains(v)) {
    return false;
  }
  visited->insert(v);

  for (auto &use : v.getUses()) {
    auto *user = use.getOwner();
    if (user->hasAttr(ConverterUtils::discreteAttrName) ||
        isa<tensor::ExtractOp>(user))
      continue;
    if (cond(&use))
      return true;
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user);
        loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
      Value tiedArg = loopOp.getTiedLoopRegionIterArg(&use);
      if (tiedArg && isUsedWithCondition(tiedArg, cond, depth + 1, visited))
        return true;
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user);
               yieldOp && !isa<scf::WhileOp>(user->getParentOp())) {
      if (depth && isUsedWithCondition(yieldOp->getParentOp()->getResult(
                                           use.getOperandNumber()),
                                       cond, depth - 1, visited))
        return true;
    } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user);
               conditionOp && use.getOperandNumber() > 0) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      if (depth &&
          isUsedWithCondition(whileOp->getResult(use.getOperandNumber() - 1),
                              cond, depth - 1, visited))
        return true;
      if (isUsedWithCondition(
              whileOp.getAfterArguments()[use.getOperandNumber() - 1], cond,
              depth, visited))
        return true;
    }
    for (auto res : user->getResults()) {
      if (isUsedWithCondition(res, cond, depth, visited))
        return true;
    }
  }
  return false;
}

// A loop-carried value may be consumed through a region argument, through a
// while after-argument, or only after the loop result. Check every semantic
// view of the same carried slot so an identity tensor.cast after the loop
// cannot hide an AddPtr/load/store use from the decomposition decision.
bool isLoopCarriedValueUsedWithCondition(
    LoopLikeOpInterface loopOp, unsigned index,
    const std::function<bool(OpOperand *)> &condition) {
  if (index >= loopOp.getRegionIterArgs().size() ||
      index >= loopOp->getNumResults())
    return false;
  if (isUsedWithCondition(loopOp.getRegionIterArgs()[index], condition))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (index < whileOp.getAfterArguments().size() &&
        isUsedWithCondition(whileOp.getAfterArguments()[index], condition))
      return true;
  }
  return isUsedWithCondition(loopOp->getResult(index), condition);
}

// The legacy BlockData route only needs address uses reached from inside the
// loop. Reuse the established traversal for the region arguments, but do not
// start another traversal from the loop result: a numerical carrier can feed
// an unrelated reduction after the loop and eventually reach an address use.
static bool isLoopCarriedAddressValueUsed(
    LoopLikeOpInterface loopOp, unsigned index,
    const std::function<bool(OpOperand *)> &condition) {
  if (index >= loopOp.getRegionIterArgs().size() ||
      index >= loopOp->getNumResults())
    return false;
  if (isUsedWithCondition(loopOp.getRegionIterArgs()[index], condition))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (index < whileOp.getAfterArguments().size() &&
        isUsedWithCondition(whileOp.getAfterArguments()[index], condition))
      return true;
  }
  return false;
}

bool needsLegacyBlockDataLoopRewrite(LoopLikeOpInterface loopOp) {
  auto hasPointerValue = [&](auto values) {
    return llvm::any_of(values, [&](Value value) {
      return containsLegacyTritonPointer(value.getType());
    });
  };

  auto isScalarPointerValue = [](Value value) {
    auto pointerType = dyn_cast<triton::PointerType>(value.getType());
    return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
  };
  auto isTensorPointerValue = [](Value value) {
    auto pointerType = dyn_cast<triton::PointerType>(value.getType());
    return pointerType && isa<ShapedType>(pointerType.getPointeeType());
  };

  SmallVector<Value> boundaryValues;
  boundaryValues.append(loopOp.getInits().begin(), loopOp.getInits().end());
  boundaryValues.append(loopOp.getRegionIterArgs().begin(),
                        loopOp.getRegionIterArgs().end());
  boundaryValues.append(loopOp->getResults().begin(),
                        loopOp->getResults().end());
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation()))
    boundaryValues.append(whileOp.getAfterArguments().begin(),
                          whileOp.getAfterArguments().end());

  bool hasScalarPointer = llvm::any_of(boundaryValues, isScalarPointerValue);
  bool hasTensorPointer = llvm::any_of(boundaryValues, isTensorPointerValue);
  // An opaque scalar-pointer loop has no legacy BlockData schema.  Leave it
  // untouched so the conversion driver reports an ordinary unsupported
  // boundary instead of entering the descriptor parser with a non-memref
  // source. Tensor-pointer loops continue through their established path.
  if (hasScalarPointer && !hasTensorPointer)
    return false;

  // Inspect the original SCF boundary, before the dialect converter remaps a
  // Triton pointer to a reinterpret-cast memref. Such loops still belong to the
  // legacy pointer conversion even though their converted init may look like
  // an ordinary memref later in rewriteLoopOp().
  if (hasPointerValue(loopOp.getInits()) ||
      hasPointerValue(loopOp.getRegionIterArgs()) ||
      llvm::any_of(loopOp->getResultTypes(), containsLegacyTritonPointer))
    return true;
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    if (hasPointerValue(whileOp.getAfterArguments()))
      return true;
  }

  // Preserve the pre-existing tensor-offset/mask path. These pointer-free
  // integer tensors are nevertheless expanded by BlockData when they feed the
  // address operands recognized by the legacy analysis.
  for (auto [index, init] : llvm::enumerate(loopOp.getInits())) {
    auto tensorType = dyn_cast<TensorType>(init.getType());
    if (!tensorType)
      continue;
    auto integerType = dyn_cast<IntegerType>(tensorType.getElementType());
    if (!integerType || integerType.getWidth() == 1)
      continue;
    if (isLoopCarriedAddressValueUsed(loopOp, index, [](OpOperand *use) {
          Operation *user = use->getOwner();
          return isa<triton::AddPtrOp>(user) ||
                 (isa<triton::LoadOp>(user) && use->getOperandNumber() == 1) ||
                 (isa<triton::StoreOp>(user) && use->getOperandNumber() == 2);
        }))
      return true;
  }
  return false;
}

// This function is util function for rewriteLoopOp that create value from data.
// Assume data is structured, and from regionIterArg from LoopLikeOpInterface.
//
// For example,
//
// %7 = scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg3 = %4) ->
// (tensor<128xi32>)  : i32 {
//    %8 = tt.addptr %5, %arg3 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
//    ...
// }
//
// is converted to
//
// %7 = scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg3 = %4,
// %arg4 = %5, %arg5 = %6) -> (tensor<128xi32>)  : i32 {
//   %scalarOffset = arith.index_cast %arg4 : index to i32
//   %scalarStride = arith.index_cast %arg5 : index to i32
//   ...
//   %newRes = arith.addi %offset, %stride : tensor<128xi32>
//   %8 = tt.addptr %5, %newRes : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// }
Value createFromData(RankedTensorType resType, const BlockData &data,
                     const Location &loc, OpBuilder &builder,
                     bool isMaskIterArg) {
  auto resShape = resType.getShape();
  Value newRes = nullptr;
  for (size_t i = 0; i < resShape.size(); i++) {
    auto axisType =
        RankedTensorType::get({resShape[i]}, resType.getElementType());
    auto axisI32Type =
        RankedTensorType::get({resShape[i]}, builder.getIntegerType(32));
    Value axisValue =
        builder.create<triton::MakeRangeOp>(loc, axisI32Type, 0, resShape[i]);
    if (axisType != axisI32Type) {
      axisValue = builder.create<arith::ExtSIOp>(loc, axisType, axisValue);
    }
    Value offset = cast<Value>(data.getOffset(i));
    Value offsetValue = builder.create<arith::IndexCastOp>(
        loc, resType.getElementType(), offset);
    offsetValue = builder.create<triton::SplatOp>(loc, axisType, offsetValue);
    Value stride = cast<Value>(data.getStride(i));
    if (!isMaskIterArg) {
      Value strideValue = builder.create<arith::IndexCastOp>(
          loc, resType.getElementType(), stride);
      strideValue = builder.create<triton::SplatOp>(loc, axisType, strideValue);
      axisValue = builder.create<arith::MulIOp>(loc, axisValue, strideValue);
    }
    axisValue = builder.create<arith::AddIOp>(loc, axisValue, offsetValue);

    for (size_t j = 0; j < resShape.size(); j++) {
      if (i != j)
        axisValue = builder.create<triton::ExpandDimsOp>(loc, axisValue, j);
    }
    axisValue = builder.create<triton::BroadcastOp>(loc, resType, axisValue);
    if (newRes) {
      newRes = builder.create<arith::AddIOp>(loc, newRes, axisValue);
    } else {
      newRes = axisValue;
    }
  }
  return newRes;
}

LogicalResult
BlockDataParser::rewriteLoopOp(LoopLikeOpInterface op,
                               ConversionPatternRewriter &rewriter,
                               llvm::SmallDenseMap<Value, BlockData> &known,
                               ArrayRef<unsigned> onlyIndexTensorSlots) {
  SmallVector<Value> newInitArgs;
  SmallVector<int64_t> iterArgIdxMap;
  SmallVector<bool> maskIterArgs;
  int64_t argCnt = 0;

  SmallVector<std::pair<int, BlockData>, 5> initArgIndexIfBlockData;
  SmallVector<std::pair<int, BlockData>, 5> knownPtrsTmp;
  llvm::SmallDenseSet<size_t> blockArgIdxSet;

  // Create a new list of init args
  for (auto [i, arg] : llvm::enumerate(op.getInits())) {
    auto mappedV = rewriter.getRemappedValue(arg);
    memref::ReinterpretCastOp reintCastOp;
    maskIterArgs.push_back(false);

    // If this init arg is supposed to be remapped, use the remapped
    // value instead.
    // In addition, if this init arg is a memref created by a reinterpret_cast
    // or a tensor of index, there is a chance that it will be used in addptr.
    // Create BlockData for each such init arg.
    if (mappedV) {
      // TODO:
      //  Passing a block argument pointer directly into a for loop not
      //  supported.
      assert(!(isa<BlockArgument>(mappedV) &&
               isa<UnrankedMemRefType>(mappedV.getType())) &&
             "cannot take pointer block argument as init arg for for loop");
      if (auto reinterpretCastOp =
              mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
        // Record memref::ReinterpretCastOp
        reintCastOp = reinterpretCastOp;
        newInitArgs.push_back(mappedV);
        iterArgIdxMap.push_back(argCnt++);
      } else {
        newInitArgs.push_back(mappedV);
        iterArgIdxMap.push_back(argCnt++);
      }
    } else {
      newInitArgs.push_back(arg);
      iterArgIdxMap.push_back(argCnt++);
    }

    auto indexTensor =
        isa<TensorType>(arg.getType()) &&
        isa<IntegerType>(cast<TensorType>(arg.getType()).getElementType()) &&
        cast<IntegerType>(cast<TensorType>(arg.getType()).getElementType())
                .getWidth() != 1 &&
        isLoopCarriedValueUsedWithCondition(op, i, [](OpOperand *use) {
          auto *user = use->getOwner();
          return isa<triton::AddPtrOp>(user) ||
                 (isa<triton::LoadOp>(user) && use->getOperandNumber() == 1) ||
                 (isa<triton::StoreOp>(user) && use->getOperandNumber() == 2);
        });

    if (!onlyIndexTensorSlots.empty() &&
        !llvm::is_contained(onlyIndexTensorSlots, static_cast<unsigned>(i)))
      indexTensor = false;

    // Handle memref::ReinterpretCastOp and tensor<Integer> specially
    if (!reintCastOp && !indexTensor)
      continue;

    BlockData data;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, data, op.getLoc(), rewriter,
                           llvm::SmallDenseMap<Value, BlockData>(0));
    } else {
      if (failed(parse(arg, data, op.getLoc(), rewriter,
                       llvm::SmallDenseMap<Value, BlockData>(0))))
        return failure();
    }

    maskIterArgs[i] =
        indexTensor &&
        isLoopCarriedValueUsedWithCondition(op, i, [](OpOperand *use) {
          auto *user = use->getOwner();
          return (isa<triton::LoadOp>(user) && use->getOperandNumber() == 1) ||
                 (isa<triton::StoreOp>(user) && use->getOperandNumber() == 2);
        });

    if (indexTensor) {
      newInitArgs.back() = nullptr;
      iterArgIdxMap.back() = -1;
      argCnt--;
    }

    // Record the BlockData for later processing
    initArgIndexIfBlockData.push_back(std::make_pair(i, data));
  }

  // Validate every structural backedge before creating constants or a new
  // loop. This prevents a partially expanded signature when an opaque memref
  // source cannot provide offsets and strides.
  auto preflightBackedge = [&](ValueRange values,
                               StringRef edgeName) -> LogicalResult {
    for (auto [index, data] : initArgIndexIfBlockData) {
      (void)data;
      if (index < 0 || static_cast<unsigned>(index) >= values.size()) {
        op->emitError("legacy BlockData preflight found a missing ")
            << edgeName << " value for carried slot " << index;
        return failure();
      }
      Value value = values[index];
      if (!canAnalyzeLegacyBlockDataBackedge(op, index, value)) {
        InFlightDiagnostic diagnostic = op->emitError(
            "legacy BlockData preflight cannot analyze carried slot ");
        diagnostic << index << " on the " << edgeName << " edge produced by ";
        if (Operation *producer = value.getDefiningOp())
          diagnostic << producer->getName();
        else
          diagnostic << "an unknown source";
        return failure();
      }
    }
    return success();
  };
  if (auto forOp = dyn_cast<scf::ForOp>(op.getOperation())) {
    if (failed(preflightBackedge(forOp.getYieldedValues(), "yield")))
      return failure();
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    if (failed(preflightBackedge(whileOp.getConditionOp().getArgs(),
                                 "condition")) ||
        failed(preflightBackedge(whileOp.getYieldOp().getOperands(), "yield")))
      return failure();
  }

  // Set insertion point to be before the for loop for new variables passed
  // into the new loop.
  auto origIp = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  // For each of the BlockData recorded in the last step, insert new
  // instructions to describe offset and stride for each dimension and append
  // them to init args
  for (auto [i, data] : initArgIndexIfBlockData) {
    // For each dimension, if the corresponding offset and stride is an
    // integer attribute, create a constant value and append them at the
    // end of init arg list, which is prepared for calculate layout info with
    // loop interation index
    for (auto &dataOffset : data.getOffsetsRef()) {
      if (isa<Attribute>(dataOffset)) {
        auto constDataOffset = cast<Attribute>(dataOffset);
        assert(isa<IntegerAttr>(constDataOffset));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataOffset).getInt()));
        newInitArgs.push_back(constOp.getResult());
        dataOffset = constOp.getResult();
      } else {
        assert(isa<IndexType>(cast<Value>(dataOffset).getType()));
        newInitArgs.push_back(cast<Value>(dataOffset));
      }
    }

    for (auto &dataStride : data.getStridesRef()) {
      if (isa<Attribute>(dataStride)) {
        auto constDataStride = cast<Attribute>(dataStride);
        assert(isa<IntegerAttr>(constDataStride));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataStride).getInt()));
        newInitArgs.push_back(constOp.getResult());
        dataStride = constOp.getResult();
      } else {
        assert(isa<IndexType>(cast<Value>(dataStride).getType()));
        newInitArgs.push_back(cast<Value>(dataStride));
      }
    }

    // Note that we want the knownPtrs to be indexed by block arg, but we
    // only have index for now. Also, the blockdata we record is the init
    // arg, but want to to use newly created block arg. These block args
    // are not created yet. We will translate this mapping later.
    knownPtrsTmp.push_back(std::make_pair(i, data));
    blockArgIdxSet.insert(i);

    // If the original init arg is a memref produced by reinterpret_cast,
    // create a new memref using new strides and offsets created above.
    // This produces a canonicalized memref, which will match what the
    // for loop generates if it modifies the memref. E.g., original
    // reinterpret_cast can produce a memref with const stride:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1] -> (d0 * 256 +
    //  s0 + d1
    //  * s1)>>
    // The new reinterpret_cast will always have dynamic stride and
    // offset:
    //  - memref<4x256xbf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1
    //  + s0 + d1 * s2)>>
    if (newInitArgs[i] &&
        newInitArgs[i].getDefiningOp<memref::ReinterpretCastOp>()) {
      SmallVector<int64_t> resultShape;
      for (auto size : data.getSizesRef()) {
        auto constSize = getConstantIntValue(size);
        assert(constSize && "expected constant size");
        resultShape.push_back(constSize.value());
      }

      // In current block data layout info, strides and offsets must be dynamic
      // value
      FailureOr<memref::ReinterpretCastOp> castOp =
          data.createCastOp(resultShape, op.getLoc(), rewriter);
      if (failed(castOp))
        return op.emitOpError(
            "could not materialize a loop-carried memref from the source type");
      if (resultShape.size() > 1) {
        auto originalOffset = dyn_cast<Value>(data.getOffsetsRef()[0]);
        for (auto &offsets : newInitArgs) {
          if (offsets == originalOffset) {
            offsets = (*castOp).getOffsets()[0];
            break;
          }
        }
        data.getOffsetsRef()[0] = (*castOp).getOffsets()[0];
      }

      LLVM_DEBUG({
        llvm::dbgs() << "new reinterpret_cast with dynamic sizes "
                        "and offsets:";
        (*castOp).print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
        llvm::dbgs() << "\n";
      });

      newInitArgs[i] = (*castOp).getResult();
    }
  }

  rewriter.restoreInsertionPoint(origIp);
  IRMapping mapping;

  // Create a new LoopOp that uses updated init args and same loop body
  LoopLikeOpInterface newOp;
  auto newInits = to_vector(
      make_filter_range(newInitArgs, [](Value v) { return v != nullptr; }));
  auto commonBodyBuilder = [&](OpBuilder &b, Location loc, bool useInit,
                               ValueRange newRegionArgs, Region &region,
                               Block::BlockArgListType regionArgs,
                               ArrayRef<bool> isUsedForRegionArgs,
                               ArrayRef<bool> maskIterArgs) {
    auto newArgIter = newRegionArgs.begin();
    for (const auto &[regionArg, isUsedForRegionArg] :
         llvm::zip(regionArgs, isUsedForRegionArgs)) {
      if (isUsedForRegionArg) {
        mapping.map(regionArg, *newArgIter);
        ++newArgIter;
      }
    }

    // Convert the book-keeping data structure to use the correct key and value.
    // Key is converted from init arg index to newly created block arg, and
    // Value's BlockData fields are converted from init arg to newly created
    // block arg

    // TODO: remove (useInit = true) logic after supporting make_tensor_ptr
    if (useInit) {
      for (auto [i, data] : knownPtrsTmp) {
        for (auto &offset : data.getOffsetsRef()) {
          offset = *newArgIter;
          ++newArgIter;
        }

        for (auto &stride : data.getStridesRef()) {
          stride = *newArgIter;
          ++newArgIter;
        }

        auto regionArg = regionArgs[i];
        auto key = mapping.lookupOrNull(regionArg);
        if (!key) {
          // Create IndexTensor regionArg from computed offset and stride data
          key = createFromData(cast<RankedTensorType>(regionArg.getType()),
                               data, op.getLoc(), rewriter, maskIterArgs[i]);
          mapping.map(regionArg, key);
        }
        known.insert(std::make_pair(key, data));
      }
    } else {
      for (auto [i, isUsedForRegionArg] :
           llvm::enumerate(isUsedForRegionArgs)) {
        if (!isUsedForRegionArg) {
          BlockData data;
          auto regionArg = regionArgs[i];
          auto regionArgType = cast<RankedTensorType>(regionArg.getType());
          data.getOffsetsRef().resize(regionArgType.getRank());
          data.getStridesRef().resize(regionArgType.getRank());
          for (auto &offset : data.getOffsetsRef()) {
            offset = *newArgIter;
            ++newArgIter;
          }
          for (auto &dim : regionArgType.getShape()) {
            data.getSizesRef().push_back(rewriter.getIndexAttr(dim));
          }
          for (auto &stride : data.getStridesRef()) {
            stride = *newArgIter;
            ++newArgIter;
          }

          auto key = mapping.lookupOrNull(regionArg);
          if (!key) {
            // Create IndexTensor regionArg from computed offset and stride data
            key = createFromData(regionArgType, data, op.getLoc(), rewriter,
                                 maskIterArgs[i]);
            mapping.map(regionArg, key);
          }
          known.insert(std::make_pair(key, data));
        }
      }
    }

    for (auto &bodyOp : region.getOps())
      b.clone(bodyOp, mapping);
  };
  for (const auto &[initArg, newInitArg] :
       llvm::zip(op.getInits(), newInitArgs)) {
    if (newInitArg) {
      mapping.map(initArg, newInitArg);
    }
  }
  SmallVector<Value> newResults;
  SmallVector<int64_t> markerSlotMap;
  if (auto forOp = dyn_cast<scf::ForOp>(op.getOperation())) {
    SmallVector<bool> usedForRegionArgs;
    for (auto newInitArg : newInitArgs) {
      usedForRegionArgs.push_back(newInitArg ? true : false);
    }
    newOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInits,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          mapping.map(forOp.getInductionVar(), iv);
          commonBodyBuilder(b, loc, true, args, forOp.getRegion(),
                            op.getRegionIterArgs(), usedForRegionArgs,
                            maskIterArgs);
        });

    // Replace only the results that correspond to the original scf.for
    auto newResultIter = newOp->result_begin();
    rewriter.setInsertionPointAfter(newOp);
    for (const auto &[res, regionArg, newIterArgIdx, mask] :
         llvm::zip_equal(op->getResults(), op.getRegionIterArgs(),
                         iterArgIdxMap, maskIterArgs)) {
      if (newIterArgIdx != -1) {
        newResults.push_back(*newResultIter);
        ++newResultIter;
      } else {
        auto key = mapping.lookup(regionArg);
        auto data = known.at(key);
        for (auto &offset : data.getOffsetsRef())
          offset =
              newOp.getTiedLoopResult(cast<BlockArgument>(cast<Value>(offset)));
        for (auto &stride : data.getStridesRef())
          stride =
              newOp.getTiedLoopResult(cast<BlockArgument>(cast<Value>(stride)));
        auto newRes =
            createFromData(cast<RankedTensorType>(regionArg.getType()), data,
                           op.getLoc(), rewriter, mask);
        newResults.push_back(newRes);
      }
    }
    markerSlotMap = iterArgIdxMap;
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    SmallVector<Type> resultTypes;
    SmallVector<bool> usedForBeforeRegionArgs;
    SmallVector<bool> usedForAfterRegionArgs;
    llvm::SmallDenseSet<size_t> blockArgIdxSetForAfter;
    SmallVector<int64_t> iterArgIdxMapForAfter;
    SmallVector<bool> maskIterArgsForAfter(whileOp->getNumResults());

    int64_t indexCnt = 0;

    for (auto newInitArg : newInitArgs) {
      usedForBeforeRegionArgs.push_back(newInitArg ? true : false);
    }
    for (size_t i = 0; i < whileOp->getNumResults(); i++) {
      auto resType = whileOp->getResultTypes()[i];
      auto indexTensor =
          isa<RankedTensorType>(resType) &&
          isa<IntegerType>(cast<RankedTensorType>(resType).getElementType()) &&
          isLoopCarriedValueUsedWithCondition(whileOp, i, [](OpOperand *use) {
            auto *user = use->getOwner();
            return isa<triton::AddPtrOp>(user) ||
                   (isa<triton::LoadOp>(user) &&
                    use->getOperandNumber() == 1) ||
                   (isa<triton::StoreOp>(user) && use->getOperandNumber() == 2);
          });
      if (indexTensor) {
        indexCnt += 2 * cast<RankedTensorType>(resType).getRank();
        usedForAfterRegionArgs.push_back(false);
        iterArgIdxMapForAfter.push_back(-1);
        maskIterArgsForAfter[i] =
            isLoopCarriedValueUsedWithCondition(whileOp, i, [](OpOperand *use) {
              auto *user = use->getOwner();
              return (isa<triton::LoadOp>(user) &&
                      use->getOperandNumber() == 1) ||
                     (isa<triton::StoreOp>(user) &&
                      use->getOperandNumber() == 2);
            });
        blockArgIdxSetForAfter.insert(i);
      } else {
        resultTypes.push_back(resType);
        usedForAfterRegionArgs.push_back(true);
        iterArgIdxMapForAfter.push_back(argCnt++);
      }
    }
    resultTypes.append(indexCnt, rewriter.getIndexType());
    newOp = rewriter.create<scf::WhileOp>(
        whileOp.getLoc(), resultTypes, newInits,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          commonBodyBuilder(b, loc, true, args, whileOp.getBefore(),
                            whileOp.getBeforeArguments(),
                            usedForBeforeRegionArgs, maskIterArgs);
        },
        [&](OpBuilder &b, Location loc, ValueRange args) {
          commonBodyBuilder(b, loc, false, args, whileOp.getAfter(),
                            whileOp.getAfterArguments(), usedForAfterRegionArgs,
                            maskIterArgsForAfter);
        });

    auto newResultIter = newOp->result_begin();
    rewriter.setInsertionPointAfter(newOp);
    for (const auto &[res, regionArg, newIterArgIdx, mask] :
         llvm::zip_equal(op->getResults(), whileOp.getAfterArguments(),
                         iterArgIdxMapForAfter, maskIterArgsForAfter)) {
      if (newIterArgIdx != -1) {
        newResults.push_back(*newResultIter);
        ++newResultIter;
      } else {
        auto key = mapping.lookup(regionArg);
        auto data = known.at(key);
        for (auto &offset : data.getOffsetsRef())
          offset = newOp->getResult(
              cast<BlockArgument>(cast<Value>(offset)).getArgNumber());
        for (auto &stride : data.getStridesRef())
          stride = newOp->getResult(
              cast<BlockArgument>(cast<Value>(stride)).getArgNumber());
        auto newRes =
            createFromData(cast<RankedTensorType>(regionArg.getType()), data,
                           op.getLoc(), rewriter, mask);
        newResults.push_back(newRes);
      }
    }

    auto conditionOp =
        cast<scf::WhileOp>(newOp.getOperation()).getConditionOp();
    if (failed(rewriteTerminator(conditionOp, rewriter, blockArgIdxSetForAfter,
                                 iterArgIdxMapForAfter, known)))
      return failure();
    markerSlotMap = iterArgIdxMapForAfter;
  }

  if (!newOp || newResults.size() != op->getNumResults())
    return op->emitError(
        "loop rewrite produced a result list with incompatible arity");

  // Copy all attributes from op to newOp
  newOp->setAttrs(op->getAttrs());
  if (!onlyIndexTensorSlots.empty()) {
    if (Attribute marker =
            op->getAttr(controlflow::kPointerDescriptorBoundaryAttr)) {
      auto descriptorSlots = dyn_cast<DenseI32ArrayAttr>(marker);
      if (!descriptorSlots) {
        rewriter.eraseOp(newOp.getOperation());
        return op->emitError("invalid pointer descriptor boundary marker");
      }

      SmallVector<int32_t> remappedSlots;
      remappedSlots.reserve(descriptorSlots.size());
      for (int32_t slot : descriptorSlots.asArrayRef()) {
        if (slot < 0 || static_cast<size_t>(slot) >= markerSlotMap.size() ||
            markerSlotMap[slot] < 0 ||
            markerSlotMap[slot] > std::numeric_limits<int32_t>::max()) {
          rewriter.eraseOp(newOp.getOperation());
          return op->emitError(
              "pointer descriptor slot cannot be remapped after range rewrite");
        }
        remappedSlots.push_back(static_cast<int32_t>(markerSlotMap[slot]));
      }
      newOp->setAttr(controlflow::kPointerDescriptorBoundaryAttr,
                     DenseI32ArrayAttr::get(op->getContext(), remappedSlots));
    }
  }
  rewriter.replaceOp(op, newResults);

  // Update the loop body. Manually invoke the rewrite logic on addptr and yield
  // in the loop body, so we can take advantage of the states we built up
  for (auto *region : newOp.getLoopRegions()) {
    for (auto &bodyOp : region->getOps()) {
      if (isDistributedTypeCustomOp(&bodyOp)) {
        rewriteStructuredCustomOp(&bodyOp, rewriter);
      } else if (auto addptrOp = dyn_cast<triton::AddPtrOp>(bodyOp)) {
        // FIXME: Constructed adaptor here does not hold the transformed op
        // info.
        auto adaptor = triton::AddPtrOp::Adaptor(addptrOp);
        if (failed(rewriteAddPtr(addptrOp, adaptor, rewriter, known)))
          return failure();
      } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(bodyOp)) {
        if (failed(rewriteAdvanceOp(advanceOp, rewriter, known)))
          return failure();
      } else if (auto makeTensorPtrOp =
                     dyn_cast<triton::MakeTensorPtrOp>(bodyOp)) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(makeTensorPtrOp);
        if (failed(rewriteMakeTensorPtrOp(
                makeTensorPtrOp,
                rewriter.getRemappedValue(makeTensorPtrOp.getBase()), rewriter,
                known)))
          return failure();
      } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(bodyOp);
                 loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(loopOp);
        // Remove UnhandledLoopOp attr before process
        rewriter.modifyOpInPlace(
            loopOp, [&]() { loopOp->removeAttr("UnhandledLoopOp"); });
        if (failed(rewriteLoopOp(loopOp, rewriter, known)))
          return failure();
      }
    }
  }

  if (!op.getRegionIterArgs().empty()) {
    auto yieldOp = cast<scf::YieldOp>(
        newOp.getLoopRegions().back()->back().getTerminator());
    if (failed(rewriteTerminator(yieldOp, rewriter, blockArgIdxSet,
                                 iterArgIdxMap, known)))
      return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new loop\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
  return success();
}

/// @brief Rewrite the triton::AddPtrOp to handle unstructured memory access.
/// @param op The triton::AddPtrOp to be rewritten.
/// @param adaptor The adaptor of the triton::AddPtrOp, used to get operands.
/// @param rewriter The pattern rewriter used to modify the IR.
/// @param data The BlockData containing information about the memory access.
LogicalResult BlockDataParser::rewriteAddPtrToUnstrucMemAcc(
    triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter, BlockData &data) {
  auto loc = op.getLoc();
  auto &offsets = data.getOffsetsRef();
  auto &blockSizes = data.getSizesRef();
  auto &strides = data.getStridesRef();
  Value ptrOffset = adaptor.getOffset();
  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto addptrRes = op.getResult();
  assert(addptrRes.hasOneUse() && "Invalid: tt.addptr has multiple users");
  auto loadOp = *(addptrRes.user_begin());

  // Prepare empty tensor for loop based scalar load
  // FIXME: We use cast here because addptr must return tensor<?x!tt.ptr<f32>>.
  // True?
  auto resTy = cast<ShapedType>(addptrRes.getType());
  auto resEPtrTy = resTy.getElementType();
  auto resETy = cast<triton::PointerType>(resEPtrTy).getPointeeType();
  Value loaded = rewriter.create<tensor::EmptyOp>(loc, blockSizes, resETy);
  SmallVector<Value> initArgs;
  initArgs.push_back(loaded);

  SmallVector<Value> forLBs;
  SmallVector<Value> forUBs;
  SmallVector<Value> forSteps;
  for (auto &s : offsets) {
    forLBs.push_back(zeroIdx);
  }
  for (auto &s : blockSizes) {
    forUBs.push_back(getValueOrCreateConstantIndexOp(rewriter, loc, s));
  }
  for (auto &s : strides) {
    forSteps.push_back(oneIdx);
  }
  SmallVector<Value> ivs;
  bool castFailed = false;
  OpBuilder builder(op);
  auto loop = createNestedLoops(
      builder, loc, 0, blockSizes.size(), forLBs, forUBs, forSteps, ivs,
      initArgs,
      [&](OpBuilder &bB, Location bLoc, SmallVector<Value> &allIVs,
          ValueRange iterArgs) {
        OpBuilder::InsertionGuard g(bB);
        bB.setInsertionPointToStart(bB.getBlock());

        Value scalarOffsetRaw =
            bB.create<tensor::ExtractOp>(bLoc, ptrOffset, allIVs);
        Value scalarOffset = bB.create<arith::IndexCastOp>(
            bLoc, bB.getIndexType(), scalarOffsetRaw);
        OpFoldResult baseOffset = bB.getIndexAttr(0);
        for (auto ofr : data.getOffsetsRef()) {
          baseOffset =
              addOpFoldResult(baseOffset, ofr, bLoc, bB, bB.getIndexType());
        }
        Value baseVal = getValueOrCreateConstantIndexOp(bB, bLoc, baseOffset);
        Value combinedOffset =
            bB.create<arith::AddIOp>(bLoc, baseVal, scalarOffset);
        // Replace offset & size. Only single element.
        data.getOffsetsRef().clear();
        data.getOffsetsRef().push_back(combinedOffset);
        data.getSizesRef().clear();
        data.getSizesRef().push_back(bB.getIndexAttr(1));
        data.getStridesRef().clear();
        data.getStridesRef().push_back(bB.getIndexAttr(1));
        FailureOr<memref::ReinterpretCastOp> castOp =
            data.createCastOp({1}, bLoc, bB);
        if (failed(castOp)) {
          castFailed = true;
          return;
        }
        rewriter.replaceOp(op, (*castOp).getResult());
        // Move tt.load using this tt.addptr into this block
        loadOp->moveAfter((*castOp).getOperation());
        loadOp->setAttr("IndirectLoad", UnitAttr::get(op.getContext()));
        bB.create<scf::YieldOp>(bLoc, iterArgs);
      });
  if (castFailed)
    return op.emitOpError("could not materialize the indirect pointer source");
  return success();
}

} // namespace triton
} // namespace mlir
