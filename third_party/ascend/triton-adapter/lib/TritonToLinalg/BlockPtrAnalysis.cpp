//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
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
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <set>

#define DEBUG_TYPE "triton-block-ptr-analysis"
namespace mlir {
namespace triton {

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
    retOffset = addOpFoldResult(retOffset, ofr, loc, builder);
  }
  return retOffset;
}

MemRefType BlockData::getResultMemrefType(int64_t offset,
                                          ArrayRef<int64_t> resultShape) const {
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto baseMemrefType = dyn_cast<BaseMemRefType>(this->source.getType());
  assert(baseMemrefType && "Invalid element type. It should be a base memref type.");
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
    auto addScalar = addOpFoldResult(lBlock.getScalarRef(),
                                     rBlock.getScalarRef(), loc, rewriter);
    this->scalar = addScalar;
  } else if (lBlock.getRank() == 0) {
    // When both lhs and rhs are scalar type with rank 0, just try passing
    // potential `scalar`
    this->scalar =
        lBlock.isScalar() ? lBlock.getScalarRef() : rBlock.getScalarRef();
  }

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(addOpFoldResult(lOffset, rOffset, loc, rewriter));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(addOpFoldResult(lStride, rStride, loc, rewriter));
  }

  // Both sizes are same implicitly under `add`
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

  assert(
      (lBlock.isScalar() ^ rBlock.isScalar()) &&
      "Currently only support one and only one scalar in function mulBlock()");

  BlockData *lb = &lBlock;
  BlockData *rb = &rBlock;
  if (lb->isScalar()) {
    std::swap(lb, rb);
  }

  // In mulBlock, `scalar` will be accumulated into `offset` and `stride`
  OpFoldResult rScalar = rb->getScalarRef();
  for (const auto &lOffset : lb->getOffsetsRef()) {
    this->offsets.push_back(mulOpFoldResult(lOffset, rScalar, loc, rewriter));
  }

  for (const auto &lStride : lb->getStridesRef()) {
    this->strides.push_back(mulOpFoldResult(lStride, rScalar, loc, rewriter));
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

  for (const auto &[lOffset, rOffset] :
       llvm::zip(lBlock.getOffsetsRef(), rBlock.getOffsetsRef())) {
    this->offsets.push_back(divOpFoldResult(lOffset, rOffset, loc, rewriter));
  }

  for (const auto &[lStride, rStride] :
       llvm::zip(lBlock.getStridesRef(), rBlock.getStridesRef())) {
    this->strides.push_back(divOpFoldResult(lStride, rStride, loc, rewriter));
  }

  this->sizes = lBlock.getSizesRef();

  this->getMemAccTypeRef().merge(lBlock.getMemAccTypeRef());
  this->getMemAccTypeRef().merge(rBlock.getMemAccTypeRef());
  // this->setMemAccTy(selectMaxMemAccTy(lBlock.getMemAccType(),
  // rBlock.getMemAccType()));
}

memref::ReinterpretCastOp BlockData::createCastOp(ArrayRef<int64_t> resultShape,
                                                  const Location &loc,
                                                  OpBuilder &builder) const {
  OpFoldResult resOffset = this->inferBlockOffset(loc, builder);
  auto resultType = this->getResultMemrefType(
      isa<Attribute>(resOffset) ? getConstantIntValue(resOffset).value()
                                : ShapedType::kDynamic,
      resultShape);

  return builder.create<memref::ReinterpretCastOp>(
      loc, resultType, this->source, resOffset, this->sizes, this->strides);
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

Value BlockDataParser::getScalarMemRef(Value ptr, Value memref,
                                       const Location &loc,
                                       ConversionPatternRewriter &rewriter) {
  assert(isa<triton::PointerType>(ptr.getType()) && "expect a scalar pointer");
  if (ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (auto castOp = memref.getDefiningOp<memref::ReinterpretCastOp>()) {
      return castOp.getResult();
    } else {
      llvm_unreachable("pointer value is defined by an unexpected op");
    }
  }

  assert(isa<BlockArgument>(ptr) &&
         "pointer should be produced by addptr or block argument");
  BlockData data;
  data.setSource(memref);
  data.getOffsetsRef().push_back(rewriter.getIndexAttr(0));
  data.getSizesRef().push_back(rewriter.getIndexAttr(1));
  data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  auto castOp = data.createCastOp(SmallVector<int64_t>(1, 1), loc, rewriter);
  return castOp.getResult();
}

void BlockDataParser::parse(
    Value operand, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  if (known.find(operand) != known.end()) {
    return data = known.lookup(operand), void();
  }

  if (isa<IntegerType>(operand.getType())) {
    data.setScalar(getOpFoldResultOfLayoutInfo(operand, rewriter));
    return;
  }

  //
  if (isa<triton::PointerType>(operand.getType())) {
    // Just consider two state: ptr<scalar> and ptr<tensor<scalar>>
    auto remappedPtr = rewriter.getRemappedValue(operand);
    assert(remappedPtr);
    if (auto op = operand.getDefiningOp()) {
      if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
        parseAddPtr(addPtrOp, data, loc, rewriter, known);
      } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(op)) {
        parseBitcast(bitcastOp, data, loc, rewriter, known);
      } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
        parseTensorPtr(makeTensorPtrOp, data, loc, rewriter, known);
      } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
        // To support
        // ptr_0 = tl.advance(ptr)
        // ptr_1 = tl.advance(ptr_0)
        parseTensorPtr(advanceOp, data, loc, rewriter, known);
      } else if (auto intToPtrOp = dyn_cast<triton::IntToPtrOp>(op)) {
        data.setSource(remappedPtr);
      } else {
        LLVM_DEBUG({ llvm::dbgs() << operand << "\n"; });
        llvm_unreachable(
            "Unexpected operand defining operation, a scalar "
            "pointer can only be produced by AddPtrOp or direct block ptr");
      }
    } else {
      data.setSource(remappedPtr);
    }
    return;
  }

  // not a scalar pointer
  if (auto addOp = operand.getDefiningOp<arith::AddIOp>()) {
    parseAdd(addOp, data, loc, rewriter, known);
  } else if (auto mulOp = operand.getDefiningOp<arith::MulIOp>()) {
    parseMul(mulOp, data, loc, rewriter, known);
  } else if (auto addPtrOp = operand.getDefiningOp<triton::AddPtrOp>()) {
    parseAddPtr(addPtrOp, data, loc, rewriter, known);
  } else if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
    parseConstSplat(constOp, data, loc, rewriter, known);
  } else if (auto broadcastOp = operand.getDefiningOp<triton::BroadcastOp>()) {
    parseBroadcast(broadcastOp, data, loc, rewriter, known);
  } else if (auto splatOp = operand.getDefiningOp<triton::SplatOp>()) {
    parseSplat(splatOp, data, loc, rewriter, known);
  } else if (auto expandDimsOp =
                 operand.getDefiningOp<triton::ExpandDimsOp>()) {
    parseExpandDims(expandDimsOp, data, loc, rewriter, known);
  } else if (auto remOp = operand.getDefiningOp<arith::RemSIOp>()) {
    parseRem(remOp, data, loc, rewriter, known);
  } else if (auto bitcastOp = operand.getDefiningOp<triton::BitcastOp>()) {
    parseBitcast(bitcastOp, data, loc, rewriter, known);
  } else if (auto extsiOp = operand.getDefiningOp<arith::ExtSIOp>()) {
    parseExtSI(extsiOp, data, loc, rewriter, known);
  } else if (auto divOp = operand.getDefiningOp<arith::DivSIOp>()) {
    parseDiv(divOp, data, loc, rewriter, known);
  } else if (auto makeRangeOp = operand.getDefiningOp<triton::MakeRangeOp>()) {
    parseMakeRange(makeRangeOp, data, loc, rewriter, known);
  } else if (auto reduceOp = operand.getDefiningOp<triton::ReduceOp>()) {
    parseReduce(reduceOp, data, loc, rewriter, known);
  } else if (auto loadOp = operand.getDefiningOp<triton::LoadOp>()) {
    parseIndirectLoad<triton::LoadOp>(loadOp, data, loc, rewriter, known);
  } else if (auto castOp = operand.getDefiningOp<arith::FPToSIOp>()) {
    parseIndirectLoad<arith::FPToSIOp>(castOp, data, loc, rewriter, known);
  } else if (auto extractSliceOp =
                 operand.getDefiningOp<tensor::ExtractSliceOp>()) {
    parseExtractSlice(extractSliceOp, data, loc, rewriter, known);
  } else if (auto forOp = operand.getDefiningOp<scf::ForOp>()) {
    parseIndirectLoad<scf::ForOp>(forOp, data, loc, rewriter, known);
  } else if (auto tensorCastOp = operand.getDefiningOp<tensor::CastOp>()) {
    // Used for identity operation.
    parse(tensorCastOp.getSource(), data, loc, rewriter, known);
  } else {
    operand.dump();
    llvm_unreachable("encountered AddPtrOp produced by unsupported operation");
  }
}

void BlockDataParser::parseAdd(
    arith::AddIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);
  data.addBlock(lBlock, rBlock, loc, rewriter);
}

void BlockDataParser::parseMul(
    arith::MulIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);

  data.mulBlock(lBlock, rBlock, loc, rewriter);
}

void BlockDataParser::parseDiv(
    arith::DivSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  BlockData lBlock, rBlock;
  parse(op.getLhs(), lBlock, loc, rewriter, known);
  parse(op.getRhs(), rBlock, loc, rewriter, known);
  data.divBlock(lBlock, rBlock, loc, rewriter);
}

// TODO : support modulos
void BlockDataParser::parseRem(
    arith::RemSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(false && "Address expression with modulo is not supported yet, it "
                  "shall be analysis at linearize.");
}

void BlockDataParser::parseMakeRange(
    triton::MakeRangeOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
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

void BlockDataParser::parseExpandDims(
    triton::ExpandDimsOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  parse(op.getSrcMutable().get(), data, loc, rewriter, known);
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
}

void BlockDataParser::parseExtractSlice(
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
  parse(extract_src, srcBlock, loc, rewriter, known);
  if (!srcBlock.hasSource()) {
    llvm_unreachable(scenarioMessages.c_str());
  }
  if (!isa<triton::LoadOp>(srcBlock.getSource().getDefiningOp())) {
    llvm_unreachable(scenarioMessages.c_str());
  }

  auto extract_result = op->getResult(0);
  auto shaped_ty = dyn_cast<RankedTensorType>(extract_result.getType());
  auto shape = shaped_ty.getShape();
  if (shape.size() > 1 || shape[0] > 1) {
    llvm_unreachable(scenarioMessages.c_str());
  }
  auto castOp = rewriter.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(shape, rewriter.getIndexType()),
      extract_result);
  auto offset = castOp.getResult();
  if (data.isEmpty()) {
    data.getOffsetsRef().push_back(offset);
    data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  } else {
    llvm_unreachable(
        "parseExtractSlice with offset already setup not yet supported");
  }
}

void BlockDataParser::parseBitcast(
    triton::BitcastOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  parse(op.getSrc(), data, loc, rewriter, known);

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
      data.setSource(remappedValue);
      LLVM_DEBUG({
        llvm::dbgs() << "Remapping bitcastOp:\n";
        llvm::dbgs() << op << "\nto \n";
        llvm::dbgs() << remappedValue << "\n";
      });
    }
  }
  data.setResElemTy(resElemPointeeTy);
}

void BlockDataParser::parseExtSI(
    arith::ExtSIOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  parse(op.getIn(), data, loc, rewriter, known);
}

void BlockDataParser::parseBroadcast(
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

  parse(src, data, loc, rewriter, known);

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
}

void BlockDataParser::parseSplat(
    triton::SplatOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());
  auto src = op.getSrc();
  auto dst = op.getResult();
  auto dstShape = dyn_cast<ShapedType>(dst.getType()).getShape();

  parse(src, data, loc, rewriter, known);

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
    op->emitError("Block data Analysis: unsupported splat pattern");
    return;
  }
  if (data.isScalar()) {
    data.getOffsetsRef()[0] = data.getScalarRef();
  }
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
                 std::is_same_v<T, triton::AdvanceOp>>
BlockDataParser::parseTensorPtr(
    T op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  Value remappedValue = rewriter.getRemappedValue(op);
  if (auto castOp = remappedValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    parseReinterpretCast(castOp, data, loc, rewriter, known);
  } else {
    llvm_unreachable("the value should be mapped to memref.reinterpret_cast");
  }
}

void BlockDataParser::parseAddPtr(
    triton::AddPtrOp op, BlockData &data, const Location &loc,
    ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  assert(data.isEmpty());

  BlockData ptrBlock, offsetBlock;
  parse(op.getPtr(), ptrBlock, op.getLoc(), rewriter, known);
  parse(op.getOffset(), offsetBlock, op.getLoc(), rewriter, known);

  assert(ptrBlock.hasSource() &&
         "Ptr field should provide source/base pointer");
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

  assert(ptrBlock.getRank() == offsetBlock.getRank() &&
         "ptr and offset should have same rank");
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

void BlockDataParser::parseReduce(
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
  parse(reduce_src, srcBlock, loc, rewriter, known);
  if (!srcBlock.hasSource()) {
    llvm_unreachable(scenarioMessages.c_str());
  }
  if (!isa<triton::LoadOp>(srcBlock.getSource().getDefiningOp())) {
    llvm_unreachable(scenarioMessages.c_str());
  }

  auto reduce_result = op->getResult(0);
  auto shaped_ty = dyn_cast<RankedTensorType>(reduce_result.getType());
  auto shape = shaped_ty.getShape();
  auto ops = llvm::map_to_vector(op.getBody()->without_terminator(),
                                 [](Operation &op) { return &op; });
  // Support only the case: scalar = tl.load(1D tensor)
  if (shape.size() != 1 || op.getAxis() != 0 || ops.size() != 1 ||
      !isa<arith::MinSIOp>(ops.front())) {
    llvm_unreachable(scenarioMessages.c_str());
  }

  auto castOp = rewriter.create<arith::IndexCastOp>(
      loc, RankedTensorType::get(shape, rewriter.getIndexType()),
      reduce_result);
  auto offset = castOp.getResult();
  if (data.isEmpty()) {
    data.getOffsetsRef().push_back(offset);
    data.getSizesRef().push_back(rewriter.getIndexAttr(shape[0]));
    data.getStridesRef().push_back(rewriter.getIndexAttr(1));
  } else {
    llvm_unreachable("parseReduce with offset already setup not yet supported");
  }
}

template <typename OpTy>
void parseIndirectLoad(OpTy op, BlockData &data, const Location &loc,
                       ConversionPatternRewriter &rewriter,
                       const llvm::SmallDenseMap<Value, BlockData> &known) {
  // FIXME: assume single result of operation
  auto opRes = op->getResult(0);
  auto opResTy = opRes.getType();
  std::vector<int64_t> resShape;
  if (auto shapedResTy = dyn_cast<ShapedType>(opResTy)) {
    // For now, we consider this is UnstrucMemAcc because we have no other info.
    // Visiting other ops may change the type due to more info.
    data.setMemAccVal(MemAccVal::UnstrucMemAcc);
    resShape = shapedResTy.getShape().vec();
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

void BlockDataParser::rewriteAddPtr(
    triton::AddPtrOp op, triton::AddPtrOp::Adaptor &adaptor,
    ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  auto insertPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(op);

  BlockData data;
  parseAddPtr(op, data, op.getLoc(), rewriter, known);

  if (auto src = data.getSource();
      data.getMemAccTypeRef().isUnstructured() &&
      !(src && isa_and_nonnull<triton::IntToPtrOp>(src.getDefiningOp()))) {
    // TODO: Based on more info, try to create a performant IR
    rewriteAddPtrToUnstrucMemAcc(op, adaptor, rewriter, data);
    LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(op) << "\n"; });
    return;
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
  for (int i = data.getSizesRef().size() - 1; i >= 0; i--) {
    auto strideConst = getConstantIntValue(data.getStridesRef()[i]);
    auto sizeConst = getConstantIntValue(data.getSizesRef()[i]);
    assert(sizeConst.has_value());
    if (sizeConst.value() == 1 && strideConst && strideConst.value() == 0) {
      data.getStridesRef()[i] = rewriter.getIndexAttr(inferedSize);
    }
    inferedSize *= sizeConst.value();
  }

  if (auto intToPtrOp =
          dyn_cast<triton::IntToPtrOp>(data.getSourceRef().getDefiningOp())) {
    auto rtype = cast<triton::PointerType>(intToPtrOp.getResult().getType());
    auto memrefType =
        MemRefType::get({ShapedType::kDynamic}, rtype.getPointeeType());
    auto hivmPointCastOp = rewriter.create<hivm::PointerCastOp>(
        intToPtrOp.getLoc(), memrefType, ValueRange{intToPtrOp.getSrc()});
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

  memref::ReinterpretCastOp castOp =
      data.createCastOp(resultShape, op.getLoc(), rewriter);
  Value src = castOp.getResult();
  LLVM_DEBUG({
    llvm::dbgs() << "cast MemRefType:\n";
    castOp.getOperation()->print(llvm::dbgs(),
                                 OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  rewriter.replaceOp(op, src);
  rewriter.restoreInsertionPoint(insertPoint);
}

OpFoldResult accumulatePotentialOffsetOnBase(
    triton::MakeTensorPtrOp op, Value base, OpFoldResult offset,
    ConversionPatternRewriter &rewriter) {
  if (auto baseRecast = base.getDefiningOp<memref::ReinterpretCastOp>()) {
    assert(isa<triton::AddPtrOp>(op.getBase().getDefiningOp()) &&
           "base of MakeTensorPtrOp only comes from native ptr or AddPtrOp");

    return addOpFoldResult(offset, baseRecast.getConstifiedMixedOffset(),
                           op.getLoc(), rewriter);
  }

  return offset;
}

// Design for load/store boundary_check.
memref::ReinterpretCastOp
createRedundantOp(triton::MakeTensorPtrOp op,
                  ConversionPatternRewriter &rewriter,
                  BlockData &data) {
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
  // Just accumulate base potential offset
  curOffsets.front() = accumulatePotentialOffsetOnBase(
      op, rewriter.getRemappedValue(op.getBase()), curOffsets.front(),
      rewriter);

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

void BlockDataParser::rewriteMakeTensorPtrOp(
    triton::MakeTensorPtrOp op, Value base,
    ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  Location loc = op.getLoc();
  BlockData data;

  auto orderSize = op.getOrder().size();
  if (orderSize > 1) {
    // Declaration of llvm::ArrayRef::slice(n, m)
    // - Chop off the first N elements of the array, and keep M elements
    //   in the array.
    // Take care that 'm' means chunk length
    for (auto [first, second] :
         llvm::zip(op.getOrder().slice(0, orderSize - 1),
                   op.getOrder().slice(1, orderSize - 1))) {
        assert(first == second + 1 && "Currently only support default order on block pointers");
    }
  }

  // Handle base is defined by tt.bitcast
  BlockDataParser::parse(op.getBase(), data, loc, rewriter, known);
  if (data.hasResElemTy()) {
    auto memrefType = dyn_cast<BaseMemRefType>(data.getSourceRef().getType())
                          .cloneWith(std::nullopt, data.getResElemTyRef());
    UnrealizedConversionCastOp castOp =
        rewriter.create<mlir::UnrealizedConversionCastOp>(loc, memrefType,
                                                          data.getSourceRef());
    data.setSource(castOp.getOutputs()[0]);
  } else {
    data.setSource(rewriter.getRemappedValue(op.getBase()));
  }

  data.getOffsetsRef() =
      std::move(llvm::map_to_vector(op.getOffsets(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));
  data.getStridesRef() =
      std::move(llvm::map_to_vector(op.getStrides(), [&](Value v) {
        return getOpFoldResultOfLayoutInfo(v, rewriter);
      }));

  SmallVector<OpFoldResult> newOffsets;
  for (auto [offset, stride] :
       llvm::zip(data.getOffsetsRef(), data.getStridesRef()))
    newOffsets.push_back(mulOpFoldResult(offset, stride, loc, rewriter));

  // 1. Consider that current base ptr may comes from `triton::AddPtrOp`,
  // which have been converted to `memref::ReinterpretCastOp` with 1D
  // shape([1,]) by `AddPtrConverter`.
  // 2. While here would also convert `triton::MakeTensorPtrOp` to
  // `memref::ReinterpretCastOp`, it will create use-def on double recast
  // which means offset&size&stride info of first one will be dropped in terms
  // of memref recast op fold specification.
  //
  // Conclusion with above two:
  // Base of MakeTensorPtrOp has been seen as origin base, so it should
  // reserve offset of first recast if it exists.
  // Here extract the offset of first recast and add it to highest dimension
  newOffsets.front() = accumulatePotentialOffsetOnBase(
      op, base, newOffsets.front(), rewriter);

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

  known[op.getResult()] = data;

  // special handling for davinci
  // create redundant reinterpret_cast op for record shape info
  auto redundantOp = createRedundantOp(op, rewriter, data);
  redundantOp->setAttr("tensor_ptr_full_shape", rewriter.getUnitAttr());

  // create reinterpret_cast op for the target block
  data.setSource(redundantOp.getResult());
  auto castOp = data.createCastOp(resultShape, loc, rewriter);
  rewriter.replaceOp(op, castOp.getResult());

  if (nd2nzFlag) {
    auto basePtr = castOp.getResult();
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
}

void BlockDataParser::rewriteAdvanceOp(
    triton::AdvanceOp op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
  OpBuilder::InsertionGuard insertionGuard(rewriter);
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();

  BlockData blockData;
  parse(op.getOperand(0), blockData, loc, rewriter, known);

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
        addOpFoldResult(mulOpFoldResult(increment, stride, loc, rewriter),
                        originalOffset, loc, rewriter);

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

  auto newOp = blockData.createCastOp(resultShape, loc, rewriter);
  rewriter.replaceOp(op, newOp.getResult());

  known[newOp.getResult()] = blockData;
}

void BlockDataParser::rewriteYieldOp(
    scf::YieldOp op, ConversionPatternRewriter &rewriter,
    const llvm::SmallDenseSet<size_t> &blockArgIdxSet, ArrayRef<int64_t> iterArgIdxMap,
    const llvm::SmallDenseMap<Value, BlockData> &known) {
  // Any inserted instruction should be before this yield
  OpBuilder::InsertionGuard insertionGuard{rewriter};
  rewriter.setInsertionPoint(op);

  auto adaptor = scf::YieldOp::Adaptor(op);

  SmallVector<BlockData, 5> initArgState;
  SmallVector<Value> operands;

  operands.reserve(op->getNumOperands());
  for (const auto &[oper, newIterArgIdx]: llvm::zip_equal(adaptor.getOperands(), iterArgIdxMap)) {
    if (newIterArgIdx != -1)
      operands.push_back(oper);
  }

  // For each of the init arg that we added additional Values in for loop, we
  // need to add corresponding Values as yield operands. The loop below gathers
  // BlockData for those values.
  for (auto [i, v] : llvm::enumerate(adaptor.getOperands())) {
    if (auto mappedV = rewriter.getRemappedValue(v)) {
      // If this value is a tensor of pointers produced by AddPtrOp,
      // we should have already converted to a ReinterpretCastOp without
      // layout information for the normal cases
      if (v.getDefiningOp<triton::AddPtrOp>() ||
          v.getDefiningOp<triton::AdvanceOp>() ||
          v.getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (auto castOp = mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
          v = castOp;
        } else {
          llvm_unreachable("mapped value defined by an unexpected op");
        }
      } else {
        // If this value is not a tensor of pointers, we will use the
        // mapped value, and rely on the conversion will happen later
        // automatically when we legalize loop body.

        // TODO:
        // The scenario where a value is a tensor of pointers but not
        // produced by AddPtrOp is not supported
        if (isa<TensorType>(mappedV.getType()) &&
            isa<triton::PointerType>(
                dyn_cast<TensorType>(mappedV.getType()).getElementType()))
          llvm_unreachable("unsupported scenario where a value is a tensor of "
                           "pointers but not produced by AddPtrOp");
        v = mappedV;
      }
    }

    if (blockArgIdxSet.find(i) == blockArgIdxSet.end())
      continue;

    auto reintCastOp = v.getDefiningOp<memref::ReinterpretCastOp>();
    assert(
        reintCastOp ||
        (isa<TensorType>(v.getType()) &&
         isa<IntegerType>(dyn_cast<TensorType>(v.getType()).getElementType())));

    BlockData state;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, state, op.getLoc(), rewriter, known);
    } else {
      parse(v, state, op.getLoc(), rewriter, known);
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
        auto constOffset = offset.get<Attribute>();
        assert(isa<IntegerAttr>(constOffset) &&
               dyn_cast<IntegerAttr>(constOffset).getInt() == 0 &&
               "attribute offsets should be zeroes");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(0));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(offset.get<Value>());
      }
    }

    for (OpFoldResult stride : state.getStridesRef()) {
      if (isa<Attribute>(stride)) {
        auto constStride = stride.get<Attribute>();
        assert(isa<IntegerAttr>(constStride) &&
               dyn_cast<IntegerAttr>(constStride).getInt() == 1 &&
               "attribute strides should be ones");
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(1));
        operands.push_back(constOp.getResult());
      } else {
        operands.push_back(stride.get<Value>());
      }
    }
  }

  // Yield is a terminator op that must be at the end of the function
  rewriter.setInsertionPointAfter(op);
  auto newOp = rewriter.replaceOpWithNewOp<scf::YieldOp>(op, operands);
  assert(op->getNumResults() == 0);

  LLVM_DEBUG({
    llvm::dbgs() << "new yield:";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

// This function is util function for rewriteLoopOp that
// check if given regionIterArg is used by MemAccOp
bool isUsedByMemAccOp(Value v, int depth = 0) {
  for (auto &use: v.getUses()) {
    auto *user = use.getOwner();
    if (user->hasAttr(ConverterUtils::discreteAttrName))
      continue;
    if (isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp, triton::AtomicCASOp>(user))
      return true;
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user);
        loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
      if(isUsedByMemAccOp(loopOp.getTiedLoopRegionIterArg(&use), depth + 1))
        return true;
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (depth && isUsedByMemAccOp(yieldOp->getParentOp()->getResult(use.getOperandNumber()), depth - 1))
        return true;
    }
    for (auto res: user->getResults()) {
      if (isUsedByMemAccOp(res, depth))
        return true;
    }
  }
  return false;
}

// This function is util function for rewriteLoopOp that
// check if given regionIterArg is used for mask
// Assuming unstructure case is handled in previous pass,
// given value is 1D tensor and stride is one
bool isUsedforMask(Value v, int depth = 0) {
  if (auto tensorType = dyn_cast<RankedTensorType>(v.getType());
      !(tensorType && tensorType.getRank() == 1))
    return false;
  for (auto &use: v.getUses()) {
    auto *user = use.getOwner();
    if ((isa<triton::LoadOp>(user) && use.getOperandNumber() == 1) ||
        (isa<triton::StoreOp>(user) && use.getOperandNumber() == 2))
      return true;
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user);
        loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
      if(isUsedforMask(loopOp.getTiedLoopRegionIterArg(&use), depth + 1))
          return true;
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (depth && isUsedforMask(yieldOp->getParentOp()->getResult(use.getOperandNumber()), depth - 1))
        return true;
    }
    for (auto res: user->getResults()) {
      if (isUsedforMask(res, depth))
        return true;
    }
  }
  return false;
}

// This function is util function for rewriteLoopOp that create value from data.
// Assume data is structured, and from regionIterArg from LoopLikeOpInterface.
//
// For example,
//
// %7 = scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg3 = %4) -> (tensor<128xi32>)  : i32 {
//    %8 = tt.addptr %5, %arg3 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
//    ...
// }
//
// is converted to
//
// %7 = scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg3 = %4, %arg4 = %5, %arg5 = %6) -> (tensor<128xi32>)  : i32 {
//   %scalarOffset = arith.index_cast %arg4 : index to i32
//   %scalarStride = arith.index_cast %arg5 : index to i32
//   ...
//   %newRes = arith.addi %offset, %stride : tensor<128xi32>
//   %8 = tt.addptr %5, %newRes : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// }
Value createFromData(RankedTensorType resType, const BlockData &data, const Location &loc, OpBuilder &builder, bool isMaskIterArg) {
  auto resShape = resType.getShape();
  Value newRes = nullptr;
  for (size_t i = 0; i < resShape.size(); i++) {
    auto axisType = RankedTensorType::get({resShape[i]}, resType.getElementType());
    auto axisI32Type = RankedTensorType::get({resShape[i]}, builder.getIntegerType(32));
    Value axisValue = builder.create<triton::MakeRangeOp>(loc, axisI32Type, 0, resShape[i]);
    if (axisType != axisI32Type) {
      axisValue = builder.create<arith::ExtSIOp>(loc, axisType, axisValue);
    }
    Value offset = cast<Value>(data.getOffset(i));
    Value offsetValue = builder.create<arith::IndexCastOp>(loc, resType.getElementType(), offset);
    offsetValue = builder.create<triton::SplatOp>(loc, axisType, offsetValue);
    Value stride = cast<Value>(data.getStride(i));
    if (!isMaskIterArg) {
      Value strideValue = builder.create<arith::IndexCastOp>(loc, resType.getElementType(), stride);
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

void BlockDataParser::rewriteLoopOp(
    LoopLikeOpInterface op, ConversionPatternRewriter &rewriter,
    llvm::SmallDenseMap<Value, BlockData> &known) {
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
      if (auto reinterpretCastOp = mappedV.getDefiningOp<memref::ReinterpretCastOp>()) {
        // Record memref::ReinterpretCastOp
        reintCastOp = reinterpretCastOp;
        newInitArgs.push_back(mappedV);
        iterArgIdxMap.push_back(argCnt++);
      } else if (auto defOp = op.getYieldedValues()[i].getDefiningOp();
                (defOp && defOp->hasAttr("MetaUse"))) {
        // When argument is MetaUse in the loop,
        // It is removed in iter_args
        newInitArgs.push_back(nullptr);
        iterArgIdxMap.push_back(-1);
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
        isUsedByMemAccOp(op.getRegionIterArgs()[i]);

    // Handle memref::ReinterpretCastOp and tensor<Integer> specially
    if (!reintCastOp && !indexTensor)
      continue;

    BlockData data;
    if (reintCastOp) {
      parseReinterpretCast(reintCastOp, data, op.getLoc(), rewriter,
                           llvm::SmallDenseMap<Value, BlockData>(0));
    } else {
      parse(arg, data, op.getLoc(), rewriter,
            llvm::SmallDenseMap<Value, BlockData>(0));
    }

    maskIterArgs[i] = indexTensor && isUsedforMask(op.getRegionIterArgs()[i]);

    // Record the BlockData for later processing
    initArgIndexIfBlockData.push_back(std::make_pair(i, data));
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
        auto constDataOffset = dataOffset.get<Attribute>();
        assert(isa<IntegerAttr>(constDataOffset));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataOffset).getInt()));
        newInitArgs.push_back(constOp.getResult());
        dataOffset = constOp.getResult();
      } else {
        assert(isa<IndexType>(dataOffset.get<Value>().getType()));
        newInitArgs.push_back(dataOffset.get<Value>());
      }
    }

    for (auto &dataStride : data.getStridesRef()) {
      if (isa<Attribute>(dataStride)) {
        auto constDataStride = dataStride.get<Attribute>();
        assert(isa<IntegerAttr>(constDataStride));
        auto constOp = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(
                             dyn_cast<IntegerAttr>(constDataStride).getInt()));
        newInitArgs.push_back(constOp.getResult());
        dataStride = constOp.getResult();
      } else {
        assert(isa<IndexType>(dataStride.get<Value>().getType()));
        newInitArgs.push_back(dataStride.get<Value>());
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
    if (newInitArgs[i] && newInitArgs[i].getDefiningOp<memref::ReinterpretCastOp>()) {
      SmallVector<int64_t> resultShape;
      for (auto size : data.getSizesRef()) {
        auto constSize = getConstantIntValue(size);
        assert(constSize && "expected constant size");
        resultShape.push_back(constSize.value());
      }

      // In current block data layout info, strides and offsets must be dynamic
      // value
      auto castOp = data.createCastOp(resultShape, op.getLoc(), rewriter);
      if (resultShape.size() > 1) {
        auto originalOffset = dyn_cast<Value>(data.getOffsetsRef()[0]);
        for (auto &offsets : newInitArgs) {
          if (offsets == originalOffset) {
            offsets = castOp.getOffsets()[0];
            break;
          }
        }
        data.getOffsetsRef()[0] = castOp.getOffsets()[0];
      }

      LLVM_DEBUG({
        llvm::dbgs() << "new reinterpret_cast with dynamic sizes "
                        "and offsets:";
        castOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
        llvm::dbgs() << "\n";
      });

      newInitArgs[i] = castOp.getResult();
    }
  }

  rewriter.restoreInsertionPoint(origIp);
  IRMapping mapping;

  // Create a new LoopOp that uses updated init args and same loop body
  LoopLikeOpInterface newOp;
  auto newInits = to_vector(make_filter_range(newInitArgs, [](Value v) { return v != nullptr; }));
  auto commonBodyBuilder = [&](OpBuilder &b, Location loc, ValueRange newRegionArgs, Region &region, Block::BlockArgListType regionArgs) {
    auto newArgIter = newRegionArgs.begin();
    for (const auto &[initArg, regionArg, newInitArg]: llvm::zip(op.getInits(), regionArgs, newInitArgs)) {
      if (newInitArg) {
        mapping.map(initArg, newInitArg);
        mapping.map(regionArg, *newArgIter);
        ++newArgIter;
      }
    }

    // Convert the book-keeping data structure to use the correct key and value.
    // Key is converted from init arg index to newly created block arg, and
    // Value's BlockData fields are converted from init arg to newly created block
    // arg
    for (auto [i, data] : knownPtrsTmp) {
      for (auto &offset: data.getOffsetsRef()) {
        offset = *newArgIter;
        ++newArgIter;
      }

      for (auto &stride: data.getStridesRef()) {
        stride = *newArgIter;
        ++newArgIter;
      }

      auto regionArg = regionArgs[i];
      auto key = mapping.lookupOrNull(regionArg);
      if (!key) {
        // Create MetaUse regionArg from computed offset and stride data
        key = createFromData(cast<RankedTensorType>(regionArg.getType()), data, op.getLoc(), rewriter, maskIterArgs[i]);
        mapping.map(regionArg, key);
      }
      known.insert(std::make_pair(key, data));
    }

    for (auto &bodyOp : region.getOps())
      b.clone(bodyOp, mapping);
  };
  
  if (auto forOp = dyn_cast<scf::ForOp>(op.getOperation())) {
    newOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInits,
      [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
        mapping.map(forOp.getInductionVar(), iv);
        commonBodyBuilder(b, loc, args, forOp.getRegion(), op.getRegionIterArgs());
      });
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    auto resultTypes = map_to_vector(newInits, [](auto v) { return v.getType(); });
    newOp = rewriter.create<scf::WhileOp>(
      whileOp.getLoc(), resultTypes, newInits,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        commonBodyBuilder(b, loc, args, whileOp.getBefore(), whileOp.getBeforeArguments());
      },
      [&](OpBuilder &b, Location loc, ValueRange args) {
        commonBodyBuilder(b, loc, args, whileOp.getAfter(), whileOp.getAfterArguments());
      });
  }

  // Replace only the results that correspond to the original scf.for
  auto newResultIter = newOp->result_begin();
  rewriter.setInsertionPointAfter(newOp);
  for (const auto &[res, regionArg, newIterArgIdx, mask]: llvm::zip_equal(op->getResults(), op.getRegionIterArgs(), iterArgIdxMap, maskIterArgs)) {
    if (newIterArgIdx != -1) {
      rewriter.replaceAllUsesWith(res, *newResultIter);
      ++newResultIter;
    } else {
      auto key = mapping.lookup(regionArg);
      auto data = known.at(key);
      for (auto &offset : data.getOffsetsRef())
        offset = newOp.getTiedLoopResult(cast<BlockArgument>(offset.get<Value>()));
      for (auto &stride : data.getStridesRef())
        stride = newOp.getTiedLoopResult(cast<BlockArgument>(stride.get<Value>()));
      auto newRes = createFromData(cast<RankedTensorType>(regionArg.getType()), data, op.getLoc(), rewriter, mask);
      rewriter.replaceAllUsesWith(res, newRes);
    }
  }
  rewriter.eraseOp(op);

  // Update the loop body. Manually invoke the rewrite logic on addptr and yield
  // in the loop body, so we can take advantage of the states we built up
  for (auto *region : newOp.getLoopRegions()) {
    for (auto &bodyOp : region->getOps()) {
      if (auto addptrOp = dyn_cast<triton::AddPtrOp>(bodyOp)) {
        // FIXME: Constructed adaptor here does not hold the transformed op info.
        auto adaptor = triton::AddPtrOp::Adaptor(addptrOp);
        rewriteAddPtr(addptrOp, adaptor, rewriter, known);
      } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(bodyOp)) {
        rewriteAdvanceOp(advanceOp, rewriter, known);
      } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(bodyOp)) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(makeTensorPtrOp);
        rewriteMakeTensorPtrOp(makeTensorPtrOp, rewriter.getRemappedValue(makeTensorPtrOp.getBase()), rewriter, known);
      } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(bodyOp);
                loopOp && !loopOp->hasAttr("ExtractedLoadOrStore")) {
        ConversionPatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(loopOp);
        rewriteLoopOp(loopOp, rewriter, known);
      }
    }
  }

  if (!op.getRegionIterArgs().empty()) {
    auto yieldOp = cast<scf::YieldOp>(newOp.getLoopRegions().back()->back().getTerminator());
    rewriteYieldOp(yieldOp, rewriter, blockArgIdxSet, iterArgIdxMap, known);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new loop\n";
    newOp.getOperation()->print(llvm::dbgs(),
                                OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });
}

/// @brief Rewrite the triton::AddPtrOp to handle unstructured memory access.
/// @param op The triton::AddPtrOp to be rewritten.
/// @param adaptor The adaptor of the triton::AddPtrOp, used to get operands.
/// @param rewriter The pattern rewriter used to modify the IR.
/// @param data The BlockData containing information about the memory access.
void BlockDataParser::rewriteAddPtrToUnstrucMemAcc(
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
        // Replace offset & size. Only single element.
        data.getOffsetsRef().clear();
        data.getOffsetsRef().push_back(scalarOffset);
        data.getSizesRef().clear();
        data.getSizesRef().push_back(bB.getIndexAttr(1));
        data.getStridesRef().clear();
        data.getStridesRef().push_back(bB.getIndexAttr(1));
        memref::ReinterpretCastOp castOp = data.createCastOp({1}, bLoc, bB);
        rewriter.replaceOp(op, castOp);
        // Move tt.load using this tt.addptr into this block
        loadOp->moveAfter(castOp);
        loadOp->setAttr("IndirectLoad", UnitAttr::get(op.getContext()));
        bB.create<scf::YieldOp>(bLoc, iterArgs);
      });
}

} // namespace triton
} // namespace mlir
