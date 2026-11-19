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

#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Utils/DebugUtils.h"
#include "ascend/include/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdlib>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

/**
 * Retrieves a boolean environment variable.
 * @param envVar The name of the environment variable.
 * @param defaultValue The default value to return if the variable is not set
 * or cannot be parsed.
 * @return true if the environment variable exists and its value is parsed as
 * "true", otherwise returns defaultValue. Parsing rules (case-insensitive):
 * "true" values: any non-empty string not equal to "0", "false", "no", "off"
 * is considered true. "false" values: an empty string or a string equal to
 * any of the false literals is considered false.
 */
bool getEnvBool(const char *envVar, bool defaultValue) {
  const char *val = std::getenv(envVar);
  if (val == nullptr) {
    return defaultValue; // variable not set
  }

  std::string s(val);
  // Convert to lowercase for easier comparison
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Common false literals
  if (s.empty() || s == "0" || s == "false" || s == "no" || s == "off") {
    return false;
  }
  // All other cases (including "1", "true", "yes", "on", etc.) are considered
  // true
  return true;
}

static llvm::SmallString<kFuncNameCap>
generateUniqueFuncName(ModuleOp moduleOp, llvm::StringRef funcNameBase) {
  llvm::SmallString<kFuncNameCap> funcName = funcNameBase;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = funcNameBase;
    funcName += ("_" + std::to_string(uniqueId++));
  }
  return funcName;
}

// Indirect load/store conversion normally consumes the already-converted
// memref adaptor directly. That is correct for ordinary pointer bases, but it
// skips BlockDataParser::materializePointer for an int_to_ptr base and loses
// the required one-element identity view. Only scalar pointer-preserving
// producers are followed here; tensor-of-pointers and arbitrary pointer
// arithmetic retain their established conversion path.
static bool isIntToPtrBasedScalarPointer(Value value) {
  auto pointerType = dyn_cast<triton::PointerType>(value.getType());
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return false;

  SmallPtrSet<Value, 8> visited;
  while (value && visited.insert(value).second) {
    if (value.getDefiningOp<triton::IntToPtrOp>())
      return true;

    Operation *producer = value.getDefiningOp();
    if (!producer)
      return false;
    if (auto addPtr = dyn_cast<triton::AddPtrOp>(producer)) {
      value = addPtr.getPtr();
      continue;
    }
    if (auto bitcast = dyn_cast<triton::BitcastOp>(producer)) {
      value = bitcast.getSrc();
      continue;
    }
    return false;
  }
  return false;
}

// A volatile indirect load whose scalar source is obtained through a Triton
// pointer bitcast needs the same one-element identity descriptor as the
// scalar-pointer path. Restrict this provenance check to scalar pointers and
// a pointer-preserving bitcast/addptr chain so ordinary dynamic memrefs and
// tensor-of-pointers keep their established ABI.
static bool isVolatileBitcastScalarPointer(Value value) {
  auto pointerType = dyn_cast<triton::PointerType>(value.getType());
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return false;

  SmallPtrSet<Value, 8> visited;
  bool sawBitcast = false;
  while (value && visited.insert(value).second) {
    if (auto bitcast = value.getDefiningOp<triton::BitcastOp>()) {
      sawBitcast = true;
      value = bitcast.getSrc();
      continue;
    }
    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      value = addPtr.getPtr();
      continue;
    }
    break;
  }
  return sawBitcast;
}

LogicalResult
BitcastConverter::matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  Value result;
  auto loc = op.getLoc();

  if (auto dstPtrTy = dyn_cast<triton::PointerType>(op.getType())) {
    auto srcPtrTy = cast<triton::PointerType>(op.getSrc().getType());
    auto resType =
        MemRefType::get({ShapedType::kDynamic}, dstPtrTy.getPointeeType());

    auto i1Ty = rewriter.getIntegerType(1);
    auto i8Ty = rewriter.getIntegerType(8);
    bool isI1toI8 = (srcPtrTy.getPointeeType() == i1Ty) &&
                    (dstPtrTy.getPointeeType() == i8Ty);
    // handling special case: ptr<i1> -> ptr<i8>, directly forward without
    // arith.bitcast
    if (isI1toI8) {
      // TypeConverter has already converted i1 to i8 memref,
      LLVM_DEBUG({
        llvm::dbgs()
            << "[BitcastConverter] Special i1->i8 pointer bitcast. Forward "
               "without arith.bitcast. srcConvertedTy="
            << adaptor.getSrc().getType() << "\n";
      });
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    result = rewriter.create<arith::BitcastOp>(loc, resType, adaptor.getSrc());
  } else {
    // handling normal case: bitcast between tensors/memrefs
    result =
        rewriter.create<arith::BitcastOp>(loc, op.getType(), adaptor.getSrc());
  }
  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult
TransposeConverter::matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  insertDebugNop(op.getLoc(), rewriter);
  auto src = adaptor.getSrc();
  auto res = ConverterUtils::getTransposedValue(src, op.getLoc(), rewriter,
                                                op.getOrder());
  rewriter.replaceOp(op, res);
  return success();
}

bool hasScalarPointerResult(scf::IfOp op) {
  bool hasPointerResult = false;
  for (Type resultType : op.getResultTypes()) {
    auto pointerType = dyn_cast<triton::PointerType>(resultType);
    if (!pointerType)
      continue;
    if (isa<ShapedType>(pointerType.getPointeeType()))
      return false;
    hasPointerResult = true;
  }
  return hasPointerResult;
}

bool isScalarPointerSelect(arith::SelectOp op) {
  auto pointerType = dyn_cast<triton::PointerType>(op.getType());
  return pointerType && !isa<ShapedType>(pointerType.getPointeeType());
}

// Convert a scalar Triton pointer type to the canonical memref descriptor used
// by TritonTypeConverter. It is reconstructed only after a complete byte
// address has crossed control flow, so it needs no dynamic layout.
static FailureOr<MemRefType>
getScalarPointerCarrierType(Type originalType,
                            const TypeConverter &typeConverter) {
  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return failure();

  Type convertedType = typeConverter.convertType(originalType);
  if (!convertedType)
    return failure();
  auto memrefType = dyn_cast<MemRefType>(convertedType);
  if (!memrefType)
    return failure();

  return memrefType;
}

static Type getIfResultCarrierType(Type originalType) {
  if (isa<triton::PointerType>(originalType))
    return IntegerType::get(originalType.getContext(), 64);

  // This converter exists only to carry scalar pointers as integer addresses.
  // Sibling results already have valid SSA types and must stay unchanged; using
  // the global type converter here would turn a tensor mask into a memref even
  // though its existing users still consume the tensor value.
  return originalType;
}

// Materialize a no-op-compatible memref cast to the common carrier. Returning
// failure for non-memref or incompatible values prevents the pointer transport
// pattern from silently changing element, shape, rank, or memory-space types.
static FailureOr<Value>
castToMemRefCarrier(Value value, MemRefType carrierType, Location loc,
                    ConversionPatternRewriter &rewriter) {
  if (value.getType() == carrierType)
    return value;

  auto sourceType = dyn_cast<MemRefType>(value.getType());
  if (!sourceType ||
      !memref::CastOp::areCastCompatible(sourceType, carrierType))
    return failure();

  return rewriter.create<memref::CastOp>(loc, carrierType, value).getResult();
}

// Dialect conversion may adapt a lane-local pointer descriptor such as
// `memref<1xT, strided<[1], offset: ?>>` to the canonical scalar-pointer
// carrier `memref<?xT>`. Address materialization must inspect the original
// descriptor: its dynamic layout offset is part of the represented address and
// cannot be recovered from the shape-only carrier type.
//
// Only unwrap a one-to-one memref materialization that preserves rank, element
// type, and memory space. Other unrealized casts may represent a real element
// or address-space conversion and must remain visible to their converters.
static Value unwrapPointerDescriptorMaterialization(Value value) {
  auto materialization = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!materialization || materialization.getInputs().size() != 1 ||
      materialization.getOutputs().size() != 1)
    return value;

  Value source = materialization.getInputs().front();
  auto sourceType = dyn_cast<MemRefType>(source.getType());
  auto targetType = dyn_cast<MemRefType>(value.getType());
  if (!sourceType || !targetType ||
      sourceType.getRank() != targetType.getRank() ||
      sourceType.getElementType() != targetType.getElementType() ||
      sourceType.getMemorySpace() != targetType.getMemorySpace())
    return value;

  return source;
}

// Converts a memref descriptor into the complete byte address represented by
// that descriptor. extract_aligned_pointer_as_index yields the aligned buffer
// pointer; the descriptor's element offset must therefore be converted to
// bytes and added explicitly before the address crosses control flow.
static FailureOr<Value>
materializePointerAddress(Value value, Location loc,
                          ConversionPatternRewriter &rewriter) {
  value = unwrapPointerDescriptorMaterialization(value);
  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return failure();
  Type elementType = memrefType.getElementType();
  if (!elementType.isIntOrFloat())
    return failure();

  // A canonical scalar PointerCast already stores the complete byte address
  // that IntToPtr received. Extracting its aligned pointer immediately and
  // casting it back to i64 is an identity round trip. In current failing
  // kernels this chain also survives into InjectSync, so folding it keeps a
  // redundant region-local form out of the backend. Only fold the canonical
  // one-dimensional carrier; descriptors with a non-zero offset or a non-unit
  // stride still require the general address materialization below.
  if (auto pointerCast = value.getDefiningOp<hivm::PointerCastOp>();
      pointerCast && pointerCast->hasAttr(kScalarPointerCarrierAttr)) {
    auto [strides, offset] = memrefType.getStridesAndOffset();
    if (pointerCast.getAddrs().size() == 1 && memrefType.getRank() == 1 &&
        offset == 0 && strides.size() == 1 && strides.front() == 1) {
      Value address = pointerCast.getAddrs().front();
      if (address.getType().isInteger(64))
        return address;
      if (address.getType().isIndex())
        return rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getI64Type(), address)
            .getResult();
    }
  }

  Value address =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, value);
  int64_t staticOffset = memrefType.getStridesAndOffset().second;
  if (staticOffset != 0) {
    Value elementOffset;
    if (ShapedType::isDynamic(staticOffset)) {
      elementOffset =
          rewriter.create<memref::ExtractStridedMetadataOp>(loc, value)
              .getOffset();
    } else {
      elementOffset =
          rewriter.create<arith::ConstantIndexOp>(loc, staticOffset);
    }

    int64_t elementBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;
    if (elementBytes != 1) {
      Value scale = rewriter.create<arith::ConstantIndexOp>(loc, elementBytes);
      elementOffset = rewriter.create<arith::MulIOp>(loc, elementOffset, scale);
    }
    address = rewriter.create<arith::AddIOp>(loc, address, elementOffset);
  }
  return rewriter
      .create<arith::IndexCastOp>(loc, rewriter.getI64Type(), address)
      .getResult();
}

LogicalResult
IfConverter::matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  const TypeConverter *typeConverter = getTypeConverter();
  if (!typeConverter)
    return rewriter.notifyMatchFailure(op, "requires a type converter");
  if (!hasScalarPointerResult(op))
    return failure();
  if (!op->hasAttr(kScalarPointerCarrierBoundaryAttr))
    return rewriter.notifyMatchFailure(
        op, "scalar-pointer if is missing its carrier boundary marker");

  SmallVector<Type> convertedResultTypes;
  convertedResultTypes.reserve(op.getNumResults());
  for (Type resultType : op.getResultTypes()) {
    convertedResultTypes.push_back(getIfResultCarrierType(resultType));
  }

  auto newIfOp = rewriter.create<scf::IfOp>(op.getLoc(), convertedResultTypes,
                                            adaptor.getCondition(),
                                            /*withElseRegion=*/true);
  newIfOp->setAttrs(op->getAttrs());
  newIfOp->setAttr(kScalarPointerCarrierBoundaryAttr,
                   UnitAttr::get(rewriter.getContext()));

  // Move the original regions instead of cloning them. Besides preserving
  // side effects, this keeps nested operations in the conversion driver's
  // worklist so their operands are remapped normally.
  rewriter.eraseBlock(newIfOp.thenBlock());
  rewriter.eraseBlock(newIfOp.elseBlock());
  rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                              newIfOp.getThenRegion().end());
  rewriter.inlineRegionBefore(op.getElseRegion(), newIfOp.getElseRegion(),
                              newIfOp.getElseRegion().end());

  SmallVector<Value> replacementResults;
  replacementResults.reserve(op.getNumResults());
  rewriter.setInsertionPointAfter(newIfOp);
  for (auto [originalResultType, newResult] :
       llvm::zip(op.getResultTypes(), newIfOp.getResults())) {
    if (!isa<triton::PointerType>(originalResultType)) {
      replacementResults.push_back(newResult);
      continue;
    }
    FailureOr<MemRefType> resultType =
        getScalarPointerCarrierType(originalResultType, *typeConverter);
    if (failed(resultType))
      return rewriter.notifyMatchFailure(
          op, "could not reconstruct a scalar pointer result");
    replacementResults.push_back(
        createScalarPointerCast(rewriter, op.getLoc(), *resultType, newResult)
            .getResult());
  }
  rewriter.replaceOp(op, replacementResults);
  return success();
}

LogicalResult PointerSelectConverter::matchAndRewrite(
    arith::SelectOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  if (!isScalarPointerSelect(op))
    return failure();

  const TypeConverter *typeConverter = getTypeConverter();
  if (!typeConverter)
    return rewriter.notifyMatchFailure(op, "requires a type converter");

  FailureOr<MemRefType> resultType =
      getScalarPointerCarrierType(op.getType(), *typeConverter);
  if (failed(resultType))
    return rewriter.notifyMatchFailure(
        op, "could not build a scalar pointer memref carrier");

  FailureOr<Value> trueAddress =
      materializePointerAddress(adaptor.getTrueValue(), op.getLoc(), rewriter);
  FailureOr<Value> falseAddress =
      materializePointerAddress(adaptor.getFalseValue(), op.getLoc(), rewriter);
  if (failed(trueAddress) || failed(falseAddress))
    return rewriter.notifyMatchFailure(
        op, "selected pointer values have no complete integer address");

  auto selectedAddress = rewriter.create<arith::SelectOp>(
      op.getLoc(), rewriter.getI64Type(), adaptor.getCondition(), *trueAddress,
      *falseAddress);
  selectedAddress->setAttrs(op->getAttrs());
  auto pointerCast = createScalarPointerCast(rewriter, op.getLoc(), *resultType,
                                             selectedAddress.getResult());
  rewriter.replaceOp(op, pointerCast.getResult());
  return success();
}

LogicalResult
YieldConverter::matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto parentIf = dyn_cast<scf::IfOp>(op->getParentOp());
  if (!parentIf || !parentIf->hasAttr(kScalarPointerCarrierBoundaryAttr))
    return failure();

  SmallVector<Value> convertedOperands(adaptor.getOperands());

  if (parentIf.getNumResults() != convertedOperands.size())
    return rewriter.notifyMatchFailure(op, "yield/result arity does not match");

  for (auto [index, targetType] : llvm::enumerate(parentIf.getResultTypes())) {
    Value &operand = convertedOperands[index];
    if (operand.getType() == targetType)
      continue;

    if (targetType.isInteger(64) && isa<MemRefType>(operand.getType())) {
      FailureOr<Value> address =
          materializePointerAddress(operand, op.getLoc(), rewriter);
      if (failed(address))
        return rewriter.notifyMatchFailure(
            op, "could not materialize a yielded pointer address");
      operand = *address;
      continue;
    }
    auto targetMemrefType = dyn_cast<MemRefType>(targetType);
    if (!targetMemrefType)
      return rewriter.notifyMatchFailure(
          op, "converted yield operand is incompatible with if result");

    FailureOr<Value> casted =
        castToMemRefCarrier(operand, targetMemrefType, op.getLoc(), rewriter);
    if (failed(casted))
      return rewriter.notifyMatchFailure(
          op, "converted yield operand is incompatible with if result");
    operand = *casted;
  }

  rewriter.replaceOpWithNewOp<scf::YieldOp>(op, convertedOperands);
  return success();
}

LogicalResult
AdvanceConverter::matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  return BlockDataParser::rewriteAdvanceOp(op, rewriter, known);
}

// ToDo:
// 1. Refactor MakeTensorPtrConverter and AdvanceConverter with
// memref::ReinterpretCastOp and memref::SubViewOp.
// Use recast to describe full shape of tensor, and use subview to represent
// current block tensor.
LogicalResult MakeTensorPtrConverter::matchAndRewrite(
    triton::MakeTensorPtrOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  return BlockDataParser::rewriteMakeTensorPtrOp(op, adaptor.getBase(),
                                                 rewriter, known);
}

LogicalResult PreciseDivConverter::matchAndRewrite(
    triton::PreciseDivFOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Value opa = adaptor.getX();
  Value opb = adaptor.getY();
  auto loc = op.getLoc();

  if (opa.getType() != opb.getType())
    return rewriter.notifyMatchFailure(op, "operands must have the same type");

  // Let DivFOp infer its result type from converted operands.  PreciseDivFOp
  // is valid for both scalar and tensor floating-point values; casting the
  // original result to RankedTensorType made scalar divisions produce a null
  // result type during partial conversion.
  auto divOp = rewriter.create<arith::DivFOp>(loc, opa, opb);
  rewriter.replaceOp(op, divOp.getResult());
  return success();
}

LogicalResult
SelectCanonicalizer::matchAndRewrite(arith::SelectOp op,
                                     PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // 0. Shortcut for scalars and bool type
  auto type = dyn_cast<TensorType>(op.getResult().getType());
  if (!type) {
    // do nothing non-tensor select
    return failure();
  }
  auto elementType = type.getElementType();
  if (elementType.isInteger(1)) {
    // do nothing with bool type
    return failure();
  }
  auto tensorShape = type.getShape();
  auto mask = op.getCondition();
  if (!isa<ShapedType>(mask.getType())) {
    // do nothing for scalar mask
    return failure();
  }

  // 1. Check for continuous masked loads.
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);

  if (isContMask.failed()) {
    mstate.eraseInsertedOps(op, rewriter);
    return rewriter.notifyMatchFailure(
        op, "Cannot lower continuous masked selects");
  }

  // 2. Get mask position
  MaskPosition maskPos = mstate.getMaskPosition(tensorShape);
  LLVM_DEBUG({
    llvm::dbgs() << "[SelectAnalysis] MaskPosition detected: "
                 << (maskPos == MaskPosition::Head     ? "Head"
                     : maskPos == MaskPosition::Tail   ? "Tail"
                     : maskPos == MaskPosition::Middle ? "Middle"
                                                       : "Unknown")
                 << "\n";
  });

  if (maskPos == MaskPosition::Unknown) {
    mstate.eraseInsertedOps(op, rewriter);
    return failure();
  }
  auto trueTensor = op.getTrueValue();
  auto falseTensor = op.getFalseValue();

  // 3. Slice and insert out the masked part
  if (maskPos == MaskPosition::Head) {
    // Slice out the masked part of true tensor
    auto extractSliceOp = mstate.getExtractSlice(trueTensor, loc, rewriter);

    // Insert out the sliced true tensor into false tensor
    auto insertSliceOp =
        mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter);

    LLVM_DEBUG({
      llvm::dbgs() << "  -> Created ExtractSlice: "
                   << *extractSliceOp.getOperation() << "\n"
                   << "  -> Created InsertSlice: "
                   << *insertSliceOp.getOperation() << "\n";
    });
    rewriter.replaceOp(op, insertSliceOp);
    return success();
  }

  // For Tail or Middle positions, we need to compute inverted dimensions
  // to handle the masking logic
  SmallVector<OpFoldResult> invertOffsets;
  SmallVector<OpFoldResult> invertFalseDims;
  SmallVector<OpFoldResult> invertTrueDims;
  OpFoldResult falseDimOp;
  OpFoldResult trueDimOp;
  int valDim = -1;
  for (int i = 0; i < mstate.getRank(); ++i) {
    const auto &offVal = mstate.offsets[i];
    const auto &dimVal = mstate.dims[i];
    auto constOffVal = getConstantIntValue(offVal);
    invertOffsets.push_back(rewriter.getIndexAttr(0));
    if (constOffVal.has_value() && constOffVal.value() == 0) {
      invertFalseDims.push_back(dimVal);
      invertTrueDims.push_back(dimVal);
    } else {
      assert(valDim == -1 &&
             "The offset in only one dimension can be not zero.");
      if (!constOffVal.has_value()) {
        valDim = i;
        falseDimOp = offVal;
      }

      invertFalseDims.push_back(offVal);
      trueDimOp = addOpFoldResult(offVal, dimVal, loc, rewriter,
                                  rewriter.getIndexType());
      invertTrueDims.push_back(trueDimOp);
    }
  }

  // Slice out the invert first masked part of false tensor
  auto falseExtractSliceOp = mstate.getExtractSlice(
      falseTensor, loc, rewriter, invertOffsets, invertFalseDims);
  // Insert out the sliced false tensor into true tensor
  auto trueInsertSliceOp =
      mstate.getInsertSlice(falseExtractSliceOp, trueTensor, loc, rewriter,
                            invertOffsets, invertFalseDims);
  // Slice out the invert first masked and masked part of inserted true tensor
  auto extractSliceOp = mstate.getExtractSlice(trueInsertSliceOp, loc, rewriter,
                                               invertOffsets, invertTrueDims);
  // Insert out the sliced true tensor into false tensor
  auto insertSliceOp =
      mstate.getInsertSlice(extractSliceOp, falseTensor, loc, rewriter,
                            invertOffsets, invertTrueDims);
  if (valDim != -1) {
    rewriter.setInsertionPointAfter(trueInsertSliceOp);
    assert(isa<Value>(falseDimOp) &&
           "Expected to be a runtime Value for dynamic dimension check.");
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value isNegative = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, cast<Value>(falseDimOp), zeroIndex);

    Value sizeIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, tensorShape[valDim]);
    Value isOutOfRange = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, cast<Value>(falseDimOp), sizeIndex);
    auto orOp = rewriter.create<arith::OrIOp>(loc, isNegative, isOutOfRange);
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{type},
                                           orOp.getResult(), true, true);

    Block *thenBlock = &ifOp.getThenRegion().front();
    rewriter.setInsertionPointToStart(thenBlock);
    rewriter.create<scf::YieldOp>(loc, ValueRange{falseTensor});

    Block *elseBlock = &ifOp.getElseRegion().front();
    rewriter.setInsertionPointToStart(elseBlock);
    falseExtractSliceOp->moveBefore(elseBlock, elseBlock->begin());
    trueInsertSliceOp->moveAfter(falseExtractSliceOp);
    extractSliceOp->moveAfter(trueInsertSliceOp);
    insertSliceOp->moveAfter(extractSliceOp);

    rewriter.setInsertionPointAfter(insertSliceOp);
    rewriter.create<scf::YieldOp>(loc, ValueRange{insertSliceOp.getResult()});
    rewriter.replaceOp(op, ifOp);
  } else { // static offsets
    rewriter.replaceOp(op, insertSliceOp);
  }
  LLVM_DEBUG({
    llvm::dbgs() << "  -> [invert] Created false tensor extractSlice: "
                 << *falseExtractSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created true tensor insertSlice: "
                 << *trueInsertSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created ExtractSlice: "
                 << *extractSliceOp.getOperation() << "\n"
                 << "  -> [invert] Created InsertSlice: "
                 << *insertSliceOp.getOperation() << "\n";
  });
  return success();
}

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly
 * applied on function arguments
 */
LogicalResult
BitcastCanonicalizer::matchAndRewrite(triton::BitcastOp bitcastOp,
                                      PatternRewriter &rewriter) const {
  Value castSrc = bitcastOp.getSrc();
  Value castRes = bitcastOp.getResult();
  Type castSrcTy = castSrc.getType();
  Type castSrcPtrTy = isa<ShapedType>(castSrcTy)
                          ? cast<ShapedType>(castSrcTy).getElementType()
                          : castSrcTy;
  if (!isa<triton::PointerType>(castSrcPtrTy))
    return failure();

  auto origBitwidth = getPointeeBitWidth(castSrc.getType());
  auto castBitwidth = getPointeeBitWidth(castRes.getType());

  if (origBitwidth == 1)
    origBitwidth = 8;
  if (castBitwidth == 1)
    castBitwidth = 8;
  if (origBitwidth != castBitwidth) {
    bitcastOp.emitError() << "Casting pointers with unmatched bitwidth!\n";
    return failure();
  }

  Operation *beforeCastOp = castSrc.getDefiningOp();
  if (beforeCastOp == nullptr) {
    return failure();
  }

  auto newRes =
      TypeSwitch<Operation *, FailureOr<Operation *>>(beforeCastOp)
          // before: addptr - bitcast - load/store
          // after: bitcast - addptr - load/store
          .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptrOp) {
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), addptrOp.getPtr());
            return rewriter.create<triton::AddPtrOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp.getResult(),
                addptrOp.getOffset());
          })
          .Case<triton::SplatOp>([&](triton::SplatOp splatOp) {
            Type newCastSrcTy =
                cast<RankedTensorType>(castRes.getType()).getElementType();

            Value splatSrc = splatOp.getSrc();
            Type splatSrcTy = splatSrc.getType();
            if (auto splatSrcTensorTy = dyn_cast<RankedTensorType>(splatSrcTy))
              newCastSrcTy =
                  splatSrcTensorTy.cloneWith(std::nullopt, newCastSrcTy);
            auto newCastOp = rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), newCastSrcTy, splatSrc);
            return rewriter.create<triton::SplatOp>(
                bitcastOp.getLoc(), castRes.getType(), newCastOp);
          })
          // before: bitcast - bitcast
          // after(fusion optimization): bitcast
          .Case<triton::BitcastOp>([&](triton::BitcastOp prevCastOp) {
            return rewriter.create<triton::BitcastOp>(
                bitcastOp.getLoc(), castRes.getType(), prevCastOp.getSrc());
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(bitcastOp,
                                               "Unknown bitcast pattern");
          });
  if (succeeded(newRes)) {
    // Preserve debug location in case beforeCastOp is erased below.
    insertDebugNop(beforeCastOp->getLoc(), rewriter);
    rewriter.replaceOp(bitcastOp, newRes.value());
    if (beforeCastOp->use_empty()) {
      rewriter.eraseOp(beforeCastOp);
    }
    return success();
  }
  return failure();
}

LogicalResult
FpToFpCanonicalizer::matchAndRewrite(triton::FpToFpOp op,
                                     PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value input = op.getSrc();
  auto resultType = op.getResult().getType();

  // Check if rounding mode is specified
  auto roundingMode = op.getRounding();
  if (roundingMode.has_value() &&
      roundingMode.value() != triton::RoundingMode::RTNE) {
    // Non-RTNE rounding modes (e.g., RTZ) should be handled by
    // TritonToHFusion pass Return failure here so this pattern doesn't match
    return failure();
  }

  // Handle RTNE (default) rounding mode with arith.truncf/extf. This can be
  // either a scalar conversion or a ranked tensor conversion.
  auto srcElemType = getElementTypeOrSelf(input.getType());
  auto dstElemType = getElementTypeOrSelf(resultType);
  if (!isa<FloatType>(srcElemType) || !isa<FloatType>(dstElemType)) {
    return op.emitError("FpToFp expects floating point types");
  }

  unsigned srcBitwidth = srcElemType.getIntOrFloatBitWidth();
  unsigned dstBitwidth = dstElemType.getIntOrFloatBitWidth();

  // Create round_mode attribute (RINT for RTNE)
  auto roundModeAttr = hfusion::RoundModeAttr::get(rewriter.getContext(),
                                                   hfusion::RoundMode::RINT);

  // A no-op conversion is valid only when the complete MLIR type is identical.
  // Equal element bitwidth alone is insufficient: FP8 formats such as
  // f8E4M3FN and f8E5M2 have different numerical semantics.
  if (input.getType() == resultType) {
    rewriter.replaceOp(op, input);
    return success();
  }

  if (srcBitwidth == dstBitwidth) {
    if (srcElemType == dstElemType) {
      return op.emitError(
          "fp_to_fp with identical element types has incompatible "
          "container or layout types");
    }

    // arith.extf/truncf require a strict bitwidth change. Materialize an f32
    // intermediate so the conversion remains a numerical cast in TTAdapter
    // IR and can follow the normal Bisheng lowering path.
    Type f32Type;
    if (auto tensorType = dyn_cast<RankedTensorType>(input.getType())) {
      f32Type =
          RankedTensorType::get(tensorType.getShape(), rewriter.getF32Type(),
                                tensorType.getEncoding());
    } else if (isa<FloatType>(input.getType())) {
      f32Type = rewriter.getF32Type();
    } else {
      return op.emitError("FpToFp expects a scalar or ranked tensor type");
    }

    auto extOp = rewriter.create<arith::ExtFOp>(loc, f32Type, input);
    extOp->setAttr("round_mode", roundModeAttr);
    auto truncOp =
        rewriter.create<arith::TruncFOp>(loc, resultType, extOp.getResult());
    truncOp->setAttr("round_mode", roundModeAttr);
    rewriter.replaceOp(op, truncOp.getResult());
  } else if (srcBitwidth > dstBitwidth) {
    // Downcast: use arith.truncf with round_mode=rint
    auto truncOp = rewriter.create<arith::TruncFOp>(loc, resultType, input);
    truncOp->setAttr("round_mode", roundModeAttr);
    rewriter.replaceOp(op, truncOp.getResult());
  } else {
    // Upcast: use arith.extf with round_mode=rint
    auto extOp = rewriter.create<arith::ExtFOp>(loc, resultType, input);
    extOp->setAttr("round_mode", roundModeAttr);
    rewriter.replaceOp(op, extOp.getResult());
  }

  return success();
}

void rewriteUserWithNewOrder(
    mlir::OpOperand *use, PatternRewriter &rewriter,
    llvm::SmallVector<int64_t, 8> &blkShapeI64, // 8: container size
    mlir::Location &loc, llvm::ArrayRef<int32_t> &order, size_t &orderSize) {
  Operation *user = use->getOwner();
  rewriter.setInsertionPointAfter(user);
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    auto loadResTy = loadOp.getResult().getType();
    auto loadResShapedTy = cast<ShapedType>(loadResTy);
    auto newLoadTy = loadResShapedTy.cloneWith(
        blkShapeI64, loadResShapedTy.getElementType());
    auto newLoadOp = rewriter.create<triton::LoadOp>(
        loc, newLoadTy, loadOp->getOperands(), loadOp->getAttrs());
    newLoadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG,
                       UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOp(loadOp, newLoadOp);
    // load contiguous data then permute. thus the permute order is as
    // follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(orderSize - 1 - order[i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, newLoadOp.getResult(),
        DenseI32ArrayAttr::get(loadOp.getContext(), permuteOrder));
    newLoadOp.getResult().replaceAllUsesExcept(permuteOp.getResult(),
                                               permuteOp);
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    // permute to contiguous then store. thus the permute order is as follows.
    SmallVector<int32_t, 8> permuteOrder; // 8: container size
    for (auto [i, v] : llvm::enumerate(order)) {
      permuteOrder.push_back(order[orderSize - 1 - i]);
    }
    auto permuteOp = rewriter.create<triton::TransOp>(
        loc, storeOp.getValue(),
        DenseI32ArrayAttr::get(storeOp.getContext(), permuteOrder));
    storeOp.getValue().replaceAllUsesExcept(permuteOp.getResult(), permuteOp);
    auto newStoreOp = rewriter.create<triton::StoreOp>(
        loc, storeOp.getPtr(), storeOp.getValue(), storeOp.getMask(),
        storeOp.getBoundaryCheck(), storeOp.getCache(), storeOp.getEvict());
    rewriter.replaceOp(storeOp, newStoreOp);
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    auto advanceResPtrTy =
        cast<triton::PointerType>(advanceOp.getResult().getType());
    auto advanceResShapedTy =
        cast<ShapedType>(advanceResPtrTy.getPointeeType());
    auto newAdvanceResShapedTy = advanceResShapedTy.cloneWith(
        blkShapeI64, advanceResShapedTy.getElementType());
    auto newAdvanceResPtrTy = triton::PointerType::get(
        newAdvanceResShapedTy, advanceResPtrTy.getAddressSpace());
    auto advanceOffsets = advanceOp.getOffsets();
    llvm::SmallVector<Value, 8> newAdvanceOffsets; // 8: container size
    for (int i = orderSize - 1; i >= 0; i--) {
      newAdvanceOffsets.push_back(advanceOffsets[order[i]]);
    }
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    auto newAdvanceOp = rewriter.create<triton::AdvanceOp>(
        loc, newAdvanceResPtrTy, advanceOp.getPtr(), newAdvanceOffsets);
    rewriter.replaceOp(advanceOp, newAdvanceOp);
    for (auto resUse : resUses)
      rewriteUserWithNewOrder(resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      rewriteUserWithNewOrder(&argUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
    for (auto &resUse : resultValue.getUses())
      rewriteUserWithNewOrder(&resUse, rewriter, blkShapeI64, loc, order,
                              orderSize);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

void markLoadUsers(mlir::OpOperand *use, PatternRewriter &rewriter) {
  Operation *user = use->getOwner();
  if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
    loadOp->setAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG,
                    UnitAttr::get(rewriter.getContext()));
  } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
    return;
  } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user)) {
    SmallVector<OpOperand *> resUses;
    for (auto &use : advanceOp->getUses())
      resUses.push_back(&use);
    for (auto resUse : resUses)
      markLoadUsers(resUse, rewriter);
  } else if (auto loopOp = dyn_cast<LoopLikeOpInterface>(user)) {
    auto initArg = use->get();
    auto iterArg = loopOp.getTiedLoopRegionIterArg(use);
    auto resultValue = loopOp.getTiedLoopResult(use);
    iterArg.setType(initArg.getType());
    resultValue.setType(initArg.getType());
    for (auto &argUse : iterArg.getUses())
      markLoadUsers(&argUse, rewriter);
    for (auto &resUse : resultValue.getUses())
      markLoadUsers(&resUse, rewriter);
  } else if (isa<scf::YieldOp>(user)) {
    return;
  } else {
    llvm_unreachable(
        "[MakeTensorPtrCanonicalizer] tt.make_tensor_ptr's result is "
        "not used by load/store/advance op");
  }
}

LogicalResult
MakeTensorPtrCanonicalizer::matchAndRewrite(triton::MakeTensorPtrOp op,
                                            PatternRewriter &rewriter) const {
  auto order = op.getOrder();
  auto orderSize = order.size();
  if (orderSize == 1) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order has single value.");
  }

  bool isPermuted = false;
  for (auto [first, second] : llvm::zip(order.slice(0, orderSize - 1),
                                        order.slice(1, orderSize - 1))) {
    if (first != second + 1) {
      isPermuted = true;
      break;
    }
  }

  auto loc = op.getLoc();
  auto base = op.getBase();
  auto shape = op.getShape();
  auto strides = op.getStrides();
  auto offsets = op.getOffsets();
  auto result = op.getResult();
  SmallVector<OpOperand *> opUses;

  for (auto &use : result.getUses())
    opUses.push_back(&use);
  for (auto use : opUses)
    markLoadUsers(use, rewriter);

  if (!isPermuted) {
    return rewriter.notifyMatchFailure(
        op, "make_tensor_ptr's order is contiguous.");
  }

  llvm::SmallVector<int32_t, 8> blkShapeI32;
  llvm::SmallVector<int64_t, 8> blkShapeI64;
  auto resPtrType = cast<triton::PointerType>(result.getType());
  if (auto resShapedTy = dyn_cast<ShapedType>(resPtrType.getPointeeType())) {
    auto resBlkShape = resShapedTy.getShape();
    for (auto [i, v] : llvm::enumerate(resBlkShape)) {
      auto reverseI = orderSize - 1 - i;
      blkShapeI32.push_back(resBlkShape[order[reverseI]]);
      blkShapeI64.push_back(resBlkShape[order[reverseI]]);
    }
  }

  llvm::SmallVector<Value, 8> newShape;
  llvm::SmallVector<Value, 8> newStrides;
  llvm::SmallVector<Value, 8> newOffsets;
  for (int i = orderSize - 1; i >= 0; i--) {
    newShape.push_back(shape[order[i]]);
    newStrides.push_back(strides[order[i]]);
    newOffsets.push_back(offsets[order[i]]);
  }

  llvm::SmallVector<int, 8> contiguousOrder;
  for (int i = orderSize - 1; i >= 0; i--)
    contiguousOrder.push_back(i);

  rewriter.setInsertionPoint(op);
  auto newMakeTensorPtrOp = rewriter.create<triton::MakeTensorPtrOp>(
      loc, base, ValueRange(newShape), ValueRange(newStrides),
      ValueRange(newOffsets), blkShapeI32, contiguousOrder);
  rewriter.replaceOp(op, newMakeTensorPtrOp);
  for (auto use : opUses)
    rewriteUserWithNewOrder(use, rewriter, blkShapeI64, loc, order, orderSize);
  return success();
}

LogicalResult
ReduceSingleCanonicalizer::matchAndRewrite(triton::ReduceOp reduceOp,
                                           PatternRewriter &rewriter) const {
  // This canonicalization only handles value reductions and value/index
  // reductions.  Multi-input reductions, such as Welford's
  // (mean, count, m2) reduction, must fall through to ReduceConverter's
  // extended lowering instead of terminating the compiler here.
  if (reduceOp.getSrcs().size() > 2)
    return rewriter.notifyMatchFailure(
        reduceOp, "only canonicalizes value and value/index reductions");
  auto src = reduceOp.getSrcs()[0];
  auto srcType = cast<RankedTensorType>(src.getType());
  auto srcShape = srcType.getShape();
  if (llvm::any_of(srcShape, [](auto s) { return s != 1; }))
    return rewriter.notifyMatchFailure(
        reduceOp, "reduce's srcs are not all with single element");
  auto loc = reduceOp->getLoc();

  // Handle Reduce Value
  auto res = reduceOp.getResult()[0];
  Value extracted;
  if (srcType.getRank() == 1) {
    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    extracted = rewriter.create<tensor::ExtractOp>(loc, src, zero.getResult())
                    .getResult();
  } else {
    auto resShape = cast<RankedTensorType>(res.getType()).getShape();
    auto collapseReassociationIndicesOptional =
        getReassociationIndicesForCollapse(srcShape, resShape);
    if (!collapseReassociationIndicesOptional.has_value()) {
      return rewriter.notifyMatchFailure(
          reduceOp, "Failure with getReassociationIndicesForCollapse call");
    }
    auto collapseReassociationIndices =
        collapseReassociationIndicesOptional.value();
    extracted = rewriter
                    .create<tensor::CollapseShapeOp>(
                        loc, src, collapseReassociationIndices)
                    .getResult();
  }
  res.replaceAllUsesWith(extracted);

  // Handle Reduce Index
  if (reduceOp.getSrcs().size() == 1)
    return success();

  auto resIdx = reduceOp.getResult()[1];
  auto zeroI32 =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
  if (srcType.getRank() == 1) {
    resIdx.replaceAllUsesWith(zeroI32);
  } else {
    auto resIdxShape = cast<RankedTensorType>(resIdx.getType()).getShape();
    auto initTensor = rewriter.create<tensor::EmptyOp>(loc, resIdxShape,
                                                       rewriter.getI32Type());
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroI32},
                                                  ValueRange{initTensor});
    resIdx.replaceAllUsesWith(fillOp.getResult(0));
  }

  return success();
}

LogicalResult DenseConstantConverter::matchAndRewrite(
    arith::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto denseAttr = cast<DenseElementsAttr>(op.getValue());
  auto loc = op.getLoc();
  auto constSplatOp = arith::ConstantOp::materialize(
      rewriter, denseAttr.getSplatValue<Attribute>(),
      denseAttr.getElementType(), loc);
  auto emptyOp = rewriter.create<tensor::EmptyOp>(
      loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
      denseAttr.getElementType());

  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{constSplatOp},
                                              ValueRange{emptyOp});

  return success();
}

LogicalResult
MakeRangeConverter::matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  insertDebugNop(loc, rewriter);
  auto type = cast<TensorType>(op.getResult().getType());
  auto shape = type.getShape();
  auto elementType = type.getElementType();
  auto context = op.getContext();

  assert(type.getShape().size() == 1 &&
         isa<IntegerType>(type.getElementType()) &&
         type.getElementType().getIntOrFloatBitWidth() == 32 &&
         "make range can only return 1D int32 tensor");

  SmallVector<AffineMap> indexingMaps{AffineMap::get(
      /* dimCount */ 1, /* symbolCount */ 0,
      {mlir::getAffineDimExpr(0, context)}, context)};

  auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);

  auto nestedBody = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                        ValueRange blockArgs) {
    Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
    Value res =
        nestedBuilder.create<arith::IndexCastOp>(loc, elementType, index);
    nestedBuilder.create<linalg::YieldOp>(loc, res);
  };

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, op->getResultTypes(), /* operands */ ValueRange{}, ValueRange{init},
      indexingMaps, ConverterUtils::getNParallelLoopsAttrs(1), nestedBody);

  linalgOp->setAttr("tt.from_make_range", mlir::UnitAttr::get(context));
  linalgOp->setAttr("tt.make_range_offset",
                    mlir::IntegerAttr::get(mlir::IndexType::get(context), 0));
  linalgOp->setAttr(
      "tt.make_range_size",
      mlir::IntegerAttr::get(mlir::IndexType::get(context), shape[0]));

  int32_t startVal = op.getStartAttr().getInt();
  if (startVal == 0) {
    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }

  // Apply start offset
  Value startScaler = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(startVal)));
  auto startInit = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
  Value startTensor = rewriter
                          .create<linalg::FillOp>(loc, ValueRange{startScaler},
                                                  ValueRange{startInit})
                          .getResult(0);
  auto addOp =
      rewriter.create<arith::AddIOp>(loc, linalgOp->getResult(0), startTensor);
  rewriter.replaceOp(op, addOp);
  return success();
}

LogicalResult
SplatConverter::matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  // These NOPs are transient debug anchors: DeduplicateDebugNopsPass collapses
  // them to one per source line before codegen.
  insertDebugNopForAllLines(loc, rewriter);
  auto shape = op.getType().getShape();
  auto init = rewriter.create<tensor::EmptyOp>(loc, shape,
                                               op.getType().getElementType());
  if (llvm::all_of(shape, [](int64_t dim) { return dim == 1; })) {
    SmallVector<Value> idx(shape.size(), rewriter.create<arith::ConstantOp>(
                                             loc, rewriter.getIndexAttr(0)));
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, adaptor.getSrc(), init,
                                                  idx);
  } else {
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        op, ValueRange{adaptor.getSrc()}, ValueRange{init});
  }
  return success();
}

LogicalResult
UnsplatConverter::matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto src = adaptor.getSrc();
  auto srcType = cast<RankedTensorType>(src.getType());
  auto shape = srcType.getShape();

  // Create index constants for all dimensions (all zeros since we're
  // extracting the single element)
  SmallVector<Value> indices;
  for (int64_t dim : shape) {
    indices.push_back(
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
  }

  // Extract the scalar element from the tensor
  auto elementType = srcType.getElementType();
  auto extractOp =
      rewriter.create<tensor::ExtractOp>(loc, elementType, src, indices);
  rewriter.replaceOp(op, extractOp.getResult());
  return success();
}

LogicalResult
ReshapeConverter::matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  insertDebugNop(loc, rewriter);
  auto src = op.getSrc();
  auto dst = op.getResult();
  Value shape = rewriter.create<arith::ConstantOp>(
      loc,
      rewriter.getI64TensorAttr(cast<ShapedType>(dst.getType()).getShape()));
  auto reshapeOp =
      rewriter.create<tensor::ReshapeOp>(loc, dst.getType(), src, shape);
  rewriter.replaceOp(op, reshapeOp.getResult());
  return success();
}

LogicalResult ExpandDimsConverter::matchAndRewrite(
    triton::ExpandDimsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  insertDebugNop(loc, rewriter);
  auto src = op.getSrc();
  auto resShape = cast<ShapedType>(op.getResult().getType()).getShape();
  auto axis = op.getAxis();

  SmallVector<ReassociationIndices> reassociation;

  auto src_last_dim = resShape.size() - 2;
  auto map_func = [&](unsigned i) -> ReassociationIndices {
    if (i < axis) {
      return i == src_last_dim ? ReassociationIndices{i, i + 1}
                               : ReassociationIndices{i};
    }
    return i == axis ? ReassociationIndices{i, i + 1}
                     : ReassociationIndices{i + 1};
  };

  reassociation = llvm::to_vector(
      llvm::map_range(llvm::seq<unsigned>(0, src_last_dim + 1), map_func));

  auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
      op.getLoc(), op.getResult().getType(), src, reassociation);
  rewriter.replaceOp(op, expandShapeOp.getResult());
  return success();
}

LogicalResult
ClampFConverter::matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto input = adaptor.getX();
  auto min_para = adaptor.getMin();
  auto max_para = adaptor.getMax();
  auto propagateNan_para = adaptor.getPropagateNan();

  if (auto input_type = dyn_cast<RankedTensorType>(input.getType())) {
    if (isa<FloatType>(min_para.getType())) {
      auto minEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      min_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{min_para},
                                             ValueRange{minEmptyTensor})
                     .result();
    }
    if (isa<FloatType>(max_para.getType())) {
      auto maxEmptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, input_type.getShape(), input_type.getElementType());
      max_para = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{max_para},
                                             ValueRange{maxEmptyTensor})
                     .result();
    }
  }

  if (propagateNan_para == PropagateNan::NONE) {
    auto minOp = rewriter.create<arith::MinNumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaxNumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else if (propagateNan_para == PropagateNan::ALL) {
    auto minOp = rewriter.create<arith::MinimumFOp>(loc, input, max_para);
    auto maxOp = rewriter.create<arith::MaximumFOp>(loc, min_para, minOp);
    rewriter.replaceOp(op, ValueRange{maxOp});
  } else {
    return failure();
  }

  return success();
}

// Here convert tt.broadcast to linalg.broadcast
//
// before
// %out = tt.broadcast %in : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
//
// after
// %collpased = tensor.collapse_shape %in [[0, 1], [2]] :
//                                    tensor<1x4x8xf32> into tensor<4x8xf32>
// %out = linalg.broadcast ins(%collpased : tensor<4x8xf32>)
//                         outs(%empty : tensor<128x4x8xf32>) dimensions = [0]
LogicalResult
BroadcastConverter::matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

  RankedTensorType sourceType =
      cast<RankedTensorType>(adaptor.getSrc().getType());
  RankedTensorType resultType = cast<RankedTensorType>(op.getType());
  auto elementType = resultType.getElementType();
  auto loc = op.getLoc();
  insertDebugNopForAllLines(loc, rewriter);
  auto initEmpty =
      rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elementType);

  SmallVector<int64_t> broadcastDims =
      ConverterUtils::getBroadcastDims(sourceType, resultType);
  SmallVector<int64_t> unbroadcastDims =
      ConverterUtils::getUnbroadcastDims(sourceType, resultType);

  SmallVector<ReassociationIndices> collapseReassociationIndices;
  auto collapseReassociationIndicesOptional =
      getReassociationIndicesForCollapse(sourceType.getShape(),
                                         unbroadcastDims);
  if (!collapseReassociationIndicesOptional.has_value()) {
    return rewriter.notifyMatchFailure(
        op, "Failure with getReassociationIndicesForCollapse call");
  }
  collapseReassociationIndices = collapseReassociationIndicesOptional.value();

  RankedTensorType collapseResultType =
      RankedTensorType::get(unbroadcastDims, sourceType.getElementType());

  auto collpasedOp = rewriter.create<tensor::CollapseShapeOp>(
      loc, collapseResultType, adaptor.getSrc(), collapseReassociationIndices);

  auto broadcastOp = rewriter.create<linalg::BroadcastOp>(
      loc, collpasedOp, initEmpty,
      rewriter.getDenseI64ArrayAttr(broadcastDims));

  rewriter.replaceOp(op, broadcastOp.getResults());
  return success();
}

// Reduce Converter
bool ReduceConverter::isReductionOpSupported(Operation *redOp) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
             arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
             arith::MaxUIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp>(redOp);
}

bool ReduceConverter::isMultiReductionOpSupported(Operation *redOp) {
  return isa<arith::SubFOp, arith::SubIOp, arith::DivFOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemFOp, arith::RemSIOp, arith::RemUIOp>(
      redOp);
}

Value ReduceConverter::cloneReduceOps(OpBuilder &builder, Value in, Value out,
                                      Value opIns, Value opOuts,
                                      triton::ReduceOp op) const {
  auto &reg = op->getRegion(0);
  assert(reg.getBlocks().size() == 1);
  auto &body = reg.getBlocks().front();
  auto numArguments = 2;
  assert(body.getNumArguments() == numArguments);

  Value ttIn = body.getArgument(0);
  Value ttOut = body.getArgument(1);

  IRMapping mapping;
  mapping.map(ttIn, in);
  mapping.map(ttOut, out);

  for (auto &op : body.without_terminator()) {
    builder.clone(op, mapping);
  }
  auto yield = cast<triton::ReduceReturnOp>(body.getTerminator());
  return mapping.lookup(yield->getOperand(0));
}

void ReduceConverter::checkIsNotCallOp(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  llvm::for_each(reductionOps, [](Operation *op) {
    assert(!isa<triton::CallOp>(op) && "tt.call ops expected to be inlined in "
                                       "tt.reduce body in ttir building stage");
  });
}

bool ReduceConverter::isSCFOpReduce(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  return (reductionOps.size() == 1 &&
          reductionOps.front()->getDialect()->getNamespace() ==
              scf::SCFDialect::getDialectNamespace());
}

bool ReduceConverter::isMultiOpReduce(
    const llvm::SmallVector<Operation *> &reductionOps) const {
  this->checkIsNotCallOp(reductionOps);

  return (reductionOps.size() > 1) ||
         (reductionOps.size() == 1 &&
          this->isMultiReductionOpSupported(reductionOps.front())) ||
         this->isSCFOpReduce(reductionOps);
}

Value ReduceConverter::computeReduceResultWithCompileFlag(
    OpBuilder &opBuilder, Location loc, Value lhs, Value rhs, Value source,
    Value initTensor, triton::ReduceOp op, bool compileOn91095Flag) const {
  // Original operation list (including all operations)
  auto originalReductionOps = this->getReductionOps(op);
  auto realReductionOps = this->getRealReductionOps(op);

  // If the size of the original operation list is greater than 1,
  // there are additional operations such as type conversion, and these
  // operations must be cloned.
  bool needClone = compileOn91095Flag || originalReductionOps.size() > 1;
  if (needClone) {
    return this->cloneReduceOps(opBuilder, lhs, rhs, source, initTensor, op);
  } else {
    assert(realReductionOps.size() == 1);
    auto rop = realReductionOps.front();
    return this->getReductionElement(lhs, rhs, loc, rop, opBuilder, false);
  }
}

LogicalResult
ReduceConverter::convertToTargetOp(triton::ReduceOp op,
                                   typename triton::ReduceOp::Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto source = adaptor.getOperands().front();
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto elemType = sourceType.getElementType();
  auto resType = op.getResult().front().getType();
  auto loc = op.getLoc();

  // Actual operation list (filtering type conversion operations, leaving only
  // actual reduce operations)
  auto realReductionOps = this->getRealReductionOps(op);

  bool multiOpReduce = this->isMultiOpReduce(realReductionOps);
  // Reduction of arbitrary operations isn't supported because using the first
  // element across the reduction dimension requires us to iterate over a
  // subview that skips over each first element.
  if (!multiOpReduce &&
      !this->isReductionOpSupported(realReductionOps.front())) {
    if (compileOn91095Flag) {
      llvm_unreachable("All reduction cases expected to be covered");
    }
    return rewriter.notifyMatchFailure(
        op, "Only support lowering reduction with single op and limited types "
            "of reduction");
  }

  auto rop = realReductionOps.front();
  auto ropLoc = rop->getLoc();
  auto axis = op.getAxis();
  auto isVectorReduce = sourceType.getRank() == 1;

  auto constantType = elemType;

  auto accBaseConstOp =
      multiOpReduce
          ? this->getMultiOpReductionBaseConstOp(rewriter, op, ropLoc,
                                                 constantType)
          : this->getReductionBaseConstOp(rewriter, rop, constantType);

  Value initTensor;
  if (isVectorReduce) {
    auto holder = rewriter.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get({}, constantType), ValueRange{});
    initTensor = rewriter
                     .create<linalg::FillOp>(loc, accBaseConstOp.getResult(),
                                             holder.getResult())
                     .getResult(0);
  } else {
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(resType).getShape(), constantType);
    initTensor =
        rewriter.create<linalg::FillOp>(loc, accBaseConstOp.getResult(), init)
            .getResult(0);
  }

  Value finalResult =
      rewriter
          .create<linalg::ReduceOp>(
              loc, ValueRange{source}, ValueRange{initTensor},
              SmallVector<int64_t>{axis},
              [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                assert(inputs.size() == 2);
                Value result = this->computeReduceResultWithCompileFlag(
                    opBuilder, loc, inputs[0], inputs[1], source, initTensor,
                    op, compileOn91095Flag);
                opBuilder.create<linalg::YieldOp>(loc, result);
              })
          .getResult(0);

  if (sourceType.getRank() == 1) {
    finalResult =
        rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
  }
  insertDebugNop(loc, rewriter);
  rewriter.replaceOp(op, finalResult);
  return success();
}

LogicalResult ReduceConverter::convertToTargetOpExtended(
    triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto operands = adaptor.getOperands();

  // BiShengIR's VReduce lowering only supports a single value input (or a
  // value/index pair).  A multi-input `tt.reduce`, such as Welford's
  // (mean, count, m2) reduction, cannot be represented by that VReduceOp.
  // Keep the current reduction body, but lower the static rank-1 scalar case
  // to an explicit scalar loop so it never reaches the variadic VReduce path.
  if (operands.size() > 2) {
    auto inputType = dyn_cast<RankedTensorType>(operands.front().getType());
    if (!inputType || inputType.getRank() != 1 || adaptor.getAxis() != 0 ||
        ShapedType::isDynamic(inputType.getShape()[0]) ||
        inputType.getShape()[0] < 1) {
      return rewriter.notifyMatchFailure(
          op, "multi-input reduce fallback requires static rank-1 axis-0 "
              "inputs");
    }
    if (op.getResult().size() != operands.size()) {
      return rewriter.notifyMatchFailure(
          op, "multi-input reduce results do not match input count");
    }

    for (auto [i, operand] : llvm::enumerate(operands)) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType || operandType.getShape() != inputType.getShape() ||
          op.getResult()[i].getType() != operandType.getElementType()) {
        return rewriter.notifyMatchFailure(
            op, "multi-input reduce fallback requires matching scalar "
                "results");
      }
    }

    auto reduceBlock = op.getBody();
    if (reduceBlock->getNumArguments() != 2 * operands.size() ||
        reduceBlock->getTerminator()->getNumOperands() != operands.size()) {
      return rewriter.notifyMatchFailure(
          op, "unexpected multi-input reduce combine region");
    }

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value upper =
        rewriter.create<arith::ConstantIndexOp>(loc, inputType.getShape()[0]);
    SmallVector<Value> initialValues;
    initialValues.reserve(operands.size());
    for (Value operand : operands)
      initialValues.push_back(
          rewriter.create<tensor::ExtractOp>(loc, operand, zero));

    auto loop = rewriter.create<scf::ForOp>(
        loc, one, upper, one, initialValues,
        [&](OpBuilder &builder, Location loopLoc, Value inductionVar,
            ValueRange iterArgs) {
          IRMapping mapping;
          for (auto [i, operand] : llvm::enumerate(operands)) {
            Value current = builder.create<tensor::ExtractOp>(loopLoc, operand,
                                                              inductionVar);
            mapping.map(reduceBlock->getArgument(i), current);
            mapping.map(reduceBlock->getArgument(i + operands.size()),
                        iterArgs[i]);
          }
          for (Operation &innerOp : reduceBlock->without_terminator())
            builder.clone(innerOp, mapping);

          SmallVector<Value> yielded;
          yielded.reserve(operands.size());
          for (Value value : reduceBlock->getTerminator()->getOperands())
            yielded.push_back(mapping.lookup(value));
          builder.create<scf::YieldOp>(loopLoc, yielded);
        });
    rewriter.replaceOp(op, loop.getResults());
    return success();
  }

  auto elemTypes = op.getElementTypes();

  auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
  const auto isScalarReduce = valueResultType == nullptr;

  SmallVector<Value> outputs;
  for (auto i = 0; i < op.getResult().size() && i < elemTypes.size(); i++) {
    auto result = dyn_cast<RankedTensorType>(op.getType(i));
    SmallVector<int64_t> resultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(result.getShape())};
    outputs.push_back(
        rewriter.create<tensor::EmptyOp>(loc, resultShape, elemTypes[i]));
  }

  auto linalgOp = rewriter.create<linalg::ReduceOp>(
      loc, adaptor.getOperands(), outputs,
      SmallVector<int64_t>{adaptor.getAxis()},
      [&](OpBuilder &b, Location loc, ValueRange inputs) {
        auto tritonReduceBlock = op.getBody();
        IRMapping mapping;
        mapping.map(tritonReduceBlock->getArguments(), inputs);

        for (auto &op : tritonReduceBlock->without_terminator()) {
          b.clone(op, mapping);
        }

        auto tritonYield = tritonReduceBlock->getTerminator();
        auto results =
            llvm::map_to_vector(tritonYield->getOperands(),
                                [&](Value val) { return mapping.lookup(val); });
        b.create<linalg::YieldOp>(loc, results);
      });

  auto params = getReduceWithIndexParams(op);
  if (failed(params)) {
    return rewriter.notifyMatchFailure(op, "meaningless reduce operation");
  } else if (params->withIndexType != ReduceWithIndexType::None) {
    addReduceWithIndexAttr(*params, rewriter, linalgOp);
  }

  insertDebugNop(loc, rewriter);
  if (isScalarReduce) {
    SmallVector<Value> reduceResults;
    for (auto i = 0; i < linalgOp.getResults().size() && i < elemTypes.size();
         i++) {
      reduceResults.push_back(rewriter.create<tensor::ExtractOp>(
          loc, elemTypes[i], linalgOp.getResults()[i], ValueRange{}));
    }
    rewriter.replaceOp(op, reduceResults);
  } else {
    rewriter.replaceOp(op, linalgOp);
  }
  return success();
}

bool ScanConverter::isReductionOpSupported(Operation *reductionOp) const {
  if (isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(
          reductionOp)) {
    return true;
  }
  if (compileOn91095Flag &&
      isa<arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
          arith::MinNumFOp, arith::MaxSIOp, arith::MinSIOp>(reductionOp)) {
    return true;
  }
  return false;
}

LogicalResult
ScanConverter::convertToTargetOp(triton::ScanOp op,
                                 typename triton::ScanOp::Adaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto reductionOps = this->getRealReductionOps(op);
  if (reductionOps.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No reduction op found in scan body");
  }

  llvm::SmallString<64> funcName;
  auto rop = reductionOps.front();
  if (this->isReductionOpSupported(reductionOps.front())) {
    bool propagateNan = true;
    bool isMinMax = false;
    if (isa<arith::AddFOp, arith::AddIOp>(rop)) {
      funcName = "triton_cumsum";
    } else if (isa<arith::MulFOp, arith::MulIOp>(rop)) {
      funcName = "triton_cumprod";
    } else if (isa<arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp>(rop)) {
      funcName = "triton_cummax";
      propagateNan = isa<arith::MaximumFOp>(rop);
      isMinMax = true;
    } else if (isa<arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp>(rop)) {
      funcName = "triton_cummin";
      propagateNan = isa<arith::MinimumFOp>(rop);
      isMinMax = true;
    }

    auto moduleOp = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPoint(moduleOp.getBody(),
                               std::prev(moduleOp.getBody()->end()));

    auto loc = op.getLoc();
    auto src = adaptor.getOperands().front();
    auto resTy = op.getResult().front().getType();
    // cummax/cummin take a trailing propagateNan flag; cumsum/cumprod do not.
    SmallVector<Type> argTypes{src.getType(), rewriter.getI32Type(),
                               rewriter.getI1Type()};
    if (isMinMax) {
      argTypes.push_back(rewriter.getI1Type());
    }
    auto libFnType = rewriter.getFunctionType(argTypes, {resTy});
    // Unique the name before creating the op: building a SymbolTable over a
    // module that already holds a symbol of this name -- a kernel named
    // "triton_cumsum", say -- trips SymbolTable's "uniquely named symbol
    // operations" assertion before any renaming can happen.
    auto uniqueName = generateUniqueFuncName(moduleOp, funcName);
    auto funcOp =
        rewriter.create<func::FuncOp>(loc, uniqueName.str(), libFnType);
    SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

    rewriter.setInsertionPoint(op);
    auto scanAxis = op.getAxis();
    auto scanReverse = op.getReverse();
    Value axis = rewriter.create<arith::ConstantIntOp>(loc, scanAxis, 32);
    Value reverseVal =
        rewriter.create<arith::ConstantIntOp>(loc, scanReverse, 1);
    SmallVector<Value> callOperands{src, axis, reverseVal};
    if (isMinMax) {
      callOperands.push_back(
          rewriter.create<arith::ConstantIntOp>(loc, propagateNan, 1));
    }
    auto callOp = rewriter.create<func::CallOp>(
        loc, funcOp.getSymNameAttr(), TypeRange({resTy}), callOperands);

    rewriter.replaceOp(op, callOp);

    return success();
  } else {
    // This branch is the associative_scan op.
    bool reverse = op.getReverse();

    auto loc = op.getLoc();

    Value scanInput = op.getOperand(0);

    auto srcType = mlir::dyn_cast<RankedTensorType>(scanInput.getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(
          op, "Expected RankedTensorType input for associative_scan");
    }

    auto elementType = srcType.getElementType();
    auto shape = srcType.getShape();
    int rank = shape.size();
    int axis = op.getAxis();

    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                                 std::to_string(axis));
    }

    if (op->getNumRegions() < 1 || op->getRegion(0).empty()) {
      return rewriter.notifyMatchFailure(op, "Missing combine region");
    }

    OpBuilder::InsertionGuard guard(rewriter);

    auto memrefType = MemRefType::get(shape, elementType);
    Value inputMemRef =
        rewriter.create<bufferization::ToBufferOp>(loc, memrefType, scanInput);

    // Wrap scan logic in a scope with UB address space for the output buffer.
    auto tensorResultType = RankedTensorType::get(shape, elementType);
    auto scopeOp =
        rewriter.create<scope::ScopeOp>(loc, TypeRange{tensorResultType});
    scopeOp.getBodyRegion().emplaceBlock();
    rewriter.setInsertionPointToEnd(&scopeOp.getBodyRegion().front());

    auto ubMemRefType = MemRefType::get(
        shape, elementType, nullptr,
        rewriter.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::UB));
    Value outputMemRef = rewriter.create<memref::AllocOp>(loc, ubMemRefType);

    auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
      auto startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      if (reverse) {
        startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                           shape[axis] - 1);
      }
      llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                        baseIdxsArray.end());
      llvm::SmallVector<Value> firstIdx = baseIdxs;
      if (axis <= firstIdx.size()) {
        firstIdx.insert(firstIdx.begin() + axis, startInd);
      } else {
        firstIdx.push_back(startInd);
      }

      Value firstVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRef, firstIdx);
      rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRef, firstIdx);

      Value axisSize =
          rewriter.create<memref::DimOp>(loc, inputMemRef, axis).getResult();
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 axisSize, one);
      auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

      // Create a loop only when the axis size is greater than 1.
      rewriter.setInsertionPointToStart(ifOp.thenBlock());

      auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value k = forOp.getInductionVar();
      if (reverse) {
        llvm::SmallVector<Value> fixInd;
        fixInd.push_back(
            rewriter
                .create<arith::ConstantIndexOp>(op.getLoc(), shape[axis] - 1)
                .getResult());
        fixInd.push_back(k);
        auto fixIndVal = rewriter.create<arith::SubIOp>(op.getLoc(), fixInd);
        k = fixIndVal.getResult();
      }
      llvm::SmallVector<Value> currIdx = baseIdxs;
      if (axis <= currIdx.size()) {
        currIdx.insert(currIdx.begin() + axis, k);
      } else {
        currIdx.push_back(k);
      }

      Value km1 = rewriter.create<arith::SubIOp>(loc, k, one);
      if (reverse) {
        km1 = rewriter.create<arith::AddIOp>(loc, k, one);
      }
      llvm::SmallVector<Value> prevIdx = baseIdxs;
      if (axis <= prevIdx.size()) {
        prevIdx.insert(prevIdx.begin() + axis, km1);
      } else {
        prevIdx.push_back(km1);
      }

      Value currentVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRef, currIdx);
      Value prevResult =
          rewriter.create<memref::LoadOp>(loc, outputMemRef, prevIdx);

      Region &combineRegion = op->getRegion(0);
      Block &combineBlock = combineRegion.front();
      IRMapping mapping;
      mapping.map(combineBlock.getArgument(0), prevResult);
      mapping.map(combineBlock.getArgument(1), currentVal);

      for (Operation &innerOp : combineBlock.without_terminator()) {
        rewriter.clone(innerOp, mapping);
      }

      Operation *yieldOp = combineBlock.getTerminator();
      Value resultVal = mapping.lookup(yieldOp->getOperand(0));

      rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRef, currIdx);

      rewriter.setInsertionPointAfter(ifOp);
    };

    // Constructing loops for non-scanning dimensions
    llvm::SmallVector<int> nonScanDims;
    for (int i = 0; i < rank; ++i) {
      if (i != axis)
        nonScanDims.push_back(i);
    }

    createSimpleNestedLoops(rewriter, loc, outputMemRef, nonScanDims,
                            processDimension);

    mlir::Type resultType = mlir::memref::getTensorTypeFromMemRefType(
        dyn_cast<mlir::MemRefType>(outputMemRef.getType()));
    Value outputTensor = rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, outputMemRef, true);
    rewriter.create<scope::ReturnOp>(loc, ValueRange{outputTensor});

    scopeOp->setAttr(hivm::TCoreTypeAttr::name,
                     hivm::TCoreTypeAttr::get(rewriter.getContext(),
                                              hivm::TCoreType::VECTOR));

    rewriter.replaceOp(op, scopeOp.getResult(0));
    return success();
  }
}

LogicalResult ScanConverter::convertToTargetOpExtended(
    triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  bool reverse = op.getReverse();

  // 1. Extract all input tensors (supports multiple inputs)
  auto operands = op->getOperands();
  if (operands.empty()) {
    return rewriter.notifyMatchFailure(op,
                                       "No input operands for extended scan");
  }

  // 2. Validate all inputs are of RankedTensorType
  llvm::SmallVector<RankedTensorType> inputTensTypes;
  for (auto operand : operands) {
    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must be RankedTensorType");
    }
    inputTensTypes.push_back(tensorTy);
  }

  // 3. Validate all input tensors have the same shape (scan operation
  // requires matching input dimensions)
  auto baseShape = inputTensTypes[0].getShape();
  int rank = baseShape.size();
  int axis = op.getAxis();
  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, "Invalid scan axis: " +
                                               std::to_string(axis));
  }
  for (size_t i = 1; i < inputTensTypes.size(); ++i) {
    if (inputTensTypes[i].getShape() != baseShape) {
      return rewriter.notifyMatchFailure(op,
                                         "All inputs must have the same shape");
    }
  }

  // 4. Prepare MemRefs for multiple inputs/outputs
  llvm::SmallVector<Value> inputMemRefs;
  llvm::SmallVector<Value> outputMemRefs;
  llvm::SmallVector<MemRefType> memRefTypes;
  for (size_t i = 0; i < inputTensTypes.size(); ++i) {
    auto &tensorTy = inputTensTypes[i];
    auto memRefTy =
        MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    memRefTypes.push_back(memRefTy);
    // Convert input tensors to MemRefs
    inputMemRefs.push_back(
        rewriter.create<bufferization::ToBufferOp>(loc, memRefTy, operands[i]));
    // Allocate MemRefs for outputs
    outputMemRefs.push_back(rewriter.create<memref::AllocOp>(loc, memRefTy));
  }

  // 5. Define scanning logic for multiple inputs/outputs
  LogicalResult loopResult = success();
  auto processDimension = [&](ArrayRef<Value> baseIdxsArray) {
    llvm::SmallVector<Value> baseIdxs(baseIdxsArray.begin(),
                                      baseIdxsArray.end());

    auto startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    if (reverse) {
      startInd = rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                         baseShape[axis] - 1);
    }

    llvm::SmallVector<Value> firstIdx = baseIdxs;
    if (axis <= firstIdx.size()) {
      firstIdx.insert(firstIdx.begin() + axis, startInd);
    } else {
      firstIdx.push_back(startInd);
    }

    // 5.1 Process the first element: directly copy multiple inputs to
    // multiple outputs (initialize cumulative results)
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      Value firstVal =
          rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], firstIdx);
      rewriter.create<memref::StoreOp>(loc, firstVal, outputMemRefs[i],
                                       firstIdx);
    }

    Value axisSize =
        rewriter.create<arith::ConstantIndexOp>(loc, baseShape[axis]);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               axisSize, one);
    auto ifOp = rewriter.create<scf::IfOp>(loc, cmp, false);

    // Create a loop only when the axis size is greater than 1.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());

    // Use a forward loop, but handle reverse indexing inside the loop.
    auto forOp = rewriter.create<scf::ForOp>(loc, one, axisSize, one);
    rewriter.setInsertionPointToStart(forOp.getBody());

    Value k = forOp.getInductionVar();

    if (reverse) {
      // Reverse scanning: Convert the forward loop index to the actual
      // reverse index. (axis_size - 1) - k
      Value axisSizeVal =
          rewriter.create<arith::ConstantIndexOp>(loc, baseShape[axis]);
      Value axisSizeMinusOne =
          rewriter.create<arith::SubIOp>(loc, axisSizeVal, one);
      k = rewriter.create<arith::SubIOp>(loc, axisSizeMinusOne, k);
    }

    llvm::SmallVector<Value> currIdx = baseIdxs;
    if (axis <= currIdx.size()) {
      currIdx.insert(currIdx.begin() + axis, k);
    } else {
      currIdx.push_back(k);
    }

    Value prevIndex;
    if (reverse) {
      prevIndex = rewriter.create<arith::AddIOp>(loc, k, one);
    } else {
      prevIndex = rewriter.create<arith::SubIOp>(loc, k, one);
    }

    llvm::SmallVector<Value> prevIdx = baseIdxs;
    if (axis <= prevIdx.size()) {
      prevIdx.insert(prevIdx.begin() + axis, prevIndex);
    } else {
      prevIdx.push_back(prevIndex);
    }

    // 5.4 Load current elements and previous cumulative results
    llvm::SmallVector<Value> currentVals;
    llvm::SmallVector<Value> prevResults;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      currentVals.push_back(
          rewriter.create<memref::LoadOp>(loc, inputMemRefs[i], currIdx));
      prevResults.push_back(
          rewriter.create<memref::LoadOp>(loc, outputMemRefs[i], prevIdx));
    }

    // 5.5 Bind parameters for custom reduction logic
    Region &combineRegion = op->getRegion(0);
    if (combineRegion.empty()) {
      op->emitError("Missing combine region in extended scan");
      loopResult = failure();
      return;
    }
    Block &combineBlock = combineRegion.front();
    // Validate that the number of reduction region arguments matches (number
    // of previous results + number of current elements)
    if (combineBlock.getNumArguments() != 2 * inputMemRefs.size()) {
      op->emitError("Combine region arguments mismatch with input count");
      loopResult = failure();
      return;
    }
    IRMapping mapping;
    for (size_t i = 0; i < inputMemRefs.size(); ++i) {
      // Bind previous results (previous value of the i-th output) to the i-th
      // argument of the reduction region
      mapping.map(combineBlock.getArgument(i), prevResults[i]);
      // Bind current elements (current value of the i-th input) to the i+N-th
      // argument of the reduction region (N is the number of inputs)
      mapping.map(combineBlock.getArgument(i + inputMemRefs.size()),
                  currentVals[i]);
    }

    // 5.6 Clone all operations within the reduction region
    for (Operation &innerOp : combineBlock.without_terminator()) {
      rewriter.clone(innerOp, mapping);
    }

    // 5.7 Extract reduction results and store them in outputMemRef
    Operation *yieldOp = combineBlock.getTerminator();
    if (yieldOp->getNumOperands() != outputMemRefs.size()) {
      op->emitError("Combine region returns mismatch with output count");
      loopResult = failure();
      return;
    }
    for (size_t i = 0; i < outputMemRefs.size(); ++i) {
      Value resultVal = mapping.lookup(yieldOp->getOperand(i));
      rewriter.create<memref::StoreOp>(loc, resultVal, outputMemRefs[i],
                                       currIdx);
    }

    rewriter.setInsertionPointAfter(ifOp);
  };

  // 6. Generate nested loops for non-scan dimensions
  llvm::SmallVector<int> nonScanDims;
  for (int i = 0; i < rank; ++i) {
    if (i != axis)
      nonScanDims.push_back(i);
  }
  createSimpleNestedLoops(rewriter, loc, outputMemRefs[0], nonScanDims,
                          processDimension);

  if (failed(loopResult)) {
    return failure();
  }

  // 7. Convert multiple output MemRefs back to tensors and replace the
  // original tt.scan operation
  llvm::SmallVector<Value> outputTensors;
  for (auto outputMemRef : outputMemRefs) {
    mlir::Type resultType = mlir::memref::getTensorTypeFromMemRefType(
        dyn_cast<mlir::MemRefType>(outputMemRef.getType()));
    outputTensors.push_back(rewriter.create<bufferization::ToTensorOp>(
        loc, resultType, outputMemRef, true));
  }
  rewriter.replaceOp(op, outputTensors);

  return success();
}

LogicalResult ExternElementwiseClOpConverter::matchAndRewrite(
    triton::ExternElementwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  if (!op.getPure()) {
    op->emitWarning() << "impure elementwise op!";
    return failure();
  }
  if (op.getSymbol().contains("__hmf_")) {
    Type dstTy = op.getResult().getType();
    bool isDstScalar = !isa<RankedTensorType>(dstTy);
    Type dstElemTy =
        isDstScalar ? dstTy : cast<RankedTensorType>(dstTy).getElementType();
    SmallVector<Type, 4> srcElemTys;
    SmallVector<Value, 4> srcs;
    for (auto src : op.getSrcs()) {
      if (!isa<RankedTensorType>(src.getType())) {
        src = rewriter.create<tensor::FromElementsOp>(
            op.getLoc(), RankedTensorType::get({int64_t{1}}, src.getType()),
            src);
      }
      srcs.push_back(src);
      srcElemTys.push_back(
          cast<RankedTensorType>(src.getType()).getElementType());
    }

    // extern libdevice ops -> hivm.hir.custom
    static constexpr llvm::StringLiteral libdeviceSuffixes[] = {
        "_fp32", "_fp16", "_bf16", "_i32", "_i64", "_u32", "_u64"};
    bool isLibdeviceOp =
        llvm::any_of(libdeviceSuffixes, [&](llvm::StringRef suffix) {
          return op.getSymbol().ends_with(suffix);
        });
    if (isLibdeviceOp) {
      auto originalTensorType = isDstScalar
                                    ? RankedTensorType::get({1}, dstElemTy)
                                    : cast<RankedTensorType>(dstTy);
      auto originalShape = llvm::to_vector(originalTensorType.getShape());
      bool needsExpand = originalTensorType.getRank() != 1;
      bool isI1Result = dstElemTy.isInteger(1);

      Type customElemType = dstElemTy;
      if (isI1Result)
        customElemType = rewriter.getI32Type();

      // Flatten inputs to 1D
      SmallVector<Value> collapsedInputs;

      for (Value operand : srcs) {
        RankedTensorType tensorTy = cast<RankedTensorType>(operand.getType());
        if (tensorTy.getRank() <= 1) {
          collapsedInputs.push_back(operand);
          continue;
        }

        int64_t totalSize = 1;
        for (int64_t dim : tensorTy.getShape()) {
          if (ShapedType::isDynamic(dim)) {
            totalSize = ShapedType::kDynamic;
            break;
          }
          totalSize *= dim;
        }

        ReassociationIndices indices;
        for (int i = 0; i < tensorTy.getRank(); ++i)
          indices.push_back(i);

        RankedTensorType collapse1DTy =
            RankedTensorType::get({totalSize}, tensorTy.getElementType());
        Value collapsedVal = rewriter.create<tensor::CollapseShapeOp>(
            op.getLoc(), collapse1DTy, operand, indices);
        collapsedInputs.push_back(collapsedVal);
      }

      int64_t outputSize = 1;
      for (int64_t dim : originalTensorType.getShape()) {
        if (ShapedType::isDynamic(dim)) {
          outputSize = ShapedType::kDynamic;
          break;
        }
        outputSize *= dim;
      }
      auto customOutputType =
          RankedTensorType::get({outputSize}, customElemType);

      SmallVector<Value> dynamicSizes;
      if (ShapedType::isDynamic(outputSize)) {
        for (Value collapsedInput : collapsedInputs) {
          auto collapsedInputType =
              dyn_cast<RankedTensorType>(collapsedInput.getType());
          if (!collapsedInputType || collapsedInputType.getRank() != 1 ||
              !collapsedInputType.isDynamicDim(0))
            continue;

          dynamicSizes.push_back(
              rewriter.create<tensor::DimOp>(op.getLoc(), collapsedInput, 0));
          break;
        }

        if (dynamicSizes.empty())
          return rewriter.notifyMatchFailure(
              op, "cannot determine the runtime size of a dynamic "
                  "libdevice output");
      }

      auto outputMemRefType =
          MemRefType::get(customOutputType.getShape(), customElemType);
      Value outputMemRef = rewriter.create<memref::AllocOp>(
          op.getLoc(), outputMemRefType, dynamicSizes);
      Value outputTensor = rewriter.create<bufferization::ToTensorOp>(
          op.getLoc(), customOutputType, outputMemRef, true, true);

      ReassociationIndices outputReassociation;
      if (needsExpand) {
        for (int i = 0; i < originalTensorType.getRank(); ++i)
          outputReassociation.push_back(i);
      }

      std::string sym =
          llvm::join(llvm::split(op.getSymbol().str(), "__hmf_"), "");
      auto customOp = rewriter.create<hivm::CustomOp>(
          op.getLoc(), sym, ValueRange{collapsedInputs},
          ValueRange{outputTensor});

      auto argAttrsArray = mlir::ArrayAttr::get(customOp->getContext(), {});
      auto pipeAttr =
          hivm::PipeAttr::get(customOp->getContext(), hivm::PIPE::PIPE_V);
      auto tcoreTypeAttr = hivm::TCoreTypeAttr::get(customOp->getContext(),
                                                    hivm::TCoreType::VECTOR);
      auto vfModeAttr =
          hivm::VFModeAttr::get(customOp->getContext(), hivm::VFMode::SIMD);

      customOp->setAttr("bitcode",
                        mlir::StringAttr::get(customOp->getContext(), ""));
      customOp->setAttr("hivm.pipe", pipeAttr);
      customOp->setAttr("hivm.tcore_type", tcoreTypeAttr);
      customOp->setAttr("hivm.vf_mode", vfModeAttr);
      customOp->setAttr("symbol",
                        mlir::StringAttr::get(customOp->getContext(), sym));
      customOp->setAttr("arg_attrs", argAttrsArray);
      customOp.setInlineMode(hivm::InlineMode::AlwaysInline);

      // Restore the result's shape and element type
      Value finalResult = customOp.getResults().front();
      if (needsExpand) {
        auto expandedType =
            RankedTensorType::get(originalShape, customElemType);
        finalResult = rewriter.create<tensor::ExpandShapeOp>(
            op->getLoc(), expandedType, finalResult, outputReassociation);
      }

      if (isI1Result) {
        auto resultTensorType = cast<RankedTensorType>(finalResult.getType());
        auto zeroAttr = DenseElementsAttr::get(
            RankedTensorType::get(resultTensorType.getShape(),
                                  rewriter.getI32Type()),
            0);
        Value zeroTensor =
            rewriter.create<arith::ConstantOp>(op->getLoc(), zeroAttr);
        finalResult = rewriter.create<arith::CmpIOp>(
            op->getLoc(), arith::CmpIPredicate::ne, finalResult, zeroTensor);
      }

      if (isDstScalar) {
        Value zero = rewriter.create<arith::ConstantOp>(
            op->getLoc(), rewriter.getIndexAttr(0));
        finalResult =
            rewriter.create<tensor::ExtractOp>(op->getLoc(), finalResult, zero);
      }

      rewriter.replaceOp(op, finalResult);
      return success();
    }

    // 1. get or create the declaration of external elementwise function
    FunctionType elemFuncType =
        FunctionType::get(rewriter.getContext(), srcElemTys, {dstElemTy});
    auto mod = SymbolTable::getNearestSymbolTable(op);
    auto extFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol()));
    // std::string symbol = op.getSymbol().str();
    if (!extFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&mod->getRegion(0).front());
      extFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(),
                                              op.getSymbol(), elemFuncType);
      extFunc.setPrivate();
      extFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                       UnitAttr::get(rewriter.getContext()));
      // set coreType for external func, otherwise InferFuncCoreTypePass will
      // fail
    }
    assert(isa<FunctionOpInterface>(
        SymbolTable::lookupSymbolIn(mod, op.getSymbol())));
    // 2. prepare the output tensor
    Value output;
    if (isDstScalar) {
      dstTy = RankedTensorType::get({(int64_t)1}, dstElemTy);
    }
    bool found = false;
    for (Value v : srcs) {
      if (v.getType() == dstTy) {
        found = true;
        output = v;
        break;
      }
    }
    if (!found) {
      output = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), cast<RankedTensorType>(dstTy).getShape(), dstElemTy);
    }
    // 3. create the linalg.map op
    auto mapOp = rewriter.create<linalg::MapOp>(
        loc,
        /*inputs=*/srcs,
        /*init=*/output,
        /*bodyBuilder=*/
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto elemOp = builder.create<func::CallOp>(loc,
                                                     /*name=*/op.getSymbol(),
                                                     /*resultType=*/dstElemTy,
                                                     /*operands=*/regionArgs);
          builder.create<linalg::YieldOp>(loc, elemOp->getResults());
        });
    if (isDstScalar) {
      // need to convert tensor back to scalar
      auto indexType = rewriter.getIndexType();
      Value zeroConstant = rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIntegerAttr(indexType, 0));
      auto extractOp = rewriter.create<tensor::ExtractOp>(
          loc, mapOp.getResults()[0], zeroConstant);
      rewriter.replaceOp(op, extractOp);
    } else {
      rewriter.replaceOp(op, mapOp);
    }
    return success();
  }
  return failure();
}

LogicalResult UnrealizedCastConverter::matchAndRewrite(
    UnrealizedConversionCastOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
JoinConverter::matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  Value emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                   resType.getElementType());

  auto shape = dyn_cast<RankedTensorType>(opa.getType()).getShape();
  auto sizes = llvm::map_to_vector(shape, [&](int64_t t) {
    return OpFoldResult(rewriter.getI64IntegerAttr(t));
  });
  sizes.push_back(rewriter.getI64IntegerAttr(1));

  int64_t rank = resType.getRank();

  // Set last dimension stride to 2 in layout
  // As last dimension size is always 1, last dimension stride here could be
  // either 1 or 2, while stride `2` could carry interleave trait and it's
  // convenient for next lower.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);
  rewriter.replaceOp(op, insert1);
  return success();
}

LogicalResult
CatConverter::matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Value opa = op.getLhs();
  Value opb = op.getRhs();
  auto loc = op.getLoc();

  auto resType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resType || resType.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "only support 1D cat");
  }

  auto inputTypeA = dyn_cast<RankedTensorType>(opa.getType());
  auto inputTypeB = dyn_cast<RankedTensorType>(opb.getType());
  if (!inputTypeA || !inputTypeB || inputTypeA.getRank() != 1 ||
      inputTypeB.getRank() != 1) {
    return rewriter.notifyMatchFailure(op, "inputs must be 1D tensors");
  }

  int64_t sizeA = inputTypeA.getShape()[0];
  int64_t sizeB = inputTypeB.getShape()[0];

  // Only handle the case where both inputs have size 1 (i.e., scalar-like)
  if (sizeA == 1 && sizeB == 1) {
    // Use scalar extract + insert
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                    resType.getElementType());

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value scalarA = rewriter.create<tensor::ExtractOp>(loc, opa, zero);
    Value scalarB = rewriter.create<tensor::ExtractOp>(loc, opb, zero);

    Value inserted0 =
        rewriter.create<tensor::InsertOp>(loc, scalarA, emptyOp, zero);
    Value inserted1 =
        rewriter.create<tensor::InsertOp>(loc, scalarB, inserted0, one);

    rewriter.replaceOp(op, inserted1);
    return success();
  }

  // General case: use tensor.insert_slice
  auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, resType.getShape(),
                                                  resType.getElementType());

  auto rank = resType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

  auto inputType = dyn_cast<RankedTensorType>(opa.getType());

  SmallVector<OpFoldResult> sizes =
      llvm::map_to_vector(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      });

  auto insert0 = rewriter.create<tensor::InsertSliceOp>(
      loc, opa, emptyOp, offsets, sizes, strides);

  offsets[0] =
      rewriter.getIndexAttr(inputType.getRank() ? inputType.getShape()[0] : 1);
  auto insert1 = rewriter.create<tensor::InsertSliceOp>(
      loc, opb, insert0, offsets, sizes, strides);

  rewriter.replaceOp(op, insert1);
  return success();
}

/// @brief Convert tt.gather to func.call. BiShengIR captures the func
///        with assumed semantics.
/// @param op The `triton::GatherOp` operation to be rewritten.
/// @param adaptor An adaptor for the operation's operands.
/// @param rewriter A pattern rewriter used to modify the IR.
/// @return A `LogicalResult` indicating whether the rewrite was successful.
LogicalResult
GatherConverter::matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value src = adaptor.getSrc();
  Value idx = adaptor.getIndices();
  Value res = op.getResult();
  auto gatherAxis = op.getAxis();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  llvm::SmallString<kFuncNameCap> funcName = gatherFuncNameBase;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = gatherFuncNameBase;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto resTy = res.getType();
  auto libFnType = rewriter.getFunctionType(
      {src.getType(), idx.getType(), rewriter.getI32Type()}, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  Value axis = rewriter.create<arith::ConstantIntOp>(loc, gatherAxis, 32);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}),
                                              ValueRange({src, idx, axis}));

  rewriter.replaceOp(op, callOp);

  return success();
}

LogicalResult
SplitConverter::matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  Value input = op.getSrc();
  auto loc = op.getLoc();
  auto inputType = cast<RankedTensorType>(input.getType());

  int64_t rank = inputType.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  // Similar to JoinConverter, here adjust last dimension stride
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  strides.back() = rewriter.getIndexAttr(2);

  auto outType = dyn_cast<RankedTensorType>(op.getOutLHS().getType());
  auto sizes = llvm::map_to_vector(outType.getShape(), [&](int64_t t) {
    return OpFoldResult(rewriter.getIndexAttr(t));
  });
  sizes.push_back(rewriter.getIndexAttr(1));

  auto slice0 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  offsets.back() = rewriter.getIndexAttr(1);
  auto slice1 = rewriter.create<tensor::ExtractSliceOp>(
      loc, outType, input, offsets, sizes, strides);

  SmallVector<Value, 2> slices = {slice0.getResult(), slice1.getResult()};
  rewriter.replaceOp(op, ValueRange(slices));
  return success();
}

/*
the element-wise most significant N bits of the 2N-bit product of x and y
%x:2 = arith.mulsi_extended %y, %z : tensor<4x?xi32>
*/
LogicalResult TritonMulhiuiConverter::matchAndRewrite(
    triton::MulhiUIOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value opl = op.getX();
  Value opr = op.getY();
  Value res = op.getResult();
  auto newMulOp = rewriter.create<arith::MulUIExtendedOp>(
      loc, res.getType(), res.getType(), opl, opr);
  // triton only need the high value
  rewriter.replaceOp(op, ValueRange{newMulOp.getHigh()});
  return success();
}

LogicalResult TritonPreciseSqrtConverter::matchAndRewrite(
    triton::PreciseSqrtOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<math::SqrtOp>(op, adaptor.getOperands());
  return success();
}

LogicalResult DevicePrintConverter::matchAndRewrite(
    triton::PrintOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  SmallVector<Type, 4> inputTypes;
  for (auto arg : op.getArgs()) {
    inputTypes.push_back(arg.getType());
  }
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  // See ScanConverter: the name is uniqued before the op is created so that a
  // kernel named "triton_print" cannot make SymbolTable assert.
  auto funcName = generateUniqueFuncName(moduleOp, printFuncNameBase);
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto prefixAttr = op.getPrefixAttr();
  funcOp->setAttr(prefixAttrName, prefixAttr);
  auto hexAttr = op.getHexAttr();
  funcOp->setAttr(hexAttrName, hexAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp, op.getArgs());

  rewriter.eraseOp(op);
  return success();
}

LogicalResult DeviceAssertConverter::matchAndRewrite(
    triton::AssertOp op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  auto msgAttr = op.getMessageAttr();
  // The frontend marks only sanitize_overflow assertions.  A user may choose
  // the same text for tl.device_assert, so message matching is not a safe
  // provenance test here or in the graph rewrite.
  if (op->hasAttr("tt.auto_overflow_assert")) {
    rewriter.eraseOp(op);
    return success();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));
  auto conditionType = op.getCondition().getType();

  auto libFnType = rewriter.getFunctionType({conditionType}, {});
  // See ScanConverter: the name is uniqued before the op is created so that a
  // kernel named "triton_assert" cannot make SymbolTable assert.
  auto funcName = generateUniqueFuncName(moduleOp, printFuncNameBase);
  auto funcOp =
      rewriter.create<func::FuncOp>(op.getLoc(), funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  funcOp->setAttr(msgAttrName, msgAttr);

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(op.getLoc(), funcOp,
                                ValueRange{op.getCondition()});

  rewriter.eraseOp(op);
  return success();
}

LogicalResult
MatmulConverter::matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto opa = adaptor.getA();
  auto opb = adaptor.getB();
  auto opc = adaptor.getC();
  auto dstType = cast<RankedTensorType>(op.getType());
  auto elemTy = dstType.getElementType();
  auto inputPrec = op.getInputPrecision();
  // Ascend does not support tf32;  map it to hf32 which provides similar
  // functionality.HF32 is only valid for fp32 x fp32 inputs; for other
  // dtypes, fall back to ieee.
  if (inputPrec == InputPrecision::TF32) {
    op->emitWarning("Ascend does not support tf32; map it to hf32.");
    inputPrec = InputPrecision::HF32;
  }
  if (inputPrec == InputPrecision::HF32) {
    auto opaElemTy = cast<RankedTensorType>(opa.getType()).getElementType();
    auto opbElemTy = cast<RankedTensorType>(opb.getType()).getElementType();
    if (!opaElemTy.isF32() || !opbElemTy.isF32()) {
      inputPrec = InputPrecision::IEEE;
    }
  }
  auto createOp = [&](auto &&rewriter, ValueRange operands,
                      ValueRange results) -> Operation * {
    if (dstType.getRank() == 2)
      return rewriter.template create<linalg::MatmulOp>(op.getLoc(), operands,
                                                        results);
    else if (dstType.getRank() == 3)
      return rewriter.template create<linalg::BatchMatmulOp>(op.getLoc(),
                                                             operands, results);
    llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
  };

  auto replaceOp = [&](auto &&rewriter, ValueRange operands,
                       ValueRange results) -> Operation * {
    if (dstType.getRank() == 2)
      return rewriter.template replaceOpWithNewOp<linalg::MatmulOp>(
          op, operands, results);
    else if (dstType.getRank() == 3)
      return rewriter.template replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, operands, results);
    llvm_unreachable("Datatype of DotOp operands could only be 2D or 3D");
  };

  Operation *matmulOp;
  if (mlir::isa<mlir::FloatType>(elemTy) && !elemTy.isF32()) {
    RankedTensorType opcFp32Ty =
        RankedTensorType::get(dstType.getShape(), rewriter.getF32Type());
    Value opcFp32 = rewriter.create<arith::ExtFOp>(op.getLoc(), opcFp32Ty, opc);
    matmulOp = createOp(rewriter, ValueRange{opa, opb}, ValueRange{opcFp32});
    auto roundModeAttr = hfusion::RoundModeAttr::get(rewriter.getContext(),
                                                     hfusion::RoundMode::RINT);
    auto truncOp = rewriter.replaceOpWithNewOp<arith::TruncFOp>(
        op, dstType, matmulOp->getResult(0));
    truncOp->setAttr("round_mode", roundModeAttr);
  } else {
    matmulOp = replaceOp(rewriter, ValueRange{opa, opb}, ValueRange{opc});
  }
  matmulOp->setAttr("input_precision",
                    rewriter.getStringAttr(stringifyInputPrecision(inputPrec)));
  return success();
}

// Fractal (zN) block geometry for ascend.dot lowering comes from Utils.h,
// shared with DotOp::inferReturnTypes.

// zN block column count b = 32 bytes / element bytes.
static int64_t nzBlockCol(Type elemTy) {
  return triton::ascend::kBytesPerFractalCol /
         (elemTy.getIntOrFloatBitWidth() / triton::ascend::kBitsPerByte);
}

static hivm::DataLayoutAttr dotNdLayout(MLIRContext *ctx) {
  return hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND);
}

static hivm::DataLayoutAttr dotFractalLayout(MLIRContext *ctx, int64_t blockRow,
                                             int64_t blockCol) {
  return hivm::DataLayoutAttr::get(
      ctx, hivm::DataLayout::Fractal, /*transpose=*/nullptr,
      DenseI64ArrayAttr::get(ctx, {blockRow, blockCol}));
}

// A 4D fractal dot operand -> 2D ND (no-op if already ND). Both operands are zN
// (non-transpose): lhs [K1,M1,16,b] -> [M,K] and rhs [N1,K1,16,b] -> [K,N],
// each = [d1*d2, d0*d3].
static Value dotFractalToND(ConversionPatternRewriter &rewriter, Location loc,
                            Value operand, bool isLhs) {
  auto operandTy = cast<RankedTensorType>(operand.getType());
  if (operandTy.getRank() != 4)
    return operand;
  auto blockedShape = operandTy.getShape();
  SmallVector<int64_t> ndShape{blockedShape[1] * blockedShape[2],
                               blockedShape[0] * blockedShape[3]};
  MLIRContext *ctx = rewriter.getContext();
  return rewriter.create<hivm::ConvertLayoutOp>(
      loc, RankedTensorType::get(ndShape, operandTy.getElementType()), operand,
      /*srcLayout=*/dotFractalLayout(ctx, blockedShape[2], blockedShape[3]),
      /*dstLayout=*/dotNdLayout(ctx), DenseI64ArrayAttr::get(ctx, ndShape),
      /*output_shape=*/ValueRange{});
}

// If `v` is a freshly-loaded buffer used only here -- to_tensor of an alloc
// with a single writer copy (the load) -- return that load copy so its DMA can
// be retargeted into the padded buffer (one copy, not load-then-copy); else
// null.
static memref::CopyOp findRedirectableLoad(Value v) {
  auto toTensor = v.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor || v.hasNUsesOrMore(2))
    return nullptr;
  Value srcBuf = toTensor.getBuffer();
  if (!srcBuf.getDefiningOp<memref::AllocOp>())
    return nullptr;
  memref::CopyOp loadCopy = nullptr;
  for (Operation *user : srcBuf.getUsers()) {
    if (user == toTensor.getOperation())
      continue;
    // Bail on a non-copy user, a source-use of srcBuf, or a second writer.
    auto copyOp = dyn_cast<memref::CopyOp>(user);
    if (!copyOp || copyOp.getTarget() != srcBuf || loadCopy)
      return nullptr;
    loadCopy = copyOp;
  }
  return loadCopy;
}

// Zero-pad `v`'s dimension `dim` up to `target` (no-op if already that size).
// Uses memref.alloc + fill(0) + subview + copy, NOT tensor.insert_slice: the
// latter lowers (in AscendNPU-IR AutoVectorizeV2) to a scalar copy loop that
// blows up cbuf.
static Value zeroPadDimTo(Value v, int64_t dim, int64_t target,
                          ConversionPatternRewriter &rewriter, Location loc) {
  auto t = cast<RankedTensorType>(v.getType());
  if (t.getShape()[dim] == target)
    return v;
  Type elemTy = t.getElementType();

  // Push the pad through a transpose (pad the mapped input dim, re-transpose).
  if (auto trans = v.getDefiningOp<linalg::TransposeOp>()) {
    ArrayRef<int64_t> perm = trans.getPermutation();
    Value paddedInput =
        zeroPadDimTo(trans.getInput(), perm[dim], target, rewriter, loc);
    SmallVector<int64_t> outShape(t.getShape().begin(), t.getShape().end());
    outShape[dim] = target;
    Value init = rewriter.create<tensor::EmptyOp>(loc, outShape, elemTy);
    return rewriter.create<linalg::TransposeOp>(loc, paddedInput, init, perm)
        .getResults()[0];
  }

  SmallVector<int64_t> paddedShape(t.getShape().begin(), t.getShape().end());
  paddedShape[dim] = target;

  // Allocate the padded buffer and zero-fill it.
  auto paddedMemTy = MemRefType::get(paddedShape, elemTy);
  Value padded = rewriter.create<memref::AllocOp>(loc, paddedMemTy);
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elemTy));
  rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{padded});

  // Subview covering the real (unpadded) region, then fill it with one copy.
  int64_t rank = t.getRank();
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes;
  for (int64_t d : t.getShape())
    sizes.push_back(rewriter.getIndexAttr(d));
  auto subviewTy =
      memref::SubViewOp::inferResultType(paddedMemTy, offsets, sizes, strides);
  Value subview = rewriter.create<memref::SubViewOp>(
      loc, cast<MemRefType>(subviewTy), padded, offsets, sizes, strides);

  if (memref::CopyOp loadCopy = findRedirectableLoad(v)) {
    rewriter.create<memref::CopyOp>(loc, loadCopy.getSource(), subview);
    rewriter.eraseOp(loadCopy);
  } else {
    auto srcBufferTy = MemRefType::get(t.getShape(), elemTy);
    Value srcBuffer =
        rewriter.create<bufferization::ToBufferOp>(loc, srcBufferTy, v);
    rewriter.create<memref::CopyOp>(loc, srcBuffer, subview);
  }

  return rewriter.create<bufferization::ToTensorOp>(
      loc, RankedTensorType::get(paddedShape, elemTy), padded,
      /*restrict=*/true, /*writable=*/true);
}

// Reconcile the ND operands' contraction (K) dim (a/b mutated in place). When
// exactly one operand is fractal, its K is block-rounded while the ND side may
// be shorter; zero-padding the ND side to the block-aligned K is safe (the
// extra K products are * 0). Only an in-one-block gap is padding; else error.
static LogicalResult
reconcileDotContractionK(triton::ascend::DotOp op, Value &a, Value &b,
                         Type elemTy, ConversionPatternRewriter &rewriter,
                         Location loc) {
  if (op.getFractalA() == op.getFractalB())
    return success();
  int64_t kA = cast<RankedTensorType>(a.getType()).getShape()[1]; // A_nd [M,kA]
  int64_t kB = cast<RankedTensorType>(b.getType()).getShape()[0]; // B_nd [kB,N]
  if (kA == kB)
    return success();
  int64_t paddedK = std::max(kA, kB), shortK = std::min(kA, kB);
  int64_t blockK = nzBlockCol(elemTy);
  if (paddedK % blockK != 0 || paddedK - shortK >= blockK)
    return op.emitError()
           << "dot: contraction dim mismatch (A K=" << kA << ", B K=" << kB
           << ") is larger than fractal padding; pad the operand "
              "explicitly";
  op.emitWarning() << "dot: auto zero-padding the non-fractal operand's K from "
                   << shortK << " to " << paddedK
                   << " to match the fractal operand's block-aligned K";
  a = zeroPadDimTo(a, /*dim=*/1, paddedK, rewriter, loc); // A_nd [M,K]
  b = zeroPadDimTo(b, /*dim=*/0, paddedK, rewriter, loc); // B_nd [K,N]
  return success();
}

// Decompose `ascend.dot` (D = A * B): fractal inputs -> ND, a plain
// linalg.matmul, then (if fractal_c) the ND result back to fractal.
LogicalResult
DotConverter::matchAndRewrite(triton::ascend::DotOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  auto resTy = cast<RankedTensorType>(op.getType());
  // The result carries the cube accumulator dtype (f32/i32); the matmul inputs
  // and K-padding use the operands' own (input) dtype.
  Type accTy = resTy.getElementType();

  Value a = op.getFractalA()
                ? dotFractalToND(rewriter, loc, adaptor.getA(), /*isLhs=*/true)
                : adaptor.getA();
  Value b = op.getFractalB()
                ? dotFractalToND(rewriter, loc, adaptor.getB(), /*isLhs=*/false)
                : adaptor.getB();
  Type inElemTy = cast<RankedTensorType>(a.getType()).getElementType();
  if (failed(reconcileDotContractionK(op, a, b, inElemTy, rewriter, loc)))
    return failure();

  // Uninitialized accumulator (accTy): mmadL1 (accumulate=false) overwrites it.
  int64_t numRows = cast<RankedTensorType>(a.getType()).getShape()[0];
  int64_t numCols = cast<RankedTensorType>(b.getType()).getShape()[1];
  auto ndResultTy = RankedTensorType::get({numRows, numCols}, accTy);
  Value acc = rewriter.create<tensor::EmptyOp>(
      loc, ArrayRef<int64_t>{numRows, numCols}, accTy);
  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, TypeRange{ndResultTy}, ValueRange{a, b}, ValueRange{acc});
  matmul->setAttr("input_precision", rewriter.getStringAttr("ieee"));
  Value ndResult = matmul->getResult(0);

  if (!op.getFractalC()) {
    rewriter.replaceOp(op, ndResult);
    return success();
  }
  // fractal_c: ND result -> the L0C accumulator fractal (block [16, 16]).
  Value fractalResult = rewriter.create<hivm::ConvertLayoutOp>(
      loc, resTy, ndResult, /*srcLayout=*/dotNdLayout(ctx),
      /*dstLayout=*/
      dotFractalLayout(ctx, triton::ascend::kFractalBlock,
                       triton::ascend::kFractalBlock),
      DenseI64ArrayAttr::get(ctx, resTy.getShape()), /*output_shape=*/
      ValueRange{});
  rewriter.replaceOp(op, fractalResult);
  return success();
}

LogicalResult
FlipOpConverter::matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());

  MLIRContext *ctx = rewriter.getContext();

  Type valuesTy = src.getType();
  Location loc = op.getLoc();

  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  if (!dimAttr) {
    op->emitError("missing 'dim' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  // Unique callee name: triton_flip, triton_flip_1, …
  std::string funcName = baseFuncName.str();
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName))
    funcName = (baseFuncName + Twine("_") + Twine(uniqueId++)).str();

  auto i64Ty = IntegerType::get(ctx, 64);
  auto libFnType =
      rewriter.getFunctionType({rankedSrcTy, i64Ty}, {rankedSrcTy});

  // Declare the callee
  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  // dim constant
  Value dimVal =
      rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);

  // Call the backend function
  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp = rewriter.create<func::CallOp>(
      loc, TypeRange({rankedSrcTy}), callee, ValueRange({src, dimVal}));

  Value finalValues = callOp.getResult(0);

  rewriter.replaceOp(op, {finalValues});
  return success();
}

LogicalResult
SortOpConverter::matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value src = adaptor.getSrc();
  auto rankedSrcTy = cast<RankedTensorType>(src.getType());
  auto srcElemTy = rankedSrcTy.getElementType();
  auto srcShape = rankedSrcTy.getShape();
  auto srcEnc = rankedSrcTy.getEncoding();

  MLIRContext *ctx = rewriter.getContext();

  Type backendElemTy = srcElemTy;
  if (srcElemTy.isInteger(8)) {
    backendElemTy = Float16Type::get(ctx); // i8 -> f16
  } else if (srcElemTy.isInteger(16)) {
    backendElemTy = Float32Type::get(ctx); // i16 -> f32
  }
  Type backendTensorTy = RankedTensorType::get(srcShape, backendElemTy, srcEnc);

  Type valuesTy = src.getType();

  Location loc = op.getLoc();
  auto dimAttr = op->getAttrOfType<IntegerAttr>("dim");
  auto descAttr = op->getAttrOfType<BoolAttr>("descending");
  if (!dimAttr || !descAttr) {
    op->emitError("missing 'dim' or 'descending' attribute");
    return failure();
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    op->emitError("must be inside a module");
    return failure();
  }

  llvm::SmallString<64> baseName("triton_sort");
  llvm::SmallString<64> funcName = baseName;
  int uniqueId = 0;
  while (SymbolTable::lookupSymbolIn(moduleOp, funcName)) {
    funcName = baseName;
    funcName += ("_" + std::to_string(uniqueId++));
  }

  auto i64Ty = IntegerType::get(ctx, 64);
  auto i1Ty = IntegerType::get(ctx, 1);
  auto libFnType = rewriter.getFunctionType({backendTensorTy, i64Ty, i1Ty},
                                            {backendTensorTy});

  auto moduleIP = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  rewriter.restoreInsertionPoint(moduleIP);

  Value srcForCall = src;
  if (backendElemTy != srcElemTy) {
    srcForCall = rewriter.create<arith::SIToFPOp>(loc, backendTensorTy, src);
  }

  Value dimVal =
      rewriter.create<arith::ConstantIntOp>(loc, dimAttr.getInt(), 64);
  Value descVal = rewriter.create<arith::ConstantIntOp>(
      loc, descAttr.getValue() ? 1 : 0, 1);

  auto callee = SymbolRefAttr::get(ctx, funcOp.getSymName());
  auto callOp =
      rewriter.create<func::CallOp>(loc, TypeRange({backendTensorTy}), callee,
                                    ValueRange({srcForCall, dimVal, descVal}));

  Value valuesFloat = callOp.getResult(0); // tensor<f16/f32>

  Value finalValues = valuesFloat;
  if (backendElemTy != srcElemTy) {
    finalValues = rewriter.create<arith::FPToSIOp>(loc, valuesTy, valuesFloat);
  }

  rewriter.replaceOp(op, {finalValues});

  return success();
}

LogicalResult
DotScaledConverter::matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  Value lhs = adaptor.getA();
  Value rhs = adaptor.getB();
  Value c = adaptor.getC();
  Value lhsScale = adaptor.getAScale();
  Value rhsScale = adaptor.getBScale();
  RankedTensorType dstType = cast<RankedTensorType>(op.getType());

  auto lhsElemType = op.getAElemType();
  auto rhsElemType = op.getBElemType();

  bool isFP8Input = (lhsElemType == triton::ScaleDotElemType::E4M3 ||
                     lhsElemType == triton::ScaleDotElemType::E5M2) &&
                    (rhsElemType == triton::ScaleDotElemType::E4M3 ||
                     rhsElemType == triton::ScaleDotElemType::E5M2);
  bool isFP4Input = (lhsElemType == triton::ScaleDotElemType::E2M1) &&
                    (rhsElemType == triton::ScaleDotElemType::E2M1);
  if ((isFP8Input || isFP4Input) && !lhsScale) {
    auto lhsTy = cast<RankedTensorType>(lhs.getType());
    int64_t k = lhsTy.getShape().back();
    if (lhsElemType == triton::ScaleDotElemType::E2M1 && op.getLhsKPack())
      k *= 2;
    auto defaultScaleTy = RankedTensorType::get(
        {dstType.getDimSize(dstType.getRank() - 2), (k + 31) / 32},
        rewriter.getI8Type());
    // The native MX path uses E8M0, whose encoding of one is 127.
    Value defaultScaleVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI8IntegerAttr(127));
    Value defaultScaleEmpty = rewriter.create<tensor::EmptyOp>(
        loc, defaultScaleTy.getShape(), defaultScaleTy.getElementType());
    lhsScale = rewriter
                   .create<linalg::FillOp>(loc, ValueRange{defaultScaleVal},
                                           ValueRange{defaultScaleEmpty})
                   .getResult(0);
  }
  if (isFP8Input || isFP4Input) {
    if (!rhsScale) {
      auto rhsTy = cast<RankedTensorType>(rhs.getType());
      int64_t k = rhsTy.getDimSize(rhsTy.getRank() - 2);
      if (rhsElemType == triton::ScaleDotElemType::E2M1 && op.getRhsKPack())
        k *= 2;
      auto defaultScaleTy = RankedTensorType::get(
          {dstType.getShape().back(), (k + 31) / 32}, rewriter.getI8Type());
      // E8M0 encodes a scale factor of one with exponent bias 127.
      Value defaultScaleVal = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI8IntegerAttr(127));
      Value defaultScaleEmpty = rewriter.create<tensor::EmptyOp>(
          loc, defaultScaleTy.getShape(), defaultScaleTy.getElementType());
      rhsScale = rewriter
                     .create<linalg::FillOp>(loc, ValueRange{defaultScaleVal},
                                             ValueRange{defaultScaleEmpty})
                     .getResult(0);
    }
    Value acc = c ? c
                  : rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

    // Get the ScaleDotElemType enum
    // Helper to convert
    auto convertFormat =
        [&](triton::ScaleDotElemType ty) -> mlir::hfusion::DataformatAttr {
      auto ctx = rewriter.getContext();
      switch (ty) {
      case triton::ScaleDotElemType::E2M1:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP4E2M1_T);
      case triton::ScaleDotElemType::E4M3:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP8E4M3_T);
      case triton::ScaleDotElemType::E5M2:
        return mlir::hfusion::DataformatAttr::get(
            ctx, mlir::hfusion::Dataformat::FP8E5M2_T);
      default:
        llvm_unreachable("unsupported ScaleDotElemType");
      }
    };

    Value matmulMxResult = rewriter.create<hfusion::MatMulMxOp>(
        loc, dstType, lhs, rhs, lhsScale, rhsScale, acc,
        /*lhsFormat(optional)*/ convertFormat(lhsElemType),
        /*rhsFormat(optional)*/ convertFormat(rhsElemType));

    Value finalResult = matmulMxResult;
    if (dstType.getElementType().isBF16()) {
      finalResult =
          rewriter.create<arith::TruncFOp>(loc, dstType, matmulMxResult);
    }
    rewriter.replaceOp(op, finalResult);
    return success();
  }

  if (!lhsScale) {
    return op.emitError("lhsScale is required for non-FP8 input");
  }

  RankedTensorType lhsTy = cast<RankedTensorType>(lhs.getType());
  RankedTensorType lhsScaleTy = cast<RankedTensorType>(lhsScale.getType());
  RankedTensorType rhsScaleTy =
      rhsScale ? cast<RankedTensorType>(rhsScale.getType()) : nullptr;
  RankedTensorType rhsTy = cast<RankedTensorType>(rhs.getType());

  Value lhsScaleOut;
  Value rhsScaleOut;
  Value c127 = rewriter.create<arith::ConstantOp>(
      op.getLoc(), rewriter.getI16Type(), rewriter.getI16IntegerAttr(127));
  Value c7 = rewriter.create<arith::ConstantOp>(
      op.getLoc(), rewriter.getI16Type(), rewriter.getI16IntegerAttr(7));
  Type i16Ty = rewriter.getI16Type();
  Type bf16Ty = rewriter.getBF16Type();
  Type fp16Ty = rewriter.getF16Type();
  Type fp32Ty = rewriter.getF32Type();
  bool fastMath = op.getFastMath();

  auto createNanSplat = [&](RankedTensorType tensorTy) -> Value {
    auto floatTy = cast<FloatType>(tensorTy.getElementType());
    auto nanAttr = rewriter.getFloatAttr(
        floatTy, APFloat::getNaN(floatTy.getFloatSemantics()));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, tensorTy.getShape(),
                                                   tensorTy.getElementType());
    return rewriter
        .create<linalg::FillOp>(
            loc, ValueRange{rewriter.create<arith::ConstantOp>(loc, nanAttr)},
            ValueRange{empty})
        .getResult(0);
  };

  auto createNaNMask = [&](Value scaleTensor,
                           RankedTensorType scaleTy) -> Value {
    if (scaleTy.getElementType().isIntOrIndex()) {
      auto bitWidth = scaleTy.getElementTypeBitWidth();
      auto allOnes = APInt::getAllOnes(bitWidth);
      auto sentinel = rewriter.create<arith::ConstantOp>(
          loc, scaleTy, DenseElementsAttr::get(scaleTy, allOnes));
      return rewriter
          .create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, scaleTensor,
                                 sentinel)
          .getResult();
    }

    return rewriter
        .create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNO, scaleTensor,
                               scaleTensor)
        .getResult();
  };

  auto applyNaNMask = [&](Value valueTensor, Value maskTensor) -> Value {
    auto valueTy = cast<RankedTensorType>(valueTensor.getType());
    Value nanTensor = createNanSplat(valueTy);
    return rewriter
        .create<arith::SelectOp>(loc, maskTensor, nanTensor, valueTensor)
        .getResult();
  };

  if (lhsScaleTy.getElementType().isIntOrIndex()) {
    RankedTensorType lhsScaleI16Ty =
        RankedTensorType::get(lhsScaleTy.getShape(), i16Ty);
    Value lhsScaleI16;
    lhsScaleI16 =
        rewriter.create<arith::ExtUIOp>(op.getLoc(), lhsScaleI16Ty, lhsScale);

    Value lhsShift7Empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), lhsScaleI16Ty.getShape(), i16Ty);
    Value lhsShift7 = rewriter
                          .create<linalg::FillOp>(op.getLoc(), ValueRange{c7},
                                                  ValueRange{lhsShift7Empty})
                          .getResult(0);
    Value lhsScaleI16Shifted =
        rewriter.create<arith::ShLIOp>(op.getLoc(), lhsScaleI16, lhsShift7);

    RankedTensorType lhsScaleBF16Ty =
        RankedTensorType::get(lhsScaleTy.getShape(), bf16Ty);
    Value lhsScaleBF16 = rewriter.create<arith::BitcastOp>(
        op.getLoc(), lhsScaleBF16Ty, lhsScaleI16Shifted);
    if (lhsTy.getElementType() == fp16Ty) {
      RankedTensorType lhsScaleFp32Ty =
          RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty);
      Value lhsScaleFp32 = rewriter.create<arith::ExtFOp>(
          op.getLoc(), lhsScaleFp32Ty, lhsScaleBF16);
      RankedTensorType lhsScaleFp16Ty =
          RankedTensorType::get(lhsScaleTy.getShape(), fp16Ty);
      lhsScaleOut = rewriter.create<arith::TruncFOp>(
          op.getLoc(), lhsScaleFp16Ty, lhsScaleFp32);
    } else {
      lhsScaleOut = lhsScaleBF16;
    }
  } else {
    lhsScaleOut =
        rewriter
            .create<arith::ExtFOp>(
                op.getLoc(),
                RankedTensorType::get(lhsScaleTy.getShape(), fp32Ty), lhsScale)
            .getResult();
  }

  if (rhsScale && rhsScaleTy.getElementType().isIntOrIndex()) {
    if (rhsScaleTy.getRank() != 2) {
      return op.emitError("rhsScale must be 2D for transpose");
    }

    SmallVector<int64_t> transposedShape = {rhsScaleTy.getShape()[1],
                                            rhsScaleTy.getShape()[0]};
    RankedTensorType transposedRhsScaleTy =
        RankedTensorType::get(transposedShape, rhsScaleTy.getElementType());

    Value transposedRhsScale = rewriter.create<triton::TransOp>(
        op.getLoc(), transposedRhsScaleTy, rhsScale,
        DenseI32ArrayAttr::get(rewriter.getContext(), ArrayRef<int32_t>{1, 0}));
    RankedTensorType rhsScaleI16Ty =
        RankedTensorType::get(transposedShape, i16Ty);
    Value rhsScaleI16;
    rhsScaleI16 = rewriter.create<arith::ExtUIOp>(op.getLoc(), rhsScaleI16Ty,
                                                  transposedRhsScale);

    Value rhsShift7Empty = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), rhsScaleI16Ty.getShape(), i16Ty);
    Value rhsShift7 = rewriter
                          .create<linalg::FillOp>(op.getLoc(), ValueRange{c7},
                                                  ValueRange{rhsShift7Empty})
                          .getResult(0);
    Value rhsScaleI16Shifted =
        rewriter.create<arith::ShLIOp>(op.getLoc(), rhsScaleI16, rhsShift7);

    RankedTensorType rhsScaleBF16Ty =
        RankedTensorType::get(transposedShape, bf16Ty);
    Value rhsScaleBF16 = rewriter.create<arith::BitcastOp>(
        op.getLoc(), rhsScaleBF16Ty, rhsScaleI16Shifted);

    if (rhsTy.getElementType() == fp16Ty) {
      RankedTensorType rhsScaleFp32Ty =
          RankedTensorType::get(transposedShape, fp32Ty);
      Value rhsScaleFp32 = rewriter.create<arith::ExtFOp>(
          op.getLoc(), rhsScaleFp32Ty, rhsScaleBF16);
      RankedTensorType rhsScaleFp16Ty =
          RankedTensorType::get(transposedShape, fp16Ty);
      rhsScaleOut = rewriter.create<arith::TruncFOp>(
          op.getLoc(), rhsScaleFp16Ty, rhsScaleFp32);
    } else {
      rhsScaleOut = rhsScaleBF16;
    }
    int64_t rhsD0 = rhsScaleTy.getShape()[1];
    int64_t rhsD1 = rhsScaleTy.getShape()[0];
    SmallVector<int64_t> rhsExpandedShape1 = {rhsD0, rhsD1, 1};
    RankedTensorType rhsExpandedTy1 =
        RankedTensorType::get(rhsExpandedShape1, rhsTy.getElementType());
    Value rhsExpanded1 = rewriter
                             .create<triton::ExpandDimsOp>(
                                 op.getLoc(), rhsExpandedTy1, rhsScaleOut,
                                 rewriter.getI32IntegerAttr(2))
                             .getResult();

    int64_t rhsDim1 = rhsTy.getShape()[0];
    if (rhsDim1 % rhsD0 != 0) {
      return op.emitError(
          "rhs dim0 must be an integer multiple of rhsScale dim0");
    }
    int64_t rhsD2 = rhsDim1 / rhsD0;
    SmallVector<int64_t> rhsBroadcastShape = {rhsD0, rhsD1, rhsD2};
    RankedTensorType rhsBroadcastTy =
        RankedTensorType::get(rhsBroadcastShape, rhsTy.getElementType());
    Value rhsBroadcasted = rewriter
                               .create<triton::BroadcastOp>(
                                   op.getLoc(), rhsBroadcastTy, rhsExpanded1)
                               .getResult();

    SmallVector<int32_t> transposeOrder = {0, 2, 1};
    Value transposedBroadcasted = rewriter.create<triton::TransOp>(
        op.getLoc(),
        RankedTensorType::get({rhsD0, rhsD2, rhsD1}, rhsTy.getElementType()),
        rhsBroadcasted,
        DenseI32ArrayAttr::get(rewriter.getContext(), transposeOrder));
    SmallVector<ReassociationIndices> rhsReassociation;
    rhsReassociation.push_back({0, 1});
    rhsReassociation.push_back({2});

    Value scaledRhs = rewriter
                          .create<tensor::CollapseShapeOp>(
                              op.getLoc(),
                              RankedTensorType::get({rhsD0 * rhsD2, rhsD1},
                                                    rhsTy.getElementType()),
                              transposedBroadcasted, rhsReassociation)
                          .getResult();

    rhs =
        rewriter.create<arith::MulFOp>(op.getLoc(), rhs, scaledRhs).getResult();

    if (!fastMath) {
      Value rhsScaleNaNMask =
          createNaNMask(transposedRhsScale, transposedRhsScaleTy);
      Value rhsExpandedMask =
          rewriter
              .create<triton::ExpandDimsOp>(
                  op.getLoc(),
                  RankedTensorType::get(rhsExpandedShape1,
                                        rewriter.getI1Type()),
                  rhsScaleNaNMask, rewriter.getI32IntegerAttr(2))
              .getResult();
      Value rhsBroadcastMask =
          rewriter
              .create<triton::BroadcastOp>(
                  op.getLoc(),
                  RankedTensorType::get(rhsBroadcastShape,
                                        rewriter.getI1Type()),
                  rhsExpandedMask)
              .getResult();
      Value transposedBroadcastMask =
          rewriter
              .create<triton::TransOp>(
                  op.getLoc(),
                  RankedTensorType::get({rhsD0, rhsD2, rhsD1},
                                        rewriter.getI1Type()),
                  rhsBroadcastMask,
                  DenseI32ArrayAttr::get(rewriter.getContext(), transposeOrder))
              .getResult();
      Value collapsedRhsMask =
          rewriter
              .create<tensor::CollapseShapeOp>(
                  op.getLoc(),
                  RankedTensorType::get({rhsD0 * rhsD2, rhsD1},
                                        rewriter.getI1Type()),
                  transposedBroadcastMask, rhsReassociation)
              .getResult();
      rhs = applyNaNMask(rhs, collapsedRhsMask);
    }
  }

  int64_t D0 = lhsScaleTy.getShape()[0];
  int64_t D1 = lhsScaleTy.getShape()[1];
  SmallVector<int64_t> expandedShape1 = {D0, D1, 1};
  RankedTensorType expandedTy1 =
      RankedTensorType::get(expandedShape1, lhsTy.getElementType());
  Value expanded1 =
      rewriter
          .create<triton::ExpandDimsOp>(op.getLoc(), expandedTy1, lhsScaleOut,
                                        rewriter.getI32IntegerAttr(2))
          .getResult();

  int64_t lhsDim1 = lhsTy.getShape()[1];
  if (lhsDim1 % D1 != 0) {
    return op.emitError(
        "lhs dim1 must be an integer multiple of lhsScale dim1");
  }
  int64_t D2 = lhsDim1 / D1;
  SmallVector<int64_t> broadcastShape = {D0, D1, D2};
  RankedTensorType broadcastTy =
      RankedTensorType::get(broadcastShape, lhsTy.getElementType());
  Value broadcasted =
      rewriter.create<triton::BroadcastOp>(op.getLoc(), broadcastTy, expanded1)
          .getResult();

  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0});
  reassociation.push_back({1, 2});

  Value scaledLhs =
      rewriter
          .create<tensor::CollapseShapeOp>(
              op.getLoc(),
              RankedTensorType::get({D0, D1 * D2}, lhsTy.getElementType()),
              broadcasted, reassociation)
          .getResult();

  Value scaledLhsFinal =
      rewriter.create<arith::MulFOp>(op.getLoc(), lhs, scaledLhs).getResult();

  if (!fastMath) {
    Value lhsScaleNaNMask = createNaNMask(lhsScale, lhsScaleTy);
    Value lhsExpandedMask =
        rewriter
            .create<triton::ExpandDimsOp>(
                op.getLoc(),
                RankedTensorType::get(expandedShape1, rewriter.getI1Type()),
                lhsScaleNaNMask, rewriter.getI32IntegerAttr(2))
            .getResult();
    Value lhsBroadcastMask =
        rewriter
            .create<triton::BroadcastOp>(
                op.getLoc(),
                RankedTensorType::get(broadcastShape, rewriter.getI1Type()),
                lhsExpandedMask)
            .getResult();
    Value collapsedLhsMask =
        rewriter
            .create<tensor::CollapseShapeOp>(
                op.getLoc(),
                RankedTensorType::get({D0, D1 * D2}, rewriter.getI1Type()),
                lhsBroadcastMask, reassociation)
            .getResult();
    scaledLhsFinal = applyNaNMask(scaledLhsFinal, collapsedLhsMask);
  }

  Operation *matmulOp;
  if (dstType.getRank() == 2) {
    matmulOp = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), ValueRange{scaledLhsFinal, rhs}, ValueRange{c});
  } else if (dstType.getRank() == 3) {
    matmulOp = rewriter.create<linalg::BatchMatmulOp>(
        op.getLoc(), ValueRange{scaledLhsFinal, rhs}, ValueRange{c});
  } else {
    return op.emitError("DotScaledOp only support 2D or 3D tensor");
  }

  rewriter.replaceOp(op, matmulOp->getResults());
  return success();
}

LogicalResult
IntToPtrConverter::matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto pointerType = dyn_cast<triton::PointerType>(op.getType());
  if (!pointerType || isa<ShapedType>(pointerType.getPointeeType()))
    return rewriter.notifyMatchFailure(
        op, "only scalar pointer reconstruction is supported");

  const TypeConverter *typeConverter = getTypeConverter();
  if (!typeConverter)
    return rewriter.notifyMatchFailure(op, "requires a type converter");
  auto resultType =
      dyn_cast<MemRefType>(typeConverter->convertType(op.getType()));
  if (!resultType)
    return rewriter.notifyMatchFailure(
        op, "pointer result did not convert to a memref type");

  // Rebuild the memref only after the integer address has crossed control
  // flow. Selecting an i64 address is a pure SSA operation and avoids the
  // backend interpreting a memref-valued merge as a GM-to-UB copy.
  auto pointerCast = createScalarPointerCast(rewriter, op.getLoc(), resultType,
                                             adaptor.getSrc());
  rewriter.replaceOp(op, pointerCast.getResult());
  return success();
}

LogicalResult
PtrToIntConverter::matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  FailureOr<Value> address =
      materializePointerAddress(adaptor.getSrc(), op.getLoc(), rewriter);
  if (failed(address)) {
    return rewriter.notifyMatchFailure(op, "input is not a memref type");
  }

  rewriter.replaceOp(op, *address);
  return success();
}

LogicalResult
IndexPutConverter::matchAndRewrite(triton::ascend::IndexPutOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  auto index = op.getIndex();
  auto value = op.getValue();
  auto dim = op.getDim();
  auto indexBoundary = op.getIndexBoundary();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto dstStride = adaptor.getDstStride();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }
  SmallVector<Type> inputTypes({ptrTy, index.getType(), value.getType(),
                                dim.getType(), indexBoundary.getType()});
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, index, value, dim, indexBoundary});
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  inputVals.append(dstStride.begin(), dstStride.end());
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult GatherOutToUbConverter::matchAndRewrite(
    triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto srcStride = op.getSrcStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();
  auto other = op.getOther();

  auto res = op.getResult();
  auto resTy = res.getType();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }

  SmallVector<Type> inputTypes(
      {srcTy, index.getType(), indexBoundary.getType(), dim.getType()});
  inputTypes.append(srcStride.getTypes().begin(), srcStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());
  if (other)
    inputTypes.push_back(other.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, index, indexBoundary, dim});
  inputVals.append(srcStride.begin(), srcStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  if (other)
    inputVals.push_back(other);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult ScatterUbToOutConverter::matchAndRewrite(
    triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto ptr = adaptor.getPtr();
  auto value = op.getValue();
  auto index = op.getIndex();
  auto indexBoundary = op.getIndexBoundary();
  auto dim = op.getDim();
  auto dstStride = op.getDstStride();
  auto endOffset = op.getEndOffset();
  auto startOffset = op.getStartOffset();

  // convert !tt.ptr<f32> to memref<?xf32>
  auto ptrTy = dyn_cast<MemRefType>(ptr.getType());
  if (!ptrTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for ptr");
  }

  SmallVector<Type> inputTypes({ptrTy, value.getType(), index.getType(),
                                indexBoundary.getType(), dim.getType()});
  inputTypes.append(dstStride.getTypes().begin(), dstStride.getTypes().end());
  inputTypes.append(endOffset.getTypes().begin(), endOffset.getTypes().end());
  inputTypes.append(startOffset.getTypes().begin(),
                    startOffset.getTypes().end());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({ptr, value, index, indexBoundary, dim});
  inputVals.append(dstStride.begin(), dstStride.end());
  inputVals.append(endOffset.begin(), endOffset.end());
  inputVals.append(startOffset.begin(), startOffset.end());
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndirectLoadConverter::matchAndRewrite(
    triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto offsets = op.getOffsets();
  auto mask = op.getMask();
  auto other = op.getOther();
  auto res = op.getResult();
  auto resTy = res.getType();

  // The adaptor may already expose the dynamic carrier created by
  // IntToPtrConverter.  Re-materialize that narrow provenance before the
  // MemRefType fast path, otherwise the indirect helper receives memref<?xT>
  // instead of the required one-element identity descriptor.
  if (isIntToPtrBasedScalarPointer(op.getSrc()) ||
      (op.getIsVolatile() && isVolatileBitcastScalarPointer(op.getSrc()))) {
    llvm::SmallDenseMap<Value, BlockData> known;
    FailureOr<Value> materialized =
        BlockDataParser::materializePointer(op.getSrc(), rewriter, known);
    if (failed(materialized))
      return rewriter.notifyMatchFailure(
          op, "unable to materialize int_to_ptr indirect-load source");
    src = *materialized;
  } else if (!isa<MemRefType>(src.getType())) {
    llvm::SmallDenseMap<Value, BlockData> known;
    FailureOr<Value> materialized =
        BlockDataParser::materializePointer(op.getSrc(), rewriter, known);
    if (failed(materialized))
      return rewriter.notifyMatchFailure(
          op, "unable to materialize indirect-load source as a memref");
    src = *materialized;
  }
  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());
  if (other)
    inputTypes.push_back(other.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);
  auto isVolatileAttr = rewriter.getBoolAttr(op.getIsVolatile());
  funcOp->setAttr("isVolatile", isVolatileAttr);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets});
  if (mask)
    inputVals.push_back(mask);
  if (other)
    inputVals.push_back(other);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  callOp->setAttr("isVolatile", isVolatileAttr);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult StrideLoadConverter::matchAndRewrite(
    triton::ascend::StrideLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto offset = adaptor.getOffset();
  auto other = adaptor.getOther();
  auto strides = adaptor.getStride();
  auto numels = adaptor.getNumel();
  auto resTy = op.getResult().getType();

  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }

  SmallVector<Type> inputTypes({srcTy});
  inputTypes.push_back(offset.getType());
  inputTypes.push_back(other.getType());
  for (Value stride : strides)
    inputTypes.push_back(stride.getType());
  for (Value numel : numels)
    inputTypes.push_back(numel.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {resTy});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  SmallVector<Value> inputVals({src});
  inputVals.push_back(offset);
  inputVals.push_back(other);
  inputVals.append(strides.begin(), strides.end());
  inputVals.append(numels.begin(), numels.end());

  rewriter.setInsertionPoint(op);
  auto callOp = rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(),
                                              TypeRange({resTy}), inputVals);
  rewriter.replaceOp(op, callOp);
  return success();
}

LogicalResult StrideStoreConverter::matchAndRewrite(
    triton::ascend::StrideStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto dst = adaptor.getDst();
  auto src = adaptor.getSrc();
  auto offset = adaptor.getOffset();
  auto strides = adaptor.getStride();
  auto numels = adaptor.getNumel();

  auto dstTy = dyn_cast<MemRefType>(dst.getType());
  if (!dstTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for dst");
  }

  SmallVector<Type> inputTypes({dstTy, src.getType(), offset.getType()});
  for (Value stride : strides)
    inputTypes.push_back(stride.getType());
  for (Value numel : numels)
    inputTypes.push_back(numel.getType());
  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  SmallVector<Value> inputVals({dst, src, offset});
  inputVals.append(strides.begin(), strides.end());
  inputVals.append(numels.begin(), numels.end());

  rewriter.setInsertionPoint(op);
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult IndirectStoreConverter::matchAndRewrite(
    triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto moduleOp = op->getParentOfType<ModuleOp>();
  rewriter.setInsertionPoint(moduleOp.getBody(),
                             std::prev(moduleOp.getBody()->end()));

  auto funcName = generateUniqueFuncName(moduleOp, funcNameBase);

  auto src = adaptor.getSrc();
  auto offsets = op.getOffsets();
  auto value = op.getValue();
  auto mask = op.getMask();

  // Keep the indirect-store ABI consistent with indirect-load for a scalar
  // address reconstructed by tt.int_to_ptr.  Ordinary dynamic memrefs retain
  // the existing direct path.
  if (isIntToPtrBasedScalarPointer(op.getSrc())) {
    llvm::SmallDenseMap<Value, BlockData> known;
    FailureOr<Value> materialized =
        BlockDataParser::materializePointer(op.getSrc(), rewriter, known);
    if (failed(materialized))
      return rewriter.notifyMatchFailure(
          op, "unable to materialize int_to_ptr indirect-store source");
    src = *materialized;
  } else if (!isa<MemRefType>(src.getType())) {
    llvm::SmallDenseMap<Value, BlockData> known;
    FailureOr<Value> materialized =
        BlockDataParser::materializePointer(op.getSrc(), rewriter, known);
    if (failed(materialized))
      return rewriter.notifyMatchFailure(
          op, "unable to materialize indirect-store source as a memref");
    src = *materialized;
  }
  // convert !tt.ptr<f32> to memref<?xf32>
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy) {
    return rewriter.notifyMatchFailure(op, "expected MemRefType for src");
  }
  SmallVector<Type> inputTypes({srcTy, offsets.getType(), value.getType()});
  if (mask)
    inputTypes.push_back(mask.getType());

  auto libFnType = rewriter.getFunctionType(inputTypes, {});
  auto funcOp = rewriter.create<func::FuncOp>(loc, funcName.str(), libFnType);
  SymbolTable::setSymbolVisibility(funcOp, SymbolTable::Visibility::Private);

  rewriter.setInsertionPoint(op);
  SmallVector<Value> inputVals({src, offsets, value});
  if (mask)
    inputVals.push_back(mask);
  rewriter.create<func::CallOp>(loc, funcOp.getSymNameAttr(), TypeRange({}),
                                inputVals);
  rewriter.eraseOp(op);
  return success();
}

IndexSelectSimdConverter::IndexSelectSimdConverter(MLIRContext *context)
    : OpConversionPattern<triton::ascend::IndexSelectSimdOp>(context) {}

LogicalResult IndexSelectSimdConverter::matchAndRewrite(
    triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  // Get converted operands
  Value src = adaptor.getSrc();
  Value indexTensor = adaptor.getIndex();
  auto srcShapeVals = adaptor.getSrcShape();
  auto srcOffsetVals = adaptor.getSrcOffset();
  auto readShapeAttr = op.getReadShape();
  int32_t dim = op.getDim();

  // Get result type
  auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
  static constexpr llvm::StringLiteral wasBoolToInt8AttrName =
      "was_bool_to_int8";
  bool wasBoolToInt8 = op->hasAttr(wasBoolToInt8AttrName);

  auto elemType = resultTensorType.getElementType();
  auto resultShape = resultTensorType.getShape();

  // Convert src (tt.ptr -> memref) to the correct memref shape
  // src is now memref<?xT> after type conversion, need to reinterpret to full
  // shape
  auto srcMemRefType = cast<MemRefType>(src.getType());

  // DenseI32ArrayAttr can be implicitly converted to ArrayRef<int32_t>
  ArrayRef<int32_t> readShape = readShapeAttr;

  // Helper lambda to convert Value to Index type if needed
  auto toIndexValue = [&](Value val) -> Value {
    if (!val.getType().isIndex()) {
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                 val);
    }
    return val;
  };

  // Build multi-dimensional memref type and determine static sizes
  // Merge two passes: build fullSrcShape and staticSizes in one loop
  SmallVector<int64_t> fullSrcShape;
  SmallVector<int64_t> staticSizes;
  SmallVector<Value> sizes;

  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    bool isDynamicDim = (i == static_cast<size_t>(dim) && readShape[i] == -1);
    int64_t staticSize;

    if (isDynamicDim) {
      // Dynamic dimension: readShape[i] == -1 indicates dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    } else if (auto constOp =
                   srcShapeVals[i].getDefiningOp<arith::ConstantIndexOp>()) {
      // Static dimension: use constant value
      staticSize = constOp.value();
      fullSrcShape.push_back(staticSize);
    } else {
      // Runtime value: must use dynamic
      staticSize = ShapedType::kDynamic;
      fullSrcShape.push_back(ShapedType::kDynamic);
      sizes.push_back(toIndexValue(srcShapeVals[i]));
    }
    staticSizes.push_back(staticSize);
  }
  auto fullSrcMemRefType = MemRefType::get(fullSrcShape, elemType);

  // Build static offsets, sizes, and strides for ReinterpretCastOp
  SmallVector<Value> offsets, strides;
  SmallVector<int64_t> staticOffsets, staticStrides;
  staticOffsets.push_back(0); // offsets are 0

  // Calculate static strides: stride[i] = product of all dimensions after i
  for (size_t i = 0; i < srcShapeVals.size(); ++i) {
    int64_t staticStride = 1;
    bool isDynamic = false;

    // Check if stride needs to be dynamic (any dimension after i is dynamic)
    for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
      if (staticSizes[j] == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      staticStride *= staticSizes[j];
    }

    if (isDynamic) {
      staticStride = ShapedType::kDynamic;
      // Compute stride dynamically: stride[i] = product of sizes after i
      Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (size_t j = i + 1; j < srcShapeVals.size(); ++j) {
        if (staticSizes[j] != ShapedType::kDynamic) {
          strideVal = rewriter.create<arith::MulIOp>(
              loc, strideVal,
              rewriter.create<arith::ConstantIndexOp>(loc, staticSizes[j]));
        } else {
          // Dynamic dimension: use runtime value
          strideVal = rewriter.create<arith::MulIOp>(
              loc, strideVal, toIndexValue(srcShapeVals[j]));
        }
      }
      strides.push_back(strideVal);
    }
    staticStrides.push_back(staticStride);
  }

  auto srcMemRef = rewriter.create<memref::ReinterpretCastOp>(
      loc, fullSrcMemRefType, src, offsets, sizes, strides, staticOffsets,
      staticSizes, staticStrides);

  // Allocate output buffer
  auto resultMemRefType = MemRefType::get(resultShape, elemType);
  auto outputBuffer = rewriter.create<memref::AllocOp>(loc, resultMemRefType);

  // Get indices tensor type for extracting
  auto indicesTensorType = cast<RankedTensorType>(indexTensor.getType());
  int64_t numIndices = indicesTensorType.getShape()[0];

  // Create for loop
  auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto numIndicesVal = rewriter.create<arith::ConstantIndexOp>(loc, numIndices);
  auto stepOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto forOp =
      rewriter.create<scf::ForOp>(loc, zeroIdx, numIndicesVal, stepOne);

  // Mark as parallel loop
  forOp->setAttr("hivm.parallel_loop", rewriter.getUnitAttr());

  // Build loop body
  Block *loopBody = forOp.getBody();
  auto savedInsertionPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(loopBody);

  // Remove the terminator temporarily
  Operation *terminator = &loopBody->back();
  rewriter.setInsertionPoint(terminator);

  Value iv = forOp.getInductionVar();

  // Extract index from indices tensor
  Value selectedIdx =
      rewriter.create<tensor::ExtractOp>(loc, indexTensor, ValueRange{iv});
  Value selectedIdxAsIndex = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIndexType(), selectedIdx);

  // Build source subview offsets/sizes/strides
  SmallVector<OpFoldResult> srcSubviewOffsets, srcSubviewSizes,
      srcSubviewStrides;

  for (size_t i = 0; i < srcOffsetVals.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      // Use the selected index for this dimension
      srcSubviewOffsets.push_back(selectedIdxAsIndex);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      // Use provided offset and read size for other dimensions
      Value offsetVal = srcOffsetVals[i];
      if (!offsetVal.getType().isIndex()) {
        offsetVal = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offsetVal);
      }
      srcSubviewOffsets.push_back(offsetVal);
      srcSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    srcSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto srcSubview = rewriter.create<memref::SubViewOp>(
      loc, srcMemRef, srcSubviewOffsets, srcSubviewSizes, srcSubviewStrides);

  // Build destination subview
  SmallVector<OpFoldResult> dstSubviewOffsets, dstSubviewSizes,
      dstSubviewStrides;
  for (size_t i = 0; i < resultShape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      dstSubviewOffsets.push_back(iv);
      dstSubviewSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      dstSubviewOffsets.push_back(rewriter.getIndexAttr(0));
      dstSubviewSizes.push_back(rewriter.getIndexAttr(readShape[i]));
    }
    dstSubviewStrides.push_back(rewriter.getIndexAttr(1));
  }

  auto dstSubview = rewriter.create<memref::SubViewOp>(
      loc, outputBuffer, dstSubviewOffsets, dstSubviewSizes, dstSubviewStrides);

  // Check if index_select is on the trailing axis (last dimension)
  if (static_cast<size_t>(dim) == fullSrcShape.size() - 1) {
    // For index_select on the trailing axis, mark as discrete memory access
    // This degrades to scalar read/write handling to avoid alignment issues
    auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    if (wasBoolToInt8)
      copyOp->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));
    copyOp->setAttr(ConverterUtils::discreteAttrName, rewriter.getUnitAttr());
  } else {
    // For index_select on non-trailing axes, add stride alignment annotation
    // This tells the backend to handle address alignment for DMA operations
    auto dstMarkOp = rewriter.create<annotation::MarkOp>(loc, dstSubview);
    dstMarkOp->setAttr(
        "hfusion.stride_align_dims",
        rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(dim)}));
    dstMarkOp->setAttr("hfusion.stride_align_value_in_byte",
                       rewriter.getDenseI32ArrayAttr({32}));

    // Copy from source to destination
    auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    if (wasBoolToInt8)
      copyOp->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));
  }

  // Restore insertion point
  rewriter.restoreInsertionPoint(savedInsertionPoint);

  // Convert memref to tensor
  auto resultTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, resultTensorType, outputBuffer, true, true);
  if (wasBoolToInt8)
    resultTensor->setAttr(wasBoolToInt8AttrName, rewriter.getBoolAttr(true));

  // Mark as index_select_simd
  resultTensor->setAttr("index_select_simd", rewriter.getUnitAttr());

  // Replace the original op
  rewriter.replaceOp(op, resultTensor);

  return success();
}

LogicalResult
HistogramConverter::matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  Value input = adaptor.getSrc();
  Value mask = adaptor.getMask();
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType || !resultType.hasStaticShape()) {
    return rewriter.notifyMatchFailure(op,
                                       "expected static shaped tensor result");
  }

  int64_t numBins = resultType.getDimSize(0);

  auto outputTensor = rewriter.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());

  auto zeroVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getIntegerAttr(resultType.getElementType(), 0));
  auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroVal},
                                                ValueRange{outputTensor});

  Value numBinsVal = rewriter.create<arith::ConstantIntOp>(loc, numBins, 64);

  SmallVector<Value, 3> inputs = {input, numBinsVal};
  if (mask) {
    inputs.push_back(mask);
  }

  auto customOp = rewriter.create<hivm::CustomOp>(
      loc, TypeRange{resultType}, "__builtin_histogram", ValueRange{inputs},
      ValueRange{fillOp.getResult(0)}, ValueRange{});

  customOp->setAttr("symbol", rewriter.getStringAttr("__builtin_histogram"));
  customOp->setAttr(
      "hivm.tcore_type",
      hivm::TCoreTypeAttr::get(rewriter.getContext(), hivm::TCoreType::VECTOR));
  customOp->setAttr("hivm.vf_mode", hivm::VFModeAttr::get(rewriter.getContext(),
                                                          hivm::VFMode::SIMT));
  customOp->setAttr("hivm.pipe", hivm::PipeAttr::get(rewriter.getContext(),
                                                     hivm::PIPE::PIPE_V));
  customOp->setAttr("gm_addr_args_indices",
                    DenseI32ArrayAttr::get(rewriter.getContext(), {}));

  rewriter.replaceOp(op, customOp->getResult(0));
  return success();
}
// ===----------------------------------------------------------------------===
// MapElementwiseDecomposeConverter
// ===----------------------------------------------------------------------===

// Promote a single scalar op to tensor-level.  Returns the tensor result
// values after recording them in valueMap.
SmallVector<Value> MapElementwiseDecomposeConverter::promoteOp(
    Operation *op, llvm::DenseMap<Value, Value> &valueMap, OpBuilder &builder,
    Location loc, ArrayRef<int64_t> tensorShape) const {

  // ---- helpers ----
  auto getTensorType = [&](Type scalarTy) -> RankedTensorType {
    return RankedTensorType::get(tensorShape, scalarTy);
  };

  auto createInit = [&](Type scalarTy) -> Value {
    return builder.create<tensor::EmptyOp>(loc, tensorShape, scalarTy);
  };

  auto getOperand = [&](Value v) -> Value {
    auto it = valueMap.find(v);
    if (it != valueMap.end())
      return it->second;
    // External constant defined outside the region — promote on first use
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      auto scalarTy = cst.getResult().getType();
      Value init = createInit(scalarTy);
      auto fill = builder.create<linalg::FillOp>(loc, cst.getResult(), init);
      valueMap[v] = fill.getResult(0);
      return fill.getResult(0);
    }
    llvm_unreachable("operand not in valueMap and not a constant");
  };

  auto record = [&](Value oldVal, Value newVal) -> SmallVector<Value> {
    valueMap[oldVal] = newVal;
    return {newVal};
  };

  // ---- binary arith: Op(lhs, rhs) → tensor Op(lhs, rhs) ----
#define PROMOTE_BINARY(OpTy)                                                   \
  if (auto a = dyn_cast<OpTy>(op)) {                                           \
    auto r = builder.create<OpTy>(loc, getOperand(a.getLhs()),                 \
                                  getOperand(a.getRhs()));                     \
    return record(a.getResult(), r.getResult());                               \
  }

  PROMOTE_BINARY(arith::AddIOp)
  PROMOTE_BINARY(arith::SubIOp)
  PROMOTE_BINARY(arith::MulIOp)
  PROMOTE_BINARY(arith::DivSIOp)
  PROMOTE_BINARY(arith::DivUIOp)
  PROMOTE_BINARY(arith::RemSIOp)
  PROMOTE_BINARY(arith::RemUIOp)
  PROMOTE_BINARY(arith::MaxSIOp)
  PROMOTE_BINARY(arith::MinSIOp)
  PROMOTE_BINARY(arith::MaxUIOp)
  PROMOTE_BINARY(arith::MinUIOp)
  PROMOTE_BINARY(arith::AddFOp)
  PROMOTE_BINARY(arith::SubFOp)
  PROMOTE_BINARY(arith::MulFOp)
  PROMOTE_BINARY(arith::DivFOp)
  PROMOTE_BINARY(arith::RemFOp)
  PROMOTE_BINARY(arith::MaxNumFOp)
  PROMOTE_BINARY(arith::MinNumFOp)
  PROMOTE_BINARY(arith::MaximumFOp)
  PROMOTE_BINARY(arith::MinimumFOp)
  PROMOTE_BINARY(arith::AndIOp)
  PROMOTE_BINARY(arith::OrIOp)
  PROMOTE_BINARY(arith::XOrIOp)
  PROMOTE_BINARY(arith::ShLIOp)
  PROMOTE_BINARY(arith::ShRSIOp)
  PROMOTE_BINARY(arith::ShRUIOp)
#undef PROMOTE_BINARY

  // ---- unary arith: Op(operand) → tensor Op(operand) ----
  if (auto a = dyn_cast<arith::NegFOp>(op)) {
    auto r = builder.create<arith::NegFOp>(loc, getOperand(a.getOperand()));
    return record(a.getResult(), r.getResult());
  }

  // ---- comparisons: preserve predicate ----
  if (auto a = dyn_cast<arith::CmpIOp>(op)) {
    auto r = builder.create<arith::CmpIOp>(
        loc, a.getPredicate(), getOperand(a.getLhs()), getOperand(a.getRhs()));
    return record(a.getResult(), r.getResult());
  }
  if (auto a = dyn_cast<arith::CmpFOp>(op)) {
    auto r = builder.create<arith::CmpFOp>(
        loc, a.getPredicate(), getOperand(a.getLhs()), getOperand(a.getRhs()));
    return record(a.getResult(), r.getResult());
  }

  // ---- select ----
  if (auto a = dyn_cast<arith::SelectOp>(op)) {
    auto r = builder.create<arith::SelectOp>(loc, getOperand(a.getCondition()),
                                             getOperand(a.getTrueValue()),
                                             getOperand(a.getFalseValue()));
    return record(a.getResult(), r.getResult());
  }

  // ---- type casts: Op(in) → Op(in) with tensor result type ----
#define PROMOTE_CAST(OpTy)                                                     \
  if (auto a = dyn_cast<OpTy>(op)) {                                           \
    auto r = builder.create<OpTy>(loc, getTensorType(a.getOut().getType()),    \
                                  getOperand(a.getIn()));                      \
    return record(a.getResult(), r.getResult());                               \
  }

  PROMOTE_CAST(arith::SIToFPOp)
  PROMOTE_CAST(arith::UIToFPOp)
  PROMOTE_CAST(arith::FPToSIOp)
  PROMOTE_CAST(arith::FPToUIOp)
  PROMOTE_CAST(arith::ExtSIOp)
  PROMOTE_CAST(arith::ExtUIOp)
  PROMOTE_CAST(arith::ExtFOp)
  PROMOTE_CAST(arith::TruncIOp)
  PROMOTE_CAST(arith::TruncFOp)
#undef PROMOTE_CAST

  // ---- constant: splat via linalg.fill ----
  if (auto a = dyn_cast<arith::ConstantOp>(op)) {
    auto scalarTy = a.getResult().getType();
    Value init = createInit(scalarTy);
    auto fill = builder.create<linalg::FillOp>(loc, a.getResult(), init);
    return record(a.getResult(), fill.getResult(0));
  }

  // ---- math ops: directly on tensors (same as PROMOTE_BINARY) ----
#define PROMOTE_MATH(OpTy)                                                     \
  if (auto a = dyn_cast<OpTy>(op)) {                                           \
    auto r = builder.create<OpTy>(loc, getOperand(a.getOperand()));            \
    return record(a.getResult(), r.getResult());                               \
  }

  PROMOTE_MATH(math::ExpOp)
  PROMOTE_MATH(math::Exp2Op)
  PROMOTE_MATH(math::SqrtOp)
  PROMOTE_MATH(math::RsqrtOp)
  PROMOTE_MATH(math::LogOp)
  PROMOTE_MATH(math::Log2Op)
  PROMOTE_MATH(math::SinOp)
  PROMOTE_MATH(math::CosOp)
  PROMOTE_MATH(math::ErfOp)
  PROMOTE_MATH(math::CeilOp)
  PROMOTE_MATH(math::FloorOp)
  PROMOTE_MATH(math::AbsFOp)
  PROMOTE_MATH(math::AbsIOp)
#undef PROMOTE_MATH

  // ---- scf.if → arith.select ----
  if (auto a = dyn_cast<scf::IfOp>(op)) {
    Value cond = getOperand(a.getCondition());
    // Process then/else regions with inherited valueMap
    SmallVector<Value> thenResults = promoteRegionBody(
        a.getThenRegion(), valueMap, builder, loc, tensorShape);
    SmallVector<Value> elseResults = promoteRegionBody(
        a.getElseRegion(), valueMap, builder, loc, tensorShape);
    for (unsigned i = 0; i < thenResults.size(); i++) {
      auto sel = builder.create<arith::SelectOp>(loc, cond, thenResults[i],
                                                 elseResults[i]);
      valueMap[a.getResult(i)] = sel.getResult();
    }
    SmallVector<Value> results;
    for (unsigned i = 0; i < thenResults.size(); i++)
      results.push_back(valueMap[a.getResult(i)]);
    return results;
  }

  // ---- scf.for ----
  if (auto a = dyn_cast<scf::ForOp>(op)) {
    SmallVector<Value> tensorInits;
    for (Value v : a.getInitArgs())
      tensorInits.push_back(getOperand(v));
    auto newFor = builder.create<scf::ForOp>(
        loc, a.getLowerBound(), a.getUpperBound(), a.getStep(), tensorInits,
        [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs) {
          llvm::DenseMap<Value, Value> loopMap = valueMap;
          loopMap[a.getInductionVar()] = iv;
          for (auto [oldArg, newArg] :
               llvm::zip(a.getRegionIterArgs(), iterArgs))
            loopMap[oldArg] = newArg;
          SmallVector<Value> yielded =
              promoteRegionBody(a.getRegion(), loopMap, b, l, tensorShape);
          b.create<scf::YieldOp>(l, yielded);
        });
    for (auto [oldR, newR] : llvm::zip(a.getResults(), newFor.getResults()))
      valueMap[oldR] = newR;
    SmallVector<Value> results;
    for (auto r : newFor.getResults())
      results.push_back(r);
    return results;
  }

  if (isa<scf::WhileOp>(op)) {
    op->emitError(
        "scf.while is not supported inside map_elementwise "
        "(scf.condition requires scalar i1, cannot promote to tensor)");
    llvm_unreachable("");
  }
  op->emitError("MapElementwiseDecomposeConverter: unsupported op");
  llvm_unreachable("unhandled op in map_elementwise region");
  llvm_unreachable("unhandled op in map_elementwise region");
}

SmallVector<Value> MapElementwiseDecomposeConverter::promoteRegionBody(
    Region &region, llvm::DenseMap<Value, Value> &valueMap, OpBuilder &builder,
    Location loc, ArrayRef<int64_t> tensorShape) const {
  llvm::DenseMap<Value, Value> localMap = valueMap;
  Block &block = region.front();
  for (Operation &op : block.without_terminator())
    promoteOp(&op, localMap, builder, loc, tensorShape);

  // Helper to get-or-create a tensor version of any value (external
  // constants are promoted on first use).
  auto getTensorValue = [&](Value v) -> Value {
    auto it = localMap.find(v);
    if (it != localMap.end())
      return it->second;
    // External constant — promote to tensor
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      auto scalarTy = cst.getResult().getType();
      Value init = builder.create<tensor::EmptyOp>(loc, tensorShape, scalarTy);
      auto fill = builder.create<linalg::FillOp>(loc, cst.getResult(), init);
      localMap[v] = fill.getResult(0);
      return fill.getResult(0);
    }
    // External but not a constant — use as-is (e.g., SSA values from
    // enclosing scope that are already tensors or valid in this context).
    return v;
  };

  Operation *term = block.getTerminator();
  SmallVector<Value> results;
  for (Value v : term->getOperands())
    results.push_back(getTensorValue(v));
  return results;
}

LogicalResult MapElementwiseDecomposeConverter::matchAndRewrite(
    triton::MapElementwiseOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();

  auto tensorType = cast<RankedTensorType>(op.getOperand(0).getType());
  auto shape = tensorType.getShape();
  int64_t rank = shape.size();
  int64_t pack = op.getPack();
  int64_t numInputs = op.getNumOperands();
  int64_t numOutputs = op.getNumResults();

  assert(rank > 0 && "map_elementwise requires ranked tensor inputs");

  Region &region = op.getRegion();
  assert(region.hasOneBlock() && "expected single-block region");
  Block &body = region.getBlocks().front();
  unsigned numRegionArgs = body.getNumArguments();

  // Build initial ValueMap: block_arg[i] → input_tensor[i/pack]
  llvm::DenseMap<Value, Value> valueMap;
  for (unsigned i = 0; i < numRegionArgs; i++) {
    unsigned inputIdx = i / pack;
    valueMap[body.getArgument(i)] = adaptor.getOperands()[inputIdx];
  }

  // Promote all non-terminator ops
  for (Operation &innerOp : body.without_terminator())
    promoteOp(&innerOp, valueMap, rewriter, loc, shape);

  // Terminator → results
  Operation *terminator = body.getTerminator();
  SmallVector<Value> results;
  for (unsigned i = 0; i < numOutputs; i++)
    results.push_back(valueMap[terminator->getOperand(i * pack)]);

  rewriter.replaceOp(op, results);
  return success();
}

} // namespace TTOpConverters
