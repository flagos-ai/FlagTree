//===--------------------- MKToTx81.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to convert operations from mk dialect to
// tx81 dialect. It converts memory operations to RdmaOp/WdmaOp and converts
// mk.dot to tx.gemm etc.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/MKToTx81/MKToTx81.h"
#include "Tx81/tx81.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "utils/FusionHelper.h"
#include "utils/utils.h"
#include "llvm/ADT/TypeSwitch.h"

// FIXME: triton/Conversion/TritonGPUToLLVM/Utility.h which defined
// TritonLLVMOpBuilder and other utilities has defined DEBUG_TYPE.
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "mk-to-tx81"

using namespace mlir;
using namespace tx;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

class MKToTx81TypeConverter : public TypeConverter {
public:
  MKToTx81TypeConverter() {
    // Add conversions for MemRef types to UI64 (representing SPM addresses)
    addConversion([](MemRefType type) -> Type {
      return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
    });

    // Add conversions for Tensor types to UI64 (representing SPM addresses)
    addConversion([](TensorType type) -> Type {
      return IntegerType::get(type.getContext(), 64, IntegerType::Unsigned);
    });

    // Keep other types as is
    addConversion([](Type type) -> Type { return type; });
  }

private:
  MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

LogicalResult convertLinalgOpToLoops(linalg::LinalgOp op,
                                     ConversionPatternRewriter &rewriter) {
  if (failed(linalg::linalgOpToLoops(rewriter, op)))
    return rewriter.notifyMatchFailure(op, "operation not supported yet.");
  rewriter.eraseOp(op);
  return success();
}

// Get format code for tensor element type
// This maps MLIR types to Tx81 format codes
Data_Format getFormatCode(MemRefType type) {
  auto elemType = type.getElementType();
  if (elemType.isF32()) {
    return Fmt_FP32;
  } else if (elemType.isF16()) {
    return Fmt_FP16;
  } else if (elemType.isBF16()) {
    return Fmt_BF16;
  } else if (elemType.isInteger(1)) {
    return Fmt_BOOL;
  } else if (elemType.isInteger(8)) {
    return Fmt_INT8;
  } else if (elemType.isInteger(16)) {
    return Fmt_INT16;
  } else if (elemType.isInteger(32)) {
    return Fmt_INT32;
  } else if (elemType.isInteger(64)) {
    return Fmt_INT64;
  } else {
    llvm_unreachable("Tx8 unsupported the element type\n");
  }
  // Default to F32 format
  return Fmt_FP32;
}

bool isSupportedType(MemRefType type) {
  auto elemType = type.getElementType();
  return elemType.isF32() || elemType.isF16() || elemType.isBF16() ||
         elemType.isInteger(8);
}

static uint64_t getElemByte(Type type) {
  static DataLayout dataLayout;
  auto typeSize = dataLayout.getTypeSize(type);
  if (!typeSize.isFixed()) {
    llvm::llvm_unreachable_internal("All element type should have fixed size.");
  }
  return typeSize.getFixedValue();
}

static std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
createMetadata(ConversionPatternRewriter &rewriter, Location loc,
               Value operand) {
  auto stridedMetadata =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
  Value indexBasePtr = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
      loc, rewriter.getIndexType(), stridedMetadata.getBaseBuffer());
  auto elemType = dyn_cast<MemRefType>(operand.getType()).getElementType();
  Value elemByte =
      rewriter.create<arith::ConstantIndexOp>(loc, getElemByte(elemType));
  Value offset = stridedMetadata.getOffset();
  Value byteOffset =
      rewriter.create<arith::MulIOp>(loc, offset.getType(), offset, elemByte);

  if (elemType.isInteger(1)) {
    auto [stride, offset] =
        cast<MemRefType>(operand.getType()).getStridesAndOffset();
    // Expected 8 bit alignment
    assert(offset % 8 == 0);

    byteOffset = offset != 0
                     ? rewriter.create<arith::ShRUIOp>(
                           loc, byteOffset.getType(), byteOffset,
                           rewriter.create<arith::ConstantIndexOp>(loc, 3))
                     : byteOffset;
  }

  Value offsetPtr = rewriter.create<arith::AddIOp>(loc, indexBasePtr.getType(),
                                                   indexBasePtr, byteOffset);
  Value i64SPMPtr = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI64Type(), offsetPtr);

  // FIXME: For multi-dimensional(rank > 2), strides need to be multiplied.
  return {i64SPMPtr, stridedMetadata.getSizes(), stridedMetadata.getStrides()};
}

static Value createAddressFromMemref(ConversionPatternRewriter &rewriter,
                                     Location loc, Value operand) {
  auto [i64SPMPtr, sizes, strides] = createMetadata(rewriter, loc, operand);
  return i64SPMPtr;
}

static SmallVector<Value, 4> padSizesToNHWC(ConversionPatternRewriter &rewriter,
                                            Location loc, ValueRange sizes) {
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  int numPad = 4 - sizes.size();
  SmallVector<Value, 4> nhwcShape;
  while (numPad--) {
    nhwcShape.push_back(one);
  }
  for (auto dim : sizes) {
    nhwcShape.push_back(dim);
  }
  return nhwcShape;
}

// The last stride is always 1, skip it, nhwcStrides.size() will be 3.
static SmallVector<Value, 4>
padStridesToNHWC(ConversionPatternRewriter &rewriter, Location loc,
                 ValueRange strides) {
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  int numPad = 4 - strides.size();
  SmallVector<Value, 4> nhwcStrides;
  while (numPad--) {
    nhwcStrides.push_back(one);
  }
  for (auto dim : strides) {
    nhwcStrides.push_back(dim);
  }
  return nhwcStrides;
}

static Value calculateElemCount(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange sizes) {
  // If we get scalar data, sizes is empty, return 1
  if (sizes.empty()) {
    return rewriter.create<arith::ConstantIndexOp>(loc, 1);
  }

  Value elemCount = sizes[0];
  for (int i = 1; i < sizes.size(); i++) {
    elemCount = rewriter.create<arith::MulIOp>(loc, elemCount.getType(),
                                               elemCount, sizes[i]);
  }
  return elemCount;
}

// Extract the operations from a linalg op region
template <typename T> llvm::SmallVector<Operation *> getRegionOps(T linalgOp) {
  auto regionBlock = linalgOp.getBody();
  return llvm::map_to_vector(regionBlock->without_terminator(),
                             [](Operation &op) { return &op; });
}

static Data_Format getFormatFromElemType(mlir::Type elemType) {
  // Convert the integer type to float type by just convert fmt.
  // So here elemType can be integer type.
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 8:
    return Fmt_INT8;
  case 16:
    return elemType.isBF16() ? Fmt_BF16 : Fmt_FP16;
  case 32:
    return elemType.isTF32() ? Fmt_TF32 : Fmt_FP32;
  default:
    llvm_unreachable("Unsupported bit width\n");
  }
  return Fmt_FP32;
}

static Data_Format getFormatFromValueType(MemRefType valueType) {
  // Convert the integer type to float type by just convert fmt.
  // So here elemType can be integer type.
  auto elemType = valueType.getElementType();
  return getFormatFromElemType(elemType);
}

// Convert integer type to float type for CGRA instruction
// Return the convert float type format code
// TODO: Directly convert memref type?
Data_Format insertConvertTypeOp(Value valuePtr, MemRefType valueType,
                                Value elemCount,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {

  // TODO: Other integer type. May need realloc the memory
  auto elemType = valueType.getElementType();

  if (!isa<IntegerType>(elemType))
    return getFormatCode(valueType);

  Data_Format fmt = Fmt_FP32;
  // Get the bit width from the element type
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 16: { // 16 bit integer
    rewriter.create<tx::INT16ToFP16Op>(loc, rewriter.getI64Type(), valuePtr,
                                       valuePtr, elemCount);
    fmt = Fmt_FP16;
    break;
  }
  case 32: { // 32 bit integer
    rewriter.create<tx::INT32ToFP32Op>(loc, rewriter.getI64Type(), valuePtr,
                                       valuePtr, elemCount,
                                       rewriter.getI16IntegerAttr(0));
    break;
  }
  default: {
    llvm_unreachable("Unsupported integer type\n");
  }
  }
  return fmt;
}

// Restore float type to integer type to for CGRA instruction
Value insertRestoreTypeOp(Value valuePtr, MemRefType valueType, Value elemCount,
                          ConversionPatternRewriter &rewriter, Location loc,
                          int16_t roundMode = RND_MODE::RND_NEAREST_EVEN) {
  // TODO: Other integer type. May need realloc the memory
  auto elemType = valueType.getElementType();
  auto newValue = valuePtr;
  if (!isa<IntegerType>(elemType))
    return newValue;

  // Get the bit width from the element type
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  switch (bitWidth) {
  case 16: { // 16 bit integer
    newValue = rewriter.create<tx::FP16ToINT16Op>(
        loc, rewriter.getI64Type(), valuePtr, valuePtr, elemCount,
        rewriter.getI16IntegerAttr(roundMode));
    break;
  }
  case 32: { // 32 bit integer
    newValue = rewriter.create<tx::FP32ToINT32Op>(
        loc, rewriter.getI64Type(), valuePtr, valuePtr, elemCount,
        rewriter.getI16IntegerAttr(roundMode));
    break;
  }
  default: {
    llvm_unreachable("Unsupported integer type\n");
  }
  }
  return newValue;
}

SmallVector<int64_t, 4> reshapeReduceShapeTo4d(ArrayRef<int64_t> inputShape,
                                               int64_t dim) {

  auto rank = inputShape.size();
  SmallVector<int64_t, 4> newShape;
  int64_t leftDimsElement = 1;
  int64_t rightDimsElement = 1;

  for (int i = 0; i < dim; i++)
    leftDimsElement *= inputShape[i];

  if (dim == inputShape.size() - 1)
    return {1, 1, leftDimsElement, inputShape[dim]};

  for (int i = dim + 1; i < rank; i++)
    rightDimsElement *= inputShape[i];

  newShape = {1, leftDimsElement, inputShape[dim], rightDimsElement}; // NHWC
  return newShape;
}

uint64_t next_power_of_two_64(uint64_t x) {
  if (x == 0) {
    return 1;
  }
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

LLVM::AtomicOrdering getOrdering(MemSemantic sem) {
  switch (sem) {
  case MemSemantic::RELAXED:
    return LLVM::AtomicOrdering::monotonic;
  case MemSemantic::ACQUIRE:
    return LLVM::AtomicOrdering::acquire;
  case MemSemantic::RELEASE:
    return LLVM::AtomicOrdering::release;
  case MemSemantic::ACQUIRE_RELEASE:
    return LLVM::AtomicOrdering::acq_rel;
  default:
    llvm_unreachable("Unexpected atomic mem semantic");
  }
}

class MemoryCopyConvertPattern : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->hasAttr("srcSpm") && op->hasAttr("dstSpm") &&
           "Can't get memory space attribute\n");
    bool isSrcSPM = op->getAttrOfType<BoolAttr>("srcSpm").getValue();
    bool isDstSPM = op->getAttrOfType<BoolAttr>("dstSpm").getValue();
    // DDR to DDR
    if (!isSrcSPM && !isDstSPM)
      return rewriter.notifyMatchFailure(
          op, "Can not copy memory from DDR to DDR.\n");

    // SPM to SPM
    if (isSrcSPM && isDstSPM) {
      SmallVector<int64_t> perm(
          cast<MemRefType>(op.getSource().getType()).getRank());
      std::iota(perm.begin(), perm.end(), 0);
      rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op, op.getSource(),
                                                       op.getTarget(), perm);
      return success();
    }

    Location loc = op.getLoc();
    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, loc, adaptor.getSource());
    auto [dstPtr, dstSizes, dstStrides] =
        createMetadata(rewriter, loc, adaptor.getTarget());
    auto srcMemrefType = cast<MemRefType>(op.getSource().getType());
    auto dstMemrefType = cast<MemRefType>(op.getTarget().getType());
    int64_t rank = srcMemrefType.getRank();
    auto elemType = srcMemrefType.getElementType();

    if (srcMemrefType.areTrailingDimsContiguous(rank) &&
        dstMemrefType.areTrailingDimsContiguous(rank)) {
      Value elemCount = calculateElemCount(rewriter, loc, dstSizes);
      if (elemType.getIntOrFloatBitWidth() == 64) {
        elemType = rewriter.getF32Type();
        elemCount = rewriter.create<arith::AddIOp>(loc, elemCount, elemCount);
      }
      elemCount = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), elemCount);

      Data_Format fmt = getFormatFromElemType(elemType);

      if (isDstSPM)
        rewriter.replaceOpWithNewOp<tx::Rdma1dOp>(op, dstPtr, srcPtr, elemCount,
                                                  fmt);
      else
        rewriter.replaceOpWithNewOp<tx::Wdma1dOp>(op, dstPtr, srcPtr, elemCount,
                                                  fmt);
      return success();
    }

    auto srcFmt = getFormatCode(cast<MemRefType>(srcMemrefType.clone(
        rewriter.getIntegerType(srcMemrefType.getElementTypeBitWidth()))));

    // Update rank to 4 if rank less than 4.
    if (rank < 4) {
      srcSizes = padSizesToNHWC(rewriter, op->getLoc(), srcSizes);
      srcStrides = padStridesToNHWC(rewriter, op->getLoc(), srcStrides);
      dstSizes = padSizesToNHWC(rewriter, op->getLoc(), dstSizes);
      dstStrides = padStridesToNHWC(rewriter, op->getLoc(), dstStrides);
      rank = 4;
    }
    int elemBytes = srcMemrefType.getElementTypeBitWidth() >> 3;
    if (isDstSPM) {
      auto rdmaOp = rewriter.create<tx::RdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          srcSizes,                              // src shape
          srcStrides,                            // src stride
          dstSizes,                              // dst shape
          dstStrides,                            // dst stride
          rewriter.getI32IntegerAttr(rank),      // rank
          rewriter.getI32IntegerAttr(elemBytes), // elem bytes
          rewriter.getI32IntegerAttr(srcFmt)     // Format
      );
    } else {
      auto wdmaOp = rewriter.create<tx::WdmaOp>(
          op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr,
          srcSizes,                              // src shape
          srcStrides,                            // src stride
          dstSizes,                              // dst shape
          dstStrides,                            // dst stride
          rewriter.getI32IntegerAttr(rank),      // rank
          rewriter.getI32IntegerAttr(elemBytes), // elem bytes
          rewriter.getI32IntegerAttr(srcFmt)     // Format
      );
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert linalg.fill to MemsetOp
class LinalgFillOpConversion : public OpConversionPattern<linalg::FillOp> {
public:
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  void preprocessI1Fill(ConversionPatternRewriter &rewriter, linalg::FillOp op,
                        SmallVector<Value> &srcSizes,
                        SmallVector<Value> &srcStrides, int64_t &rank,
                        Type &inputType) const {

    auto elementCount = calculateElemCount(rewriter, op->getLoc(), srcSizes);
    SmallVector<Value> memoryLinearizedSizes{rewriter.create<arith::ShRUIOp>(
        op.getLoc(), elementCount.getType(), elementCount,
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 4))};

    SmallVector<Value> memoryLinearizedStrides{
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1)};

    srcSizes = memoryLinearizedSizes;
    srcStrides = memoryLinearizedStrides;
    rank = 1;
    inputType = rewriter.getF16Type();
  }

  bool isMemoryContiguousType(MemRefType type) const {
    auto shape = type.getShape();
    auto rank = type.getRank();
    auto firstNonOne = std::find_if_not(shape.begin(), shape.end(),
                                        [](int64_t dim) { return dim == 1; });
    int leadingOnes = std::distance(shape.begin(), firstNonOne);
    return type.areTrailingDimsContiguous(rank - leadingOnes);
  }

  bool isSupportedBitWidthAndType(int bitWidth, MemRefType type) const {
    assert(isMemoryContiguousType(type) && "Type's memory must be contiguous");
    return bitWidth == 16 || bitWidth == 32 ||
           (bitWidth == 1 && type.getNumElements() >= 16);
  }

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the value to fill with
    Value fillValue = op.getInputs()[0]; // adaptor.getValue();

    if (op.getOutputs().size() != 1)
      return rewriter.notifyMatchFailure(op, "Only support single output\n");

    auto rank = cast<MemRefType>(op.getOutputs()[0].getType()).getRank();
    if (rank == 0) {
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getInputs()[0],
                                       adaptor.getOutputs()[0]);
      rewriter.eraseOp(op);
      return success();
    }

    auto [srcPtr, srcSizes, srcStrides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto inputType = op.getInputs()[0].getType();
    auto outputType = cast<MemRefType>(op.getOutputs()[0].getType());
    auto bitWidth = inputType.getIntOrFloatBitWidth();

    if (!isSupportedBitWidthAndType(bitWidth, outputType)) {
      return convertLinalgOpToLoops(op, rewriter);
    }

    fillValue = rewriter.create<arith::BitcastOp>(
        op.getLoc(), rewriter.getIntegerType(bitWidth), fillValue);
    fillValue = bitWidth != 32
                    ? rewriter.create<arith::ExtSIOp>(
                          op.getLoc(), rewriter.getI32Type(), fillValue)
                    : fillValue;

    if (bitWidth == 1) {
      preprocessI1Fill(rewriter, op, srcSizes, srcStrides, rank, inputType);
    }
    Data_Format fmt = getFormatFromElemType(inputType);

    // NOTE: When encounter NaN, use xor + addvs to simulate memset operation
    // will get wrong result.
    auto resultOp = rewriter.create<tx::MemsetOp>(
        op.getLoc(), rewriter.getI64Type(), srcPtr, fillValue, srcSizes,
        srcStrides, rewriter.getI32IntegerAttr(rank),
        rewriter.getI16IntegerAttr(fmt));

    rewriter.eraseOp(op);

    return success();
  }
};

class TransposeOpConversion : public OpConversionPattern<linalg::TransposeOp> {
public:
  using OpConversionPattern<linalg::TransposeOp>::OpConversionPattern;

  bool convertToGatherScatter(linalg::TransposeOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto perms = op.getPermutation();
    auto src = op.getInput();
    auto dst = op.getInit();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    auto srcShape = srcType.getShape();
    auto srcStrides = srcType.getStridesAndOffset().first;
    auto dstShape = dstType.getShape();
    auto dstStrides = dstType.getStridesAndOffset().first;
    unsigned bitWidth = srcType.getElementTypeBitWidth();

    if (!srcType.hasStaticShape() || !dstType.hasStaticShape() ||
        llvm::any_of(srcStrides, ShapedType::isDynamic) ||
        llvm::any_of(dstStrides, ShapedType::isDynamic)) {
      LDBG("TransposeOpConversion: dynamic shape/strides not supported\n");
      return false;
    }

    // Get inner bits
    uint64_t bits = bitWidth;
    int64_t contiguousStride = 1;
    size_t rank = perms.size();
    for (; rank > 0 && srcStrides[perms[rank - 1]] == contiguousStride &&
           dstStrides[rank - 1] == contiguousStride;
         --rank) {
      bits *= dstShape[rank - 1];
      contiguousStride *= dstShape[rank - 1];
    }

    if (bits & 0x7 || rank > 3) {
      LDBG("TransposeOpConversion: bits not byte aligned or rank not "
           "supported\n");
      return false;
    }

    unsigned bytes = bits >> 3;
    SmallVector<uint32_t> srcStrideArgs(3);
    SmallVector<uint32_t> srcIterArgs(3, 1);
    SmallVector<uint32_t> dstStrideArgs(3);
    SmallVector<uint32_t> dstIterArgs(3, 1);

    for (size_t i = 0; i < rank; ++i) {
      uint64_t srcStrideBits = srcStrides[perms[i]] * bitWidth;
      uint64_t dstStrideBits = dstStrides[i] * bitWidth;
      if (srcStrideBits & 0x7 || dstStrideBits & 0x7) {
        LDBG("TransposeOpConversion: stride not byte aligned\n");
        return false;
      }
      srcStrideArgs[i + 3 - rank] = srcStrideBits >> 3;
      dstStrideArgs[i + 3 - rank] = dstStrideBits >> 3;
      srcIterArgs[i + 3 - rank] = srcShape[perms[i]];
      dstIterArgs[i + 3 - rank] = dstShape[i];
    }

    auto srcPtr = createAddressFromMemref(rewriter, op->getLoc(), src);
    auto dstPtr = createAddressFromMemref(rewriter, op->getLoc(), dst);

    rewriter.create<tx::GatherScatter>(
        op.getLoc(), rewriter.getI64Type(), srcPtr, dstPtr, bytes,
        srcStrideArgs[0], srcStrideArgs[1], srcStrideArgs[2], srcIterArgs[0],
        srcIterArgs[1], srcIterArgs[2], dstStrideArgs[0], dstStrideArgs[1],
        dstStrideArgs[2], dstIterArgs[0], dstIterArgs[1], dstIterArgs[2]);

    rewriter.eraseOp(op);
    return true;
  }

  template <typename transTy>
  LogicalResult transposeChannel(linalg::TransposeOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    auto src = op.getInput();
    auto dst = op.getInit();
    auto srcType = cast<MemRefType>(src.getType());
    auto dstType = cast<MemRefType>(dst.getType());
    SmallVector<int32_t, 4> srcShape(srcType.getShape().begin(),
                                     srcType.getShape().end());
    SmallVector<int32_t, 4> dstShape(dstType.getShape().begin(),
                                     dstType.getShape().end());

    auto srcPtr = createAddressFromMemref(rewriter, op->getLoc(), src);
    auto dstPtr = createAddressFromMemref(rewriter, op->getLoc(), dst);

    // TODO: Through fmt conversion to support more element types.
    if (!isSupportedType(srcType)) {
      return rewriter.notifyMatchFailure(op, "Unsupported element type\n");
    }
    Data_Format fmt = getFormatCode(srcType);

    auto newOp =
        rewriter.create<transTy>(op->getLoc(), rewriter.getI64Type(), srcPtr,
                                 dstPtr, srcShape, dstShape, fmt);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(linalg::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto perm = op.getPermutation();

    if (perm == ArrayRef<int64_t>({0, 2, 3, 1})) {
      return transposeChannel<tx::Nchw2nhwc>(op, adaptor, rewriter);
    }

    if (perm == ArrayRef<int64_t>({0, 3, 1, 2})) {
      return transposeChannel<tx::Nhwc2nchw>(op, adaptor, rewriter);
    }

    if (convertToGatherScatter(op, adaptor, rewriter))
      return success();

    // Default handling of remaining cases.
    // TODO:  Convert higher rank to tx.
    return convertLinalgOpToLoops(op, rewriter);
  }
};

class ReciprocalOpConversionPattern
    : public OpConversionPattern<linalg::ReciprocalOp> {
public:
  using OpConversionPattern<linalg::ReciprocalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::ReciprocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto [inputPtr, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getInputs()[0]);
    auto outputPtr =
        createAddressFromMemref(rewriter, loc, adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    // Tx neural engine not support fp32 for input
    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    Data_Format srcFmt = getFormatCode(inputType);

    rewriter.create<tx::RecipVVOp>(loc, rewriter.getI64Type(), inputPtr,
                                   outputPtr, elemCount,
                                   rewriter.getI16IntegerAttr(srcFmt));
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// mk.dot to tx.gemm Conversion Pattern
//===----------------------------------------------------------------------===//

class MKDotToTx81GemmOpConversion : public OpConversionPattern<mk::DotOp> {
public:
  using OpConversionPattern<mk::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mk::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto a = adaptor.getA();
    auto b = adaptor.getB();
    auto dst = adaptor.getInits();
    auto aType = cast<MemRefType>(a.getType());     // (K/64)xMx64
    auto bType = cast<MemRefType>(b.getType());     // (N/64)xKx64
    auto dstType = cast<MemRefType>(dst.getType()); // (N/64)xMx64
    int32_t M = aType.getShape()[1];
    int32_t K = bType.getShape()[1];
    int32_t N = bType.getShape()[0] * bType.getShape()[2];
    auto dims = rewriter.getI32ArrayAttr({M, K, N});
    Data_Format srcFmt = getFormatCode(aType);
    Data_Format dstFmt = getFormatCode(dstType);

    auto aPtr = createAddressFromMemref(rewriter, loc, a);
    auto bPtr = createAddressFromMemref(rewriter, loc, b);
    auto dstPtr = createAddressFromMemref(rewriter, loc, dst);

    // Assume input type is same. Tx neural engine not support fp32 for input
    // FIXME: There are encoding differences between f32 and tf32 for certain
    // special values (e.g., Inf and NaN). We assume that such special values do
    // not occur.
    if (aType.getElementType().isF32()) {
      // Warning for neural engine that fp32 is not supported
      LLVM_DEBUG(llvm::dbgs() << "Neural engine not support FP32. Convert FP32 "
                                 "to TF32 for tx.Gemm Op\n");
      srcFmt = Data_Format::Fmt_TF32;
    }

    auto zero =
        rewriter.create<arith::ConstantIntOp>(loc, 0, rewriter.getI64Type());

    // Create GemmOp
    rewriter.create<tx::GemmOp>(
        loc, rewriter.getI64Type(),
        aPtr,               // src_a (Matrix A in SPM)
        bPtr,               // src_b (Matrix B in SPM)
        dstPtr,             // src_bias. Unused for now.
        dstPtr,             // dst,
        dims,               // dimensions [M,K,N]
        op.getEnPsumAttr(), // en_psum. Used as accumulate buffer
        dstPtr, //  The address of psum in SPM, Always same to output
        rewriter.getBoolAttr(false),                   // trans_src_a
        rewriter.getBoolAttr(true),                    // trans_src_b.
        rewriter.getI32IntegerAttr(1),                 // batch_src_a
        rewriter.getI32IntegerAttr(1),                 // batch_src_b
        rewriter.getI32IntegerAttr(ActFuncMode::None), // relu_mode.
        rewriter.getBoolAttr(false),                   // en_bias
        rewriter.getBoolAttr(false),                   // en_neg_scale
        zero,                                          // src_neg_scale
        rewriter.getBoolAttr(false),                   // en_pos_scale
        zero,                                          // src_pos_scale
        rewriter.getI32IntegerAttr(srcFmt),            // src_fmt
        rewriter.getI32IntegerAttr(dstFmt)             // dst_fmt
    );

    rewriter.eraseOp(op);
    return success();
  }
};

class MKDequantOpConversionPattern : public OpConversionPattern<mk::DequantOp> {
  using OpConversionPattern<mk::DequantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mk::DequantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto inputPtr = createAddressFromMemref(rewriter, loc, adaptor.getSrc());
    auto scalePtr = createAddressFromMemref(rewriter, loc, adaptor.getScale());
    auto outputPtr = createAddressFromMemref(rewriter, loc, adaptor.getInit());

    auto outputType = cast<MemRefType>(adaptor.getInit().getType());
    auto elemCount = outputType.getNumElements();
    auto elemType = outputType.getElementType();
    assert(elemType.isF16() || elemType.isBF16());
    if (elemType.isBF16())
      rewriter.create<tx::MXFPScaleBF16Op>(loc, TypeRange{}, inputPtr, scalePtr,
                                           outputPtr, elemCount);
    else
      rewriter.create<tx::MXFPScaleFP16Op>(loc, TypeRange{}, inputPtr, scalePtr,
                                           outputPtr, elemCount);
    rewriter.eraseOp(op);
    return success();
  }
};

class GatherConvertPattern : public OpConversionPattern<mlir::mk::GatherOp> {
public:
  using OpConversionPattern<mlir::mk::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto indices = adaptor.getIndices();
    auto indicesType = cast<MemRefType>(indices.getType());
    auto shape = indicesType.getShape();

    auto axis = op.getAxis();

    int64_t numElems = indicesType.getNumElements();
    auto strides = computeStrides(shape);
    for (int64_t idx = 0; idx < numElems; idx += 1) {
      auto tensorIdx = delinearize(idx, strides);

      SmallVector<Value> idxIndex(tensorIdx.size());
      std::transform(tensorIdx.begin(), tensorIdx.end(), idxIndex.begin(),
                     [&](auto val) {
                       return rewriter.create<arith::ConstantIndexOp>(loc, val);
                     });
      // Read the index value from indices tensor
      Value indexValue =
          rewriter.create<memref::LoadOp>(loc, indices, idxIndex);

      // Read value from source using computed indices
      SmallVector<Value> inputIndex = idxIndex;
      assert(axis < inputIndex.size() && axis >= 0 &&
             "Axis index out of bounds");
      inputIndex[axis] = rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), indexValue);

      Value gatheredValue =
          rewriter.create<memref::LoadOp>(loc, adaptor.getSrc(), inputIndex);

      // Write value to destination
      rewriter.create<memref::StoreOp>(loc, gatheredValue, adaptor.getDst(),
                                       idxIndex);
    }

    rewriter.eraseOp(op);

    return success();
  }
};

class MKSigmoidToTx81SigmoidOpConversion
    : public OpConversionPattern<mlir::mk::SigmoidOp> {
public:
  using OpConversionPattern<mlir::mk::SigmoidOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [input, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getSrc());
    auto [dst, dstSizes, dstStrides] =
        createMetadata(rewriter, loc, adaptor.getZeroes());
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    // Tx neural engine not support fp32 for input
    auto inputType = dyn_cast<MemRefType>(op.getSrc().getType());
    assert(isSupportedType(inputType) &&
           "Unsupported element type for Sigmoid operation\n");
    Data_Format srcFmt = getFormatCode(inputType);

    rewriter.create<tx::Sigmoid>(loc, rewriter.getI64Type(), input, dst,
                                 elemCount, rewriter.getI16IntegerAttr(srcFmt));
    rewriter.eraseOp(op);

    return success();
  }
};

struct MKGeluToTx81GeluOpConversion
    : public OpConversionPattern<mlir::mk::GeluOp> {
  using OpConversionPattern<mlir::mk::GeluOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::mk::GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [input, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getSrc());
    auto [dst, dstSizes, dstStrides] =
        createMetadata(rewriter, loc, adaptor.getZeroes());
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);
    // Check input type: only support f16, bf16, f32.
    auto inputType = dyn_cast<MemRefType>(op.getSrc().getType());
    assert(isSupportedType(inputType) &&
           "Unsupported element type for Gelu operation\n");
    Data_Format srcFmt = getFormatCode(inputType);

    auto geluMode = static_cast<GeluMode>(op.getGeluMode());
    switch (geluMode) {
    case GeluMode::None: {
      rewriter.create<tx::GeluNone>(loc, rewriter.getI64Type(), input, dst,
                                    elemCount,
                                    rewriter.getI16IntegerAttr(srcFmt));
      break;
    }
    case GeluMode::Tanh: {
      auto immAddr = createAddressFromMemref(rewriter, loc, adaptor.getImm());
      rewriter.create<tx::GeluTanh>(loc, rewriter.getI64Type(), input, immAddr,
                                    dst, elemCount,
                                    rewriter.getI16IntegerAttr(srcFmt));
      break;
    }
    default: {
      llvm::report_fatal_error("Unsupported gelu mode!");
    }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class MKBit2FPOpConversionPattern : public OpConversionPattern<mk::Bit2FpOp> {
public:
  using OpConversionPattern<mlir::mk::Bit2FpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::Bit2FpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [inputPtr, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getSrc());
    auto outputPtr = createAddressFromMemref(rewriter, loc, adaptor.getInit());

    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    Data_Format srcFmt = getFormatCode(outputType);

    rewriter.create<tx::Bit2FpOp>(loc, rewriter.getI64Type(), inputPtr,
                                  outputPtr, elemCount,
                                  rewriter.getI16IntegerAttr(srcFmt));
    rewriter.eraseOp(op);

    return success();
  }
};

class MKMaskMoveOpConversionPattern
    : public OpConversionPattern<mk::MaskMoveOp> {
public:
  using OpConversionPattern<mlir::mk::MaskMoveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::mk::MaskMoveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [inputPtr, sizes, strides] =
        createMetadata(rewriter, loc, adaptor.getSource());
    auto outputPtr = createAddressFromMemref(rewriter, loc, adaptor.getInit());
    auto maskPtr = createAddressFromMemref(rewriter, loc, adaptor.getMask());
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getSource().getType());
    Data_Format srcFmt = getFormatCode(inputType);

    rewriter.create<tx::MaskMoveOp>(loc, rewriter.getI64Type(), inputPtr,
                                    outputPtr, elemCount, maskPtr,
                                    rewriter.getI32IntegerAttr(srcFmt));
    rewriter.eraseOp(op);

    return success();
  }
};

template <typename MKOpT, typename TxOpT>
struct MKRelationVVOpConversionPattern : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input0 = adaptor.getInput0();
    auto input1 = adaptor.getInput1();
    auto output = adaptor.getInit();
    auto inputType = cast<MemRefType>(input0.getType());

    auto loc = op.getLoc();

    auto [input0Ptr, sizes, strides] = createMetadata(rewriter, loc, input0);
    auto input1Ptr = createAddressFromMemref(rewriter, loc, input1);
    auto outputPtr = createAddressFromMemref(rewriter, loc, output);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto tx81Op = rewriter.create<TxOpT>(
        loc, rewriter.getI64Type(), input0Ptr, input1Ptr, outputPtr, elemCount,
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename MKOpT, typename TxOpT>
struct MKArithVSOpConversionPattern : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto val = adaptor.getValue();
    auto output = adaptor.getInit();
    auto inputType = cast<MemRefType>(input.getType());

    auto loc = op.getLoc();

    // Store float value on a integer type 32bit memory
    auto bitWidth = inputType.getElementTypeBitWidth();
    assert(bitWidth == 16 || bitWidth == 32);
    auto bitcastType =
        bitWidth == 16 ? rewriter.getI16Type() : rewriter.getI32Type();
    Value i32Value =
        rewriter.create<arith::BitcastOp>(op.getLoc(), bitcastType, val);
    // Extend bitcode to 32 bit
    if (bitWidth == 16) {
      i32Value = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI32Type(), i32Value);
    }

    auto elemCount = inputType.getNumElements();
    auto inputPtr = createAddressFromMemref(rewriter, loc, input);
    auto outputPtr = createAddressFromMemref(rewriter, loc, output);
    auto tx81Op = rewriter.create<TxOpT>(
        loc, rewriter.getI64Type(), inputPtr, i32Value, outputPtr,
        rewriter.create<arith::ConstantIndexOp>(loc, elemCount),
        rewriter.getI16IntegerAttr(RND_MODE::RND_NEAREST_EVEN), // Round mode
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename MKOpT, typename TxOpT>
struct MKRelationVSOpConversionPattern : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getInput();
    auto val = adaptor.getValue();
    auto output = adaptor.getInit();
    auto inputType = cast<MemRefType>(input.getType());

    auto loc = op.getLoc();

    // Store float value on a integer type 32bit memory
    auto bitWidth = inputType.getElementTypeBitWidth();
    assert(bitWidth == 16 || bitWidth == 32);
    auto bitcastType =
        bitWidth == 16 ? rewriter.getI16Type() : rewriter.getI32Type();
    Value i32Value =
        rewriter.create<arith::BitcastOp>(op.getLoc(), bitcastType, val);
    // Extend bitcode to 32 bit
    if (bitWidth == 16) {
      i32Value = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI32Type(), i32Value);
    }

    // BoolRelationVSOp need 8 bit alignment
    auto elemCount = inputType.getNumElements();
    auto outputType = cast<MemRefType>(output.getType());
    if (outputType.getElementType().isInteger(1)) {
      elemCount = ((elemCount + 7) / 8) * 8;
      op->emitRemark() << "element count was expanded to a multiple of 8, may "
                          "access memory out of bounds!";
    }

    auto inputPtr = createAddressFromMemref(rewriter, loc, input);
    auto outputPtr = createAddressFromMemref(rewriter, loc, output);
    auto tx81Op = rewriter.create<TxOpT>(
        loc, rewriter.getI64Type(), inputPtr, i32Value, outputPtr,
        rewriter.create<arith::ConstantIndexOp>(loc, elemCount),
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename MKOpT, typename TxOpT>
struct MKArgMinMaxConversionPattern : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = adaptor.getSrc();
    auto outVal = adaptor.getValue();
    auto outIdx = adaptor.getIndex();
    auto inputType = cast<MemRefType>(input.getType());
    auto valueType = cast<MemRefType>(outVal.getType());
    auto indexType = cast<MemRefType>(outIdx.getType());
    auto inputShape = inputType.getShape();

    assert(!inputType.getElementType().isInteger() &&
           "mk.argmax/argmin op's input type should not be integer.");

    auto loc = op.getLoc();
    auto reduceDim = op.getAxis();
    int64_t innerSize = inputShape.empty() ? 1 : inputShape.back();
    // NOTE: LinalgToMK Pass already sliced input to a vector
    assert(inputType.getRank() == 1);
    // TODO: support input's stride != 1
    auto [strides, offset] = inputType.getStridesAndOffset();
    assert(strides[0] == 1);

    auto inputPtr = createAddressFromMemref(rewriter, loc, input);
    auto outValPtr = createAddressFromMemref(rewriter, loc, outVal);
    auto outIdxPtr = createAddressFromMemref(rewriter, loc, outIdx);

    auto tx81Op = rewriter.create<TxOpT>(
        loc, TypeRange{}, inputPtr, outValPtr, outIdxPtr,
        rewriter.getI32IntegerAttr(innerSize),
        rewriter.getI16IntegerAttr(getFormatCode(valueType)));
    rewriter.eraseOp(op);
    return success();
  }
};

struct ElementwiseConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult convertIsNaNOp(linalg::GenericOp op, OpAdaptor adapter,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adapter.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adapter.getOutputs()[0]);
    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto elemCount = inputType.getNumElements();
    assert((elemCount % 8) == 0 &&
           "ElemCount must be a multiple of 8 due to ElementwiseRewrite pass!");

    auto elemCountValue = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto fmt = getFormatCode(inputType);
    rewriter.create<tx::BoolUnEqualVV>(loc,                   // loc
                                       rewriter.getI64Type(), // result type
                                       input,                 // input0
                                       input,                 // input1
                                       output,                // out
                                       elemCountValue,        // elem_count
                                       rewriter.getI16IntegerAttr(fmt) // fmt
    );
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult convertUnaryOp(linalg::GenericOp op, OpAdaptor adapter,
                               ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adapter.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adapter.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    // Data format after conversion
    Data_Format srcFmt = getFormatCode(inputType);

    // Create the unary operation
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input, output, elemCount,
                           rewriter.getI16IntegerAttr(srcFmt));

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult convertBinaryOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    // Data format after conversion
    Data_Format srcFmt = getFormatCode(inputType);

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCount,
                           rewriter.getI16IntegerAttr(0), // Round mode
                           rewriter.getI16IntegerAttr(srcFmt));

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult
  convertBoolBinaryLogicOp(linalg::GenericOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto bitWidth = inputType.getElementType().getIntOrFloatBitWidth();
    auto elemCount = inputType.getNumElements();

    // If bit width is 1 and element count is not divisible by 8, expand
    // the number of elements to a multiple of 8.
    if (bitWidth == 1 && elemCount % 8) {
      elemCount = ((elemCount + 7) / 8) * 8;
    }

    elemCount *= bitWidth;

    // Creat new element count value.
    Value elemCountValue =
        rewriter.create<arith::ConstantIndexOp>(loc, elemCount);
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCountValue);

    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult ZeroPointConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto output = createAddressFromMemref(rewriter, op->getLoc(),
                                          adaptor.getOutputs()[0]);
    auto elemCount =
        cast<MemRefType>(op->getOperandTypes()[0]).getNumElements();

    rewriter.create<TxOpT>(loc, input, output, 0, (uint32_t)elemCount);
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult NormalConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input, output,
                           elemCount);
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult
  RoundConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 RND_MODE roundMode = RND_MODE::RND_NEAREST_EVEN) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);
    // TODO: Fix attribute
    auto result = rewriter.create<TxOpT>(
        loc,
        rewriter.getI64Type(),                // Result type
        input,                                // Input
        output,                               // Output
        elemCount,                            // Element count
        rewriter.getI16IntegerAttr(roundMode) // Round mode
    );
    rewriter.eraseOp(op);
    return success();
  }

  template <typename TxOpT>
  LogicalResult BoolRelationVVOp(linalg::GenericOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    assert(inputType.getNumElements() % 8 == 0);
    Data_Format srcFmt = getFormatCode(inputType);

    // Create the elementwise operation
    // TODO: Fix attribute
    rewriter.create<TxOpT>(loc, rewriter.getI64Type(), input0, input1, output,
                           elemCount,
                           rewriter.getI16IntegerAttr(srcFmt) // Format
    );

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult FmaConvertOp(linalg::GenericOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto input2 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[2]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());

    auto mulResult = rewriter.create<tx::MulVVOp>(
        loc, rewriter.getI64Type(), input0, input1, output, elemCount,
        rewriter.getI16IntegerAttr(0), // Round mode
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));
    auto addResult = rewriter.create<tx::AddVVOp>(
        loc, rewriter.getI64Type(), output, input2, output, elemCount,
        rewriter.getI16IntegerAttr(0), // Round mode
        rewriter.getI16IntegerAttr(getFormatCode(inputType)));
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult convertDivIntOp(linalg::GenericOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    auto input0 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto input1 =
        createAddressFromMemref(rewriter, loc, adaptor.getInputs()[1]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    // Data format after conversion
    Data_Format srcFmt =
        insertConvertTypeOp(input0, inputType, elemCount, rewriter, loc);
    if (adaptor.getInputs()[0] != adaptor.getInputs()[1]) {
      // If input0 and input1 are not the same, we need to convert input1 type
      insertConvertTypeOp(input1, inputType, elemCount, rewriter, loc);
    }

    if (adaptor.getInputs()[0] != adaptor.getOutputs()[0] &&
        adaptor.getInputs()[1] != adaptor.getOutputs()[0])
      // If input and output are not the same, we need to convert output type
      insertConvertTypeOp(output, inputType, elemCount, rewriter, loc);

    rewriter.create<tx::DivVVOp>(loc, rewriter.getI64Type(), input0, input1,
                                 output, elemCount,
                                 rewriter.getI16IntegerAttr(0), // Round mode
                                 rewriter.getI16IntegerAttr(srcFmt));

    insertRestoreTypeOp(output, inputType, elemCount, rewriter, loc,
                        RND_MODE::RND_ZERO);

    if (adaptor.getInputs()[0] != adaptor.getOutputs()[0]) {
      insertRestoreTypeOp(input0, inputType, elemCount, rewriter, loc);
    }
    if (adaptor.getInputs()[1] != adaptor.getOutputs()[0] &&
        adaptor.getInputs()[1] != adaptor.getInputs()[0]) {
      insertRestoreTypeOp(input1, inputType, elemCount, rewriter, loc);
    }

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  convertRoundOp(linalg::GenericOp op, OpAdaptor adaptor,
                 ConversionPatternRewriter &rewriter,
                 RND_MODE roundMode = RND_MODE::RND_NEAREST_EVEN) const {
    Location loc = op->getLoc();
    auto input = createAddressFromMemref(rewriter, loc, adaptor.getInputs()[0]);
    auto [output, sizes, strides] =
        createMetadata(rewriter, op->getLoc(), adaptor.getOutputs()[0]);
    auto elemCount = calculateElemCount(rewriter, op->getLoc(), sizes);

    // Use IEEE round to nearest mode
    auto fpToInt = rewriter.create<tx::FP32ToINT32Op>(
        loc, rewriter.getI64Type(), input, output, elemCount,
        rewriter.getI16IntegerAttr(roundMode)); // Round mode
    auto intToFp = rewriter.create<tx::INT32ToFP32Op>(
        loc, rewriter.getI64Type(), output, output, elemCount,
        rewriter.getI16IntegerAttr(0)); // Round mode

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto regionOps = getRegionOps<linalg::GenericOp>(op);

    if (!op.getOutputs().empty() &&
        cast<MemRefType>(op.getOutputs()[0].getType()).getRank() == 0)
      return convertLinalgOpToLoops(op, rewriter);

    // Check if the operation is elementwise
    if (op.getIteratorTypesArray().front() !=
        mlir::utils::IteratorType::parallel)
      return rewriter.notifyMatchFailure(op, "Only support elementwise op.");

    // WORKAROUND: Select op input0 is bool(i1), cmp op result is bool(i1)
    // I64/F64 lowering to llvm
    // NOTE: May exist scf.if which has not output
    if (regionOps.size() != 1 ||
        (!op.getOutputs().empty() &&
         dyn_cast<MemRefType>(op.getOutputs()[0].getType())
                 .getElementType()
                 .getIntOrFloatBitWidth() == 64) ||
        (!op.getInputs().empty() &&
         dyn_cast<MemRefType>(op.getInputs()[0].getType())
                 .getElementType()
                 .getIntOrFloatBitWidth() == 64)) {
      return convertLinalgOpToLoops(op, rewriter);
    }

    auto elemWiseOp = regionOps[0];
    return llvm::TypeSwitch<Operation *, LogicalResult>(elemWiseOp)
        .Case<math::TanhOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::Tanh>(op, adaptor, rewriter);
        })
        .Case<arith::NegFOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::NegVVOp>(op, adaptor, rewriter);
        })
        .Case<math::IsNaNOp>([&](auto elemWiseOp) {
          return convertIsNaNOp(op, adaptor, rewriter);
        })
        .Case<arith::AddFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::AddVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::SubFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::SubVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::MulFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MulVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::DivSIOp, arith::DivUIOp>([&](auto elemWiseOp) {
          return convertDivIntOp(op, adaptor, rewriter);
        })
        .Case<arith::MaximumFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MaxVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::MinimumFOp>([&](auto elemWiseOp) {
          return convertBinaryOp<tx::MinVVOp>(op, adaptor, rewriter);
        })
        .Case<arith::AndIOp>([&](auto elemWiseOp) {
          return convertBoolBinaryLogicOp<tx::BoolAndV>(op, adaptor, rewriter);
        })
        .Case<arith::OrIOp>([&](auto elemWiseOp) {
          return convertBoolBinaryLogicOp<tx::BoolOrV>(op, adaptor, rewriter);
        })
        .Case<arith::XOrIOp>([&](auto elemWiseOp) {
          return convertBoolBinaryLogicOp<tx::BoolXorV>(op, adaptor, rewriter);
        })
        .Case<math::AbsFOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::AbsVVOp>(op, adaptor, rewriter);
        })
        .Case<math::CeilOp>([&](auto elemWiseOp) {
          return convertRoundOp(op, adaptor, rewriter, RND_MODE::RND_POS_INF);
        })
        .Case<math::FloorOp>([&](auto elemWiseOp) {
          return convertRoundOp(op, adaptor, rewriter, RND_MODE::RND_NEG_INF);
        })
        .Case<math::TruncOp>([&](auto elemWiseOp) {
          return convertRoundOp(op, adaptor, rewriter, RND_MODE::RND_ZERO);
        })
        .Case<math::RoundOp>([&](auto elemWiseOp) {
          return convertRoundOp(op, adaptor, rewriter);
        })
        .Case<math::SqrtOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::SqrtVVOp>(op, adaptor, rewriter);
        })
        .Case<math::RsqrtOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::RsqrtVVOp>(op, adaptor, rewriter);
        })
        .Case<math::LogOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::LnOp>(op, adaptor, rewriter);
        })
        .Case<math::Log2Op>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::Log2Op>(op, adaptor, rewriter);
        })
        .Case<math::ExpOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::ExpOp>(op, adaptor, rewriter);
        })
        .Case<math::Exp2Op>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::Pow2Op>(op, adaptor, rewriter);
        })
        .Case<math::SinOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::SinOp>(op, adaptor, rewriter);
        })
        .Case<math::CosOp>([&](auto elemWiseOp) {
          return convertUnaryOp<tx::CosOp>(op, adaptor, rewriter);
        })
        .Case<arith::ExtFOp>([&](auto elemWiseOp) {
          auto inputType = elemWiseOp.getIn().getType();
          auto targetType = elemWiseOp.getOut().getType();
          if (inputType.isF16() && targetType.isF32())
            return NormalConvertOp<tx::FP16ToFP32Op>(op, adaptor, rewriter);
          else if (inputType.isBF16() && targetType.isF32())
            return NormalConvertOp<tx::BF16ToFP32Op>(op, adaptor, rewriter);

          else if (isa<Float8E5M2Type>(inputType) && targetType.isBF16())
            return NormalConvertOp<tx::FP8E5M2ToBF16Op>(op, adaptor, rewriter);
          else if (isa<Float8E4M3Type>(inputType) && targetType.isBF16())
            return NormalConvertOp<tx::FP8E4M3ToBF16Op>(op, adaptor, rewriter);
          else if (isa<Float8E4M3FNType>(inputType) && targetType.isBF16())
            return NormalConvertOp<tx::FP8E4M3FNToBF16Op>(op, adaptor,
                                                          rewriter);
          else if (isa<Float4E2M1FNType>(inputType) && targetType.isBF16())
            return NormalConvertOp<tx::FP4E2M1ToBF16Op>(op, adaptor, rewriter);

          else if (isa<Float8E5M2Type>(inputType) && targetType.isF16())
            return NormalConvertOp<tx::FP8E5M2ToFP16Op>(op, adaptor, rewriter);
          else if (isa<Float8E4M3Type>(inputType) && targetType.isF16())
            return NormalConvertOp<tx::FP8E4M3ToFP16Op>(op, adaptor, rewriter);
          else if (isa<Float8E4M3FNType>(inputType) && targetType.isF16())
            return NormalConvertOp<tx::FP8E4M3FNToFP16Op>(op, adaptor,
                                                          rewriter);
          else if (isa<Float4E2M1FNType>(inputType) && targetType.isF16())
            return NormalConvertOp<tx::FP4E2M1ToFP16Op>(op, adaptor, rewriter);
          else
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for ExtFOp "
                    "conversion");
        })
        .Case<math::FmaOp>([&](auto elemWiseOp) {
          return FmaConvertOp(op, adaptor, rewriter);
        })
        .Case<arith::SIToFPOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType())
                                .getElementType();

          if (inputType.isInteger(8) && outputType.isF32()) {
            return ZeroPointConvertOp<tx::INT8ToFP32Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(8) && outputType.isF16()) {
            return ZeroPointConvertOp<tx::INT8ToFP16Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(16) && outputType.isF32()) {
            return RoundConvertOp<tx::INT16ToFP32Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(16) && outputType.isF16()) {
            return NormalConvertOp<tx::INT16ToFP16Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(32) && outputType.isF16()) {
            return RoundConvertOp<tx::INT32ToFP16Op>(op, adaptor, rewriter);
          } else if (inputType.isInteger(32) && outputType.isF32()) {
            return RoundConvertOp<tx::INT32ToFP32Op>(op, adaptor, rewriter);
          } else {
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for integer to "
                    "FP conversion");
          }
        })
        .Case<arith::FPToSIOp>([&](auto elemWiseOp) {
          // TODO: Need add more int to fp convert.
          auto inputType = dyn_cast<MemRefType>(op.getInputs()[0].getType())
                               .getElementType();
          auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType())
                                .getElementType();
          // arith.fptosi: Cast from a value interpreted as floating-point to
          // the nearest (rounding towards zero) signed integer value.
          if (inputType.isF16() && outputType.isInteger(8)) {
            return RoundConvertOp<tx::FP16ToINT8Op>(op, adaptor, rewriter,
                                                    RND_MODE::RND_ZERO);
          } else if (inputType.isF16() && outputType.isInteger(16)) {
            return RoundConvertOp<tx::FP16ToINT16Op>(op, adaptor, rewriter,
                                                     RND_MODE::RND_ZERO);
          } else if (inputType.isF16() && outputType.isInteger(32)) {
            return RoundConvertOp<tx::FP16ToINT32Op>(op, adaptor, rewriter,
                                                     RND_MODE::RND_ZERO);
          } else if (inputType.isF32() && outputType.isInteger(8)) {
            return RoundConvertOp<tx::FP32ToINT8Op>(op, adaptor, rewriter,
                                                    RND_MODE::RND_ZERO);
          } else if (inputType.isF32() && outputType.isInteger(16)) {
            return RoundConvertOp<tx::FP32ToINT16Op>(op, adaptor, rewriter,
                                                     RND_MODE::RND_ZERO);
          } else if (inputType.isF32() && outputType.isInteger(32)) {
            return RoundConvertOp<tx::FP32ToINT32Op>(op, adaptor, rewriter,
                                                     RND_MODE::RND_ZERO);
          } else {
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for fp to "
                    "integer conversion");
          }
        })
        .Case<arith::CmpFOp>([&](auto elemWiseOp) {
          arith::CmpFPredicate predicate = elemWiseOp.getPredicate();
          switch (predicate) {
          case arith::CmpFPredicate::OEQ:
          case arith::CmpFPredicate::UEQ:
            return BoolRelationVVOp<tx::BoolEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::ONE:
          case arith::CmpFPredicate::UNE:
            return BoolRelationVVOp<tx::BoolUnEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OGE:
          case arith::CmpFPredicate::UGE:
            return BoolRelationVVOp<tx::BoolGreaterEqualVV>(op, adaptor,
                                                            rewriter);
          case arith::CmpFPredicate::OGT:
          case arith::CmpFPredicate::UGT:
            return BoolRelationVVOp<tx::BoolGreaterVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OLE:
          case arith::CmpFPredicate::ULE:
            return BoolRelationVVOp<tx::BoolLessEqualVV>(op, adaptor, rewriter);
          case arith::CmpFPredicate::OLT:
          case arith::CmpFPredicate::ULT:
            return BoolRelationVVOp<tx::BoolLessThenVV>(op, adaptor, rewriter);
          default:
            llvm_unreachable("Not yet supported");
            break;
          }
        })
        .Case<arith::TruncFOp>([&](auto elemWiseOp) {
          // May exist elemWiseOp has no result
          auto resultType = elemWiseOp->getResult(0).getType();
          if (resultType.isF16())
            return RoundConvertOp<tx::FP32ToFP16Op>(op, adaptor, rewriter);
          else if (resultType.isBF16())
            return RoundConvertOp<tx::FP32ToBF16Op>(op, adaptor, rewriter);
          else
            return rewriter.notifyMatchFailure(
                op, "Unsupported input/output type combination for trunc "
                    "conversion");
        })
        .Default([&](auto elemWiseOp) {
          // Affine dialect should handled before this pass. So here lower it
          // to scf.for
          return convertLinalgOpToLoops(op, rewriter);
        });
  }
};

struct LinalgReduceConversion : public OpConversionPattern<linalg::ReduceOp> {
  using OpConversionPattern<linalg::ReduceOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(linalg::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto reductionOps = getRegionOps(op);
    // If there is only one reduction operation, try to convert it to Tx81
    // reduce op.
    // TODO: Delete, here use to check linalg-to-mk pass has finished conversion
    // for target supported reduction ops
    if (reductionOps.size() == 1) {
      auto redOp = reductionOps[0];

      auto inputType = cast<MemRefType>(op.getInputs()[0].getType());
      auto elementType = inputType.getElementType();
      // Check if linalg-to-mk pass has finished conversion for target supported
      // reduction ops
      assert(!(isReductionOpAndTypeSupportedByTarget(redOp, elementType) ||
               isReduceToElementWiseOpAndTypeSupportedByTarget(
                   redOp, elementType, inputType.getNumElements(),
                   inputType.getRank())));
    }

    return convertLinalgOpToLoops(op, rewriter);
  }
};

template <typename MKOpT, typename Tx81Op>
struct MKReduceOpConversion : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

public:
  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto input = op.getSrc();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!isSupportedType(inputType)) {
      return failure();
    }
    // TODO: Check init buffer has no init value

    auto axis = op.getAxis();
    assert(axis == 3 || axis == 2);
    auto loc = op->getLoc();

    auto srcPtr = createAddressFromMemref(rewriter, loc, input);
    auto outputPtr = createAddressFromMemref(rewriter, loc, op.getInit());

    auto format = getFormatCode(inputType);

    rewriter.replaceOpWithNewOp<Tx81Op>(
        op, TypeRange{}, srcPtr, outputPtr,
        rewriter.getUI32IntegerAttr(axis == 3 ? 0 /*reduce C dim*/
                                              : 1 /*reduce W dim*/),
        op.getNhwcShapeAttr(), rewriter.getI16IntegerAttr(format));
    return success();
  }
};

template <typename MKOpT, typename TxOpT>
struct BarrierConversion : public OpConversionPattern<MKOpT> {
  using OpConversionPattern<MKOpT>::OpConversionPattern;
  using OpAdaptor = typename MKOpT::Adaptor;

  LogicalResult
  matchAndRewrite(MKOpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    rewriter.create<TxOpT>(loc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct PrintConversion : public OpConversionPattern<mk::PrintOp> {
  using OpConversionPattern<mk::PrintOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(mk::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // printf scalar value.
    if (printScalar(op)) {
      if (op.getNumOperands() == 0) {
        createRuntimePrintScalarCall(rewriter, op.getPrefix(), std::nullopt);
      } else {
        createRuntimePrintScalarCall(rewriter, op.getPrefix(),
                                     adaptor.getOperands()[0], op.getHex(),
                                     op.getIsSigned()[0]);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // print memref value.
    createPrintMemrefCall(op, rewriter);

    rewriter.eraseOp(op);
    return success();
  }

private:
  static std::string getFormatSubstr(Type type, bool hex = false,
                                     std::optional<int> width = std::nullopt,
                                     bool isSigned = false) {
    // If the `value` is a pointer, just return %p.
    if (isa<LLVM::LLVMPointerType>(type)) {
      return "%p";
    }
    // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in
    // the type (so 4 for fp16, 8 for int32, 16 for int64).
    if (hex) {
      // Ignore `width` for `hex` values, pad to typeWidth.
      std::string ret =
          "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
      if (type.getIntOrFloatBitWidth() > 32) {
        ret += "ll";
      }
      ret += "x";
      return ret;
    }

    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    }

    if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + (isSigned ? "lli" : "llu");
      else
        return prefix + (isSigned ? "i" : "u");
    }
    assert(false && "not supported type");
    return "";
  }

  // For printf, need to extend int32 or float64.
  static Value printfPromoteValue(RewriterBase &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    auto loc = UnknownLoc::get(context);
    auto b = LLVM::TritonLLVMOpBuilder(loc, rewriter);

    bool isUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (isUnsigned) {
        return b.zext(ui32_ty, value);
      } else {
        return b.sext(i32_ty, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      return b.fpext(f64_ty, value);
    }

    return value;
  }

  static LLVM::LLVMFuncOp
  getOrAddPrintFuncDecl(ConversionPatternRewriter &rewriter,
                        StringRef funcName = "__Print") {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType = {ptr_ty(ctx)};
    auto funcType =
        LLVM::LLVMFunctionType::get(i32_ty, argsType, /*isVarArg*/ true);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                             funcType);
  }

  static bool printScalar(mk::PrintOp op) {
    // Simply use printf if no operand or the operand is scalar.
    if (op.getNumOperands() == 0)
      return true;

    assert(op.getNumOperands() == 1);
    Type oprType = op.getOperands()[0].getType();
    return (oprType.isIntOrIndexOrFloat() || isa<triton::PointerType>(oprType));
  }

  static void createRuntimePrintScalarCall(ConversionPatternRewriter &rewriter,
                                           StringRef prefix,
                                           std::optional<Value> arg,
                                           bool hex = false,
                                           bool isSigned = false) {
    assert(!prefix.empty() && "printf with empty string not supported");
    auto loc = UnknownLoc::get(rewriter.getContext());
    auto b = LLVM::TritonLLVMOpBuilder(loc, rewriter);

    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);
    os << prefix;
    if (arg.has_value())
      os << getFormatSubstr(arg.value().getType(), hex, std::nullopt, isSigned);

    llvm::SmallString<64> formatStrNewline(formatStr);
    formatStrNewline.push_back('\n');
    formatStrNewline.push_back('\0');
    Value formatStrValue = LLVM::addStringToModule(
        loc, rewriter, "printfFormat_", formatStrNewline);

    SmallVector<Value> allArgs{formatStrValue};
    if (arg.has_value())
      allArgs.push_back(printfPromoteValue(rewriter, arg.value()));
    b.call(getOrAddPrintFuncDecl(rewriter), allArgs);
  }

  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtr = LLVM::LLVMPointerType::get(context);
    return LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtr, true);
  }

  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             StringRef funcName = "__Print") {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return SymbolRefAttr::get(context, funcName);

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName,
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, funcName);
  }

  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value), 0);
    }

    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }

  static void createPrintMemrefCall(mk::PrintOp op,
                                    ConversionPatternRewriter &rewriter) {
    auto loc = op->getLoc();
    auto context = rewriter.getContext();
    auto memRefType = llvm::cast<MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    Type memElementType = memRefType.getElementType();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    std::string formatSpecifierStr = getFormatSubstr(
        memElementType, op.getHex(), std::nullopt, op.getIsSigned()[0]);
    formatSpecifierStr += " \0";
    auto prefix = op.getPrefix();
    std::string prefixNewline = "\n" + prefix.str();
    Value prefixValue = getOrCreateGlobalString(
        loc, rewriter, "frmt_prefix" + prefix.str(),
        StringRef(prefixNewline.c_str(), 128), parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec" + formatSpecifierStr,
        StringRef(formatSpecifierStr.c_str(), 8), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // print prefix firstly.
    rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                  prefixValue);

    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      rewriter.setInsertionPointToEnd(loop.getBody());

      if (i != e - 1)
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                                      newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    Value elementLoad =
        rewriter.create<memref::LoadOp>(loc, op.getOperands()[0], loopIvs);
    if (elementLoad.getType() == rewriter.getF32Type())
      elementLoad = rewriter.create<mlir::LLVM::FPExtOp>(
          loc, rewriter.getF64Type(), elementLoad);
    else if (elementLoad.getType() == rewriter.getI8Type())
      elementLoad = rewriter.create<mlir::LLVM::SExtOp>(
          loc, rewriter.getI32Type(), elementLoad);
    rewriter.create<LLVM::CallOp>(
        loc, getPrintfType(context), printfRef,
        ArrayRef<Value>({formatSpecifierCst, elementLoad}));
  }
};
struct AtomicRMWOpConversion : public OpConversionPattern<mk::AtomicRMWOp> {
  using OpConversionPattern<mk::AtomicRMWOp>::OpConversionPattern;

  LLVM::AtomicBinOp getAtomicBinOp(RMWOp op, Type type) const {
    switch (op) {
    case RMWOp::AND:
      return LLVM::AtomicBinOp::_and;
    case RMWOp::OR:
      return LLVM::AtomicBinOp::_or;
    case RMWOp::XOR:
      return LLVM::AtomicBinOp::_xor;
    case RMWOp::ADD:
      return LLVM::AtomicBinOp::add;
    case RMWOp::FADD:
      return LLVM::AtomicBinOp::fadd;
    case RMWOp::MAX:
      return type.isIntOrIndex() ? LLVM::AtomicBinOp::max
                                 : LLVM::AtomicBinOp::fmax;
    case RMWOp::MIN:
      return type.isIntOrIndex() ? LLVM::AtomicBinOp::min
                                 : LLVM::AtomicBinOp::fmin;
    case RMWOp::UMAX:
      return LLVM::AtomicBinOp::umax;
    case RMWOp::UMIN:
      return LLVM::AtomicBinOp::umin;
    case RMWOp::XCHG:
      return LLVM::AtomicBinOp::xchg;
    default:
      llvm_unreachable("Unexpected atomic op");
    }
  }

  LogicalResult
  matchAndRewrite(mk::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = adaptor.getVal();

    // TODO: Support sem and scope
    if (isa<MemRefType>(val.getType()))
      return failure();

    assert(0 && "Now, tsingmicro backend only support memref type. Don't "
                "support llvm atomic op conversion.");
    auto ptr =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getPtr());

    auto opKind = getAtomicBinOp(op.getAtomicRmwOp(), val.getType());
    auto ordering = getOrdering(op.getSem());
    ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), ptr);

    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(op, opKind, ptr, val,
                                                   ordering);

    return success();
  }
};

struct AtomicCASOpConversion : public OpConversionPattern<mk::AtomicCASOp> {
  using OpConversionPattern<mk::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mk::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto cmp = adaptor.getCmp();
    auto val = adaptor.getVal();
    // TODO: Support sem and scope
    if (isa<MemRefType>(val.getType()))
      return failure();

    assert(0 && "Now, tsingmicro backend only support memref type. Don't "
                "support llvm atomic op conversion. It will be supported in "
                "the future by using  llvm.bitcast and llvm.atomic.cmpxchg.");
    auto ptr =
        createAddressFromMemref(rewriter, op->getLoc(), adaptor.getPtr());

    ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext()), ptr);

    auto ordering = getOrdering(op.getSem());
    auto failureOrdering = ordering != LLVM::AtomicOrdering::monotonic
                               ? LLVM::AtomicOrdering::acquire
                               : ordering;
    // TODO: Use llvm.bitcast to support other types: f32, etc.
    Value cmpXchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, ptr, cmp, val, ordering, failureOrdering);
    Value oldVal = rewriter.create<LLVM::ExtractValueOp>(loc, cmpXchg, 0);
    rewriter.replaceOp(op, oldVal);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Legalize magic kernel operations to be convertible to Tx81 operations
// patterns
//===----------------------------------------------------------------------===//
namespace {

struct LinalgCopyRewrite : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.getInputs().size() == 1 && op.getOutputs().size() == 1 &&
           "LinalgCopyRewrite only supports single input and output");
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getInputs()[0],
                                                op.getOutputs()[0]);
    return success();
  }
};

} // namespace

void mlir::triton::populateMKToTx81CanonicalizationPatterns(
    RewritePatternSet &patterns) {
  // Backend op canonicalization
  patterns.add<LinalgCopyRewrite>(patterns.getContext());
}

void mlir::triton::populateMKToTx81ConversionPatterns(
    RewritePatternSet &patterns) {

  MKToTx81TypeConverter typeConverter;

  // Add type conversion patterns
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);

  // NOTE: Only convert ops that have been legalized to be convertible to Tx81
  // ops.
  // Convert only float type input/output ops to Tx81 ops except some special
  // ops that support any types.
  // clang-format off
  patterns.add<ElementwiseConversion>(patterns.getContext());

  patterns.add<MemoryCopyConvertPattern,
               LinalgReduceConversion,
               MKReduceOpConversion<mk::ReduceMaxOp, tx::ReduceMaxOp>,
               MKReduceOpConversion<mk::ReduceMinOp, tx::ReduceMinOp>,
               MKReduceOpConversion<mk::ReduceSumOp, tx::ReduceSumOp>,
               TransposeOpConversion,
               ReciprocalOpConversionPattern,
               LinalgFillOpConversion,
               MKDotToTx81GemmOpConversion,
               MKDequantOpConversionPattern,
               MKSigmoidToTx81SigmoidOpConversion,
               MKGeluToTx81GeluOpConversion,
               MKArgMinMaxConversionPattern<mk::ArgMinOp, tx::ArgMinOp>,
               MKArgMinMaxConversionPattern<mk::ArgMaxOp, tx::ArgMaxOp>,
               MKMaskMoveOpConversionPattern,
               MKBit2FPOpConversionPattern,
               MKRelationVSOpConversionPattern<mk::BoolEqualVS,tx::BoolEqualVS>,
               MKRelationVSOpConversionPattern<mk::EqualVS,tx::EqualVS>,
               MKRelationVSOpConversionPattern<mk::LessThenVS, tx::LessThenVS>,
               MKRelationVVOpConversionPattern<mk::UnEqualVV, tx::UnEqualVV>,
               MKArithVSOpConversionPattern<mk::AddVS, tx::AddVSOp>,
               MKArithVSOpConversionPattern<mk::SubVS, tx::SubVSOp>,
               MKArithVSOpConversionPattern<mk::MulVS, tx::MulVSOp>,
               MKRelationVVOpConversionPattern<mk::EqualVV,tx::EqualVV>,
               GatherConvertPattern,
               BarrierConversion<mk::BarrierOp, tx::BarrierOp>,
               BarrierConversion<mk::AtomicBarrierInOp,tx::AtomicBarrierInOp>,
               BarrierConversion<mk::AtomicBarrierOutOp,tx::AtomicBarrierOutOp>,
               PrintConversion,
               AtomicRMWOpConversion,
               AtomicCASOpConversion>(
      patterns.getContext());
  // clang-format on
}
