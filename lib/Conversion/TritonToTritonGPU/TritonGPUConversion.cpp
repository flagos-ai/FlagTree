#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

#include <algorithm>
#include <numeric>
#ifdef __TLE__
#include <optional>
#endif

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#ifdef __TLE__
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "tle/dialect/include/IR/Dialect.h" // flagtree tle raw
#endif
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

//
// TypeConverter
//
TritonGPUTypeConverter::TritonGPUTypeConverter(MLIRContext *context,
                                               int numWarps, int threadsPerWarp,
                                               int numCTAs,
                                               bool enableSourceRemat)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  // Add encoding for tensor
  addConversion([this](RankedTensorType tensorType) -> RankedTensorType {
#ifdef __TLE__
    return convertRankedTensorType(tensorType, this->numWarps);
#else
    // types with encoding are already in the right format
    // TODO: check for layout encodings more specifically
    if (tensorType.getEncoding())
      return tensorType;
    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::gpu::BlockedEncodingAttr encoding =
        getDefaultBlockedEncoding(this->context, shape, this->numWarps,
                                  this->threadsPerWarp, this->numCTAs);
    return tensorType.cloneWithEncoding(encoding);
#endif
  });

  // Add encoding for tensor pointer
  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    // Check whether tensor pointer `tt.ptr<tensor<>>`
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (pointeeTensorType == nullptr)
      return ptrType;

    // Add layout into the tensor
    auto convertedTensorType = convertType(pointeeTensorType);
    return triton::PointerType::get(convertedTensorType,
                                    ptrType.getAddressSpace());
  });

#ifdef __TLE__
  addConversion([this](Value value) -> std::optional<Type> {
    Type type = value.getType();
    int valueNumWarps = getNumWarps(value);
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return convertRankedTensorType(tensorType, valueNumWarps);

    if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      auto pointeeTensorType =
          dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      if (pointeeTensorType)
        return triton::PointerType::get(
            convertRankedTensorType(pointeeTensorType, valueNumWarps),
            ptrType.getAddressSpace());
    }

    return std::nullopt;
  });
#endif

  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  if (enableSourceRemat) {
    addSourceMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                                ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, tensorType,
                                                inputs)
          .getResult(0);
    });
  }

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                              ValueRange inputs, Location loc) {
    auto cast =
        triton::gpu::ConvertLayoutOp::create(builder, loc, tensorType, inputs);
    return cast.getResult();
  });
}

#ifdef __TLE__
int TritonGPUTypeConverter::getNumWarps(Value value) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (Block *owner = blockArg.getOwner()) {
      if (Region *region = owner->getParent()) {
        if (region->getParentOp())
          return lookupNumWarps(region);
      }
    }
  }
  if (Operation *op = value.getDefiningOp()) {
    if (std::optional<int> contextualNumWarps = maybeLookupNumWarps(op))
      return *contextualNumWarps;
  }
  return numWarps;
}

RankedTensorType
TritonGPUTypeConverter::convertRankedTensorType(RankedTensorType tensorType,
                                                int contextualNumWarps) const {
  // Types with encoding are already in the right format.
  // TODO: check for layout encodings more specifically.
  if (tensorType.getEncoding())
    return tensorType;
  ArrayRef<int64_t> shape = tensorType.getShape();
  triton::gpu::BlockedEncodingAttr encoding = getDefaultBlockedEncoding(
      context, shape, contextualNumWarps, threadsPerWarp, numCTAs);
  return tensorType.cloneWithEncoding(encoding);
}
#endif

//
// TritonGPUConversion
//
TritonGPUConversionTarget::TritonGPUConversionTarget(
    MLIRContext &context, TritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  // TODO: we should also verify ops of TritonGPUDialect
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  // Some ops from SCF are illegal
  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

#ifdef __TLE__
  // flagtree tle raw
  addDynamicallyLegalOp<triton::gpu::LocalAllocOp, triton::gpu::LocalStoreOp,
                        triton::gpu::LocalLoadOp>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });
#endif
  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect, ub::UBDialect,
#ifdef __TLE__
                             LLVM::LLVMDialect // flagtree tle raw
#endif
                             >(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });

  // We have requirements for the data layouts
  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        cast<RankedTensorType>(dotOp.getA().getType()).getEncoding();
    Attribute bEncoding =
        cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
    if (aEncoding && isa<triton::gpu::DotOperandEncodingAttr>(aEncoding) &&
        bEncoding && isa<triton::gpu::DotOperandEncodingAttr>(bEncoding))
      return true;
    return false;
  });
  addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) -> bool {
    for (auto arg : funcOp.getArguments()) {
      if (auto tensor = dyn_cast<RankedTensorType>(arg.getType())) {
        if (!tensor.getEncoding())
          return false;
      }
    }
    return true;
  });

#ifdef __TLE__
  // flagtree tle raw
  addDynamicallyLegalDialect<triton::tle::TleDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    return hasLegalRegions && typeConverter.isLegal(op);
  });
}
#endif

bool TritonGPUConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {
  bool hasLegalRegions = true;
  for (auto &region : op->getRegions()) {
    hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
  }
  if (hasLegalRegions && typeConverter.isLegal(op)) {
    return true;
  }
  return false;
}

// This function returns the layout to use for gather/scatter indices. The
// `gather4` and `scatter4` TMA instructions require 4 consecutive indices.
// Thus, threads issuing these instructions must have all 4 index elements
// available.
static RankedTensorType getNewIndicesType(RankedTensorType type,
                                          unsigned numThreads,
                                          unsigned numWarps) {
  assert(type.getRank() == 1);
  auto enc = cast<DistributedEncodingTrait>(type.getEncoding());

  // Technically any layout where we have a pack of 4 neighbouring elements plus
  // broadcasted over the warp dimension is okay but for now we just pick a
  // layout.
  std::array<unsigned, 2> sizePerThread{1, 4};
  std::array<unsigned, 2> threadsPerWarp = {numThreads, 1};
  std::array<unsigned, 2> order = {1, 0};
  std::array<unsigned, 2> warpsPerCta = {1, numWarps};

  MLIRContext *ctx = type.getContext();
  auto ctaLayout = CTAEncodingAttr::getDefault(ctx, /*rank=*/2);
  auto parentEncoding = BlockedEncodingAttr::get(
      ctx, sizePerThread, threadsPerWarp, warpsPerCta, order, ctaLayout);
  auto newEncoding = SliceEncodingAttr::get(ctx, /*dim=*/0, parentEncoding);
  if (enc == newEncoding)
    return {};

  return type.cloneWithEncoding(newEncoding);
}

// Function for converting any gather or scatter op that requires a specific
// index layout. This also handles converting result types if there are any.
static LogicalResult convertGatherScatterIndices(Operation *op,
                                                 OpOperand &indices,
                                                 ConversionPatternRewriter &b) {
  auto type = cast<RankedTensorType>(indices.get().getType());
  RankedTensorType newType =
      getNewIndicesType(type, lookupThreadsPerWarp(b), lookupNumWarps(op));
  if (!newType)
    return failure();
  Value index =
      ConvertLayoutOp::create(b, op->getLoc(), newType, indices.get());
  indices.set(index);
  return success();
}

LogicalResult impl::convertGatherScatterOp(
    Operation *op, ValueRange operands, OpOperand &xOffsetsMutable,
    const TypeConverter &typeConverter, ConversionPatternRewriter &rewriter) {
  LogicalResult result = success();
  rewriter.modifyOpInPlace(op, [&] {
    for (auto [operand, value] : llvm::zip(op->getOpOperands(), operands))
      operand.set(value);
    for (OpResult result : op->getOpResults())
#ifdef __TLE__
      result.setType(typeConverter.convertType(result));
#else
      result.setType(typeConverter.convertType(result.getType()));
#endif
    result = convertGatherScatterIndices(op, xOffsetsMutable, rewriter);
  });
  return result;
}
