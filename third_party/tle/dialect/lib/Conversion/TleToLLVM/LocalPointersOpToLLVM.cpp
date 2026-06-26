#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"

#include <optional>

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"
namespace {

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

// Extract pointee type from tle.remote_pointers result type.
static Type getRemotePointeeType(Type resultTy) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(resultTy))
    resultTy = tensorTy.getElementType();
  if (auto ptrTy = dyn_cast<triton::PointerType>(resultTy))
    return ptrTy.getPointeeType();
  return Type();
}

// Return bit width for scalar int/float types, or std::nullopt otherwise.
static std::optional<int> getScalarBitWidth(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return intTy.getWidth();
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return floatTy.getWidth();
  return std::nullopt;
}

// Map a shared-memory pointer (addrspace=3) to a cluster-shared pointer (addrspace=7).
Value mapSharedToClusterPointer(ConversionPatternRewriter &rewriter,
                                Location loc, Value ptr, Value ctaId) {
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  if (!ptrTy)
    return Value();
  const unsigned sharedAddrSpace =
      static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared);
  const unsigned clusterSharedAddrSpace =
      static_cast<unsigned>(NVVM::NVVMMemorySpace::SharedCluster);
  if (ptrTy.getAddressSpace() == clusterSharedAddrSpace)
    return ptr;
  if (ptrTy.getAddressSpace() != sharedAddrSpace)
    return Value();
  auto clusterPtrTy =
      LLVM::LLVMPointerType::get(rewriter.getContext(), clusterSharedAddrSpace);
  return NVVM::MapaOp::create(rewriter, loc, clusterPtrTy, ptr, ctaId);
}



// Declare the external flagcxGetIntraPointerC function in the LLVM IR module.
static LLVM::LLVMFuncOp getOrInsertGetPeerPointer(ModuleOp module,
                                                  MLIRContext *ctx) {

  const char *funcName = "flagcxGetIntraPointerC";
  if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
    return func;
  auto i32Ty = IntegerType::get(ctx, 32);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto floatTy = Float32Type::get(ctx);
  auto PtrTy = LLVM::LLVMPointerType::get(ctx, 1);

  auto funcType =
      LLVM::LLVMFunctionType::get(PtrTy, {PtrTy, i64Ty, i32Ty}, false);

  OpBuilder builder(module.getBodyRegion());
  auto func =
      builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);


  func.setLinkage(LLVM::Linkage::External);
  return func;
}

struct LocalPointersOpConversion
    : public ConvertOpToLLVMPattern<tle::LocalPointersOp> {
  LocalPointersOpConversion(LLVMTypeConverter &typeConverter,
                            const TargetInfoBase &targetInfo,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(tle::LocalPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto typeConverter = getTypeConverter();
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[LocalPointersOpConversion] " << msg << "\n";
      return rewriter.notifyMatchFailure(op, msg);
    };

    auto memDescTy = cast<ttg::MemDescType>(op.getSrc().getType());
    auto resultTensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto resultPtrTy = dyn_cast<triton::PointerType>(op.getResult().getType());
    if (!resultTensorTy && !resultPtrTy)
      return reportFailure("local_pointers result must be tensor<ptr> or ptr");
    auto ptrTy =
        resultTensorTy
            ? cast<triton::PointerType>(resultTensorTy.getElementType())
            : resultPtrTy;
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto llvmPtrTy =
        cast<LLVM::LLVMPointerType>(typeConverter->convertType(ptrTy));
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);
    auto i32Ty = rewriter.getIntegerType(32);
    auto ensureI32 = [&](Value v) -> Value {
      if (v.getType() == i32Ty)
        return v;
      if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
        if (intTy.getWidth() > 32)
          return rewriter.create<LLVM::TruncOp>(loc, i32Ty, v);
        if (intTy.isUnsigned())
          return rewriter.create<LLVM::ZExtOp>(loc, i32Ty, v);
        return rewriter.create<LLVM::SExtOp>(loc, i32Ty, v);
      }
      return Value();
    };

    auto sharedEnc = cast<ttg::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kOffset = str_attr("offset");
    LinearLayout regLayout;
    if (resultTensorTy) {
      if (!resultTensorTy.getEncoding())
        return reportFailure(
            "tensor local_pointers result must carry an encoding");
      regLayout = ttg::toLinearLayout(resultTensorTy);
    }
    for (Value operand : op.getIndices()) {
      if (resultTensorTy) {
        auto idxTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!idxTy)
          return reportFailure("tensor result requires ranked-tensor indices");
        if (resultTensorTy.getEncoding() && idxTy.getEncoding() &&
            resultTensorTy.getEncoding() != idxTy.getEncoding())
          return reportFailure(
              "indices tensor encoding must match result encoding");
      } else if (!isa<IntegerType>(operand.getType())) {
        return reportFailure("scalar result requires scalar integer indices");
      }
    }

    const size_t outSize = resultTensorTy ? regLayout.getInDimSize(kReg) : 1;
    SmallVector<Value> outVals(outSize, Value());

    TritonLLVMOpBuilder b(loc, rewriter);
    int elemBits = llvmElemTy.getIntOrFloatBitWidth();
    assert(elemBits % 8 == 0 && "element bitwidth must be byte addressable");
    int elemBytes = elemBits / 8;
    Value elemBytesVal =
        elemBytes > 1 ? b.i32_val(static_cast<int32_t>(elemBytes)) : Value();
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i8PtrTy = LLVM::LLVMPointerType::get(ctx, llvmPtrTy.getAddressSpace());

    SmallVector<unsigned> bufferShape;
    for (int64_t dim : memDescTy.getShape())
      bufferShape.push_back(static_cast<unsigned>(dim));
    auto bufferRank = bufferShape.size();
    auto smemOffsets = smemObj.getOffsets();
    if (smemOffsets.size() != bufferRank)
      return reportFailure("shared memory offsets rank mismatch");

    auto indexVals = adaptor.getIndices();
    const bool hasExplicitIndices = !indexVals.empty();
    if (hasExplicitIndices) {
      if (indexVals.size() != bufferRank)
        return reportFailure("indices must provide buffer-rank values");
    } else {
      if (!resultTensorTy && bufferRank != 0)
        return reportFailure(
            "zero-index scalar local_pointers requires rank-0 buffer");
      if (resultTensorTy && resultTensorTy.getShape() != memDescTy.getShape())
        return reportFailure(
            "zero-index tensor local_pointers requires full buffer shape");
    }

    SmallVector<SmallVector<Value>> indexElems;
    if (hasExplicitIndices) {
      indexElems.reserve(indexVals.size());
      for (Value indexVal : indexVals) {
        if (resultTensorTy) {
          auto elems = unpackLLElements(loc, indexVal, rewriter);
          if (elems.size() != outVals.size())
            return reportFailure(
                "indices tensors must match local_pointers result shape");
          indexElems.push_back(std::move(elems));
        } else {
          Value scalar = ensureI32(indexVal);
          if (!scalar)
            return reportFailure("scalar indices must lower to i32 values");
          indexElems.push_back(SmallVector<Value>{scalar});
        }
      }
    } else if (resultTensorTy) {
      auto fullCoords =
          emitIndices(loc, rewriter, targetInfo, resultTensorTy.getEncoding(),
                      resultTensorTy,
                      /*withCTAOffset=*/false);
      if (fullCoords.size() != outVals.size())
        return reportFailure(
            "failed to synthesize full indices for local_pointers");
      indexElems.assign(bufferRank, SmallVector<Value>{});
      for (size_t idx = 0; idx < fullCoords.size(); ++idx) {
        if (fullCoords[idx].size() != bufferRank)
          return reportFailure("synthesized full indices rank mismatch");
        for (size_t dim = 0; dim < bufferRank; ++dim) {
          Value coord = ensureI32(fullCoords[idx][dim]);
          if (!coord)
            return reportFailure(
                "synthesized full indices must lower to i32 values");
          indexElems[dim].push_back(coord);
        }
      }
    }

    for (size_t idx = 0; idx < outVals.size(); ++idx) {
      SmallVector<Value> idxCoords;
      idxCoords.reserve(bufferRank);
      for (size_t dim = 0; dim < indexElems.size(); ++dim) {
        Value val = ensureI32(indexElems[dim][idx]);
        if (!val)
          return reportFailure("indices must lower to i32 scalars");
        Value offset = smemOffsets[dim];
        Value offVal = ensureI32(offset);
        if (!offVal)
          return reportFailure("shared memory offsets must be i32");
        idxCoords.push_back(b.add(val, offVal));
      }

      Value elemOffset;
      if (bufferRank == 0) {
        elemOffset = b.i32_val(0);
      } else if (auto paddedEnc =
                     dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEnc)) {
        auto order = ttg::getOrder(sharedEnc, memDescTy.getShape());
        elemOffset =
            LLVM::linearize(rewriter, loc, idxCoords, bufferShape, order);
      } else {
        auto dimNames = standardOutDimNames(ctx, bufferRank);
        SmallVector<std::pair<StringAttr, Value>> logicalOffsets;
        logicalOffsets.reserve(bufferRank);
        for (auto [dim, offset] : llvm::zip_equal(dimNames, idxCoords))
          logicalOffsets.push_back({dim, offset});
        LinearLayout sharedLayout = ttg::toLinearLayout(memDescTy);
        sharedLayout = sharedLayout.sublayout({kOffset}, dimNames);
        LinearLayout invSharedLayout = sharedLayout.invert();

        // Be robust to non-canonical input ordering produced by upstream
        // transformations: reorder offsets to match the inverted layout's
        // expected in-dim order before applying the mapping.
        SmallVector<std::pair<StringAttr, Value>> orderedLogicalOffsets;
        orderedLogicalOffsets.reserve(invSharedLayout.getNumInDims());
        for (StringAttr inDim : invSharedLayout.getInDimNames()) {
          bool found = false;
          for (auto &logical : logicalOffsets) {
            if (logical.first == inDim) {
              orderedLogicalOffsets.push_back(logical);
              found = true;
              break;
            }
          }
          if (!found)
            return reportFailure(
                "missing logical offset for inverted shared-layout in-dim");
        }

        auto remappedOffsets = applyLinearLayout(loc, rewriter, invSharedLayout,
                                                 orderedLogicalOffsets);
        if (remappedOffsets.empty())
          return reportFailure("failed to remap shared-memory linear offsets");

        bool foundOffset = false;
        for (auto &mapped : remappedOffsets) {
          if (mapped.first == kOffset) {
            elemOffset = mapped.second;
            foundOffset = true;
            break;
          }
        }
        if (!foundOffset)
          return reportFailure(
              "remapped shared layout does not contain offset");
      }

      Value byteOffset = elemOffset;
      if (elemBytes > 1)
        byteOffset = b.mul(byteOffset, elemBytesVal);
      if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(sharedEnc)) {
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, elemBits,
                                      byteOffset, /*offsetInBytes=*/true);
        byteOffset = b.add(byteOffset, padOffset);
      }

      Value ptrI8 = b.bitcast(smemObj.getBase(), i8PtrTy);
      Value advanced = b.gep(i8PtrTy, i8Ty, ptrI8, byteOffset,
                             LLVM::GEPNoWrapFlags::inbounds);
      outVals[idx] = b.bitcast(advanced, llvmPtrTy);
    }

    if (resultTensorTy) {
      Value result =
          packLLElements(loc, typeConverter, outVals, rewriter, resultTensorTy);
      rewriter.replaceOp(op, result);
    } else {
      rewriter.replaceOp(op, outVals.front());
    }
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

LogicalResult lowerClusterSpace(Location loc, ValueRange srcElems,
                                ValueRange shardElems,
                                ConversionPatternRewriter &rewriter,
                                SmallVectorImpl<Value> &resultPtrs) {

  auto reportFailure = [&](StringRef msg) -> LogicalResult {
    llvm::errs() << "[RemotePointersOpConversion][cluster] " << msg << "\n";
    return failure();
  };

  auto ensureI32 = [&](Value v) -> Value {
    if (!v)
      return Value();
    if (v.getType().isInteger(32))
      return v;
    auto intTy = dyn_cast<IntegerType>(v.getType());
    if (!intTy)
      return Value();
    if (intTy.getWidth() > 32)
      return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), v);
    if (intTy.isUnsigned())
      return rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI32Type(), v);
    return rewriter.create<LLVM::SExtOp>(loc, rewriter.getI32Type(), v);
  };

  resultPtrs.reserve(srcElems.size());

  for (auto [idx, srcPtr] : llvm::enumerate(srcElems)) {
    if (!isa<LLVM::LLVMPointerType>(srcPtr.getType()))
      return reportFailure("source elements must lower to LLVM pointers");

    Value shardVal =
        shardElems.size() == 1 ? shardElems.front() : shardElems[idx];

    Value ctaId = ensureI32(shardVal);
    if (!ctaId)
      return reportFailure("shard_id must lower to i32");

    Value mappedPtr = mapSharedToClusterPointer(rewriter, loc, srcPtr, ctaId);

    if (!mappedPtr)
      return reportFailure("expected shared/cluster-shared address space");

    resultPtrs.push_back(mappedPtr);
  }

  return success();
}

LogicalResult lowerDeviceSpace(Location loc, ValueRange srcElems,
                               ValueRange shardElems, Value offsetVal,
                               int elemBytes,
                               ConversionPatternRewriter &rewriter,
                               SmallVectorImpl<Value> &resultPtrs) {
  ModuleOp module =
      rewriter.getInsertionPoint()->getParentOp()->getParentOfType<ModuleOp>();

  if (!module) {
    return rewriter.notifyMatchFailure(loc, "expected module context");
  }
  auto func = getOrInsertGetPeerPointer(module, rewriter.getContext());

  Value memPtr = srcElems[0];
  Value peer = shardElems[0];

  auto i64Ty = rewriter.getI64Type();

  Value offsetI64;
  if (offsetVal && offsetVal.getType() != i64Ty) {
    auto offsetIntTy = dyn_cast<IntegerType>(offsetVal.getType());
    if (!offsetIntTy)
      return failure();
    if (offsetIntTy.getWidth() < 64)
      offsetI64 = rewriter.create<arith::ExtSIOp>(loc, i64Ty, offsetVal);
    else
      offsetI64 = rewriter.create<arith::TruncIOp>(loc, i64Ty, offsetVal);
  } else if (offsetVal) {
    offsetI64 = offsetVal;
  } else {
    return failure();
  }

  Value byteOffset = offsetI64;
  if (elemBytes != 1) {
    Value elemBytesVal =
        rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    byteOffset = rewriter.create<arith::MulIOp>(loc, offsetI64, elemBytesVal);
  }

  auto intTy = dyn_cast<IntegerType>(memPtr.getType());
  if (!intTy || intTy.getWidth() != 64)
    return failure();

  auto ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx, 1);

  Value comm_dev_ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, memPtr);

  auto isGlobalAddrSpace = [&](auto ptrTy) -> bool {
    auto addrSpace = ptrTy.getAddressSpace();
    return addrSpace == static_cast<unsigned>(NVVM::NVVMMemorySpace::Global);
  };

  if (!isGlobalAddrSpace(ptrTy))
    return failure();

  auto getPeerPtrCall = rewriter.create<LLVM::CallOp>(
      loc, TypeRange{func.getFunctionType().getReturnType()},
      FlatSymbolRefAttr::get(func), ValueRange{comm_dev_ptr, byteOffset, peer});

  Value peerPtr = getPeerPtrCall.getResult();
  auto peerPtrTy = dyn_cast<LLVM::LLVMPointerType>(peerPtr.getType());

  if (!isGlobalAddrSpace(peerPtrTy))
    return failure();
  resultPtrs.push_back(peerPtr);

  return success();
}

LogicalResult lowerNodeSpace(Location loc, ValueRange srcElems,
                             ValueRange shardElems,
                             ConversionPatternRewriter &rewriter,
                             SmallVectorImpl<Value> &resultPtrs) {
  return failure(); // Not implemented yet
}

struct RemotePointersOpConversion
    : public ConvertOpToLLVMPattern<tle::RemotePointersOp> {
  RemotePointersOpConversion(LLVMTypeConverter &typeConverter,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::RemotePointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    auto reportFailure = [&](StringRef msg) -> LogicalResult {
      llvm::errs() << "[RemotePointersOpConversion] " << msg << "\n";
      return rewriter.notifyMatchFailure(op, msg);
    };
    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (srcElems.empty())
      return reportFailure("expected non-empty source pointer elements");

    auto shardElems = unpackLLElements(loc, adaptor.getShardId(), rewriter);
    if (shardElems.empty())
      return reportFailure("expected non-empty shard_id elements");
    if (shardElems.size() != 1 && shardElems.size() != srcElems.size())
      return reportFailure(
          "shard_id must be scalar or match source pointer element count");

    auto space = adaptor.getSpace();

    Value offsetVal;
    if (adaptor.getOffset()) {
      auto offsetElems = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      if (offsetElems.size() != 1)
        return reportFailure("offset must be scalar");
      offsetVal = offsetElems[0];
    }

    SmallVector<Value> mappedPtrs;
    if (space == "cluster") {
      if (failed(lowerClusterSpace(loc, srcElems, shardElems, rewriter,
                                   mappedPtrs))) {
        return rewriter.notifyMatchFailure(op, "cluster lowering failed");
      }
    } else if (space == "device") {
      int elemBytes = 1;
      if (offsetVal) {
        Type resultPointeeTy = getRemotePointeeType(op.getType());
        if (!resultPointeeTy)
          return reportFailure("result must be tt.ptr or tensor<tt.ptr>");
        auto elemBits = getScalarBitWidth(resultPointeeTy);
        if (!elemBits.has_value())
          return reportFailure(
              "result pointee type must be scalar int or float");
        elemBytes = elemBits.value() / 8;
      }
      if (failed(lowerDeviceSpace(loc, srcElems, shardElems, offsetVal,
                                  elemBytes, rewriter, mappedPtrs))) {
        return rewriter.notifyMatchFailure(op, "device lowering failed");
      }
    } else if (space == "node") {
      if (failed(lowerNodeSpace(loc, srcElems, shardElems, rewriter,
                                mappedPtrs))) {
        return rewriter.notifyMatchFailure(op, "node lowering failed");
      }
    } else {
      return reportFailure("unsupported remote space: " + space.str());
    }

    Value packed =
        packLLElements(loc, typeConverter, mappedPtrs, rewriter, op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }
};

} // namespace

void tle::populateLocalPointersOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LocalPointersOpConversion>(typeConverter, targetInfo, benefit);
}

void tle::populateRemotePointersOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  (void)targetInfo;
  patterns.add<RemotePointersOpConversion>(typeConverter, benefit);
}
