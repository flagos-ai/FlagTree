//===------------------- Tx81MemrefToLLVM.cpp------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Tx81MemrefToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/AllocLikeConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "utils/utils.h"
#include <cstdint>
#include <vector>

#define DEBUG_TYPE "tx81-memref-to-llvm"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "tsingmicro-tx81/Conversion/Tx81MemrefToLLVM/Passes.h.inc"

// Used for allocate spm memory
uint64_t spmPointer = 0x10000;

namespace {

//===----------------------------------------------------------------------===//
// Tx81 Custom MemRef Op Conversion Patterns
//===----------------------------------------------------------------------===//

struct TsmMemRefAllocOpLowering : public AllocLikeOpLLVMLowering {
  TsmMemRefAllocOpLowering(const LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<Value, Value>
  allocateBufferFromSPM(ConversionPatternRewriter &rewriter, Location loc,
                        Operation *op) const {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    MemRefType memRefType = getMemRefResultType(op);

    // Align spm address.
    if (allocOp.getAlignment().has_value()) {
      auto alignment = allocOp.getAlignment().value();
      spmPointer = (spmPointer + alignment - 1) & ~(alignment - 1);
    }
    Value spmOffsetOp = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(), rewriter.getI32IntegerAttr(spmPointer));
    Type elementType = typeConverter->convertType(memRefType.getElementType());
    auto elementPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value spmAddr = rewriter.create<LLVM::ZeroOp>(loc, elementPtrType);

    spmAddr = rewriter.create<LLVM::PtrToIntOp>(op->getLoc(),
                                                rewriter.getI64Type(), spmAddr);
    spmAddr = rewriter.create<LLVM::AddOp>(op->getLoc(), rewriter.getI64Type(),
                                           spmAddr, spmOffsetOp);

    spmAddr = rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), elementPtrType,
                                                spmAddr);
    Value allocatedPtr = spmAddr;
    if (!allocatedPtr)
      return std::make_tuple(Value(), Value());
    Value alignedPtr = allocatedPtr;

    // update spm pointer
    auto elemCount = memRefType.getNumElements();
    auto bitWidth = memRefType.getElementTypeBitWidth();

    uint64_t totalByte = (elemCount * bitWidth + 7) / 8;
    spmPointer += totalByte;

    return std::make_tuple(allocatedPtr, alignedPtr);
  }

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    return allocateBufferFromSPM(rewriter, loc, op);
  }
};

template <typename MemrefOp>
struct MemrefLoadOrStoreOpLowering : public ConvertOpToLLVMPattern<MemrefOp> {

  using ConvertOpToLLVMPattern<MemrefOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename MemrefOp::Adaptor;

  LogicalResult
  matchAndRewrite(MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto type = op.getMemRefType();
    bool isMaskEle = type.getElementType().isInteger(1);
    MemRefDescriptor memRefDescriptor(adaptor.getMemref());
    Type indexType = ConvertToLLVMPattern::getTypeConverter()->getIndexType();

    Value dataPtr, Offset;
    if (isMaskEle) {
      dataPtr = memRefDescriptor.alignedPtr(rewriter, loc);
      Value index = memRefDescriptor.offset(rewriter, loc);
      auto indices = adaptor.getIndices();
      for (int i = 0, e = indices.size(); i < e; ++i) {
        Value stride = memRefDescriptor.stride(rewriter, loc, i);
        Value increment = rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
        index = rewriter.create<LLVM::AddOp>(loc, index, increment);
      }

      Value MaskC = rewriter.create<LLVM::ConstantOp>(loc, indexType, 7);
      Value ShrAmtC = rewriter.create<LLVM::ConstantOp>(loc, indexType, 3);
      Offset = rewriter.create<LLVM::AndOp>(loc, index, MaskC);
      Offset =
          rewriter.create<LLVM::TruncOp>(loc, rewriter.getI8Type(), Offset);
      index = rewriter.create<LLVM::AShrOp>(loc, index, ShrAmtC);
      dataPtr = rewriter.create<LLVM::GEPOp>(
          loc, memRefDescriptor.getElementPtrType(), rewriter.getI8Type(),
          dataPtr, index);
    } else {
      dataPtr = ConvertToLLVMPattern::getStridedElementPtr(
          op.getLoc(), type, adaptor.getMemref(), adaptor.getIndices(),
          rewriter);
    }

    // TODO: Add spm offset according the memory space
    auto intPtrType = ConvertToLLVMPattern::getIntPtrType(
        memRefDescriptor.getElementPtrType().getAddressSpace());
    Value ptrValue =
        rewriter.create<LLVM::PtrToIntOp>(op.getLoc(), intPtrType, dataPtr);

    // Workaround: Should add memory space analysis pass.
    Operation *opBase = op;
    if (!opBase->hasAttr("isSpm")) {
      return rewriter.notifyMatchFailure(
          op, "Load/Store should have isSpm attribute.");
    }
    int isSpm =
        cast<IntegerAttr>(opBase->getAttr("isSpm")).getValue().getSExtValue();

    Value adjustedPtr = dataPtr;
    if (isSpm) {
      // Get the module for function declarations
      auto module = op->template getParentOfType<ModuleOp>();
      // Types for function declaration
      SmallVector<Type, 5> argTypes = {
          rewriter.getI64Type() // offset
      };

      auto i8PtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(),
          *ConvertToLLVMPattern::getTypeConverter()->getMemRefAddressSpace(
              type));
      // Declare the function
      Value funcPtr = triton::utils::declareTx81Function(
          module, rewriter, op.getLoc(), "get_spm_memory_mapping_wrapper",
          i8PtrTy, argTypes);

      // Create the call to __Rdma
      auto spmMemoryAddrPtr = rewriter.create<LLVM::CallOp>(
          op.getLoc(), TypeRange{i8PtrTy},
          "get_spm_memory_mapping_wrapper", // funcPtr,
          ValueRange{ptrValue});

      adjustedPtr = spmMemoryAddrPtr.getResult();
    }

    // Wether need memoryspace cast
    if constexpr (std::is_same<MemrefOp, memref::LoadOp>()) {
      if (isMaskEle) {
        Value newVal = rewriter.create<LLVM::LoadOp>(loc, rewriter.getI8Type(),
                                                     adjustedPtr, 0, false,
                                                     op.getNontemporal());
        Value Zero =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 0);
        Value MaskC =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 1);
        MaskC = rewriter.create<LLVM::ShlOp>(loc, MaskC, Offset);
        newVal = rewriter.create<LLVM::AndOp>(loc, newVal, MaskC);
        newVal = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                               newVal, Zero);
        rewriter.replaceOp(op, ValueRange{newVal});
      } else {
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
            op, op.getType(), adjustedPtr, 0, false, op.getNontemporal());
      }
    } else {
      Value StoreVal = adaptor.getValue();
      if (isMaskEle) {
        Value srcVal = rewriter.create<LLVM::LoadOp>(loc, rewriter.getI8Type(),
                                                     adjustedPtr, 0, false,
                                                     op.getNontemporal());
        Value MaskC =
            rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI8Type(), 1);
        MaskC = rewriter.create<LLVM::ShlOp>(loc, MaskC, Offset);
        Value TrueVal = rewriter.create<LLVM::OrOp>(loc, srcVal, MaskC);
        Value FalseVal = rewriter.create<LLVM::XOrOp>(loc, TrueVal, MaskC);
        StoreVal =
            rewriter.create<LLVM::SelectOp>(loc, StoreVal, TrueVal, FalseVal);
      }

      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, StoreVal, adjustedPtr, 0,
                                                 false, op.getNontemporal());
    }

    return success();
  }
};

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = castOp.getSource().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  /// Extracts allocated, aligned pointers and offset from a ranked or unranked
  /// memref type. In unranked case, the fields are extracted from the
  /// underlying ranked descriptor.
  void extractPointersAndOffset(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &typeConverter,
                                Value originalOperand, Value convertedOperand,
                                Value *allocatedPtr, Value *alignedPtr,
                                Value *offset = nullptr) const {
    Type operandType = originalOperand.getType();
    if (isa<MemRefType>(operandType)) {
      MemRefDescriptor desc(convertedOperand);
      *allocatedPtr = desc.allocatedPtr(rewriter, loc);
      *alignedPtr = desc.alignedPtr(rewriter, loc);
      if (offset != nullptr)
        *offset = desc.offset(rewriter, loc);
      return;
    }

    // These will all cause assert()s on unconvertible types.
    unsigned memorySpace = *typeConverter.getMemRefAddressSpace(
        cast<UnrankedMemRefType>(operandType));
    auto elementPtrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(), memorySpace);

    // Extract pointer to the underlying ranked memref descriptor and cast it to
    // ElemType**.
    UnrankedMemRefDescriptor unrankedDesc(convertedOperand);

    // FIXME: workaround, take memRefDescPtr as naked ptr.
    Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);
    *allocatedPtr = underlyingDescPtr;
    *alignedPtr = underlyingDescPtr;

    if (offset != nullptr) {
      *offset = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getI32IntegerAttr(0));
    }
  }

  LogicalResult convertSourceMemRefToDescriptor(
      ConversionPatternRewriter &rewriter, Type srcType,
      memref::ReinterpretCastOp castOp,
      memref::ReinterpretCastOp::Adaptor adaptor, Value *descriptor) const {
    MemRefType targetMemRefType =
        cast<MemRefType>(castOp.getResult().getType());
    auto llvmTargetDescriptorTy = dyn_cast_or_null<LLVM::LLVMStructType>(
        typeConverter->convertType(targetMemRefType));
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::poison(rewriter, loc, llvmTargetDescriptorTy);

    // Set allocated and aligned pointers.
    Value allocatedPtr, alignedPtr;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             castOp.getSource(), adaptor.getSource(),
                             &allocatedPtr, &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.getOffsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.getSizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.getStrides()[dynStrideId++]);
      else
        desc.setConstantStride(rewriter, loc, i, castOp.getStaticStride(i));
    }
    *descriptor = desc;
    return success();
  }
};

/// Materialize the MemRef descriptor represented by the results of
/// ExtractStridedMetadataOp.
class ExtractStridedMetadataOpLowering
    : public ConvertOpToLLVMPattern<memref::ExtractStridedMetadataOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractStridedMetadataOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp extractStridedMetadataOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!LLVM::isCompatibleType(adaptor.getOperands().front().getType()))
      return failure();

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.getSource());
    Location loc = extractStridedMetadataOp.getLoc();
    Value source = extractStridedMetadataOp.getSource();

    auto sourceMemRefType = cast<MemRefType>(source.getType());
    int64_t rank = sourceMemRefType.getRank();
    SmallVector<Value> results;
    results.reserve(2 + rank * 2);

    // Base buffer.
    Value baseBuffer = sourceMemRef.allocatedPtr(rewriter, loc);
    Value alignedBuffer = sourceMemRef.alignedPtr(rewriter, loc);
    MemRefDescriptor dstMemRef = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(),
        cast<MemRefType>(extractStridedMetadataOp.getBaseBuffer().getType()),
        baseBuffer, alignedBuffer);
    results.push_back((Value)dstMemRef);

    // Offset.
    results.push_back(sourceMemRef.offset(rewriter, loc));

    // Sizes.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.size(rewriter, loc, i));
    // Strides.
    for (unsigned i = 0; i < rank; ++i)
      results.push_back(sourceMemRef.stride(rewriter, loc, i));

    rewriter.replaceOp(extractStridedMetadataOp, results);
    return success();
  }
};

/// Unpack the pointer returned by a memref.extract_aligned_pointer_as_index.
class ConvertExtractAlignedPointerAsIndex
    : public ConvertOpToLLVMPattern<memref::ExtractAlignedPointerAsIndexOp> {
public:
  using ConvertOpToLLVMPattern<
      memref::ExtractAlignedPointerAsIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractAlignedPointerAsIndexOp extractOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BaseMemRefType sourceTy = extractOp.getSource().getType();

    Value alignedPtr;
    if (sourceTy.hasRank()) {
      MemRefDescriptor desc(adaptor.getSource());
      alignedPtr = desc.alignedPtr(rewriter, extractOp->getLoc());
    } else {
      auto elementPtrTy = LLVM::LLVMPointerType::get(
          rewriter.getContext(), sourceTy.getMemorySpaceAsInt());

      UnrankedMemRefDescriptor desc(adaptor.getSource());
      Value descPtr = desc.memRefDescPtr(rewriter, extractOp->getLoc());

      alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
          rewriter, extractOp->getLoc(), *getTypeConverter(), descPtr,
          elementPtrTy);
    }

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(
        extractOp, getTypeConverter()->getIndexType(), alignedPtr);
    return success();
  }
};

// Copy from llvm-project MemrefToLLVM.cpp
// FIXME: Use ptr dialect to fix the error between un-ranked memref to llvm ptr
struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern<memref::CastOp>::ConvertOpToLLVMPattern;

public:
  LogicalResult match(memref::CastOp memRefCastOp) const override {
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // memref::CastOp reduce to bitcast in the ranked MemRef case and can be
    // used for type erasure. For now they must preserve underlying element type
    // and require source and result type to have the same rank. Therefore,
    // perform a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (isa<MemRefType>(srcType) && isa<MemRefType>(dstType))
      return success(typeConverter->convertType(srcType) ==
                     typeConverter->convertType(dstType));

    // At least one of the operands is unranked type
    assert(isa<UnrankedMemRefType>(srcType) ||
           isa<UnrankedMemRefType>(dstType));

    // Unranked to unranked cast is disallowed
    return !(isa<UnrankedMemRefType>(srcType) &&
             isa<UnrankedMemRefType>(dstType))
               ? success()
               : failure();
  }

  void rewrite(memref::CastOp memRefCastOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = typeConverter->convertType(memRefCastOp.getType());
    auto loc = memRefCastOp.getLoc();

    // For ranked/ranked case, just keep the original descriptor.
    if (isa<MemRefType>(srcType) && isa<MemRefType>(dstType))
      return rewriter.replaceOp(memRefCastOp, {adaptor.getSource()});

    if (isa<MemRefType>(srcType) && isa<UnrankedMemRefType>(dstType)) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = cast<MemRefType>(srcType);
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, adaptor.getSource(), rewriter);

      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(rank));
      // poison = PoisonOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::poison(rewriter, loc, targetStructType);
      // d1 = InsertValueOp poison, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, ptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, ptr);
      rewriter.replaceOp(memRefCastOp, (Value)memRefDesc);

    } else if (isa<UnrankedMemRefType>(srcType) && isa<MemRefType>(dstType)) {
      // Casting from unranked type to ranked.
      // The operation is assumed to be doing a correct cast. If the destination
      // type mismatches the unranked the type, it is undefined behavior.
      UnrankedMemRefDescriptor memRefDesc(adaptor.getSource());
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);

      auto desc = MemRefDescriptor::poison(rewriter, loc, targetStructType);
      // FIXME: workaround, take memRefDescPtr as naked ptr.
      desc.setAllocatedPtr(rewriter, loc, ptr);
      desc.setAlignedPtr(rewriter, loc, ptr);

      rewriter.replaceOp(memRefCastOp, SmallVector<Value>{desc});
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

} // namespace

void mlir::triton::populateTx81MemrefToLLVMConversionPatterns(
    RewritePatternSet &patterns, LLVMTypeConverter &converter) {
  // clang-format off
  patterns.add<TsmMemRefAllocOpLowering,
                MemRefReinterpretCastOpLowering,
                ExtractStridedMetadataOpLowering,
                ConvertExtractAlignedPointerAsIndex,
                MemRefCastOpLowering,
                MemrefLoadOrStoreOpLowering<memref::LoadOp>,
                MemrefLoadOrStoreOpLowering<memref::StoreOp>>(
                  converter);
  // clang-format on
}
