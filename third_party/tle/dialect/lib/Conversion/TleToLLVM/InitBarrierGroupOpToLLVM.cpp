// MIT License
//
// Copyright (c) 2026 The FlagOS Contributors

#include "tle/dialect/include/Conversion/TleToLLVM/InitBarrierGroupOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>

namespace {

using namespace mlir;
namespace tle = mlir::triton::tle;

constexpr unsigned kNVVMConstantAddressSpace = 4;
constexpr unsigned kNVVMSharedAddressSpace = 3;

Value createOffsetTable(tle::InitBarrierGroupOp op,
                        ArrayRef<int32_t> offsets,
                        ConversionPatternRewriter &rewriter) {
  std::string bytes;
  bytes.reserve(offsets.size() * sizeof(int32_t));
  for (int32_t offset : offsets) {
    uint32_t value = static_cast<uint32_t>(offset);
    for (unsigned byte = 0; byte < sizeof(value); ++byte)
      bytes.push_back(static_cast<char>((value >> (byte * 8)) & 0xff));
  }

  ModuleOp module = op->getParentOfType<ModuleOp>();
  unsigned tableNumber = 0;
  SmallString<48> symbolName;
  do {
    symbolName.clear();
    (Twine("__tle_barrier_offsets_") + Twine(tableNumber++))
        .toStringRef(symbolName);
  } while (module.lookupSymbol(symbolName));

  auto *context = rewriter.getContext();
  Type i8Type = rewriter.getI8Type();
  auto tableType = LLVM::LLVMArrayType::get(i8Type, bytes.size());
  LLVM::GlobalOp table;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    table = LLVM::GlobalOp::create(
        rewriter, UnknownLoc::get(context), tableType, /*isConstant=*/true,
        LLVM::Linkage::Internal, symbolName,
        rewriter.getStringAttr(StringRef(bytes.data(), bytes.size())),
        /*alignment=*/4, kNVVMConstantAddressSpace);
  }

  auto tablePointerType =
      LLVM::LLVMPointerType::get(context, kNVVMConstantAddressSpace);
  return LLVM::AddressOfOp::create(rewriter, op.getLoc(), tablePointerType,
                                   table.getSymName());
}

struct InitBarrierGroupOpConversion
    : public ConvertOpToLLVMPattern<tle::InitBarrierGroupOp> {
  using ConvertOpToLLVMPattern<
      tle::InitBarrierGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(tle::InitBarrierGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Location loc = op.getLoc();
    auto *context = rewriter.getContext();
    TritonLLVMOpBuilder builder(loc, rewriter);

    std::map<int32_t, SmallVector<int32_t>> offsetsByCount;
    for (auto [offset, count] :
         llvm::zip_equal(op.getOffsets(), op.getCounts()))
      offsetsByCount[count].push_back(offset);

    SmallVector<int32_t> orderedOffsets;
    for (auto &[count, offsets] : offsetsByCount)
      orderedOffsets.append(offsets);
    Value tableBase = createOffsetTable(op, orderedOffsets, rewriter);

    auto function = op->getParentOfType<FunctionOpInterface>();
    if (!function)
      return op.emitOpError("cannot find parent function");
    Value sharedBase = LLVM::getStackPointer(rewriter, function);
    auto constantPointerType =
        LLVM::LLVMPointerType::get(context, kNVVMConstantAddressSpace);
    auto sharedPointerType =
        LLVM::LLVMPointerType::get(context, kNVVMSharedAddressSpace);
    Type i8Type = rewriter.getI8Type();
    Type i32Type = rewriter.getI32Type();

    Value threadId = NVVM::ThreadIdXOp::create(rewriter, loc, i32Type);
    Value workerCount = builder.i32_val(op.getWorkerCount());
    Value isWorker = builder.icmp_ult(threadId, workerCount);
    Value zero = builder.i32_val(0);

    int32_t tableGroupBase = 0;
    for (auto &[participantCount, offsets] : offsetsByCount) {
      int32_t groupOffset = 0;
      while (groupOffset < static_cast<int32_t>(offsets.size())) {
        int32_t chunkSize = std::min(
            static_cast<int32_t>(op.getWorkerCount()),
            static_cast<int32_t>(offsets.size()) - groupOffset);
        Value isInChunk =
            builder.icmp_ult(threadId, builder.i32_val(chunkSize));
        Value active = builder.and_(isWorker, isInChunk);
        Value safeThreadId = builder.select(active, threadId, zero);
        Value tableIndex = builder.add(
            safeThreadId, builder.i32_val(tableGroupBase + groupOffset));
        Value byteOffset = builder.mul(tableIndex, builder.i32_val(4));
        Value offsetPointer = builder.gep(constantPointerType, i8Type,
                                          tableBase, byteOffset);
        Value sharedOffset = LLVM::LoadOp::create(
            rewriter, loc, i32Type, offsetPointer, /*alignment=*/4,
            /*isVolatile=*/false, /*isNonTemporal=*/false);
        Value barrierPointer = builder.gep(sharedPointerType, i8Type,
                                           sharedBase, sharedOffset);

        ::mlir::triton::PTXBuilder ptxBuilder;
        std::string ptx = "@$0 mbarrier.init.shared::cta.b64 [$1], " +
                          std::to_string(participantCount) + ";";
        auto &init = *ptxBuilder.create(ptx);
        init({ptxBuilder.newOperand(active, "b"),
              ptxBuilder.newOperand(barrierPointer, "r")},
             /*onlyAttachMLIRArgs=*/true);
        ptxBuilder.launch(rewriter, loc, void_ty(context));
        groupOffset += chunkSize;
      }
      tableGroupBase += static_cast<int32_t>(offsets.size());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void tle::populateInitBarrierGroupOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<InitBarrierGroupOpConversion>(typeConverter, benefit);
}
