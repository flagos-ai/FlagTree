// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/NvidiaPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Circular strategy memory layout of profiled data (total: N bytes).
// Assuming we record data from warp 0, 2, 7 so buffer looks like:
//  +-----------------------------------------------+
//  | warp 0 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 2 data (N/3 bytes)                       |
//  +-----------------------------------------------+
//  | warp 7 data (N/3 bytes)                       |
//  +-----------------------------------------------+

struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dataPack =
        lowerCircularStoreOpHelper(op, adaptor.getSegment(), rewriter);

    uint32_t addrSpace = dataPack.addrSpace;
    if (addrSpace == 1) {
      auto mod = op.getOperation()->getParentOfType<ModuleOp>();
      int numWarps = proton::gpu::getTotalNumWarps(mod);
      PTXBuilder builder;
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      if (numWarps > 1) {
        auto stInst = builder.create<>("st")->o("global").o("cg").v(2).b(32);
        auto *ptrOpr = builder.newAddrOperand(dataPack.ptr, "l");

        PTXBuilder::Operand *valOpr;
        SmallVector<std::pair<Value, std::string>> vecVals;
        auto unPackedVals = unpackLLVector(loc, dataPack.record, rewriter);
        vecVals.push_back({unPackedVals[0], "r"});
        vecVals.push_back({unPackedVals[1], "r"});
        valOpr = builder.newListOperand(vecVals);
        stInst(ptrOpr, valOpr).predicate(dataPack.isWriter, "b");
        builder.launch(rewriter, loc, void_ty(rewriter.getContext()));
      } else {
        // Non-vectorized version for num_warps=1 to handle potential
        // misalignment
        auto stInst = builder.create<>("st")->o("global").o("cg").b(32);

        auto unPackedVals = unpackLLVector(loc, dataPack.record, rewriter);

        // First store: write first 32-bit value at base address
        auto *ptrOpr0 = builder.newAddrOperand(dataPack.ptr, "l", 0);
        auto *valOpr0 = builder.newOperand(unPackedVals[0], "r");
        stInst(ptrOpr0, valOpr0).predicate(dataPack.isWriter, "b");

        // Second store: write second 32-bit value at offset +4 bytes
        auto *ptrOpr1 = builder.newAddrOperand(dataPack.ptr, "l", 4);
        auto *valOpr1 = builder.newOperand(unPackedVals[1], "r");
        stInst(ptrOpr1, valOpr1).predicate(dataPack.isWriter, "b");

        builder.launch(rewriter, loc, void_ty(rewriter.getContext()));
      }
    } else if (addrSpace == 3) {
      targetInfo.getTritonTargetInfo().storeDShared(
          rewriter, loc, dataPack.ptr, std::nullopt, dataPack.record,
          /*pred=*/dataPack.isWriter);
    } else {
      llvm::report_fatal_error("unsupported address space in circular store");
    }
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::proton::gpu::NVIDIA {
void populateProtonGPUOpNvidiaPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfo &targetInfo,
                                       PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::NVIDIA
