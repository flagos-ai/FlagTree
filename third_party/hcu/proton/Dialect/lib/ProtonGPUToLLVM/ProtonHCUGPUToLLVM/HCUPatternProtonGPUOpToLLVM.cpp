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

#include "Conversion/ProtonGPUToLLVM/ProtonHCUGPUToLLVM/HCUPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonHCUGPUToLLVM/TargetInfo.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

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
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto dataPack =
        lowerCircularStoreOpHelper(op, adaptor.getSegment(), rewriter);

    uint32_t addrSpace = dataPack.addrSpace;
    if (addrSpace == 1) {
      // TODO(crobeck): see what buffer ops performance looks like here for
      // global mem (address space 1) compared to predicated ops to shared
      // memory
      llvm::report_fatal_error("unimplemented");
    } else if (addrSpace == 3) {
      targetInfo.getTritonTargetInfo().storeDShared(
          rewriter, loc, dataPack.ptr, std::nullopt, dataPack.record,
          dataPack.isWriter);
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

namespace mlir::triton::proton::gpu::HCU {
void populateProtonGPUOpHCUPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::HCU
