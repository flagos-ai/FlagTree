// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Conversion/ProtonGPUToLLVM/PatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/AMDPatternProtonGPUOpToLLVM.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton::gpu {
#define GEN_PASS_DEF_CONVERTPROTONAMDGPUTOLLVM
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton::proton::gpu
} // namespace mlir

namespace {

class ProtonLLVMConversionTarget : public ConversionTarget {
public:
  explicit ProtonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addIllegalDialect<mlir::triton::proton::gpu::ProtonGPUDialect>();
    addIllegalDialect<mlir::triton::proton::ProtonDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertProtonAMDGPUToLLVM
    : public mlir::triton::proton::gpu::impl::ConvertProtonAMDGPUToLLVMBase<
          ConvertProtonAMDGPUToLLVM> {
  explicit ConvertProtonAMDGPUToLLVM(std::string arch) { this->arch = arch; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    auto tritonTargetInfo = mlir::triton::AMD::TargetInfo(arch);
    auto protonTargetInfo =
        mlir::triton::proton::gpu::AMD::TargetInfo(tritonTargetInfo, arch);
    mlir::LowerToLLVMOptions option(context);
    TritonGPUToLLVMTypeConverter typeConverter(context, option,
                                               tritonTargetInfo);
    populateTypeConversions(typeConverter, protonTargetInfo);
    mlir::triton::proton::gpu::populateProtonGPUOpPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::triton::proton::gpu::AMD::populateProtonGPUOpAMDPatterns(
        typeConverter, patterns, protonTargetInfo, 1);
    mlir::triton::AMD::populateMaskedOpsToLLVMPatterns(patterns,
                                                       tritonTargetInfo);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->arch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->arch);
      return signalPassFailure();
    }
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    auto convTarget = ProtonLLVMConversionTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonAMDGPUToLLVMPass(std::string arch) {
  return std::make_unique<ConvertProtonAMDGPUToLLVM>(arch);
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
