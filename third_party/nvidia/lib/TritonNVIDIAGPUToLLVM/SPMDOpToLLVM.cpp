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

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

static Value getNumPrograms(OpBuilder &rewriter, int numCTAs, Location loc,
                            ProgramIDDim axis) {
  if (numCTAs == 1) {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::GridDimXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::GridDimYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::GridDimZOp::create(rewriter, loc, i32_ty);
    }
  } else {
    switch (axis) {
    case ProgramIDDim::X:
      return NVVM::ClusterDimXOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Y:
      return NVVM::ClusterDimYOp::create(rewriter, loc, i32_ty);
    case ProgramIDDim::Z:
      return NVVM::ClusterDimZOp::create(rewriter, loc, i32_ty);
    }
  }
  llvm_unreachable("invalid axis");
}

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
        op->getParentOfType<ModuleOp>());

    rewriter.replaceOp(
        op, getNumPrograms(rewriter, numCTAs, op.getLoc(), op.getAxis()));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
