#include "PatternPopulation.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

// Register this SPMD lowering for IFWA TMA global scratch handling.
static Value getNumPrograms(OpBuilder &rewriter, int numCTAs, Location loc,
                            ProgramIDDim axis) {
  Value physicalGridDim;
  switch (axis) {
  case ProgramIDDim::X:
    physicalGridDim = NVVM::GridDimXOp::create(rewriter, loc, i32_ty);
    break;
  case ProgramIDDim::Y:
    physicalGridDim = NVVM::GridDimYOp::create(rewriter, loc, i32_ty);
    break;
  case ProgramIDDim::Z:
    physicalGridDim = NVVM::GridDimZOp::create(rewriter, loc, i32_ty);
    break;
  }

  if (numCTAs == 1 || axis != ProgramIDDim::X)
    return physicalGridDim;

  auto builder = TritonLLVMOpBuilder(loc, rewriter);
  return builder.udiv(physicalGridDim, builder.i32_val(numCTAs));
}

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
        op->getParentOfType<ModuleOp>());

    rewriter.replaceOp(
        op, getNumPrograms(rewriter, numCTAs, op.getLoc(), op.getAxis()));
    return success();
  }
};

// ---- gpu::ThreadIdOp → GPNPU thread-id intrinsic ----------------------------

struct GpuThreadIdOpConversion
    : public ConvertOpToLLVMPattern<::mlir::gpu::ThreadIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(::mlir::gpu::ThreadIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto i32Ty = rewriter.getI32Type();

    StringRef intrinsicName;
    switch (op.getDimension()) {
    case ::mlir::gpu::Dimension::x:
      intrinsicName = "llvm.nvvm.read.ptx.sreg.tid.x";
      break;
    case ::mlir::gpu::Dimension::y:
      intrinsicName = "llvm.nvvm.read.ptx.sreg.tid.y";
      break;
    case ::mlir::gpu::Dimension::z:
      intrinsicName = "llvm.nvvm.read.ptx.sreg.tid.z";
      break;
    }

    Value result =
        LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsicName,
                                        TypeRange(i32Ty), ValueRange())
            .getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// ---- triton::gpu::WarpIdOp → tid / threadsPerWarp --------------------------

struct WarpIdOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::WarpIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::WarpIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto i32Ty = rewriter.getI32Type();

    if (triton::gpu::lookupNumWarps(op) == 1) {
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    // Read thread ID through the GPNPU intrinsic.
    Value tid = LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                                "llvm.nvvm.read.ptx.sreg.tid.x",
                                                TypeRange(i32Ty), ValueRange())
                    .getResult(0);

    // warpId = tid / threadsPerWarp
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpId = b.udiv(tid, b.i32_val(threadsPerWarp));

    rewriter.replaceOp(op, warpId);
    return success();
  }
};

} // namespace

void mlir::triton::Intellifusion::populateGPNPUSPMDOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<GpuThreadIdOpConversion>(typeConverter, benefit);
  patterns.add<WarpIdOpConversion>(typeConverter, benefit);
}
