#include "Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::Intellifusion {

using namespace mlir::triton::nvidia_gpu;

class GPNPUAssignDescriptorMemoryLayouts
    : public ttg::AssignDescriptorMemoryLayouts {
public:
  GPNPUAssignDescriptorMemoryLayouts() = default;

private:
  Attribute buildFallbackSharedEncoding(mlir::MLIRContext *ctx,
                                        ArrayRef<int64_t> shape,
                                        ArrayRef<unsigned> order,
                                        ttg::CGAEncodingAttr cgaLayout,
                                        Type elementType) override;
  Attribute getCompatibleSharedEncoding(Attribute enc, ArrayRef<int64_t> shape,
                                        Type elementType) override;
  bool isCompatibleSharedEncoding(Attribute enc) override;
};

bool GPNPUAssignDescriptorMemoryLayouts::isCompatibleSharedEncoding(
    Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc))
    return !nvmma.getTransposed();
  return false;
}

Attribute GPNPUAssignDescriptorMemoryLayouts::getCompatibleSharedEncoding(
    Attribute enc, ArrayRef<int64_t> shape, Type elementType) {
  if (isCompatibleSharedEncoding(enc))
    return enc;

  auto sharedLinear = dyn_cast<ttg::SharedLinearEncodingAttr>(enc);
  if (!sharedLinear)
    return {};

  auto *ctx = enc.getContext();
  auto cgaLayout = ttg::getCGALayout(sharedLinear);
  auto order = ttg::getOrder(sharedLinear, shape);
  auto sharedLinearLayout = ttg::toLinearLayout(shape, sharedLinear);
  auto isEquivalent = [&](ttg::NVMMASharedEncodingAttr candidate) {
    auto candidateLayout = ttg::nvmmaSharedToLinearLayout(
        shape, candidate, ttg::TMAMode::Tiled, /*disableSwizzle=*/false,
        /*emitErrors=*/false);
    return succeeded(candidateLayout) && *candidateLayout == sharedLinearLayout;
  };

  SmallVector<ttg::NVMMASharedEncodingAttr> preferredCandidates;
  // TMA descriptors only support non-transposed layouts. Preserve Triton's
  // default shape/order-based choice when it already matches this
  // shared_linear layout. The full candidate scan below is only a fallback for
  // equivalent non-transposed layouts not selected by the heuristic builder.
  for (bool fp4Padded : {false, true}) {
    auto preferred = ttg::NVMMASharedEncodingAttr::get(
        ctx, shape, order, cgaLayout, elementType, fp4Padded);
    preferredCandidates.push_back(preferred);
    if (isEquivalent(preferred))
      return preferred;
  }

  unsigned elementBitWidth = std::max(8u, elementType.getIntOrFloatBitWidth());
  for (bool fp4Padded : {false, true}) {
    for (unsigned swizzle : {0u, 32u, 64u, 128u}) {
      auto candidate = ttg::NVMMASharedEncodingAttr::get(
          ctx, swizzle, /*transposed=*/false, elementBitWidth, fp4Padded,
          cgaLayout);
      if (llvm::is_contained(preferredCandidates, candidate))
        continue;
      if (isEquivalent(candidate))
        return candidate;
    }
  }

  return {};
}

// Build fallback encoding given shape, order, cga layout and element type
Attribute GPNPUAssignDescriptorMemoryLayouts::buildFallbackSharedEncoding(
    mlir::MLIRContext *ctx, ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
    ttg::CGAEncodingAttr cgaLayout, Type elementType) {
  return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, cgaLayout,
                                           elementType, /*fp4Padded*/ false);
}

class GPNPUOptimizeDescriptorEncodingPass
    : public PassWrapper<GPNPUOptimizeDescriptorEncodingPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      GPNPUOptimizeDescriptorEncodingPass)

  StringRef getArgument() const final {
    return "gpnpu-optimize-descriptor-encoding";
  }
  StringRef getDescription() const final {
    return "set GPNPU-compatible encodings on tensor descriptor types";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                    nvidia_gpu::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    GPNPUAssignDescriptorMemoryLayouts assignMemoryLayouts;
    assignMemoryLayouts.assignMemoryLayouts(m);
  }
};

std::unique_ptr<Pass> createGPNPUOptimizeDescriptorEncodingPass() {
  return std::make_unique<GPNPUOptimizeDescriptorEncodingPass>();
}

} // namespace mlir::triton::Intellifusion
