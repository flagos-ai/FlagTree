#ifndef TRITON_HCU_EXPLICIT_WARP_SPECIALIZE_H
#define TRITON_HCU_EXPLICIT_WARP_SPECIALIZE_H

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace mlir::triton::HCU {
struct ExplicitWarpDomain {
  triton::FuncOp function;
  unsigned base, total;
  SmallVector<int32_t> starts;
};
// A direct terminal specialization has no parked-worker state loop or result
// exchange. Captures still require their independent uniformity/rematerializing
// proof. This is structural IR analysis, not a frontend protocol/certificate.
inline std::optional<ExplicitWarpDomain>
terminalWarpDomain(gpu::WarpSpecializeOp ws, unsigned maxWaves) {
  auto mod = ws->getParentOfType<ModuleOp>();
  auto base = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto fn = dyn_cast<triton::FuncOp>(ws->getParentOp());
  if (!base || base.getInt() < 1 || uint64_t(base.getInt()) > maxWaves ||
      !llvm::isPowerOf2_64(base.getInt()) || !fn ||
      !llvm::hasSingleElement(fn.getBody()) || ws.getNumResults() ||
      ws.getPartitionRegions().empty() ||
      gpu::TritonGPUDialect::getNumCTAs(mod) != 1 ||
      gpu::TritonGPUDialect::getThreadsPerWarp(mod) != 64)
    return std::nullopt;
  unsigned count = 0;
  mod.walk([&](gpu::WarpSpecializeOp) { ++count; });
  if (count != 1) return std::nullopt;
  for (Operation &op : fn.getBody().front()) {
    if (&op == ws.getOperation() ||
        isa<gpu::LocalAllocOp, gpu::LocalDeallocOp, triton::ReturnOp>(&op) ||
        isPure(&op)) continue;
    return std::nullopt;
  }
  ExplicitWarpDomain domain{fn, unsigned(base.getInt()), unsigned(base.getInt()), {}};
  for (int32_t waves : ws.getPartitionNumWarps()) {
    if (waves < 1 || !llvm::isPowerOf2_32(waves) ||
        unsigned(waves) > maxWaves - domain.total)
      return std::nullopt;
    domain.starts.push_back(domain.total);
    domain.total += waves;
  }
  return domain;
}
} // namespace mlir::triton::HCU
#endif
