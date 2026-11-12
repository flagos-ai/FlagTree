#ifndef TRITON_HCU_EXPLICIT_WARP_SPECIALIZE_H
#define TRITON_HCU_EXPLICIT_WARP_SPECIALIZE_H

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace mlir::triton::HCU {
struct ExplicitWarpDomain {
  triton::FuncOp function;
  unsigned base, total;
  SmallVector<int32_t> starts;
};

// Captures must be reconstructible from the function ABI. Each IR stage
// supplies the operations it can rematerialize, while sharing the SSA proof.
class UniformWarpCaptures {
public:
  using SupportedOp = llvm::function_ref<bool(Operation *)>;
  using SupportedArg = llvm::function_ref<bool(Type)>;

  UniformWarpCaptures(Block &entry, SupportedOp supported,
                      SupportedArg supportedArg = nullptr)
      : entry(entry), supported(supported), supportedArg(supportedArg) {}

  bool operator()(Value value) {
    if (auto it = proven.find(value); it != proven.end())
      return it->second;
    bool uniform = false;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      uniform = arg.getOwner() == &entry &&
                (!supportedArg || supportedArg(arg.getType()));
    } else if (Operation *op = value.getDefiningOp()) {
      uniform = supported(op) && llvm::all_of(op->getOperands(), [&](Value input) {
        return (*this)(input);
      });
    }
    proven.try_emplace(value, uniform);
    return uniform;
  }

private:
  Block &entry;
  SupportedOp supported;
  SupportedArg supportedArg;
  llvm::DenseMap<Value, bool> proven;
};

inline bool hasUniformTritonWarpCaptures(gpu::WarpSpecializeOp ws,
                                        triton::FuncOp fn) {
  auto isScalar = [](Type type) {
    return isa<IntegerType, IndexType, FloatType, triton::PointerType>(type);
  };
  auto supported = [&](Operation *op) {
    if (auto alloc = dyn_cast<gpu::LocalAllocOp>(op))
      return !alloc.getSrc();
    if (isa<gpu::MemDescIndexOp, gpu::MemDescSubsliceOp,
            gpu::MemDescReinterpretOp>(op))
      return true;
    return llvm::all_of(op->getResultTypes(), isScalar) && isPure(op) &&
           (op->getName().getStringRef().starts_with("arith.") ||
            isa<triton::AddPtrOp>(op));
  };
  UniformWarpCaptures proof(fn.getBody().front(), supported, isScalar);
  return llvm::all_of(ws.getExplicitCaptures(), [&](Value capture) {
    return (isScalar(capture.getType()) ||
            isa<gpu::MemDescType>(capture.getType())) &&
           proof(capture);
  });
}
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
