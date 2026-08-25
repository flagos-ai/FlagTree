#include "mlir/Analysis/Liveness.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <map>
#include <string>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUGLOBALSCRATCHALLOCATIONPASS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

static int32_t roundUp(int32_t val, int32_t step) {
  auto t = val + step - 1;
  return t - (t % step);
}

static uint32_t getGridBarrierScratchBytes(Operation *op) {
  if (op->getName().getStringRef() != "tle.distributed_barrier") {
    return 0;
  }

  auto kind = op->getAttrOfType<StringAttr>("group_kind");
  if (!kind)
    return 0;
  if (kind.getValue() == "grid")
    return 4;
  if (kind.getValue() != "grid_axis_group")
    return 0;

  auto domainShape =
      op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape");
  auto axes = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  assert(domainShape && axes &&
         "verified grid_axis_group metadata is required");
  uint64_t groupCount = 1;
  for (auto [axis, dim] : llvm::enumerate(domainShape.asArrayRef())) {
    if (llvm::is_contained(axes.asArrayRef(), static_cast<int32_t>(axis)))
      continue;
    groupCount *= static_cast<uint32_t>(dim);
  }
  assert(groupCount <= std::numeric_limits<uint32_t>::max() / 4 &&
         "grid_axis_group scratch size must be verified");
  return static_cast<uint32_t>(groupCount * 4);
}

#ifdef __TLE__
namespace {

constexpr llvm::StringLiteral kGridBarrierScratchOnlyAttr =
    "tle.grid_barrier_scratch_only";

struct GridBarrierScratchLayout {
  std::map<std::string, int32_t> offsets;
  int32_t size = 0;
};

static std::string getGridBarrierScratchKey(Operation *op) {
  auto kind = op->getAttrOfType<StringAttr>("group_kind");
  assert(kind && "grid barrier scratch requires verified group metadata");
  if (kind.getValue() == "grid")
    return "grid";

  auto domainShape =
      op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape");
  auto axes = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  assert(kind.getValue() == "grid_axis_group" && domainShape && axes &&
         "grid axis barrier scratch requires verified group metadata");

  std::string key;
  llvm::raw_string_ostream os(key);
  os << "grid_axis_group:domain=";
  llvm::interleaveComma(domainShape.asArrayRef(), os);
  os << ":axes=";
  llvm::interleaveComma(axes.asArrayRef(), os);
  os.flush();
  return key;
}

static GridBarrierScratchLayout buildGridBarrierScratchLayout(ModuleOp mod) {
  GridBarrierScratchLayout layout;
  mod->walk([&](Operation *op) {
    uint32_t nbytes = getGridBarrierScratchBytes(op);
    if (nbytes == 0)
      return;

    std::string key = getGridBarrierScratchKey(op);
    if (layout.offsets.find(key) != layout.offsets.end())
      return;

    layout.size = roundUp(layout.size, 4);
    layout.offsets.emplace(std::move(key), layout.size);
    layout.size += static_cast<int32_t>(nbytes);
  });
  layout.size = roundUp(layout.size, 4);
  return layout;
}

static bool usesOnlyReusableGridBarrierScratch(Operation *op) {
  return op->hasAttr(kGridBarrierScratchOnlyAttr);
}

} // namespace
#endif

static void allocateGMem(Operation *parentOp,
                         llvm::SetVector<Operation *> &callStack
#ifdef __TLE__
                         , const GridBarrierScratchLayout &barrierLayout
#endif
                         ) {
  // Recursively visit any dependency functions
  parentOp->walk([&](triton::CallOp call) {
    auto callable = call.resolveCallable();
    if (!callable->hasAttr("ttg.global_scratch_memory_size")) {
      auto inserted = callStack.insert(parentOp);
      assert(inserted && "call cycle detected");
      allocateGMem(callable, callStack
#ifdef __TLE__
                   , barrierLayout
#endif
      );
      callStack.remove(parentOp);
    }
  });

  MLIRContext *ctx = parentOp->getContext();
  OpBuilder builder(ctx);
#ifdef __TLE__
  bool hasReusableGridBarrierScratch = false;
  parentOp->walk([&](Operation *op) {
    if (getGridBarrierScratchBytes(op) != 0) {
      hasReusableGridBarrierScratch = true;
      return;
    }
    if (auto call = dyn_cast<triton::CallOp>(op)) {
      if (usesOnlyReusableGridBarrierScratch(call.resolveCallable()))
        hasReusableGridBarrierScratch = true;
    }
  });
  int32_t offset = hasReusableGridBarrierScratch ? barrierLayout.size : 0;
  bool hasNonReusableScratch = false;
#else
  int32_t offset = 0;
#endif
  uint32_t largestAlignment = 1;
#ifdef __TLE__
  if (hasReusableGridBarrierScratch)
    largestAlignment = 4;
#endif

  // General scratch allocations still use conservative bump allocation. Grid
  // barriers are reusable phase counters, so collectives with the same
  // participant signature share one canonical slot. A synchronous device call
  // whose scratch consists only of those counters can use the same frame.
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    uint32_t nbytes = 0;
    uint32_t align = 0;
    if (auto alloc = dyn_cast<triton::gpu::GlobalScratchAllocOp>(op)) {
      nbytes = alloc.getNbytes();
      align = alloc.getAlignment();
    } else if ((nbytes = getGridBarrierScratchBytes(op)) != 0) {
#ifdef __TLE__
      auto it = barrierLayout.offsets.find(getGridBarrierScratchKey(op));
      assert(it != barrierLayout.offsets.end());
      op->setAttr("ttg.global_scratch_memory_offset",
                  builder.getI32IntegerAttr(it->second));
      return;
#else
      align = 4;
#endif
    } else if (auto callOp = dyn_cast<triton::CallOp>(op)) {
      auto callable = callOp.resolveCallable();
      auto nbytes_attr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_size");
      auto align_attr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_alignment");
      assert(nbytes_attr);
      assert(align_attr);

      nbytes = nbytes_attr.getValue().getZExtValue();
      align = align_attr.getValue().getZExtValue();
#ifdef __TLE__
      if (nbytes > 0 && usesOnlyReusableGridBarrierScratch(callable)) {
        op->setAttr("ttg.global_scratch_memory_offset",
                    builder.getI32IntegerAttr(0));
        return;
      }
#endif
    }
    if (nbytes > 0) {
#ifdef __TLE__
      hasNonReusableScratch = true;
#endif
      offset = roundUp(offset, align);
      op->setAttr("ttg.global_scratch_memory_offset",
                  builder.getI32IntegerAttr(offset));
      offset += nbytes;
      largestAlignment = std::max(largestAlignment, align);
    }
  });
  int32_t totalMemorySize = roundUp(offset, largestAlignment);
  parentOp->setAttr("ttg.global_scratch_memory_size",
                    builder.getI32IntegerAttr(totalMemorySize));
  parentOp->setAttr("ttg.global_scratch_memory_alignment",
                    builder.getI32IntegerAttr(largestAlignment));
#ifdef __TLE__
  parentOp->removeAttr(kGridBarrierScratchOnlyAttr);
  if (hasReusableGridBarrierScratch && !hasNonReusableScratch)
    parentOp->setAttr(kGridBarrierScratchOnlyAttr, builder.getUnitAttr());
#endif
}

namespace {
class TritonGPUGlobalScratchAllocationPass
    : public mlir::triton::gpu::impl::TritonGPUGlobalScratchAllocationPassBase<
          TritonGPUGlobalScratchAllocationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool seenKernel = false;

    SetVector<Operation *> callStack;
#ifdef __TLE__
    GridBarrierScratchLayout barrierLayout =
        buildGridBarrierScratchLayout(mod);
#endif
    mod->walk([&](triton::FuncOp func) {
      allocateGMem(func, callStack
#ifdef __TLE__
                   , barrierLayout
#endif
      );

      if (func.getVisibility() == SymbolTable::Visibility::Public) {
        assert(!seenKernel);
        seenKernel = true;
        auto size =
            func->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_size");
        auto align = func->getAttrOfType<IntegerAttr>(
            "ttg.global_scratch_memory_alignment");
        assert(size);
        assert(align);
        mod->setAttr("ttg.global_scratch_memory_size", size);
        mod->setAttr("ttg.global_scratch_memory_alignment", align);
      }
    });
    assert(seenKernel);
  }
};
} // namespace
