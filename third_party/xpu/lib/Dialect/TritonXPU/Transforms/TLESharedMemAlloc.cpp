//===----------------------------------------------------------------------===//
// TLE Shared-Memory Alloc: assign a static byte offset to every cluster-shared
// (scope == smem) TLE buffer inside the cluster SM window.
//
// The TLE frontend (`tle.gpu.alloc(..., scope=tle.gpu.smem)`) stamps each
// `ttg.local_alloc` with the discardable attribute `xpu.mem_scope = "smem"`.
// Such a buffer is NOT a per-core LM allocation: all `core_num` cores share one
// copy that lives in the module-level `global_smem` window (addrspace 2). The
// GM->SM copy and the per-core `local_ptr` lowerings (LoadStoreOpToLLVM) each
// recompute their base pointer as `global_smem + sm_offset`, so they need a
// concrete byte offset for every smem buffer.
//
// A single lowering pattern cannot assign these offsets: patterns are stateless
// and see one op at a time, but several smem buffers (bias + gamma + beta in
// LayerNorm) must be packed side-by-side without overlapping, and without
// colliding with the reduce/scan scratch that grows UP from offset 0. This tiny
// module pass owns that global view.
//
// LAYOUT (mirrors LoopInvariantStaging's stage_sm convention):
//   * SM is 256KB, shared cluster-wide.
//   * reduce/scan scratch grows UP from offset 0; the lower
//   `kScratchReserveBytes`
//     (128KB) is reserved for it.
//   * TLE smem buffers are stacked DOWN from the top (256KB), each 64B-aligned,
//     in deterministic walk order. `running` is the current top; each buffer is
//     placed at `running - align64(bytes)`.
//   * assert that the final `running` stays >= kScratchReserveBytes, i.e. all
//     smem buffers together fit in the top 128KB.
//
// PRECONDITION -- co-existence with stage_sm: LoopInvariantStaging's stage_sm
// stacks its staging buffer DOWN from the same 256KB top, and neither allocator
// knows about the other, so if both ever ran on one kernel they would OVERLAP.
// Today they cannot: stage_sm only runs in compiler.py's non-TLE branch and is
// additionally env-gated on TRITONXPU_LOOP_INVARIANT_STAGING, while this pass
// only runs in the TLE branch. Anyone enabling both must first give them a
// shared bump allocator for the top-of-SM region.
//
// All sizes are compile-time constants (buffer widths are constexpr in TLE), so
// the offsets are static i32 attributes (`xpu.sm_offset`).

//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-tle-sm-alloc"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTLESHAREDMEMALLOC
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// SM window layout constants (must match LoopInvariantStaging.cpp:322-325 so
// the TLE smem buffers and the stage_sm staging buffer share the same top-of-SM
// region and neither steps on the reduce/scan scratch below).
constexpr int32_t kSMTotalBytes = 256 * 1024;
constexpr int32_t kScratchReserveBytes = 128 * 1024;

// Round up to the next 64-byte boundary (SM DMA alignment).
static int64_t align64(int64_t n) {
  return (n + 63) & ~static_cast<int64_t>(63);
}

} // namespace

struct TritonXPUTLESharedMemAllocPass
    : public impl::TritonXPUTLESharedMemAllocBase<
          TritonXPUTLESharedMemAllocPass> {
  using impl::TritonXPUTLESharedMemAllocBase<
      TritonXPUTLESharedMemAllocPass>::TritonXPUTLESharedMemAllocBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    // Stack smem buffers DOWN from the top of SM, 64B-aligned, in walk order.
    int64_t running = kSMTotalBytes;
    auto walkResult = mod.walk([&](triton::gpu::LocalAllocOp op) {
      auto scopeAttr = op->getAttrOfType<StringAttr>("xpu.mem_scope");
      if (!scopeAttr || scopeAttr.getValue() != "smem")
        return WalkResult::advance();

      auto memTy = dyn_cast<triton::gpu::MemDescType>(op.getType());
      if (!memTy) {
        op.emitError("smem-scoped local_alloc has non-MemDesc result type");
        return WalkResult::interrupt();
      }

      int64_t numElems = 1;
      for (int64_t d : memTy.getShape())
        numElems *= d;
      unsigned elemBits = memTy.getElementType().getIntOrFloatBitWidth();
      // Sub-byte element types would round to 0 bytes and make two buffers
      // share an offset. TLE has no sub-byte smem buffer today; reject rather
      // than silently alias.
      if (elemBits < 8 || elemBits % 8 != 0) {
        op.emitError(
            "smem-scoped local_alloc has a sub-byte / non-byte-aligned "
            "element type (")
            << elemBits << " bits)";
        return WalkResult::interrupt();
      }
      int64_t bufBytes = align64(numElems * (elemBits / 8u));
      running -= bufBytes;
      op->setAttr("xpu.sm_offset",
                  IntegerAttr::get(IntegerType::get(ctx, 32),
                                   static_cast<int32_t>(running)));
      LLVM_DEBUG(llvm::dbgs()
                 << "[TLESharedMemAlloc] smem buffer bytes=" << bufBytes
                 << " -> sm_offset=" << running << "\n");
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    // The TLE smem buffers must not intrude into the reduce/scan scratch that
    // grows UP from offset 0 (lower 128KB reserved). This also subsumes a plain
    // overflow check: `running` only ever decreases.
    if (running < kScratchReserveBytes) {
      mod.emitError("TLE smem buffers (")
          << (kSMTotalBytes - running) << " bytes) exceed the "
          << (kSMTotalBytes - kScratchReserveBytes)
          << "-byte staging window reserved above the reduce/scan scratch";
      signalPassFailure();
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
