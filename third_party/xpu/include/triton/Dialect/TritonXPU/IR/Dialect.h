#ifndef TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_

// TritonXPUDialect
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h" // cf
#include "triton/Dialect/Triton/IR/Dialect.h"           // arith/scf/math/triton
#include "triton/Dialect/TritonGPU/IR/Dialect.h"        // SliceEncodingAttr

#include "triton/Dialect/TritonXPU/IR/Dialect.h.inc"
#include "triton/Dialect/TritonXPU/IR/TritonXPUEnums.h.inc" // TritonXPUDialect

// TritonXPUAttr
#include "mlir/IR/Attributes.h"
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrInterfaces.h.inc"
namespace mlir {
namespace triton {
namespace xpu {
// Bring CTAEncodingAttr into the xpu namespace so the tablegen-generated
// declaration `CTAEncodingAttr getCTALayout() const;` (inserted by the
// LayoutEncodingTrait interface) resolves correctly.
using ::mlir::triton::gpu::CTAEncodingAttr;
} // namespace xpu
} // namespace triton
} // namespace mlir
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.h.inc"

// TritonXPUOps
#define GET_OP_CLASSES
#include "triton/Dialect/TritonXPU/IR/Ops.h.inc"

// TritonXPUTypes
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/MathExtras.h"
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/Types.h.inc"

namespace mlir {
namespace triton {
namespace xpu {

unsigned getTotalElemsPerThread(Type eltTy);

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy);

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape, Type eltTy);

unsigned getGroupSize(Attribute layout);

// Return a blocked encoding where the shape is distributed contiguously amongst
// the threads, warps, CTAs with 1 element per threads.
triton::xpu::ClusterLayoutAttr
getDefaultClusterEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          uint32_t buffer_size, uint32_t core_num);

// Unencoded TLE tensors, in one place.
//
// A tensor facing a `tt.store` in a TLE kernel carries no encoding by design:
// `TLECoreTiling::encodeComputePtrs` leaves output pointers unencoded so the
// store verifier's value-type == ptr-type keeps holding. The lowering reads
// such a type as the default cluster encoding at buffer_size=128 / core_num=64,
// so an analysis asking a per-core question about one has to answer it the same
// way. Returns `type` untouched when it already has an encoding, or is not a
// ranked tensor.
Type withTLEDefaultEncoding(Type type);

// The encoded sibling of an unencoded TLE tensor, when the IR has one.
//
// The TLE type conversion pairs every unencoded type with a `convert_layout` to
// or from the encoded form the compute chain uses, so the sibling is one hop
// away.
// Prefer it over `withTLEDefaultEncoding` whenever a Value is in hand: the two
// agree only at the original shape, because the default encoding splits the
// last dim across cores while the TLE encoding splits dim 0, so dividing the
// last dim by a vector width inflates the default reading -- and
// `tle_vload`/`tle_vstore` are sized from that count. Falls back to
// `withTLEDefaultEncoding` when no sibling is found.
Type tleEncodedFacingType(Value value);

// Same lookup, null Type when the IR has no sibling. A caller that sizes a
// memory access must use this one and fail loudly instead: the fallback above
// agrees with the sibling only at the original shape, so guessing there would
// mis-size the access silently.
Type tleEncodedFacingTypeOrNull(Value value);

// The `tle_local_ptr` behind a TLE `tt.load`, looking through the identity
// `convert_layout` the TLE type conversion leaves on SM pointers. Null when the
// producer is something else, or when a cvt in the way actually relayouts.
triton::xpu::TLELocalPtrOp getTLELocalPtrThroughCvt(Value ptr);

// Whether a TLE buffer is the cluster-shared kind (`scope=tle.gpu.smem`, i.e.
// `xpu.mem_scope == "smem"` on the alloc). Reading one is supported; WRITING
// one is not, and must be refused before the rewrite: a tle_vstore there would
// have every core in the cluster write the same window with no ownership rule.
// The lowering asserts on it, which in a release build is silence rather than a
// stop.
bool tleBufferIsSmem(Value buffer);

// Whether a TLE buffer read is expressible as whole vector loads, and if so its
// uniform element addend in `off`. LM buffers are always admissible (the
// lowering ignores their index tensor); a `scope=smem` buffer is admissible
// only when its index is `[uniform +] tl.arange(0, n)`, the one form
// `tle_vload` can address. Every site that decides whether a TLE load is
// vectorizable has to ask this, or the analysis and the rewrite disagree.
//
// `allowSmem` false refuses every scope=smem read regardless of its index: that
// is the `tle-smem-vec` escape hatch, which the driver clears on the last rung
// of the TLE retune ladder so an unsliceable kernel gets the scalar SM read
// rather than no kernel at all. LM reads are unaffected.
bool tleSmemSliceOffset(triton::xpu::TLELocalPtrOp lp, Value &off,
                        bool allowSmem = true);

// Whether an SM `tle_local_ptr` load is a GATHER: a cluster-shared buffer read
// with a data-dependent index that is NOT a plain slice (the slice case goes
// to tle_vload). No index-arithmetic recognition -- ANY non-slice index is a
// gather, lowered to one SM vgather (`vgathers`) per result vector with the
// index as a per-lane offset. This is what lets the group-norm per-channel
// weight read (and any tl.gather over an SM buffer) vectorize instead of
// dropping the affine chain to scalar. bf16 stays on the scalar SM path.
bool tleSmemGather(triton::xpu::TLELocalPtrOp lp);

SmallVector<unsigned>
getCoresPerClusterWithUniqueData(Attribute layout,
                                 ArrayRef<int64_t> tensorShape);

SmallVector<unsigned>
getCoresPerGroupWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape);

SmallVector<unsigned> getUniqueContigPerCore(Attribute layout,
                                             ArrayRef<int64_t> shape);

//===----------------------------------------------------------------------===//
// TLE buffer memory spaces and budgets
//
// These are header-inline on purpose: both TritonXPUTransforms and
// TritonXPUToLLVM need them, and the latter does not link the former.
//===----------------------------------------------------------------------===//

/// XPU address spaces as seen by `!tt.ptr<T, N>` and by the XTDK backend.
constexpr unsigned kLocalMemAddrSpace = 0;  // per-core LM
constexpr unsigned kGlobalMemAddrSpace = 1; // GM
constexpr unsigned kSharedMemAddrSpace = 2; // cluster-shared SM

/// Cores in one XPU cluster. LM is hardware-replicated per core; SM is a single
/// block shared by all of them, which is why the two budgets differ by 64x in
/// units even though both are quoted per core.
constexpr unsigned kCoresPerCluster = 64;

/// Per-core LM budget: min(XTDK stack limit, 8KB physical).
constexpr unsigned kLocalMemPerCoreBytes = 8192;

/// SM is 256KB per cluster, with the top 64 bytes reserved for `.sh.amo`.
constexpr unsigned kSharedMemTotalBytes = 256 * 1024;
constexpr unsigned kSharedMemAmoReserveBytes = 64;
constexpr unsigned kSharedMemBudgetBytes =
    kSharedMemTotalBytes - kSharedMemAmoReserveBytes;

/// Byte offset assigned to an SM `ttg.local_alloc` inside the shared block.
constexpr llvm::StringLiteral kSharedMemOffsetAttrName = "xpu.smem_offset";

/// The `ClusterLayoutAttr` stamped on a TLE `ttg.local_alloc` describing how
/// the tile is cut across cores. Without it the legacy flat cut applies.
constexpr llvm::StringLiteral kTileLayoutAttrName = "xpu.tile_layout";

/// True for `#triton_xpu.smem`, i.e. the cluster-shared SM block.
inline bool isSharedMemSpace(Attribute memorySpace) {
  return isa_and_nonnull<SharedMemorySpaceAttr>(memorySpace);
}

/// True if `type` is a `ttg.memdesc` living in SM.
inline bool isSharedMemDesc(Type type) {
  auto memDesc = dyn_cast<triton::gpu::MemDescType>(type);
  return memDesc && isSharedMemSpace(memDesc.getMemorySpace());
}

/// LM (0) or SM (2) for a `ttg.memdesc`; `#ttg.shared_memory` keeps meaning LM.
inline unsigned getMemDescAddrSpace(Type type) {
  return isSharedMemDesc(type) ? kSharedMemAddrSpace : kLocalMemAddrSpace;
}

/// True for the address spaces a TLE buffer can live in. Address space is not a
/// boolean here: LM = 0, GM = 1, SM = 2, so `!= 1` and `== 0` are both wrong.
inline bool isTLEBufferAddrSpace(unsigned addrSpace) {
  return addrSpace == kLocalMemAddrSpace || addrSpace == kSharedMemAddrSpace;
}

/// Elements of `shape` that land on a single core, matching
/// `XPUTLELocalAllocOpConversion` exactly -- any divergence silently
/// mis-budgets by up to 64x.
inline unsigned getPerCoreElems(Operation *allocOp, ArrayRef<int64_t> shape,
                                unsigned groupSize) {
  if (auto tileLayout =
          allocOp->getAttrOfType<ClusterLayoutAttr>(kTileLayoutAttrName)) {
    auto coresPerGroup = tileLayout.getCoresPerGroup();
    auto groupsPerCluster = tileLayout.getGroupsPerCluster();
    unsigned rank = shape.size();
    if (rank == 1 && groupsPerCluster[0] > 1)
      return tileLayout.getSizePerCore()[0]; // LargeN result buffer
    unsigned elemsPerCore = 1;
    for (unsigned d = 0; d < rank; ++d) {
      unsigned coresAlongDim = coresPerGroup[d] * groupsPerCluster[d];
      elemsPerCore *= (shape[d] + coresAlongDim - 1) / coresAlongDim;
    }
    return elemsPerCore;
  }
  unsigned tileElems = 1;
  for (auto dim : shape)
    tileElems *= dim;
  if (groupSize == 0)
    groupSize = 1;
  return (tileElems + groupSize - 1) / groupSize;
}

/// Bytes one core spends on `allocOp`, 64-byte aligned like the allocator.
inline unsigned getPerCoreBytes(Operation *allocOp, unsigned groupSize) {
  auto memDesc =
      cast<triton::gpu::MemDescType>(allocOp->getResult(0).getType());
  unsigned elems = getPerCoreElems(allocOp, memDesc.getShape(), groupSize);
  unsigned eltBytes =
      (memDesc.getElementType().getIntOrFloatBitWidth() + 7) / 8;
  return llvm::alignTo(elems * eltBytes, 64u);
}

/// Bytes `allocOp` occupies in the shared block. Unlike LM, an SM region is not
/// replicated: the whole cluster's worth of per-core slices lives in one block.
inline unsigned getSharedMemRegionBytes(Operation *allocOp,
                                        unsigned groupSize) {
  return getPerCoreBytes(allocOp, groupSize) * kCoresPerCluster;
}

} // namespace xpu
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONXPU_IR_DIALECT_H_
