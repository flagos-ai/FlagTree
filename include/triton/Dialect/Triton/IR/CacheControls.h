#ifndef TRITON_DIALECT_TRITON_IR_CACHE_CONTROLS_H
#define TRITON_DIALECT_TRITON_IR_CACHE_CONTROLS_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

inline CacheModifier getCacheModifier(Operation *op) {
  auto attr = op->getAttrOfType<CacheModifierAttr>("cache");
  return attr ? attr.getValue() : CacheModifier::NONE;
}

inline EvictionPolicy getEvictionPolicy(Operation *op) {
  auto attr = op->getAttrOfType<EvictionPolicyAttr>("evict");
  return attr ? attr.getValue() : EvictionPolicy::NORMAL;
}

inline bool isMusaLSUCache(CacheModifier cache) {
  return cache >= CacheModifier::MUSA_I0_O0_L0 &&
         cache <= CacheModifier::MUSA_I5_O5_L1;
}

inline bool isPPUAIUCache(CacheModifier cache) {
  return cache >= CacheModifier::PPU_AIU_KP1 &&
         cache <= CacheModifier::PPU_AIU_KP6;
}

inline bool isL1Eviction(EvictionPolicy evict) {
  return evict >= EvictionPolicy::L1_EVICT_NORMAL &&
         evict <= EvictionPolicy::L1_NO_ALLOCATE;
}

inline bool isL2Eviction(EvictionPolicy evict) {
  return evict >= EvictionPolicy::L2_EVICT_NORMAL &&
         evict <= EvictionPolicy::L2_EVICT_UNCHANGED;
}

inline bool hasExplicitCacheControl(Operation *op) {
  return getCacheModifier(op) != CacheModifier::NONE ||
         getEvictionPolicy(op) != EvictionPolicy::NORMAL;
}

// PTX versions before 9.4 have a ptxas cache-hint operand issue
// (upstream #11660/#11848). Reuse the version selected by the NVIDIA backend.
inline constexpr int kMinPtxVersionForCpAsyncL2Hint = 94;
inline bool usesNvidiaL2Policy(EvictionPolicy evict) {
  return isL2Eviction(evict) || evict == EvictionPolicy::EVICT_FIRST ||
         evict == EvictionPolicy::EVICT_LAST;
}

// Hints do not constrain transport selection. Match upstream NVIDIA lowering:
// only explicit CA overrides the preferred CG on a 16-byte cp.async.
inline CacheModifier getNvidiaAsyncCacheModifier(CacheModifier cache,
                                                 unsigned copyBytes) {
  return copyBytes == 16 && cache != CacheModifier::CA ? CacheModifier::CG
                                                       : CacheModifier::CA;
}

inline EvictionPolicy getNvidiaAsyncEvictionPolicy(EvictionPolicy evict,
                                                   int ptxVersion) {
  return usesNvidiaL2Policy(evict) &&
                 ptxVersion >= kMinPtxVersionForCpAsyncL2Hint
             ? evict
             : EvictionPolicy::NORMAL;
}

// Report adjustments where a selected instruction consumes the formal cache /
// evict fields. No extra IR attributes, propagation pass or fallback is needed.
inline void remarkCacheControlConversion(Operation *op, StringRef path,
                                         CacheModifier cache,
                                         EvictionPolicy evict,
                                         StringRef reason) {
  if (!hasExplicitCacheControl(op) ||
      (getCacheModifier(op) == cache && getEvictionPolicy(op) == evict))
    return;
  op->emitRemark() << "cache hint on " << path << ": (cache="
                   << stringifyCacheModifier(getCacheModifier(op)) << ", evict="
                   << stringifyEvictionPolicy(getEvictionPolicy(op))
                   << ") -> (cache=" << stringifyCacheModifier(cache)
                   << ", evict=" << stringifyEvictionPolicy(evict) << "); "
                   << reason;
}

} // namespace mlir::triton
#endif
