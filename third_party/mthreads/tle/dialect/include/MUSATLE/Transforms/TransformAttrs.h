#ifndef MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H
#define MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::musa_tle {

// Internal attributes shared across independently built MThreads analysis,
// transform, and LLVM-lowering passes.  Keep their spellings in one place so
// producers and consumers cannot silently drift apart.
inline constexpr llvm::StringLiteral
    kMUSATLELocalPointerAsyncStoreAttr("musa_tle.local_ptr_async_store");
inline constexpr llvm::StringLiteral
    kMUSATLEStaticWarpSpecializeAttr("musa_tle.static_warp_specialize");
inline constexpr llvm::StringLiteral
    kMUSATLEConsumerEpochSyncAttr("musa_tle.consumer_epoch_sync");
inline constexpr llvm::StringLiteral
    kMUSATLEConsumerEpochRendezvousRequiredAttr(
        "musa_tle.consumer_epoch_rendezvous_required");
inline constexpr llvm::StringLiteral
    kMUSATLEExplicitCtaSyncAttr("musa_tle.explicit_cta_sync");
inline constexpr llvm::StringLiteral
    kMUSATLEPipeTMEReuseSyncAttr("musa_tle.pipe_tme_reuse_sync");
inline constexpr llvm::StringLiteral
    kMUSATLEPipePayloadCopyAttr("musa_tle.pipe_payload_copy");
inline constexpr llvm::StringLiteral
    kMUSATLEExplicitSqmmaAttr("musa_tle.explicit_sqmma");
inline constexpr llvm::StringLiteral kMUSATLEPipeLocalStoreWaitGroupsAttr(
    "musa_tle.pipe_local_store_wait_groups");
inline constexpr llvm::StringLiteral
    kMUSATLEPipeLocalStoreGroupAttr("musa_tle.pipe_local_store_group");
inline constexpr llvm::StringLiteral
    kMUSATLEPipeBarrierRingAttr("musa_tle.pipe_barrier_ring");
inline constexpr llvm::StringLiteral
    kMUSATLEPipeAsyncWaitAttr("musa_tle.pipe_async_wait");
inline constexpr llvm::StringLiteral
    kMUSATLELayoutConversionSyncAttr("musa_tle.layout_conversion_sync");
inline constexpr llvm::StringLiteral
    kMUSATLEPartitionMaterializationAttr("musa_tle.partition_materialization");
inline constexpr llvm::StringLiteral
    kMUSATLEBarrierGroupAttr("musa_tle.barrier_group");

} // namespace mlir::triton::musa_tle

#endif // MTHREADS_MUSATLE_TRANSFORMS_TRANSFORM_ATTRS_H
