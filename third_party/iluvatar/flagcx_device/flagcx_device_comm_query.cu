/*************************************************************************
 * Copyright 2025-     FlagOS Contributors
 *
 * Iluvatar CoreX device-side FlagCX adapter for TLE Distributed.
 *
 * This translation unit deliberately avoids the NVIDIA device binding in
 * bindings/ir/nvidia/. It uses the shared FlagCX default IPC layout and
 * compiles with the CoreX ivcore toolchain. The exported C ABI is kept equal
 * to the symbols emitted by FlagTree's TLE lowering.
 ************************************************************************/

// The CoreX clang does not define the CUDA compiler macro for -x ivcore, while
// FlagCX's device headers use it to select device-only declarations. The
// adapter supplies the small set of device macros it needs before including
// the shared default device layout.
//
// These definitions adapt the device-compilation interfaces declared in:
//   flagcx/adaptor/include/device_api/device_utils.h
//   flagcx/adaptor/include/device_api/platform_traits.h
//   flagcx/adaptor/include/device_api/flagcx_device_core.h
#define FLAGCX_ADAPTOR_DEVICE_UTILS_H_
#define FLAGCX_DEFAULT_PLATFORM_TRAITS_H_
#define FLAGCX_DEVICE_COMPILE 1
#define FLAGCX_CHECK_DEVICE_CC 1
#define FLAGCX_IR_EXTERN_C extern "C"
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#define FLAGCX_DEVICE_INLINE_DECORATOR __device__ __attribute__((always_inline))
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ __attribute__((always_inline))
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__ __constant__
// ivcore11 cannot select llvm.nvvm.membar.sys, so system-scope fences use the
// device-wide fence, which lowers to ml_lsa_wbinv and writes back/invalidates
// the LSA caches. That is the coherence level peer (P2P) accesses need here.
#define FLAGCX_DEVICE_THREAD_FENCE __threadfence
#define FLAGCX_DEVICE_SYNC_THREADS __syncthreads
#define FLAGCX_THREAD_IDX_X threadIdx.x
#define FLAGCX_BLOCK_IDX_X blockIdx.x
#define FLAGCX_BLOCK_DIM_X blockDim.x
#define FLAGCX_GRID_DIM_X gridDim.x
#define FLAGCX_DEVICE_STREAM_PTR void *
#define FLAGCX_SIMT_WIDTH 64
#define FLAGCX_SHARED __shared__
#define FLAGCX_MAYBE_UNUSED __attribute__((unused))
// FlagCX >= #555 tags by-value kernel pointers with FLAGCX_DEVICE_GLOBAL_PTR
// (defined in device_utils.h, which this adapter stubs out). Empty on CoreX:
// the qualifier is only meaningful for XPU address-space 1.
#ifndef FLAGCX_DEVICE_GLOBAL_PTR
#define FLAGCX_DEVICE_GLOBAL_PTR
#endif

#include <cuda_runtime.h>

#include <stdint.h>

#include "device_api/platform_traits.h"

//
// This is the CoreX device replacement for:
//   flagcx/adaptor/include/device_api/default_platform_traits.h
struct DefaultPlatform {};

template <> struct PlatformTraits<DefaultPlatform> {
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      return static_cast<int>(FLAGCX_THREAD_IDX_X & (FLAGCX_SIMT_WIDTH - 1));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t lanemaskLt() {
      const uint32_t lane =
          static_cast<uint32_t>(FLAGCX_THREAD_IDX_X & (FLAGCX_SIMT_WIDTH - 1));
      return lane == 0 ? 0ull : ((1ull << lane) - 1ull);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t activemask() {
      return static_cast<uint32_t>(__activemask());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(uint64_t mask = ~uint64_t{0}) {
      (void)mask;
      __syncwarp();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(uint64_t value) {
      return __popcll(value);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int, int) {
      __syncthreads();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iteration) {
      (void)iteration;
      asm volatile("");
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      __threadfence();
    }
  };

  struct Atomic {
    // Barrier flags are written by a peer device through the P2P mapping, and a
    // plain __atomic_load_n can be served from a cache line that never observes
    // those peer writes, so a spin-wait hangs even though the peer already
    // stored the epoch. The access pattern below matches the one IXCCL (the
    // vendor's NCCL port) uses for exactly this job, see ixccl
    // src/device/op128.h and prims_simple.h under __IVCORE_ARCH__:
    //   * a system-scope load is a plain volatile load with no fence, so the
    //     spin loop stays cheap
    //   * a system-scope store fences first, then stores volatile, which is
    //     what publishes the producer's prior writes before the flag
    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    load(T *ptr, flagcxDeviceMemoryOrder_t order) {
      switch (order) {
      case flagcxDeviceMemoryOrderRelaxed:
        return __atomic_load_n(ptr, __ATOMIC_RELAXED);
      case flagcxDeviceMemoryOrderRelease:
        return __atomic_load_n(ptr, __ATOMIC_RELAXED);
      case flagcxDeviceMemoryOrderAcquire:
      case flagcxDeviceMemoryOrderAcqRel:
      default:
        return *const_cast<volatile T *>(ptr);
      }
    }

    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    store(T *ptr, T value, flagcxDeviceMemoryOrder_t order) {
      switch (order) {
      case flagcxDeviceMemoryOrderRelaxed:
        __atomic_store_n(ptr, value, __ATOMIC_RELAXED);
        return;
      case flagcxDeviceMemoryOrderRelease:
      case flagcxDeviceMemoryOrderAcquire:
      case flagcxDeviceMemoryOrderAcqRel:
      default:
        FLAGCX_DEVICE_THREAD_FENCE();
        *const_cast<volatile T *>(ptr) = value;
        return;
      }
    }

    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAdd(T *ptr, T value, flagcxDeviceMemoryOrder_t order) {
      (void)order;
      return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    compareExchange(T *ptr, T &expected, T desired,
                    flagcxDeviceMemoryOrder_t order) {
      (void)order;
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
  };

  struct CoopThread {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return 1; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {}
  };

  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return static_cast<int>(FLAGCX_BLOCK_DIM_X);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { __syncthreads(); }
  };

  template <int N> struct CoopTile {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= Intrin::simtWidth,
                  "N must be a power of two and no greater than simtWidth");

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::lane() % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t laneMask() const {
      const uint32_t lane =
          static_cast<uint32_t>(FLAGCX_THREAD_IDX_X & (FLAGCX_SIMT_WIDTH - 1));
      const uint32_t base = lane & ~(static_cast<uint32_t>(N) - 1u);
      return (~uint64_t{0} >> (Intrin::simtWidth - N)) << base;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { __syncwarp(); }
  };

  struct CoopTileSpan {
    int first;
    int count;
    int id;
    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int first_, int count_, int id_)
        : first(first_), count(count_), id(id_) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X) - first;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return count; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { __syncthreads(); }
  };

  struct CoopLanes {
    uint64_t mask;
    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopLanes(uint64_t mask_)
        : mask(mask_) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      const uint32_t lane =
          static_cast<uint32_t>(FLAGCX_THREAD_IDX_X & (FLAGCX_SIMT_WIDTH - 1));
      return __popcll(mask & (lane == 0 ? 0ull : ((1ull << lane) - 1ull)));
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return __popcll(mask); }
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t getLmask() const { return mask; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { __syncwarp(); }
  };

  using CoopWarp = CoopTile<FLAGCX_SIMT_WIDTH>;
  using CoopAny = PlatformCoop;
};

#include "device_api/flagcx_device_core.h"
#include "device_api/flagcx_device_enums.h"

namespace {

enum class IntraBarrierAction { Arrive, Wait, Sync };

// Corresponds to flagcxMakeCoopFromKind()/flagcxMakeCoopFromKindEx() in:
//   flagcx/bindings/ir/flagcx_device_scalar_ir_impl.h
//
// flagcxCoopAny erases the coop type behind a vtable, so every threadRank()/
// size()/sync() becomes an indirect call through a function-pointer table in
// __constant__ memory. The CoreX device-link path cannot resolve those
// relocations when the bitcode is linked into a Triton kernel, and the barrier
// hangs. The coop kind is a compile-time constant at every TLE call site, so
// the adapter dispatches it statically and instantiates the barrier on a
// concrete coop type instead. This keeps the exported C ABI unchanged.
template <typename Coop>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxIntraBarrierOn(const flagcxDevComm &comm, uint32_t index, bool multimem,
                     flagcxDeviceMemoryOrder_t order,
                     IntraBarrierAction action) {
  flagcxTeam team = flagcxTeamIntra(comm);
  flagcxDevBarrier<flagcxTeamTagIntra, Coop> bar(Coop(), comm, team, index,
                                                 multimem);
  switch (action) {
  case IntraBarrierAction::Arrive:
    bar.arrive(order);
    return;
  case IntraBarrierAction::Wait:
    bar.wait(order);
    return;
  case IntraBarrierAction::Sync:
    bar.sync(order);
    return;
  }
}

static FLAGCX_DEVICE_INLINE_DECORATOR void flagcxIntraBarrierDispatch(
    const void *commOpaque, flagcxCoopKind_t coopKind, uint32_t index,
    bool multimem, flagcxDeviceMemoryOrder_t order, IntraBarrierAction action) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  switch (coopKind) {
  case FLAGCX_COOP_BLOCK:
    flagcxIntraBarrierOn<flagcxCoopBlock>(*comm, index, multimem, order,
                                          action);
    return;
  case FLAGCX_COOP_WARP:
    flagcxIntraBarrierOn<flagcxCoopWarp>(*comm, index, multimem, order, action);
    return;
  case FLAGCX_COOP_THREAD:
    flagcxIntraBarrierOn<flagcxCoopThread>(*comm, index, multimem, order,
                                           action);
    return;
  default: // fail-safe: no-op sync
    flagcxIntraBarrierOn<flagcxCoopThread>(*comm, index, multimem, order,
                                           action);
    return;
  }
}

} // namespace

#define FLAGCX_GLOBAL_AS __attribute__((address_space(1)))
using flagcxGlobalConstPtr = const void FLAGCX_GLOBAL_AS *;
using flagcxGlobalPtr = void FLAGCX_GLOBAL_AS *;

// Corresponds to Category 1 (Comm Queries) in:
//   flagcx/bindings/ir/flagcx_device_wrapper_impl.h
extern "C" __device__ __attribute__((noinline)) int
flagcxDevCommGetRank(flagcxGlobalConstPtr commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getRank();
}

extern "C" __device__ __attribute__((noinline)) int
flagcxDevCommGetSize(flagcxGlobalConstPtr commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getSize();
}

extern "C" __device__ __attribute__((noinline)) int
flagcxDevCommGetIntraRank(flagcxGlobalConstPtr commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getIntraRank();
}

extern "C" __device__ __attribute__((noinline)) int
flagcxDevCommGetIntraSize(flagcxGlobalConstPtr commOpaque) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  return comm->getIntraSize();
}

// Corresponds to Category 6 (Scalar Barrier — Intra) in:
//   flagcx/bindings/ir/flagcx_device_scalar_ir_impl.h
extern "C" __device__ __attribute__((noinline)) void
flagcxIntraBarrierArriveS(flagcxGlobalConstPtr commOpaque,
                          flagcxCoopKind_t coopKind, uint32_t index,
                          bool multimem, flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierDispatch((const void *)commOpaque, coopKind, index,
                             multimem, order, IntraBarrierAction::Arrive);
}

extern "C" __device__ __attribute__((noinline)) void
flagcxIntraBarrierWaitS(flagcxGlobalConstPtr commOpaque,
                        flagcxCoopKind_t coopKind, uint32_t index,
                        bool multimem, flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierDispatch((const void *)commOpaque, coopKind, index,
                             multimem, order, IntraBarrierAction::Wait);
}

extern "C" __device__ __attribute__((noinline)) void
flagcxIntraBarrierSyncS(flagcxGlobalConstPtr commOpaque,
                        flagcxCoopKind_t coopKind, uint32_t index,
                        bool multimem, flagcxDeviceMemoryOrder_t order) {
  flagcxIntraBarrierDispatch((const void *)commOpaque, coopKind, index,
                             multimem, order, IntraBarrierAction::Sync);
}

// Corresponds to Category 4 (Pointer Access), flagcxGetIntraPointerC(), in:
//   flagcx/bindings/ir/flagcx_device_wrapper_impl.h
extern "C" __device__ __attribute__((noinline)) flagcxGlobalPtr
flagcxGetIntraPointerC(flagcxGlobalConstPtr memOpaque, size_t offset,
                       int peer) {
  const flagcxDevMem *mem = (const flagcxDevMem *)memOpaque;
  return (flagcxGlobalPtr)flagcxGetIntraPointer(*mem, offset, peer);
}
