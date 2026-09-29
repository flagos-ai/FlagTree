#include <stdint.h>

// Narrow UserHopper D8 boundary for the live TLE warp-specialized kernel.
//
// Exactly one thread in the 4-warp dispatch partition owns the descriptorless
// TMA1D state. Synchronization across those 128 dispatch threads stays in TLE
// (`tl.debug_barrier`); this helper must never use `__syncthreads()`, because
// the loader and math partitions do not call it.

static constexpr unsigned long long kEvictFirst = 0x12f0000000000000ull;
static constexpr unsigned long long kEvictNormal = 0x1000000000000000ull;

enum D8Tma1dOp : int {
  kInit = 0,
  kLoad = 1,
  kStore = 2,
  kFenceInit = 3,
};

__device__ __forceinline__ void d8_mbarrier_init(unsigned long long *barrier) {
  const unsigned int smem =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
               :
               : "r"(smem)
               : "memory");
}

__device__ __forceinline__ void
d8_mbarrier_arrive_expect_tx(unsigned long long *barrier, int num_bytes) {
  const unsigned int smem =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(smem), "r"(num_bytes)
               : "memory");
}

__device__ __forceinline__ void d8_mbarrier_wait(unsigned long long *barrier,
                                                 unsigned int &phase) {
  const unsigned int smem =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "{\n\t"
      ".reg .pred done;\n\t"
      "D8_LIVE_WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n\t"
      "@done bra D8_LIVE_DONE_%=;\n\t"
      "bra D8_LIVE_WAIT_%=;\n\t"
      "D8_LIVE_DONE_%=:\n\t"
      "}"
      :
      : "r"(smem), "r"(phase), "r"(0x989680)
      : "memory");
  phase ^= 1;
}

__device__ __forceinline__ void d8_tma_load_1d(void *smem_dst,
                                               const void *gmem_src,
                                               unsigned long long *barrier,
                                               int num_bytes) {
  const unsigned int dst =
      static_cast<unsigned int>(__cvta_generic_to_shared(smem_dst));
  const unsigned int bar =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;"
      :
      : "r"(dst), "l"(gmem_src), "r"(num_bytes), "r"(bar), "l"(kEvictFirst)
      : "memory");
}

__device__ __forceinline__ void
d8_tma_store_1d(void *gmem_dst, const void *smem_src, int num_bytes) {
  const unsigned int src =
      static_cast<unsigned int>(__cvta_generic_to_shared(smem_src));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
               "[%0], [%1], %2, %3;"
               :
               : "l"(gmem_dst), "r"(src), "r"(num_bytes), "l"(kEvictNormal)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

__device__ __forceinline__ void d8_tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

extern "C" __device__ void TleD8Tma1d(const uint64_t token_stage_addr,
                                      const uint64_t token_state_addr,
                                      const uint64_t global_token_addr,
                                      const int num_bytes, const int active,
                                      const int op) {
  // Both addresses come from TLE's dynamic-SMEM allocator and are converted to
  // generic addresses before crossing the raw ABI. Keeping all shared storage
  // in that allocator preserves the alignment of v23's existing TMA/WGMMA
  // buffers; a CUDA static-shared section would shift `extern shared`.
  auto *token_stage = reinterpret_cast<unsigned char *>(
      static_cast<uintptr_t>(token_stage_addr));
  auto *token_state = reinterpret_cast<unsigned long long *>(
      static_cast<uintptr_t>(token_state_addr));
  const unsigned int stage =
      static_cast<unsigned int>(__cvta_generic_to_shared(token_stage));
  const unsigned int barrier =
      static_cast<unsigned int>(__cvta_generic_to_shared(token_state));
  const unsigned int phase = barrier + sizeof(unsigned long long);

  // PR787 always-inlines raw CUDA regions and currently loses their LLVM
  // `convergent` attributes. Keep lane ownership opaque to LLVM: a C++ owner
  // early-return lets it fuse consecutive calls into one divergent CFG and
  // strand a dispatch-role barrier inside that divergence.
  if (op == kInit) {
    asm volatile("{\n\t"
                 ".reg .u32 tid, role_lane;\n\t"
                 ".reg .pred owner;\n\t"
                 "mov.u32 tid, %%tid.x;\n\t"
                 "and.b32 role_lane, tid, 127;\n\t"
                 "setp.eq.u32 owner, role_lane, 0;\n\t"
                 "@owner st.shared.u32 [%1], 0;\n\t"
                 "@owner mbarrier.init.shared::cta.b64 [%0], 1;\n\t"
                 "}"
                 :
                 : "r"(barrier), "r"(phase)
                 : "memory");
  } else if (op == kFenceInit) {
    // All four dispatch warps execute the init fence after the first role sync.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  } else if (op == kLoad) {
    // Owner/active/size are PTX predicates, not LLVM control flow.
    asm volatile("{\n\t"
                 ".reg .u32 tid, role_lane, low_bits;\n\t"
                 ".reg .pred owner, enabled, size_ok, aligned;\n\t"
                 "mov.u32 tid, %%tid.x;\n\t"
                 "and.b32 role_lane, tid, 127;\n\t"
                 "setp.eq.u32 owner, role_lane, 0;\n\t"
                 "setp.ne.s32 enabled, %5, 0;\n\t"
                 "and.pred owner, owner, enabled;\n\t"
                 "setp.gt.s32 size_ok, %2, 0;\n\t"
                 "and.pred owner, owner, size_ok;\n\t"
                 "setp.le.s32 size_ok, %2, 4096;\n\t"
                 "and.pred owner, owner, size_ok;\n\t"
                 "and.b32 low_bits, %2, 15;\n\t"
                 "setp.eq.u32 aligned, low_bits, 0;\n\t"
                 "and.pred owner, owner, aligned;\n\t"
                 "@owner cp.async.bulk.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1], %2, [%3], %4;\n\t"
                 "}"
                 :
                 : "r"(stage), "l"(global_token_addr), "r"(num_bytes),
                   "r"(barrier), "l"(kEvictFirst), "r"(active)
                 : "memory");
  } else if (op == kStore) {
    // The owner-only wait loop remains inside one opaque PTX block, preserving
    // a straight-line LLVM CFG between raw calls and role barriers.
    asm volatile(
        "{\n\t"
        ".reg .u32 tid, role_lane, low_bits, current_phase, next_phase;\n\t"
        ".reg .pred owner, enabled, size_ok, aligned, done;\n\t"
        "mov.u32 tid, %%tid.x;\n\t"
        "and.b32 role_lane, tid, 127;\n\t"
        "setp.eq.u32 owner, role_lane, 0;\n\t"
        "setp.ne.s32 enabled, %6, 0;\n\t"
        "and.pred owner, owner, enabled;\n\t"
        "setp.gt.s32 size_ok, %4, 0;\n\t"
        "and.pred owner, owner, size_ok;\n\t"
        "setp.le.s32 size_ok, %4, 4096;\n\t"
        "and.pred owner, owner, size_ok;\n\t"
        "and.b32 low_bits, %4, 15;\n\t"
        "setp.eq.u32 aligned, low_bits, 0;\n\t"
        "and.pred owner, owner, aligned;\n\t"
        "@!owner bra D8_LIVE_STORE_DONE_%=;\n\t"
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %4;\n\t"
        "ld.shared.u32 current_phase, [%1];\n\t"
        "D8_LIVE_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 "
        "done, [%0], current_phase, 0x989680;\n\t"
        "@done bra D8_LIVE_WAIT_DONE_%=;\n\t"
        "bra D8_LIVE_WAIT_%=;\n\t"
        "D8_LIVE_WAIT_DONE_%=:\n\t"
        "xor.b32 next_phase, current_phase, 1;\n\t"
        "st.shared.u32 [%1], next_phase;\n\t"
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
        "[%2], [%3], %4, %5;\n\t"
        "cp.async.bulk.commit_group;\n\t"
        "cp.async.bulk.wait_group 0;\n\t"
        "D8_LIVE_STORE_DONE_%=:\n\t"
        "}"
        :
        : "r"(barrier), "r"(phase), "l"(global_token_addr), "r"(stage),
          "r"(num_bytes), "l"(kEvictNormal), "r"(active)
        : "memory");
  }
}
