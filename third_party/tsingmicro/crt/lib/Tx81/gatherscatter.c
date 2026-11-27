//===------------------------ gatherscatter.c -----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::GatherScatter see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81_run.h"
#include <stdio.h>
#include <stdlib.h>

static bool gatherscatter_spm_addr_oob(const Tx81MemAddrRange *range) {
  return range->min_addr < TX81_GS_SPM_ADDR_MIN ||
         range->max_addr > TX81_GS_SPM_ADDR_MAX;
}

#define dma_logging()                                                          \
  do {                                                                         \
    __EP_LOG__(3,                                                              \
               "[func @%s] gatherscatter -- src: %p, dst: %p, bytes: %u ("     \
               "src_strideN: %u, src_strideH: %u, src_strideW: %u, "           \
               "src_iterN: %u, src_iterH: %u, src_iterW: %u, "                 \
               "dst_strideN: %u, dst_strideH: %u, dst_strideW: %u, "           \
               "dst_iterN: %u, dst_iterH: %u, dst_iterW: %u)\n",               \
               kernel_name, src, dst, bytes, src_strideN, src_strideH,         \
               src_strideW, src_iterN, src_iterH, src_iterW, dst_strideN,      \
               dst_strideH, dst_strideW, dst_iterN, dst_iterH, dst_iterW);     \
  } while (false)

void __GatherScatter(uint64_t *src, uint64_t *dst, uint32_t bytes,
                     uint32_t src_strideN, uint32_t src_strideH,
                     uint32_t src_strideW, uint32_t src_iterN,
                     uint32_t src_iterH, uint32_t src_iterW,
                     uint32_t dst_strideN, uint32_t dst_strideH,
                     uint32_t dst_strideW, uint32_t dst_iterN,
                     uint32_t dst_iterH, uint32_t dst_iterW, uint32_t action,
                     const char *kernel_name) {
  INTRNISIC_RUN_SWITCH;

  if (is_dma_action_logging(action)) {
    dma_logging();
  }

  if (is_dma_action_checking(action)) {
    Tx81MemAddrRange src_range = compute_gatherscatter_src_addr_range(
        src, bytes, src_strideN, src_strideH, src_strideW, src_iterN, src_iterH,
        src_iterW);

    DMA_LOG_COND(
        dma_oob_log_count++,
        __EP_LOG__(3,
                   "[func @%s] gatherscatter src (%p), src_range_size %u, "
                   "valid spm range [%#x, %#x), dma_oob_log_count (%d)\n",
                   kernel_name, src,
                   (unsigned int)(src_range.max_addr - src_range.min_addr),
                   TX81_GS_SPM_ADDR_MIN, TX81_GS_SPM_ADDR_MAX,
                   dma_oob_log_count));

    if (gatherscatter_spm_addr_oob(&src_range)) {
      // DMA_LOG_COND(dma_oob_count, dma_logging());
      DMA_LOG_COND(
          dma_oob_count++,
          __EP_LOG__(3,
                     "[func @%s] fatal error: spm memory OOB, "
                     "gatherscatter src (%p), range [%#lx, %#lx), "
                     "valid [%#x, %#x), dma_oob_count (%d)\n",
                     kernel_name, src, (unsigned long)src_range.min_addr,
                     (unsigned long)src_range.max_addr, TX81_GS_SPM_ADDR_MIN,
                     TX81_GS_SPM_ADDR_MAX, dma_oob_count));
      if (get_dma_check_abort()) {
        abort();
      }
    }

    Tx81MemAddrRange dst_range = compute_gatherscatter_dst_addr_range(
        dst, bytes, dst_strideN, dst_strideH, dst_strideW, dst_iterN, dst_iterH,
        dst_iterW);

    DMA_LOG_COND(
        dma_oob_log_count++,
        __EP_LOG__(3,
                   "[func @%s] gatherscatter dst (%p), dst_range_size %u, "
                   "valid spm range [%#x, %#x), dma_oob_log_count (%d)\n",
                   kernel_name, dst,
                   (unsigned int)(dst_range.max_addr - dst_range.min_addr),
                   TX81_GS_SPM_ADDR_MIN, TX81_GS_SPM_ADDR_MAX,
                   dma_oob_log_count));

    if (gatherscatter_spm_addr_oob(&dst_range)) {
      DMA_LOG_COND(
          dma_oob_count++,
          __EP_LOG__(3,
                     "[func @%s] fatal error: spm memory OOB, "
                     "gatherscatter dst (%p), range [%#lx, %#lx), "
                     "valid [%#x, %#x), dma_oob_count (%d)\n",
                     kernel_name, dst, (unsigned long)dst_range.min_addr,
                     (unsigned long)dst_range.max_addr, TX81_GS_SPM_ADDR_MIN,
                     TX81_GS_SPM_ADDR_MAX, dma_oob_count));
      if (get_dma_check_abort()) {
        abort();
      }
    }
  }

  // Create command buffer.
  RcsDataMove *cmd = g_intrinsic()->datamove_pointer;
  RcsDataMoveInstr inst = {I_CGRA,
                           {
                               0,
                           },
                           {
                               0,
                           }};

  St_StrideIteration src_si = {src_strideW, src_iterW,   src_strideH,
                               src_iterH,   src_strideN, src_iterN};
  St_StrideIteration dst_si = {dst_strideW, dst_iterW,   dst_strideH,
                               dst_iterH,   dst_strideN, dst_iterN};

  cmd->GatherScatter(&inst, (uint64_t)src, (uint64_t)dst, bytes, &src_si,
                     &dst_si);

  // Dispatch the command to accelerator
  RcsExecute(&inst);
  RcsWaitfinish();
  // SYNCHRONOUS_INTRINSIC_SWITCH;
  // Destroy the command buffer.
}
