//===------------------------ gemm.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::TsmGemm, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __Gemm(int64_t *srcA, int64_t *srcB, int64_t *srcBias, int64_t *dst,
            int32_t *dims, bool enPsum, int64_t *psum, bool enTransA,
            bool enTransB, int64_t batchSizeA, int64_t batchSizeB,
            int32_t reluMode, bool enBias, bool enNegScale, int64_t *negScale,
            bool enPosScale, int64_t *posScale, int64_t srcFmt,
            int64_t dstFmt) {
  // Create gemm command buffer.
  TsmGemm *gemm = g_intrinsic()->gemm_pointer;
  TsmNeInstr inst = {I_NEUR,
                     {
                         0,
                     },
                     {
                         0,
                     }};

  gemm->AddInput(&inst, (uint64_t)srcA, (uint64_t)srcB, (Data_Format)srcFmt);
  gemm->ConfigMKN(&inst, (uint32_t)dims[0], (uint32_t)dims[1],
                  (uint32_t)dims[2]);
  gemm->AddOutput(&inst, (uint64_t)dst, (Data_Format)dstFmt);
  gemm->SetPsum(&inst, enPsum, (uint64_t)psum, (Data_Format)dstFmt);
  gemm->SetTransflag(&inst, (uint8_t)enTransA, (uint8_t)enTransB);
  // TODO:
  // gemm->SetQuant();
  gemm->ConfigBatch(&inst, (uint32_t)batchSizeA, (uint32_t)batchSizeB);
  gemm->AddBias(&inst, enBias, (uint64_t)srcBias);
  gemm->SetNegativeAxisScale(&inst, enNegScale, (uint64_t)negScale);
  gemm->SetPositiveAxisScale(&inst, enPosScale, (uint64_t)posScale);
  switch (reluMode) {
  case ENRelu:
    gemm->EnableRelu(&inst);
    break;
  case ENLeakRelu:
    gemm->EnableLeakyRelu(&inst);
    break;
  default:
    break;
  }

  // Dispatch the command to accelerator
  TsmExecute(&inst);
}
