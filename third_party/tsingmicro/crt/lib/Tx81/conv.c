//===------------------------ conv.c --------------------------------------===//
//
//
//===----------------------------------------------------------------------===//
//
// Runtime API of MLIR operation tx::TsmConv, see Tx81Ops.td for detail.
//
//===----------------------------------------------------------------------===//

#include "tx81.h"

// The arguments list is aligned with TsmConv in Tx81Ops.td
void __Conv(int64_t opType, int64_t *srcAct, int64_t *srcActDims,
            int64_t *weight, int64_t *weightDims, bool enBias, int64_t *bias,
            bool enNegScale, int64_t *negScale, bool enPosScale,
            int64_t *posScale, bool enSparse, int64_t *sparse, bool enPsum,
            int64_t *psum, int64_t *pads, int64_t *unpads, int64_t *strides,
            int64_t *dilations, bool enLeakyRelu, int64_t srcActFmt,
            int64_t weightFmt, int64_t dstFmt, int64_t *dst, int64_t *dstDims) {
  // Create convolution command buffer.
  TsmConv *conv = g_intrinsic()->conv_pointer;
  TsmNeInstr inst = {I_NEUR,
                     {
                         0,
                     },
                     {
                         0,
                     }};

  // Convert to nhwc format
  Data_Shape shape = {(uint16_t)srcActDims[0], (uint16_t)srcActDims[1],
                      (uint16_t)srcActDims[2], (uint16_t)srcActDims[3]};

  Data_Shape wshape = {(uint16_t)weightDims[0], (uint16_t)weightDims[1],
                       (uint16_t)weightDims[2], (uint16_t)weightDims[3]};

  Data_Shape dstShape = {(uint16_t)dstDims[0], (uint16_t)dstDims[1],
                         (uint16_t)dstDims[2], (uint16_t)dstDims[3]};

  conv->AddInput(&inst, (int64_t)srcAct, shape, (Data_Format)srcActFmt);
  conv->AddWeight(&inst, (uint64_t)weight, wshape, (Data_Format)weightFmt);
  conv->AddBias(&inst, enBias, (uint64_t)bias);
  conv->AddOutput(&inst, (uint64_t)dst, dstShape, (Data_Format)dstFmt);
  conv->SetOpType(&inst, opType);
  conv->SetNegativeAxisScale(&inst, enNegScale, (uint64_t)negScale);
  conv->SetPositiveAxisScale(&inst, enPosScale, (uint64_t)posScale);
  conv->SetSparse(&inst, enSparse, (uint64_t)sparse);
  // FIXME: Should we have psum format instead?
  conv->SetPsum(&inst, enPsum, (uint64_t)psum, (Data_Format)dstFmt);
  conv->SetPads(&inst, pads[0], pads[1], pads[2], pads[3]);
  conv->SetUnPads(&inst, unpads[0], unpads[1], unpads[2], unpads[3]);
  conv->SetKernelStrides(&inst, strides[0], strides[1], strides[2], strides[3]);
  conv->SetDilations(&inst, dilations[0], dilations[1]);
  if (enLeakyRelu)
    conv->EnableLeakyRelu(&inst);
  else
    conv->EnableRelu(&inst);

  // Dispatch the command to accelerator
  TsmExecute(&inst);

  // Destroy the command buffer.
}
