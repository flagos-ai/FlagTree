#include "kernel_operator.h"
using namespace AscendC;

// Independent matrix-transport reference: preserve aligned UB rows from the
// load through the byte-counted GM store. Arithmetic and permutations are SIMD.
extern "C" __global__ __aicore__ void matrix_transport_aiv(GM_ADDR x, GM_ADDR y,
                                                           GM_ADDR indices,
                                                           uint32_t transpose) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> inputBuffer, outputBuffer, indexBuffer;
  uint32_t elements = transpose ? 128 : 8192;
  pipe.InitBuffer(inputBuffer, elements * sizeof(float));
  pipe.InitBuffer(outputBuffer, transpose ? 512 : 32);
  pipe.InitBuffer(indexBuffer, transpose ? 512 : 32);
  auto input = inputBuffer.Get<float>();
  auto output = outputBuffer.Get<float>();
  auto index = indexBuffer.Get<uint32_t>();
  GlobalTensor<float> gx, gy;
  GlobalTensor<uint32_t> gi;
  gx.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
  gy.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
  gi.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(indices));
  DataCopyPadExtParams<float> padding;
  if (transpose) {
    DataCopy(index, gi, 128);
    pipe_barrier(PIPE_ALL);
  }
  uint32_t programs = transpose ? 56 : 256;
  for (uint32_t program = GetBlockIdx(); program < programs;
       program += GetBlockNum()) {
    if (transpose) {
      uint32_t columns = 221 - program * 4;
      columns = columns < 4 ? columns : 4;
      Duplicate(input, float(0), 128);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams load(11, columns * 8, (442 - columns * 2) * 4, 0, 0);
      DataCopyPad(input, gx[program * 8], load, padding);
      pipe_barrier(PIPE_ALL);
      Adds(input, input, float(1), 128);
      pipe_barrier(PIPE_ALL);
      Gather(output, input, index, uint32_t(0), 128);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams store(columns, 22 * 4, 1, (418 - 22) * 4, 0);
      DataCopyPad(gy[program * 4 * 418], output, store);
    } else {
      DataCopyExtParams load(1024, 16, (1024 - 4) * 4, 0, 0);
      DataCopyPad(input, gx[program * 4], load, padding);
      pipe_barrier(PIPE_ALL);
      Adds(input, input, float(1), 8192);
      pipe_barrier(PIPE_ALL);
      DataCopyExtParams store(1024, 16, 0, (1024 - 4) * 4, 0);
      DataCopyPad(gy[program * 4], input, store);
    }
    pipe_barrier(PIPE_ALL);
  }
}
