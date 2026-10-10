#include <cuda_runtime.h>
#include <__clang_cuda_ivcorex_intrinsics.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <int Rows, int Blocks>
__global__ void sme_copy(const unsigned char *src, unsigned char *dst,
                         unsigned stride, unsigned columnOffset,
                         unsigned sharedOffset) {
  constexpr unsigned tileBytes = Rows * Blocks * 64;
  constexpr unsigned storageBytes = tileBytes + 128;
  __shared__ __align__(64) unsigned char storage[storageBytes];
  for (unsigned i = threadIdx.x; i < storageBytes; i += blockDim.x)
    storage[i] = 0xa5;
  __syncthreads();

  // One whole 64-lane warp issues one tile; the other warp only consumes it.
  if (threadIdx.x < 64) {
    v4u32 desc;
    const auto address = reinterpret_cast<unsigned long long>(src);
    desc.x = static_cast<unsigned>(address);
    desc.y = static_cast<unsigned>(address >> 32);
    desc.z = ~0u;
    desc.w = stride;
    const unsigned sharedAddress = static_cast<unsigned>(
        reinterpret_cast<unsigned long long>(storage + 64 + sharedOffset));
    if constexpr (Rows == 1 && Blocks == 1)
      __ivcorex_sme_load_1x1b64(sharedAddress, desc, columnOffset, 0);
    else if constexpr (Rows == 1 && Blocks == 4)
      __ivcorex_sme_load_1x4b64(sharedAddress, desc, columnOffset, 0);
    else if constexpr (Rows == 1 && Blocks == 8)
      __ivcorex_sme_load_1x8b64(sharedAddress, desc, columnOffset, 0);
    else if constexpr (Rows == 4 && Blocks == 1)
      __ivcorex_sme_load_4x1b64(sharedAddress, desc, columnOffset, 0);
    else if constexpr (Rows == 8 && Blocks == 1)
      __ivcorex_sme_load_8x1b64(sharedAddress, desc, columnOffset, 0);
    else if constexpr (Rows == 16 && Blocks == 1)
      __ivcorex_sme_load_16x1b64(sharedAddress, desc, columnOffset, 0);
    // Bit 3 selects the G2S counter; a zero count drains this warp's SME.
    __ivcorex_sl_waitcnt(8);
  }
  __syncthreads();
  if (threadIdx.x >= 64)
    for (unsigned i = threadIdx.x - 64; i < storageBytes; i += 64)
      dst[i] = storage[i];
}

// Keep immediate-encoding probes separate from the runnable raw-layout copies.
template <unsigned Stride, unsigned ColumnOffset>
__global__ void sme_imm_alignment(const unsigned char *src,
                                  unsigned char *dst) {
  __shared__ __align__(64) unsigned char storage[1024];
  if (threadIdx.x < 64) {
    ld_sme_16x1b64_rowb16(storage, src, Stride, 0, 0, ColumnOffset, 0);
    __ivcorex_sl_waitcnt(8);
  }
  __syncthreads();
  for (unsigned i = threadIdx.x; i < sizeof(storage); i += blockDim.x)
    dst[i] = storage[i];
}

template __global__ void sme_imm_alignment<512, 64>(const unsigned char *,
                                                  unsigned char *);
template __global__ void sme_imm_alignment<512, 32>(const unsigned char *,
                                                  unsigned char *);
template __global__ void sme_imm_alignment<544, 64>(const unsigned char *,
                                                  unsigned char *);

static void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s: %s (%d)\n", operation,
                 cudaGetErrorString(error), static_cast<int>(error));
    std::exit(2);
  }
}

template <int Rows, int Blocks>
static bool runCase(unsigned stride, unsigned columnOffset,
                    unsigned sharedOffset) {
  constexpr unsigned rowBytes = Blocks * 64;
  constexpr unsigned storageBytes = Rows * rowBytes + 128;
  std::vector<unsigned char> input(Rows * stride + rowBytes);
  unsigned state = 0x31415926;
  for (auto &value : input) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    value = static_cast<unsigned char>(state);
  }
  std::vector<unsigned char> expected(storageBytes, 0xa5);
  for (unsigned row = 0; row < Rows; ++row)
    std::memcpy(expected.data() + 64 + sharedOffset + row * rowBytes,
                input.data() + row * stride + columnOffset, rowBytes);
  std::vector<unsigned char> output(storageBytes);
  unsigned char *deviceInput = nullptr;
  unsigned char *deviceOutput = nullptr;
  check(cudaMalloc(&deviceInput, input.size()), "cudaMalloc input");
  check(cudaMalloc(&deviceOutput, output.size()), "cudaMalloc output");
  check(cudaMemcpy(deviceInput, input.data(), input.size(),
                   cudaMemcpyHostToDevice), "cudaMemcpy input");
  bool matches = true;
  for (unsigned iteration = 0; iteration < 5; ++iteration) {
    check(cudaMemset(deviceOutput, 0, output.size()), "cudaMemset output");
    sme_copy<Rows, Blocks><<<1, 128>>>(deviceInput, deviceOutput, stride,
                                      columnOffset, sharedOffset);
    check(cudaGetLastError(), "SME launch");
    check(cudaDeviceSynchronize(), "SME synchronize");
    check(cudaMemcpy(output.data(), deviceOutput, output.size(),
                     cudaMemcpyDeviceToHost), "cudaMemcpy output");
    for (unsigned i = 0; i < storageBytes; ++i) {
      if (output[i] != expected[i]) {
        std::fprintf(stderr,
                     "FAIL %dx%db64 stride=%u col=%u shared=%u iteration=%u "
                     "byte=%u got=%u expected=%u\n",
                     Rows, Blocks, stride, columnOffset, sharedOffset,
                     iteration, i, static_cast<unsigned>(output[i]),
                     static_cast<unsigned>(expected[i]));
        matches = false;
        break;
      }
    }
    if (!matches)
      break;
  }
  check(cudaFree(deviceInput), "cudaFree input");
  check(cudaFree(deviceOutput), "cudaFree output");
  if (matches)
    std::printf("PASS %dx%db64 stride=%u col=%u shared=%u bytes=%u repeats=5\n",
                Rows, Blocks, stride, columnOffset, sharedOffset,
                Rows * rowBytes);
  return matches;
}

static bool runShapes(unsigned stride, unsigned columnOffset,
                      unsigned sharedOffset) {
  bool matches = true;
  matches &= runCase<1, 1>(stride, columnOffset, sharedOffset);
  matches &= runCase<1, 4>(stride, columnOffset, sharedOffset);
  matches &= runCase<1, 8>(stride, columnOffset, sharedOffset);
  matches &= runCase<4, 1>(stride, columnOffset, sharedOffset);
  matches &= runCase<8, 1>(stride, columnOffset, sharedOffset);
  matches &= runCase<16, 1>(stride, columnOffset, sharedOffset);
  return matches;
}

int main(int argc, char **argv) {
  const char *mode = argc == 2 ? argv[1] : "legal";
  if (argc > 2 || (std::strcmp(mode, "legal") != 0 &&
                   std::strcmp(mode, "unaligned-stride") != 0 &&
                   std::strcmp(mode, "unaligned-offset") != 0)) {
    std::fprintf(stderr,
                 "usage: %s [legal|unaligned-stride|unaligned-offset]\n",
                 argv[0]);
    return 2;
  }
  int count = 0;
  const auto error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess || count == 0) {
    std::fprintf(stderr, "GPU_UNAVAILABLE cudaGetDeviceCount: %s (%d), count=%d\n",
                 cudaGetErrorString(error), static_cast<int>(error), count);
    return 77;
  }
  check(cudaSetDevice(0), "cudaSetDevice");
  cudaDeviceProp props;
  check(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  std::printf("device=%s warpSize=%d mode=%s\n", props.name, props.warpSize, mode);
  if (props.warpSize != 64) {
    std::fprintf(stderr, "requires a 64-lane Iluvatar warp\n");
    return 2;
  }
  bool matches;
  if (std::strcmp(mode, "legal") == 0) {
    matches = runShapes(512, 0, 0);
    matches &= runShapes(640, 64, 4);
  } else {
    std::fprintf(stderr,
                 "EXPERIMENT: outside SDK alignment contract, not a supported "
                 "configuration\n");
    matches = std::strcmp(mode, "unaligned-stride") == 0
                  ? runShapes(544, 0, 0)
                  : runShapes(640, 32, 0);
  }
  return matches ? 0 : 1;
}
