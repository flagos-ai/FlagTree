#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define DEVICE __attribute__((device, always_inline))
#define GLOBAL __attribute__((address_space(1)))
#define SHARED __attribute__((address_space(3)))

// Separate, non-overloaded entry points keep decltype(&function) unambiguous.
#define ABI_CASE(NAME, TYPE)                                                   \
  DEVICE void Copy_##NAME(TYPE *dst, int64_t n, const TYPE *src, int64_t m) {  \
    for (int64_t i = 0; i < n && i < m; ++i)                                   \
      dst[i] = src[i];                                                         \
  }                                                                            \
  DEVICE void Scalar_##NAME(TYPE *dst, int64_t n, TYPE value) {                \
    for (int64_t i = 0; i < n; ++i)                                            \
      dst[i] = value;                                                          \
  }

ABI_CASE(Bool, bool)
ABI_CASE(I8, int8_t)
ABI_CASE(U8, uint8_t)
ABI_CASE(I16, int16_t)
ABI_CASE(U16, uint16_t)
ABI_CASE(I32, int32_t)
ABI_CASE(U32, uint32_t)
ABI_CASE(I64, int64_t)
ABI_CASE(U64, uint64_t)
ABI_CASE(F16, __half)
ABI_CASE(BF16, __nv_bfloat16)
ABI_CASE(F32, float)
ABI_CASE(F64, double)

DEVICE float fpx_to_float(__nv_fp8_e4m3 value) {
  return static_cast<float>(value);
}

DEVICE float fpx_to_float(__half value) { return __half2float(value); }

DEVICE float fpx_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

#define DOT_CASE(NAME, TYPE)                                                   \
  DEVICE void Dot_##NAME(TYPE *a, int64_t na, TYPE *b, int64_t nb,             \
                         GLOBAL float *out) {                                  \
    const int lane = threadIdx.x % 32;                                         \
    const int warp = threadIdx.x / 32;                                         \
    const int num_warps = blockDim.x / 32;                                     \
    __shared__ float partials[32];                                             \
    float cv = 0.0F;                                                           \
    for (int64_t i = 0; i < na && i < nb; ++i) {                               \
      float av = fpx_to_float(a[i]);                                           \
      float bv = fpx_to_float(b[i]);                                           \
      cv += av * bv;                                                           \
    }                                                                          \
    for (int offset = 16; offset > 0; offset /= 2)                             \
      cv += __shfl_xor_sync(0xFFFFFFFF, cv, offset);                           \
    if (lane == 0)                                                             \
      partials[warp] = cv;                                                     \
    __syncthreads();                                                           \
    cv = 0.0F;                                                                 \
    for (unsigned i = 0; i < num_warps; ++i)                                   \
      cv += partials[i];                                                       \
    if (threadIdx.x == 0)                                                      \
      *out = cv;                                                               \
  }

DOT_CASE(FP8, __nv_fp8_e4m3)
DOT_CASE(FP16, __half)
DOT_CASE(BF16, __nv_bfloat16)

DEVICE void BoolByte(int32_t *out, int64_t n, const bool *mask, int64_t m,
                     bool scalar) {
  for (int64_t i = 0; i < n && i < m; ++i)
    out[i] = __builtin_bit_cast(uint8_t, mask[i]) +
             10 * __builtin_bit_cast(uint8_t, scalar);
}

DEVICE void Step(float *x, int64_t n, float delta, bool forward) {
  for (int64_t i = 0; i < n; ++i)
    x[i] += forward ? delta : -delta;
}

DEVICE void MaskedModify(float *x, int64_t n, const bool *mask, int64_t m) {
  for (int64_t i = 0; i < n && i < m; ++i) {
    if (mask[i])
      x[i] = x[i] * 2.0f + 0.5f;
  }
}

DEVICE void MixedReturns(float *x, int64_t nx, int32_t *y, int64_t ny,
                         bool *mask, int64_t nm) {
  for (int64_t i = 0; i < nx; ++i)
    x[i] += 0.5f;
  for (int64_t i = 0; i < ny; ++i)
    y[i] = 2 * y[i] + 3;
  for (int64_t i = 0; i < nm; ++i)
    mask[i] = !mask[i];
}

DEVICE void Swap(float *x, int64_t nx, float *y, int64_t ny) {
  for (int64_t i = 0; i < nx && i < ny; ++i) {
    float value = x[i];
    x[i] = y[i];
    y[i] = value;
  }
}

DEVICE void Stamp(int32_t *x, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    x[i] += static_cast<int32_t>(threadIdx.x) * 10000 +
            static_cast<int32_t>(i) * 1000000;
}

DEVICE void InspectFragment(const int32_t *x, int64_t n,
                            GLOBAL int32_t *observed, GLOBAL int32_t *lengths,
                            int32_t capacity) {
  lengths[threadIdx.x] = static_cast<int32_t>(n);
  // A wrong fragment length must fail an assertion, not overrun the output.
  for (int64_t i = 0; i < n && i < capacity; ++i)
    observed[static_cast<int64_t>(threadIdx.x) * capacity + i] = x[i];
}

DEVICE void StoreOnly(const int32_t *x, int64_t n, GLOBAL int32_t *out) {
  for (int64_t i = 0; i < n; ++i)
    out[threadIdx.x + i * blockDim.x] = x[i] + 7;
}

// Fragment + ordinary tensor descriptor + rank-two buffered descriptor +
// float/bool/int scalars + a global pointer, all in one signature.
DEVICE void MixedMemory(float *fragment, int64_t count, SHARED float *ta,
                        SHARED float *tp, int64_t to, int64_t tn, int64_t ts,
                        SHARED float *ba, SHARED float *bp, int64_t bo,
                        int64_t br, int64_t bc, int64_t bs0, int64_t bs1,
                        float scale, bool flag, int32_t bias,
                        GLOBAL float *side) {
  for (int64_t i = 0; i < count; ++i) {
    int64_t col = threadIdx.x + i * blockDim.x;
    if (col < tn && col < br * bc) {
      float value = fragment[i] + tp[to + col * ts] * scale + bias;
      if (!flag)
        value = -value;
      fragment[i] = value;
      tp[to + col * ts] = value + 1.0f;
      bp[bo + (col / bc) * bs0 + (col % bc) * bs1] = value + 2.0f;
      side[col] = value + 3.0f;
    }
  }
  __syncthreads();
}
