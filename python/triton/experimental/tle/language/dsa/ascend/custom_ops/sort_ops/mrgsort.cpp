#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"

extern "C" {

__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_mrgsort(
    memref_t<__ubuf__ float, 1> *src_proposals, const uint64_t off0,
    const uint64_t off1, const uint64_t off2, const uint64_t off3,
    const uint32_t len0, const uint32_t len1, const uint32_t len2,
    const uint32_t len3, const bool if_exhausted_suspension,
    const uint32_t valid_bit, const uint32_t repeat_times,
    memref_t<__ubuf__ float, 1> *dst) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(((len0 >= 0) && (len0 <= 4095)) &&
         "mrgsort: len0 must be in range [0, 4095]");
  assert(((len1 >= 0) && (len1 <= 4095)) &&
         "mrgsort: len1 must be in range [0, 4095]");
  assert(((len2 >= 0) && (len2 <= 4095)) &&
         "mrgsort: len2 must be in range [0, 4095]");
  assert(((len3 >= 0) && (len3 <= 4095)) &&
         "mrgsort: len3 must be in range [0, 4095]");
  assert(((valid_bit == 3) || (valid_bit == 7) || (valid_bit == 15)) &&
         "mrgsort: valid_bit valid values are 3, 7, 15");
#endif
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  __ubuf__ float *addr_array[4];
  addr_array[0] = src_ptr + off0 * npp;
  addr_array[1] = src_ptr + off1 * npp;
  addr_array[2] = src_ptr + off2 * npp;
  addr_array[3] = src_ptr + off3 * npp;
  uint64_t config = 0;
  // Xt[7:0]: repeat time, Xt[11:8]: 4-bit mask signal,
  // Xt[12]: 1-enable input list exhausted suspension
  config |= (repeat_times & 0xFF);
  config |= (uint64_t(valid_bit & 0xF) << 8);
  config |= (uint64_t(if_exhausted_suspension & 0x1) << 12);
  uint64_t lens = 0;
  lens |= (uint64_t(len0 & 0xFFFF));
  lens |= (uint64_t(len1 & 0xFFFF) << 16);
  lens |= (uint64_t(len2 & 0xFFFF) << 32);
  lens |= (uint64_t(len3 & 0xFFFF) << 48);
  auto dst_ptr = dst->aligned + dst->offset;
  vmrgsort4(dst_ptr, addr_array, lens, config);
}

} // extern "C"
