#include "Utils.h"
#include "Vector/Arange/ArangeUtils.h"
#include "Vector/Cast/CastUtils.h"
#include "Vector/Sort/SortUtils.h"

extern "C" {

__aiv__ __attribute__((always_inline)) void _mlir_ciface_custom_sort32(
    memref_t<__ubuf__ float, 1> *src0, memref_t<__ubuf__ int32_t, 1> *src1,
    const int32_t repeat_times, memref_t<__ubuf__ float, 1> *dst) {
#ifdef ENABLE_CPU_TRACE_INTRINSIC
  assert(((repeat_times >= 0) && (repeat_times <= 255)) &&
         "sort32: repeat_times must be in range [0, 255]");
#endif
  auto src0_ptr = src0->aligned + src0->offset;
  auto src1_ptr = (__ubuf__ uint32_t *)(src1->aligned + src1->offset);
  auto dst_ptr = dst->aligned + dst->offset;
  vbitsort(dst_ptr, src0_ptr, src1_ptr, repeat_times);
}

} // extern "C"
