#ifndef _TANGRT_DEVICE_ASSERT_H_
#define _TANGRT_DEVICE_ASSERT_H_

#include <assert.h>

#include <utility>

#include "tang_rt/device_functions.h"

extern "C" {
// #pragma push_macro("size_t")
// #define size_t unsigned
__device__ void __assertfail(const char *__message,
                             const char *__file,
                             unsigned    __line,
                             const char *__function,
                             unsigned    __charSize)
//__attribute__((noreturn))
{
  __pt_printf("%d: block: [%d,%d,%d], thread: [%d,%d,%d] Assertion failed.\n",
              __line,
              blockIdx.x,
              blockIdx.y,
              blockIdx.z,
              threadIdx.x,
              threadIdx.y,
              threadIdx.z);
  asm volatile("exit\n\t" ::: "memory");
}
// #undef size_t
// #pragma pop_macro("size_t")

// In order for standard assert() macro on linux to work we need to
// provide device-side __assert_fail()
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file,
                                            unsigned    __line,
                                            const char *__function) {
  __assertfail(__message, __file, __line, __function, sizeof(char));
}
}  // end extern "C"

#endif
