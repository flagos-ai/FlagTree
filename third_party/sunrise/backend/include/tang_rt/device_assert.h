// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
