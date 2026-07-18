// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _TAPTI_VERSION_
#define _TAPTI_VERSION_

#include <stdint.h>
#include "tapti_result.h"

#define TAPTI_API_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif  //! __cplusplus

#if defined(_MSC_VER)
#define TAPTI_DEPRECATED __declspec(deprecated)
#define TAPTI_API_EXPORT __declspec(dllexport)
#define TAPTI_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define TAPTI_DEPRECATED __attribute__((deprecated))
#define TAPTI_API_EXPORT __attribute__((visibility("default")))
#define TAPTI_API_IMPORT __attribute__((visibility("default")))
#else
#define TAPTI_DEPRECATED
#define TAPTI_API_EXPORT
#define TAPTI_API_IMPORT
#endif  //! UNKNOWN COMPILER

#if defined(tapti_shared_EXPORTS)
#define TAPTI_API TAPTI_API_EXPORT
#else
#define TAPTI_API TAPTI_API_IMPORT
#endif  //! For user

TAptiResult TAPTI_API taptiGetVersion(uint32_t *version);

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif // _TAPTI_VERSION_
