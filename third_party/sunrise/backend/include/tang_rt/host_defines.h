/*
Copyright declaration.
*/

#ifndef _TANG_RT_INCLUDE_HOST_DEFINES_H_
#define _TANG_RT_INCLUDE_HOST_DEFINES_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define __dparm(x) = x
#else
#define __dparm(x)
#endif

#ifdef __TANGC_MAJOR__

#ifndef __device__
#define __device__  __Tdevice__
#define __Tdevice__ __attribute__((Tdevice))
#endif

#ifndef __global__
#define __global__  __Tglobal__
#define __Tglobal__ __attribute__((Tglobal))
#endif

#ifndef __constant__
#define __constant__  __Tconstant__
#define __Tconstant__ __attribute__((Tconstant))
#endif

#ifndef __host__
#define __host__  __Thost__
#define __Thost__ __attribute__((Thost))
#endif

#ifndef __shared__
#define __shared__  __Tshared__
#define __Tshared__ __attribute__((Tshared))
#endif

#ifndef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif

#endif

#if defined(_MSC_VER)
#define TANGRT_DEPRECATED __declspec(deprecated)
#define TANGRT_API_EXPORT __declspec(dllexport)
#define TANGRT_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define TANG_DEPRECATED __attribute__((deprecated))
#define TANG_API_EXPORT __attribute__((visibility("default")))
#define TANG_API_IMPORT __attribute__((visibility("default")))
#else
#define TANG_DEPRECATED
#define TANG_API_EXPORT
#define TANG_API_IMPORT
#endif  // unknown compiler, may needs extra care.

#if defined(tangrt_shared_EXPORTS)
#define TANGRT_API_PUBLIC TANG_API_EXPORT
#elif !defined(__TANGRT_API_VERSION_INTERNAL)
#define TANGRT_API_PUBLIC TANG_API_IMPORT
#else
#define TANGRT_API_PUBLIC
#endif

/**************************************************
 * _ptds: Per-Thread-Default-Stream API use ptds to
 * run commands.
 * _ptsz suffix: Per-Thread-Stream-Zero API use ptds to
 * run commands when the given stream is null.
 * See the following code for details:
 * @code
 * tangError_t tangMemcpyAsync_ptsz(..., tangStream_t stream) {
 *   return tangMemcpyAsyncImpl(..., stream ? stream : TA_STREAM_PER_THREAD);
 * }
 * tangError_t tangMemcpyAsync(..., tangStream_t stream) {
 *   return tangMemcpyAsyncImpl(..., stream ? stream : TA_STREAM_LEGACY);
 * }
 * @endcode
 **************************************************/
#if defined(__TANGRT_API_PER_THREAD_DEFAULT_STREAM)
#define __TANGRT_API_PTDS(api) api##_ptds
#define __TANGRT_API_PTSZ(api) api##_ptsz
#else
#define __TANGRT_API_PTDS(api) api
#define __TANGRT_API_PTSZ(api) api
#endif  //! __TANGRT_API_PER_THREAD_DEFAULT_STREAM

#if defined(__TANGRT_API_VERSION_INTERNAL)
#undef __TANGRT_API_PTDS
#undef __TANGRT_API_PTSZ
#define __TANGRT_API_PTDS(api) api
#define __TANGRT_API_PTSZ(api) api
#endif  // __TANGRT_API_VERSION_INTERNAL

#endif  //! _TANG_RT_INCLUDE_HOST_DEFINES_H_
