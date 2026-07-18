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
