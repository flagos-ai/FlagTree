#ifndef _TANG_RUNTIME_H_
#define _TANG_RUNTIME_H_
#include "tang_rt/version.h"
#include "tang_rt/driver_types.h"
#include "tang_rt/vector_types.h"
#include "tang_runtime_api.h"

#ifndef TA_STREAM_LEGACY
#define TA_STREAM_LEGACY ((tangStream_t)0x01)
#endif  //! TA_STREAM_LEGACY

#ifndef TA_STREAM_PER_THREAD
#define TA_STREAM_PER_THREAD ((tangStream_t)0x02)
#endif  //! TA_STREAM_PER_THREAD

#ifndef tangStreamLegacy
#define tangStreamLegacy ((tangStream_t)0x01)
#endif  //! tangStreamLegacy

#ifndef tangStreamPerThread
#define tangStreamPerThread ((tangStream_t)0x02)
#endif  //! tangStreamPerThread

#ifdef __cplusplus
extern "C" {
#endif  //! __cplusplus

#ifdef __cplusplus
}
#endif  //! __cplusplus

#endif  //! _TANG_RUNTIME_H_
