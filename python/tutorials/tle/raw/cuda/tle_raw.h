#pragma once

#include <stddef.h>

#define TLE_DEVICE_INLINE __attribute__((device, always_inline)) inline
#define TLE_RAW_FUNC TLE_DEVICE_INLINE void

namespace tle_raw
{

// Layout fields are constants supplied by the generated wrapper, not template
// arguments. const prevents mutation; constant propagation requires inlining.
template<typename T>
struct TensorFragment
{
    const T* const handle;
    const unsigned size;

    TLE_DEVICE_INLINE const T& operator[](size_t index) const
    {
        return handle[index];
    }
};

template<typename T>
struct PlainBuffer1D
{
    T* const handle;
    const size_t stride;

    TLE_DEVICE_INLINE T& operator[](size_t index) const
    {
        return handle[index * stride];
    }
};

TLE_DEVICE_INLINE unsigned thread_id()
{
    unsigned tid;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tid));
    return tid;
}

}
