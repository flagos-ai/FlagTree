/*
Copyright declaration.
*/

// cuda/include/vector_types.h

#ifndef _TANG_RT_INCLUDE_VECTOR_TYPES_H_
#define _TANG_RT_INCLUDE_VECTOR_TYPES_H_

#include "tang_rt/host_defines.h"

/**
 * Struct for data in 3D
 */

#if defined(__DIM3_TYPE__)
typedef dim3 __DIM3_TYPE__;
#else
typedef struct dim3 {
  unsigned x;  ///< x
  unsigned y;  ///< y
  unsigned z;  ///< z
#ifdef __cplusplus
#if __cplusplus >= 201103L
  constexpr dim3(unsigned _x = 1, unsigned _y = 1, unsigned _z = 1)
    : x(_x), y(_y), z(_z) {}
#else
  dim3(unsigned _x = 1, unsigned _y = 1, unsigned _z = 1)
    : x(_x), y(_y), z(_z) {}
#endif  //! __cplusplus >= 201103
#endif  //! __cplusplus
} dim3;
#endif  //! no __DIM3_TYPE__

#endif  //! _TANG_RT_INCLUDE_VECTOR_TYPES_H_
