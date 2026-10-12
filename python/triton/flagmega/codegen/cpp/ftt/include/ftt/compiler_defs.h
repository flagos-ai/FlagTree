/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

// FTT (FlagMega Tensor Template) compiler and arch detection.
//
// Modelled on nncase NTT's compiler_defs.h / arch dispatch: a single macro
// selects which arch backend the headers specialize against, so kernel code
// stays arch-neutral and only `arch/<name>/*` changes per target.

#if defined(__ILUVATAR__) || defined(FTT_ARCH_ILUVATAR)
#define FTT_ARCH_ILUVATAR 1
#endif

#if !defined(FTT_ARCH_ILUVATAR)
#error "FTT currently supports only the Iluvatar arch; define FTT_ARCH_ILUVATAR."
#endif

// Device/host function qualifiers.  Iluvatar's COREX toolchain accepts the
// CUDA-style attributes, so the shapes here match what the vendor clang wants.
#if defined(__CUDACC__)
#define FTT_DEVICE __device__
#define FTT_HOST __host__
#define FTT_HOST_DEVICE __host__ __device__
#define FTT_GLOBAL extern "C" __global__
#else
#define FTT_DEVICE
#define FTT_HOST
#define FTT_HOST_DEVICE
#define FTT_GLOBAL extern "C"
#endif

// Device-compilation guard.
//
// A CUDA-style source is preprocessed twice, and only the device pass may use
// the vendor builtins and the special registers; the host pass needs a stub
// for the same function so the translation unit still parses.  COREX marks
// that pass with `__CUDA_ARCH__`, like nvcc -- it does NOT define `__ivcore__`
// (verified against the driver's own `-dM` output for both passes).  Guarding
// on the wrong macro is silent: every primitive takes its host stub, so
// `thread_id()` returns 0 on device and a kernel computes one lane's worth of
// a correct-looking answer.  Spell the guard only through this macro.
#if defined(__CUDA_ARCH__)
#define FTT_DEVICE_PASS 1
#endif

#define FTT_INLINE inline __attribute__((always_inline))
#define FTT_DEVICE_INLINE FTT_DEVICE FTT_INLINE

// Loop hints.  Kernel bodies are generated with statically known trip counts
// wherever the IR knows them, so full unrolling is the common case.
#define FTT_UNROLL _Pragma("unroll")
#define FTT_NO_UNROLL _Pragma("nounroll")

namespace ftt {

using dim_t = long long;
using size_type = unsigned long long;

} // namespace ftt
