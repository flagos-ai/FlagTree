/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "compiler_defs.h"

// FlagMega runtime ABI for generated C++ packages.
//
// A generated package is built into one shared library and called from the
// Python runtime through `ctypes`, so the boundary has to be plain C: no
// name mangling, no C++ objects crossing it, no exceptions escaping.  Three
// symbols make up the contract, and `codegen/cpp/package.py` writes the
// host side of them from the artifact's own runtime binding:
//
//   flagmega_launch(void **arguments, void *stream)
//       Launch the entry in the artifact's declared grid.  `arguments` is a
//       flat array of device pointers in exactly the order the artifact's
//       `runtime_binding["signature"]` lists -- external buffers first, then
//       memory pools.  The Python side passes `tensor.data_ptr()` values, so
//       every entry is already a device address; the generated host function
//       casts each to its element type.  `stream` is a `cudaStream_t` the
//       caller owns (the current torch stream, in practice).
//
//   flagmega_argument_count()
//       The expected length of that array, so a signature mismatch between a
//       rebuilt library and a stale manifest is caught before a launch reads
//       past the end of the array.
//
//   flagmega_abi_version()
//       Bumped whenever the three signatures above change shape.
//
// The launch function returns the backend's own error code (0 on success),
// not a FlagMega-specific one: the Python side reports it through the same
// path as any other driver error.

#define FLAGMEGA_CPP_ABI_VERSION 1

extern "C" {

// Defined by every generated package.
int flagmega_launch(void **arguments, void *stream);
int flagmega_argument_count();
int flagmega_abi_version();

} // extern "C"

namespace ftt {
namespace runtime {

// Grid/block geometry for one generated entry.  The values come from the
// artifact's launch contract, so they are compile-time constants in the
// generated source rather than runtime parameters.
struct launch_geometry {
    int grid_x;
    int grid_y;
    int grid_z;
    int block_threads;
};

// Select this CTA's slice of a replicated byte pool.
//
// A `block`-scoped memory pool holds one independent copy per CTA, laid out
// contiguously, so a buffer at pool offset `o` lives at
// `scope_index * scope_nbytes + o` for this CTA.  The buffer plan's own
// offsets are relative to the slice, which is why the slice base is applied
// here rather than folded into each address.
FTT_DEVICE_INLINE unsigned char *block_scope_base(unsigned char *pool,
                                                  dim_t scope_nbytes) {
#if defined(FTT_DEVICE_PASS)
    return pool + static_cast<dim_t>(blockIdx.x) * scope_nbytes;
#else
    (void)scope_nbytes;
    return pool;
#endif
}

} // namespace runtime
} // namespace ftt
