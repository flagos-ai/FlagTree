/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

// FTT -- FlagMega Tensor Template.
//
// A small C++ header library for FlagMega's generated kernels, modelled on
// nncase's NTT but scoped to what the FlagMega IR actually emits.  Two
// deliberate differences from NTT:
//
//   * Shapes are always fixed.  The FlagMega IR resolves every extent before
//     the codegen package plan exists, so the dynamic-dimension machinery
//     NTT carries would be dead weight here.
//   * Tensors are views only.  Generated kernels never allocate; buffers come
//     from the artifact's workspace / block_local_data / arguments with
//     offsets fixed by the buffer plan.
//
// Arch dispatch follows NTT's pattern: one macro selects the arch directory,
// and only `arch/<name>/*` changes per target.  Iluvatar BI-V150 is the first
// arch; adding another means adding a sibling directory with `arch_types.h`
// and `primitive_ops.h`, not touching the layers above.

#include "compiler_defs.h"
#include "shape.h"
#include "tensor.h"

#if defined(FTT_ARCH_ILUVATAR)
#include "arch/iluvatar/arch_types.h"
#include "arch/iluvatar/primitive_ops.h"
#else
#error "No FTT arch selected."
#endif

#include "apply.h"
#include "primitive_ops.h"
