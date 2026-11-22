/* Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#pragma once

// FTT -- FlagMega Tensor Template.
//
// Dimension, shape and vector infrastructure adapted from nncase NTT.
// Tensor views use caller-owned storage from the generated buffer plan.
//
// Arch dispatch follows NTT's pattern: one macro selects the arch directory,
// and only `arch/<name>/*` changes per target.  Iluvatar BI-V150 is the first
// arch; adding another means adding a sibling directory with `arch_types.h`
// and `primitive_ops.h`, not touching the layers above.

#include "compiler_defs.h"
#include "shape.h"
#include "tensor.h"
#include "vector_ops.h"
#include "utility.h"
#include "apply.h"

#if defined(FTT_ARCH_ILUVATAR)
#include "arch/iluvatar/arch_types.h"
#include "arch/iluvatar/primitive_ops.h"
#elif defined(FTT_ARCH_NVIDIA)
#include "arch/nvidia/arch_types.h"
#include "arch/nvidia/primitive_ops.h"
#endif

#if defined(FTT_ARCH_ILUVATAR) || defined(FTT_ARCH_NVIDIA)
#include "primitive_ops.h"
#endif
