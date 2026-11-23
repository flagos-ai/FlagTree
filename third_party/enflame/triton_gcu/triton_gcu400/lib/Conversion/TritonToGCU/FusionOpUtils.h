/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef KURAMA_TRITON_TO_GCU_FUSIONOPUTILS_H
#define KURAMA_TRITON_TO_GCU_FUSIONOPUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace triton {
namespace gcu {

using ValueRemapFn = llvm::function_ref<Value(Value)>;

Value getAllocaInputValue(Value val, ValueRemapFn remapFn);

bool canUseAllocaForValue(Value val, ValueRemapFn remapFn);

/// True when `maskType` may be a histogram mask that lives in a private
/// per-subthread stack buffer: one subthread owns every lane, so a private copy
/// is complete, and the byte-per-lane buffer fits the private budget.  The
/// fusion analysis uses this to decide whether the mask producer may write
/// straight into a `memref.alloca`; the `triton_gcu.histogram` lowering uses
/// the same predicate to decide whether to reuse that alloca or stage through
/// L1.  Both sides must agree, otherwise the fusion can hand the lowering an
/// alloca that the lowering then copies instead of reusing.
bool fitsPrivateHistogramMask(RankedTensorType maskType);

bool canUseAllocaForVboolValue(Value val);

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // KURAMA_TRITON_TO_GCU_FUSIONOPUTILS_H
