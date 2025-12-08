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
#ifndef TRITON_GCU_CONVERSION_PASSES_H
#define TRITON_GCU_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

namespace mlir {
/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace mlir

#endif // TRITON_GCU_CONVERSION_PASSES_H
