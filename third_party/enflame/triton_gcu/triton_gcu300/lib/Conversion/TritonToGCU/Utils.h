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
#ifndef KURAMA_TRITON_TO_GCU_UTILS_H_
#define KURAMA_TRITON_TO_GCU_UTILS_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace triton {
namespace gcu {

bool get_bool_env(const char *name);
SmallVector<unsigned> getWarpsPerCTA(Attribute layout);
SmallVector<unsigned> getElemsPerThread(Type type);
unsigned getTotalElemsPerThread(Type type);
unsigned getNumWarpsPerCTA(Type type);
unsigned getBpe(Type type);

inline int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  // C/C++'s integer division rounds towards 0.
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}
} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // KURAMA_TRITON_TO_GCU_UTILS_H_
