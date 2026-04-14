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
#ifndef KURAMA_COMMON_CONSTANT_UTIL_H_
#define KURAMA_COMMON_CONSTANT_UTIL_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace triton {
namespace gcu {
Value createConstantZero(OpBuilder &builder, Location loc, Type elemType);
Value createConstantNaN(OpBuilder &builder, Location loc, Type elemType);
Value createConstantInf(OpBuilder &builder, Location loc, Type elemType,
                        bool isNegative = false);
} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // KURAMA_COMMON_CONSTANT_UTIL_H_
