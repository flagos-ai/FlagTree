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
#ifndef MATH_EXT_TYPES
#define MATH_EXT_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialect/MathExt/IR/MathExtDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//
// Types
//
// I32 Type
// MathExt_I32 -> I32
// MathExt_I32Tensor -> I32Tensor
def MathExt_I32Like : AnyTypeOf<[I32, I32Tensor]>;

def MathExt_Int : AnyTypeOf<[I32, I64], "integer">;
def MathExt_IntTensor : TensorOf<[MathExt_Int]>;
def MathExt_IntLike : AnyTypeOf<[MathExt_Int, MathExt_IntTensor]>;

#endif //MATH_EXT_TYPES
