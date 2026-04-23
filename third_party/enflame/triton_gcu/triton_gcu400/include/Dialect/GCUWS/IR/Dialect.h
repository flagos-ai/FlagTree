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
#ifndef DIALECT_GCUWS_IR_GCUWSDIALECT_H_
#define DIALECT_GCUWS_IR_GCUWSDIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "Dialect/GCUWS/IR/Dialect.h.inc"
#include "Dialect/GCUWS/IR/GCUWSAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/GCUWS/IR/GCUWSAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/GCUWS/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "Dialect/GCUWS/IR/GCUWSOpInterfaces.h.inc"
#include "Dialect/GCUWS/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace gcuws {} // namespace gcuws
} // namespace triton
} // namespace mlir

#endif // DIALECT_GCUWS_IR_GCUWSDIALECT_H_
