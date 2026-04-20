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
#ifndef GCU_DIALECT_GCU_IR_DIALECT_H
#define GCU_DIALECT_GCU_IR_DIALECT_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Dialect/GCU/IR/Dialect.h.inc"
#include "Dialect/GCU/IR/OpsEnums.h.inc"
#include "Dialect/GCU/IR/Traits.h"
#include "Dialect/GCU/IR/Types.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/GCU/IR/OpsAttributes.h.inc"
#define GET_OP_CLASSES
#include "Dialect/GCU/IR/Ops.h.inc"

namespace mlir {
namespace gcu {} // namespace gcu
} // namespace mlir

#endif // GCU_DIALECT_GCU_IR_DIALECT_H
