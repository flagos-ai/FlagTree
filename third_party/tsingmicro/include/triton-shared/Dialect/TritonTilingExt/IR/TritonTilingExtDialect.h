//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRITON_TILING_EXT_IR_TRITON_TILING_EXT_DIALECT_H_
#define MLIR_DIALECT_TRITON_TILING_EXT_IR_TRITON_TILING_EXT_DIALECT_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

//===----------------------------------------------------------------------===//
// TritonTilingExt Operations
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOpsDialect.h.inc"

// Include the generated interface declarations.
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtInterfaces.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonTilingExt operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtOps.h.inc"

namespace mlir {

namespace ttx {

// -----------------------------------------------------------------------------
// BufferizableOpInterface
// -----------------------------------------------------------------------------
// All TritonTilingExtOps need to support bufferization: the process of
// allocating buffers for tensors, thereby converting inputs and outputs of
// tensor type to memref. This process is done by implementing the
// "BufferizableOpInterface". We implement the interface for TritonTilingExtOps
// through an external model instead of directly in TritonTilingExtOps.td to be
// consistent with other ops in the mlir project. See some examples here:
// - mlir/lib/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.cpp
// - mlir/lib/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.cpp
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

// -----------------------------------------------------------------------------
// TilingInterface
// -----------------------------------------------------------------------------
// The three methods `getTiledImplementation`, `getResultTilePosition`, and
// `generateResultTileValue` are implemented as part of the TilingInterface.
// (see TilingInterface.td). These three methods are re-used across
// all TritonTilingExtOps, while others method are implemented individually by
// each operator depending on their use cases.
template <typename TritonTilingExtOpTy>
FailureOr<TilingResult> getTiledImplementation(TritonTilingExtOpTy op,
                                               OpBuilder &b,
                                               ArrayRef<OpFoldResult> offsets,
                                               ArrayRef<OpFoldResult> sizes);

template <typename TritonTilingExtOpTy>
LogicalResult getResultTilePosition(TritonTilingExtOpTy op, OpBuilder &b,
                                    unsigned resultNumber,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVector<OpFoldResult> &resultOffsets,
                                    SmallVector<OpFoldResult> &resultSizes);

template <typename TritonTilingExtOpTy>
FailureOr<TilingResult>
generateResultTileValue(TritonTilingExtOpTy op, OpBuilder &b,
                        unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes);

// -----------------------------------------------------------------------------
// MemoryEffectsOpInterface
// -----------------------------------------------------------------------------
// Implementation of the MemoryEffectsOpInterface for TritonTilingExtOps.
// This allows DCE pass to determine if a TritonTilingExtOp is safe to be
// removed. see TritonTilingExtOps.td for more details.
template <typename TritonTilingExtOpTy>
void getEffects(
    TritonTilingExtOpTy op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects);

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
// Utility method to extract a slice from the input source using either
// tensor::ExtractSlice or memref::SubView
Value getSlice(OpBuilder &b, Location loc, Value source,
               ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
               ArrayRef<OpFoldResult> strides);

} // namespace ttx
} // namespace mlir

#endif // MLIR_DIALECT_TRITON_TILING_EXT_IR_TRITON_TILING_EXT_DIALECT_H_
