//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H
#define TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnstructuredToMemrefPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_UNSTRUCTUREDTOMEMREF_UNSTRUCTUREDTOMEMREF_H
