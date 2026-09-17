//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_UNSTRUCTUREDTOMK_UNSTRUCTUREDTOMK_H
#define TRITON_CONVERSION_UNSTRUCTUREDTOMK_UNSTRUCTUREDTOMK_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUnstructuredToMKPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_UNSTRUCTUREDTOMK_UNSTRUCTUREDTOMK_H
