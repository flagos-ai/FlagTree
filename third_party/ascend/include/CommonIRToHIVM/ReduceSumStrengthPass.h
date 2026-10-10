/*
 * Copyright 2026- Xcoresigma Technology Co., Ltd
 */

#ifndef TRITON_ADAPTER_COMMONIR_TO_HIVM_REDUCESUMSTRENGTHPASS_H
#define TRITON_ADAPTER_COMMONIR_TO_HIVM_REDUCESUMSTRENGTHPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createReduceSumStrengthPass();

std::unique_ptr<OperationPass<ModuleOp>>
createReduceSumStrengthPass(bool enable, int32_t splitFactor);

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_COMMONIR_TO_HIVM_REDUCESUMSTRENGTHPASS_H
