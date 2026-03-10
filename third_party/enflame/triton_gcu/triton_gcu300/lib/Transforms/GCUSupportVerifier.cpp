/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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
#include <string>

#include "Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_GCU64TYPEVERIFIERPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {
struct GCU64TypeVerifierPass
    : public mlir::impl::GCU64TypeVerifierPassBase<GCU64TypeVerifierPass> {
  using Base::Base;

  void runOnOperation() override {
    auto gpuModuleOp = getOperation();
    gpuModuleOp.walk([&](::mlir::triton::FuncOp funcOp) {
      // http://git.enflame.cn/sw/FlagGems/-/blob/enflame_flaggems_vllm/src/flag_gems/ops/to.py
      // filter `to` kernel in flaggems vllm
      std::string funcName(funcOp.getName());
      if (funcName.find("to_dtype_func") != std::string::npos) {
        return;
      }
      for (auto type : funcOp.getFunctionType().getInputs()) {
        if (llvm::isa<PointerType>(type)) {
          if (dyn_cast<PointerType>(type).getPointeeType().isIntOrFloat() &&
              64 == dyn_cast<PointerType>(type)
                        .getPointeeType()
                        .getIntOrFloatBitWidth()) {
            funcOp.emitError("64-bit data type not supported on GCU300!");
            if (!test_mode)
              signalPassFailure();
          }
        }
        if (type.isIntOrFloat() && 64 == type.getIntOrFloatBitWidth()) {
          funcOp.emitError("64-bit data type not supported on GCU300!");
          if (!test_mode)
            signalPassFailure();
        }
      }
    });
  }
};
} // namespace
