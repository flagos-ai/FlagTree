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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_GCU64TYPEVERIFIERPASS
#include "Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

bool is_64_width(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 64;
  if (auto floatTy = dyn_cast<FloatType>(type))
    return floatTy.getWidth() == 64;
  if (auto ptrTy = dyn_cast<PointerType>(type))
    return is_64_width(ptrTy.getPointeeType());
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return is_64_width(tensorTy.getElementType());
  return false;
}

bool opUses64Bit(Operation *op) {
  auto uses64 = [](Type ty) { return is_64_width(ty); };
  return llvm::any_of(op->getResults(),
                      [&](Value v) { return uses64(v.getType()); }) ||
         llvm::any_of(op->getOperands(),
                      [&](Value v) { return uses64(v.getType()); }) ||
         llvm::any_of(op->getRegions(), [&](Region &region) {
           return llvm::any_of(region, [&](Block &block) {
             return llvm::any_of(block.getArguments(), [&](BlockArgument arg) {
               return uses64(arg.getType());
             });
           });
         });
}

struct GCU64TypeVerifierPass
    : public mlir::impl::GCU64TypeVerifierPassBase<GCU64TypeVerifierPass> {
  using Base::Base;

  void runOnOperation() override {
    auto gpuModuleOp = getOperation();
    gpuModuleOp.walk([&](::mlir::triton::FuncOp funcOp) {
      // http://git.enflame.cn/sw/FlagGems/-/blob/enflame_flaggems_vllm/src/flag_gems/ops/to.py
      // filter `to` kernel in flaggems vllm
      std::string funcName(funcOp.getName());
      if (funcName.find("to_dtype_func") != std::string::npos)
        return;

      auto report = [&](Operation *op, const char *msg) {
        op->emitError(msg);
        if (!test_mode)
          signalPassFailure();
      };

      for (auto type : funcOp.getFunctionType().getInputs()) {
        if (is_64_width(type))
          report(funcOp.getOperation(),
                 "64-bit data type not supported on GCU300!: function input "
                 "argument has 64-bit type");
      }

      // If you want to check if the i64 type is used inside funcop, please set
      // the ENABLE_I64_CHECK_INNER=1 environment variable
      const char *check_inner_i64 = std::getenv("ENABLE_I64_CHECK_INNER");
      if (check_inner_i64 && (std::string(check_inner_i64) == "1" ||
                              std::string(check_inner_i64) == "true")) {
        funcOp.walk([&](Operation *op) {
          if (op == funcOp.getOperation())
            return;
          if (opUses64Bit(op)) {
            report(op, "64-bit data type not supported on GCU300!: internal "
                       "operation uses 64-bit type: ");
            op->print(llvm::errs());
            llvm::errs() << "\n";
          }
        });
      }
    });
  }
};
} // namespace
