/*
 * Copyright 2023 - 2024 Enflame.All Rights Reserved.
 *
 */
#include <map>
#include <string>
#include <vector>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_GCUFLATTENTRITONFUNCPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {
static uint64_t uniqueId = 1;

struct FlattenTritonFuncPass
    : public mlir::impl::GCUFlattenTritonFuncPassBase<FlattenTritonFuncPass> {
  using Base::Base;

  void runOnOperation() override;

private:
  void flattenFuncOp(
      Operation *funcOp,
      std::map<llvm::StringRef, std::vector<Operation *>> &funcName2CallOps);

  gpu::GPUModuleOp moduleOp_;
};
} // namespace

void FlattenTritonFuncPass::runOnOperation() {
  moduleOp_ = getOperation();

  std::vector<Operation *> publicFuncOps;
  for (auto ttFunc : moduleOp_.getOps<triton::FuncOp>()) {
    if (ttFunc.isPublic()) {
      publicFuncOps.push_back(ttFunc.getOperation());
    }
  }

  for (auto &publicFuncOp : publicFuncOps) {
    std::map<llvm::StringRef, std::vector<Operation *>> funcName2CallOps;
    funcName2CallOps.clear();
    flattenFuncOp(publicFuncOp, funcName2CallOps);
  }
}

void FlattenTritonFuncPass::flattenFuncOp(
    Operation *funcOp,
    std::map<llvm::StringRef, std::vector<Operation *>> &funcName2CallOps) {
  auto func = llvm::dyn_cast<triton::FuncOp>(funcOp);
  auto curFuncName = func.getName();

  std::vector<Operation *> calleeFuncOps;

  func.walk([&](triton::CallOp call) {
    auto calleeFuncName = call.getCallee();
    auto calleeFunc =
        moduleOp_.template lookupSymbol<triton::FuncOp>(calleeFuncName);

    if (curFuncName == calleeFuncName) {
      std::string o;
      llvm::raw_string_ostream os(o);
      funcOp->print(os);
      os.str();
      llvm_unreachable(
          (std::string("unsupported recursive call: \n") + o).c_str());
    }

    if (calleeFunc.isExternal())
      return;

    if (funcName2CallOps[calleeFuncName].size() != 0) {
      std::string newFuncName =
          std::string(calleeFuncName) + "_clone" + std::to_string(uniqueId++);

      auto calleeFuncClone = calleeFunc.clone();
      calleeFuncClone.setName(newFuncName);
      call.setCallee(newFuncName);

      moduleOp_.insert(calleeFunc, calleeFuncClone);

      calleeFuncOps.push_back(calleeFuncClone.getOperation());
      funcName2CallOps[newFuncName].push_back(call.getOperation());
    } else {
      calleeFuncOps.push_back(calleeFunc.getOperation());
      funcName2CallOps[calleeFuncName].push_back(call.getOperation());
    }
  });

  for (unsigned int i = 0; i < calleeFuncOps.size(); i++) {
    flattenFuncOp(calleeFuncOps[i], funcName2CallOps);
  }
}
