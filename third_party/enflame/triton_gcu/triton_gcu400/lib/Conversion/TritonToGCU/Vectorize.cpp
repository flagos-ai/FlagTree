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

#include <string>

#include "Dialect/GCU/IR/Dialect.h"
#include "Utility.h"
#include "Vectorize.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir {
namespace triton {
namespace gcu {

void VectorizeHandlerInterface::finalizeMappedResults(
    const VectorizeContext &context, Operation *originalOp,
    Operation *clonedOp) const {
  SmallVector<Type> resultTypes;
  auto typeInterface = dyn_cast<InferTypeOpInterface>(clonedOp);
  if (typeInterface &&
      succeeded(typeInterface.inferReturnTypes(
          clonedOp->getContext(), clonedOp->getLoc(), clonedOp->getOperands(),
          clonedOp->getAttrDictionary(), clonedOp->getPropertiesStorage(),
          clonedOp->getRegions(), resultTypes))) {
    assert(resultTypes.size() == originalOp->getNumResults() &&
           "number of result types must match number of original op results");
    for (unsigned i = 0; i < resultTypes.size(); ++i) {
      clonedOp->getResult(i).setType(resultTypes[i]);
      context.mapper.map(originalOp->getResult(i), clonedOp->getResult(i));
    }
  } else {
    for (auto [result, newResult] :
         llvm::zip_equal(originalOp->getResults(), clonedOp->getResults())) {
      auto vectorTy = VectorType::get(ArrayRef<int64_t>{context.vectorLength},
                                      getElementTypeOrSelf(result.getType()));
      newResult.setType(vectorTy);
      context.mapper.map(result, newResult);
    }
  }
}

Operation *SelectVectorizeHandler::rewriteOp(const VectorizeContext &context,
                                             Operation *op) const {
  auto selectOp = cast<arith::SelectOp>(op);
  auto condition = selectOp.getCondition();
  auto mapValue = context.mapper.lookup(condition);
  assert(mapValue && "condition must be mapped before rewrite");
  if (getElementTypeOrSelf(mapValue.getType()).isInteger(8)) {
    context.mapper.map(
        condition,
        context.builder
            .create<mlir::gcu::VectorConvertOp>(
                context.loc,
                VectorType::get(ArrayRef<int64_t>{context.vectorLength},
                                context.builder.getIntegerType(1)),
                mapValue)
            .getResult(0));
  }
  auto clonedOp = context.builder.clone(*op, context.mapper);
  context.mapper.map(condition, mapValue);
  return clonedOp;
}

Operation *
ExternElementwiseVectorizeHandler::rewriteOp(const VectorizeContext &context,
                                             Operation *op) const {
  auto externElementwiseOp = cast<triton::ExternElementwiseOp>(op);
  auto operands = llvm::to_vector(
      llvm::map_range(externElementwiseOp.getOperands(), [&](auto operand) {
        return context.mapper.lookup(operand);
      }));
  auto symbol = externElementwiseOp.getSymbol();
  auto resultTy =
      VectorType::get(ArrayRef<int64_t>{context.vectorLength},
                      getElementTypeOrSelf(externElementwiseOp.getResult()));
  Operation *rewriteOp;
  if (mlir::triton::gcu::isNvLibDeviceSymbol(symbol)) {
    std::string efSymbol = "__ef_v";
    efSymbol += symbol.drop_front(strlen("__nv_"));
    rewriteOp = context.builder.create<mlir::gcu::ExternElementwiseOp>(
        context.loc, resultTy, operands, efSymbol);
  } else if (mlir::triton::gcu::isMixedPrecisionSymbol(symbol)) {
    rewriteOp = context.builder.create<mlir::gcu::ExternElementwiseOp>(
        context.loc, resultTy, operands, symbol);
  } else {
    llvm_unreachable(
        ("unsupported extern elementwise: " + symbol).str().c_str());
  }
  return rewriteOp;
}

void ExternElementwiseVectorizeHandler::finalizeMappedResults(
    const VectorizeContext &context, Operation *originalOp,
    Operation *clonedOp) const {
  context.mapper.map(originalOp->getResults(), clonedOp->getResults());
}

VectorizeEngine::VectorizeEngine() {
  addHandler(std::make_unique<DefaultVectorizeHandler>());
}

Operation *VectorizeEngine::vectorize(Operation *op,
                                      const VectorizeContext &context) const {
  for (const auto &handler : llvm::reverse(handlers)) {
    if (!handler->match(op)) {
      continue;
    }
    auto rewriteOp = handler->rewriteOp(context, op);
    handler->finalizeMappedResults(context, op, rewriteOp);
    return rewriteOp;
  }
  assert(false && "no vectorize handler matched");
  return nullptr;
}

} // namespace gcu
} // namespace triton
} // namespace mlir
