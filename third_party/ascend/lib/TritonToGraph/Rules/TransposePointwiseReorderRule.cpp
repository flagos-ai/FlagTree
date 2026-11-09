/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/GraphOptimizationRule.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

struct MatchedChain {
  triton::DotOp dot;
  unsigned operandIndex;
  triton::TransOp trans;
  SmallVector<Operation *> unaryOps;
};

enum class UnaryMoveKind { ElementTypeCast, TypePreserving };

bool hasSingleResultAndOperand(Operation *operation, Value expectedResult) {
  return operation && operation->getNumOperands() == 1 &&
         operation->getNumResults() == 1 &&
         operation->getResult(0) == expectedResult &&
         operation->getNumRegions() == 0;
}

bool hasOnlyExpectedUse(Value value, Operation *expectedUser) {
  if (!value || !expectedUser || !value.hasOneUse())
    return false;
  return value.use_begin()->getOwner() == expectedUser;
}

std::optional<UnaryMoveKind> classifyTrustedUnary(Operation *operation) {
  if (!operation)
    return std::nullopt;

  if (isa<arith::TruncFOp, arith::ExtFOp, arith::BitcastOp>(operation))
    return UnaryMoveKind::ElementTypeCast;

  if (operation->hasTrait<OpTrait::Elementwise>())
    return UnaryMoveKind::TypePreserving;

  return std::nullopt;
}

bool hasCompatibleUnaryTensorTypes(Operation *operation, UnaryMoveKind kind) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1)
    return false;

  auto operandType =
      dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  auto resultType =
      dyn_cast<RankedTensorType>(operation->getResult(0).getType());
  if (!operandType || !resultType || !operandType.hasStaticShape() ||
      !resultType.hasStaticShape())
    return false;

  if (operandType.getShape() != resultType.getShape() ||
      operandType.getEncoding() != resultType.getEncoding())
    return false;

  return kind != UnaryMoveKind::TypePreserving || operandType == resultType;
}

bool isTrustedUnary(Operation *operation, Value expectedResult,
                    UnaryMoveKind kind) {
  return hasSingleResultAndOperand(operation, expectedResult) &&
         isMemoryEffectFree(operation) &&
         hasCompatibleUnaryTensorTypes(operation, kind);
}

bool hasValidTransposeOrder(triton::TransOp trans,
                            RankedTensorType sourceType) {
  if (!trans.getOrderAttr())
    return false;

  ArrayRef<int32_t> order = trans.getOrder();
  if (order.empty() || order.size() != sourceType.getRank())
    return false;

  SmallVector<unsigned char> seen(order.size(), 0);
  for (int32_t axis : order) {
    if (axis < 0 || static_cast<size_t>(axis) >= order.size() || seen[axis])
      return false;
    seen[axis] = true;
  }
  return true;
}

bool inferTransposeReturnTypes(triton::TransOp trans, Value input,
                               SmallVectorImpl<Type> &inferredTypes) {
  if (!trans || !input || !isa<triton::gpu::TensorOrMemDesc>(input.getType()))
    return false;

  auto rankedInputType = dyn_cast<RankedTensorType>(input.getType());
  if (!rankedInputType || !hasValidTransposeOrder(trans, rankedInputType))
    return false;

  auto inputType = cast<triton::gpu::TensorOrMemDesc>(input.getType());
  if (Attribute encoding = inputType.getEncoding()) {
    Attribute probeEncoding;
    auto *interface =
        encoding.getDialect()
            .getRegisteredInterface<triton::DialectInferLayoutInterface>();
    if (!interface || failed(interface->inferTransOpEncoding(
                          encoding, rankedInputType.getShape(),
                          trans.getOrder(), probeEncoding, trans.getLoc())))
      return false;
  }

  return succeeded(triton::TransOp::inferReturnTypes(
             trans->getContext(), trans.getLoc(), ValueRange{input},
             trans->getAttrDictionary(), trans->getPropertiesStorage(),
             trans->getRegions(), inferredTypes)) &&
         inferredTypes.size() == 1;
}

std::optional<MatchedChain> matchCandidate(triton::DotOp dot,
                                           unsigned operandIndex) {
  if (operandIndex > 1 || !dot || dot->getNumRegions() != 0)
    return std::nullopt;

  // Deliberately select only A or B.  The accumulator C is never read by
  // this rule.
  Value value = operandIndex == 0 ? dot.getA() : dot.getB();
  Block *block = dot->getBlock();
  if (!value || !block)
    return std::nullopt;

  Operation *expectedUser = dot.getOperation();
  SmallVector<Operation *> reverseUnaryOps;
  while (Operation *operation = value.getDefiningOp()) {
    std::optional<UnaryMoveKind> kind = classifyTrustedUnary(operation);
    if (!kind)
      break;

    if (operation->getBlock() != block ||
        !hasOnlyExpectedUse(value, expectedUser) ||
        !isTrustedUnary(operation, value, *kind))
      return std::nullopt;

    reverseUnaryOps.push_back(operation);
    value = operation->getOperand(0);
    expectedUser = operation;
  }

  if (reverseUnaryOps.empty())
    return std::nullopt;

  triton::TransOp trans = value.getDefiningOp<triton::TransOp>();
  if (!trans || trans->getBlock() != block ||
      !hasSingleResultAndOperand(trans, value) ||
      !hasOnlyExpectedUse(value, expectedUser))
    return std::nullopt;

  auto sourceType = dyn_cast<RankedTensorType>(trans.getSrc().getType());
  auto resultType = dyn_cast<RankedTensorType>(trans.getResult().getType());
  if (!sourceType || !resultType || !sourceType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      !hasValidTransposeOrder(trans, sourceType))
    return std::nullopt;

  SmallVector<Type, 1> originalInferredTypes;
  if (!inferTransposeReturnTypes(trans, trans.getSrc(),
                                 originalInferredTypes) ||
      originalInferredTypes.front() != trans.getResult().getType())
    return std::nullopt;

  SmallVector<Operation *> unaryOps(reverseUnaryOps.rbegin(),
                                    reverseUnaryOps.rend());
  return MatchedChain{dot, operandIndex, trans, std::move(unaryOps)};
}

std::optional<RankedTensorType> getMovedUnaryResultType(Operation *unary,
                                                        Value input) {
  if (!unary || unary->getNumResults() != 1)
    return std::nullopt;

  std::optional<UnaryMoveKind> kind = classifyTrustedUnary(unary);
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto oldResultType =
      dyn_cast<RankedTensorType>(unary->getResult(0).getType());
  if (!kind || !inputType || !oldResultType || !inputType.hasStaticShape() ||
      !oldResultType.hasStaticShape())
    return std::nullopt;

  if (*kind == UnaryMoveKind::TypePreserving)
    return inputType;
  return inputType.clone(oldResultType.getElementType());
}

Operation *createMovedUnary(IRRewriter &rewriter, Operation *unary, Value input,
                            RankedTensorType resultType) {
  IRMapping mapping;
  mapping.map(unary->getOperand(0), input);
  Operation *replacement = rewriter.clone(*unary, mapping);
  replacement->getResult(0).setType(resultType);
  return replacement;
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created))
    rewriter.eraseOp(operation);
}

class TransposePointwiseReorderPlan final : public RewritePlan {
public:
  TransposePointwiseReorderPlan(MatchedChain chain, unsigned epoch)
      : dot(chain.dot), dotOperation(chain.dot.getOperation()),
        operandIndex(chain.operandIndex), trans(chain.trans),
        transOperation(chain.trans.getOperation()),
        unaryOps(std::move(chain.unaryOps)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::TransposePointwiseReorder;
  }

  unsigned getBenefit() const override {
    return static_cast<unsigned>(unaryOps.size());
  }

  Operation *getAnchor() const override { return dotOperation; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !dot)
      return failure();

    triton::FuncOp function = dot->getParentOfType<triton::FuncOp>();
    if (!function ||
        function.getOperation() != context.getFunction().getOperation())
      return failure();

    std::optional<MatchedChain> current = matchCandidate(dot, operandIndex);
    if (!current || current->trans.getOperation() != transOperation ||
        current->unaryOps.size() != unaryOps.size())
      return failure();

    for (auto [currentUnary, storedUnary] :
         llvm::zip(current->unaryOps, unaryOps)) {
      if (currentUnary != storedUnary)
        return failure();
    }
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    // Apply defensively revalidates as well: a plan must never depend on a
    // stale operation/use graph even if a caller accidentally skips the
    // driver's revalidation gate.
    Block *block = dot ? dot->getBlock() : nullptr;
    std::optional<MatchedChain> current = matchCandidate(dot, operandIndex);
    if (!block || !current || current->trans.getOperation() != transOperation ||
        current->unaryOps.size() != unaryOps.size())
      return failure();
    for (auto [currentUnary, storedUnary] :
         llvm::zip(current->unaryOps, unaryOps)) {
      if (currentUnary != storedUnary)
        return failure();
    }

    Value originalDotOperand = operandIndex == 0 ? dot.getA() : dot.getB();
    rewriter.setInsertionPoint(trans);
    Value currentValue = trans.getSrc();
    SmallVector<Operation *> created;
    for (Operation *unary : unaryOps) {
      std::optional<RankedTensorType> movedResultType =
          getMovedUnaryResultType(unary, currentValue);
      if (!movedResultType) {
        eraseCreatedOperations(rewriter, created);
        return failure();
      }

      Operation *replacement =
          createMovedUnary(rewriter, unary, currentValue, *movedResultType);
      if (!replacement) {
        eraseCreatedOperations(rewriter, created);
        return failure();
      }
      created.push_back(replacement);
      currentValue = replacement->getResult(0);
    }

    // Preflight the inferred result before calling the generated builder. In
    // particular, an unsupported operand encoding must reject the plan rather
    // than reaching a builder that may assert while inferring its result.
    SmallVector<Type, 1> inferredTransposeTypes;
    if (!inferTransposeReturnTypes(trans, currentValue,
                                   inferredTransposeTypes) ||
        inferredTransposeTypes.front() != originalDotOperand.getType()) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    triton::TransOp replacementTrans = rewriter.create<triton::TransOp>(
        trans.getLoc(), currentValue, trans.getOrderAttr());
    replacementTrans->setAttrs(trans->getAttrs());
    created.push_back(replacementTrans.getOperation());

    // Nothing original is changed until every replacement has passed its own
    // verifier and the inferred transpose result exactly matches dot A or B.
    for (Operation *operation : created) {
      if (failed(mlir::verify(operation))) {
        eraseCreatedOperations(rewriter, created);
        return failure();
      }
    }
    if (replacementTrans.getResult().getType() !=
        originalDotOperand.getType()) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }

    rewriter.modifyOpInPlace(dot.getOperation(), [&] {
      dot->setOperand(operandIndex, replacementTrans.getResult());
    });

    for (Operation *unary : llvm::reverse(unaryOps))
      rewriter.eraseOp(unary);
    rewriter.eraseOp(trans);
    return success();
  }

private:
  triton::DotOp dot;
  Operation *dotOperation;
  unsigned operandIndex;
  triton::TransOp trans;
  Operation *transOperation;
  SmallVector<Operation *> unaryOps;
  unsigned epoch;
};

class TransposePointwiseReorderRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::TransposePointwiseReorder;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](triton::DotOp dot) {
      for (unsigned operandIndex : {0u, 1u}) {
        std::optional<MatchedChain> chain = matchCandidate(dot, operandIndex);
        if (chain) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[" DEBUG_TYPE "] matched graph optimization rule "
                     << static_cast<unsigned>(getId()) << " ("
                     << getGraphOptimizationRuleName(getId()) << ") at "
                     << dot.getLoc() << ": operand=" << operandIndex
                     << " pointwiseOps=" << chain->unaryOps.size() << "\n");
          plans.push_back(std::make_unique<TransposePointwiseReorderPlan>(
              std::move(*chain), context.getEpoch()));
        }
      }
    });
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule>
cfg::createTransposePointwiseReorderRule() {
  return std::make_unique<TransposePointwiseReorderRule>();
}
