/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
 */

#ifndef TRITON_TO_GRAPH_INDEPENDENT_ROW_REDUCTION_ANALYSIS_H
#define TRITON_TO_GRAPH_INDEPENDENT_ROW_REDUCTION_ANALYSIS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <utility>

namespace mlir {
namespace triton {
namespace cfg {

enum class IndependentRowReductionKind : uint8_t {
  Other,
  SingleMoment,
  TwoMoment,
};

inline bool isIndependentRowReduction(ReduceOp reduce) {
  if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1)
    return false;
  auto source = dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
  if (!source || !source.hasStaticShape())
    return false;
  Type result = reduce.getResults().front().getType();
  if (source.getRank() == 1 && reduce.getAxis() == 0)
    return !isa<RankedTensorType>(result);
  auto rankedResult = dyn_cast<RankedTensorType>(result);
  return source.getRank() == 2 && reduce.getAxis() == 1 && rankedResult &&
         rankedResult.hasStaticShape() && rankedResult.getRank() == 1 &&
         source.getShape()[0] == rankedResult.getShape()[0];
}

inline bool isMergeStyleReduction(ReduceOp reduce) {
  if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1)
    return false;
  auto source = dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
  auto result =
      dyn_cast<RankedTensorType>(reduce.getResults().front().getType());
  return source && result && source.hasStaticShape() &&
         result.hasStaticShape() && source.getRank() == 2 &&
         result.getRank() == 1 && reduce.getAxis() == 0 &&
         source.getShape()[1] == result.getShape()[0];
}

inline Value stripIdentityReshapes(Value value) {
  while (auto reshape = value.getDefiningOp<triton::ReshapeOp>()) {
    Value source = reshape.getOperand();
    if (source.getType() != value.getType())
      break;
    value = source;
  }
  return value;
}

inline Value getSquareOperand(Value value) {
  value = stripIdentityReshapes(value);
  if (auto multiply = value.getDefiningOp<arith::MulFOp>()) {
    Value lhs = stripIdentityReshapes(multiply.getLhs());
    Value rhs = stripIdentityReshapes(multiply.getRhs());
    return lhs == rhs ? lhs : Value();
  }
  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    Value lhs = stripIdentityReshapes(multiply.getLhs());
    Value rhs = stripIdentityReshapes(multiply.getRhs());
    return lhs == rhs ? lhs : Value();
  }
  return Value();
}

inline bool isMomentDataflowCarrier(Operation *operation) {
  return operation->hasTrait<OpTrait::Elementwise>() ||
         isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
             triton::ReshapeOp>(operation);
}

inline void collectMomentDataflow(Value root, llvm::DenseSet<Value> &values) {
  SmallVector<Value> worklist;
  values.insert(root);
  worklist.push_back(root);
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (!isMomentDataflowCarrier(user))
        continue;
      for (Value result : user->getResults())
        if (values.insert(result).second)
          worklist.push_back(result);
    }
  }
}

inline bool hasMomentDownstreamUse(Value reductionResult) {
  llvm::DenseSet<Value> flow;
  collectMomentDataflow(reductionResult, flow);
  return flow.size() > 1;
}

inline bool hasJointMomentDownstreamUse(triton::FuncOp function,
                                        Value directResult,
                                        Value squareResult) {
  llvm::DenseSet<Value> directFlow;
  llvm::DenseSet<Value> squareFlow;
  collectMomentDataflow(directResult, directFlow);
  collectMomentDataflow(squareResult, squareFlow);
  bool joined = false;
  function.walk([&](Operation *operation) {
    if (joined || !isMomentDataflowCarrier(operation))
      return;
    bool consumesDirect = false;
    bool consumesSquare = false;
    for (Value operand : operation->getOperands()) {
      consumesDirect |= directFlow.contains(operand);
      consumesSquare |= squareFlow.contains(operand);
    }
    joined = consumesDirect && consumesSquare;
  });
  return joined;
}

inline IndependentRowReductionKind
classifyIndependentRowReduction(triton::FuncOp function) {
  SmallVector<triton::ReduceOp> reductions;
  bool hasMergeStyleReduction = false;
  function.walk([&](triton::ReduceOp reduce) {
    hasMergeStyleReduction |= isMergeStyleReduction(reduce);
    if (isIndependentRowReduction(reduce))
      reductions.push_back(reduce);
  });
  if (hasMergeStyleReduction || reductions.empty())
    return IndependentRowReductionKind::Other;

  SmallVector<std::pair<triton::ReduceOp, Value>> squareReductions;
  SmallVector<triton::ReduceOp> directReductions;
  for (triton::ReduceOp reduce : reductions) {
    Value squareOperand = getSquareOperand(reduce.getSrcs().front());
    if (squareOperand)
      squareReductions.push_back({reduce, squareOperand});
    else
      directReductions.push_back(reduce);
  }
  if (reductions.size() == 1)
    return squareReductions.size() == 1 &&
                   hasMomentDownstreamUse(
                       squareReductions.front().first.getResults().front())
               ? IndependentRowReductionKind::SingleMoment
               : IndependentRowReductionKind::Other;

  if (reductions.size() != 2 || squareReductions.size() != 1 ||
      directReductions.size() != 1)
    return IndependentRowReductionKind::Other;
  return stripIdentityReshapes(directReductions.front().getSrcs().front()) ==
                     squareReductions.front().second &&
                 hasMomentDownstreamUse(
                     directReductions.front().getResults().front()) &&
                 hasMomentDownstreamUse(
                     squareReductions.front().first.getResults().front()) &&
                 hasJointMomentDownstreamUse(
                     function, directReductions.front().getResults().front(),
                     squareReductions.front().first.getResults().front())
             ? IndependentRowReductionKind::TwoMoment
             : IndependentRowReductionKind::Other;
}

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif
