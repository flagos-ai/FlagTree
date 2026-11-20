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

#include <cstddef>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "SplitMatmul";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

constexpr llvm::StringLiteral kMatmulAtLeastOnceHint = "matmul_at_least_once";

namespace {

struct MatmulInputs {
  Value a;
  Value b;
  Value bias;
};

struct SplitInfo {
  bool mayNotExec;
  Value outerInValue;
  Value outerOutValue;
  bool shouldSplit;
  bool supported = false;
};

} // namespace

static inline MatmulInputs parseMatmulInputs(linalg::MatmulOp matmulOp) {
  auto inits = matmulOp.getDpsInits();
  auto inputs = matmulOp.getDpsInputs();

  return {inputs[0], inputs[1], inits[0]};
}

// The user is responsible for checking biasDefOp is not null.
static bool operationIsFillZero(Operation *op) {
  auto fillOp = dyn_cast<linalg::FillOp>(op);
  if (!fillOp) {
    return false;
  }
  auto filledVal = fillOp.getInputs()[0];
  return matchPattern(filledVal, m_Zero()) ||
         matchPattern(filledVal, m_AnyZeroFloat());
}

// This generally should always be true, but just for safety...
static bool isFloatOrInt(RankedTensorType tensorType) {
  auto elmType = tensorType.getElementType();
  return isa<FloatType, IntegerType>(elmType);
}

static bool scfMayNotExec(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case([&](scf::ForOp forOp) {
        IntegerAttr ubAttr;
        IntegerAttr lbAttr;
        if (matchPattern(forOp.getUpperBound(), m_Constant(&ubAttr)) &&
            matchPattern(forOp.getLowerBound(), m_Constant(&lbAttr))) {
          return ubAttr.getValue().sle(lbAttr.getValue());
        }
        return true;
      })
      .Case([&](scf::IfOp ifOp) {
        return !matchPattern(ifOp.getCondition(), m_One());
      })
      .Default([&](auto) { return false; });
}

/**
 * Search outward from args to obtain the following information:
 * 1. Whether args is only used and updated by matmul: bool argsLimitedInMatmul
 * 2. Whether matmul is guaranteed to execute: bool mayNotExec
 * 3. Outermost initial value: Value outerInValue
 * 4. Outermost for or if, i.e., where to insert if/else.
 */
static Value searchInArgsChain(Value nextValueOfC, bool &argsLimitedInMatmul,
                               bool &mayNotExec, Value &outerInValue) {
  auto op = nextValueOfC.getDefiningOp();
  auto parentOp = op->getParentOp();
  if (outerInValue.getDefiningOp()) {
    return nextValueOfC;
  }

  // update argsLimitedInMatmul
  Value nextSearchValue = nextValueOfC;
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    auto blockArg = dyn_cast_if_present<BlockArgument>(outerInValue);
    if (!blockArg || blockArg.getOwner() != forOp.getBody()) {
      argsLimitedInMatmul = false;
      return nextValueOfC;
    }
    int argIdx = blockArg.getArgNumber() - 1;
    for (auto &use : blockArg.getUses()) {
      auto user = use.getOwner();
      // Allowed: the op itself (mmad or inner for/if that chains to mmad).
      auto userInBlock = CVPipeline::getAncestorInBlock(user, op->getBlock());
      if (userInBlock == op) {
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(userInBlock)) {
        argsLimitedInMatmul = (use.getOperandNumber() == argIdx);
      } else {
        argsLimitedInMatmul = false;
        break;
      }
    }
    // check res is only used by yield
    for (auto &use : nextValueOfC.getUses()) {
      auto user = use.getOwner();
      auto userInBlock = CVPipeline::getAncestorInBlock(user, op->getBlock());
      auto yieldOp = dyn_cast<scf::YieldOp>(userInBlock);
      if (!yieldOp) {
        argsLimitedInMatmul = false;
        break;
      } else if (yieldOp->getBlock() != forOp.getBody()) {
        argsLimitedInMatmul = false;
        break;
      } else if (use.getOperandNumber() != argIdx) {
        argsLimitedInMatmul = false;
        break;
      }
    }

    if (argsLimitedInMatmul) {
      outerInValue = forOp.getInitArgs()[argIdx];
      nextSearchValue = forOp->getResult(argIdx);
    }
  } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
    if (!ifOp.elseBlock()) {
      argsLimitedInMatmul = false;
    }
    auto otherYieldOp =
        op->getBlock() == ifOp.thenBlock()
            ? cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator())
            : cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto opYieldOp =
        op->getBlock() == ifOp.thenBlock()
            ? cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator())
            : cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    int resultIdx = -1;
    for (unsigned i = 0; i < otherYieldOp->getNumOperands(); ++i) {
      if (otherYieldOp->getOperand(i) == outerInValue &&
          opYieldOp->getOperand(i) == nextSearchValue) {
        resultIdx = i;
        break;
      }
    }
    if (resultIdx == -1) {
      argsLimitedInMatmul = false;
    } else {
      nextSearchValue = ifOp.getResult(resultIdx);
    }
  } else {
    argsLimitedInMatmul = false;
    LOG_DEBUG("WARN: no for/if out to matmul.");
  }

  if (!argsLimitedInMatmul) {
    return nextValueOfC;
  }
  // update mayNotExec
  mayNotExec = mayNotExec || scfMayNotExec(parentOp);
  return searchInArgsChain(nextSearchValue, argsLimitedInMatmul, mayNotExec,
                           outerInValue);
}

/**
 * Traces the chain of users of a given value to find a matching operation.
 * This function recursively follows the use-def chain through various operation
 * types (ViewLikeOpInterface, scf::ForOp, scf::YieldOp) to locate an operation
 * that matches the specified criteria.
 *
 * Parameters:
 *   value        - The starting value whose users are to be traced.
 *   needSingle   - If true, requires that the value has exactly one user in its
 * parent block. If multiple users exist, returns std::nullopt. isMatchedOp  - A
 * predicate function that determines if an operation is the target match.
 *                  Returns true if the operation matches the desired criteria.
 *   isSkipOp     - A predicate function that determines if an operation should
 * be skipped. If true and the operation has a single result, continues tracing
 * through that result.
 *
 * Returns:
 *   The matched operation if found, or std::nullopt if no match is found or if
 *   the single-user requirement is violated.
 */
std::optional<Operation *>
traceChainUser(Value value, bool needSingle,
               const std::function<bool(Operation *, Value v)> &isMatchedOp,
               const std::function<bool(Operation *, Value v)> &isSkipOp) {
  Operation *nextUser = nullptr;
  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto userInblock =
        CVPipeline::getAncestorInBlock(user, value.getParentBlock());
    if (nextUser && userInblock != nextUser && needSingle) {
      return std::nullopt; // not single user
    } else if (!nextUser && userInblock) {
      nextUser = userInblock;
    }
  }

  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (llvm::isa<ViewLikeOpInterface, tensor::ExtractSliceOp>(user)) {
      return traceChainUser(user->getResult(0), needSingle, isMatchedOp,
                            isSkipOp);
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      auto initArgs = forOp.getInitArgs();
      int initIndx = -1;
      int useCnt = 0;
      for (auto [i, arg] : llvm::enumerate(initArgs)) {
        if (arg == value) {
          initIndx = i;
          useCnt++;
        }
      }
      if (useCnt == 1) {
        return traceChainUser(forOp.getRegionIterArgs()[initIndx], needSingle,
                              isMatchedOp, isSkipOp);
      }
      return std::nullopt;
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto operands = yieldOp.getOperands();
      auto it = llvm::find(operands, value);
      if (it == operands.end()) {
        return std::nullopt;
      }
      auto index = std::distance(operands.begin(), it);

      Operation *parentOp = yieldOp->getParentOp();
      if (!parentOp ||
          static_cast<size_t>(index) >= parentOp->getNumResults()) {
        return std::nullopt;
      }

      return traceChainUser(parentOp->getResult(index), needSingle, isMatchedOp,
                            isSkipOp);
    }

    if (isMatchedOp(user, value)) {
      return user;
    }
    if (isSkipOp(user, value)) {
      if (user->getNumResults() == 1) {
        return traceChainUser(user->getResult(0), needSingle, isMatchedOp,
                              isSkipOp);
      }
      return std::nullopt;
    }
  }
  return std::nullopt;
}

static bool hasCrossBlockCascadeL0CConsumer(Value result,
                                            linalg::MatmulOp matmulOp) {
  auto matchMatmulC = [](Operation *op, Value value) {
    if (auto nextMatmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      auto inputs = parseMatmulInputs(nextMatmulOp);
      return inputs.a != value && inputs.b != value && inputs.bias == value;
    }
    return false;
  };
  auto usedByL0C =
      traceChainUser(result, true, matchMatmulC,
                     [](Operation *op, Value value) { return false; });

  // No cascade consumer, or the cascade consumer is in the same block as
  // `result` (i.e. inside the current for-loop body).
  if (!usedByL0C.has_value() ||
      usedByL0C.value()->getBlock() == result.getParentBlock()) {
    return false;
  }

  // Only support the cascade for in the same block
  auto cascadeMatmul = llvm::dyn_cast<linalg::MatmulOp>(usedByL0C.value());
  auto cascadeForOp = cascadeMatmul
                          ? llvm::dyn_cast_if_present<scf::ForOp>(
                                cascadeMatmul->getBlock()->getParentOp())
                          : nullptr;
  auto currentForOp = llvm::dyn_cast_if_present<scf::ForOp>(
      matmulOp->getBlock()->getParentOp());
  if (cascadeForOp && currentForOp &&
      cascadeForOp->getBlock() != currentForOp->getBlock()) {
    return false;
  }
  currentForOp->setAttr(CVPipeline::kMayNotExecNPU,
                        BoolAttr::get(matmulOp.getContext(), true));
  return true;
}

/**
 * Checks whether `initVal` is defined by a linalg::MatmulOp (L0C -> L0C
 * cascade producer) that lives in a different block than `matmulOp`'s
 * for-loop. Such a cascade scenario should be split.
 */
static bool hasCrossBlockCascadeL0CProducer(Value initVal,
                                            linalg::MatmulOp matmulOp) {
  auto defMatmul = dyn_cast_if_present<linalg::MatmulOp>(
      hivm::traceDefOp<linalg::MatmulOp>(initVal).value_or(nullptr));
  if (!defMatmul) {
    return false;
  }
  // Cascade producer must be outside `matmulOp`'s block to be considered
  // a cascade scenario at all.
  auto defInMatmulBlock =
      CVPipeline::getAncestorInBlock(defMatmul, matmulOp->getBlock());
  if (defInMatmulBlock) {
    return false;
  }
  // Only support the cascade for in the same block
  auto cascadeForOp = llvm::dyn_cast_if_present<scf::ForOp>(
      defMatmul->getBlock()->getParentOp());
  auto currentForOp = llvm::dyn_cast_if_present<scf::ForOp>(
      matmulOp->getBlock()->getParentOp());
  if (cascadeForOp && currentForOp &&
      cascadeForOp->getBlock() != currentForOp->getBlock()) {
    return false;
  }
  return true;
}

static bool isSubviewFromGlobalMemory(ViewLikeOpInterface viewOp) {
  // Subview ops may be nested many layers deep through reinterpretation or
  // other subviews. like, subview (subview (reinterpret_cast (subview
  // (reinterpret_cast (arg0))))) so we need Search and only keep same block
  // view-like op.
  if (!viewOp) {
    return false;
  }
  Value source = viewOp.getViewSource();
  while (true) {
    LOG_DEBUG("Check view source: " << source << "\n");
    if (auto blockArg = dyn_cast<BlockArgument>(source)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parentOp)) {
        return true;
      } else if (isa<LoopLikeOpInterface>(parentOp)) {
        auto argIndex = blockArg.getArgNumber();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
          if (argIndex > 0) {
            source = forOp.getInitArgs()[argIndex - 1];
          }
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
          source = whileOp.getInits()[argIndex];
        } else {
          LOG_DEBUG("Subview source block argument is from unknow block.");
          return false;
        }
      }
    }
    // From other view-like op
    if (auto viewLike = dyn_cast<ViewLikeOpInterface>(source.getDefiningOp())) {
      source = viewLike.getViewSource();
      continue;
    }
    LOG_DEBUG(
        "Subview source defining op is not ViewLikeOpInterface: " << source);
    return false;
  }
  return false;
}

/**
 * Determines whether a matmul operation should be split based on its output
 * usage. A split is needed when the matmul output is used in ways that would
 * cause hardware issues:
 * 1. (Specific scenarios Filter) Output is used by a nested matmul (L0C -> L0C)
 * which causes NPUIR fixpipe errors.
 * 2. Output is used as input A or B of another matmul (L0C -> L1), which
 * requires materialization to global memory.
 *
 * Parameters:
 *   matmulOp     - The matmul operation to check.
 *   outerOutValue   - The outermost result value in the use-def chain.
 *   outerInValue - The outermost initial value (bias) in the use-def chain.
 *
 * Returns:
 *   true if the matmul should be split due to output usage, false otherwise.
 */
static bool isOutputFilter(linalg::MatmulOp matmulOp, Value &outerOutValue,
                           Value &outerInValue) {
  // Specific scenario: used by store and nextUser is nested
  auto matchStoreGm = [](Operation *op, Value value) {
    if (auto nextStoreGM =
            dyn_cast<bufferization::MaterializeInDestinationOp>(op)) {
      Value dest = nextStoreGM.getDest();
      auto viewOp = dest.getDefiningOp<ViewLikeOpInterface>();
      return isSubviewFromGlobalMemory(viewOp);
    } else if (auto hivmStore = dyn_cast<hivm::StoreOp>(op)) {
      auto dest = hivmStore.getDst();
      auto viewOp = dest.getDefiningOp<ViewLikeOpInterface>();
      return isSubviewFromGlobalMemory(viewOp);
    }
    return false;
  };
  auto storeToGM =
      traceChainUser(outerOutValue, false, matchStoreGm,
                     [](Operation *op, Value value) { return false; });
  if (storeToGM.has_value() &&
      storeToGM.value()->getBlock() != outerOutValue.getParentBlock()) {
    LOG_DEBUG("(store) Split because avoiding NPUIR insert fixpipe errors. "
              << matmulOp);
    return true;
  }

  // Specific scenario: used by single L0C and nextUser is nested
  auto matchMatmulC = [](Operation *op, Value value) {
    if (auto nextMatmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      auto inputs = parseMatmulInputs(nextMatmulOp);
      return inputs.a != value && inputs.b != value && inputs.bias == value;
    }
    return false;
  };
  auto usedByL0C =
      traceChainUser(outerOutValue, true, matchMatmulC,
                     [](Operation *op, Value value) { return false; });
  if (usedByL0C.has_value() &&
      usedByL0C.value()->getBlock() != outerOutValue.getParentBlock()) {
    LOG_DEBUG("(first) Split because avoiding NPUIR insert fixpipe errors. "
              << matmulOp);
    return true;
  }
  return false;
}
static bool shouldSplitByOutput(linalg::MatmulOp matmulOp, Value &outerOutValue,
                                Value &outerInValue) {
  if (isOutputFilter(matmulOp, outerOutValue, outerInValue)) {
    return true;
  }
  // used by any L1
  auto matchMatmulAB = [](Operation *op, Value value) {
    if (auto nextMatmulOp = dyn_cast<linalg::MatmulOp>(op)) {
      auto inputs = parseMatmulInputs(nextMatmulOp);
      return inputs.a == value || inputs.b == value;
    }
    return false;
  };
  auto skipCubeop = [=](Operation *op, Value value) {
    return isa<linalg::TransposeOp>(op);
  };
  auto usedByL1 =
      traceChainUser(outerOutValue, false, matchMatmulAB, skipCubeop);
  if (usedByL1.has_value() &&
      usedByL1.value()->getBlock() != outerOutValue.getParentBlock()) {
    LOG_DEBUG("Split avoid L0C -> L1. " << matmulOp); // S01-S08
    return true;
  }

  return false;
}

/**
 * Determines whether a matmul operation should be split based on its input
 * (bias) source. The split is avoided when:
 * 1. The bias is filled with zeros (no need to split).
 * 2. The bias comes from another matmul output (L0C remaining in the chain).
 *
 * Parameters:
 *   matmulOp     - The matmul operation to check.
 *   outerOutValue   - The outermost result value in the use-def chain.
 *   outerInValue - The outermost initial value (bias) in the use-def chain.
 *
 * Returns:
 *   true if the matmul should be split due to input source, false otherwise.
 */
static bool isInputFilter(linalg::MatmulOp matmulOp, Value &outerOutValue,
                          Value &outerInValue) {
  // Specific scenario: outerInValue from L0C and nextUser is nested
  auto defMatmul = dyn_cast_if_present<linalg::MatmulOp>(
      hivm::traceDefOp<linalg::MatmulOp>(outerInValue).value_or(nullptr));
  if (defMatmul) {
    auto defInMatmulBlock =
        CVPipeline::getAncestorInBlock(defMatmul, matmulOp->getBlock());
    if (!defInMatmulBlock) {
      LOG_DEBUG("(Second) Split because avoiding NPUIR insert fixpipe errors. "
                << matmulOp);
      return true;
    }
  }
  return false;
}
static bool shouldSplitByInput(linalg::MatmulOp matmulOp, Value &outerOutValue,
                               Value &outerInValue) {
  if (isInputFilter(matmulOp, outerOutValue, outerInValue)) {
    return true;
  }
  auto outerInOp = outerInValue.getDefiningOp();
  if (!outerInOp) {
    return true;
  }

  if (operationIsFillZero(outerInValue.getDefiningOp())) {
    LOG_DEBUG("Not split because bias is zero. " << matmulOp);
    return false;
  }

  auto broadcastOp = dyn_cast<linalg::BroadcastOp>(outerInOp);
  if (broadcastOp) {
    if (auto btUsage = CVPipeline::getBTSizeFromValidBroadcastOp(broadcastOp)) {
      if (btUsage != -1 && btUsage <= CVPipeline::CACHE_TABLE_BUFFER_SIZE) {
        LOG_DEBUG("Not split because broadcast bias is small. " << matmulOp);
        return false;
      }
    }
  }

  auto defMatmul = dyn_cast_if_present<linalg::MatmulOp>(
      hivm::traceDefOp<linalg::MatmulOp>(outerInValue).value_or(nullptr));
  if (defMatmul) {
    LOG_DEBUG("Not split because L0C remain. " << matmulOp);
    return false;
  }
  return true;
}

/**
 * Verifies whether a matmul operation is valid for split transformation.
 * Returns false if the matmul should be skipped due to:
 * 1. Already processed matmul (has kLoopCarriedL0C attribute)
 * 2. Illegal matmul (missing inits or inputs)
 * 3. Memref-type matmul (not tensor mode)
 */
static bool verifyMatmul(linalg::MatmulOp matmulOp) {
  if (matmulOp->hasAttr(CVPipeline::kLoopCarriedL0C)) {
    return false;
  }
  auto inits = matmulOp.getDpsInits();
  auto inputs = matmulOp.getDpsInputs();
  if (inits.empty() || inputs.size() < 2) {
    LOG_DEBUG("Not split because op is illegal: " << matmulOp);
    return false;
  }

  auto bias = matmulOp.getDpsInits()[0];
  auto outputType = dyn_cast<RankedTensorType>(bias.getType());
  if (!outputType) {
    LOG_DEBUG("Not split because not tensor mode matmul: " << matmulOp);
    return false;
  }
  if (!isFloatOrInt(outputType)) {
    LOG_DEBUG("Not split because not integer or float: " << matmulOp);
    return false;
  }
  return true;
}

// Supports the most common case, where a loop-carried matmul is
// directly under a not executed for-loop
static std::optional<SplitInfo> handleMayNotExec(linalg::MatmulOp matmulOp,
                                                 Value outerInValue) {
  auto *parentOp = matmulOp->getBlock()->getParentOp();
  auto forOp = llvm::dyn_cast<scf::ForOp>(parentOp);
  auto inputs = parseMatmulInputs(matmulOp);
  auto bias = inputs.bias;
  if (!forOp || !scfMayNotExec(forOp)) {
    return std::nullopt;
  }
  auto blockArg = llvm::dyn_cast<BlockArgument>(bias);
  if (!blockArg || blockArg.getOwner()->getParentOp() != forOp) {
    return std::nullopt;
  }
  auto initVal = forOp.getTiedLoopInit(blockArg)->get();
  auto result = forOp.getTiedLoopResult(blockArg);

  // only support the scenario that initVal is zero not broadcast
  if (operationIsFillZero(outerInValue.getDefiningOp())) {
    // Check if the current matmul is used by a subsequent cross-block
    // cascade matmul (L0C -> L0C).
    if (hasCrossBlockCascadeL0CConsumer(result, matmulOp)) {
      return SplitInfo{false, initVal, result, false, false};
    }
  }

  // Check if the current matmul's bias (L0C) is defined by a cross-block
  // cascade matmul (L0C -> L0C).
  if (hasCrossBlockCascadeL0CProducer(initVal, matmulOp)) {
    return SplitInfo{true, initVal, result, false, true};
  }

  return SplitInfo{true, initVal, result, true, false};
}

/**
 * Determines whether a matmul operation should be split based on comprehensive
 * analysis. This function performs a complete analysis of the matmul
 * operation's context, including:
 * 1. Tracing the use-def chain of the output to find the outermost scope.
 * 2. Evaluating input-based split criteria via shouldSplitByInput() and
 * shouldSplitByOutput().
 *
 * Parameters:
 *   matmulOp - The matmul operation to analyze.
 *
 * Returns:
 *   An optional SplitInfo structure containing:
 *   - mayNotExec: Whether the matmul might not execute (e.g., in a conditional
 * or empty loop).
 *   - outerInValue: The outermost initial value (original bias) in the chain.
 *   - outerOutValue: The outermost result value in the chain.
 *   - shouldSplit: Boolean indicating whether the split transformation should
 * be applied. Returns std::nullopt if analysis cannot be performed.
 */
static std::optional<SplitInfo> shouldSplit(linalg::MatmulOp matmulOp,
                                            bool needSplitAll) {

  if (!verifyMatmul(matmulOp)) {
    return std::nullopt;
  }

  auto matmulInput = parseMatmulInputs(matmulOp);
  if (needSplitAll) {
    LOG_DEBUG("Split because needSplitAll is true. " << matmulOp);
    return SplitInfo{false, matmulInput.bias, matmulOp.getResult(0), true,
                     false};
  }

  bool argsLimitedInMatmul = true;
  bool mayNotExec = false;
  Value outerInValue = matmulInput.bias;
  auto outerOutValue = searchInArgsChain(
      matmulOp.getResult(0), argsLimitedInMatmul, mayNotExec, outerInValue);
  if (utils::getAnnotateOpWithAttr(outerOutValue, kMatmulAtLeastOnceHint)
          .has_value()) {
    mayNotExec = false;
  }
  if (!argsLimitedInMatmul) {
    LOG_DEBUG("Split because bias is not limited in args" << matmulOp); // S25
    return SplitInfo{mayNotExec, outerInValue, outerOutValue, true, false};
  }

  if (mayNotExec) {
    auto splitInfoOpt = handleMayNotExec(matmulOp, outerInValue);
    if (splitInfoOpt.has_value()) {
      return splitInfoOpt;
    }
  }

  if (!shouldSplitByInput(matmulOp, outerOutValue, outerInValue) &&
      !shouldSplitByOutput(matmulOp, outerOutValue, outerInValue)) {
    return SplitInfo{mayNotExec, outerInValue, outerOutValue, false, false};
  }

  return SplitInfo{mayNotExec, outerInValue, outerOutValue, true, false};
}

/**
 * Collects all the scf::ForOps that gate the execution of the current
 * matmul in a cascade scenario.
 */
static SmallVector<scf::ForOp> collectCascadeForOps(const SplitInfo &splitInfo,
                                                    scf::ForOp currentForOp) {
  SmallVector<scf::ForOp> result;
  if (!currentForOp) {
    return result;
  }
  result.push_back(currentForOp);

  SmallPtrSet<Operation *, 8> seen;
  seen.insert(currentForOp);

  Value curIn = splitInfo.outerInValue;
  while (true) {
    auto defMatmul = dyn_cast_if_present<linalg::MatmulOp>(
        hivm::traceDefOp<linalg::MatmulOp>(curIn).value_or(nullptr));
    if (!defMatmul) {
      break;
    }
    if (auto producerForOp = llvm::dyn_cast_if_present<scf::ForOp>(
            defMatmul->getBlock()->getParentOp())) {
      if (seen.insert(producerForOp).second) {
        result.push_back(producerForOp);
      }
    }
    // Continue tracing upward along the producer matmul's bias input.
    auto inputs = parseMatmulInputs(defMatmul);
    curIn = inputs.bias;
  }

  return result;
}

/**
 * Handles the mayNotExec case by creating select operations.
 * This function creates a select operation to choose between the actual result
 * and a zero-filled result based on whether the loop will execute.
 *
 * Returns the select result if mayNotExec is handled, otherwise returns the
 * original outerOutValue.
 */
static Value handleMayNotExecSelect(linalg::MatmulOp matmulOp,
                                    PatternRewriter &rewriter,
                                    SplitInfo &splitInfo,
                                    bool replaceUses = true) {
  auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(
      splitInfo.outerOutValue.getDefiningOp());
  if (!forOp) {
    return splitInfo.outerOutValue;
  }

  auto outputType =
      dyn_cast<RankedTensorType>(parseMatmulInputs(matmulOp).bias.getType());
  if (!outputType) {
    return splitInfo.outerOutValue;
  }
  auto elmType = outputType.getElementType();
  Location loc = matmulOp.getLoc();

  rewriter.setInsertionPointAfterValue(splitInfo.outerOutValue);

  SmallVector<scf::ForOp> cascadeForOps =
      collectCascadeForOps(splitInfo, forOp);

  Value executed = nullptr;
  for (auto loopOp : cascadeForOps) {
    auto lb = loopOp.getLowerBound();
    auto ub = loopOp.getUpperBound();
    Value cmp =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, ub, lb);
    executed =
        executed ? rewriter.create<arith::OrIOp>(loc, executed, cmp) : cmp;
  }

  Value zeroValue;
  if (auto floatType = dyn_cast<FloatType>(elmType)) {
    APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
    zeroValue =
        rewriter.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat)
            .getResult();
  } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
    zeroValue =
        rewriter.create<arith::ConstantIntOp>(loc, intType, 0).getResult();
  }
  auto fillOp =
      rewriter.create<linalg::FillOp>(loc, zeroValue, splitInfo.outerOutValue);
  auto selectOp = rewriter.create<arith::SelectOp>(
      loc, executed, splitInfo.outerOutValue, fillOp.getResult(0));

  if (replaceUses) {
    // Replace uses of outerOutValue with select result, except for the select
    // and fill operations
    splitInfo.outerOutValue.replaceUsesWithIf(
        selectOp.getResult(), [&](OpOperand &operand) {
          return operand.getOwner() != selectOp && operand.getOwner() != fillOp;
        });
  }

  forOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                 rewriter.getUnitAttr());

  return selectOp.getResult();
}

static LogicalResult splitMatmul(linalg::MatmulOp matmulOp,
                                 PatternRewriter &rewriter,
                                 SplitInfo splitInfo) {
  auto outputType =
      dyn_cast<RankedTensorType>(parseMatmulInputs(matmulOp).bias.getType());
  if (!outputType) {
    LOG_DEBUG("Not tensor mode: " << matmulOp
                                  << "; the caller does not ensure the "
                                     "assumption. Cowardly doing nothing");
    return failure();
  }
  auto elmType = outputType.getElementType();

  Location loc = matmulOp.getLoc();

  // [Step 1] Create tensor.empty for the new accumulator tensor
  // Same shape and type as original matmul output
  auto outerDefOp = splitInfo.outerOutValue.getDefiningOp();
  rewriter.setInsertionPoint(outerDefOp);
  SmallVector<Value> dynamicSizes;
  for (int64_t i = 0; i < outputType.getRank(); ++i) {
    if (outputType.isDynamicDim(i)) {
      dynamicSizes.push_back(rewriter.create<tensor::DimOp>(
          loc, parseMatmulInputs(matmulOp).bias, i));
    }
  }
  auto emptyOp =
      rewriter.create<tensor::EmptyOp>(loc, outputType, dynamicSizes);
  Value zeroValue;
  if (auto floatType = dyn_cast<FloatType>(elmType)) {
    APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
    zeroValue =
        rewriter.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat)
            .getResult();
  } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
    zeroValue =
        rewriter.create<arith::ConstantIntOp>(loc, intType, 0).getResult();
  }
  auto fillOp = rewriter.create<linalg::FillOp>(emptyOp.getLoc(), zeroValue,
                                                emptyOp.getResult());
  auto zeroVal = fillOp.getResult(0);
  if (outerDefOp == matmulOp.getOperation()) {
    // The matmul input and accumulator may alias. Replace only the accumulator.
    matmulOp.getDpsInitsMutable()[0].assign(zeroVal);
  } else {
    splitInfo.outerInValue.replaceUsesWithIf(zeroVal, [&](OpOperand &opop) {
      return opop.getOwner() == outerDefOp;
    });
  }

  auto forOp = llvm::dyn_cast_if_present<scf::ForOp>(
      splitInfo.outerOutValue.getDefiningOp());
  if (!splitInfo.mayNotExec || !forOp) {
    // [Step 2] Create new matmul using zero-filled tensor as accumulator
    // New matmul runs entirely on CUBE with no VECTOR dependency
    auto inputs = parseMatmulInputs(matmulOp);
    auto a = inputs.a;
    auto b = inputs.b;
    auto bias = inputs.bias;
    rewriter.setInsertionPoint(matmulOp);
    auto newMatmul = rewriter.create<linalg::MatmulOp>(loc, ValueRange{a, b},
                                                       ValueRange{bias});
    NamedAttrList attrs(matmulOp->getAttrDictionary());
    constexpr StringLiteral kShouldRemoveAttrs[] = {"operandSegmentSizes",
                                                    "res_attrs", "arg_attrs"};
    for (auto attr : kShouldRemoveAttrs) {
      attrs.erase(attr);
    }
    newMatmul->setAttrs(attrs);
    if (splitInfo.outerOutValue == matmulOp.getResult(0)) {
      splitInfo.outerOutValue = newMatmul.getResult(0);
    }
    rewriter.replaceOp(matmulOp, newMatmul);
  }

  auto newOutValue = splitInfo.outerOutValue;

  // [Step 3] Create add: add(new_matmul_result, outs_value)
  // This is the "c" in a*b+c, added after the matmul result
  rewriter.setInsertionPointAfterValue(splitInfo.outerOutValue);

  constexpr size_t kMaxPreservedUsers = 2;
  SmallPtrSet<Operation *, kMaxPreservedUsers> preservedUsers;
  if (splitInfo.mayNotExec && forOp) {
    auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBound();

    Value executed =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, ub, lb);
    Value zeroValue;
    if (auto floatType = dyn_cast<FloatType>(elmType)) {
      APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
      zeroValue =
          rewriter.create<arith::ConstantFloatOp>(loc, floatType, zeroAPFloat)
              .getResult();
    } else if (auto intType = dyn_cast<IntegerType>(elmType)) {
      zeroValue =
          rewriter.create<arith::ConstantIntOp>(loc, intType, 0).getResult();
    }
    auto fillOp = rewriter.create<linalg::FillOp>(loc, zeroValue,
                                                  splitInfo.outerOutValue);
    auto selectOp = rewriter.create<arith::SelectOp>(
        loc, executed, splitInfo.outerOutValue, fillOp.getResult(0));
    fillOp->setAttr(CVPipeline::kForMayNotExec, rewriter.getUnitAttr());
    selectOp->setAttr(CVPipeline::kForMayNotExec, rewriter.getUnitAttr());
    newOutValue = selectOp.getResult();
    preservedUsers.insert(selectOp);
    preservedUsers.insert(fillOp);
    forOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                   rewriter.getUnitAttr());
  }

  Operation *addOp;
  if (isa<FloatType>(elmType)) {
    addOp =
        rewriter.create<arith::AddFOp>(loc, newOutValue, splitInfo.outerInValue)
            .getOperation();
  } else {
    addOp =
        rewriter.create<arith::AddIOp>(loc, newOutValue, splitInfo.outerInValue)
            .getOperation();
  }
  if (preservedUsers.empty()) {
    preservedUsers.insert(addOp);
  }
  addOp->setAttr(CVPipeline::kAddFromMatmul, rewriter.getUnitAttr());
  splitInfo.outerOutValue.replaceUsesWithIf(
      addOp->getResult(0), [&](OpOperand &operand) {
        return !preservedUsers.contains(operand.getOwner());
      });

  return success();
}

LogicalResult
SplitMatmulPattern::matchAndRewrite(linalg::MatmulOp matmulOp,
                                    PatternRewriter &rewriter) const {
  if (CVPipeline::hasFallbackAttr(matmulOp->getParentOfType<ModuleOp>())) {
    return failure();
  }

  LOG_DEBUG("check matmulOp = " << matmulOp);
  auto splitInfoOpt = shouldSplit(matmulOp, needSplitAll);
  if (!splitInfoOpt.has_value()) {
    return failure();
  }
  auto splitInfo = splitInfoOpt.value();
  LOG_DEBUG("-------------------");
  LOG_DEBUG("matmulOp = " << matmulOp);
  LOG_DEBUG("outerOutValue = " << splitInfo.outerOutValue);
  LOG_DEBUG("outerInValue = " << splitInfo.outerInValue);
  LOG_DEBUG("mayNotExec = " << splitInfo.mayNotExec);
  LOG_DEBUG("shouldSplit = " << splitInfo.shouldSplit);
  LOG_DEBUG("-------------------");
  matmulOp->setAttr(CVPipeline::kLoopCarriedL0C, rewriter.getUnitAttr());

  if (auto markOpOpt = utils::getAnnotateOpWithAttr(splitInfo.outerOutValue,
                                                    kMatmulAtLeastOnceHint);
      markOpOpt.has_value()) {
    auto markOp = markOpOpt.value();
    markOp->removeAttr(kMatmulAtLeastOnceHint);
    if (markOp->getAttrs().empty() ||
        (markOp->getAttrs().size() == 1 &&
         markOp->getAttrs().front().getName() == "effects")) {
      rewriter.eraseOp(markOp);
    }
    matmulOp->getParentOp()->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                                     rewriter.getUnitAttr());
  }

  if (splitInfo.shouldSplit) {
    return splitMatmul(matmulOp, rewriter, splitInfo);
  }

  if (splitInfo.mayNotExec) {
    if (splitInfo.supported) {
      handleMayNotExecSelect(matmulOp, rewriter, splitInfo);
    } else {
      matmulOp->setAttr(CVPipeline::kMayNotExec, rewriter.getUnitAttr());
    }
  }

  return success();
}
