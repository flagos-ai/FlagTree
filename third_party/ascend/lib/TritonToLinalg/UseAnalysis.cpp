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

#include "ascend/include/TritonToLinalg/UseAnalysis.h"
#include "ascend/include/Utils/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;
using namespace dataflow;

#define DEBUG_TYPE "triton-use-analysis"

std::string stringifyUseType(UseType useTy) {
  std::string ret;
  if (useTy == UseType::MetaUse) {
    ret = "MetaUse";
  } else if (useTy == UseType::DataUse) {
    ret = "DataUse";
  } else if (useTy == UseType::MixUse) {
    ret = "MixUse";
  } else if (useTy == UseType::Undefined) {
    ret = "Undefined";
  }
  return ret;
}

#if LLVM_VERSION_MAJOR >= 20
LogicalResult
triton::UseAnalysis::visitOperation(Operation *op, ArrayRef<UseInfo *> operands,
                                    ArrayRef<const UseInfo *> results) {
#else
void triton::UseAnalysis::visitOperation(Operation *op,
                                         ArrayRef<UseInfo *> operands,
                                         ArrayRef<const UseInfo *> results) {
#endif

  if (op->getResults().size() == 1) {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (resultType && isa<triton::PointerType>(resultType.getElementType())) {
      for (auto opnd : operands) {
        propagateUse(opnd, UseType::MetaUse);
      }
    }
  }

  TypeSwitch<Operation *>(op)
      .Case<triton::LoadOp>([&](auto load) {
        propagateUse(operands[0], UseType::MetaUse);
        auto mask = load.getMask();
        auto other = load.getOther();
        if (mask) {
          assert(mask != other && "mask and other cannot be the same");
          propagateUse(operands[1], UseType::MetaUse);
        }
        if (other) {
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::PrintOp>([&](auto print) {
        for (auto operand : operands)
          propagateUse(operand, UseType::DataUse);
      })
      .Case<triton::AssertOp>(
          [&](auto assert) { propagateUse(operands[0], UseType::DataUse); })
      .Case<triton::StoreOp>([&](auto store) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        auto value = store.getValue();
        auto mask = store.getMask();
        if (mask) {
          assert(mask != value && "mask and data cannot be the same");
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::ascend::IndirectStoreOp>([&](auto store) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        propagateUse(operands[2], UseType::DataUse);
        auto mask = store.getMask();
        if (mask) {
          propagateUse(operands[3], UseType::DataUse);
        }
      })
      // Consider triton::AtomicRMWOp as store operation
      .Case<triton::AtomicRMWOp>([&](auto atomicOp) {
        propagateUse(operands[0], UseType::MixUse);
        propagateUse(operands[1], UseType::DataUse);
        auto value = atomicOp.getVal();
        auto mask = atomicOp.getMask();
        if (mask) {
          assert(mask != value && "mask and data cannot be the same");
          propagateUse(operands[2], UseType::MetaUse);
        }
      })
      .Case<triton::AtomicCASOp>([&](auto atomicOp) {
        propagateUse(operands[0], UseType::MetaUse);
        propagateUse(operands[1], UseType::DataUse);
        propagateUse(operands[2], UseType::DataUse);
        auto value = atomicOp.getVal();
      })
      .Case<triton::DotOp>([&](auto dot) {
        propagateResults(operands[0], results);
        propagateResults(operands[1], results);

        auto opc = dot.getC();
        triton::SplatOp splat;
        if (opc) {
          splat = opc.template getDefiningOp<triton::SplatOp>();
        }

        if (opc && splat && splat.getSrc().getDefiningOp<arith::ConstantOp>()) {
          propagateUse(operands[2], UseType::MetaUse);
        } else {
          propagateUse(operands[2], UseType::DataUse);
        }
      })
      .Case<LoopLikeOpInterface>([&](auto loopOp) {
        for (const auto &[yield, init, result] : llvm::zip_equal(
                 loopOp.getYieldedValues(), loopOp.getInits(), results)) {
          propagateResults(getLatticeElement(yield), {result});
          propagateResults(getLatticeElement(init), {result});
        }
      })
      .Case<triton::ReduceOp>([&](auto reduceOp) {
        for (auto operand : operands) {
          propagateUse(operand, UseType::DataUse);
        }
      })
      .Case<tensor::ExtractOp>([&](auto extractOp) {
        for (auto operand : operands) {
          propagateUse(operand, UseType::DataUse);
        }
      })
      .Case<hivm::FixpipeOp>(
          [&](auto fixpipeOp) { propagateUse(operands[0], UseType::DataUse); })
      .Case<hivm::CopyOp>(
          [&](auto copyOp) { propagateUse(operands[0], UseType::DataUse); })
      .Case<hivm::CustomOp, hivm::CustomMacroOp>([&](auto customOp) {
        for (auto operand : operands) {
          propagateUse(operand, UseType::MixUse);
        }
      })
      .Default([&](Operation *op) {
        // this condition account for tt.addptr
        for (auto operand : operands) {
          propagateResults(operand, results);
        }
      });
#if LLVM_VERSION_MAJOR >= 20
  return success();
#endif
}

void setMixUseRecursively(Operation *rootOp, bool applyRoot = true) {
  traverseBackwardUpdateOperandChainIf(
      rootOp,
      // ConditionFn
      [rootOp, applyRoot](Operation *curOp) {
        for (auto res : curOp->getResults()) {
          auto tensorType = dyn_cast<RankedTensorType>(res.getType());
          if (tensorType &&
              isa<triton::PointerType>(tensorType.getElementType()))
            return false;
        }
        return isMetaUse(curOp) && (curOp != rootOp || applyRoot);
      },
      // StopFn
      [rootOp](Operation *curOp) {
        return isa<triton::LoadOp>(curOp) && curOp != rootOp;
      },
      // ActionFn
      [](OpBuilder &b, Operation *op) {
        LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(b.getContext())); });
        op->removeAttr("MetaUse");
      });
}

static void setMixUseFromValue(Value v) {
  if (auto *defOp = v.getDefiningOp()) {
    setMixUseRecursively(defOp);
    return;
  }

  auto blockArg = dyn_cast<BlockArgument>(v);
  if (!blockArg) {
    return;
  }

  auto *parentOp = blockArg.getOwner()->getParentOp();
  auto loopLikeOp = dyn_cast_or_null<LoopLikeOpInterface>(parentOp);
  if (!loopLikeOp) {
    return;
  }

  if (OpOperand *init = loopLikeOp.getTiedLoopInit(blockArg)) {
    if (auto *initDefOp = init->get().getDefiningOp())
      setMixUseRecursively(initDefOp);
  }

  if (OpOperand *yielded = loopLikeOp.getTiedLoopYieldedValue(blockArg)) {
    if (auto *yieldDefOp = yielded->get().getDefiningOp())
      setMixUseRecursively(yieldDefOp);
  }
}

std::optional<bool> isIterArgMixUse(Value v, Value target,
                                    const DataFlowSolver &solver) {
  auto defOp = v.getDefiningOp();
  auto *use = solver.lookupState<UseInfo>(v);
  if ((use && use->type == UseType::DataUse) ||
      isa_and_nonnull<LoopLikeOpInterface, scf::IfOp>(defOp))
    return true;
  if (v == target)
    return false;
  if (!defOp)
    return std::nullopt;
  for (auto oper : defOp->getOperands()) {
    auto res = isIterArgMixUse(oper, target, solver);
    if (res.has_value())
      return res.value() || !isMetaUse(defOp);
  }
  return std::nullopt;
}

void postProcessWhileOp(scf::WhileOp op, const DataFlowSolver &solver) {
  for (const auto &[res, arg] :
       llvm::zip_equal(op->getResults(), op.getConditionOp().getArgs())) {
    auto *defOp = arg.getDefiningOp();
    if (!defOp)
      continue;
    auto *use = solver.lookupState<UseInfo>(res);
    if (use && use->type == UseType::DataUse)
      setMixUseRecursively(defOp);
  }
  for (const auto &[yield, regionArg] : llvm::zip_equal(
           op.getYieldOp().getOperands(), op.getBeforeArguments())) {
    auto *defOp = yield.getDefiningOp();
    if (!defOp)
      continue;
    if (isIterArgMixUse(yield, regionArg, solver).value_or(false))
      setMixUseRecursively(defOp);
  }
}

void postProcessLoopOp(LoopLikeOpInterface loopOp,
                       const DataFlowSolver &solver) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp.getOperation())) {
    postProcessWhileOp(whileOp, solver);
    return;
  }
  for (const auto &[res, yield, regionArg] :
       llvm::zip_equal(loopOp->getResults(), loopOp.getYieldedValues(),
                       loopOp.getRegionIterArgs())) {
    auto *defOp = yield.getDefiningOp();
    if (!defOp)
      continue;
    auto *use = solver.lookupState<UseInfo>(res);
    if ((use && use->type == UseType::DataUse) ||
        isIterArgMixUse(yield, regionArg, solver).value_or(false))
      setMixUseRecursively(defOp);
  }
}

LogicalResult triton::runUseAnalysis(triton::FuncOp &funcOp) {
  MLIRContext *context = funcOp.getContext();
  SymbolTableCollection symbolTable;

  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<UseAnalysis>(symbolTable);
  if (failed(solver.initializeAndRun(funcOp))) {
    return failure();
  }
  auto &os = llvm::dbgs();
  // Walk the func op, convert tags on operands to tags on operations
  funcOp.walk([&](Operation *op) {
    LLVM_DEBUG({ os << "[UseAnalysis] op is " << *op << "\n"; });
    UseType useType = UseType::Undefined;
    for (auto result : op->getResults()) {
      LLVM_DEBUG({ os << "[UseAnalysis] ===> result is " << result << "\n"; });
      auto use = solver.lookupState<UseInfo>(result);
      assert(use && "Lattice value not found");
      auto thisUseType = use->type;
      LLVM_DEBUG({
        os << "[UseAnalysis] ==========> useType is "
           << stringifyUseType(thisUseType) << "\n";
      });
      if (thisUseType == UseType::Undefined) {
        continue;
      }
      if (useType == UseType::Undefined) {
        useType = thisUseType;
      }
      if (thisUseType == UseType::MixUse || thisUseType != useType) {
        useType = UseType::MixUse;
        break;
      }
    }

    if (useType == UseType::Undefined) {
      LLVM_DEBUG({ op->setAttr("Undefined", UnitAttr::get(context)); });
      return;
    } else if (useType == UseType::MetaUse) {
      auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
      if (memEffect) {
        if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(op)) {
          LLVM_DEBUG(
              { os << "force protecting side-effect op:" << *op << "\n"; });
          op->setAttr("DataUse", UnitAttr::get(context));
          return;
        }
      }
      if (!isa<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp,
               scope::ScopeOp, triton::ReduceOp>(op)) {
        assert(op->getNumResults() == 1 &&
               "Ops used for meta computation are expected to have one result");
      }
      for (auto it = 0; it < op->getNumResults(); ++it) {
        // Only set the tag if the operation uses tensors
        if (isa<ShapedType>(op->getResult(it).getType()) ||
            (isa<triton::LoadOp>(op) &&
             op->hasAttr(ConverterUtils::discreteAttrName)) ||
            (isa<triton::BitcastOp>(op) &&
             isa<PointerType>(op->getResult(it).getType()))) {
          // Setting tag for erasing op later
          op->setAttr("MetaUse", UnitAttr::get(context));
        }
      }
      return;
    } else if (useType == UseType::DataUse) {
      LLVM_DEBUG({ op->setAttr("DataUse", UnitAttr::get(context)); });
      return;
    }

    assert(useType == UseType::MixUse);

    // If the operation only produces scalars, no need to clone it
    bool shapedResult = true;
    for (auto result : op->getResults())
      shapedResult &= isa<ShapedType>(result.getType());
    if (!shapedResult ||
        isa<LoopLikeOpInterface, scf::IfOp, arith::SelectOp>(op)) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }
    llvm::SetVector<Operation *> metaUsers;
    for (auto result : op->getResults()) {
      for (auto user : result.getUsers()) {
        TypeSwitch<Operation *>(user)
            .Case<triton::LoadOp>([&](auto load) {
              auto ptr = load.getPtr();
              auto mask = load.getMask();
              auto other = load.getOther();
              if (result == ptr || result == mask || result == other) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::StoreOp>([&](auto store) {
              auto ptr = store.getPtr();
              auto mask = store.getMask();
              if (result == ptr || result == mask) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::ascend::IndirectStoreOp>([&](auto indirectstore) {
              auto src = indirectstore.getSrc();
              // Only src is MetaUse (see visitOperation). offsets/value/mask
              // are DataUse
              if (result == src) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::AtomicRMWOp>([&](auto atomicOp) {
              auto ptr = atomicOp.getPtr();
              auto mask = atomicOp.getMask();
              if (result == ptr || result == mask)
                metaUsers.insert(user);
            })
            .Case<triton::AtomicCASOp>([&](auto atomicOp) {
              auto ptr = atomicOp.getPtr();
              if (result == ptr)
                metaUsers.insert(user);
            })
            .Case<triton::DotOp>([&](auto dot) {
              auto opc = dot.getC();
              triton::SplatOp splat;
              if (opc) {
                splat = opc.template getDefiningOp<triton::SplatOp>();
              }

              if (opc && splat &&
                  splat.getSrc().getDefiningOp<arith::ConstantOp>()) {
                metaUsers.insert(user);
              }
            })
            .Case<triton::PrintOp>([&](auto print) {})
            .Case<hivm::CustomOp, hivm::CustomMacroOp>([](auto) {
              // Custom operands are MixUse in visitOperation, even when all
              // results are used only as memory addresses. Keep the call on
              // the materialized value, not a MetaUse clone that is erased
              // before the custom operation's operands are converted.
            })
            .Default([&](Operation *op) {
              bool allMeta = true;
              for (auto res : op->getResults()) {
                auto resUse = solver.lookupState<UseInfo>(res);
                if (resUse->type != UseType::MetaUse) {
                  allMeta = false;
                  break;
                }
              }
              if (allMeta) {
                metaUsers.insert(user);
              }
            });
      }
    }

    // If the operation doesn't have direct meta users, no need to clone it
    if (metaUsers.empty()) {
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      return;
    }

    if (isa<LoopLikeOpInterface, scf::IfOp>(op))
      return;

    if (isa<triton::LoadOp>(op))
      return;

    // Clone the operation; switch all meta users to use the clone
    OpBuilder builder(op);
    auto clone = builder.clone(*op);
    LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });

    // Setting tag for erasing op later
    clone->setAttr("MetaUse", UnitAttr::get(context));

    for (auto [res_i, result] : llvm::enumerate(op->getResults())) {
      for (auto user : metaUsers) {
        for (auto &operand : user->getOpOperands()) {
          if (operand.get() == result) {
            operand.set(clone->getResult(res_i));
          }
        }
      }
    }
  });
  LLVM_DEBUG({
    os << "[UseAnalysis] Before post-process, funcOp is " << *funcOp << "\n";
  });
  // Post-process
  funcOp.walk([&](Operation *op) {
    // Handle indirect load and store case.
    // For example, load(1st) -> computeOp -> load(2nd),
    // or load -> computeOp -> store
    // The first load is IndirectLoadInterfaceOp.
    // Do not inplace replace MetaUse by MixUse. Because the condition checking
    // depends on that the op has the attr of MetaUse.
    // Handle the indirect load interface op
    // We first trace from the 1st load to the 2nd load with the ops between
    // them marked as MixUse. Then we traceback from the 2nd load to mark defs
    // MixUse.
    if (opIsIndirectLoad(op) || opIsIndirectCalc(op)) {
      LLVM_DEBUG({
        os << "[UseAnalysis] Found indirect load interface op: " << *op << "\n";
      });
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      // Modify the users of this op's result.
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [op](Operation *curOp) { return isMetaUse(curOp) && curOp != op; },
          /*stopFn*/
          [&](Operation *curOp) {
            // triton::LoadOp or triton::StoreOp without MetaUse means
            // it is an indirect load or store
            // instead of the load providing the offset.
            // The pattern is as follows,
            // load -> ops -> load
            // load -> ops -> store
            // We need to ensure the intermediate ops are marked MixUse
            // so that they will be replaced instead of be erased without
            // conversion.
            return (isa<triton::LoadOp, triton::StoreOp,
                        triton::ascend::IndirectStoreOp>(curOp)) &&
                   !isMetaUse(curOp);
          },
          /*actionFn*/
          [](OpBuilder &b, Operation *op) { setMixUseRecursively(op); },
          stopOps);
      LLVM_DEBUG({
        os << "[UseAnalysis] stopOps are \n";
        for (auto [idx, stopOp] : llvm::enumerate(stopOps))
          os << idx << ": " << *stopOp << "\n";
      });
      LLVM_DEBUG({
        os << "[UseAnalysis] After trace, funcOp is " << *funcOp << "\n";
      });
      for (auto *stopOp : stopOps)
        setMixUseRecursively(stopOp, /*applyRoot=*/false);
      LLVM_DEBUG({
        os << "[UseAnalysis] After traceback of stopOp, funcOp is " << *funcOp
           << "\n";
      });
      // Modify this op.
      LLVM_DEBUG({ op->setAttr("MixUse", UnitAttr::get(context)); });
      op->removeAttr("MetaUse");
    }
    if (op->hasAttr(ConverterUtils::discreteAttrName))
      setMixUseRecursively(op);
    if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
      postProcessLoopOp(loopOp, solver);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Value> yields(ifOp.thenYield().getOperands());
      if (!ifOp.getElseRegion().empty())
        yields.append(llvm::to_vector(ifOp.elseYield().getOperands()));
      for (auto yield : yields) {
        setMixUseFromValue(yield);
      }
    } else if (auto atomicRmwOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      auto mask = atomicRmwOp.getMask();
      if (mask && op->hasAttr(ConverterUtils::discreteMaskAttrName))
        setMixUseRecursively(mask.getDefiningOp());
    }
  });
  // Remove MetaUse in case of MixUse existing in the op
  funcOp.walk([&](Operation *op) {
    if (isMetaUse(op) && isMixUse(op)) {
      op->removeAttr("MetaUse");
    }
  });
  // Masked load with non-scalar tensor `other` lowers as
  // arith.select(mask, loaded, other) so the mask SSA must stay live.
  funcOp.walk([&](triton::LoadOp load) {
    Value mask = load.getMask();
    Value other = load.getOther();
    if (!mask || !other)
      return;
    if (other.getDefiningOp<triton::SplatOp>())
      return;
    if (auto c = other.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseElementsAttr>(c.getValue())) {
        if (dense.isSplat())
          return;
      }
    }
    if (auto *def = mask.getDefiningOp())
      setMixUseRecursively(def);
  });
  // hivm.custom present library call, shouldn't be metause
  funcOp.walk([&](hivm::CustomOp op) {
    if (isMetaUse(op)) {
      op->removeAttr("MetaUse");
    }
  });
  funcOp.walk([&](hivm::CustomMacroOp op) {
    if (isMetaUse(op)) {
      op->removeAttr("MetaUse");
    }
  });
  LLVM_DEBUG({
    os << "[UseAnalysis] After post-process, funcOp is " << *funcOp << "\n";
  });
  return success();
}

static bool isScalarI1ToI8PointerBitcast(Operation *op) {
  auto bitcast = dyn_cast<triton::BitcastOp>(op);
  if (!bitcast)
    return false;

  auto sourceType = dyn_cast<triton::PointerType>(bitcast.getSrc().getType());
  auto resultType = dyn_cast<triton::PointerType>(bitcast.getType());
  if (!sourceType || !resultType)
    return false;

  return sourceType.getPointeeType().isInteger(1) &&
         resultType.getPointeeType().isInteger(8);
}

MetaUseEraser::MetaUseEraser(MLIRContext *context)
    : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

LogicalResult MetaUseEraser::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  LLVM_DEBUG({
    int64_t count = 0;
    for (auto result : op->getResults()) {
      count += std::distance(result.use_begin(), result.use_end());
    }
    llvm::dbgs() << "Number of user: " << count << "\n";
  });
  if (isa<triton::AddPtrOp>(op)) {
    return rewriter.notifyMatchFailure(op,
                                       "AddPtrOp will be handled separately");
  }
  if (isScalarI1ToI8PointerBitcast(op))
    return rewriter.notifyMatchFailure(
        op, "scalar i1-to-i8 pointer bitcast requires conversion");
  if (isMetaUse(op)) {
    rewriter.eraseOp(op);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "requires meta ops");
}
