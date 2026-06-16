/*
 * Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
 * All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License"); you may
 *    not use this file except in compliance with the License. You may obtain
 *    a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *    License for the specific language governing permissions and limitations
 *    under the License.
 */

// MMA reduce thread-locality optimization (Iluvatar).
//
// This is an Iluvatar-specific sibling of the upstream
// `tritongpu-optimize-thread-locality` pass. It reuses the same loop-rewrite
// strategy (split the register-resident part of the reduce axis into a trailing
// dimension, do the cheap thread-local reduce inside the loop, carry the
// partial, and do the single cross-lane reduce once after the loop), but
// extends it in two ways needed by FlashAttention online-softmax on the
// Iluvatar `#mma` layout:
//
//   1. Generic (non-blocked) source encodings: the rank+1 "thread locality"
//      view is built directly from the source LinearLayout, routing the
//      register-driven bits of the reduce axis into the new trailing dim. This
//      keeps the reshape a free view (no data movement).
//
//   2. Rescaled accumulators: FA computes `l = l * alpha + sum(p)` rather than
//      `l = l + sum(p)`. Because `*alpha` (a per-row scalar) distributes over
//      the add-reduce, the rescale can be applied to the carried partial each
//      iteration. The update op-chain is rebuilt on the split partial and the
//      rescale operand (`alpha`, shape [M]) is broadcast into the partial's
//      [M, lanePart] layout.
//
// Only the `addf` combiner with a multiplicative rescale chain is supported,
// which is exactly what the softmax running-sum needs (and matches the scope of
// the v3.2 `MMAReduce`/`noWarpReduce` feature this replaces).

#include "TritonILUVATARGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONILUVATARGPUMMAREDUCETHREADLOCALITY
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

namespace ttg = mlir::triton::gpu;

namespace {

class TritonILUVATARGPUMMAReduceThreadLocalityPass
    : public impl::TritonILUVATARGPUMMAReduceThreadLocalityBase<
          TritonILUVATARGPUMMAReduceThreadLocalityPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    DenseSet<triton::ReduceOp> reduceOps;
    mod.walk([&](triton::ReduceOp reduce) -> void {
      if (isCandidate(reduce))
        reduceOps.insert(reduce);
    });

    IRRewriter builder(&getContext());
    for (auto reduce : reduceOps)
      rewrite(builder, mod, reduce);
  }

private:
  //===--------------------------------------------------------------------===//
  // Matching
  //===--------------------------------------------------------------------===//

  bool isCandidate(triton::ReduceOp reduce) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    int64_t rank = srcType.getShape().size();
    auto srcEncoding = srcType.getEncoding();

    // The combiner inside the reduce must be `addf` so that the per-iteration
    // rescale distributes over it.
    auto reductionOp = getReductionOp(reduce);
    if (!reductionOp || !isa<arith::AddFOp>(reductionOp.value()))
      return false;

    // Blocked encodings are handled by the upstream pass; here we target the
    // distributed encodings (e.g. `#mma`) it cannot.
    if (isa<ttg::BlockedEncodingAttr>(srcEncoding))
      return false;
    if (!isa<ttg::DistributedEncodingTrait>(srcEncoding) || rank <= 1)
      return false;
    // The rewrite assumes the reduction is on the innermost dim.
    if (reduce.getAxis() != rank - 1)
      return false;
    // Must admit a register-isolating free-view split of the reduce axis.
    if (!getThreadLocalityOptimizedEncoding(reduce).has_value())
      return false;

    int elemsPerThread = ttg::getElemsPerThread(srcType)[reduce.getAxis()];
    if (elemsPerThread <= 1)
      return false;

    if (!reduce->hasOneUse())
      return false;
    Operation *update = *(reduce->getUsers().begin());
    // The accumulator update must be `addf` (the running sum).
    if (!isa<arith::AddFOp>(update) || update->getNumOperands() != 2)
      return false;
    if (!update->hasOneUse())
      return false;
    OpOperand &yieldUse = *(update->getUses().begin());
    auto yieldOp = dyn_cast<scf::YieldOp>(yieldUse.getOwner());
    if (!yieldOp)
      return false;
    unsigned argNum = yieldUse.getOperandNumber();

    auto forOp = dyn_cast<scf::ForOp>(reduce->getBlock()->getParentOp());
    if (!forOp || forOp.getBody() != yieldOp->getBlock())
      return false;
    Value oldAccum = forOp.getInitArgs()[argNum];
    if (!oldAccum.getDefiningOp<arith::ConstantOp>())
      return false;

    Value blockArg = forOp.getRegionIterArgs()[argNum];
    Value reduceResult = reduce->getResult(0);
    Value accumOperand = (update->getOperand(0) == reduceResult)
                             ? update->getOperand(1)
                             : update->getOperand(0);
    // The accumulator operand must reach the loop-carried block arg through a
    // multiplicative rescale chain whose other operands are liftable.
    return isRebuildableChain(accumOperand, blockArg, reduceResult, oldAccum);
  }

  // Returns true iff `v` reaches `blockArg` through a chain of `mulf` ops whose
  // off-path operands are liftable per-row tensors (broadcastable to the
  // partial accumulator shape).
  bool isRebuildableChain(Value v, Value blockArg, Value reduceResult,
                          Value oldAccum) const {
    if (v == blockArg || v == reduceResult)
      return true;
    Operation *def = v.getDefiningOp();
    if (def && isa<arith::MulFOp>(def) &&
        dependsOn(v, blockArg, reduceResult)) {
      for (Value o : def->getOperands())
        if (!isRebuildableChain(o, blockArg, reduceResult, oldAccum))
          return false;
      return true;
    }
    // An on-path value that is not a `mulf` cannot be rebuilt safely.
    if (dependsOn(v, blockArg, reduceResult))
      return false;
    return isLiftable(v, oldAccum);
  }

  bool isLiftable(Value v, Value oldAccum) const {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    auto a = dyn_cast<RankedTensorType>(oldAccum.getType());
    return t && a && t.getRank() == a.getRank() && t.getShape() == a.getShape();
  }

  bool dependsOn(Value v, Value a, Value b) const {
    llvm::SmallPtrSet<Operation *, 16> visited;
    return dependsOnImpl(v, a, b, visited);
  }
  bool dependsOnImpl(Value v, Value a, Value b,
                     llvm::SmallPtrSet<Operation *, 16> &visited) const {
    if (v == a || v == b)
      return true;
    Operation *def = v.getDefiningOp();
    if (!def || !visited.insert(def).second)
      return false;
    for (Value o : def->getOperands())
      if (dependsOnImpl(o, a, b, visited))
        return true;
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Rewriting
  //===--------------------------------------------------------------------===//

  void rewrite(IRRewriter &builder, ModuleOp mod,
               triton::ReduceOp reduce) const {
    builder.setInsertionPoint(reduce);
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    int64_t rank = srcType.getShape().size();

    Attribute view3d = getThreadLocalityOptimizedEncoding(reduce).value();
    auto viewOpTensorShape = getThreadLocalityOptimizedShape(reduce);
    auto viewOpTensorType = RankedTensorType::get(
        viewOpTensorShape, srcType.getElementType(), view3d);
    Attribute slice2d = ttg::SliceEncodingAttr::get(
        mod.getContext(), rank, cast<ttg::DistributedEncodingTrait>(view3d));

    Operation *oldUpdate = *(reduce->getUsers().begin());
    OpOperand &yieldUse = *(oldUpdate->getUses().begin());
    unsigned argNum = yieldUse.getOperandNumber();
    auto forOp = dyn_cast<scf::ForOp>(reduce->getBlock()->getParentOp());
    Value blockArg = forOp.getRegionIterArgs()[argNum];
    auto blockArgNum = cast<BlockArgument>(blockArg).getArgNumber();
    Value oldAccum = forOp.getInitArgs()[argNum];
    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    // The partial accumulator: shape = view shape minus the trailing register
    // dim, layout = slice2d.
    SmallVector<int64_t> accumShape(viewOpTensorShape.begin(),
                                    viewOpTensorShape.end() - 1);
    auto partialType = RankedTensorType::get(
        accumShape, cast<RankedTensorType>(oldAccum.getType()).getElementType(),
        slice2d);

    auto newAccum =
        createAccum(builder, reduce, oldAccum, viewOpTensorShape, slice2d);
    auto newLoop = replaceForOpWithNewSignature(
        builder, forOp, ValueRange{newAccum->getResult(0)});
    auto newReduce = createReduce(builder, reduce, viewOpTensorType);
    Value reduceResult = reduce->getResult(0);
    auto newUpdate = createUpdate(builder, newLoop, newReduce, oldUpdate,
                                  blockArg, reduceResult, partialType);
    createYield(builder, newLoop, oldYield, newUpdate->getResult(0),
                blockArgNum);
    auto newReduce2 = createPostLoopReduce(builder, newLoop, reduce);
    Type destType = oldAccum.getType();
    auto cvtLayout = createConvertLayout(builder, destType, newReduce2);
    auto finalOp = incorporateOriginalAccumulatorValue(builder, oldUpdate,
                                                       cvtLayout, oldAccum);
    // The loop-carried accumulator result (now a passthrough constant) may have
    // multiple post-loop uses (e.g. FA `acc / l_i` and the store of `l_i`);
    // route all of them to the reduced+rescaled final value.
    newLoop.getResult(argNum).replaceAllUsesWith(finalOp->getResult(0));

    oldYield.erase();
    forOp.erase();
  }

  // Rebuild the accumulator-side value chain on the split partial. `blockArg`
  // maps to the new partial block arg; `reduceResult` maps to the thread-local
  // reduce; off-path operands (the rescale, e.g. `alpha`) are broadcast into
  // the partial layout.
  Value rebuildPartial(OpBuilder &builder, Value v, Value blockArg,
                       Value newArg, Value reduceResult, Value newReduce,
                       RankedTensorType partialType) const {
    if (v == blockArg)
      return newArg;
    if (v == reduceResult)
      return newReduce;
    Operation *def = v.getDefiningOp();
    if (def && isa<arith::MulFOp>(def) &&
        dependsOn(v, blockArg, reduceResult)) {
      IRMapping mapping;
      for (Value o : def->getOperands())
        mapping.map(o, rebuildPartial(builder, o, blockArg, newArg,
                                      reduceResult, newReduce, partialType));
      return cloneWithInferType(builder, def, mapping)->getResult(0);
    }
    return liftToPartial(builder, v, partialType);
  }

  // Broadcast a per-row value `v` (shape [M], 1D) to the partial accumulator
  // shape/layout ([M, lanePart], slice2d) via expand_dims + broadcast +
  // convert_layout.
  Value liftToPartial(OpBuilder &builder, Value v,
                      RankedTensorType partialType) const {
    Location loc = v.getLoc();
    int64_t newDimAxis = partialType.getRank() - 1;
    auto expanded = triton::ExpandDimsOp::create(builder, loc, v, newDimAxis);
    auto expandedTy = cast<RankedTensorType>(expanded.getType());
    auto bType = RankedTensorType::get(partialType.getShape(),
                                       partialType.getElementType(),
                                       expandedTy.getEncoding());
    Value bcast = triton::BroadcastOp::create(builder, loc, bType, expanded);
    return triton::gpu::ConvertLayoutOp::create(builder, loc, partialType,
                                                bcast);
  }

  Operation *createUpdate(OpBuilder &builder, scf::ForOp &loop,
                          Operation *newReduce, Operation *oldUpdate,
                          Value blockArg, Value reduceResult,
                          RankedTensorType partialType) const {
    auto newArgNum = loop.getBody()->getNumArguments() - 1;
    auto newArg = loop.getBody()->getArgument(newArgNum);
    builder.setInsertionPointAfter(newReduce);
    IRMapping mapping;
    for (Value operand : oldUpdate->getOperands()) {
      Value mapped =
          (operand == reduceResult)
              ? newReduce->getResult(0)
              : rebuildPartial(builder, operand, blockArg, newArg, reduceResult,
                               newReduce->getResult(0), partialType);
      mapping.map(operand, mapped);
    }
    return cloneWithInferType(builder, oldUpdate, mapping);
  }

  //===--------------------------------------------------------------------===//
  // Shared machinery (adapted from upstream OptimizeThreadLocality)
  //===--------------------------------------------------------------------===//

  std::optional<Operation *> getReductionOp(triton::ReduceOp reduce) const {
    if (reduce->getNumRegions() != 1)
      return std::nullopt;
    Region &region = reduce->getRegion(0);
    if (region.getBlocks().size() != 1)
      return std::nullopt;
    Block &block = region.front();
    auto body = block.without_terminator();
    if (std::distance(body.begin(), body.end()) != 1)
      return std::nullopt;
    return std::optional<Operation *>(&block.front());
  }

  Operation *incorporateOriginalAccumulatorValue(OpBuilder &builder,
                                                 Operation *oldUpdate,
                                                 Operation *cvtLayout,
                                                 Value oldAccum) const {
    builder.setInsertionPointAfter(cvtLayout);
    IRMapping mapping;
    mapping.map(oldUpdate->getOperand(0), oldAccum);
    mapping.map(oldUpdate->getOperand(1), cvtLayout->getResult(0));
    return cloneWithInferType(builder, oldUpdate, mapping);
  }

  Operation *createConvertLayout(OpBuilder &builder, Type destType,
                                 Operation *newReduce) const {
    builder.setInsertionPointAfter(newReduce);
    return ttg::ConvertLayoutOp::create(builder, newReduce->getLoc(), destType,
                                        newReduce->getResult(0));
  }

  Operation *createPostLoopReduce(OpBuilder &builder, scf::ForOp &loop,
                                  triton::ReduceOp &reduce) const {
    auto resultIndex =
        loop.getBody()->getNumArguments() - 1 - loop.getNumInductionVars();
    auto newLoopResult = loop.getResult(resultIndex);
    builder.setInsertionPointAfter(loop);
    IRMapping mapping;
    mapping.map(*(reduce.getOperands().begin()), newLoopResult);
    return cloneWithInferType(builder, &(*reduce), mapping);
  }

  Operation *createYield(OpBuilder &builder, scf::ForOp &loop,
                         scf::YieldOp &oldYield, Value newUpdate,
                         int oldAccumBlockArgNum) const {
    builder.setInsertionPoint(oldYield);
    SmallVector<Value> yieldValues = llvm::to_vector(oldYield.getOperands());
    yieldValues[oldAccumBlockArgNum - 1] =
        loop.getBody()->getArgument(oldAccumBlockArgNum);
    yieldValues.push_back(newUpdate);
    return scf::YieldOp::create(builder, oldYield.getLoc(), yieldValues);
  }

  Operation *createReduce(OpBuilder &builder, triton::ReduceOp reduce,
                          Type viewOpTensorType) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto rank = srcType.getShape().size();
    builder.setInsertionPointAfter(reduce);
    IRMapping mapping;
    for (auto operand : reduce.getOperands()) {
      auto viewOp = triton::ReshapeOp::create(
          builder, reduce.getLoc(), viewOpTensorType, operand,
          /*allowReorder=*/true, /*efficientLayout=*/true);
      mapping.map(operand, viewOp);
    }
    auto newReduce = cloneWithInferType(builder, &(*reduce), mapping);
    newReduce->setAttr("axis", builder.getI32IntegerAttr(rank));
    if (auto typeInfer = dyn_cast<InferTypeOpInterface>(newReduce)) {
      SmallVector<Type, 1> newTypes;
      if (succeeded(typeInfer.inferReturnTypes(
              newReduce->getContext(), newReduce->getLoc(),
              newReduce->getOperands(), newReduce->getAttrDictionary(),
              newReduce->getPropertiesStorage(), newReduce->getRegions(),
              newTypes))) {
        for (size_t i = 0; i < newTypes.size(); i++)
          newReduce->getResult(i).setType(newTypes[i]);
      }
    }
    return newReduce;
  }

  std::optional<TypedAttr> getNeutralElement(Operation *op) const {
    return mlir::arith::getNeutralElement(op);
  }

  Operation *createAccum(OpBuilder &builder, triton::ReduceOp reduce,
                         Value &oldAccum, SmallVector<int64_t> &shape,
                         Attribute &slice2d) const {
    SmallVector<int64_t> accumShape(shape.begin(), shape.end() - 1);
    auto elemType = cast<RankedTensorType>(oldAccum.getType()).getElementType();
    auto accumType = RankedTensorType::get(accumShape, elemType, slice2d);
    builder.setInsertionPointAfter(oldAccum.getDefiningOp());
    auto reductionOp = getReductionOp(reduce);
    assert(reductionOp && "Processing a reduce that is not supported!");
    auto neutralVal = getNeutralElement(reductionOp.value());
    assert(neutralVal && "Could not find neutral value for reduction op!");
    auto denseAttr = DenseElementsAttr::get(accumType, neutralVal.value());
    return arith::ConstantOp::create(builder, oldAccum.getLoc(), accumType,
                                     denseAttr);
  }

  SmallVector<int64_t>
  getThreadLocalityOptimizedShape(triton::ReduceOp reduce) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto srcShape = srcType.getShape();
    auto rank = srcShape.size();
    auto elemsPerThread = ttg::getElemsPerThread(srcType)[reduce.getAxis()];
    SmallVector<int64_t> viewOpTensorShape(srcShape.begin(), srcShape.end());
    viewOpTensorShape.push_back(1);
    viewOpTensorShape[reduce.getAxis()] /= elemsPerThread;
    viewOpTensorShape[rank] = elemsPerThread;
    return viewOpTensorShape;
  }

  // Build the rank+1 "thread locality" view encoding from the source
  // LinearLayout: route the register-driven bits of the reduce axis into the
  // trailing dim, keep lane/warp bits in the shrunk axis dim. Bases are
  // preserved so the reshape is a free view. Returns nullopt when the register
  // part cannot be cleanly isolated.
  std::optional<Attribute>
  getThreadLocalityOptimizedEncoding(triton::ReduceOp reduce) const {
    auto srcType = cast<RankedTensorType>(reduce.getOperands()[0].getType());
    auto srcShape = srcType.getShape();
    int rank = srcShape.size();
    int axis = reduce.getAxis();
    auto *ctx = srcType.getContext();

    int ept = ttg::getElemsPerThread(srcType)[axis];
    if (ept <= 1 || !llvm::isPowerOf2_32(ept))
      return std::nullopt;
    int eptLog2 = llvm::Log2_32(ept);

    triton::LinearLayout srcLL =
        ttg::toLinearLayout(srcShape, srcType.getEncoding());
    auto kReg = StringAttr::get(ctx, "register");
    auto regIt = srcLL.getBases().find(kReg);
    if (regIt == srcLL.getBases().end())
      return std::nullopt;
    auto axisDim = triton::standardOutDimNames(ctx, rank)[axis];
    int axisOutIdx = srcLL.getOutDimIndex(axisDim);
    int axisSize = srcShape[axis];
    int axisBits = llvm::Log2_32(axisSize);

    // Bit positions of the reduce-axis output driven purely by register.
    llvm::SmallDenseSet<int> regBits;
    for (const auto &b : regIt->second) {
      int axisVal = b[axisOutIdx];
      if (axisVal == 0)
        continue;
      bool pureAxis = true;
      for (int d = 0; d < (int)b.size(); ++d)
        if (d != axisOutIdx && b[d] != 0)
          pureAxis = false;
      if (!pureAxis || !llvm::isPowerOf2_32(axisVal))
        return std::nullopt;
      regBits.insert(llvm::Log2_32(axisVal));
    }
    if ((int)regBits.size() != eptLog2)
      return std::nullopt;

    SmallVector<int> bitToAxis(axisBits, -1);
    SmallVector<int> bitToReg(axisBits, -1);
    int axisPos = 0, regPos = 0;
    for (int b = 0; b < axisBits; ++b) {
      if (regBits.contains(b))
        bitToReg[b] = regPos++;
      else
        bitToAxis[b] = axisPos++;
    }

    std::vector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>>
        newBases;
    for (const auto &[inDim, inBases] : srcLL.getBases()) {
      bool isReg = (inDim == kReg);
      std::vector<std::vector<int32_t>> nb;
      nb.reserve(inBases.size());
      for (const auto &b : inBases) {
        std::vector<int32_t> v(b.begin(), b.end());
        int axisVal = v[axisOutIdx];
        int newAxis = 0, regVal = 0;
        for (int bit = 0; bit < axisBits; ++bit) {
          if (!(axisVal & (1 << bit)))
            continue;
          if (bitToReg[bit] >= 0) {
            if (!isReg)
              return std::nullopt;
            regVal |= (1 << bitToReg[bit]);
          } else {
            newAxis |= (1 << bitToAxis[bit]);
          }
        }
        v[axisOutIdx] = newAxis;
        v.push_back(regVal);
        nb.push_back(std::move(v));
      }
      newBases.emplace_back(inDim, std::move(nb));
    }

    auto trailingDim = triton::standardOutDimNames(ctx, rank + 1)[rank];
    SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
    for (StringAttr d : srcLL.getOutDimNames()) {
      int sz = (d == axisDim) ? (axisSize / ept) : srcLL.getOutDimSize(d);
      newOutDims.push_back({d, sz});
    }
    newOutDims.push_back({trailingDim, ept});

    triton::LinearLayout viewLL(newBases, newOutDims,
                                /*requireSurjective=*/srcLL.isSurjective());
    return std::optional<Attribute>(
        ttg::LinearEncodingAttr::get(ctx, std::move(viewLL)));
  }
};

} // namespace

std::unique_ptr<Pass> createTritonILUVATARGPUMMAReduceThreadLocalityPass() {
  return std::make_unique<TritonILUVATARGPUMMAReduceThreadLocalityPass>();
}

} // namespace mlir
