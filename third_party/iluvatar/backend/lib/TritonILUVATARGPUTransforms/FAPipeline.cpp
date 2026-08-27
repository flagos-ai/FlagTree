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

#include "TritonILUVATARGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <optional>

namespace mlir {

#define GEN_PASS_DECL_TRITONILUVATARGPUFAPIPELINE
#define GEN_PASS_DEF_TRITONILUVATARGPUFAPIPELINE
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

namespace tt = triton;
namespace ttg = triton::gpu;

namespace {

// One dot operand staged through shared memory by SmeLoad:
//   tt.load -> ttg.local_alloc -> ttg.local_load -> tt.dot
struct SmeOperand {
  tt::LoadOp load;
  ttg::LocalAllocOp alloc;
  ttg::LocalLoadOp localLoad;
};

struct FALoopMatch {
  scf::ForOp loop;
  SmeOperand k;
  SmeOperand v;
  tt::DotOp qkDot;
  tt::DotOp pvDot;
  // K's pointer travels through the loop as an iter arg, so the next tile's
  // address is already computed in the body.
  BlockArgument kPtrArg;
  Value nextKPtr;
};

bool isSmeLoad(tt::LoadOp load) {
  auto ty = dyn_cast<RankedTensorType>(load.getResult().getType());
  if (!ty)
    return false;
  auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(ty.getEncoding());
  return blocked && blocked.getIsSme();
}

std::optional<SmeOperand> matchSmeOperand(Value dotOperand, Block *body) {
  auto localLoad = dotOperand.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad || localLoad->getBlock() != body || !localLoad->hasOneUse())
    return std::nullopt;
  auto alloc = localLoad.getSrc().getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc || alloc->getBlock() != body || !alloc->hasOneUse() ||
      !alloc.getSrc())
    return std::nullopt;
  auto load = alloc.getSrc().getDefiningOp<tt::LoadOp>();
  if (!load || load->getBlock() != body || !load->hasOneUse() ||
      !isSmeLoad(load))
    return std::nullopt;
  return SmeOperand{load, alloc, localLoad};
}

bool isIluvatarMmaDot(tt::DotOp dot) {
  auto ty = dyn_cast<RankedTensorType>(dot.getResult().getType());
  return ty && isa<ttg::IluvatarMmaEncodingAttr>(ty.getEncoding());
}

// Match v3.2 IluvatarFlashAttentionPipeline: only f16/bf16. Prefetching K pins
// a shared buffer across the whole loop; for fp32 that extra tile pushes
// BLOCK_M=BLOCK_N=128 past the 128KB shared limit (Q/K/V/P reuse breaks).
bool matchDtype(Value value) {
  auto ty = dyn_cast<RankedTensorType>(value.getType());
  if (!ty)
    return false;
  Type elementTy = ty.getElementType();
  return elementTy.isF16() || elementTy.isBF16();
}

// The load is replayed for another iteration by cloning it, so everything it
// reads apart from the pointer has to be available at both the new sites (once
// before the loop, once one iteration ahead).
bool onlyPtrComesFromLoop(tt::LoadOp load, scf::ForOp loop) {
  for (Value operand : llvm::drop_begin(load->getOperands())) {
    if (auto arg = dyn_cast<BlockArgument>(operand)) {
      if (arg.getOwner() == loop.getBody())
        return false;
      continue;
    }
    if (loop->isProperAncestor(operand.getDefiningOp()))
      return false;
  }
  return true;
}

std::optional<unsigned> getIterArgIndex(scf::ForOp loop, BlockArgument arg) {
  if (arg.getOwner() != loop.getBody())
    return std::nullopt;
  unsigned number = arg.getArgNumber();
  if (number < loop.getNumInductionVars())
    return std::nullopt;
  unsigned idx = number - loop.getNumInductionVars();
  if (idx >= loop.getNumRegionIterArgs())
    return std::nullopt;
  return idx;
}

bool dependsOn(Value value, Operation *producer, Block *body) {
  SmallVector<Value> worklist{value};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != body || !seen.insert(def).second)
      continue;
    if (def == producer)
      return true;
    llvm::append_range(worklist, def->getOperands());
    // What a region reads is not an operand of the op holding it, so those
    // values have to be picked up from the nested ops -- the masked branch of a
    // causal flash-attention loop hides the dependency exactly this way.
    if (def->getNumRegions() != 0)
      def->walk([&](Operation *nested) {
        llvm::append_range(worklist, nested->getOperands());
      });
  }
  return false;
}

std::optional<FALoopMatch> matchFALoop(scf::ForOp loop) {
  Block *body = loop.getBody();

  // The buffer this pass keeps alive across the whole loop cannot reuse space
  // that a nested loop's own (multi-buffered) staging occupies, so on such a
  // loop the added shared footprint costs more occupancy than the prefetch buys
  // back.
  bool hasNestedLoop = false;
  body->walk([&](LoopLikeOpInterface) { hasNestedLoop = true; });
  if (hasNestedLoop)
    return std::nullopt;

  SmallVector<tt::DotOp> dots;
  for (Operation &op : body->without_terminator()) {
    if (auto dot = dyn_cast<tt::DotOp>(&op))
      dots.push_back(dot);
  }
  if (dots.size() != 2)
    return std::nullopt;
  if (!isIluvatarMmaDot(dots[0]) || !isIluvatarMmaDot(dots[1]))
    return std::nullopt;
  if (!matchDtype(dots[0].getA()) || !matchDtype(dots[0].getB()) ||
      !matchDtype(dots[1].getA()) || !matchDtype(dots[1].getB()))
    return std::nullopt;

  // Both dots take their A operand from shared memory: A of the first is the K
  // tile, A of the second the V tile. That is the flash-attention shape this
  // pass targets; anything else is left alone.
  std::optional<SmeOperand> k = matchSmeOperand(dots[0].getA(), body);
  std::optional<SmeOperand> v = matchSmeOperand(dots[1].getA(), body);
  if (!k || !v)
    return std::nullopt;

  // The second dot has to consume the first one's result, i.e. the loop really
  // is softmax(QK)*V. That is what puts the whole softmax between the point
  // where K stops being needed and the point where the next tile is wanted.
  if (!dependsOn(dots[1].getB(), dots[0], body))
    return std::nullopt;

  auto kPtrArg = dyn_cast<BlockArgument>(k->load.getPtr());
  if (!kPtrArg)
    return std::nullopt;
  std::optional<unsigned> idx = getIterArgIndex(loop, kPtrArg);
  if (!idx)
    return std::nullopt;
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  if (*idx >= yieldOp.getNumOperands())
    return std::nullopt;
  Value nextKPtr = yieldOp.getOperand(*idx);
  Operation *nextKPtrOp = nextKPtr.getDefiningOp();
  if (!nextKPtrOp || nextKPtrOp->getBlock() != body ||
      !isa<tt::AddPtrOp>(nextKPtrOp))
    return std::nullopt;

  if (!onlyPtrComesFromLoop(k->load, loop))
    return std::nullopt;

  return FALoopMatch{loop, *k, *v, dots[0], dots[1], kPtrArg, nextKPtr};
}

LogicalResult rewriteFALoop(FALoopMatch &match) {
  scf::ForOp loop = match.loop;
  Operation *nextKPtrOp = match.nextKPtr.getDefiningOp();
  IRRewriter builder(loop.getContext());

  // Stage the first K tile before the loop. The buffer has to be mutable so
  // that later iterations can overwrite it in place.
  builder.setInsertionPoint(loop);
  Value initKPtr = loop.getTiedLoopInit(match.kPtrArg)->get();
  Operation *initLoad = builder.clone(*match.k.load.getOperation());
  initLoad->setOperand(0, initKPtr);
  auto allocTy = match.k.alloc.getType();
  auto bufTy = ttg::MemDescType::get(
      allocTy.getShape(), allocTy.getElementType(), allocTy.getEncoding(),
      allocTy.getMemorySpace(), /*mutableMemory=*/true);
  auto buf = ttg::LocalAllocOp::create(builder, match.k.alloc.getLoc(), bufTy,
                                       initLoad->getResult(0));

  scf::ForOp newLoop =
      replaceForOpWithNewSignature(builder, loop, {buf.getResult()});
  loop.erase();
  Value curBuf = newLoop.getRegionIterArgs().back();

  // The dot now reads the tile the previous iteration fetched.
  match.k.localLoad.getSrcMutable().assign(curBuf);

  // Issue the next iteration's fetch once the current tile has been consumed,
  // i.e. after the QK dot, so the transfer overlaps softmax and the PV dot.
  // On the last iteration the address is clamped to the current tile: the
  // fetched data is dead, but the access stays in bounds.
  if (match.pvDot->isBeforeInBlock(nextKPtrOp))
    nextKPtrOp->moveBefore(match.pvDot.getOperation());
  builder.setInsertionPoint(match.pvDot);
  Location loc = match.k.load.getLoc();
  Value nextIv = arith::AddIOp::create(builder, loc, newLoop.getInductionVar(),
                                       newLoop.getStep());
  Value hasNext = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        nextIv, newLoop.getUpperBound());
  Value nextPtr = arith::SelectOp::create(builder, loc, hasNext, match.nextKPtr,
                                          match.kPtrArg);
  Operation *nextLoad = builder.clone(*match.k.load.getOperation());
  nextLoad->setOperand(0, nextPtr);
  ttg::LocalStoreOp::create(builder, loc, nextLoad->getResult(0), curBuf);

  match.k.alloc.erase();
  match.k.load.erase();

  appendToForOpYield(newLoop, {curBuf});
  return success();
}

} // anonymous namespace

class TritonILUVATARGPUFAPipelinePass
    : public impl::TritonILUVATARGPUFAPipelineBase<
          TritonILUVATARGPUFAPipelinePass> {

public:
  using Base =
      impl::TritonILUVATARGPUFAPipelineBase<TritonILUVATARGPUFAPipelinePass>;

  TritonILUVATARGPUFAPipelinePass() = default;
  explicit TritonILUVATARGPUFAPipelinePass(int32_t numStages) {
    this->numStages = numStages;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool runPrefetch = this->numStages <= 1;
    if (runPrefetch) {
      std::string env =
          tt::tools::getStrEnv("TRITON_ILUVATAR_MR_FA_PIPELINE_OPT");
      if (auto enabled = tt::tools::isEnvValueBool(env))
        runPrefetch = *enabled;
    }
    if (!runPrefetch)
      return;

    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : loops) {
      std::optional<FALoopMatch> match = matchFALoop(loop);
      if (!match)
        continue;
      (void)rewriteFALoop(*match);
    }
  }
};

std::unique_ptr<Pass> createTritonILUVATARGPUFAPipelinePass(int numStages) {
  return std::make_unique<TritonILUVATARGPUFAPipelinePass>(numStages);
}

} // namespace mlir
