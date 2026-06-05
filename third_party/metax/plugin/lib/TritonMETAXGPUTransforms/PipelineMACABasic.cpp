/*
 * 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights
 * Reserved.
 */
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#ifdef USE_MACA
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#endif

using llvm::MapVector;
using namespace mlir;
using ::mlir::triton::gpu::MACAMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::SwizzledSharedEncodingAttr;
using namespace maca::debug;
namespace ttg = triton::gpu;
namespace tt = triton;

#define int_attr(num) builder.getI64IntegerAttr(num)

namespace {

// Pass named attrs (e.g., tt.contiguity) from Triton to Triton
void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
#ifdef USE_MACA
  NamedAttrList attrs = op->getAttrs();
#else
  NamedAttrList attrs = op->getDiscardableAttrs();
#endif
  // Collect the attributes to propagate: the ones in dictAttrs and not yet on
  // the operation.
  SmallVector<NamedAttribute> toPropagate;
  for (const NamedAttribute attr : dictAttrs.getValue()) {
    if (!attrs.get(attr.getName()))
      toPropagate.push_back(attr);
  }
  // If we found any, let's set them here as a single step.
  if (toPropagate.size()) {
    attrs.append(toPropagate);
#ifdef USE_MACA
    op->setAttrs(attrs);
#else
    op->setDiscardableAttrs(attrs);
#endif
  }
}

class LoopPipeliner {
  /// Cache of ForOp and YieldOp related to this pipeliner.
  scf::ForOp forOp;
  scf::YieldOp yieldOp;

  /// Loads to be pipelined
  SetVector<Value> validLoads;
#ifdef USE_MACA
  /// Loads to be generated.
  SetVector<Value> newLoads;
  /// ops other than loads to be generated.
  SmallVector<Value> newOps;
  /// numStages allocated for shared memory.
  int numStagesForSHM = 2;

  /// control the pipeline load num, default -1: all the load
  int pipelineLoadNum = -1;

  /// if collect all ops can be pipelined.
  bool isFullStage = false;

  /// if only use [1 x shared_buffer_shape] in pipeline
  bool isSingleShm = false;

  /// deps op of dot operands in loop
  SetVector<Operation *> dotsDeps;

  /// deps op with only blocked layout.
  SetVector<Operation *> blockedDeps;

  /// deps op with only B2S or B2Dot layout conversion.
  SetVector<Operation *> storedDeps;

  /// deps op of dot operands out of loop
  SetVector<Operation *> dotsDepsOutOfLoop;

  /// dot deps => local load
  DenseMap<Value, Value> elemsMapping;

  /// load => dot operands for yield op.
  DenseMap<Value, Value> nextDotOperands;

  /// cvt => buffer at stage N
  DenseMap<Value, Value> cvtStageBuffer;

  /// cvt => dep op
  DenseMap<Value, Value> cvtMapping;

  /// valid cvts
  SetVector<Value> validCvts;

  /// load => dot, the dot which use the load
  DenseMap<Value, Value> loadDotMapping;
#endif
  /// The value that each load will be mapped to (after layout conversion)
  DenseMap<Value, Value> loadsMapping;
  /// load => buffer
  DenseMap<Value, Value> loadsBuffer;
  /// load => buffer type (with shared layout after swizzling)
#ifdef USE_MACA
  DenseMap<Value, ttg::MemDescType> loadsBufferType;
#else
  DenseMap<Value, RankedTensorType> loadsBufferType;
#endif
  /// load => buffer at stage N
  DenseMap<Value, SmallVector<Value>> loadStageBuffer;
  /// load => at stage N
  DenseMap<Value, SmallVector<Value>> loadStage;
  /// load => after extract
  DenseMap<Value, Value> loadsExtract;

  /// Iterator values
  Value pipelineIterIdx;
  Value loopIterIdx;
  Value nextIV;

  /// Yield values
  SmallVector<Value> nextBuffers;
  SmallVector<Operation *> newLocalStores;
  SmallVector<Value> nextLoads;
  SmallVector<Value> nextOps;
  SmallVector<Value> extractSlices;
  SmallVector<Value> yieldValues;

  /// The number of stages in the pipeline.
  /// Stages in the range of [0, numStages-1) are in the prologue.
  /// numStages-1 is appended after the loop body.
  int numStages;

  /// Arg indicies
  size_t bufferIdx, newLoadIdx, newOpIdx, sliceIdx, depArgsBeginIdx, ivIndex;
  DenseMap<BlockArgument, size_t> depArgsIdx;

  /// value (in loop) => value at stage N
  DenseMap<Value, SmallVector<Value>> valueMapping;
  /// loop iter arg => value
  DenseMap<BlockArgument, Value> depArgsMapping;
  /// forOp value => newForOp value
  IRMapping mapping;
  /// forOp value => prefetch value
  IRMapping nextMapping;

  /// Dependency ops by program order
  SmallVector<Operation *> orderedDeps;

  /// arg => source operand defined stages
  DenseMap<BlockArgument, DenseSet<int>> immediateArgStages;

  /// block arguments that loads depend on
  SetVector<BlockArgument> depArgs;

  /// operation => source operand defined stages
  DenseMap<Operation *, DenseSet<int>> immediateOpStages;

  /// operations that loads depend on
  SetVector<Operation *> depOps;

  /// Collect all pipelinable ops
  LogicalResult collectOps(SetVector<Operation *> &ops);

  /// Collect values that `v` depends on and are defined inside the loop
  void collectValueDep(Value v, int stage, SetVector<Value> &opDeps);

  /// Collect all op dependencies
  void collectDeps(SetVector<Operation *> &ops,
                   MapVector<Operation *, SetVector<Value>> &opDeps);

  /// Check if none of the ops has valid uses
  LogicalResult checkOpUses(SetVector<Operation *> &ops);

  /// Check if ops have dependencies that are not pipelinable
  void checkOpDeps(SetVector<Operation *> &ops);

  void createBufferTypes();

  void createOrderedDeps();

  /// Return the stage at which `v` is defined prior to `stage`
  int getValueDefStage(Value v, int stage);

  /// Map `origin` to `newValue` at `stage`
  void setValueMapping(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at `stage` according to the association between
  /// yieldOp and forOp
  void setValueMappingYield(Value origin, Value newValue, int stage);

  /// Map `origin` to `newValue` at the next stage according to the association
  /// between yieldOp and forOp
  void setValueMappingYield(scf::ForOp newForOp, Value origin, Value newValue);

  /// Return the value mapped to `origin` at `stage`, if it exists.
  Value lookupOrDefault(Value origin, int stage);

  /// Get the load mask for `loadOp`, given the mapped mask `mappedMask` (if
  /// exists) and the current iteration's `loopCond`.
  Value getLoadMask(triton::LoadOp loadOp, Value mappedMask, Value loopCond,
                    OpBuilder &builder);

  /// Return an empty buffer of size <numStages, ...>
  ttg::LocalAllocOp allocateEmptyBuffer(triton::LoadOp loadOp,
                                        OpBuilder &builder);

  /// Collect all args of the new loop
  SmallVector<Value> collectNewLoopArgs();

  /// Clone the forOp and return the new forOp
  scf::ForOp cloneForOp(ArrayRef<Value> newLoopArgs, OpBuilder &builder);

  /// Prefetch the next iteration for `newForOp`
  void prefetchNextIteration(scf::ForOp newForOp, OpBuilder &builder);

  /// Assemble `newForOp`'s yield op
  void finalizeYield(scf::ForOp newForOp, OpBuilder &builder);

public:
  LoopPipeliner(scf::ForOp forOp, int numStages, int pipelineLoadNum,
                bool isFullStage, bool isSingleShm)
      : forOp(forOp), numStages(numStages), pipelineLoadNum(pipelineLoadNum),
        isFullStage(isFullStage), isSingleShm(isSingleShm) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  /// Collect loads to pipeline. Return success if we can pipeline this loop
  LogicalResult initialize();

  /// Emit pipelined loads (before loop body)
  void emitPrologue();

  /// emit pipelined loads (after loop body)
  void emitEpilogue();

  /// create the new ForOp (add new args & insert prefetched ops)
  scf::ForOp createNewForOp();

  friend class TritonMETAXGPUPipelineMACAPass;
};

/// Collect loads to pipeline. Return success if we can pipeline this loop
LogicalResult LoopPipeliner::collectOps(SetVector<Operation *> &ops) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  mlir::triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp)
    if (auto loadOp = dyn_cast<triton::LoadOp>(&op)) {
#ifndef USE_MACA
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);

      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy || tensorTy.getRank() < 2)
        continue;
      auto ty = tensorTy.getElementType()
                    .cast<triton::PointerType>()
                    .getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();
      if (width >= 32)
        ops.insert(loadOp);
#else
      auto ptr = loadOp.getPtr();
      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy || tensorTy.getRank() < 2)
        continue;
      if (pipelineLoadNum == -1 || ops.size() < pipelineLoadNum)
        ops.insert(loadOp);
#endif
    }

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::collectValueDep(Value v, int stage,
                                    SetVector<Value> &deps) {
  // Loop-invariant value, skip
  // if (v.getParentRegion() != &forOp.getLoopBody())
  if (v.getParentRegion() != forOp.getRegion())
    return;

  // Since we only need to peel the loop numStages-1 times, don't worry
  // about depends that are too far away
  if (stage < 0)
    return;

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getArgNumber() > 0) {
      deps.insert(v);
      collectValueDep(yieldOp->getOperand(arg.getArgNumber() - 1), stage - 1,
                      deps);
    }
  } else { // value
    Operation *defOp = v.getDefiningOp();
    deps.insert(v);
#ifdef USE_MACA
    // recursively collect all nested ops deps.
    defOp->walk<WalkOrder::PreOrder>([&](Operation *nested) {
      for (OpOperand &operand : nested->getOpOperands()) {
        Operation *def = operand.get().getDefiningOp();
        if ((def && !defOp->isAncestor(def)) ||
            isa<BlockArgument>(operand.get()))
          collectValueDep(operand.get(), stage, deps);
      }
    });
#else
    for (Value op : v.getDefiningOp()->getOperands())
      collectValueDep(op, stage, deps);
#endif
  }
}

void LoopPipeliner::collectDeps(
    SetVector<Operation *> &ops,
    MapVector<Operation *, SetVector<Value>> &valueDeps) {
  for (auto op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      valueDeps[op] = deps;
    }
  }
}

LogicalResult LoopPipeliner::checkOpUses(SetVector<Operation *> &ops) {
  DenseSet<Operation *> invalidOps;
  // Collect all ops' dependencies
  MapVector<Operation *, SetVector<Value>> opDeps;
  collectDeps(ops, opDeps);

  for (Operation *op : ops) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      // Don't pipeline valid loads that depend on other valid loads
      // (Because if a valid load depends on another valid load, this load needs
      // to wait on the other load in the prologue, which is against the point
      // of the pipeline pass)
      bool isCandidate = true;
      for (Operation *other : ops)
        if (isa<triton::LoadOp>(other))
          if (opDeps[op].contains(other->getResult(0))) {
            isCandidate = false;
            break;
          }
      // We only pipeline loads that have one covert_layout (to dot_op) use
      // TODO: lift this constraint in the future
      if (isCandidate && loadOp.getResult().hasOneUse()) {
#ifdef USE_MACA
        SetVector<Operation *> dotDeps_;
#endif
        isCandidate = false;
        Operation *use = *loadOp.getResult().getUsers().begin();

        // Advance to the first conversion as long as the use resides in shared
        // memory and it has a single use itself
        while (use) {
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            break;
          auto tensorType =
              dyn_cast<RankedTensorType>(use->getResult(0).getType());
          auto memType =
              dyn_cast<ttg::MemDescType>(use->getResult(0).getType());
          if (tensorType && !mlir::isa<ttg::SwizzledSharedEncodingAttr>(
                                tensorType.getEncoding()))
            break;
          if (memType && !mlir::isa<ttg::SwizzledSharedEncodingAttr>(
                             memType.getEncoding()))
            break;
#ifdef USE_MACA
          dotDeps_.insert(use);
#endif
          use = *use->getResult(0).getUsers().begin();
        }

        if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
          if (auto tensorType = dyn_cast<RankedTensorType>(
                  convertLayout.getResult().getType())) {
            if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
                    tensorType.getEncoding())) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
              Operation *dot = *convertLayout.getResult().getUsers().begin();
              if (auto dotOp = dyn_cast<triton::DotOp>(dot)) {
                loadDotMapping[loadOp] = dotOp;
              }
#ifdef USE_MACA
              dotDeps_.insert(use);
              for (Operation *op : dotDeps_) {
                dotsDeps.insert(op);
                elemsMapping[op->getResult(0)] = convertLayout;
              }
#endif
            }
          }
        }

        if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(use)) {
          if (auto tensorType = dyn_cast<RankedTensorType>(
                  convertLayout.getResult().getType())) {
            if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
                    tensorType.getEncoding())) {
              isCandidate = true;
              loadsMapping[loadOp] = convertLayout;
#ifdef USE_MACA
              dotDeps_.insert(use);
              for (Operation *op : dotDeps_) {
                dotsDeps.insert(op);
                elemsMapping[op->getResult(0)] = convertLayout;
              }
#endif
            }
          }
        }
      } else
        isCandidate = false;

      if (!isCandidate)
        invalidOps.insert(loadOp);
      else
        validLoads.insert(loadOp);
    }
  }

  for (Operation *op : invalidOps)
    ops.remove(op);

  if (ops.empty())
    return failure();
  else
    return success();
}

void LoopPipeliner::checkOpDeps(SetVector<Operation *> &ops) {
  SetVector<BlockArgument> nonImmediateDepArgs;
  SetVector<Operation *> nonImmediateOps;
  for (Operation *op : ops) {
    for (Value v : op->getOperands()) {
      SetVector<Value> deps;
      collectValueDep(v, numStages - 1, deps);
      int defStage = getValueDefStage(v, numStages - 1);
      assert(defStage >= 0 &&
             "newLoopArgs has null args without a define op. Consider either "
             "rewrite the loop to reduce cross iteration dependencies or "
             "increase the num_stages value.");
      for (auto dep : deps) {
        auto immediate = mlir::isa<BlockArgument>(deps.front());
        if (auto arg = dyn_cast<BlockArgument>(dep)) {
          depArgs.insert(arg);
          if (immediate)
            immediateArgStages[arg].insert(defStage);
          else
            nonImmediateDepArgs.insert(arg);
        } else {
          depOps.insert(dep.getDefiningOp());
          if (immediate)
            immediateOpStages[dep.getDefiningOp()].insert(defStage);
          else
            nonImmediateOps.insert(dep.getDefiningOp());
        }
      }
    }
  }

  // XXX: We could remove the following constraints if we can rematerialize in
  // the loop.
  // Check if immediateDepArgs and nonImmediateDepArgs are disjoint.
  for (auto &[arg, stages] : immediateArgStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an argument provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateDepArgs.contains(arg) &&
             stages.contains(numStages - 2)) &&
           "Loop-carried arguments provide values for both immediate and "
           "non-immediate operands of loads. Please consider removing "
           "pre/post load instructions dependency on this argument.");
  }

  // Check if immediateOps and nonImmediateOps are disjoint.
  for (auto &[op, stages] : immediateOpStages) {
    assert(stages.size() == 1 &&
           "Triton doesn't support an operation provides values for "
           "immediate operands of loads from multiple stages. Consider "
           "removing post load instructions dependency on this argument.");
    assert(!(nonImmediateOps.contains(op) && stages.contains(numStages - 2)) &&
           "Operations provide values for both immediate and "
           "non-immediate operands of loads.  Please consider "
           "removing pre/post load instructions dependency on this "
           "operation.");
  }
}

// helpers
void LoopPipeliner::setValueMapping(Value origin, Value newValue, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    valueMapping[origin] = SmallVector<Value>(numStages);
  valueMapping[origin][stage] = newValue;
}

void LoopPipeliner::setValueMappingYield(Value origin, Value newValue,
                                         int stage) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto value = forOp.getRegionIterArgs()[yieldIdx];
      setValueMapping(value, newValue, stage);
    }
  }
}

void LoopPipeliner::setValueMappingYield(scf::ForOp newForOp, Value origin,
                                         Value newValue) {
  for (OpOperand &operand : origin.getUses()) {
    if (operand.getOwner() == yieldOp) {
      auto yieldIdx = operand.getOperandNumber();
      auto depYieldIdx = depArgsIdx[forOp.getRegionIterArgs()[yieldIdx]];
      auto originArg = forOp.getRegionIterArgs()[yieldIdx];
      nextMapping.map(originArg, newValue);
      auto newArg = newForOp.getRegionIterArgs()[depYieldIdx];
#ifdef USE_MACA
      if (depArgsMapping.find(newArg) == depArgsMapping.end())
#else
      if (!depArgsMapping.contains(newArg))
#endif
        depArgsMapping[newArg] = newValue;
    }
  }
}

Value LoopPipeliner::lookupOrDefault(Value origin, int stage) {
  if (valueMapping.find(origin) == valueMapping.end())
    return origin;
  return valueMapping[origin][stage];
}

void LoopPipeliner::createBufferTypes() {
  for (auto loadCvt : loadsMapping) {
    auto loadOp = loadCvt.first;
    Value cvt = loadCvt.second;
    auto enc = cast<RankedTensorType>(cvt.getType()).getEncoding();
    auto dotOpEnc = cast<ttg::DotOperandEncodingAttr>(enc);
    auto ty = cast<RankedTensorType>(loadOp.getType());
#ifdef USE_MACA
    //
    // match pattern:
    //   %0 load ptr
    //   %1 lalloc %0->shared
    //   %2 trans %1->shared1
    //   %3 lload %2->dot
    // we get tensor type of %1, so it is necessary to create shared encoding
    // with trans flag.
    Operation *cvt_ = cvt.getDefiningOp();
    Value cvtSrc;
    if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(cvt_))
      cvtSrc = convertLayout.getSrc();
    if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(cvt_))
      cvtSrc = convertLayout.getSrc();
    Operation *trans = cvtSrc.getDefiningOp();
    Attribute memorySpace;
    MLIRContext *ctx = loadOp.getContext();
    auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    if (auto transOp = llvm::dyn_cast<triton::TransOp>(trans)) {
      auto dstTy =
          dyn_cast<triton::gpu::TensorOrMemDesc>(transOp.getSrc().getType());
      SmallVector<int64_t> bufferShape(dstTy.getShape().begin(),
                                       dstTy.getShape().end());
      unsigned bitWidth = dstTy.getElementType().getIntOrFloatBitWidth();
      bufferShape.insert(bufferShape.begin(), numStagesForSHM);
      auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
          ty.getContext(), dotOpEnc, dstTy.getShape(), ttg::getOrder(dstTy),
          ttg::getCTALayout(dstTy.getEncoding()), dstTy.getElementType(), true);

      loadsBufferType[loadOp] =
          ttg::MemDescType::get(bufferShape, dstTy.getElementType(), sharedEnc,
                                sharedMemorySpace, /*mutable*/ true);
    } else {
      auto dstTy = dyn_cast<triton::gpu::TensorOrMemDesc>(cvtSrc.getType());
      SmallVector<int64_t> bufferShape(dstTy.getShape().begin(),
                                       dstTy.getShape().end());
      unsigned bitWidth = dstTy.getElementType().getIntOrFloatBitWidth();
      bufferShape.insert(bufferShape.begin(), numStagesForSHM);
      auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
          ty.getContext(), dotOpEnc, dstTy.getShape(), ttg::getOrder(dstTy),
          ttg::getCTALayout(dstTy.getEncoding()), dstTy.getElementType());
      loadsBufferType[loadOp] =
          ttg::MemDescType::get(bufferShape, dstTy.getElementType(), sharedEnc,
                                sharedMemorySpace, /*mutable*/ true);
    }
#else
    unsigned bitWidth = dotOpEnc.getMMAv2kWidth()
                            ? 32 / dotOpEnc.getMMAv2kWidth()
                            : ty.getElementType().getIntOrFloatBitWidth();
    bufferShape.insert(bufferShape.begin(), numStages);
    auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
        ty.getContext(), dotOpEnc, ty.getShape(), ttg::getOrder(ty), bitWidth);
    loadsBufferType[loadOp] =
        RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
#endif
  }
}

void LoopPipeliner::createOrderedDeps() {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (depOps.contains(&op))
      orderedDeps.push_back(&op);
    else if (op.getNumResults() > 0 && validLoads.contains(op.getResult(0)))
      orderedDeps.push_back(&op);
  }
  assert(depOps.size() + validLoads.size() == orderedDeps.size() &&
         "depOps contains invalid values");
}

int LoopPipeliner::getValueDefStage(Value v, int stage) {
  if (stage < 0)
    return -1;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (arg.getArgNumber() > 0)
      return getValueDefStage(yieldOp->getOperand(arg.getArgNumber() - 1),
                              stage - 1);
    llvm_unreachable("Loop induction variable should not be a dependency");
  } else
    return stage;
}

ttg::LocalAllocOp LoopPipeliner::allocateEmptyBuffer(triton::LoadOp loadOp,
                                                     OpBuilder &builder) {

  // Allocate a buffer for each pipelined tensor
  // shape: e.g. (numStages==4), <32x64xbf16> -> <4x32x64xbf16>
  Value convertLayout = loadsMapping[loadOp];
  if (auto tensorType = dyn_cast<RankedTensorType>(convertLayout.getType()))
    return builder.create<ttg::LocalAllocOp>(convertLayout.getLoc(),
                                             loadsBufferType[loadOp], Value());
  llvm_unreachable("Async copy's return should be of RankedTensorType");
}

LogicalResult LoopPipeliner::initialize() {
  // All ops that maybe pipelined
  SetVector<Operation *> ops;

  if (isFullStage) {
    bool failed =
        collectValidOp(forOp, yieldOp, numStages, loadsMapping, elemsMapping,
                       validLoads, dotsDeps, dotsDepsOutOfLoop, ops)
            .failed();
    setParentFunctionOpAttrDebug(forOp, DebugOption::kCheckCollectValidOp,
                                 failed ? DebugResult::kFail
                                        : DebugResult::kSuccess);
    if (failed)
      return failure();
  } else {
    if (collectOps(ops).failed())
      return failure();

    if (checkOpUses(ops).failed())
      return failure();
  }

  if (isSingleShm) {
    numStagesForSHM = 1;
    // TODO: support more load in single shared memory mode
    if (validLoads.size() != 1)
      return failure();
  }

  checkOpDeps(ops);

  createBufferTypes();

  createOrderedDeps();

  return success();
}

Value LoopPipeliner::getLoadMask(triton::LoadOp loadOp, Value mappedMask,
                                 Value loopCond, OpBuilder &builder) {
  Type maskType = triton::getI1SameShape(loadOp.getType());
  Value mask = loadOp.getMask();
  Value newMask;
  if (mask) {
    Value cond = loopCond;
    if (isa<RankedTensorType>(maskType)) {
      cond = builder.create<triton::SplatOp>(mask.getLoc(), maskType, loopCond);
    }
    newMask = builder.create<arith::AndIOp>(mask.getLoc(), mappedMask, cond);
  } else {
    if (isa<RankedTensorType>(maskType)) {
      newMask = builder.create<triton::SplatOp>(loopCond.getLoc(), maskType,
                                                loopCond);
    } else {
      newMask = loopCond;
    }
  }
  return newMask;
}

void LoopPipeliner::emitPrologue() {
  OpBuilder builder(forOp);
  // Get init operands for loop carried values
#ifdef USE_MACA
  for (auto [arg, operand] :
       llvm::zip(forOp.getRegionIterArgs(), forOp.getInitsMutable())) {
    setValueMapping(arg, operand.get(), 0);
  }
#else
  for (BlockArgument &arg : forOp.getRegionIterArgs()) {
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    setValueMapping(arg, operand.get(), 0);
  }
#endif

  // Emit prologue from [0, numStage-1)
  Value iv = forOp.getLowerBound();
#ifdef USE_MACA
  pipelineIterIdx =
      builder.create<arith::ConstantIntOp>(iv.getLoc(), numStagesForSHM, 32);

  // allocate buffer according to cvt
  for (Value v : validLoads) {
    auto loadOp = dyn_cast<tt::LoadOp>(v.getDefiningOp());
    auto cvt = loadsMapping[v];
    if (!validCvts.contains(cvt)) {
      cvtStageBuffer[cvt] = allocateEmptyBuffer(loadOp, builder);
      validCvts.insert(cvt);
    }
  }
#else
  pipelineIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
#endif
  for (int stage = 0; stage < numStages - 1; ++stage) {
    // Special handling for induction variable as the increment is implicit
    if (stage != 0)
      iv = builder.create<arith::AddIOp>(iv.getLoc(), iv, forOp.getStep());
    setValueMapping(forOp.getInductionVar(), iv, stage);

    // Special handling for loop condition as there is no condition in ForOp
    Value loopCond = builder.create<arith::CmpIOp>(
        iv.getLoc(), arith::CmpIPredicate::slt, iv, forOp.getUpperBound());
    for (Operation *op : orderedDeps) {
      Operation *newOp = nullptr;
      if (validLoads.contains(op->getResult(0))) {
        auto load = cast<triton::LoadOp>(op);
#ifndef USE_MACA
        // Allocate empty buffer
        if (stage == 0) {
          loadsBuffer[load] = allocateEmptyBuffer(load, builder);
          loadStageBuffer[load] = {loadsBuffer[load]};
        }
#endif
        // load => copy async
        if (auto loadOp = llvm::dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
#ifdef USE_MACA
          Operation *newLoadOp = nullptr;
          newOp = builder.create<triton::LoadOp>(
              loadOp.getLoc(), loadOp.getResult().getType(),
              lookupOrDefault(loadOp.getPtr(), stage), newMask,
              lookupOrDefault(loadOp.getOther(), stage),
              loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
          loadStage[loadOp].push_back(newOp->getResult(0));
          if (stage > 1) {
            auto newLoadOp = llvm::dyn_cast<triton::LoadOp>(newOp);
            newLoads.insert(newLoadOp);
          }
#else
          newOp = builder.create<ttg::InsertSliceAsyncOp>(
              op->getLoc(), loadsBuffer[loadOp].getType(),
              lookupOrDefault(loadOp.getPtr(), stage),
              loadStageBuffer[loadOp][stage], pipelineIterIdx, newMask,
              lookupOrDefault(loadOp.getOther(), stage), loadOp.getCache(),
              loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
          builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
          loadStageBuffer[loadOp].push_back(newOp->getResult(0));
#endif
        } else
          llvm_unreachable("This should be LoadOp");
      } else {
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          Value newMask =
              getLoadMask(loadOp, lookupOrDefault(loadOp.getMask(), stage),
                          loopCond, builder);
          newOp = builder.create<triton::LoadOp>(
              loadOp.getLoc(), loadOp.getResult().getType(),
              lookupOrDefault(loadOp.getPtr(), stage), newMask,
              lookupOrDefault(loadOp.getOther(), stage),
              loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
              loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
#ifdef USE_MACA
          addNamedAttrs(newOp, op->getAttrDictionary());
#else
          addNamedAttrs(newOp, op->getDiscardableAttrDictionary());
#endif
        } else
          newOp = builder.clone(*op);
#ifdef USE_MACA
        auto callback = [&](OpOperand *newOperand) {
          auto it = valueMapping.find(newOperand->get());
          if (it != valueMapping.end()) {
            Value replacement = it->second[stage];
            newOperand->set(replacement);
          }
        };
        // recrusively update all nested op operands.
        newOp->walk<WalkOrder::PreOrder>([&](Operation *nested) {
          for (OpOperand &operand : nested->getOpOperands()) {
            Operation *def = operand.get().getDefiningOp();
            if ((def && !newOp->isAncestor(def)) ||
                isa<BlockArgument>(operand.get()))
              callback(&operand);
          }
        });
#else
        for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
          auto it = valueMapping.find(op->getOperand(opIdx));
          if (it != valueMapping.end()) {
            Value v = it->second[stage];
            assert(v && "Value not found in valueMapping");
            newOp->setOperand(opIdx, v);
          } // else, op at opIdx is a loop-invariant value
        }
#endif
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults())) {
        Value originResult = op->getResult(dstIdx);
        if (validLoads.contains(originResult))
          break;
        setValueMapping(originResult, newOp->getResult(dstIdx), stage);
        // Update mapping for loop-carried values (args)
        setValueMappingYield(op->getResult(dstIdx), newOp->getResult(dstIdx),
                             stage + 1);
      }
    } // for (Operation *op : orderedDeps)

#ifndef USE_MACA
    // Update pipeline index
    pipelineIterIdx = builder.create<arith::AddIOp>(
        iv.getLoc(), pipelineIterIdx,
        builder.create<arith::ConstantIntOp>(iv.getLoc(), 1, 32));
#endif
    // Some values have not been used by any ops in the loop body
    for (BlockArgument arg : forOp.getRegionIterArgs())
      setValueMappingYield(arg, valueMapping[arg][stage], stage + 1);
  } // for (int stage = 0; stage < numStages - 1; ++stage)

  // async.wait & extract_slice
  // Operation *asyncWait =
  // builder.create<ttg::AsyncWaitOp>(validLoads.front().getLoc(),
  //                                  validLoads.size() * (numStages - 2));
  loopIterIdx = builder.create<arith::ConstantIntOp>(iv.getLoc(), 0, 32);
#ifdef USE_MACA
  IRMapping prefetchMapping;
  SetVector<Operation *> dotDepsWithOutDot;
  for (Value v : validLoads) {
    prefetchMapping.map(v, loadStage[v][0]);
  }
  for (Operation *op : dotsDeps) {
    bool isValid = true;
    for (Value v : op->getOperands()) {
      auto tensor = dyn_cast<RankedTensorType>(v.getType());
      if (tensor) {
        if (isa<BlockedEncodingAttr>(tensor.getEncoding())) {
          continue;
        }
      }
      isValid = false;
    }
    if (isValid) {
      dotDepsWithOutDot.insert(op);
    } else {
      continue;
    }
    for (Value v : op->getResults()) {
      auto tensor = dyn_cast<RankedTensorType>(v.getType());
      if (tensor) {
        if (isa<BlockedEncodingAttr>(tensor.getEncoding())) {
          continue;
        }
      }
      isValid = false;
    }
    if (isValid) {
      blockedDeps.insert(op);
    } else {
      storedDeps.insert(op);
    }
  }
  dotsDeps = dotDepsWithOutDot;

  // stage 0
  // %0 = load %arg
  // %1 = other blocked ops %0
  // %2 = local_store %1
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // set cvtMapping of dotsDeps.
  genDotDeps(builder, forOp, validLoads, prefetchMapping, prefetchMapping,
             loadsMapping, elemsMapping, dotsDeps, nextDotOperands,
             cvtStageBuffer, cvtMapping, zero, false);
  for (Value cvt : validCvts) {
    loadsExtract[cvt] = prefetchMapping.lookupOrDefault(cvtMapping[cvt]);
  }
  if (numStages > 2) {
    for (Value v : validLoads) {
      prefetchMapping.map(v, loadStage[v][1]);
    }
    // stage 1 if numStages > 3
    // %0 = load %arg
    // %1 = other blocked ops %0

    // stage 1 if numStages == 3
    // %0 = load %arg

    // set cvtMapping of blockedDeps.
    if (numStages > 3) {
      genDotDeps(builder, forOp, validLoads, prefetchMapping, prefetchMapping,
                 loadsMapping, elemsMapping, blockedDeps, nextDotOperands,
                 cvtStageBuffer, cvtMapping, zero, false);
      for (Value cvt : validCvts) {
        newOps.push_back(prefetchMapping.lookupOrDefault(cvtMapping[cvt]));
      }
    } else {
      for (Value load : validLoads) {
        newOps.push_back(loadStage[load][1]);
      }
    }
  }
  // stage n (n > 1)
  // %0 = load
#endif
  loopIterIdx = builder.create<arith::AddIOp>(
      loopIterIdx.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(loopIterIdx.getLoc(), 1, 32));
}

void LoopPipeliner::emitEpilogue() {
  // If there's any outstanding async copies, we need to wait for them.
  OpBuilder builder(forOp);
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointAfter(forOp);
#ifdef USE_MACA
  builder.create<ttg::BarrierSharedOp>(forOp.getLoc());
  for (auto cvt : validCvts)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), cvtStageBuffer[cvt]);
#else
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
#endif
}

SmallVector<Value> LoopPipeliner::collectNewLoopArgs() {
  // Order of new args:
  //   (original args)
  //   (insertSliceAsync buffer at stage numStages - 1) for each load
  //   (extracted tensor) for each load
  //   (depArgs at stage numStages - 1)
  //   (depArgs at stage numStages - 2)
  //   ...
  //   (iv at stage numStages - 2)
  //   (pipeline iteration index)
  //   (loop iteration index)

  // We need this to update operands for yield
  // original block arg => new arg's idx
  SmallVector<Value> newLoopArgs;
#ifdef USE_MACA
  for (auto v : forOp.getInitArgs())
    newLoopArgs.push_back(v);
#else
  for (auto v : forOp.getIterOperands())
    newLoopArgs.push_back(v);
#endif

  bufferIdx = newLoopArgs.size();
#ifdef USE_MACA
  for (auto cvt : validCvts)
    newLoopArgs.push_back(cvtStageBuffer[cvt]);
  newOpIdx = newLoopArgs.size();
  for (auto op : newOps)
    newLoopArgs.push_back(op);
  newLoadIdx = newLoopArgs.size();
  for (auto load : newLoads)
    newLoopArgs.push_back(load);
#else
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(loadStageBuffer[loadOp].back());
#endif

  sliceIdx = newLoopArgs.size();
#ifdef USE_MACA
  for (auto cvt : validCvts)
    newLoopArgs.push_back(loadsExtract[cvt]);
#else
  for (auto loadOp : validLoads)
    newLoopArgs.push_back(loadsExtract[loadOp]);
#endif

  depArgsBeginIdx = newLoopArgs.size();
  for (auto depArg : depArgs) {
    depArgsIdx[depArg] = newLoopArgs.size();
    if (immediateArgStages[depArg].contains(numStages - 2))
      // Peel off post load ops in numStage-1
      newLoopArgs.push_back(valueMapping[depArg][numStages - 2]);
    else
      newLoopArgs.push_back(valueMapping[depArg][numStages - 1]);
  }

  ivIndex = newLoopArgs.size();
  newLoopArgs.push_back(valueMapping[forOp.getInductionVar()][numStages - 2]);
  newLoopArgs.push_back(pipelineIterIdx);
  newLoopArgs.push_back(loopIterIdx);
  return newLoopArgs;
}

scf::ForOp LoopPipeliner::cloneForOp(ArrayRef<Value> newLoopArgs,
                                     OpBuilder &builder) {
  // Clone the original ForOp
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newLoopArgs);

  // Set mapping on body of the new ForOp
  builder.setInsertionPointToStart(newForOp.getBody());
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // Clone the loop body, replace original args with args of the new ForOp.
  // We want to find cvt ops that match the following pattern:
  // %0 = load %ptr
  // %1 (dotOperand) = cvt %0
  for (Operation &op : forOp.getBody()->without_terminator()) {
#ifdef USE_MACA
    if (blockedDeps.contains(&op))
      continue;
    if (auto cvtOp = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
      auto result = op.getResult(0);
      auto it = std::find(validCvts.begin(), validCvts.end(), result);
      if (it != validCvts.end()) {
        auto loadArgIdx = std::distance(validCvts.begin(), it);
        auto cvtDstTy = cast<RankedTensorType>(result.getType());
        if (mlir::isa<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding())) {
          Operation *prev_op = cvtOp.getSrc().getDefiningOp();
          auto transOp = dyn_cast<triton::TransOp>(prev_op);
          if (transOp) {
            // match pattern
            // %0 = load %ptr
            // %1 = lalloc(blocked->shared)
            // %2 = trans %1
            // %3 = lload(shared->dot) %2
            Operation *cvt1 = transOp.getSrc().getDefiningOp();
            // auto cvtOp1 = dyn_cast<triton::gpu::ConvertLayoutOp>(cvt1);
            auto cvtOp1 = dyn_cast<triton::gpu::LocalAllocOp>(cvt1);
            if (cvtOp1) {
              Value newTrans = builder.create<triton::TransOp>(
                  transOp.getLoc(),
                  newForOp.getRegionIterArgs()[sliceIdx + loadArgIdx],
                  transOp.getOrder());
              mapping.map(transOp.getResult(), newTrans);
              nextMapping.map(transOp.getResult(), newTrans);
              auto cvt = builder.create<ttg::LocalLoadOp>(transOp.getLoc(),
                                                          cvtDstTy, newTrans);
              mapping.map(result, cvt.getResult());
              nextMapping.map(result, cvt.getResult());
              continue;
            }
          }
          // We replace the use new load use with a convert layout
          auto dotEnc =
              dyn_cast<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding());
          auto cvt = builder.create<ttg::LocalLoadOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[sliceIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          nextMapping.map(result, cvt.getResult());
          continue;
        }
      }
    }
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      // match pattern
      // %0 = load %ptr
      // %1 = cvt(blocked->dot) %0
      auto result = op.getResult(0);
      auto cvtDstTy = cast<RankedTensorType>(result.getType());
      if (mlir::isa<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding())) {
        auto it = std::find(validCvts.begin(), validCvts.end(), result);
        if (it != validCvts.end()) {
          // We replace the use new load use with a convert layout
          auto loadArgIdx = std::distance(validCvts.begin(), it);
          auto dotEnc =
              dyn_cast<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding());
          auto cvt = builder.create<ttg::LocalLoadOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[sliceIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          nextMapping.map(result, cvt.getResult());
          continue;
        }
      }
    }
#else
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto result = op.getResult(0);
      auto cvtDstTy = cast<RankedTensorType>(result.getType());
      if (mlir::isa<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding())) {
        auto it =
            std::find(validLoads.begin(), validLoads.end(), op.getOperand(0));
        if (it != validLoads.end()) {
          // We replace the use new load use with a convert layout
          auto loadArgIdx = std::distance(validLoads.begin(), it);
          auto cvt = builder.create<ttg::ConvertLayoutOp>(
              result.getLoc(), cvtDstTy,
              newForOp.getRegionIterArgs()[sliceIdx + loadArgIdx]);
          mapping.map(result, cvt.getResult());
          continue;
        }
      }
    }
#endif
    cloneWithInferType(builder, &op, mapping);
  }

  return newForOp;
}

void LoopPipeliner::prefetchNextIteration(scf::ForOp newForOp,
                                          OpBuilder &builder) {
  builder.setInsertionPointToStart(newForOp.getBody());

#ifdef USE_MACA
  size_t initArgIdx = 0;
  for (auto v : forOp.getRegionIterArgs()) {
    BlockArgument nextArg = newForOp.getRegionIterArgs()[initArgIdx];
    nextMapping.map(v, nextArg);
    ++initArgIdx;
  }
#endif
  // Map the dep args of the next iteration to the dep args of the current
  size_t argIdx = 0;
  for (auto depArg : depArgs) {
    BlockArgument nextArg =
        newForOp.getRegionIterArgs()[argIdx + depArgsBeginIdx];
    nextMapping.map(depArg, nextArg);
    ++argIdx;
  }

  // Special handling for iv & loop condition
  Value curIV = newForOp.getRegionIterArgs()[ivIndex];
  nextIV = builder.create<arith::AddIOp>(newForOp.getInductionVar().getLoc(),
                                         curIV, newForOp.getStep());
  Value nextLoopCond =
      builder.create<arith::CmpIOp>(nextIV.getLoc(), arith::CmpIPredicate::slt,
                                    nextIV, newForOp.getUpperBound());

  pipelineIterIdx = newForOp.getRegionIterArgs()[ivIndex + 1];
#ifdef USE_MACA
  Value insertSliceIndex =
      builder.create<arith::RemSIOp>(nextIV.getLoc(), pipelineIterIdx,
                                     builder.create<arith::ConstantIntOp>(
                                         nextIV.getLoc(), numStagesForSHM, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex =
      builder.create<arith::RemSIOp>(nextIV.getLoc(), loopIterIdx,
                                     builder.create<arith::ConstantIntOp>(
                                         nextIV.getLoc(), numStagesForSHM, 32));
  Value extractSliceIndexI64 = builder.create<arith::IndexCastOp>(
      nextIV.getLoc(), builder.getIndexType(), extractSliceIndex);
#else
  Value insertSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
  loopIterIdx = newForOp.getRegionIterArgs()[ivIndex + 2];
  Value extractSliceIndex = builder.create<arith::RemSIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), numStages, 32));
#endif

#ifdef USE_MACA
  // initialize next buffer with loadStageBuffer
  SmallVector<Value> genLoads;
  if (numStages == 3) {
    int bufferArgIdx = 0;
    for (auto load : newOps) {
      BlockArgument nextArg =
          newForOp.getRegionIterArgs()[bufferArgIdx + newOpIdx];
      nextLoads.push_back(nextArg);
      ++bufferArgIdx;
    }
  }
  int bufferArgIdx = 0;
  for (auto loadOp : newLoads) {
    BlockArgument nextArg =
        newForOp.getRegionIterArgs()[bufferArgIdx + newLoadIdx];
    nextLoads.push_back(nextArg);
    ++bufferArgIdx;
  }

#endif
  IRMapping curMapping = nextMapping;
  for (Operation *op : orderedDeps)
    if (!validLoads.contains(op->getResult(0))) {
      if (immediateOpStages[op].contains(numStages - 2))
        // A post load op that provides values for numStage - 2
        curMapping.map(forOp.getInductionVar(), curIV);
      else
        curMapping.map(forOp.getInductionVar(), nextIV);
      Operation *nextOp;
      if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
        auto newMask =
            getLoadMask(loadOp, curMapping.lookupOrDefault(loadOp.getMask()),
                        nextLoopCond, builder);
        nextOp = builder.create<triton::LoadOp>(
            loadOp.getLoc(), loadOp.getResult().getType(),
            curMapping.lookupOrDefault(loadOp.getPtr()), newMask,
            curMapping.lookupOrDefault(loadOp.getOther()),
            loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
            loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
#ifdef USE_MACA
        addNamedAttrs(nextOp, op->getAttrDictionary());
#else
        addNamedAttrs(nextOp, op->getDiscardableAttrDictionary());
#endif
        curMapping.map(loadOp.getResult(), nextOp->getResult(0));
        nextMapping.map(loadOp.getResult(), nextOp->getResult(0));
      } else {
        nextOp = builder.clone(*op, curMapping);
        for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
          nextMapping.map(op->getResult(dstIdx), nextOp->getResult(dstIdx));
      }

      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
    }

  // loads -> async loads
  for (Operation *op : orderedDeps) {
    Operation *nextOp = nullptr;
    // Update loading mask
    if (validLoads.contains(op->getResult(0))) {
      auto loadOp = llvm::cast<triton::LoadOp>(op);
      auto mask = loadOp.getMask();
      auto newMask =
          getLoadMask(loadOp, nextMapping.lookupOrDefault(loadOp.getMask()),
                      nextLoopCond, builder);
      if (mask) {
        // If mask is defined outside the loop, don't update the map more than
        // once
        if (!(forOp.isDefinedOutsideOfLoop(mask) && nextMapping.contains(mask)))
          nextMapping.map(loadOp.getMask(), newMask);
        newMask = nextMapping.lookupOrDefault(mask);
      }
#ifdef USE_MACA
      Operation *newLoadOp = nullptr;
      newLoadOp = builder.create<triton::LoadOp>(
          loadOp.getLoc(), loadOp.getResult().getType(),
          nextMapping.lookupOrDefault(loadOp.getPtr()), newMask,
          nextMapping.lookupOrDefault(loadOp.getOther()),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      nextLoads.push_back(newLoadOp->getResult(0));
      genLoads.push_back(newLoadOp->getResult(0));
      auto curLoadOp = *nextLoads.begin();
      nextMapping.map(loadOp.getResult(), curLoadOp);
      nextLoads.erase(nextLoads.begin());
#else
      Value insertAsyncOp = builder.create<ttg::InsertSliceAsyncOp>(
          op->getLoc(), loadsBuffer[loadOp].getType(),
          nextMapping.lookupOrDefault(loadOp.getPtr()),
          newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()],
          insertSliceIndex, newMask,
          nextMapping.lookupOrDefault(loadOp.getOther()), loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile(), /*axis*/ 0);
      builder.create<ttg::AsyncCommitGroupOp>(op->getLoc());
      nextBuffers.push_back(insertAsyncOp);
      // Extract slice
      auto bufferType = insertAsyncOp.getType().cast<RankedTensorType>();
      auto bufferShape = bufferType.getShape();
      auto sliceType = loadsMapping[loadOp].getType().cast<RankedTensorType>();
      sliceType = RankedTensorType::get({bufferShape[1], bufferShape[2]},
                                        sliceType.getElementType(),
                                        loadsBufferType[loadOp].getEncoding());

      nextOp = builder.create<ttg::ExtractSliceOp>(
          op->getLoc(), sliceType, insertAsyncOp,
          SmallVector<OpFoldResult>{extractSliceIndex, int_attr(0),
                                    int_attr(0)},
          SmallVector<OpFoldResult>{int_attr(1),
                                    int_attr(sliceType.getShape()[0]),
                                    int_attr(sliceType.getShape()[1])},
          SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
      extractSlices.push_back(nextOp->getResult(0));

      // Update mapping of results
      for (unsigned dstIdx : llvm::seq(unsigned(0), op->getNumResults()))
        // If this is a loop-carried value, update the mapping for yield
        setValueMappingYield(newForOp, op->getResult(dstIdx),
                             nextOp->getResult(dstIdx));
#endif
    }
  }

  // Some values have not been used by any ops in the loop body
  for (BlockArgument arg : forOp.getRegionIterArgs())
    setValueMappingYield(newForOp, arg,
                         newForOp.getRegionIterArgs()[depArgsIdx[arg]]);

  builder.setInsertionPointToEnd(newForOp.getBody());

  // async.wait & extract_slice
#ifdef USE_MACA
  Location anchor = validLoads.front().getLoc();
  Operation *asyncWait = builder.create<ttg::SchedBoundOp>(anchor);
  Operation *arriveOp = nullptr;
  Operation *barrierOp = builder.create<ttg::BarrierSharedOp>(anchor);
  if (isFullStage) {
    auto mmaElemTy =
        dyn_cast<RankedTensorType>(validCvts[0].getType()).getElementType();
    // mma_other_inst_num - 2 because other instruction like ldg may insert
    // between mma and other instructions.
    int mma_other_inst_num = getMmaInstNum(mmaElemTy);
    mma_other_inst_num = mma_other_inst_num == 0 ? 1 : mma_other_inst_num;
    barrierOp->moveBefore(genLoads.front().getDefiningOp());
    Operation *iglp =
        builder.create<ttg::IGLPOp>(anchor, 3, -1, 4, 4, mma_other_inst_num, 6);
    iglp->moveAfter(barrierOp);
  } else {
    barrierOp->moveAfter(genLoads.front().getDefiningOp());
  }

  // generate non-shared memory deps
  DenseMap<Value, Value> nextStageBuffer;
  for (Value cvt : validCvts) {
    auto curBuffer =
        newForOp.getRegionIterArgs()[bufferIdx + nextBuffers.size()];
    nextStageBuffer[cvt] = curBuffer;
    nextBuffers.push_back(curBuffer);
  }
  // set cvtMapping of blockedDeps
  auto genDeps =
      genDotDeps(builder, forOp, validLoads, nextMapping, mapping, loadsMapping,
                 elemsMapping, blockedDeps, nextDotOperands, nextStageBuffer,
                 cvtMapping, extractSliceIndex, false);
  if (newOps.size() > 0 && numStages > 3) {
    for (Value cvt : validCvts) {
      auto orig = cvtMapping[cvt];
      auto nextOp = nextMapping.lookupOrDefault(orig);
      auto curOp = newForOp.getRegionIterArgs()[newOpIdx + nextOps.size()];
      nextOps.push_back(nextOp);
      nextMapping.map(orig, curOp);
    }
  }
  // set cvtMapping of storedDeps
  auto genStoredDeps =
      genDotDeps(builder, forOp, validLoads, nextMapping, mapping, loadsMapping,
                 elemsMapping, storedDeps, nextDotOperands, nextStageBuffer,
                 cvtMapping, extractSliceIndex, false);
  for (Value cvt : validCvts) {
    auto nextOp = nextMapping.lookupOrDefault(cvtMapping[cvt]);
    extractSlices.push_back(nextOp);
  }

  if (isSingleShm) {
    for (Operation *op : orderedDeps) {
      if (!validLoads.contains(op->getResult(0))) {
        if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
          auto firstLoad =
              nextMapping.lookupOrDefault(loadOp.getResult()).getDefiningOp();
          barrierOp->moveBefore(firstLoad);
          break;
        }
      }
    }
  }
  if (isSingleShm) {
    // TODO: support more load in single shared memory mode
    auto firstDot = mapping.lookupOrDefault(loadDotMapping[validLoads.front()])
                        .getDefiningOp();
    for (auto it = genStoredDeps.rbegin(); it != genStoredDeps.rend(); ++it) {
      // for single shared memory we need to move the local_store after the dot,
      // like that: %3 = local_alloc [1 x buffer_shape] %7 = tt.load %8 =
      // memdesc_subview %3 [0, 0, 0] local_store %7 %8 scf.for(..., arg*=%8,
      // ...)
      //    ...
      //    %14 = tt.load
      //    ...
      //    %24 = tt.dot * arg*
      //    local_store %14 %8
      (*it)->moveAfter(firstDot);
    }
  } else {
    for (auto it = genStoredDeps.rbegin(); it != genStoredDeps.rend(); ++it) {
      // move localstore after asyncWait
      (*it)->moveAfter(asyncWait);
    }
  }

  for (auto it = genDeps.rbegin(); it != genDeps.rend(); ++it) {
    (*it)->moveAfter(asyncWait);
  }
  asyncWait->erase();
#else
  Operation *asyncWait = builder.create<ttg::AsyncWaitOp>(
      validLoads[0].getLoc(), validLoads.size() * (numStages - 2));
  for (auto it = extractSlices.rbegin(); it != extractSlices.rend(); ++it) {
    // move extract_slice after asyncWait
    it->getDefiningOp()->moveAfter(asyncWait);
  }
#endif

  // Bump iteration count
  pipelineIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), pipelineIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
  loopIterIdx = builder.create<arith::AddIOp>(
      nextIV.getLoc(), loopIterIdx,
      builder.create<arith::ConstantIntOp>(nextIV.getLoc(), 1, 32));
}

void LoopPipeliner::finalizeYield(scf::ForOp newForOp, OpBuilder &builder) {
  SmallVector<Value> yieldValues;
  for (Value v : yieldOp->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (Value nextBuffer : nextBuffers)
    yieldValues.push_back(nextBuffer);
  for (Value nextOp : nextOps)
    yieldValues.push_back(nextOp);
  for (Value nextLoad : nextLoads)
    yieldValues.push_back(nextLoad);
  for (Value nextSlice : extractSlices)
    yieldValues.push_back(nextSlice);

  for (size_t i = depArgsBeginIdx; i < ivIndex; ++i) {
    auto arg = newForOp.getRegionIterArgs()[i];
    assert(depArgsMapping.count(arg) && "Missing loop-carried value");
    yieldValues.push_back(depArgsMapping[arg]);
  }
  yieldValues.push_back(nextIV);
  yieldValues.push_back(pipelineIterIdx);
  yieldValues.push_back(loopIterIdx);

  builder.setInsertionPointToEnd(newForOp.getBody());
  builder.create<scf::YieldOp>(yieldOp->getLoc(), yieldValues);
}

scf::ForOp LoopPipeliner::createNewForOp() {
  OpBuilder builder(forOp);
  auto newLoopArgs = collectNewLoopArgs();
  auto newForOp = cloneForOp(newLoopArgs, builder);
  prefetchNextIteration(newForOp, builder);
  finalizeYield(newForOp, builder);
  return newForOp;
}
} // anonymous namespace

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUPipelineMACAPass
    : public TritonMETAXGPUPipelineMACABase<TritonMETAXGPUPipelineMACAPass> {
public:
  TritonMETAXGPUPipelineMACAPass() = default;
  TritonMETAXGPUPipelineMACAPass(int numStages, int pipelineLoadNum,
                                 bool isFullStage, bool isSingleShm) {
    this->numStages = numStages;
    this->pipelineLoadNum = pipelineLoadNum;
    this->isFullStage = isFullStage;
    this->isSingleShm = isSingleShm;
  }

  void runOnOperation() override {
    int numStages = this->numStages;
    int pipelineLoadNum = this->pipelineLoadNum;
    bool isFullStage = this->isFullStage;
    bool isSingleShm = this->isSingleShm;

    // just now, for single shared memory we only support pipeline one load
    if (numStages <= 1 || pipelineLoadNum == 0 ||
        (isSingleShm && pipelineLoadNum != 1))
      return;

    // Do the pipelining
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      LoopPipeliner pipeliner(forOp, numStages, pipelineLoadNum, isFullStage,
                              isSingleShm);

      if (pipeliner.initialize().failed())
        return;

      pipeliner.emitPrologue();
      scf::ForOp newForOp = pipeliner.createNewForOp();
      pipeliner.emitEpilogue();

      // Replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineMACAPass(int numStages, int pipelineLoadNum,
                                           bool isFullStage, bool isSingleShm) {
  return std::make_unique<TritonMETAXGPUPipelineMACAPass>(
      numStages, pipelineLoadNum, isFullStage, isSingleShm);
}
