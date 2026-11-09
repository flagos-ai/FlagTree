#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "BufferOpsEmitter.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TDMUtility.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "WarpSpecializeUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

#include <algorithm>
#include <functional>
#include <memory>

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::HCU::getVectorSize;
using ::mlir::LLVM::HCU::llLoad;
using ::mlir::LLVM::HCU::llStore;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::HCU::ISAFamily;

namespace {

// A masked raw buffer load already returns all-zero bits for the OOB
// sentinel. Recognize the original constant before unpack/insert-element
// construction hides it from BufferEmitter's LLVM constant matcher. Only an
// explicit all-bit-zero constant qualifies: -0, NaNs, mixed values and unknown
// SSA values keep their original fallback selects. No mask fact is inferred.
static bool isAllZeroBitsConstant(Value value) {
  Attribute attr;
  if (!value || !matchPattern(value, m_Constant(&attr)))
    return false;
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().isZero();
  if (auto real = dyn_cast<FloatAttr>(attr))
    return real.getValue().bitcastToAPInt().isZero();
  if (auto integers = dyn_cast<DenseIntElementsAttr>(attr))
    return integers.isSplat() && integers.getSplatValue<APInt>().isZero();
  if (auto reals = dyn_cast<DenseFPElementsAttr>(attr))
    return reals.isSplat() &&
           reals.getSplatValue<APFloat>().bitcastToAPInt().isZero();
  return false;
}

static Value peelBufferMaskAggregate(Value value);

// Prove only explicit scalar integer uniformity. In particular, a scalar
// LLVM type, a load result, and an arbitrary PHI are not uniformity proofs.
// The CF recurrence case is the ordinary SCF-to-CF integer induction schema:
// one dominating initial edge and one backedge with uniform arithmetic and
// control. No warp-specialization marker or matrix configuration is used.
static bool isRematerializationUniform(Value root,
                                      bool loopCycleControlsOnly = false) {
  DenseMap<Value, bool> memo;
  llvm::SmallDenseSet<Value, 8> induction;
  llvm::SmallDenseSet<Value, 64> active;
  unsigned steps = 0;
  auto arithmetic = [](Operation *op) {
    // A defined unsigned quotient/remainder of CTA-uniform values remains
    // uniform for every nonzero constant divisor, not just powers of two.
    // This is a frontier proof: the existing division stays at its original
    // position and is reused, never speculated or cloned at another lane.
    if (isa<arith::DivUIOp, arith::RemUIOp, LLVM::UDivOp, LLVM::URemOp>(op)) {
      APInt divisor;
      return matchPattern(op->getOperand(1), m_ConstantInt(&divisor)) &&
             !divisor.isZero();
    }
    return isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp,
               arith::MulIOp, arith::AndIOp, arith::OrIOp, arith::XOrIOp,
               arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
               arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
               arith::CmpIOp, arith::SelectOp,
               LLVM::ConstantOp, LLVM::AddOp, LLVM::SubOp, LLVM::MulOp,
               LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::ShlOp,
               LLVM::LShrOp, LLVM::AShrOp, LLVM::SExtOp, LLVM::ZExtOp,
               LLVM::TruncOp, LLVM::ICmpOp, LLVM::SelectOp>(op);
  };
  std::function<bool(Value)> prove = [&](Value value) -> bool {
    if (loopCycleControlsOnly) {
      value = peelBufferMaskAggregate(value);
      if (!value)
        return false;
    }
    if (!isa<IntegerType>(value.getType()) || ++steps > 512)
      return false;
    if (induction.contains(value))
      return true;
    if (auto it = memo.find(value); it != memo.end())
      return it->second;
    if (!active.insert(value).second)
      return false;
    auto finish = [&](bool result) {
      active.erase(value);
      memo[value] = result;
      return result;
    };
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *block = arg.getOwner();
      Operation *parent = block->getParentOp();
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(parent)) {
        auto marker = func->getAttrOfType<IntegerAttr>("nvvm.kernel");
        if (block == &func.getBody().front() && triton::isKernel(func) &&
            marker && marker.getValue().isOne())
          return finish(true);
      }
      // Explicit capture is an identity edge, not a new uniformity fact.
      if (auto parts = dyn_cast<WarpSpecializePartitionsOp>(parent)) {
        if (block == &block->getParent()->front()) {
          auto ws = parts.getParentOp();
          return finish(arg.getArgNumber() < ws.getExplicitCaptures().size() &&
                        prove(ws.getExplicitCaptures()[arg.getArgNumber()]));
        }
        // Non-entry region arguments are ordinary CFG values, not captures.
      }
      if (!block->hasNoPredecessors()) {
        SmallVector<std::pair<Block *, Value>> incoming;
        bool valid = true;
        for (Block *pred : block->getPredecessors()) {
          Operation *term = pred->getTerminator();
          if (auto branch = dyn_cast<cf::BranchOp>(term)) {
            incoming.emplace_back(pred, branch.getDestOperands()[arg.getArgNumber()]);
          } else if (auto branch = dyn_cast<LLVM::BrOp>(term)) {
            incoming.emplace_back(pred, branch.getDestOperands()[arg.getArgNumber()]);
          } else {
            valid = false;
          }
        }
        if (valid && incoming.size() == 2) {
          DominanceInfo dominance(parent);
          auto a = incoming[0], b = incoming[1];
          if (dominance.dominates(block, a.first))
            std::swap(a, b);
          // Exactly one initial edge and one actual backedge.
          if (!dominance.dominates(block, a.first) &&
              dominance.dominates(block, b.first) && prove(a.second)) {
            induction.insert(value);
            valid = prove(b.second);
            // A recurrence depends on control that can return to its
            // backedge. A divergent branch after the loop cannot change the
            // induction value. Keep existing callers' conservative scope;
            // buffer-mask factoring opts into the natural-cycle scope.
            llvm::SmallPtrSet<Block *, 32> cycle;
            if (loopCycleControlsOnly) {
              SmallVector<Block *> reverse{b.first};
              while (valid && !reverse.empty()) {
                Block *cursor = reverse.pop_back_val();
                if (!cycle.insert(cursor).second)
                  continue;
                if (cycle.size() > 64) { valid = false; break; }
                if (cursor == block)
                  continue;
                for (Block *pred : cursor->getPredecessors())
                  if (pred->getParent() == block->getParent() &&
                      dominance.dominates(block, pred))
                    reverse.push_back(pred);
              }
              valid = valid && cycle.contains(block);
            }
            SmallVector<Block *> pending{block};
            llvm::SmallPtrSet<Block *, 32> seen;
            while (valid && !pending.empty()) {
              Block *cursor = pending.pop_back_val();
              if (!seen.insert(cursor).second)
                continue;
              if (seen.size() > 64) { valid = false; break; }
              Operation *term = cursor->getTerminator();
              if (auto branch = dyn_cast<cf::CondBranchOp>(term))
                valid = prove(branch.getCondition());
              else if (auto branch = dyn_cast<LLVM::CondBrOp>(term))
                valid = prove(branch.getCondition());
              else if (!isa<cf::BranchOp, LLVM::BrOp>(term))
                valid = term->getNumSuccessors() == 0;
              for (Block *next : cursor->getSuccessors())
                if (next != block && next->getParent() == block->getParent() &&
                    dominance.dominates(block, next) &&
                    (!loopCycleControlsOnly || cycle.contains(next)))
                  pending.push_back(next);
            }
            induction.erase(value);
            return finish(valid);
          }
        }
      }
      return finish(false);
    }
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumRegions() || op->getNumSuccessors() ||
        op->getNumResults() != 1)
      return finish(false);
    StringRef name = op->getName().getStringRef();
    if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(op) ||
        name == "rocdl.workgroup.id.x" || name == "rocdl.workgroup.id.y" ||
        name == "rocdl.workgroup.id.z" || name == "rocdl.grid.size.x" ||
        name == "rocdl.grid.size.y" || name == "rocdl.grid.size.z")
      return finish(true);
    return finish(arithmetic(op) && llvm::all_of(op->getOperands(), prove));
  };
  return prove(root);
}

// Factor a proved uniform Boolean conjunct out of a raw-buffer load mask.
// NUM_RECORDS=0 suppresses the same accesses as the existing OOB offset,
// without making the per-lane address depend on a loop-uniform condition.
// Keep the original vector length, offsets, and nonuniform mask conjuncts.
// This deliberately reuses the existing integer/CFG uniformity proof: a
// scalar LLVM type, an arbitrary PHI, or a lane/load result is insufficient.
// Prove the inputs to an integer comparison separately. If the comparison
// is a loop exit, the induction proof may revisit it under the induction
// hypothesis. Starting the query at the comparison itself would reject that
// valid recursion as a value cycle.
static bool isUniformBranchCondition(Value condition) {
  Operation *definition = condition.getDefiningOp();
  if (definition && isa<arith::CmpIOp, LLVM::ICmpOp>(definition))
    return llvm::all_of(definition->getOperands(), [](Value operand) {
      return isRematerializationUniform(operand, true);
    });
  return isRematerializationUniform(condition, true);
}

// A uniform mask is not an EXEC proof. A terminal gfx936 partition
// starts on a wave boundary, and retains all its lanes when its own and
// the enclosing kernel's control flow are uniform. Inspect the original
// CFG before individual lowering patterns introduce masked branches.
static bool preservesFullWaveExecution(Region &region,
                                       Operation *specialization = nullptr) {
  for (Block &block : region) {
    Operation *terminator = block.getTerminator();
    if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
      if (!isUniformBranchCondition(branch.getCondition()))
        return false;
    } else if (auto branch = dyn_cast<LLVM::CondBrOp>(terminator)) {
      if (!isUniformBranchCondition(branch.getCondition()))
        return false;
    } else if (terminator->getNumSuccessors() &&
               !isa<cf::BranchOp, LLVM::BrOp>(terminator)) {
      return false;
    }
    for (Operation &op : block.without_terminator()) {
      if (&op == specialization)
        continue;
      if (op.getNumRegions() ||
          isa<CallOpInterface, LLVM::CallIntrinsicOp, LLVM::InlineAsmOp,
              triton::ElementwiseInlineAsmOp>(op))
        return false;
    }
  }
  return true;
}

static llvm::DenseSet<Operation *> collectFullWaveBufferCopies(ModuleOp module,
                                                              StringRef arch) {
  llvm::DenseSet<Operation *> copies;
  if (arch != "gfx936" || TritonGPUDialect::getNumCTAs(module) != 1 ||
      TritonGPUDialect::getThreadsPerWarp(module) != 64)
    return copies;
  auto base = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto total = module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  if (!base || !total || base.getInt() < 1 || total.getInt() > 16)
    return copies;

  module.walk([&](WarpSpecializeOp ws) {
    auto kernel = dyn_cast<LLVM::LLVMFuncOp>(ws->getParentOp());
    auto starts = ws.getWarpGroupStartIds();
    if (!kernel || !triton::isKernel(kernel) || ws.getNumResults() ||
        !starts || starts->size() != ws.getPartitionRegions().size())
      return;
    // The manual specialization pass establishes these exact wave intervals.
    int64_t next = base.getInt();
    for (auto [start, waves] : llvm::zip(*starts, ws.getPartitionNumWarps())) {
      if (start != next || waves < 1)
        return;
      next += waves;
    }
    if (next != total.getInt() ||
        !preservesFullWaveExecution(kernel.getBody(), ws.getOperation()))
      return;
    SmallVector<Region *> regions{&ws.getDefaultRegion()};
    llvm::append_range(regions, ws.getPartitionRegions());
    for (Region *region : regions) {
      if (!preservesFullWaveExecution(*region))
        continue;
      for (Block &block : *region)
        for (auto copy : block.getOps<triton::hcugpu::BufferLoadToLocalOp>())
          copies.insert(copy.getOperation());
    }
  });
  return copies;
}

static Value peelBufferMaskAggregate(Value value) {
  unsigned steps = 0;
  while (true) {
    // Tensor splats currently introduce same-type integer bitcasts. They
    // are identity edges, not additional uniformity facts.
    if (auto cast = value.getDefiningOp<LLVM::BitcastOp>()) {
      if (cast.getArg().getType() != value.getType() ||
          !isa<IntegerType>(value.getType()) || ++steps > 512)
        return {};
      value = cast.getArg();
      continue;
    }
    auto extract = value.getDefiningOp<LLVM::ExtractValueOp>();
    if (!extract)
      return value;
    auto position = extract.getPosition();
    auto type =
        dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
    if (position.size() != 1 || !type || type.isOpaque() ||
        position[0] < 0 || position[0] >= type.getBody().size() ||
        type.getBody()[position[0]] != value.getType())
      return {};
    Value container = extract.getContainer();
    while (true) {
      if (++steps > 512)
        return {};
      auto insert = container.getDefiningOp<LLVM::InsertValueOp>();
      if (!insert || insert.getPosition().size() != 1 ||
          insert.getContainer().getType() != type || insert.getType() != type)
        return {};
      int64_t field = insert.getPosition()[0];
      if (field < 0 || field >= type.getBody().size() ||
          insert.getValue().getType() != type.getBody()[field])
        return {};
      if (field == position[0]) {
        value = insert.getValue();
        break;
      }
      container = insert.getContainer();
    }
  }
}

static Value findUniformBufferMaskFactor(Value value, unsigned depth = 0) {
  if (!value || !value.getType().isInteger(1) || depth > 32)
    return {};
  value = peelBufferMaskAggregate(value);
  if (!value)
    return {};
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant)))
    return {};
  if (isRematerializationUniform(value, /*loopCycleControlsOnly=*/true))
    return value;
  Operation *op = value.getDefiningOp();
  if (!op || !isa<arith::AndIOp, LLVM::AndOp>(op))
    return {};
  if (Value factor = findUniformBufferMaskFactor(op->getOperand(0), depth + 1))
    return factor;
  return findUniformBufferMaskFactor(op->getOperand(1), depth + 1);
}

static bool collectBufferMaskResidual(Value value, Value factor,
                                      SmallVectorImpl<Value> &residual,
                                      bool &found, unsigned depth = 0) {
  if (depth > 32 || !value.getType().isInteger(1))
    return false;
  // Unknown aggregates remain opaque residual predicates, never proof roots.
  if (Value peeled = peelBufferMaskAggregate(value))
    value = peeled;
  if (value == factor) {
    found = true;
    return true;
  }
  Operation *op = value.getDefiningOp();
  if (op && isa<arith::AndIOp, LLVM::AndOp>(op))
    return collectBufferMaskResidual(op->getOperand(0), factor, residual,
                                     found, depth + 1) &&
           collectBufferMaskResidual(op->getOperand(1), factor, residual,
                                     found, depth + 1);
  residual.push_back(value);
  return true;
}

static bool factorUniformBufferLoadMask(
    ConversionPatternRewriter &rewriter, Location loc, Value &resource,
    SmallVectorImpl<Value> &maskElems, unsigned vec) {
  if (maskElems.empty())
    return false;
  Value factor = findUniformBufferMaskFactor(maskElems.front());
  auto descriptor = resource.getDefiningOp<ROCDL::MakeBufferRsrcOp>();
  if (!factor || !descriptor || descriptor->getNumOperands() != 4 ||
      !descriptor->getOperand(2).getType().isInteger(32))
    return false;

  // All emitted transactions must contain the exact same proved factor.
  // Plan first, so an unsupported predicate leaves the old lowering intact.
  SmallVector<SmallVector<Value>> residuals;
  for (unsigned i = 0; i < maskElems.size(); i += vec) {
    SmallVector<Value> residual;
    bool found = false;
    if (!collectBufferMaskResidual(maskElems[i], factor, residual, found) ||
        !found)
      return false;
    residuals.push_back(std::move(residual));
  }

  TritonLLVMOpBuilder b(loc, rewriter);
  Value records = b.select(factor, descriptor->getOperand(2), b.i32_val(0));
  resource = rewriter.createOrFold<ROCDL::MakeBufferRsrcOp>(
      loc, resource.getType(), descriptor->getOperand(0),
      descriptor->getOperand(1), records, descriptor->getOperand(3));
  for (unsigned i = 0, j = 0; i < maskElems.size(); i += vec, ++j) {
    Value pred = b.int_val(1, 1);
    for (Value leaf : residuals[j])
      pred = b.and_(pred, leaf);
    maskElems[i] = pred;
  }
  return true;
}



// Store descriptors need the complete acyclic integer grid-index DAG.
// Preserve the existing load/rematerialization acceptance domain. Every new
// operation is deterministic integer arithmetic on recursively uniform inputs;
// unknown calls, loads, thread/lane IDs and unproved CFG values remain rejected.
static bool isBufferStoreRebaseUniform(Value root) {
  DenseMap<Value,bool> memo;
  llvm::SmallDenseSet<Value,64> active;
  unsigned steps=0;
  std::function<bool(Value)> prove=[&](Value value)->bool {
    if (!isa<IntegerType>(value.getType()) || ++steps>512) return false;
    if (auto it=memo.find(value);it!=memo.end()) return it->second;
    if (isRematerializationUniform(value)) {memo[value]=true;return true;}
    if (!active.insert(value).second) return false;
    auto finish=[&](bool result) {active.erase(value);memo[value]=result;return result;};
    Operation *op=value.getDefiningOp();
    if (!op || op->getNumRegions() || op->getNumSuccessors() || op->getNumResults()!=1)
      return finish(false);
    StringRef name=op->getName().getStringRef();
    bool arithmetic=isa<arith::AddIOp,arith::SubIOp,arith::MulIOp,
      arith::AndIOp,arith::OrIOp,arith::XOrIOp,arith::ShLIOp,arith::ShRUIOp,
      arith::ShRSIOp,arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp,
      arith::CmpIOp,arith::SelectOp,arith::DivSIOp,arith::DivUIOp,
      arith::RemSIOp,arith::RemUIOp,arith::MinSIOp,arith::MinUIOp,
      arith::MaxSIOp,arith::MaxUIOp,LLVM::AddOp,LLVM::SubOp,LLVM::MulOp,
      LLVM::AndOp,LLVM::OrOp,LLVM::XOrOp,LLVM::ShlOp,LLVM::LShrOp,
      LLVM::AShrOp,LLVM::SExtOp,LLVM::ZExtOp,LLVM::TruncOp,LLVM::ICmpOp,
      LLVM::SelectOp,LLVM::SDivOp,LLVM::UDivOp,LLVM::SRemOp,LLVM::URemOp>(op)
      || name=="llvm.intr.smin" || name=="llvm.intr.smax"
      || name=="llvm.intr.umin" || name=="llvm.intr.umax";
    return finish(arithmetic && llvm::all_of(op->getOperands(),prove));
  };
  return prove(root);
}

// A deliberately small range proof for explicit offset terms. Bounds apply
// to every tensor lane, independently of masks. Unknown arguments/loads/PHIs
// and arithmetic that might wrap fall back to the existing offset lowering.
static std::optional<uint64_t> getNonNegativeOffsetMax(Value value,
                                                     unsigned depth = 0) {
  if (depth > 16)
    return std::nullopt;
  auto ty = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  if (!ty || (ty.getWidth() != 32 && ty.getWidth() != 64))
    return std::nullopt;
  constexpr uint64_t limit = INT32_MAX;
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    if (auto integer = dyn_cast<IntegerAttr>(attr)) {
      const APInt &v = integer.getValue();
      if (!v.isNegative() && v.getActiveBits() <= 31)
        return v.getZExtValue();
    }
    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
      if (dense.getNumElements() > 4096 && !dense.isSplat())
        return std::nullopt;
      uint64_t maximum = 0;
      for (const APInt &v : dense.getValues<APInt>()) {
        if (v.isNegative() || v.getActiveBits() > 31)
          return std::nullopt;
        maximum = std::max(maximum, v.getZExtValue());
      }
      return maximum;
    }
    return std::nullopt;
  }
  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (range.getStart() >= 0 && range.getEnd() > range.getStart())
      return uint64_t(range.getEnd() - 1);
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (isa<triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
          ConvertLayoutOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op))
    return getNonNegativeOffsetMax(op->getOperand(0), depth + 1);
  if (!isa<arith::AddIOp, arith::MulIOp, arith::AndIOp>(op))
    return std::nullopt;
  auto a = getNonNegativeOffsetMax(op->getOperand(0), depth + 1);
  auto b = getNonNegativeOffsetMax(op->getOperand(1), depth + 1);
  // AND with any nonnegative bounded value clears the sign bit even when
  // the other operand is unknown (including an arbitrary kernel argument).
  if (isa<arith::AndIOp>(op))
    return a && b ? std::min(*a, *b) : (a ? a : b);
  if (!a || !b)
    return std::nullopt;
  if (isa<arith::AddIOp>(op) && *a <= limit - *b)
    return *a + *b;
  if (isa<arith::MulIOp>(op) && (!*b || *a <= limit / *b))
    return *a * *b;
  return std::nullopt;
}

#include "BufferStoreScalarOffset.inc"

std::optional<const char *> getHCUGPUMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  // The default HCUHSA LLVM Sync Scope is "system", so no string is
  // provided here
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

std::pair<bool, bool> getOrderingFlags(MemSemantic memOrdering) {
  bool emitReleaseFence = false;
  bool emitAcquireFence = false;
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    // In this case, no memory fences are needed
    break;
  case MemSemantic::RELEASE:
    emitReleaseFence = true;
    break;
  case MemSemantic::ACQUIRE:
    emitAcquireFence = true;
    break;
  case MemSemantic::ACQUIRE_RELEASE:
    emitAcquireFence = true;
    emitReleaseFence = true;
  default:
    // default == acq_rel, so we emit the same barriers
    emitAcquireFence = true;
    emitReleaseFence = true;
  }
  return {emitAcquireFence, emitReleaseFence};
}

LogicalResult emitFence(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc, MemSemantic memOrdering,
                        MemSyncScope memScope, bool preAtomic) {
  // This function emits an LLVM::FenceOp which will get lowered by the
  // LLVM backend to the right scope and ordering instructions, as
  // described in the "atomicrmw" entries for "global" address-space,
  // in the "HCUHSA Memory Model Code Sequences GFX942"
  // table in https://llvm.org/docs/HCUGPUUsage.html#memory-model-gfx942
  //
  // Triton supports three scopes for atomic access
  // 1. System
  // 2. GPU (default) ('Agent' for HCUGPU)
  // 3. CTA ('Workgroup' for HCUGPU)
  //
  // and 4 orderings
  // 1. Relaxed
  // 2. Acquire
  // 3. Release
  // 4. AcquireRelease
  //
  // The following table shows the scope and ordering instructions that
  // are emitted by this function for each combination of scope and ordering
  // for buffer-atomic instructions.
  //
  // Note: In the following comments, "[buffer-atomic_0.. buffer-atomic_n]"
  // represents a sequence of buffer-atomic instructions that are lowered from
  // a single tl.atomic_*
  //
  // Unordered(Relaxed):
  //   agent/workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //                    No scope/ordering instrs are required.
  //   system: //TODO:
  // Acquire:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2.
  //              No scope/ordering instrs are required.
  //   agent: Instr seq: [buffer-atomic_0.. buffer-atomic_n],
  //                     s_waitcnt vmcnt(0), buffer_inv sc1=1
  //          Waves across an agent may use different L1 and L2.
  //          Atomic ops bypass L1 and operate on L2.
  //          s_waitcnt vmcnt(0) ensures that the atomicrmw has completed
  //          before invalidating the cache. buffer_inv sc1=1 will a) L1:
  //          invalidate cache b) L2: Invalidate non-coherently modified lines
  //          if multiple L2s are configured, NOP otherwise. This buffer_inv
  //          ensures that following loads do not see stale global values.
  //   system: //TODO:
  //
  // Release:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2 so all
  //              previous global writes of a waver are visible to all other
  //              waves in the workgroup. LDS operations for all waves are
  //              executed in a total global ordering and are observed by all
  //              waves in the workgroup. So LDS stores issued before the
  //              release will be visible to LDS loads after the read of the
  //              released buffer-atomic. So, swait_cnt lgkmcnt is not
  //              required.
  //   agent: Instr seq: buffer_wbl2 sc1=1, s_waitcnt vmcnt(0),
  //                     [buffer-atomic_0.. buffer-atomic_n]
  //          buffer_wbl2 sc1=1 ensures that dirtly L2 lines are visible to
  //          CUs that don't use the same L2.
  //          From SIMemoryLegalizer.cpp SIGfx940CacheControl::insertRelease:
  //            "Inserting a "S_WAITCNT vmcnt(0)" before is not required
  //             because the hardware does not reorder memory operations by
  //             the same wave with respect to a following "BUFFER_WBL2".
  //             The "BUFFER_WBL2" is guaranteed to initiate writeback of
  //             any dirty cache lines of earlier writes by the same wave.
  //             A "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback
  //             has completed.""
  //   system: //TODO:
  //
  // AcquireRelease:
  //   Instr seq: Release scope/order insts,
  //              [buffer-atomic_0..buffer-atomic_n],
  //              Acquire scope/order instrs.
  //
  // LLVM::FenceOp lowering will emit the required cache ops and s_waitcnt
  // vmcnt(0) instrs

  auto [emitReleaseFence, emitAcquireFence] = getOrderingFlags(memOrdering);
  if (MemSyncScope::SYSTEM == memScope)
    return rewriter.notifyMatchFailure(
        op, "System memory scope is not supported for Buffer Atomic Ops");
  auto scopeStr = getHCUGPUMemScopeStr(memScope);
  if (!scopeStr)
    return rewriter.notifyMatchFailure(
        op, "Unsupported memory scope for Buffer Atomic Ops");

  StringAttr scope = mlir::StringAttr::get(loc.getContext(), *scopeStr);

  if (emitReleaseFence && preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    LLVM::FenceOp::create(rewriter, loc, TypeRange{},
                          LLVM::AtomicOrdering::acquire, scope);
  }
  return success();
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const HCU::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

std::pair<Block *, Block *> emitBranch(RewriterBase &rewriter, Location loc,
                                       Value cond) {
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *after =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *body = rewriter.createBlock(after);
  rewriter.setInsertionPointToEnd(currentBlock);
  LLVM::CondBrOp::create(rewriter, loc, cond, body, after);
  rewriter.setInsertionPointToStart(body);
  LLVM::BrOp::create(rewriter, loc, after);
  rewriter.setInsertionPointToStart(body);
  return {body, after};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const HCU::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = LLVM::ConstantOp::create(builder, loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(RewriterBase &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Return a tensor of pointers with the same type of `basePtr` and the same
  // shape of `offset`
  Type getPointerTypeWithShape(Value basePtr, Value offset) const {
    Type basePtrType = basePtr.getType();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    return offsetType.cloneWith(std::nullopt, basePtrType);
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const HCU::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Contains some helper functions for direct to lds loads.
struct DirectToLdsLoadConversionBase : public LoadStoreConversionBase {
  explicit DirectToLdsLoadConversionBase(
      const HCU::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // direct to lds loads do not support per lane shared offsets. We need to
  // ensure that we write coalesced into shared memory. This means we cannot
  // exceed the supported load width because splitting them would cause strided
  // (non coalesced) writes. Additionally:
  //   1) For *non* swizzled shared encodings we check if they result in
  //      coalesced writes and can then lower them directly to the intrinsics.
  //   2) For swizzled shared encodings we need to transfer the swizzling to the
  //      source pointers. For now this is done by swizzling the pointers
  //      between the lane of a warp via permute. This only works if the swizzle
  //      pattern does not exchange elements between warps which holds for all
  //      our swizzle patterns. There is still a check performed to not silently
  //      produce wrong results if we invalidate the condition in the future
  LogicalResult canWriteCoalesced(RewriterBase &rewriter, Operation *op,
                                  RankedTensorType srcTy, MemDescType dstTy,
                                  unsigned vectorSize,
                                  bool requiresSrcPtrSwizzling,
                                  bool isBuffer = false) const {
    if (targetInfo.supportsDirectToLDSScattering()) {
      return success();
    }

    int vecBits = vectorSize * dstTy.getElementTypeBitWidth();
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(vecBits, isBuffer)) {
      LDBG(*op << " results in unsupported load bitwidth: " << vecBits);
      return failure();
    }
    // Compute the blocked -> shared linear layout to check preconditions
    LinearLayout srcLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

    unsigned threadsPerWarp = lookupThreadsPerWarp(rewriter);
    if (!requiresSrcPtrSwizzling &&
        !LLVM::HCU::canCoalesceWriteIntoSharedMemory(
            rewriter, srcToSharedLayout, threadsPerWarp, vectorSize)) {
      LDBG(*op << " does not write coalesced into LDS and is not swizzled");
      return failure();
    }

    if (requiresSrcPtrSwizzling &&
        !LLVM::HCU::doesSwizzleInsideWarp(rewriter, srcToSharedLayout,
                                          threadsPerWarp)) {
      LDBG(*op << " does swizzle across warp boundaries");
      return failure();
    }
    return success();
  }

  // For each load emit the computation to get the lane id offset which holds
  // the source pointers/offsets we need to store to shared memory
  SmallVector<Value>
  emitSwizzledLaneOffsets(RewriterBase &rewriter, Operation *op,
                          RankedTensorType srcTy, MemDescType swizzledTy,
                          MemDescType flatTy, Value llDst, Type resElemTy,
                          unsigned vec) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Create regToShared layout for the swizzled and flat encoding
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    auto sharedSwizz = triton::gpu::toLinearLayout(swizzledTy);
    auto sharedFlat = triton::gpu::toLinearLayout(flatTy);

    auto regToSharedSwizzled = regLayout.invertAndCompose(sharedSwizz);
    auto regToSharedFlat = regLayout.invertAndCompose(sharedFlat);

    MLIRContext *ctx = rewriter.getContext();
    StringAttr kBlock = str_attr("block");
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value blockId = b.i32_val(0);

    int numberOfLoads = regToSharedSwizzled.getInDimSize(kRegister) / vec;

    // For each load compute the difference between the flat and the swizzled
    // linear offsets into shared memory
    // TODO (alex): this is only correct as long as the lds view is a contiguous
    // block. So this can break if we slice along the 2 minor dimensions
    SmallVector<Value> swizzledOffsets;
    swizzledOffsets.reserve(numberOfLoads);
    auto vecVal = b.i32_val(vec);
    for (int i = 0; i < numberOfLoads; i++) {
      auto regId = b.i32_val(i * vec);

      std::array<std::pair<StringAttr, Value>, 4> indices{{
          {kRegister, regId},
          {kLane, laneId},
          {kWarp, warpId},
          {kBlock, blockId},
      }};

      Value swizzledOffset =
          applyLinearLayout(loc, rewriter, regToSharedSwizzled, indices)[0]
              .second;
      Value flatOffset =
          applyLinearLayout(loc, rewriter, regToSharedFlat, indices)[0].second;

      // Normalize the offset by vecTy to obtain the offset in lanes
      auto laneOffet = b.sdiv(b.sub(swizzledOffset, flatOffset), vecVal);
      swizzledOffsets.push_back(laneOffet);
    }
    return swizzledOffsets;
  }

  // Swizzle the mask (1bit) based on selectLane via ballot
  Value shuffleMask(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                    Location loc, const TargetInfoBase &targetInfo,
                    Value selectLane, Value mask) const {
    auto warpMask =
        targetInfo.ballot(rewriter, loc, rewriter.getI64Type(), mask);
    // Extract the selectLane bit
    auto bitMask = b.lshr(warpMask, b.zext(rewriter.getI64Type(), selectLane));
    return b.trunc(i1_ty, bitMask);
  }

  SmallVector<Value> zipLoadValues(RewriterBase &rewriter, Location loc,
                                   unsigned vec, ArrayRef<Value> srcElems,
                                   Type srcTy, ArrayRef<Value> maskElems,
                                   ArrayRef<Value> otherElems, Type otherTy,
                                   ArrayRef<Value> swizzledLaneOffsets) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadVals;
    auto structTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(), ArrayRef<Type>{srcTy, i1_ty, otherTy, i32_ty});
    for (int i = 0; i < srcElems.size(); i++) {
      Value packedArr = LLVM::UndefOp::create(rewriter, loc, structTy);
      // src
      packedArr = b.insert_val(packedArr, srcElems[i], 0);
      // mask
      auto maskElem = maskElems.empty() ? b.true_val() : maskElems[i];
      packedArr = b.insert_val(packedArr, maskElem, 1);
      // other
      if (!otherElems.empty())
        packedArr = b.insert_val(packedArr, otherElems[i], 2);
      // swizzleOffset are per vec so we need to duplicate values vec times
      auto swizzleOffset = swizzledLaneOffsets.empty()
                               ? b.i32_val(0)
                               : swizzledLaneOffsets[i / vec];
      packedArr = b.insert_val(packedArr, swizzleOffset, 3);

      loadVals.push_back(packedArr);
    }
    return loadVals;
  }

  auto unzipLoadValues(RewriterBase &rewriter, Location loc, int startIdx,
                       ArrayRef<Value> values, Type srcTy, Type otherTy,
                       bool hasOther, unsigned vec) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto structElem = values[startIdx];
    Value offsetElem = b.extract_val(srcTy, structElem, 0);
    Value maskElem = b.extract_val(i1_ty, structElem, 1);
    // Gather other elements
    SmallVector<Value> otherElems;
    if (hasOther) {
      for (int i = 0; i < vec; i++) {
        otherElems.push_back(b.extract_val(otherTy, values[startIdx + i], 2));
      }
    }

    Value swizzleLaneOffset = b.extract_val(i32_ty, structElem, 3);

    return std::make_tuple(offsetElem, maskElem, std::move(otherElems),
                           swizzleLaneOffset);
  }

  // Replay a closed scalar-integer address DAG at the selected lane. Only
  // gfx936 kernel-entry integer parameters use the uniform dispatch ABI;
  // other block arguments remain unsupported. No convergent op is moved.
  Value tryRematerializeIntegerAtLane(RewriterBase &rewriter, Location loc,
                                    Value offset, Value selectedLane,
                                    bool allowUniformFrontier = false) const {
    bool isPredicate = offset.getType().isInteger(1);
    if ((!offset.getType().isInteger(32) && !isPredicate) ||
        lookupThreadsPerWarp(rewriter) != 64)
      return {};
    DenseMap<Value, Value> scalarized;
    unsigned aggregateSteps = 0;
    auto scalarize = [&](Value original) -> Value {
      if (auto it = scalarized.find(original); it != scalarized.end())
        return it->second;
      Value value = original;
      while (auto extract = value.getDefiningOp<LLVM::ExtractValueOp>()) {
        auto position = extract.getPosition();
        auto type =
            dyn_cast<LLVM::LLVMStructType>(extract.getContainer().getType());
        if (position.size() != 1 || !type || type.isOpaque() ||
            position[0] < 0 || position[0] >= type.getBody().size() ||
            type.getBody()[position[0]] != value.getType())
          return {};
        Value container = extract.getContainer();
        while (true) {
          if (++aggregateSteps > (allowUniformFrontier ? 4096u : 512u)) {
            LDBG("selected-lane: aggregate scan budget exhausted");
            return {};
          }
          auto insert = container.getDefiningOp<LLVM::InsertValueOp>();
          if (!insert || insert.getPosition().size() != 1 ||
              insert.getContainer().getType() != type ||
              insert.getType() != type)
            return {};
          int64_t field = insert.getPosition()[0];
          if (field < 0 || field >= type.getBody().size() ||
              insert.getValue().getType() != type.getBody()[field])
            return {};
          if (field == position[0]) {
            value = insert.getValue();
            break;
          }
          container = insert.getContainer();
        }
      }
      scalarized[original] = value;
      return value;
    };
    auto scalarInt = [](Type type) {
      return type.isInteger(1) || type.isInteger(32) || type.isInteger(64);
    };
    llvm::SmallDenseSet<Value, 64> visited;
    SmallVector<Operation *> ordered;
    SmallVector<Value> tids;
    SmallVector<Value> uniformKernelArgs;
    llvm::SmallDenseSet<Value, 64> uniformFrontier;
    std::function<bool(Value)> visit = [&](Value original) {
      Value value = scalarize(original);
      if (!value || !scalarInt(value.getType())) {
        LDBG("selected-lane: cannot scalarize " << original);
        return false;
      }
      // Keep the older explicit-uniform-split path exactly as before. Reused
      // uniform frontier values are not cloned DAG nodes; account for them
      // separately from the unchanged 64-node cloning budget. Whole offsets
      // can require more bounded aggregate unpacking than the older varying
      // fragment (4096 versus 512 insert/extract visits).
      if (allowUniformFrontier && isRematerializationUniform(value)) {
        if (uniformFrontier.insert(value).second)
          uniformKernelArgs.push_back(value);
        return true;
      }
      if (!visited.insert(value).second)
        return true;
      if (visited.size() > 64) {
        LDBG("selected-lane: value budget exhausted");
        return false;
      }
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        auto function = dyn_cast<LLVM::LLVMFuncOp>(
            argument.getOwner()->getParentOp());
        auto kernelMarker = function
                                ? function->getAttrOfType<IntegerAttr>("nvvm.kernel")
                                : IntegerAttr{};
        // FuncOpConversion sets nvvm.kernel for HCU kernel entries too.
        // Require its marker and the entry block, not merely a scalar type
        // or the visibility of an arbitrary LLVM function.
        if (targetInfo.getArch() != "gfx936" || !function ||
            !triton::isKernel(function) || !kernelMarker ||
            !kernelMarker.getValue().isOne() ||
            argument.getOwner() != &function.getBody().front()) {
          LDBG("selected-lane: unproven uniform argument " << value);
          return false;
        }
        uniformKernelArgs.push_back(value);
        return true;
      }
      Operation *op = value.getDefiningOp();
      if (!op || op->getNumResults() != 1 || op->getNumRegions() ||
          op->getNumSuccessors())
        return false;
      bool physicalThreadX = isa<ROCDL::ThreadIdXOp>(op);
      // getThreadId's exact pre-ROCDL spelling; replace the whole i32 leaf.
      if (auto cast = dyn_cast<arith::IndexCastOp>(op))
        if (cast.getIn().getType().isIndex() && value.getType().isInteger(32))
          if (auto tid = cast.getIn().getDefiningOp<::mlir::gpu::ThreadIdOp>())
            physicalThreadX = tid.getDimension() == ::mlir::gpu::Dimension::x;
      if (physicalThreadX) {
        if (!value.getType().isInteger(32))
          return false;
        tids.push_back(value);
        return true;
      }
      bool allowed = isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::AndOp,
                         LLVM::OrOp, LLVM::XOrOp, LLVM::SExtOp, LLVM::ZExtOp,
                         LLVM::TruncOp, LLVM::ICmpOp, LLVM::SelectOp>(op);
      if (auto constant = dyn_cast<LLVM::ConstantOp>(op))
        allowed = isa<IntegerAttr>(constant.getValue());
      if (isa<LLVM::BitcastOp>(op))
        allowed = op->getOperand(0).getType() == value.getType();
      if (isa<LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::UDivOp,
              LLVM::URemOp>(op)) {
        Value rhs = scalarize(op->getOperand(1));
        APInt amount;
        if (!rhs || !matchPattern(rhs, m_ConstantInt(&amount)))
          return false;
        if (isa<LLVM::UDivOp, LLVM::URemOp>(op))
          allowed = !amount.isNegative() && amount.isPowerOf2();
        else
          allowed = amount.ult(cast<IntegerType>(value.getType()).getWidth());
      }
      // Relative thread IDs and other scalar integer address arithmetic may
      // still be in arith during this conversion. Match the same closed
      // semantics, only on the newly proven-frontier path. Do not lower or
      // move an unknown op merely because its result type is scalar.
      if (allowUniformFrontier) {
        allowed |= isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                       arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                       arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                       arith::CmpIOp, arith::SelectOp>(op);
        if (auto constant = dyn_cast<arith::ConstantOp>(op))
          allowed |= isa<IntegerAttr>(constant.getValue());
        if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp,
                arith::DivUIOp, arith::RemUIOp>(op)) {
          Value rhs = scalarize(op->getOperand(1));
          APInt amount;
          if (!rhs || !matchPattern(rhs, m_ConstantInt(&amount)))
            return false;
          if (isa<arith::DivUIOp, arith::RemUIOp>(op))
            allowed = !amount.isNegative() && amount.isPowerOf2();
          else
            allowed = amount.ult(cast<IntegerType>(value.getType()).getWidth());
        }
      }
      if (!allowed) {
        LDBG("selected-lane: unsupported operation " << *op);
        return false;
      }
      for (Value operand : op->getOperands())
        if (!visit(operand))
          return false;
      ordered.push_back(op);
      return true;
    };
    if (!visit(offset) || (!isPredicate && tids.empty())) {
      LDBG("selected-lane: fallback, frontier=" << allowUniformFrontier
           << ", values=" << visited.size() << ", aggregates=" << aggregateSteps
           << ", thread leaves=" << tids.size());
      return {};
    }
    LDBG("selected-lane: success, frontier=" << allowUniformFrontier
         << ", values=" << visited.size() << ", aggregates=" << aggregateSteps);
    // Validation above emitted no IR. Preserve physical warp bits and replace
    // all workitem-id leaves together. Reuse only ABI-proven entry parameters;
    // no program-id, PHI, device-function parameter or memory read is guessed.
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value tid = ROCDL::ThreadIdXOp::create(rewriter, loc, i32_ty);
    Value selectedTid = b.or_(b.and_(tid, b.i32_val(-64)), selectedLane);
    IRMapping mapping;
    for (Value argument : uniformKernelArgs)
      mapping.map(argument, argument);
    for (Value leaf : tids)
      mapping.map(leaf, selectedTid);
    for (Operation *op : ordered) {
      for (Value operand : op->getOperands())
        mapping.map(operand, mapping.lookup(scalarized.lookup(operand)));
      rewriter.clone(*op, mapping);
    }
    return mapping.lookup(scalarized.lookup(offset));
  }

  void applySwizzling(RewriterBase &rewriter, Location loc, Value &srcOrOffset,
                      Value &mask, Value laneId,
                      Value swizzleLaneOffset,
                      bool rematerializeOffset = false,
                      bool wholeOffset = false,
                      bool fullWave = false) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    // laneId + swizzleOffset will always stay inside the warp [0,
    // threadsPerWarp) because we only swizzle inside a warp
    Value swizzledLaneId = b.add(laneId, swizzleLaneOffset);
    // Shuffle based on swizzleLaneId to apply the swizzling
    Value recomputed;
    if (rematerializeOffset) {
      recomputed = tryRematerializeIntegerAtLane(
          rewriter, loc, srcOrOffset, swizzledLaneId, wholeOffset);
      // Preserve every previously successful explicit split. Only its failed
      // closed-DAG attempt needs the additional uniform-frontier proof.
      if (!recomputed && !wholeOffset)
        recomputed = tryRematerializeIntegerAtLane(
            rewriter, loc, srcOrOffset, swizzledLaneId, true);
    }
    srcOrOffset = recomputed ? recomputed
                            : targetInfo.shuffleIdx(rewriter, loc, srcOrOffset,
                                                    swizzledLaneId);

    if (mask) {
      Value remappedMask;
      if (fullWave)
        remappedMask = tryRematerializeIntegerAtLane(
            rewriter, loc, mask, swizzledLaneId, /*allowUniformFrontier=*/true);
      mask = remappedMask ? remappedMask
                          : shuffleMask(rewriter, b, loc, targetInfo,
                                        swizzledLaneId, mask);
    }
  }

  LogicalResult lowerDirectToLDSLoad(
      RewriterBase &rewriter, Location loc, RankedTensorType srcTy,
      MemDescType dstTy, SmallVector<Value> loadVals, Value llDst,
      Type resElemTy, unsigned vec, triton::HCU::ISAFamily isaFamily,
      std::function<SmallVector<Value>(RewriterBase &, Location,
                                       ArrayRef<Value>, Value, int, VectorType,
                                       Value)>
          lowerInst) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();

    // Build src to shared layout and remove broadcasted registers
    auto srcLayout = triton::gpu::toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    loadVals = removeBroadcastSrc.apply(loadVals);

    LinearLayout sharedLayout;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto cvt = srcLayout.invertAndCompose(sharedLayout);
    if (!cvt.isTrivialOver({str_attr("block")})) {
      return emitError(
          loc,
          "direct to lds loads do not support non-trivial block dimension");
    }
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    Value ctaMulticastMask;
    if (isaFamily == ISAFamily::GFX1250) {
      ctaMulticastMask = LLVM::HCU::emitCtaMulticastMask(
          rewriter, loc, targetInfo.getClusterCTAId(rewriter, loc), srcLayout);
    }

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, resElemTy, rewriter);
    auto affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    auto maskSpanAffineOffset = SharedMemoryObject::getMaskSpanOffsets(dstTy);

    Value laneId, warpId;
    if (ISAFamily::CDNA3 == isaFamily || ISAFamily::CDNA4 == isaFamily) {
      // On GFX9, there is no dedicated hardware instruction to read `wave_id`.
      // The value is instead computed from `workitem.id.x`. Per the GFX9 ABI,
      // `workitem.id.x` is initialized in a vector register, and vector
      // instructions are generated for IR operations that depend on `wave_id`.
      //
      // A `v_readfirstlane` instruction is inserted at the end of these vector
      // sequences to transfer the value from a vector register to a scalar
      // register, initializing `$m0`.

      // When this sequence occurs inside a loop, the MachineLICM pass does not
      // hoist it because `v_readfirstlane` is convergent. Since both
      // `workitem.id.x` and `wave_id` are constant at runtime, their
      // computation can be hoisted to the nearest isolated region entry.
      // A warp-specialized worker cannot use an SSA value from the kernel
      // entry, and its wave ID must remain relative to its own partition.
      auto insertPt = rewriter.saveInsertionPoint();
      Region *scope = triton::HCU::warpSpecializeScope(
          &insertPt.getBlock()->front());
      rewriter.setInsertionPointToStart(&scope->front());

      std::tie(laneId, warpId) = getLaneAndWarpId(rewriter, loc);
      auto call = LLVM::createLLVMIntrinsicCallOp(
          rewriter, loc, "llvm.amdgcn.readfirstlane", {i32_ty}, {warpId});
      warpId = call.getResult(0);
      rewriter.restoreInsertionPoint(insertPt);
    } else {
      std::tie(laneId, warpId) = getLaneAndWarpId(rewriter, loc);
    }

    auto calcPaddedOffset = [&](Value smemOffset) {
      TritonLLVMOpBuilder b(loc, rewriter);
      auto bitwidth = dstTy.getElementTypeBitWidth();
      if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
              dstTy.getEncoding())) {
        // Apply the offset needed for padding.
        Value padOffset = emitPadding(loc, rewriter, paddedEnc, bitwidth,
                                      smemOffset, /*offsetInBytes=*/true);
        smemOffset = b.add(smemOffset, padOffset);
      }
      return smemOffset;
    };

    auto lowerInstForwardMulticastMask =
        [&](RewriterBase &rewriter, Location loc, ArrayRef<Value> vals,
            Value shmemAddr, int idx, VectorType vecTy) {
          return lowerInst(rewriter, loc, vals, shmemAddr, idx, vecTy,
                           ctaMulticastMask);
        };

    // If we do not support scattering (GFX9) the address should be the start
    // address (scalar) of the warp
    laneId = targetInfo.supportsDirectToLDSScattering() ? laneId : b.i32_val(0);
    lowerLdSt(loc, ctx, cvt, loadVals, resElemTy, smemObj.getBase(),
              calcPaddedOffset, affineOffset, maskSpanAffineOffset, laneId,
              warpId, rewriter, targetInfo, vec, lowerInstForwardMulticastMask);
    return success();
  }

  void emitOtherStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter, VectorType vecTy,
                      Value mask, ArrayRef<Value> otherElems, Value shmemAddr,
                      Value laneId, bool requiresSrcPtrSwizzling,
                      Value swizzleLaneOffset) const {
    TritonLLVMOpBuilder b(loc, rewriter);
    Value storeVal = packElementRangeIntoVector(rewriter, typeConverter, loc,
                                                vecTy, otherElems, 0);
    Type ptrTy = shmemAddr.getType();
    Value ldsAddr = shmemAddr;
    // When scattering is unsupported, shmemAddr is the warp base address.
    // Use shmemAddr + lane_id [+ swizzleOffset] to compute each lane's address.
    if (!targetInfo.supportsDirectToLDSScattering()) {
      ldsAddr = b.gep(ptrTy, vecTy, shmemAddr, laneId);
      if (requiresSrcPtrSwizzling)
        ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzleLaneOffset);
    }
    llStore(rewriter, loc, ldsAddr, storeVal, b.icmp_ne(mask, b.true_val()),
            CacheModifier::NONE, targetInfo.requiresAliasInfoForAsyncOps());
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const HCU::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> otherElems;
    if (other)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    Value multicastMask;
    auto mod = op->getParentOfType<ModuleOp>();
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs > 1) {
      Value clusterCTAId = targetInfo.getClusterCTAId(rewriter, loc);
      auto regLayout =
          triton::gpu::toLinearLayout(cast<RankedTensorType>(ptr.getType()));
      multicastMask = LLVM::HCU::emitCtaMulticastMask(rewriter, loc,
                                                      clusterCTAId, regLayout);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);

      Value loadVal = llLoad(rewriter, loc, ptr, vecTy, pred, falseVal,
                             multicastMask, cacheMod, op.getIsVolatile());
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::BufferLoadOp>,
      public LoadStoreConversionBase {
  BufferLoadOpConversion(LLVMTypeConverter &converter,
                         const HCU::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::BufferLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::HCU::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto cacheMod = op.getCache();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);
    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offset
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    assert(offsetElems.size() == numElems);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // Get the `other` value (if any)
    SmallVector<Value> otherElems;
    if (llOther && !isAllZeroBitsConstant(other))
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // FIXME: Explicitly disable cache swizzling(introduced by oai commit
    // 7fffa0df) for
    //        buffer ops on HCUs as more hcu-specific details (stride field
    //        particularly) should be considered.
    // Create the resource descriptor and then emit the buffer_load intrinsic(s)
    // Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    auto module = op->getParentOfType<ModuleOp>();
    auto target = module->getAttrOfType<StringAttr>("ttg.target");
    // Nonzero, negative-zero, and unknown fallback values retain the entire
    // old predicate/select path. This optimization changes regular raw loads
    // only; direct-to-LDS transfers and their synchronization are untouched.
    if (mask && otherElems.empty() && target &&
        target.getValue() == "hip:gfx936")
      factorUniformBufferLoadMask(rewriter, loc, rsrcDesc, maskElems, vec);
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);
      Value loadVal = bufferEmitter.emitLoad(
          vecTy, rsrcDesc, offsetElems[vecStart], pred, falseVal, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::BufferLoadToLocalOp>,
      public DirectToLdsLoadConversionBase {
  BufferLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                const HCU::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass),
        fullWaveCopies(collectFullWaveBufferCopies(axisAnalysisPass.getModuleOp(),
                                                    targetInfo.getArch())) {}

  const llvm::DenseSet<Operation *> fullWaveCopies;

  LogicalResult
  matchAndRewrite(triton::hcugpu::BufferLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::HCU::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // Original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llDst = adaptor.getDest();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    RankedTensorType ptrType =
        cast<RankedTensorType>(getPointerTypeWithShape(ptr, offset));

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> otherElems;
    // gfx936 raw.ptr G2L writes zero for the emitter's OOB sentinel, even
    // when the destination slot held nonzero data. Explicit +0 therefore
    // needs neither conditional DMA nor a second LDS zero store. Preserve
    // all other bit patterns (including -0), and other architectures.
    if (llOther && !(targetInfo.getArch() == "gfx936" &&
                     isAllZeroBitsConstant(op.getOther())))
      otherElems = unpackLLElements(loc, llOther, rewriter);

    auto dstTy = op.getDest().getType();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto dstEnc = dstTy.getEncoding();

    // If the op has a contiguity hint use it to increase the vector size.
    vec = std::max(vec, op.getContiguity());

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool requiresSrcPtrSwizzling =
        !targetInfo.supportsDirectToLDSScattering() && maybeSwizzledEnc &&
        maybeSwizzledEnc.getMaxPhase() != 1;
    if (failed(canWriteCoalesced(rewriter, op, ptrType, dstTy, vec,
                                 requiresSrcPtrSwizzling, /*isBuffer=*/true))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;

    if (requiresSrcPtrSwizzling) {
      // TODO (alex): this is only correct as long as the lds view is a
      // contiguous block. So this can break if we slice along the 2 minor
      // dimensions.
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCTALayout());
      flatDstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, ptrType, dstTy, flatDstTy, llDst, resElemTy, vec);
    }

    // Keep an explicit uniform term outside the lane shuffle.  Splat is the
    // same uniform/non-uniform boundary used by CanonicalizePointers; do not
    // infer uniformity merely from a lowered scalar LLVM type.
    Value uniformOffset, uniformSource, varyingSource;
    if (requiresSrcPtrSwizzling &&
        cast<RankedTensorType>(offset.getType()).getElementType().isInteger(32)) {
      Value sum = offset;
      if (auto trunc = sum.getDefiningOp<arith::TruncIOp>())
        sum = trunc.getIn();
      if (auto add = sum.getDefiningOp<arith::AddIOp>()) {
        auto splat = add.getLhs().getDefiningOp<triton::SplatOp>();
        Value varying = add.getRhs();
        if (!splat) {
          splat = add.getRhs().getDefiningOp<triton::SplatOp>();
          varying = add.getLhs();
        }
        if (splat && (splat.getSrc().getType().isInteger(32) ||
                      splat.getSrc().getType().isInteger(64))) {
          Value llUniform = rewriter.getRemappedValue(splat.getSrc());
          Value llVarying = rewriter.getRemappedValue(varying);
          if (llUniform && llVarying) {
            auto varyingElems = unpackLLElements(loc, llVarying, rewriter);
            if (varyingElems.size() == offsetElems.size()) {
              // Truncation distributes over addition modulo 2^32.  Keep the
              // element-to-byte conversion and mask/OOB select in the emitter.
              uniformSource = splat.getSrc();
              varyingSource = varying;
              uniformOffset = llUniform.getType().isInteger(32)
                                  ? llUniform
                                  : b.trunc(i32_ty, llUniform);
              for (auto &elem : varyingElems)
                if (!elem.getType().isInteger(32))
                  elem = b.trunc(i32_ty, elem);
              offsetElems = std::move(varyingElems);
            }
          }
        }
      }
    }

    auto offsetTy = offsetElems[0].getType();
    bool hasOther = !otherElems.empty();
    auto otherTy = hasOther ? otherElems[0].getType() : i1_ty;
    // Zip buffer_offset, mask, other, swizzleOffsets for lowerLdSt
    auto loadVals =
        zipLoadValues(rewriter, loc, vec, offsetElems, offsetTy, maskElems,
                      otherElems, otherTy, swizzledLaneOffsets);

    // FIXME: Explicitly disable cache swizzling(introduced by oai commit
    // 83229ced0479) for
    //        buffer atomic ops on HCUs as more hcu-specific details (stride
    //        field particularly) should be considered.
    // Create the resource descriptor and then emit the buffer_loads to lds
    // based on the collected shared addresses and vector size
    // Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    bool rebaseUniform = false;
    Value uniformByteOffset;
    if (uniformOffset && targetInfo.getArch() == "gfx936" &&
        isRematerializationUniform(uniformOffset)) {
      auto uMax = getNonNegativeOffsetMax(uniformSource);
      auto vMax = getNonNegativeOffsetMax(varyingSource);
      unsigned elemBytes = std::max(8u, resElemTy.getIntOrFloatBitWidth()) / 8;
      uint64_t vectorBytes = uint64_t(vec) * elemBytes;
      // This also proves the peeled i64->i32 truncation and element->byte
      // conversion exact, and keeps every valid byte below the mask sentinel.
      uint64_t maxElementOffset = (uint64_t(INT32_MAX) -
                                  (vectorBytes - 1)) / elemBytes;
      rebaseUniform = uMax && vMax && *uMax && *vMax <= maxElementOffset &&
                      *uMax <= maxElementOffset - *vMax;
      if (rebaseUniform)
        uniformByteOffset = b.mul(uniformOffset, b.i32_val(elemBytes));
    }
    Value rsrcDesc = rebaseUniform
        ? bufferEmitter.createResourceDescriptor(llPtr, Value{}, uniformByteOffset)
        : bufferEmitter.createResourceDescriptor(llPtr);

    Value threadPred = emitRedundantThreadPredicate(
        getFreeVariableMasks(ptrType), rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitBufferLoadLds =
        [this, &op, &b, &bufferEmitter, &rsrcDesc, laneId = laneId, threadPred,
         offsetTy, otherTy, hasOther, requiresSrcPtrSwizzling, uniformOffset,
         rebaseUniform, fullWave = fullWaveCopies.contains(op.getOperation())](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> loadVals,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value multicastMask) -> SmallVector<Value> {
      auto [offsetElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadVals, offsetTy, otherTy,
                          hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits, /*isBuffer=*/true));
      Value vecBytesVal = b.i32_val(vecBits / 8);

      Value maybeSwizzledMaskElem = maskElem;
      if (requiresSrcPtrSwizzling)
        applySwizzling(rewriter, loc, offsetElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset,
                       /*rematerializeOffset=*/true,
                       /*wholeOffset=*/!uniformOffset, fullWave);
      if (uniformOffset && !rebaseUniform)
        offsetElem = b.add(offsetElem, uniformOffset);

      // If other=0.0 we remove other in canonicalizePointers and we can use out
      // of bounds to store 0 to LDS. So if we have other values we need to
      // predicate to not overwrite the other stores
      Value cond =
          hasOther ? b.and_(threadPred, maybeSwizzledMaskElem) : threadPred;

      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      auto bufferLoadToLds = bufferEmitter.emitLoadToLds(
          vecTy, vecBytesVal, rsrcDesc, offsetElem, shmemAddr,
          hasOther ? b.true_val() : maybeSwizzledMaskElem, op.getCache());
      if (targetInfo.requiresAliasInfoForAsyncOps())
        HCU::addAsyncCopyAliasScope(bufferLoadToLds);

      rewriter.setInsertionPointToStart(afterLoadBlock);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, requiresSrcPtrSwizzling,
                       swizzleLaneOffset);
      }

      return {};
    };

    auto res = lowerDirectToLDSLoad(
        rewriter, loc, ptrType, flatDstTy, loadVals, llDst, resElemTy, vec,
        targetInfo.getISAFamily(), emitBufferLoadLds);
    if (failed(res)) {
      return failure();
    }

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public DirectToLdsLoadConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const HCU::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();

    auto dstTy = op.getResult().getType();
    auto dstEnc = dstTy.getEncoding();
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    Value llDst = adaptor.getResult();

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(op.getSrc(), axisAnalysisPass);
    auto maskElements = getMaskElemsAndUpdateVeclen(
        rewriter, loc, adaptor.getMask(), op.getMask(), vec);

    auto srcElems = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> otherElems;
    if (op.getOther())
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);

    // If the op has a contiguity hint use it to increase the vector size.
    vec = std::max(vec, op.getContiguity());

    // For padded encodings restrict vec by the min interval
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstEnc)) {
      vec = std::min(vec, padEnc.getMinInterval());
    }

    auto maybeSwizzledEnc = dyn_cast<SwizzledSharedEncodingAttr>(dstEnc);
    bool requiresSrcPtrSwizzling =
        !targetInfo.supportsDirectToLDSScattering() && maybeSwizzledEnc &&
        maybeSwizzledEnc.getMaxPhase() != 1;

    if (failed(canWriteCoalesced(rewriter, op, srcTy, dstTy, vec,
                                 requiresSrcPtrSwizzling))) {
      return failure();
    }

    // For swizzled layouts we need to use the non swizzled layout to compute
    // the LDS addresses since we gather into LDS
    auto flatDstTy = dstTy;
    SmallVector<Value> swizzledLaneOffsets;
    if (requiresSrcPtrSwizzling) {
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), maybeSwizzledEnc.getVec(), 1, 1,
          maybeSwizzledEnc.getOrder(), maybeSwizzledEnc.getCTALayout());
      flatDstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, srcTy, dstTy, flatDstTy, llDst, resElemTy, vec);
    }

    Type srcPtrTy = srcElems[0].getType();
    bool hasOther = !otherElems.empty();
    Type otherTy = hasOther ? otherElems[0].getType() : i1_ty;
    // Zip buffer_offset, mask, other, swizzleOffsets for lowerLdSt
    SmallVector<Value> loadVals =
        zipLoadValues(rewriter, loc, vec, srcElems, srcPtrTy, maskElements,
                      otherElems, otherTy, swizzledLaneOffsets);

    Value threadPred = emitRedundantThreadPredicate(getFreeVariableMasks(srcTy),
                                                    rewriter, loc, targetInfo);

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    auto emitGlobalLoadLds =
        [this, &op, &b, laneId = laneId, threadPred, srcPtrTy, otherTy,
         hasOther, requiresSrcPtrSwizzling](
            RewriterBase &rewriter, Location loc, ArrayRef<Value> loadValues,
            Value shmemAddr, int startIdx, VectorType vecTy,
            Value multicastMask) -> SmallVector<Value> {
      auto [srcElem, maskElem, otherElems, swizzleLaneOffset] =
          unzipLoadValues(rewriter, loc, startIdx, loadValues, srcPtrTy,
                          otherTy, hasOther, vecTy.getNumElements());
      int vecBits = vecTy.getNumElements() * vecTy.getElementTypeBitWidth();
      assert(targetInfo.supportsDirectToLdsLoadBitWidth(vecBits));
      Value maybeSwizzledMaskElem = maskElem;

      if (requiresSrcPtrSwizzling)
        applySwizzling(rewriter, loc, srcElem, maybeSwizzledMaskElem, laneId,
                       swizzleLaneOffset);

      // Predicate load based on threadPred && swizzledMask
      auto cond = b.and_(threadPred, maybeSwizzledMaskElem);
      auto [loadBlock, afterLoadBlock] = emitBranch(rewriter, loc, cond);

      emitAsyncLoad(rewriter, loc, targetInfo, vecBits, srcElem, shmemAddr,
                    op.getCache(), multicastMask);

      rewriter.setInsertionPointToStart(afterLoadBlock);

      if (hasOther) {
        emitOtherStore(rewriter, loc, this->getTypeConverter(), vecTy, maskElem,
                       otherElems, shmemAddr, laneId, requiresSrcPtrSwizzling,
                       swizzleLaneOffset);
      }

      return {};
    };

    auto res = lowerDirectToLDSLoad(
        rewriter, loc, srcTy, flatDstTy, loadVals, llDst, resElemTy, vec,
        targetInfo.getISAFamily(), emitGlobalLoadLds);
    if (failed(res)) {
      return failure();
    }

    // Drop the result token.
    Value zero = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                          IntegerType::get(op.getContext(), 32),
                                          rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }

  void emitAsyncLoad(RewriterBase &rewriter, Location loc,
                     HCU::TargetInfo targetInfo, int vecBits, Value srcPtr,
                     Value shmemAddr, triton::CacheModifier cacheMod,
                     Value multicastMask) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int32_t cacheModifiers =
        mlir::LLVM::HCU::getCtrlBitsForCacheModifierOnTarget(
            cacheMod, /*isLoad=*/true, targetInfo);

    if (llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily())) {
      auto globalLoadLdsOp = ROCDL::GlobalLoadLDSOp::create(
          rewriter, loc, srcPtr, shmemAddr, vecBits / 8,
          /*offset=*/0, cacheModifiers, nullptr, nullptr, nullptr);
      if (targetInfo.requiresAliasInfoForAsyncOps())
        HCU::addAsyncCopyAliasScope(globalLoadLdsOp);
    } else if (targetInfo.getISAFamily() == ISAFamily::GFX1250) {
      if (cacheMod != triton::CacheModifier::NONE) {
        emitRemark(loc) << "cache modifiers not yet implemented on gfx1250";
      }
      if (multicastMask) {
        std::string intrinsic =
            "llvm.amdgcn.cluster.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers),
             multicastMask});
      } else {
        std::string intrinsic =
            "llvm.amdgcn.global.load.async.to.lds.b" + std::to_string(vecBits);
        auto globalLoadLdsOp = LLVM::createLLVMIntrinsicCallOp(
            rewriter, loc, intrinsic, {},
            {srcPtr, shmemAddr, b.i32_val(0), b.i32_val(cacheModifiers)});
      }
    }
  }
};

struct AsyncTDMCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::hcugpu::AsyncTDMCopyGlobalToLocalOp>,
      public LoadStoreConversionBase {
  AsyncTDMCopyGlobalToLocalOpConversion(
      LLVMTypeConverter &converter, const HCU::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::AsyncTDMCopyGlobalToLocalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto tensorDescTy = op.getDesc().getType();
    auto smemTy = op.getResult().getType();
    auto paddedEnc =
        llvm::dyn_cast<PaddedSharedEncodingAttr>(smemTy.getEncoding());
    Type elementType = getTypeConverter()->convertType(smemTy.getElementType());

    unsigned padInterval = 0;
    unsigned padAmount = 0;
    if (paddedEnc) {
      assert(paddedEnc.getIntervals().size() == 1 &&
             paddedEnc.getPaddings().size() == 1);
      padInterval = paddedEnc.getIntervals()[0];
      padAmount = paddedEnc.getPaddings()[0];
    }

    auto mod = op->getParentOfType<ModuleOp>();
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs > 1)
      return rewriter.notifyMatchFailure(op, "NYI: Support multicast.");

    SmallVector<Value> desc =
        unpackLLElements(loc, adaptor.getDesc(), rewriter);

    SmallVector<int64_t> blockShape =
        llvm::to_vector(tensorDescTy.getBlockType().getShape());

    // 2D tensors: 12 dwords (group0: 4, group1: 8)
    // 3D-5D tensors: 20 dwords (group0: 4, group1: 8, group2: 4, group3: 4)
    assert((blockShape.size() <= 2 && desc.size() == 12) ||
           (blockShape.size() > 2 && desc.size() == 20));

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getResult(), elementType, rewriter);
    Value dstPtr = dstMemObj.getBase();
    SmallVector<Value> offset = adaptor.getIndices();
    int numWarps = triton::gpu::lookupNumWarps(op);

    Value barrierPtr = nullptr;
    if (op.getBarrier()) {
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(),
          typeConverter->convertType(
              op.getBarrier().getType().getElementType()),
          rewriter);
      barrierPtr = smemObj.getBase();
    }

    mlir::LLVM::HCU::emitTDMOperation(rewriter, loc, getTypeConverter(), desc,
                                      blockShape, numWarps, padInterval,
                                      padAmount, offset, dstPtr, op.getPred(),
                                      elementType, barrierPtr, /*isLoad=*/true);

    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncTDMCopyLocalToGlobalOpConversion
    : public ConvertOpToLLVMPattern<
          triton::hcugpu::AsyncTDMCopyLocalToGlobalOp>,
      public LoadStoreConversionBase {
  AsyncTDMCopyLocalToGlobalOpConversion(
      LLVMTypeConverter &converter, const HCU::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::AsyncTDMCopyLocalToGlobalOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto tensorDescTy = op.getDesc().getType();
    auto smemTy = op.getSrc().getType();
    Type elementType = getTypeConverter()->convertType(smemTy.getElementType());

    SmallVector<Value> desc =
        unpackLLElements(loc, adaptor.getDesc(), rewriter);

    SmallVector<int64_t> blockShape =
        llvm::to_vector(tensorDescTy.getBlockType().getShape());

    // 2D tensors: 12 dwords (group0: 4, group1: 8)
    // 3D-5D tensors: 20 dwords (group0: 4, group1: 8, group2: 4, group3: 4)
    assert((blockShape.size() <= 2 && desc.size() == 12) ||
           (blockShape.size() > 2 && desc.size() == 20));

    auto dstMemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getSrc(), elementType, rewriter);
    Value dstPtr = dstMemObj.getBase();
    SmallVector<Value> offset = adaptor.getIndices();
    int numWarps = triton::gpu::lookupNumWarps(op);

    mlir::LLVM::HCU::emitTDMOperation(
        rewriter, loc, getTypeConverter(), desc, blockShape, numWarps,
        /*padInterval=*/0, /*padAmount=*/0, offset, dstPtr, b.true_val(),
        elementType, /*barrierPtr=*/nullptr,
        /*isLoad=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const HCU::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

struct BufferAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::BufferAtomicRMWOp>,
      public LoadStoreConversionBase {
  BufferAtomicRMWOpConversion(LLVMTypeConverter &converter,
                              const HCU::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::BufferAtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::HCU::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // v4f16 and v4bf16 variants of buffer atomics do not exist.
    // only v2f16 and v2bf16.
    if (valueElemTy.isBF16() || valueElemTy.isF16()) {
      // We clamp to the only supported vectorization width here (2).
      // In ConvertToBufferOps we check that we have a large enough vector size
      assert(vec >= 2);
      vec = 2u;
      // The max width of a buffer atomic op is 64-bits
      // Some types like F32 don't have a 2x vectorized version
    } else if (valueElemTy.isF32() || valueElemTy.isF64() ||
               valueElemTy.isInteger(32) || valueElemTy.isInteger(64)) {
      vec = 1u;
    }

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // FIXME: Explicitly disable cache swizzling(introduced by oai commit
    // 83229ced0479) for
    //        buffer atomic ops on HCUs as more hcu-specific details (stride
    //        field particularly) should be considered.
    // Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops) see: https://llvm.org/docs/HCUGPUUsage.html#memory-model-gfx942
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastRMWOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    //    We set GLC=1, to return the old value. Atomics in GFX942 execute with
    //    either device (default) or system scope (controlled by the sc1 flag).
    //    This is distinct from the memory scope of the atomic (i.e, the memory
    //    fences which appear before/after the ops).

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = std::distance(opUsers.begin(), opUsers.end()) > 0;
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);

      Value loadVal = bufferEmitter.emitAtomicRMW(
          atomicRmwAttr, vecTy, rsrcDesc, offsetElems[vecStart], storeVal, pred,
          hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastRMWOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Acquire Fence post-atomic
    if (failed(emitFence(op, rewriter, lastRMWOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferAtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::BufferAtomicCASOp>,
      public LoadStoreConversionBase {
  BufferAtomicCASOpConversion(LLVMTypeConverter &converter,
                              const HCU::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::BufferAtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::HCU::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value cmp = op.getCmp();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llVal = adaptor.getVal();
    Value llCmp = adaptor.getCmp();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = val.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    // Max supported vectorization for i32 and i64 is 1x
    // on CDNA3 and CDNA4
    // BUFFER_ATOMIC_CMPSWAP(i32) and BUFFER_ATOMIC_CMPSWAP_X2(i64)
    unsigned vec = 1u;

    // Get the offsets, val, and cmp
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valElems = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> cmpElems = unpackLLElements(loc, llCmp, rewriter);

    // FIXME: Explicitly disable cache swizzling(introduced by oai commit
    // 83229ced0479) for
    //        buffer atomic ops on HCUs as more hcu-specific details (stride
    //        field particularly) should be considered.
    // Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops)
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    // Release Fence pre-atomic
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastCASOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = !op.getResult().getUsers().empty();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value pred = threadPred;
      // Create the store val
      Value casStoreVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valElems, vecStart);
      // Create the cmp val
      Value casCmpVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          cmpElems, vecStart);

      Value loadVal =
          bufferEmitter.emitAtomicCAS(vecTy, rsrcDesc, offsetElems[vecStart],
                                      casCmpVal, casStoreVal, pred, hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastCASOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Emit post-atomic acquire fence
    if (failed(emitFence(op, rewriter, lastCASOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::BufferStoreOp>,
      public LoadStoreConversionBase {
  BufferStoreOpConversion(LLVMTypeConverter &converter,
                          const HCU::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::BufferStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::HCU::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto cacheMod = op.getCache();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // FIXME: Explicitly disable cache swizzling(introduced by oai commit
    // 7fffa0df) for
    //        buffer ops on HCUs as more hcu-specific details (stride field
    //        particularly) should be considered.
    // Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    // Rebase only an explicit splatted uniform term. A runtime unsigned
    // bound handles arbitrary origin bits; the independently proven local
    // range guarantees exact byte arithmetic. This is a module opt-in, not
    // an input proof. Unknown layout/arithmetic keeps the original path.
    Value storeUniformByteOffset;
    auto storeModule = op->getParentOfType<ModuleOp>();
    auto rebaseAttr = storeModule->getAttrOfType<BoolAttr>(
        "ttg.hcu.rebase_buffer_store");
    if (rebaseAttr && rebaseAttr.getValue() &&
        targetInfo.getArch() == "gfx936" && !llStride &&
        cast<RankedTensorType>(offset.getType()).getElementType().isInteger(32)) {
      Value sum = offset;
      if (auto trunc = sum.getDefiningOp<arith::TruncIOp>()) sum = trunc.getIn();
      if (auto add = sum.getDefiningOp<arith::AddIOp>()) {
        for (unsigned side = 0; side != 2; ++side) {
          Value uniformTensor = add->getOperand(side);
          if (auto ext = uniformTensor.getDefiningOp<arith::ExtSIOp>())
            uniformTensor = ext.getIn();
          else if (auto ext = uniformTensor.getDefiningOp<arith::ExtUIOp>())
            uniformTensor = ext.getIn();
          auto splat = uniformTensor.getDefiningOp<triton::SplatOp>();
          Value varying = add->getOperand(1-side);
          if (!splat ||
              (!splat.getSrc().getType().isInteger(32) &&
               !splat.getSrc().getType().isInteger(64))) continue;
          Value uniform = rewriter.getRemappedValue(splat.getSrc());
          Value local = rewriter.getRemappedValue(varying);
          if (!uniform || !local || !isBufferStoreRebaseUniform(uniform)) continue;
          unsigned elemBytes = std::max(8u, valueElemTy.getIntOrFloatBitWidth())/8;
          uint64_t vectorBytes = uint64_t(vec)*elemBytes;
          if (!vectorBytes || vectorBytes > uint64_t(INT32_MAX)) continue;
          uint64_t maxElement = (uint64_t(INT32_MAX)-(vectorBytes-1))/elemBytes;
          // A uniform runtime interval also handles dynamic nonnegative
          // strides. Each bound is clamped to zero on failure; arithmetic on
          // these bounded i64 values cannot overflow. Failure only chooses
          // the original descriptor/offset path, never an unchecked access.
          using Bound = std::pair<Value,Value>; // nonnegative maximum, valid
          DenseMap<Value,Bound> knownBounds;
          unsigned boundSteps = 0;
          std::function<std::optional<Bound>(Value,unsigned)> bound;
          bound = [&](Value source,unsigned depth)->std::optional<Bound> {
            if (depth>16 || ++boundSteps>512) return std::nullopt;
            if (auto it=knownBounds.find(source);it!=knownBounds.end()) return it->second;
            auto type=dyn_cast<IntegerType>(getElementTypeOrSelf(source.getType()));
            if (!type || (type.getWidth()!=32 && type.getWidth()!=64)) return std::nullopt;
            auto save=[&](Value upper,Value valid)->std::optional<Bound> {
              Bound result{b.select(valid,upper,b.i64_val(0)),valid};
              knownBounds[source]=result;return result;
            };
            if (auto maximum=getNonNegativeOffsetMax(source)) {
              bool fits=*maximum<=maxElement;
              return save(b.i64_val(fits?*maximum:0),b.i1_val(fits));
            }
            Operation *def=source.getDefiningOp();
            if (!def) return std::nullopt;
            if (auto splatValue=dyn_cast<triton::SplatOp>(def)) {
              Value scalar=rewriter.getRemappedValue(splatValue.getSrc());
              if (!scalar || !isBufferStoreRebaseUniform(scalar)) return std::nullopt;
              if (scalar.getType().isInteger(32)) scalar=b.sext(i64_ty,scalar);
              else if (!scalar.getType().isInteger(64)) return std::nullopt;
              // Unsigned comparison rejects negative signed values too.
              Value valid=b.icmp_ule(scalar,b.i64_val(maxElement));
              return save(scalar,valid);
            }
            if (isa<triton::ExpandDimsOp,triton::BroadcastOp,ConvertLayoutOp,
                    arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(def)) {
              auto inner=bound(def->getOperand(0),depth+1);
              if (inner) knownBounds[source]=*inner;
              return inner;
            }
            if (!isa<arith::AddIOp,arith::MulIOp>(def)) return std::nullopt;
            auto lhs=bound(def->getOperand(0),depth+1);
            auto rhs=bound(def->getOperand(1),depth+1);
            if (!lhs || !rhs) return std::nullopt;
            Value upper;
            if (isa<arith::AddIOp>(def)) upper=b.add(lhs->first,rhs->first);
            else upper=b.mul(lhs->first,rhs->first);
            Value valid=b.and_(b.and_(lhs->second,rhs->second),
                               b.icmp_ule(upper,b.i64_val(maxElement)));
            return save(upper,valid);
          };
          auto range=bound(varying,0);
          if (!range) continue;
          auto locals = unpackLLElements(loc, local, rewriter);
          if (locals.size() != offsetElems.size()) continue;
          if (!uniform.getType().isInteger(32)) uniform = b.trunc(i32_ty,uniform);
          Value threshold=b.trunc(i32_ty,b.sub(b.i64_val(maxElement),range->first));
          Value fits=b.and_(range->second,b.icmp_ule(uniform,threshold));
          Value chosen = b.select(fits,uniform,b.i32_val(0));
          Value remainder = b.sub(uniform,chosen);
          for (Value &element : locals) {
            if (!element.getType().isInteger(32)) element=b.trunc(i32_ty,element);
            element=b.add(element,remainder);
          }
          offsetElems=std::move(locals);
          storeUniformByteOffset=b.mul(chosen,b.i32_val(elemBytes));
          break;
        }
      }
    }
    Value rsrcDesc = storeUniformByteOffset
        ? bufferEmitter.createResourceDescriptor(llPtr,Value{},storeUniformByteOffset)
        : bufferEmitter.createResourceDescriptor(llPtr);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    // SOFFSET can bypass resource bounds on some buffer modes. The planner
    // independently proves the complete original lowered scalar offset, for
    // every possible bit pattern of its inputs (without using the predicate),
    // covers its full i16 within the unchanged raw descriptor. A lost tensor
    // producer during conversion cannot supply or invalidate that proof.
    // Unknown arithmetic or rebased resources keep emitStore per element.
    auto scalarOffsetAttr = storeModule->getAttrOfType<BoolAttr>(
        "ttg.hcu.buffer_store_scalar_offset");
    auto splitImmediateAttr = storeModule->getAttrOfType<BoolAttr>(
        "ttg.hcu.buffer_store_split_immediate");
    auto preserveMaskedBaseAttr = storeModule->getAttrOfType<BoolAttr>(
        "ttg.hcu.buffer_store_preserve_masked_base");
    bool useScalarOffset = false;
    if (scalarOffsetAttr && scalarOffsetAttr.getValue() &&
        targetInfo.getArch() == "gfx936" &&
        (vec == 1 || vec == 2 || vec == 4 || vec == 8) && !llStride &&
        !storeUniformByteOffset && valueElemTy.getIntOrFloatBitWidth() == 16 &&
        cacheMod == triton::CacheModifier::NONE) {
      // Each plan() must prove the full original offset <= MaxElement and
      // the exact nonnegative core+constant identity before any IR is emitted.
      useScalarOffset = true;
    }
    BufferStoreScalarOffset scalarOffsets;
    unsigned scalarOffsetSites = 0;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      if (useScalarOffset) {
        if (auto plan = scalarOffsets.plan(offsetElems[vecStart], vec)) {
          Value core = scalarOffsets.emitCore(plan, rewriter, loc);
          Value bytes = b.mul(b.i32_val(2), core);
          Value masked = b.select(pred, bytes, b.i32_val(INT32_MIN));
          // Preserve the original vector store's payload and predicate. Only
          // the independently proved address decomposition changes; there is
          // no scalarization, mask relaxation, descriptor or cache change.
          Type payloadType = i16_ty;
          if (vec == 2) payloadType = i32_ty;
          else if (vec > 2) payloadType = LLVM::getVectorType(i32_ty, vec / 2);
          Value payload = b.bitcast(storeVal, payloadType);
          Value voffset = masked;
          uint64_t scalarBytes = plan->constant * 2;
          if (splitImmediateAttr && splitImmediateAttr.getValue()) {
            // gfx936 MUBUF has a 12-bit byte immediate. Keep the original
            // predicate/sentinel, and expose only this low displacement in
            // VOFFSET so instruction selection may fold it into offset:imm.
            uint64_t lowBytes = scalarBytes & 4095;
            scalarBytes -= lowBytes;
            // The full-domain plan gives coreBytes+allBytes <= NR-2.
            // For a false predicate INT_MIN+lowBytes <= 0x80000fff.
            // Both cases prove this unsigned addition cannot wrap.
            if (lowBytes) {
              // Exact i32 register identity. Keep the bounded masked base
              // opaque to arithmetic reassociation across the select. This
              // requests instruction selection; it is not a range proof.
              // Tied allocation/CSE may still add copies or merge identities.
              if (preserveMaskedBaseAttr && preserveMaskedBaseAttr.getValue())
                masked = LLVM::InlineAsmOp::create(
                    rewriter, loc, masked.getType(), ValueRange{masked},
                    "", "=v,0", false, false, LLVM::TailCallKind::None,
                    LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
                    ArrayAttr::get(ctx, {})).getRes();
              voffset = b.add(masked, b.i32_val(lowBytes),
                              LLVM::IntegerOverflowFlags::nuw);
            }
          }
          SmallVector<Value, 5> args{payload, rsrcDesc, voffset,
              b.i32_val(scalarBytes), b.i32_val(0)};
          ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, TypeRange{}, args,
                                            ArrayRef<NamedAttribute>());
          ++scalarOffsetSites;
          continue;
        }
      }
      bufferEmitter.emitStore(rsrcDesc, offsetElems[vecStart], storeVal, pred,
                              cacheMod);
    } // end vec

    if (scalarOffsetSites)
      op.emitRemark("buffer-store scalar-offset proven sites: ") << scalarOffsetSites;
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const HCU::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op, "Unknown HCUGPU memory ordering");
    auto scope = getHCUGPUMemScopeStr(op.getScope());
    if (!scope)
      return rewriter.notifyMatchFailure(op, "Unknown HCUGPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    Type valueElemIntTy{};
    if (!valueElemTy.isSignlessInteger()) {
      valueElemIntTy = rewriter.getIntegerType(valueElemNBits);
    }
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    SmallVector<Value> resultVals(elemsPerThread);

    auto successOrdering = *atomicMemOrdering;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto scopeStr = StringRef(scope.value());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      if (tensorTy && (i & ~regMask) != i) {
        resultVals[i] = resultVals[i & ~regMask];
        continue;
      }
      Value casVal = valElements[i];
      Value casCmp = cmpElements[i];
      Value casPtr = ptrElements[i];
      if (valueElemIntTy) {
        casVal = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casVal);
        casCmp = LLVM::BitcastOp::create(rewriter, loc, valueElemIntTy, casCmp);
      }
      // use op
      if (tensorTy) { // for tensor
        auto retType = valueElemTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        Value undefVal = b.undef(valueElemTy);
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));
        endBlock->addArgument({valueElemTy}, {loc});

        rewriter.setInsertionPointToEnd(curBlock);
        LLVM::CondBrOp::create(rewriter, loc, threadPred, atomicBlock, endBlock,
                               undefVal);

        rewriter.setInsertionPointToEnd(atomicBlock);

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        // Extract the new_loaded value from the pair.
        Value ret;
        if (valueElemIntTy) {
          ret = b.extract_val(valueElemIntTy, cmpxchg, 0);
          ret = LLVM::BitcastOp::create(rewriter, loc, valueElemTy, ret);
        } else {
          ret = b.extract_val(valueElemTy, cmpxchg, 0);
        }

        LLVM::BrOp::create(rewriter, loc, ret, endBlock);
        rewriter.setInsertionPointToStart(endBlock);
        resultVals[i] = endBlock->getArgument(0);
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, casCmp, casVal, successOrdering,
            failureOrdering, scopeStr);

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded;
          if (valueElemIntTy) {
            newLoaded = b.extract_val(valueElemIntTy, cmpxchg, 0);
            newLoaded =
                LLVM::BitcastOp::create(rewriter, loc, valueElemTy, newLoaded);
          } else {
            newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          }
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        LLVM::BrOp::create(rewriter, loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        GCNBuilder BuilderMemfenceLDS;
        BuilderMemfenceLDS.create("s_waitcnt lgkmcnt(0)")->operator()();
        BuilderMemfenceLDS.launch(rewriter, loc, void_ty(ctx));
        b.barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

bool supportsGlobalAtomicF16PackedAndDpp(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  default:
    break;
  }
  return false;
}

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const HCU::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getHCUGPUMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    auto emitter =
        LLVM::HCU::AtomicRMWEmitter(targetInfo, *binOp, *memOrder, *scopeStr);

    Value val = op.getVal();
    Value ptr = op.getPtr();
    Value opResult = op.getResult();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    int numElems = 1;
    // In the case of unpaired f16 elements utilize dpp instructions to
    // accelerate atomics. Here is an algorithm of lowering
    // tt::atomicRmwOp(%ptr, %val, %mask):
    // 0. Group thread by pairs. Master thread is (tid % 2 == 0);
    // 1. All the threads send %val to (tid - 1) thread via dppUpdateOp shl, so
    //    all the masters receive value from secondary threads;
    // 2. Take into account parity in the %mask value, build control flow
    //    structures according to it;
    // 3. Generate llvm::atomicRmwOp in the threads enabled by %mask value;
    // 4. All the threads send result of generated operation to (tid + 1) thread
    //    via dppUpdateOp shl, so all secondary thread also receive their
    //    result.
    //
    // This approach enables us to use half the active threads committing atomic
    // requests to avoid generating of code providing unified access to f16
    // element and reduce contention.
    bool applyPackingF16 = false;
    auto vec = getVectorSize(ptr, axisAnalysisPass);
    if (llMask) {
      vec = std::min<unsigned>(vec, getMaskAlignment(op.getMask()));
    }

    // CDNA3/CDNA4 arch allows to accelerate its atomics with LDS reduction
    // algorithm, which is only applicable for atomics with no return. Otherwise
    // we have to deal with an additional overhead.
    bool enableIntraWaveReduce =
        llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily()) &&
        tensorTy && opResult.use_empty();

    // TODO: support data types less than 32 bits
    enableIntraWaveReduce &= valueElemTy.getIntOrFloatBitWidth() >= 32;

    if (tensorTy) {
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      // Force F16 packing in the case it's not coming in as packed, but the
      // ISA can support packed atomic instructions.
      applyPackingF16 =
          supportsGlobalAtomicF16PackedAndDpp(targetInfo.getISAFamily()) &&
          vec == 1 && isF16Ty && atomicRmwAttr == RMWOp::FADD &&
          !enableIntraWaveReduce;
      numElems = tensorTy.getNumElements();

      auto threadOrder = getThreadOrder(tensorTy);
      auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
      if (axisInfo) {
        unsigned contigWithinLanes =
            axisInfo->getContiguity(threadOrder.front());
        enableIntraWaveReduce &= contigWithinLanes == 1;
      } else {
        enableIntraWaveReduce = false;
      }
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto tid = getThreadId(rewriter, loc);

    bool needLdsStaging = !tensorTy && !opResult.use_empty();
    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset") && needLdsStaging
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      if (applyPackingF16) {
        resultVals[i] = emitter.emitPairedAtomicForEvenTID(
            rewriter, ptrElements[i], valElements[i], rmwMask);
      } else {
        Value valElement;
        if (vec == 1) {
          valElement = valElements[i];
        } else {
          Value vecVal = b.undef(vecTy);
          for (size_t ii = 0; ii < vec; ++ii)
            vecVal = b.insert_element(vecTy, vecVal, valElements[i + ii],
                                      b.i32_val(ii));
          valElement = vecVal;
        }

        // If we have a single tl.atomic_rmw that is lowered into multiple
        // llvm.atomic_rmw, and we set the ordering for each to aql_rel (the
        // default if no sem value is explicitly set in the DSL level
        // tl.atomic_add. The llvm backend will insert extra buffer invalidates
        // and L2 write backs causing a perforance degration. To avoid this we
        // set the ordering to release for the first, acquire for the last, and
        // relaxed for anything in between so that only a single set of
        // buffer_inv and buffer_wbl2 instructions are inserted by the backend
        // for any "cluster" of atomic ops.
        if ((vec > 1 || elemsPerThread > 1) &&
            op.getSem() == MemSemantic::ACQUIRE_RELEASE) {
          if (i == 0) {
            // First
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::release);
          } else if (i == elemsPerThread - vec) {
            // Last
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::acquire);
          } else {
            // Middle
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::monotonic);
          }
        }

        Value retVal =
            emitter.emitAtomicRMW(rewriter, ptrElements[i], valElement, rmwMask,
                                  atomicSharedMemBase, enableIntraWaveReduce);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        } else {
          if (!atomicSharedMemBase.has_value()) {
            rewriter.eraseOp(op);
            return success();
          }
          Value atomPtr = *atomicSharedMemBase;
          b.barrier();
          Value ret = b.load(valueElemTy, atomPtr);

          rewriter.replaceOp(op, {ret});
          return success();
        }
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<hcugpu::AsyncWaitOp> {
  AsyncWaitOpConversion(LLVMTypeConverter &converter,
                        const HCU::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(hcugpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4: {
      // global.load.lds uses vmcnt to synchronize
      // The rocdl op stores all available counters in a single int32 value (v).
      // The vmcnt (6 bits) is split into a lower 3:0 and higher 5:4 parts.
      // The lower part is stored in bits 3:0 of v and the higher part in bits
      // 15:14. We have to set all other bits in v to 1 to signal we are not
      // interested in those.

      // Clamp vmcnt to 6bits; a lower vmcnt will produce a conservative wait
      unsigned vmCnt = std::min(63u, op.getNumInst());

      // Extract low and high bits and combine while setting all other bits to 1
      unsigned lowBits = vmCnt & 0xF;
      unsigned highBits = vmCnt >> 4 << 14;
      unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set

      // HCU extend: bit 7 to enable backend hcu-update-wait-by-reverse-search
      // optimize. see http://172.20.48.13/browse/DCUSW-3577 to know more
      // details.
      unsigned enableOptWaitCntBits = 0x1 << 7;
      unsigned waitValue =
          lowBits | highBits | otherCnts | enableOptWaitCntBits;

      ROCDL::SWaitcntOp::create(rewriter, loc, waitValue);
      break;
    }
    case ISAFamily::GFX1250: {
      // Clamp asyncCnt to 6bits(hw imit); lower means conservative
      unsigned asyncCnt = std::min(63u, op.getNumInst());
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                      "llvm.amdgcn.s.wait.asynccnt", {},
                                      {b.i16_val(asyncCnt)});
      break;
    }
    default:
      return rewriter.notifyMatchFailure(
          op, "Only supported on CDNA target architecture");
    }

    // Drop the result AsyncToken
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const HCU::TargetInfo &targetInfo;
};

struct AsyncTDMWaitConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::AsyncTDMWait> {
  AsyncTDMWaitConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::hcugpu::AsyncTDMWait op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::createLLVMIntrinsicCallOp(rewriter, loc,
                                    "llvm.amdgcn.s.wait.tensorcnt", {},
                                    {b.i16_val(op.getNum())});
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Drop the result AsyncToken
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncCopyMbarrierArriveOpConversion
    : public ConvertOpToLLVMPattern<triton::hcugpu::AsyncCopyMbarrierArriveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::hcugpu::AsyncCopyMbarrierArriveOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(),
        typeConverter->convertType(op.getBarrier().getType().getElementType()),
        rewriter);
    LLVM::createLLVMIntrinsicCallOp(
        rewriter, loc, "llvm.amdgcn.ds.atomic.async.barrier.arrive.b64",
        void_ty(getContext()), smemObj.getBase());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::triton::HCU {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns
      .add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
           StoreOpConversion, BufferLoadOpConversion,
           BufferLoadToLocalOpConversion, BufferStoreOpConversion,
           BufferAtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion,
           BufferAtomicCASOpConversion, AsyncTDMCopyGlobalToLocalOpConversion,
           AsyncTDMCopyLocalToGlobalOpConversion>(typeConverter, targetInfo,
                                                  axisInfoAnalysis, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncTDMWaitConversion>(typeConverter, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
  patterns.add<AsyncCopyMbarrierArriveOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::HCU
