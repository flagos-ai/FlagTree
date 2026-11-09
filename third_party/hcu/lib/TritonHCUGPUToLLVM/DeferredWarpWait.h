#ifndef TRITON_HCU_DEFERRED_WARP_WAIT_H
#define TRITON_HCU_DEFERRED_WARP_WAIT_H

#include "WarpSpecializeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::HCU {
namespace deferred_wait {

// Trace memory events, not a particular producer/consumer protocol. A full
// wait may cross one CTA rendezvous only if no role can observe any of its
// pending writes before the following rendezvous. No memory event is moved.
class WindowTrace {
  struct Window {
    Value root;
    SmallVector<int64_t> begin, end;
    SmallVector<unsigned> axes;
  };
  struct Event {
    unsigned phase;
    Window window;
  };
  struct Wait {
    Operation *op;
    unsigned phase;
    SmallVector<Window> pending;
    bool globalWrite;
  };
  gpu::WarpSpecializeOp ws;
  unsigned budget = 1 << 20;
  DenseMap<Value, APInt> environment;
  DenseSet<Value> closed;
  SmallVector<Event> events;
  SmallVector<Wait> waits;
  SmallVector<unsigned> phaseCounts;
  DenseSet<unsigned> globalWritePhases;

  bool spend() { return budget && --budget; }

  std::optional<APInt> integer(Value value, unsigned depth = 0) {
    if (!spend() || depth == 64 || !value.getType().isIntOrIndex())
      return std::nullopt;
    value = resolveWarpSpecializeCapture(value);
    if (auto it = environment.find(value); it != environment.end())
      return it->second;
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)))
      return constant;
    Operation *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    auto lhs = op->getNumOperands() ? integer(op->getOperand(0), depth + 1)
                                    : std::nullopt;
    if (!lhs)
      return std::nullopt;
    if (isa<arith::ExtSIOp>(op))
      return lhs->sext(cast<IntegerType>(value.getType()).getWidth());
    if (isa<arith::ExtUIOp>(op))
      return lhs->zext(cast<IntegerType>(value.getType()).getWidth());
    if (isa<arith::TruncIOp>(op))
      return lhs->trunc(cast<IntegerType>(value.getType()).getWidth());
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return integer(lhs->isZero() ? select.getFalseValue()
                                  : select.getTrueValue(), depth + 1);
    if (op->getNumOperands() != 2)
      return std::nullopt;
    auto rhs = integer(op->getOperand(1), depth + 1);
    if (!rhs || lhs->getBitWidth() != rhs->getBitWidth())
      return std::nullopt;
    if (isa<arith::AddIOp>(op)) return *lhs + *rhs;
    if (isa<arith::SubIOp>(op)) return *lhs - *rhs;
    if (isa<arith::MulIOp>(op)) return *lhs * *rhs;
    if (isa<arith::AndIOp>(op)) return *lhs & *rhs;
    if (isa<arith::OrIOp>(op)) return *lhs | *rhs;
    if (isa<arith::XOrIOp>(op)) return *lhs ^ *rhs;
    if (isa<arith::RemUIOp>(op) && !rhs->isZero()) return lhs->urem(*rhs);
    if (isa<arith::DivUIOp>(op) && !rhs->isZero()) return lhs->udiv(*rhs);
    if (isa<arith::RemSIOp, arith::DivSIOp>(op) && !rhs->isZero() &&
        !(lhs->isMinSignedValue() && rhs->isAllOnes()))
      return isa<arith::RemSIOp>(op) ? lhs->srem(*rhs) : lhs->sdiv(*rhs);
    if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(op) &&
        rhs->ult(lhs->getBitWidth())) {
      unsigned n = rhs->getZExtValue();
      return isa<arith::ShLIOp>(op) ? lhs->shl(n) :
             isa<arith::ShRUIOp>(op) ? lhs->lshr(n) : lhs->ashr(n);
    }
    return std::nullopt;
  }

  std::optional<Window> window(Value value, unsigned depth = 0) {
    if (!spend() || depth == 64)
      return std::nullopt;
    value = resolveWarpSpecializeCapture(value);
    if (auto alloc = value.getDefiningOp<gpu::LocalAllocOp>()) {
      if (ws->isAncestor(alloc))
        return std::nullopt;
      auto type = cast<gpu::MemDescType>(value.getType());
      // These shared encodings are injective over a root allocation. Other
      // encodings need their own physical-alias proof before being admitted.
      if (!isa<gpu::SwizzledSharedEncodingAttr>(type.getEncoding()) &&
          !(isa<gpu::SharedLinearEncodingAttr>(type.getEncoding()) &&
            gpu::toLinearLayout(type).isInvertible()))
        return std::nullopt;
      Window out{value, {}, {}, {}};
      for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
        if (extent <= 0) return std::nullopt;
        out.begin.push_back(0);
        out.end.push_back(extent);
        out.axes.push_back(axis);
      }
      return out;
    }
    auto index = value.getDefiningOp<gpu::MemDescIndexOp>();
    auto slice = value.getDefiningOp<gpu::MemDescSubsliceOp>();
    if (!index && !slice)
      return std::nullopt;
    auto source = window(index ? index.getSrc() : slice.getSrc(), depth + 1);
    if (!source) return std::nullopt;
    if (index) {
      auto n = integer(index.getIndex());
      if (!n || !n->isSignedIntN(64) || source->axes.empty())
        return std::nullopt;
      unsigned axis = source->axes.front();
      int64_t offset = n->getSExtValue();
      if (offset < 0 || offset >= source->end[axis] - source->begin[axis])
        return std::nullopt;
      source->begin[axis] += offset;
      source->end[axis] = source->begin[axis] + 1;
      source->axes.erase(source->axes.begin());
    } else {
      auto shape = slice.getType().getShape();
      if (source->axes.size() != shape.size()) return std::nullopt;
      for (auto [i, offset] : llvm::enumerate(slice.getOffsets())) {
        unsigned axis = source->axes[i];
        if (offset < 0 || shape[i] <= 0 ||
            offset > source->end[axis] - source->begin[axis] - shape[i])
          return std::nullopt;
        source->begin[axis] += offset;
        source->end[axis] = source->begin[axis] + shape[i];
      }
    }
    return source;
  }

  // Reject descriptor escape/alias anywhere, including before the WS entry.
  // Being type-correct or having a pure view trait is not an alias proof.
  bool close(Value value) {
    if (!spend()) return false;
    if (!closed.insert(value).second) return true;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (op == ws.getOperation()) {
        for (Region *region : ws.getPartitionRegions())
          if (!close(region->getArgument(use.getOperandNumber()))) return false;
      } else if (isa<gpu::MemDescIndexOp, gpu::MemDescSubsliceOp>(op)) {
        if (!close(op->getResult(0))) return false;
      } else if (isa<gpu::LocalLoadOp, gpu::LocalStoreOp,
                     gpu::AsyncCopyGlobalToLocalOp,
                     hcugpu::BufferLoadToLocalOp>(op)) {
        if (!ws->isAncestor(op)) return false;
      } else if (isa<gpu::LocalDeallocOp>(op)) {
        if (op->getBlock() != ws->getBlock() || !ws->isBeforeInBlock(op))
          return false;
      } else {
        return false;
      }
    }
    return true;
  }

  bool block(Block &body, unsigned &phase, SmallVector<Window> &pending,
             bool &globalWrite) {
    for (Operation &op : body.without_terminator()) {
      if (!spend()) return false;
      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        auto lb = integer(loop.getLowerBound()), ub = integer(loop.getUpperBound()),
             step = integer(loop.getStep());
        if (!lb || !ub || !step || !lb->isSignedIntN(64) ||
            !ub->isSignedIntN(64) || !step->isSignedIntN(64) ||
            step->getSExtValue() <= 0)
          return false;
        SmallVector<std::optional<APInt>> carried;
        for (Value init : loop.getInitArgs()) carried.push_back(integer(init));
        int64_t i = lb->getSExtValue();
        while (i < ub->getSExtValue()) {
          environment[loop.getInductionVar()] = APInt(lb->getBitWidth(), i, true);
          for (auto [arg, number] : llvm::zip(loop.getRegionIterArgs(), carried)) {
            environment.erase(arg);
            if (number) environment[arg] = *number;
          }
          if (!block(*loop.getBody(), phase, pending, globalWrite)) return false;
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          carried.clear();
          for (Value next : yield.getOperands()) carried.push_back(integer(next));
          if (llvm::AddOverflow(i, step->getSExtValue(), i) ||
              !APInt(64, i, true).isSignedIntN(lb->getBitWidth()))
            return false;
        }
        for (auto [result, number] : llvm::zip(loop.getResults(), carried)) {
          environment.erase(result);
          if (number) environment[result] = *number;
        }
        environment.erase(loop.getInductionVar());
        for (Value arg : loop.getRegionIterArgs()) environment.erase(arg);
        continue;
      }
      if (auto bar = dyn_cast<gpu::LocalBarrierOp>(&op)) {
        if (bar->hasAttr("hcu.cta_scope")) {
          if (!isa<UnitAttr>(bar->getAttr("hcu.cta_scope"))) return false;
          ++phase;
        }
        continue;
      }
      if (auto wait = dyn_cast<hcugpu::AsyncWaitOp>(&op)) {
        if (wait.getNumInst() == 0) {
          Operation *first = wait->getNextNode();
          Operation *second = first ? first->getNextNode() : nullptr;
          auto cta = [](Operation *p) {
            return isa_and_nonnull<gpu::LocalBarrierOp>(p) &&
                   isa_and_nonnull<UnitAttr>(p->getAttr("hcu.cta_scope"));
          };
          if (wait.getResult().use_empty() && cta(first) && cta(second) &&
              !pending.empty())
            waits.push_back({wait, phase, pending, globalWrite});
          pending.clear(); // vmcnt0 completes every issued copy, not one token.
          globalWrite = false;
        }
        continue;
      }
      Value memory;
      bool asynchronous = false;
      if (auto read = dyn_cast<gpu::LocalLoadOp>(&op)) memory = read.getSrc();
      else if (auto store = dyn_cast<gpu::LocalStoreOp>(&op)) memory = store.getDst();
      else if (auto copy = dyn_cast<gpu::AsyncCopyGlobalToLocalOp>(&op)) {
        auto pointer = dyn_cast<PointerType>(getElementTypeOrSelf(copy.getSrc()));
        if (!pointer || pointer.getAddressSpace() != 1) return false;
        memory = copy.getResult(); asynchronous = true;
      } else if (auto copy = dyn_cast<hcugpu::BufferLoadToLocalOp>(&op)) {
        if (cast<PointerType>(copy.getPtr().getType()).getAddressSpace() != 1)
          return false;
        memory = copy.getDest(); asynchronous = true;
      }
      if (memory) {
        auto region = window(memory);
        if (!region || !close(region->root)) return false;
        events.push_back({phase, *region});
        if (asynchronous) pending.push_back(*region);
        continue;
      }
      // These views are resolved at each access under its current loop SSA
      // environment. Their closed use graph was checked independently.
      if (isa<gpu::MemDescIndexOp, gpu::MemDescSubsliceOp,
              gpu::AsyncCommitGroupOp, hcugpu::AccumulatorFenceOp>(op))
        continue;
      // Pure tensor arithmetic may contain pure control flow, but cannot
      // hide a descriptor alias or a shared access from this trace.
      if (llvm::any_of(op.getOperandTypes(), [](Type t) { return isa<gpu::MemDescType>(t); }) ||
          llvm::any_of(op.getResultTypes(), [](Type t) { return isa<gpu::MemDescType>(t); }))
        return false;
      if (isPure(&op)) continue;
      auto effects = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!effects || op.getNumRegions()) return false;
      // tt.load's effect interface labels reads GlobalMemory even for raw
      // shared/generic pointers. Such accesses lack a closed memdesc alias
      // proof, so do not mistake them for disjoint AS1 global accesses.
      for (Type type : op.getOperandTypes())
        if (auto ptr = dyn_cast<PointerType>(getElementTypeOrSelf(type));
            ptr && ptr.getAddressSpace() != 1)
          return false;
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto &effect : instances) {
        if (effect.getResource() != GlobalMemory::get()) return false;
        if (!isa<MemoryEffects::Read>(effect.getEffect())) {
          globalWrite = true;
          globalWritePhases.insert(phase);
        }
      }
    }
    return true;
  }

  static bool overlaps(const Window &a, const Window &b) {
    if (a.root != b.root) return false;
    for (unsigned i = 0; i < a.begin.size(); ++i)
      if (a.end[i] <= b.begin[i] || b.end[i] <= a.begin[i]) return false;
    return true;
  }

public:
  explicit WindowTrace(gpu::WarpSpecializeOp ws) : ws(ws) {}
  void run() {
    SmallVector<Region *> roles{&ws.getDefaultRegion()};
    llvm::append_range(roles, ws.getPartitionRegions());
    for (Region *role : roles) {
      if (!llvm::hasSingleElement(*role)) return;
      unsigned phase = 0;
      SmallVector<Window> pending;
      bool writes = false;
      if (!block(role->front(), phase, pending, writes)) return;
      phaseCounts.push_back(phase);
      environment.clear();
    }
    if (!llvm::all_equal(phaseCounts)) return;
    DenseMap<Operation *, bool> safe;
    for (const Wait &wait : waits) {
      auto [it, inserted] = safe.try_emplace(wait.op, true);
      it->second &= !wait.globalWrite;
      // Another role may overwrite a global source after the rendezvous
      // which originally followed vmcnt0. Without a global alias proof,
      // preserve completion before every global write in the crossed phase.
      it->second &= !globalWritePhases.contains(wait.phase + 1);
      for (const Event &event : events) {
        if (!spend()) return;
        if (event.phase != wait.phase + 1) continue;
        if (llvm::any_of(wait.pending, [&](const Window &pending) {
              return overlaps(event.window, pending);
            })) it->second = false;
      }
    }
    for (auto [wait, legal] : safe)
      if (legal) wait->moveAfter(wait->getNextNode());
  }
};
} // namespace deferred_wait
} // namespace mlir::triton::HCU
#endif
