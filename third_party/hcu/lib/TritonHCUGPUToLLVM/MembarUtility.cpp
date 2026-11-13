#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "triton/Analysis/Allocation.h"
#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::HCU {
namespace {
namespace ttg = mlir::triton::gpu;

bool isI32Constant(Value value, int64_t expected) {
  llvm::APInt integer;
  return value.getType().isInteger(32) &&
         matchPattern(value, m_ConstantInt(&integer)) &&
         integer.getSExtValue() == expected;
}

bool isIncrementByOne(Value value, Value previous) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  return add && add.getLhs() == previous && isI32Constant(add.getRhs(), 1);
}

bool isModuloTwoCounter(Value value, Value previous) {
  auto select = value.getDefiningOp<arith::SelectOp>();
  auto cmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>()
                    : arith::CmpIOp{};
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
         cmp.getLhs() == select.getTrueValue() &&
         isI32Constant(cmp.getRhs(), 2) &&
         isI32Constant(select.getFalseValue(), 0) &&
         isIncrementByOne(select.getTrueValue(), previous);
}

bool isOrdinaryGlobalLoad(Operation *op) {
  if (!isa<triton::LoadOp, hcugpu::BufferLoadOp>(op))
    return false;
  // Reject local-pointer loads even when they use an ordinary load operation.
  for (Value operand : op->getOperands()) {
    Type type = operand.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    if (auto pointer = dyn_cast<triton::PointerType>(type))
      if (pointer.getAddressSpace() != 1)
        return false;
  }
  return op->getNumResults() == 1;
}

// Explicitly exclude pure operations whose lowering may use LDS scratch
// (convert_layout/reduce/scan), inline assembly, and all unknown operations.
bool isRegisterOnlyPrefetchOp(Operation *op) {
  return op->getName().getDialectNamespace() == "arith" ||
         isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
             triton::AddPtrOp, triton::MakeRangeOp>(op);
}

bool hasSingleCTALayout(Attribute encoding) {
  auto layout = dyn_cast<ttg::LayoutEncodingTrait>(encoding);
  return layout &&
         llvm::all_of(layout.getCTALayout().getCTAsPerCGA(),
                      [](unsigned count) { return count == 1; }) &&
         llvm::all_of(layout.getCTALayout().getCTASplitNum(),
                      [](unsigned count) { return count == 1; });
}

// A dynamic loop bound is allowed only when its scalar integer dependency
// tree is CTA-uniform. Never infer uniformity for a memory load or unknown op.
bool isCTAUniformInteger(Value value, Block *entry,
                         llvm::DenseMap<Value, bool> &memo) {
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  memo[value] = false;
  if (!isa<IntegerType>(value.getType()))
    return false;
  if (auto argument = dyn_cast<BlockArgument>(value))
    return memo[value] = argument.getOwner() == entry;
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != entry || op->getNumRegions())
    return false;
  if (isa<arith::ConstantOp, triton::GetProgramIdOp,
          triton::GetNumProgramsOp>(op))
    return memo[value] = true;
  if (op->getName().getDialectNamespace() != "arith")
    return false;
  bool uniform = llvm::all_of(op->getOperands(), [&](Value operand) {
    return isCTAUniformInteger(operand, entry, memo);
  });
  return memo[value] = uniform;
}

// Every descriptor derived from this root must be the one complete tile in
// entry/body/exit, used exactly once by a store and once by a load. In
// particular, no descriptor may escape, gain an alias, or feed an unknown op.
bool hasOnlyTwoSlotTileViews(ttg::LocalAllocOp root, Block *entry, Block *body,
                            Block *exit, Value counter, Value initial,
                            bool registerDrain = false) {
  DominanceInfo dominance(root->getParentOp());
  unsigned deallocs = 0;
  llvm::SmallPtrSet<Block *, 4> seen;
  for (Operation *user : root.getResult().getUsers()) {
    if (isa<ttg::LocalDeallocOp>(user)) {
      if ((!registerDrain && user->getBlock() != exit) ||
          (registerDrain && !dominance.dominates(exit, user->getBlock())) ||
          ++deallocs != 1)
        return false;
      continue;
    }
    auto index = dyn_cast<ttg::MemDescIndexOp>(user);
    if (!index || !seen.insert(user->getBlock()).second)
      return false;
    Block *block = user->getBlock();
    if (block == entry) {
      if (index.getIndex() != initial)
        return false;
    } else if (block == body || (!registerDrain && block == exit)) {
      if (!isModuloTwoCounter(index.getIndex(), counter))
        return false;
    } else {
      return false;
    }
    ttg::LocalStoreOp store;
    ttg::LocalLoadOp load;
    for (Operation *access : index.getResult().getUsers()) {
      if (access->getBlock() != block)
        return false;
      if (auto candidate = dyn_cast<ttg::LocalStoreOp>(access)) {
        if (store || candidate.getDst() != index.getResult())
          return false;
        store = candidate;
      } else if (auto candidate = dyn_cast<ttg::LocalLoadOp>(access)) {
        if (load || candidate.getSrc() != index.getResult() ||
            candidate.getToken())
          return false;
        load = candidate;
      } else {
        return false;
      }
    }
    if (!store || !load || !store->isBeforeInBlock(load))
      return false;
  }
  return seen.size() == (registerDrain ? 2 : 3) &&
         (!registerDrain || deallocs == 1);
}

// Prove only a same-iteration WAR between full planes of one ring
// allocation, including a shifted descriptor queue. The existing head publication barrier must remain:
// next iteration reuses the slot read by the previous iteration.
bool filterAlternatingSlotWAR(Operation *before, Operation *after) {
  auto read = dyn_cast<ttg::LocalLoadOp>(before);
  auto copy = dyn_cast<hcugpu::BufferLoadToLocalOp>(after);
  if (!read || !copy || !read.getToken() || read->getBlock() != copy->getBlock() ||
      !read->isBeforeInBlock(copy))
    return false;
  auto mod = before->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  auto ctas = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  if (!target || target.getValue() != "hip:gfx936" || !ctas || ctas.getInt() != 1)
    return false;
  auto desc = dyn_cast<BlockArgument>(read.getSrc());
  auto view = copy.getDest().getDefiningOp<ttg::MemDescIndexOp>();
  auto root = view ? view.getSrc().getDefiningOp<ttg::LocalAllocOp>()
                   : ttg::LocalAllocOp{};
  auto wrap = view ? view.getIndex().getDefiningOp<arith::SelectOp>()
                   : arith::SelectOp{};
  auto increment = wrap ? wrap.getTrueValue().getDefiningOp<arith::AddIOp>()
                        : arith::AddIOp{};
  auto counter = increment ? dyn_cast<BlockArgument>(increment.getLhs())
                           : BlockArgument{};
  if (!desc || !view || !root || !wrap || !counter ||
      desc.getOwner() != counter.getOwner())
    return false;
  auto srcTy = root.getType(), dstTy = view.getType();
  if (srcTy.getRank() != 3 ||
      (srcTy.getShape()[0] != 2 && srcTy.getShape()[0] != 4))
    return false;
  const int64_t slots = srcTy.getShape()[0];
  auto wrapCmp = wrap.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!wrapCmp || wrapCmp.getPredicate() != arith::CmpIPredicate::slt ||
      wrapCmp.getLhs() != wrap.getTrueValue() ||
      !isI32Constant(wrapCmp.getRhs(), slots) ||
      !isI32Constant(wrap.getFalseValue(), 0) ||
      !isIncrementByOne(wrap.getTrueValue(), counter))
    return false;
  Block *header = desc.getOwner(), *body = before->getBlock();
  auto func = dyn_cast<triton::FuncOp>(header->getParentOp());
  if (!func || func.getBody().empty())
    return false;
  Block *entry = &func.getBody().front();
  auto pre = dyn_cast<cf::BranchOp>(entry->getTerminator());
  auto back = dyn_cast<cf::BranchOp>(body->getTerminator());
  auto cond = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  llvm::SmallPtrSet<Block *, 2> preds(header->pred_begin(), header->pred_end());
  if (root->getBlock() != entry || root.getSrc() ||
      !root.isSharedMemoryAlloc() || !pre || !back || !cond ||
      pre.getDest() != header || back.getDest() != header ||
      cond.getTrueDest() != body || cond.getFalseDest() == body ||
      !cond.getTrueDestOperands().empty() ||
      !llvm::hasSingleElement(body->getPredecessors()) ||
      *body->pred_begin() != header || body->getNumArguments() ||
      preds.size() != 2 || !preds.contains(entry) || !preds.contains(body) ||
      pre.getDestOperands().size() != header->getNumArguments() ||
      back.getDestOperands().size() != header->getNumArguments() ||
      back.getDestOperands()[counter.getArgNumber()] != view.getIndex())
    return false;
  auto constantSlot = [&](Value v) -> int64_t {
    llvm::APInt value;
    if (!v.getType().isInteger(32) || !matchPattern(v, m_ConstantInt(&value)) ||
        value.isNegative() || value.getZExtValue() >= uint64_t(slots))
      return -1;
    return value.getZExtValue();
  };
  const int64_t initial = constantSlot(pre.getDestOperands()[counter.getArgNumber()]);
  if (initial < 0)
    return false;
  // Induct over the queue independently of block-argument positions. If a
  // descriptor has phase p relative to the current counter, a shifted phi
  // on its backedge must have phase p+1; root[next] is terminal phase zero.
  SmallVector<Value> seen;
  BlockArgument cursor = desc;
  int64_t readPhase = -1, phase = -1;
  while (true) {
    if (cursor.getOwner() != header || cursor.getType() != dstTy ||
        llvm::is_contained(seen, Value(cursor)) || seen.size() >= unsigned(slots - 1))
      return false;
    seen.push_back(cursor);
    auto initView = pre.getDestOperands()[cursor.getArgNumber()]
                        .getDefiningOp<ttg::MemDescIndexOp>();
    int64_t initialSlot = initView ? constantSlot(initView.getIndex()) : -1;
    if (!initView || initView.getSrc() != root.getResult() ||
        initView.getType() != dstTy || initialSlot < 0)
      return false;
    int64_t actualPhase = (initialSlot - initial + slots) % slots;
    if (readPhase < 0)
      readPhase = phase = actualPhase;
    if (actualPhase != phase || readPhase == 1)
      return false;
    Value incoming = back.getDestOperands()[cursor.getArgNumber()];
    if (incoming == view.getResult()) {
      if (phase != 0)
        return false;
      break;
    }
    cursor = dyn_cast<BlockArgument>(incoming);
    if (!cursor)
      return false;
    phase = (phase + 1) % slots;
  }
  // Check CTA-uniform loop control, independently of argument positions or
  // a constant K. The ring counter itself is uniform by the induction above.
  auto cmp = dyn_cast<arith::CmpIOp>(header->front());
  auto iv = cmp ? dyn_cast<BlockArgument>(cmp.getLhs()) : BlockArgument{};
  if (!cmp || !iv || iv.getOwner() != header ||
      header->getOperations().size() != 2 || cond.getCondition() != cmp.getResult())
    return false;
  llvm::DenseMap<Value, bool> uniform;
  auto step = back.getDestOperands()[iv.getArgNumber()].getDefiningOp<arith::AddIOp>();
  if (!isCTAUniformInteger(cmp.getRhs(), entry, uniform) ||
      !isCTAUniformInteger(pre.getDestOperands()[iv.getArgNumber()], entry, uniform) ||
      !step || step.getLhs() != iv ||
      !isCTAUniformInteger(step.getRhs(), entry, uniform))
    return false;
  auto wait = read.getToken().getDefiningOp<hcugpu::AsyncWaitOp>();
  auto barrier = wait ? dyn_cast_or_null<ttg::LocalBarrierOp>(wait->getNextNode())
                      : ttg::LocalBarrierOp{};
  if (!wait || wait->getBlock() != body ||
      (slots == 2 && wait.getNumInst() != 0) ||
      !barrier || !barrier->isBeforeInBlock(read))
    return false;
  for (Operation &op : *body) {
    if (&op == wait.getOperation()) break;
    if (!isRegisterOnlyPrefetchOp(&op)) return false;
  }
  if (dstTy.getRank() != 2 ||
      !isa<ttg::SwizzledSharedEncodingAttr>(srcTy.getEncoding()) ||
      !hasSingleCTALayout(srcTy.getEncoding()) ||
      srcTy.getEncoding() != dstTy.getEncoding() ||
      srcTy.getElementType() != dstTy.getElementType() ||
      srcTy.getMemorySpace() != dstTy.getMemorySpace() ||
      srcTy.getMutableMemory() != dstTy.getMutableMemory() ||
      read.getSrc().getType() != dstTy ||
      srcTy.getShape() != srcTy.getAllocShape() ||
      dstTy.getShape() != dstTy.getAllocShape() ||
      srcTy.getShape().drop_front() != dstTy.getShape() ||
      ArrayRef<int64_t>(ttg::getAllocationShapePerCTA(srcTy)) != srcTy.getShape() ||
      ArrayRef<int64_t>(ttg::getAllocationShapePerCTA(dstTy)) != dstTy.getShape() ||
      !(srcTy.getElementType().isF16() || srcTy.getElementType().isBF16()) ||
      !llvm::all_of(dstTy.getShape(), [](int64_t d) { return d > 0; }))
    return false;
  // Same physical full-plane stride as common getFullSlotInterval and
  // MemDescIndex lowering. read=(counter+readPhase)%slots, write=counter+1.
  // Only this same-iteration WAR pair is disjoint. The head LDS barrier above
  // still completes every prior-iteration read before slot reuse. Partial VM
  // waits are unchanged: publication RAW and every other effect retain their
  // original handling. No async token is treated as a reverse-WAR exemption.
  return true;
}

// Ignore only a read-after-write dependency covered by an AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::hcugpu::MatrixLoadToLocalOp,
                     triton::hcugpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };

  // op1 precedes op2 in the membar analysis. Waiting for a copy before its
  // local read does not protect that read from the next iteration's copy.
  // Filtering the reverse (WAR) dependency lets a fast wave overwrite a
  // single-buffer tile while another wave is still reading it.
  return isAsyncLoad(op1) && isLocalLoadWithAsyncWaitToken(op2);
};

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::hcugpu::InitBarrierOp,
                     triton::hcugpu::ArriveBarrierOp,
                     triton::hcugpu::AsyncCopyMbarrierArriveOp,
                     triton::hcugpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareTwoSlotPrefetchWAR(ModuleOp mod) {
  auto mode = mod->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  auto ctas = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  auto wave = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  auto warps = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto buffers = mod->getAttrOfType<IntegerAttr>("ttg.hcu.sync_lds_buffers");
  bool registerDrain = buffers && buffers.getInt() == 2 && mode &&
                       mode.getInt() == 1;
  if (!mode || (mode.getInt() != 4 && !registerDrain) ||
      !target || target.getValue() != "hip:gfx936" ||
      !ctas || ctas.getInt() != 1 || !wave || wave.getInt() != 64 || !warps ||
      (warps.getInt() != 4 && warps.getInt() != 8))
    return {};
  auto functions = llvm::to_vector(mod.getOps<triton::FuncOp>());
  if (functions.size() != 1)
    return {};
  Region &region = functions.front().getBody();
  if ((!registerDrain && region.getBlocks().size() != 4) ||
      region.getBlocks().size() < 4)
    return {};
  auto block = region.begin();
  Block *entry = &*block++;
  Block *header = &*block++;
  Block *body = &*block++;
  Block *exit = &*block;
  auto pre = dyn_cast<cf::BranchOp>(entry->getTerminator());
  auto condition = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  auto back = dyn_cast<cf::BranchOp>(body->getTerminator());
  if (!entry->hasNoPredecessors() || !pre || pre.getDest() != header ||
      !condition || condition.getTrueDest() != body ||
      condition.getFalseDest() != exit || !back || back.getDest() != header ||
      !llvm::hasSingleElement(body->getPredecessors()) ||
      *body->pred_begin() != header ||
      !llvm::hasSingleElement(exit->getPredecessors()) ||
      *exit->pred_begin() != header ||
      (!registerDrain && !isa<triton::ReturnOp>(exit->getTerminator())) ||
      header->getNumArguments() != 7 || body->getNumArguments() != 0 ||
      exit->getNumArguments() != 0 || pre.getDestOperands().size() != 7 ||
      back.getDestOperands().size() != 7 ||
      !condition.getTrueDestOperands().empty() ||
      !condition.getFalseDestOperands().empty())
    return {};
  llvm::SmallPtrSet<Block *, 2> predecessors(header->pred_begin(),
                                             header->pred_end());
  if (predecessors.size() != 2 || !predecessors.contains(entry) ||
      !predecessors.contains(body) || header->getOperations().size() != 2)
    return {};
  auto cmp = dyn_cast<arith::CmpIOp>(header->front());
  llvm::APInt bound;
  if (!cmp || condition.getCondition() != cmp.getResult() ||
      cmp.getPredicate() != arith::CmpIPredicate::slt ||
      cmp.getLhs() != header->getArgument(0) ||
      !cmp.getRhs().getType().isInteger(32) ||
      (!registerDrain &&
       (!matchPattern(cmp.getRhs(), m_ConstantInt(&bound)) || bound.isNegative())) ||
      !isI32Constant(pre.getDestOperands()[0], 0))
    return {};
  llvm::DenseMap<Value, bool> uniform;
  if (registerDrain && !isCTAUniformInteger(cmp.getRhs(), entry, uniform))
    return {};
  auto increment = back.getDestOperands()[0].getDefiningOp<arith::AddIOp>();
  llvm::APInt step;
  if (!increment || increment.getLhs() != header->getArgument(0) ||
      !matchPattern(increment.getRhs(), m_ConstantInt(&step)) ||
      !step.isStrictlyPositive() || (!registerDrain && step != 1))
    return {};
  unsigned counterIndex = registerDrain ? 4 : 2;
  Value counter = header->getArgument(counterIndex);
  Value initial = pre.getDestOperands()[counterIndex];
  Value next = back.getDestOperands()[counterIndex];
  if ((!isI32Constant(initial, 0) && !isI32Constant(initial, 1)) ||
      !isModuloTwoCounter(next, counter))
    return {};

  SmallVector<ttg::LocalStoreOp> stores;
  SmallVector<ttg::LocalLoadOp> loads;
  SmallVector<Operation *> globals;
  SmallVector<ttg::MemDescIndexOp> views;
  triton::DotOp dot;
  for (Operation &op : body->without_terminator()) {
    if (op.getNumRegions() != 0)
      return {};
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
      stores.push_back(store);
    else if (auto load = dyn_cast<ttg::LocalLoadOp>(op))
      loads.push_back(load);
    else if (auto index = dyn_cast<ttg::MemDescIndexOp>(op))
      views.push_back(index);
    else if (auto candidate = dyn_cast<triton::DotOp>(op)) {
      if (dot)
        return {};
      dot = candidate;
    } else if (isOrdinaryGlobalLoad(&op))
      globals.push_back(&op);
    else if (!isRegisterOnlyPrefetchOp(&op))
      return {};
  }
  if (stores.size() != 2 || loads.size() != 2 || globals.size() != 2 ||
      views.size() != 2 || !dot || dot.getType().getRank() != 2 ||
      !hasSingleCTALayout(dot.getType().getEncoding()) ||
      dot.getA() != header->getArgument(5) ||
      dot.getB() != header->getArgument(6) ||
      dot.getC() != header->getArgument(1) ||
      back.getDestOperands()[1] != dot.getResult())
    return {};
  for (auto store : stores)
    for (auto load : loads)
      if (!store->isBeforeInBlock(load))
        return {};

  SmallVector<ttg::LocalAllocOp> roots;
  for (unsigned i = 0; i < 2; ++i) {
    auto view = stores[i].getDst().getDefiningOp<ttg::MemDescIndexOp>();
    auto root = view ? view.getSrc().getDefiningOp<ttg::LocalAllocOp>()
                     : ttg::LocalAllocOp{};
    auto operand = dyn_cast<ttg::DotOperandEncodingAttr>(
        loads[i].getType().getEncoding());
    if (!root || root->getBlock() != entry || root.getSrc() ||
        root.getType().getRank() != 3 || root.getType().getShape()[0] != 2 ||
        !isa<ttg::SwizzledSharedEncodingAttr>(root.getType().getEncoding()) ||
        !hasSingleCTALayout(root.getType().getEncoding()) || !operand ||
        operand.getOpIdx() != i ||
        operand.getParent() != dot.getType().getEncoding() ||
        !hasSingleCTALayout(stores[i].getSrc().getType().getEncoding()) ||
        !root.isSharedMemoryAlloc() ||
        !(root.getType().getElementType().isF16() ||
          root.getType().getElementType().isBF16()) ||
        view.getIndex() != next || loads[i].getSrc() != view.getResult() ||
        (registerDrain
             ? stores[i].getSrc() != globals[i]->getResult(0)
             : (stores[i].getSrc() != header->getArgument(3 + i) ||
                back.getDestOperands()[3 + i] != globals[i]->getResult(0))) ||
        globals[i]->getResult(0).getType() != stores[i].getSrc().getType() ||
        back.getDestOperands()[5 + i] != loads[i].getResult() ||
        !hasOnlyTwoSlotTileViews(root, entry, body, exit, counter, initial,
                                registerDrain))
      return {};
    auto initialGlobal = pre.getDestOperands()[3 + i].getDefiningOp();
    auto initialLocal = pre.getDestOperands()[5 + i]
                            .getDefiningOp<ttg::LocalLoadOp>();
    auto initialView = initialLocal
                           ? initialLocal.getSrc()
                                 .getDefiningOp<ttg::MemDescIndexOp>()
                           : ttg::MemDescIndexOp{};
    if ((!registerDrain &&
         (!initialGlobal || initialGlobal->getBlock() != entry ||
          !isOrdinaryGlobalLoad(initialGlobal))) || !initialLocal ||
        initialLocal->getBlock() != entry || !initialView ||
        initialView.getSrc() != root.getResult())
      return {};
    roots.push_back(root);
  }
  if (roots[0] == roots[1])
    return {};
  // Refuse regions/calls/async/assembly anywhere in the function. Extra
  // register-drain/epilogue blocks cannot access the proved roots; every
  // root user was checked above, including deallocation dominated by exit.
  for (Block &bb : region)
    for (Operation &op : bb)
      if (op.getNumRegions() || isa<CallOpInterface>(op) ||
          op.getName().getStringRef().contains("async") ||
          op.getName().getStringRef().contains("asm"))
        return {};

  // All validation precedes mutation. Synchronize prologue reads once, then
  // retain every store->read publication barrier inserted by ordinary membar.
  // For S_j -> B_j -> R_j on slot j%2, R_j -> B_(j+1) -> S_(j+2)
  // protects the next reuse; only the exact main-loop R->S WAR pairs differ
  // physically across the adjacent iterations. Single-slot/other CFGs never
  // reach this point. LocalBarrier lowering waits ds=0 before CTA rendezvous.
  OpBuilder builder(pre);
  ttg::LocalBarrierOp::create(builder, pre.getLoc());
  return {{loads[0].getOperation(), stores[0].getOperation()},
          {loads[1].getOperation(), stores[1].getOperation()}};
}

// Private, read-only proof at the final CF boundary. The fixed pipeline
// queue template deliberately rejects unsupported alias/control-flow forms.
llvm::SmallVector<std::pair<Operation *, Operation *>>
prepareAsyncSlotPrefetchWAR(ModuleOp mod) {
  auto mode = mod->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
  auto enable = mod->getAttrOfType<IntegerAttr>("ttg.hcu.async_slot_war");
  auto target = mod->getAttrOfType<StringAttr>("ttg.target");
  auto ctas = mod->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  auto wave = mod->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  auto warps = mod->getAttrOfType<IntegerAttr>("ttg.num-warps");
  if (!enable || enable.getInt() != 1 || !mode ||
      (mode.getInt() != 6 && mode.getInt() != 7) || !target ||
      target.getValue() != "hip:gfx936" || !ctas || ctas.getInt() != 1 ||
      !wave || wave.getInt() != 64 || !warps ||
      (warps.getInt() != 4 && warps.getInt() != 8))
    return {};
  auto funcs = llvm::to_vector(mod.getOps<triton::FuncOp>());
  if (funcs.size() != 1 || funcs[0].getBody().getBlocks().size() != 4)
    return {};
  auto bb = funcs[0].getBody().begin();
  Block *entry = &*bb++, *header = &*bb++, *body = &*bb++, *exit = &*bb;
  auto pre = dyn_cast<cf::BranchOp>(entry->getTerminator());
  auto cond = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  auto back = dyn_cast<cf::BranchOp>(body->getTerminator());
  if (!entry->hasNoPredecessors() || !pre || pre.getDest() != header ||
      !cond || cond.getTrueDest() != body || cond.getFalseDest() != exit ||
      !back || back.getDest() != header || !isa<triton::ReturnOp>(exit->getTerminator()) ||
      !llvm::hasSingleElement(body->getPredecessors()) || *body->pred_begin() != header ||
      !llvm::hasSingleElement(exit->getPredecessors()) || *exit->pred_begin() != header ||
      body->getNumArguments() || exit->getNumArguments() ||
      !cond.getTrueDestOperands().empty() || !cond.getFalseDestOperands().empty() ||
      header->getOperations().size() != 2 || header->getNumArguments() < 9)
    return {};
  llvm::SmallPtrSet<Block *, 2> preds(header->pred_begin(), header->pred_end());
  auto cmp = dyn_cast<arith::CmpIOp>(header->front());
  llvm::APInt bound;
  if (preds.size() != 2 || !preds.contains(entry) || !preds.contains(body) ||
      !cmp || cond.getCondition() != cmp.getResult() ||
      cmp.getPredicate() != arith::CmpIPredicate::slt || cmp.getLhs() != header->getArgument(0) ||
      !matchPattern(cmp.getRhs(), m_ConstantInt(&bound)) || bound.isNegative() ||
      pre.getDestOperands().size() != header->getNumArguments() ||
      back.getDestOperands().size() != header->getNumArguments() ||
      !isI32Constant(pre.getDestOperands()[0], 0) ||
      !isIncrementByOne(back.getDestOperands()[0], header->getArgument(0)))
    return {};

  SmallVector<ttg::LocalLoadOp> reads;
  SmallVector<hcugpu::BufferLoadToLocalOp> copies;
  triton::DotOp dot;
  hcugpu::AsyncWaitOp wait;
  ttg::LocalBarrierOp publication;
  for (Operation &op : body->without_terminator()) {
    if (op.getNumRegions())
      return {};
    if (auto v = dyn_cast<ttg::LocalLoadOp>(op))
      reads.push_back(v);
    else if (auto v = dyn_cast<hcugpu::BufferLoadToLocalOp>(op))
      copies.push_back(v);
    else if (auto v = dyn_cast<triton::DotOp>(op)) {
      if (dot)
        return {};
      dot = v;
    } else if (auto v = dyn_cast<hcugpu::AsyncWaitOp>(op)) {
      if (wait)
        return {};
      wait = v;
    } else if (auto v = dyn_cast<ttg::LocalBarrierOp>(op)) {
      if (publication)
        return {};
      publication = v;
    } else if (!isa<ttg::MemDescIndexOp, ttg::AsyncCommitGroupOp, ROCDL::SchedBarrier>(op) &&
             !isRegisterOnlyPrefetchOp(&op))
      return {};
  }
  auto mma = dot ? dyn_cast<ttg::AMDMfmaEncodingAttr>(dot.getType().getEncoding())
                 : ttg::AMDMfmaEncodingAttr{};
  if (reads.size() != 2 || copies.size() != 2 || !dot || dot.getType().getRank() != 2 || !mma ||
      mma.getMmacLayout() != ttg::MmacLayout::LEGACY ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      !hasSingleCTALayout(mma) || !wait || !publication ||
      wait->getNextNode() != publication.getOperation() || wait->getNumOperands() != 2)
    return {};
  SmallVector<ttg::LocalAllocOp> roots;
  SmallVector<Value> descriptors;
  SmallVector<Operation *> deallocOps;
  llvm::DenseMap<Value, unsigned> owner;
  llvm::SmallVector<std::pair<Operation *, Operation *>> pairs;
  auto getConstant = [](Value v) -> int64_t {
    llvm::APInt x;
    return v.getType().isInteger(32) && matchPattern(v, m_ConstantInt(&x))
               ? x.getSExtValue() : -1;
  };
  unsigned slots = 0;
  Value counter = header->getArgument(2), next = back.getDestOperands()[2];
  int64_t initial = getConstant(pre.getDestOperands()[2]);
  for (unsigned operand = 0; operand < 2; ++operand) {
    auto read = reads[operand];
    auto enc = dyn_cast<ttg::DotOperandEncodingAttr>(read.getType().getEncoding());
    auto view = copies[operand].getDest().getDefiningOp<ttg::MemDescIndexOp>();
    auto root = view ? view.getSrc().getDefiningOp<ttg::LocalAllocOp>() : ttg::LocalAllocOp{};
    if (!enc || enc.getOpIdx() != operand || enc.getParent() != mma ||
        !root || root->getBlock() != entry || root.getSrc() || !root.isSharedMemoryAlloc() ||
        !llvm::all_of(root.getType().getShape(), [](int64_t dim) { return dim > 0; }) ||
        root.getType().getAllocShape() != root.getType().getShape() ||
        root.getType().getRank() != 3 ||
        !isa<ttg::SwizzledSharedEncodingAttr, ttg::SharedLinearEncodingAttr>(root.getType().getEncoding()) ||
        !hasSingleCTALayout(root.getType().getEncoding()) || !hasSingleCTALayout(cast<RankedTensorType>(copies[operand].getOffsets().getType()).getEncoding()) ||
        !(root.getType().getElementType().isF16() || root.getType().getElementType().isBF16()))
      return {};
    int64_t nDim = root.getType().getShape()[0];
    if (nDim < 2 || nDim > 4)
      return {};
    unsigned n = static_cast<unsigned>(nDim);
    if ((slots && slots != n) || initial < 0 || initial >= n ||
        header->getNumArguments() != 4 * n + 1 || view.getIndex() != next)
      return {};
    slots = n;
    auto select = next.getDefiningOp<arith::SelectOp>();
    auto wrapCmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
    if (!wrapCmp || wrapCmp.getPredicate() != arith::CmpIPredicate::slt ||
        wrapCmp.getLhs() != select.getTrueValue() || !isI32Constant(wrapCmp.getRhs(), n) ||
        !isI32Constant(select.getFalseValue(), 0) || !isIncrementByOne(select.getTrueValue(), counter))
      return {};
    unsigned tokenBase = 3 + operand * (n - 1), descBase = 3 + (2 + operand) * (n - 1);
    unsigned dotArg = 3 + 4 * (n - 1) + operand;
    if (read.getSrc() != header->getArgument(descBase) || read.getToken() != wait.getResult() ||
        wait->getOperand(operand) != header->getArgument(tokenBase) ||
        !publication->isBeforeInBlock(read) || !read->isBeforeInBlock(copies[0]) ||
        !read->isBeforeInBlock(copies[1]) || !publication->isBeforeInBlock(copies[operand]) ||
        dot->getOperand(operand) != header->getArgument(dotArg) ||
        back.getDestOperands()[dotArg] != read.getResult())
      return {};
    for (unsigned i = 0; i < n - 1; ++i) {
      Value desc = header->getArgument(descBase + i);
      auto initialView = pre.getDestOperands()[descBase + i].getDefiningOp<ttg::MemDescIndexOp>();
      auto commit = pre.getDestOperands()[tokenBase + i].getDefiningOp<ttg::AsyncCommitGroupOp>();
      auto producer = commit && commit->getNumOperands() == 1
                          ? commit->getOperand(0).getDefiningOp<hcugpu::BufferLoadToLocalOp>()
                          : hcugpu::BufferLoadToLocalOp{};
      int64_t idx = initialView ? getConstant(initialView.getIndex()) : -1;
      if (!initialView || initialView.getSrc() != root.getResult() || idx < 0 || idx >= n ||
          !producer || producer->getBlock() != entry || producer.getDest() != initialView.getResult())
        return {};
      // Invariant: slot(phi) = (counter + phase) % n. A shifted
      // backedge phi contributes phase(q)-1; root[Cnext] contributes 0.
      unsigned phase = (idx - initial + n) % n;
      if (i + 1 < n - 1) {
        auto nextInitial = pre.getDestOperands()[descBase + i + 1].getDefiningOp<ttg::MemDescIndexOp>();
        int64_t q = nextInitial ? getConstant(nextInitial.getIndex()) : -1;
        if (q < 0 || q >= n || phase != (q - initial - 1 + 2 * n) % n ||
            back.getDestOperands()[descBase + i] != header->getArgument(descBase + i + 1) ||
            back.getDestOperands()[tokenBase + i] != header->getArgument(tokenBase + i + 1))
          return {};
      } else {
        auto last = back.getDestOperands()[tokenBase + i].getDefiningOp<ttg::AsyncCommitGroupOp>();
        if (phase != 0 || back.getDestOperands()[descBase + i] != view.getResult() ||
            !last || last->getNumOperands() != 1 || last->getOperand(0) != copies[operand].getResult())
          return {};
      }
      // The future copy is root[Cnext], hence phase 1 in this iteration.
      if (i == 0 && phase == 1)
        return {};
      descriptors.push_back(desc);
      owner[desc] = operand;
    }
    auto initRead = pre.getDestOperands()[dotArg].getDefiningOp<ttg::LocalLoadOp>();
    auto initView = initRead ? initRead.getSrc().getDefiningOp<ttg::MemDescIndexOp>() : ttg::MemDescIndexOp{};
    if (!initRead || initRead->getBlock() != entry || !initView ||
        initView.getSrc() != root.getResult())
      return {};
    unsigned bodyViews = 0, deallocs = 0;
    Operation *deallocOp = nullptr;
    for (Operation *user : root.getResult().getUsers()) {
      if (auto index = dyn_cast<ttg::MemDescIndexOp>(user)) {
        if (index->getBlock() == body) {
          if (index != view)
            return {};
          ++bodyViews;
        } else if (index->getBlock() != entry || getConstant(index.getIndex()) < 0 ||
                 getConstant(index.getIndex()) >= n)
          return {};
        if (index.getType().getShape() != root.getType().getShape().drop_front() ||
            index.getType().getEncoding() != root.getType().getEncoding() ||
            index.getType().getElementType() != root.getType().getElementType() ||
            index.getType().getMemorySpace() != root.getType().getMemorySpace() ||
            index.getType().getMutableMemory() != root.getType().getMutableMemory())
          return {};
        descriptors.push_back(index.getResult());
        owner[index.getResult()] = operand;
      } else if (isa<ttg::LocalDeallocOp>(user) && user->getBlock() == exit) {
        ++deallocs;
        deallocOp = user;
      } else
        return {};
    }
    if (bodyViews != 1 || deallocs != 1)
      return {};
    deallocOps.push_back(deallocOp);
    roots.push_back(root);
    pairs.emplace_back(read.getOperation(), copies[operand].getOperation());
  }
  if (roots[0] == roots[1] || roots[0].getType().getElementType() != roots[1].getType().getElementType() ||
      dot.getC() != header->getArgument(1) || back.getDestOperands()[1] != dot.getResult())
    return {};
  for (Value desc : descriptors)
    for (OpOperand &use : desc.getUses()) {
      Operation *user = use.getOwner();
      if (auto load = dyn_cast<ttg::LocalLoadOp>(user)) {
        Value token = load.getToken();
        auto w = token ? token.getDefiningOp<hcugpu::AsyncWaitOp>() : hcugpu::AsyncWaitOp{};
        if (load.getSrc() != desc || !w || w->getBlock() != user->getBlock() || !w->isBeforeInBlock(user) ||
            (user->getBlock() == exit && !user->isBeforeInBlock(deallocOps[owner[desc]])))
          return {};
      } else if (auto copy = dyn_cast<hcugpu::BufferLoadToLocalOp>(user)) {
        if (copy.getDest() != desc || copy.getOther() || copy.getStride() ||
            (copy->getBlock() != entry && copy->getBlock() != body) ||
            cast<triton::PointerType>(copy.getPtr().getType()).getAddressSpace() != 1)
          return {};
      } else if (auto branch = dyn_cast<cf::BranchOp>(user)) {
        unsigned arg = use.getOperandNumber();
        if (branch.getDest() != header || (user->getBlock() != entry && user->getBlock() != body) ||
            arg >= header->getNumArguments() || !owner.count(header->getArgument(arg)) ||
            owner[header->getArgument(arg)] != owner[desc])
          return {};
      } else
        return {};
    }
  if (llvm::any_of(funcs[0].getArguments(), [](Value v) {
        return isa<ttg::MemDescType>(v.getType());
      }))
    return {};
  auto knownDescriptor = [&](Value v) {
    return !isa<ttg::MemDescType>(v.getType()) || owner.count(v) ||
           llvm::any_of(roots, [&](ttg::LocalAllocOp r) { return r.getResult() == v; });
  };
  unsigned allocations = 0;
  for (Block &block : funcs[0].getBody())
    for (Operation &op : block) {
      if (op.getNumRegions() || isa<CallOpInterface>(op) ||
          !llvm::all_of(op.getOperands(), knownDescriptor) ||
          !llvm::all_of(op.getResults(), knownDescriptor))
        return {};
      if (isa<ttg::LocalAllocOp>(op))
        ++allocations;
      if (auto read = dyn_cast<ttg::LocalLoadOp>(op))
        if (!owner.count(read.getSrc()))
          return {};
      if (auto copy = dyn_cast<hcugpu::BufferLoadToLocalOp>(op))
        if (!owner.count(copy.getDest()))
          return {};
      if (auto otherDot = dyn_cast<triton::DotOp>(op))
        if (otherDot.getA().getType() != dot.getA().getType() ||
            otherDot.getB().getType() != dot.getB().getType() ||
            otherDot.getType() != dot.getType())
          return {};
      if (!isRegisterOnlyPrefetchOp(&op) && op.getName().getStringRef() != "tt.get_program_id" &&
          !isa<cf::BranchOp, cf::CondBranchOp, triton::ReturnOp, triton::StoreOp,
               triton::DotOp, hcugpu::BufferStoreOp, ttg::ConvertLayoutOp, ttg::LocalAllocOp,
               ttg::LocalDeallocOp, ttg::MemDescIndexOp, ttg::LocalLoadOp,
               hcugpu::BufferLoadToLocalOp, hcugpu::AsyncWaitOp, ttg::AsyncCommitGroupOp,
               ttg::LocalBarrierOp, ROCDL::SchedBarrier>(op))
        return {};
    }
  // Entry/backedge equations establish slot=(counter+phase)%N inductively.
  // The read phase differs from copy phase 1. On every backedge, the next
  // mandatory publication barrier waits DS0 and rendezvous before reuse.
  // Keep all publication waits/RAW/WAW checks; exempt only these exact WARs.
  return allocations == 2 ? pairs : decltype(pairs){};
}

bool membarFilter(Operation *op1, Operation *op2) {
  return (filterAlternatingSlotWAR(op1, op2) ||
          filterAsyncLocalLoadsDependencies(op1, op2) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}
} // namespace mlir::triton::HCU
