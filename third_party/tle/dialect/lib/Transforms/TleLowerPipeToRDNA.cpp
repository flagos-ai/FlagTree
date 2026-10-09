/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Lower CTA-scoped SMEM `tle.pipe.*` ops to the AMD/RDNA hardware mbarrier ops
// (triton::amdgpu::{Init,Arrive,Wait}BarrierOp). This is the RDNA analogue of
// TleLowerPipeToNvws.cpp, but correctness-first: it does not model warp
// specialization or async/TMA transport. Every participant is the whole CTA, so
// each per-stage channel is a pair of i64 LDS mbarriers (full/empty) and all
// counts collapse to the CTA thread count. When warp_specialize lands on RDNA
// the whole-CTA counts become per-wave task counts.
//
// The AMD backend is only present in builds that enable the amd codegen
// backend; the TLE transforms library is shared across backends. We therefore
// gate everything that touches the AMD dialect behind TLE_AMD_PIPE (set by the
// TLE CMake only when amd is built). The pass and its tablegen-generated factory
// still exist on every backend (so GEN_PASS_REGISTRATION resolves); on non-amd
// builds runOnOperation is a no-op and any surviving tle.pipe op is diagnosed by
// the backend that owns it.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <string>

#ifdef TLE_AMD_PIPE
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#endif

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_DEF_TRITONTLELOWERPIPETORDNA
#include "tle/dialect/include/Transforms/Passes.h.inc"

#ifdef TLE_AMD_PIPE
namespace amdg = mlir::triton::amdgpu;

namespace {

//===----------------------------------------------------------------------===//
// Pipe IR accessors (mirrors TleLowerPipeToNvws.cpp)
//===----------------------------------------------------------------------===//

static int64_t getPipeCapacity(Operation *op) {
  return op->getAttrOfType<IntegerAttr>("capacity").getInt();
}

static OperandRange getPipeFields(Operation *op) {
  if (auto p = dyn_cast<PipeCreateOp>(op))
    return p.getFields();
  if (auto p = dyn_cast<PipeWriterAcquireOp>(op))
    return p.getFields();
  if (auto p = dyn_cast<PipeWriterCommitOp>(op))
    return p.getFields();
  if (auto p = dyn_cast<PipeWriterCloseOp>(op))
    return p.getFields();
  if (auto p = dyn_cast<PipeReaderWaitOp>(op))
    return p.getFields();
  if (auto p = dyn_cast<PipeReaderReleaseOp>(op))
    return p.getFields();
  return cast<PipeDrainOp>(op).getFields();
}

static Value getPipeIdentity(Operation *op) {
  if (auto p = dyn_cast<PipeCreateOp>(op))
    return p.getIdentity();
  if (auto p = dyn_cast<PipeWriterAcquireOp>(op))
    return p.getIdentity();
  if (auto p = dyn_cast<PipeWriterCommitOp>(op))
    return p.getIdentity();
  if (auto p = dyn_cast<PipeWriterCloseOp>(op))
    return p.getIdentity();
  if (auto p = dyn_cast<PipeReaderWaitOp>(op))
    return p.getIdentity();
  if (auto p = dyn_cast<PipeReaderReleaseOp>(op))
    return p.getIdentity();
  return cast<PipeDrainOp>(op).getIdentity();
}

static bool isPipeLifecycleOp(Operation *op) {
  return isa<PipeCreateOp, PipeWriterAcquireOp, PipeWriterCommitOp,
             PipeWriterCloseOp, PipeReaderWaitOp, PipeReaderReleaseOp,
             PipeDrainOp>(op);
}

static bool containsPipeLifecycleOp(tt::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isPipeLifecycleOp(op))
      found = true;
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Inline noinline pipe helper calls (so every lifecycle op binds a concrete
// pipe instance). The inverse is restored later by TleRestorePipeFunctionCalls,
// exactly as on the NVWS path.
//===----------------------------------------------------------------------===//

static LogicalResult inlinePipeCall(tt::CallOp call, tt::FuncOp callee,
                                    int64_t callId) {
  if (callee.isExternal())
    return call.emitOpError("cannot inline external callee containing pipe ops");
  Region &body = callee.getBody();
  if (!body.hasOneBlock())
    return call.emitOpError("cannot inline multi-block callee containing pipe "
                            "ops before pipe lowering");
  Block &block = body.front();
  auto returnOp = dyn_cast<tt::ReturnOp>(block.getTerminator());
  if (!returnOp)
    return call.emitOpError("callee containing pipe ops must terminate with "
                            "tt.return before pipe lowering");
  if (returnOp.getNumOperands() != call.getNumResults())
    return call.emitOpError("callee return count does not match call results");

  OpBuilder builder(call);
  SmallVector<Type> argumentTypes = llvm::to_vector(llvm::map_range(
      call.getOperands(), [](Value value) { return value.getType(); }));
  auto begin = PipeCallBeginOp::create(builder, call.getLoc(), argumentTypes,
                                       call.getOperands(), callee.getName(),
                                       callId);
  IRMapping mapping;
  for (auto [arg, alias] : llvm::zip(block.getArguments(), begin.getAliases()))
    mapping.map(arg, alias);
  for (Operation &op : block.getOperations()) {
    if (&op == returnOp.getOperation())
      continue;
    builder.clone(op, mapping);
  }
  PipeCallEndOp::create(builder, call.getLoc(), callee.getName(), callId);
  for (auto [result, returned] :
       llvm::zip(call.getResults(), returnOp.getOperands()))
    result.replaceAllUsesWith(mapping.lookupOrDefault(returned));
  call.erase();
  return success();
}

static LogicalResult inlinePipeHelperCalls(ModuleOp module) {
  bool changed = true;
  int64_t nextCallId = 0;
  while (changed) {
    changed = false;
    SmallVector<tt::CallOp> calls;
    module.walk([&](tt::CallOp call) {
      auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
      if (callee && containsPipeLifecycleOp(callee))
        calls.push_back(call);
    });
    for (tt::CallOp call : calls) {
      if (!call->getBlock())
        continue;
      auto callee = module.lookupSymbol<tt::FuncOp>(call.getCallee());
      if (!callee || !containsPipeLifecycleOp(callee))
        continue;
      if (failed(inlinePipeCall(call, callee, nextCallId++)))
        return failure();
      changed = true;
    }
  }
  for (tt::FuncOp func :
       llvm::make_early_inc_range(module.getOps<tt::FuncOp>())) {
    if (!containsPipeLifecycleOp(func))
      continue;
    if (func.getVisibility() != SymbolTable::Visibility::Public &&
        SymbolTable::symbolKnownUseEmpty(func, module)) {
      func.erase();
      continue;
    }
    if (func.getVisibility() != SymbolTable::Visibility::Public)
      return func.emitOpError("contains pipe ops but still has call sites after "
                              "pipe helper inlining");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pipe identity key (mirrors TleLowerPipeToNvws.cpp). A pipe endpoint inside a
// warp_specialize partition sees the identity/fields as isolated block-args, so
// resolve those back to the captured value — otherwise a writer in a partition
// and a reader in the default region key to different pipes.
//===----------------------------------------------------------------------===//

static Value canonicalizePipeField(Value field) {
  while (true) {
    if (auto result = dyn_cast<OpResult>(field)) {
      if (auto begin = dyn_cast<PipeCallBeginOp>(result.getOwner())) {
        field = begin.getArguments()[result.getResultNumber()];
        continue;
      }
    }
    if (auto blockArg = dyn_cast<BlockArgument>(field)) {
      if (auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
              blockArg.getOwner()->getParentOp())) {
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(
                partitions->getParentOp())) {
          field = wsOp.getExplicitCaptures()[blockArg.getArgNumber()];
          continue;
        }
      }
    }
    break;
  }
  return field;
}

static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << getPipeCapacity(op) << "|";
  op->getAttr("scope").print(os);
  os << "|" << canonicalizePipeField(getPipeIdentity(op)).getAsOpaquePointer()
     << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getPipeFields(op))
    os << canonicalizePipeField(field).getAsOpaquePointer() << ",";
  return key;
}

static bool isOneShotPipe(PipeCreateOp op) {
  if (auto oneShot = op->getAttrOfType<BoolAttr>("one_shot"))
    return oneShot.getValue();
  return false;
}

//===----------------------------------------------------------------------===//
// Shared-memory helpers
//===----------------------------------------------------------------------===//

// Thread count of the partition (or the whole CTA, with no warp specialization)
// that `op` runs in. lookupNumWarps is warp-specialization-aware: inside a
// warp_specialize partition it returns that partition's num_warps. Each thread
// arrives with count 1, so a barrier sized to this count flips after exactly the
// partition's threads arrive.
static FailureOr<int32_t> getThreadCount(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (numWarps <= 0 || threadsPerWarp <= 0) {
    op->emitOpError("requires positive num_warps and threads_per_warp "
                    "to infer pipe participant count");
    return failure();
  }
  return numWarps * threadsPerWarp;
}

static Attribute getSharedEncoding(MLIRContext *context, int64_t rank) {
  SmallVector<unsigned> order;
  for (int64_t dim = rank - 1; dim >= 0; --dim)
    order.push_back(static_cast<unsigned>(dim));
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(context, rank);
  return ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, order,
                                              ctaLayout);
}

static ttg::MemDescType getBarrierArrayType(MLIRContext *context,
                                            int64_t capacity) {
  return ttg::MemDescType::get({capacity, 1}, IntegerType::get(context, 64),
                               getSharedEncoding(context, 2),
                               ttg::SharedMemorySpaceAttr::get(context),
                               /*mutableMemory=*/true);
}

static ttg::MemDescType getBarrierSlotType(MLIRContext *context) {
  return ttg::MemDescType::get({1}, IntegerType::get(context, 64),
                               getSharedEncoding(context, 1),
                               ttg::SharedMemorySpaceAttr::get(context),
                               /*mutableMemory=*/true);
}

//===----------------------------------------------------------------------===//
// Per-pipe lowering state
//===----------------------------------------------------------------------===//

struct RDNAPipeState {
  Value fullArray;  // {C,1} i64 LDS: writer -> reader (data ready)
  Value emptyArray; // {C,1} i64 LDS: reader -> writer (slot free)
  Value closeTags;  // {C,1} i32 LDS close flags, or null for one-shot pipes
  Value drainBar;   // {1} i64 LDS drain rendezvous, or null if no drain
  ttg::MemDescType barSlotType;      // {1} i64
  ttg::MemDescType closeTagSlotType; // {1} i32 (null for one-shot)
  int64_t capacity = 0;
  bool oneShot = false;
};

// Per-endpoint thread counts for a pipe. With warp specialization the producer
// and consumer live in different partitions, so each barrier is sized by the
// partition that actually arrives on it (via lookupNumWarps on that endpoint's
// op), not by the whole CTA.
struct PipeCounts {
  int32_t producerCnt = 0; // threads that arrive on `full` (writer partition)
  int32_t consumerCnt = 0; // threads that arrive on `empty` (reader partition)
  int32_t drainCnt = 0;    // threads that rendezvous at drain
  bool hasDrain = false;
};

//===----------------------------------------------------------------------===//
// warp_specialize capture plumbing. Barrier/close-tag/drain allocations live in
// the enclosing scope, but a writer/reader inside an isolated-from-above
// partition cannot reference them directly — the value must be threaded in as a
// warp_specialize capture. These mirror TleLowerPipeToNvws.cpp.
//===----------------------------------------------------------------------===//

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingWarpSpecializePartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
      return std::make_pair(
          cast<ttg::WarpSpecializeOp>(partitions->getParentOp()), region);
    region = parent->getParentRegion();
  }
  return std::nullopt;
}

static bool isDefinedInsideRegion(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = value.getDefiningOp();
  return def && region->isAncestor(def->getParentRegion());
}

// If `useOp` is inside a warp_specialize partition and `value` is defined
// outside it, find-or-add `value` as an explicit capture and return the
// partition's block argument for it; otherwise return `value` unchanged.
static Value getWarpSpecializeCaptureForUse(Operation *useOp, Value value) {
  auto partition = getEnclosingWarpSpecializePartition(useOp);
  if (!partition)
    return value;
  ttg::WarpSpecializeOp wsOp = partition->first;
  Region *region = partition->second;
  if (isDefinedInsideRegion(value, region))
    return value;
  for (auto indexed : llvm::enumerate(wsOp.getExplicitCaptures()))
    if (indexed.value() == value)
      return region->getArgument(indexed.index());
  wsOp->insertOperands(wsOp.getNumOperands(), value);
  unsigned captureIndex = wsOp.getNumOperands() - 1;
  for (Region *partitionRegion : wsOp.getPartitionRegions())
    partitionRegion->addArgument(value.getType(), value.getLoc());
  return region->getArgument(captureIndex);
}

// Index one i64 mbarrier slot out of a {C,1} barrier array, threading the array
// through a warp_specialize capture when `source` is inside a partition.
static Value barrierSlot(OpBuilder &builder, Location loc,
                         const RDNAPipeState &st, Value array, Value stage,
                         Operation *source) {
  array = getWarpSpecializeCaptureForUse(source, array);
  return ttg::MemDescIndexOp::create(builder, loc, st.barSlotType, array, stage);
}

static Value i32Const(OpBuilder &builder, Location loc, int64_t v) {
  return arith::ConstantIntOp::create(builder, loc, v, 32);
}

// tle.pipe $phase is an i1 parity; the AMD WaitBarrier takes an i32 phase. With
// the 1-bit phase field (see BarrierOpToLLVM.cpp) the parity maps straight over.
static Value phaseToI32(OpBuilder &builder, Location loc, Value phaseI1) {
  return arith::ExtUIOp::create(builder, loc, builder.getI32Type(), phaseI1);
}

//===----------------------------------------------------------------------===//
// Close-tag ring (i32 LDS, one slot per stage). Only non-one-shot pipes carry
// it; reader_wait reads it to answer `is_closed`, writer_close sets it.
//===----------------------------------------------------------------------===//

static RankedTensorType closeTagTensorType(Operation *op, OpBuilder &builder,
                                           ArrayRef<int64_t> shape) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  int numCTAs = ttg::TritonGPUDialect::getNumCTAs(module);
  Attribute encoding = ttg::getDefaultBlockedEncoding(context, shape, numWarps,
                                                      threadsPerWarp, numCTAs);
  return RankedTensorType::get(shape, builder.getI32Type(), encoding);
}

static Value closeTagSplat(OpBuilder &builder, Location loc,
                           RankedTensorType type, bool value) {
  Value scalar = i32Const(builder, loc, value ? 1 : 0);
  return tt::SplatOp::create(builder, loc, type, scalar);
}

static void storeCloseTag(OpBuilder &builder, Location loc,
                          const RDNAPipeState &st, Value stage, bool value,
                          Operation *source) {
  Value closeTags = getWarpSpecializeCaptureForUse(source, st.closeTags);
  Value slot = ttg::MemDescIndexOp::create(builder, loc, st.closeTagSlotType,
                                           closeTags, stage);
  RankedTensorType tagType = closeTagTensorType(source, builder, {1});
  Value tag = closeTagSplat(builder, loc, tagType, value);
  ttg::LocalStoreOp::create(builder, loc, tag, slot);
}

static Value loadCloseTag(OpBuilder &builder, Location loc,
                          const RDNAPipeState &st, Value stage,
                          Operation *source) {
  Value closeTags = getWarpSpecializeCaptureForUse(source, st.closeTags);
  Value slot = ttg::MemDescIndexOp::create(builder, loc, st.closeTagSlotType,
                                           closeTags, stage);
  RankedTensorType tagType = closeTagTensorType(source, builder, {1});
  Value tagTensor = ttg::LocalLoadOp::create(builder, loc, tagType, slot);
  Value tagI32 =
      tt::UnsplatOp::create(builder, loc, builder.getI32Type(), tagTensor);
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, tagI32,
                               i32Const(builder, loc, 0));
}

//===----------------------------------------------------------------------===//
// Build the barriers for one pipe.create.
//===----------------------------------------------------------------------===//

static FailureOr<RDNAPipeState> createPipeState(PipeCreateOp op,
                                                const PipeCounts &counts) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  MLIRContext *context = op->getContext();

  RDNAPipeState st;
  st.capacity = getPipeCapacity(op);
  st.oneShot = isOneShotPipe(op);
  st.barSlotType = getBarrierSlotType(context);

  st.fullArray =
      ttg::LocalAllocOp::create(builder, loc, getBarrierArrayType(context, st.capacity));
  st.emptyArray =
      ttg::LocalAllocOp::create(builder, loc, getBarrierArrayType(context, st.capacity));

  // Init each stage's `full` with the producer count (phase 0: the producer
  // arrives to publish) and `empty` with the consumer count but already flipped
  // (init_phase = 1: slots start free, so the first writer_acquire on parity 0
  // proceeds). Seeding the phase at init replaces an all-waves pre-arrive, which
  // would miscount once producer/consumer are warp-specialized sub-groups.
  for (int64_t s = 0; s < st.capacity; ++s) {
    Value idx = i32Const(builder, loc, s);
    Value full = barrierSlot(builder, loc, st, st.fullArray, idx, op);
    Value empty = barrierSlot(builder, loc, st, st.emptyArray, idx, op);
    amdg::InitBarrierOp::create(builder, loc, full, counts.producerCnt);
    amdg::InitBarrierOp::create(builder, loc, empty, counts.consumerCnt,
                                /*init_phase=*/1);
  }

  if (!st.oneShot) {
    // The close ring is initialized from a ranked tensor splat, whose element
    // count must be a power of two (Triton tensor invariant). Pad the ring to
    // PowerOf2Ceil(capacity); the extra slots are never indexed (stage < C).
    // The i64 barrier arrays above are memdescs, not tensors, so they can stay
    // at the logical capacity.
    int64_t controlCapacity = llvm::PowerOf2Ceil(st.capacity);
    st.closeTagSlotType =
        ttg::MemDescType::get({1}, builder.getI32Type(), getSharedEncoding(context, 1),
                              ttg::SharedMemorySpaceAttr::get(context),
                              /*mutableMemory=*/true);
    auto closeTagArrayType =
        ttg::MemDescType::get({controlCapacity, 1}, builder.getI32Type(),
                              getSharedEncoding(context, 2),
                              ttg::SharedMemorySpaceAttr::get(context),
                              /*mutableMemory=*/true);
    RankedTensorType initType =
        closeTagTensorType(op, builder, {controlCapacity, 1});
    Value init = closeTagSplat(builder, loc, initType, /*value=*/false);
    st.closeTags =
        ttg::LocalAllocOp::create(builder, loc, closeTagArrayType, init);
  }

  if (counts.hasDrain) {
    st.drainBar =
        ttg::LocalAllocOp::create(builder, loc, getBarrierSlotType(context));
    amdg::InitBarrierOp::create(builder, loc, st.drainBar, counts.drainCnt);
    mlir::gpu::BarrierOp::create(builder, loc);
  }
  return st;
}

} // namespace

struct TritonTleLowerPipeToRDNA
    : public impl::TritonTleLowerPipeToRDNABase<TritonTleLowerPipeToRDNA> {
  using Base = impl::TritonTleLowerPipeToRDNABase<TritonTleLowerPipeToRDNA>;

  // The amd dialect is declared as a dependent dialect here (not in Passes.td)
  // so the generated base stays free of the amd header on non-amd builds. It
  // must be registered before the multi-threaded pass run; loading it inside
  // runOnOperation would abort ("Loading a dialect while multi-threaded").
  void getDependentDialects(DialectRegistry &registry) const override {
    Base::getDependentDialects(registry);
    registry.insert<amdg::TritonAMDGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(inlinePipeHelperCalls(module)))
      return signalPassFailure();

    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (isPipeLifecycleOp(op))
        ops.push_back(op);
    });
    if (ops.empty())
      return;

    // Pre-scan: size each pipe's barriers by the partition each endpoint runs
    // in. The producer/consumer may be different warp_specialize partitions, so
    // `full` is sized by a writer op's thread count and `empty` by a reader
    // op's; a drain adds its own rendezvous barrier.
    std::map<std::string, PipeCounts> countsByKey;
    for (Operation *op : ops) {
      PipeCounts &c = countsByKey[getPipeKey(op)];
      int32_t *slot = nullptr;
      if (isa<PipeWriterCommitOp, PipeWriterAcquireOp, PipeWriterCloseOp>(op))
        slot = &c.producerCnt;
      else if (isa<PipeReaderWaitOp, PipeReaderReleaseOp>(op))
        slot = &c.consumerCnt;
      else if (isa<PipeDrainOp>(op)) {
        slot = &c.drainCnt;
        c.hasDrain = true;
      } else {
        continue; // create: fallback filled below
      }
      FailureOr<int32_t> cnt = getThreadCount(op);
      if (failed(cnt))
        return signalPassFailure();
      *slot = *cnt;
    }
    // Degenerate pipes (missing a writer or reader) fall back to the create op's
    // own thread count.
    for (Operation *op : ops) {
      if (!isa<PipeCreateOp>(op))
        continue;
      FailureOr<int32_t> cnt = getThreadCount(op);
      if (failed(cnt))
        return signalPassFailure();
      PipeCounts &c = countsByKey[getPipeKey(op)];
      if (c.producerCnt == 0)
        c.producerCnt = *cnt;
      if (c.consumerCnt == 0)
        c.consumerCnt = *cnt;
    }

    std::map<std::string, RDNAPipeState> states;
    SmallVector<PipeCreateOp> creates;

    for (Operation *op : ops) {
      std::string key = getPipeKey(op);
      if (auto create = dyn_cast<PipeCreateOp>(op)) {
        FailureOr<RDNAPipeState> st =
            createPipeState(create, countsByKey[key]);
        if (failed(st))
          return signalPassFailure();
        states[key] = *st;
        creates.push_back(create);
        continue;
      }

      auto it = states.find(key);
      if (it == states.end()) {
        op->emitOpError("pipe op has no matching tle.pipe.create in this "
                        "module");
        return signalPassFailure();
      }
      RDNAPipeState &st = it->second;
      OpBuilder builder(op);
      Location loc = op->getLoc();

      if (auto acq = dyn_cast<PipeWriterAcquireOp>(op)) {
        Value slot =
            barrierSlot(builder, loc, st, st.emptyArray, acq.getStage(), op);
        amdg::WaitBarrierOp::create(builder, loc, slot,
                                    phaseToI32(builder, loc, acq.getPhase()));
      } else if (auto commit = dyn_cast<PipeWriterCommitOp>(op)) {
        Value slot =
            barrierSlot(builder, loc, st, st.fullArray, commit.getStage(), op);
        amdg::ArriveBarrierOp::create(builder, loc, slot, /*count=*/1);
      } else if (auto close = dyn_cast<PipeWriterCloseOp>(op)) {
        if (!st.oneShot)
          storeCloseTag(builder, loc, st, close.getStage(), /*value=*/true, op);
        // Publish so a waiting reader wakes and observes the close tag.
        Value slot =
            barrierSlot(builder, loc, st, st.fullArray, close.getStage(), op);
        amdg::ArriveBarrierOp::create(builder, loc, slot, /*count=*/1);
      } else if (auto wait = dyn_cast<PipeReaderWaitOp>(op)) {
        Value slot =
            barrierSlot(builder, loc, st, st.fullArray, wait.getStage(), op);
        amdg::WaitBarrierOp::create(builder, loc, slot,
                                    phaseToI32(builder, loc, wait.getPhase()));
        Value closed;
        if (st.oneShot)
          closed = arith::ConstantIntOp::create(builder, loc, 0, 1);
        else
          closed = loadCloseTag(builder, loc, st, wait.getStage(), op);
        wait.getIsClosed().replaceAllUsesWith(closed);
      } else if (auto rel = dyn_cast<PipeReaderReleaseOp>(op)) {
        Value slot =
            barrierSlot(builder, loc, st, st.emptyArray, rel.getStage(), op);
        amdg::ArriveBarrierOp::create(builder, loc, slot, /*count=*/1);
      } else if (isa<PipeDrainOp>(op)) {
        Value drainBar = getWarpSpecializeCaptureForUse(op, st.drainBar);
        amdg::ArriveBarrierOp::create(builder, loc, drainBar, /*count=*/1);
        amdg::WaitBarrierOp::create(builder, loc, drainBar,
                                    i32Const(builder, loc, 0));
      }
      op->erase();
    }

    // Replace each pipe identity with a sentinel so any lingering uses (e.g.
    // pipe.call_begin aliases, restored later) stay well-typed, then drop the
    // create ops.
    for (PipeCreateOp create : creates) {
      OpBuilder builder(create);
      Value sentinel = i32Const(builder, create.getLoc(), 0);
      create.getIdentity().replaceAllUsesWith(sentinel);
      create.erase();
    }
  }
};

#else // !TLE_AMD_PIPE

struct TritonTleLowerPipeToRDNA
    : public impl::TritonTleLowerPipeToRDNABase<TritonTleLowerPipeToRDNA> {
  void runOnOperation() override {
    // AMD backend not built: nothing to lower. Any surviving tle.pipe op is
    // reported by the backend that owns it.
  }
};

#endif // TLE_AMD_PIPE

} // namespace mlir::triton::tle
