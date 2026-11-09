#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"

#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferOuterScope";
#define LDBG(...)                                                              \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;
using namespace hivm;

namespace mlir {
namespace triton {

// Maximum number of flag allocation attempts per transfer group
static constexpr int kMaxFlagAttempts = 16;
// Flag ID 15 is reserved for pipe synchronization (e.g. PIPE_S) and must not
// be allocated to cross-core transfers. Usable IDs are 0..MAX_FLAG_ID (14).
static constexpr int kReservedPipeFlagId = 15;

// --- Attribute helpers ---

static int getFlagFromSyncOp(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("flag_id")) {
    return attr.getInt();
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
    return attr.getInt();
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("flag")) {
    return attr.getInt();
  }
  return -1;
}

static int getBlockId(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(mlir::CVPipeline::kBlockId))
    return attr.getInt();
  return -1;
}

static int getTransferId(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(mlir::CVPipeline::kTransferId))
    return attr.getInt();
  return -1;
}

// --- main_loop attribute helpers ---

/// Walk up from `op` (exclusive) and return the first ancestor satisfying
/// `pred`; nullptr if none.
static Operation *findAncestorOp(Operation *op,
                                 llvm::function_ref<bool(Operation *)> pred) {
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp())
    if (pred(cur))
      return cur;
  return nullptr;
}

/// Nearest ancestor carrying the main_loop attribute, if any.
static Operation *findMainLoopOp(Operation *op) {
  return findAncestorOp(
      op, [](Operation *cur) { return CVPipeline::isMainLoopOp(cur); });
}

/// True if the sync op is nested inside a main_loop op (ForOp/WhileOp).
static bool parentOpHasMainLoopAttr(Operation *syncOp) {
  if (!syncOp) {
    return false;
  }
  return findMainLoopOp(syncOp) != nullptr;
}

// --- Tag-driven transfer op classification helpers ---

/// True if op is inside a VECTOR scope; for direction derivation only
static bool isInsideVectorScope(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return false;
  }
  if (auto tcoreAttr = scopeOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type"))
    return tcoreAttr.getTcoretype() == TCoreType::VECTOR;
  return false;
}

/// True if op declares MemoryEffects Write on buffer
static bool hasWriteEffectOn(Operation *op, Value buffer) {
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface) {
    return false;
  }
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  iface.getEffects(effects);
  return llvm::any_of(effects, [&](const auto &effect) {
    return isa<MemoryEffects::Write>(effect.getEffect()) &&
           effect.getValue() == buffer;
  });
}

/// Walk single-use memref chain from receiverOp to the first op producing a
/// tensor. Returns nullptr when:
/// - head result is not a memref
/// - chain forks (multi-use), leaves the block, or hits an op that does not
///   produce exactly one result
static Operation *findChainTensorTerminal(Operation *receiverOp) {
  if (receiverOp->getNumResults() != 1) {
    return nullptr;
  }
  Value cur = receiverOp->getResult(0);
  Block *block = receiverOp->getBlock();
  // Terminates: SSA def-use is acyclic; same-block users appear strictly
  // after their def, so the walk never revisits a value.
  while (cur && isa<MemRefType>(cur.getType())) {
    if (!cur.hasOneUse()) {
      return nullptr; // forked chain: ambiguous
    }
    Operation *user = *cur.getUsers().begin();
    if (user->getBlock() != block || user->getNumResults() != 1) {
      return nullptr;
    }
    Value result = user->getResult(0);
    if (isa<RankedTensorType>(result.getType())) {
      return user;
    }
    cur = result;
  }
  return nullptr;
}

/// First block_id found in a block's ops, if any
static std::optional<int> getFirstBlockId(Block *block) {
  for (Operation &op : *block)
    if (auto id = CVPipeline::getOpBlockId(&op))
      return id;
  return std::nullopt;
}

// --- Operation search helpers ---

/// Find sync op with a specific flag, searching forward or backward in a block
static Operation *findSyncOpWithFlag(Block *block, Operation *start, int flag,
                                     bool forward, bool wantWait) {
  if (!block) {
    return nullptr;
  }
  auto it = start->getIterator();
  if (forward) {
    for (auto e = block->end(); it != e; ++it) {
      Operation *op = &*it;
      if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
        continue;
      }
      if (getFlagFromSyncOp(op) != flag) {
        continue;
      }
      if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) {
        return op;
      }
      if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) {
        return op;
      }
    }
  } else {
    if (it == block->begin()) {
      return nullptr;
    }
    do {
      --it;
      Operation *op = &*it;
      if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
        continue;
      }
      if (getFlagFromSyncOp(op) != flag) {
        continue;
      }
      if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) {
        return op;
      }
      if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) {
        return op;
      }
    } while (it != block->begin());
  }
  return nullptr;
}

// ============================================================================
// Step 1: Collect transfer group info
// ============================================================================

/// Collect all ops with ssbuffer.transfer_id, grouped by transfer_id
static int
collectOpsByTransferId(ModuleOp module,
                       DenseMap<int, SmallVector<Operation *>> &opsByTid) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(mlir::CVPipeline::kTransferId)) {
      return;
    }
    int tid = getTransferId(op);
    if (tid >= 0) {
      opsByTid[tid].push_back(op);
    }
  });
  LDBG("Collected " << opsByTid.size() << " transfer groups.");

  for (auto &p : opsByTid) {
    LDBG("  tid=" << p.first << " has " << p.second.size() << " ops.");
    DenseMap<int, int> blockIdCount;
    for (auto *op : p.second) {
      int bid = getBlockId(op);
      blockIdCount[bid]++;
    }
    for (auto &bp : blockIdCount) {
      LDBG("    block_id=" << bp.first << ": " << bp.second << " ops.");
    }
  }
  return 0;
}

/// Collect alloc/mark pairs from transfer ops in the group.
/// Identifies the correct cross-core buffer (ub/cbuf) used by each transfer op,
/// ignoring local buffers (cc on CUBE side) that are not part of the data
/// transfer.
static int collectBufferAllocs(const SmallVector<Operation *> &ops,
                               TransferGroupInfo &info) {
  // Helper: find the annotation.mark for a given alloc op
  auto findMarkForAlloc = [](Operation *allocOp) -> Operation * {
    Value allocResult = allocOp->getResult(0);
    for (auto *user : allocResult.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        return user;
    }
    return nullptr;
  };

  // Cross-core buffer = alloc defining the recorded buffer operand
  if (info.senderChain.transferOp && info.senderChain.bufferOperand) {
    if (auto *defOp = info.senderChain.bufferOperand.getDefiningOp()) {
      if (isa<memref::AllocOp>(defOp)) {
        info.senderBuf.allocOp = defOp;
        info.senderBuf.markOp = findMarkForAlloc(defOp);
        LDBG("Sender cross-core buffer: alloc from transferOp buffer operand.");
      }
    }
  }

  if (info.receiverChain.transferOp && info.receiverChain.bufferOperand) {
    if (auto *defOp = info.receiverChain.bufferOperand.getDefiningOp()) {
      if (isa<memref::AllocOp>(defOp)) {
        info.receiverBuf.allocOp = defOp;
        info.receiverBuf.markOp = findMarkForAlloc(defOp);
        LDBG("Receiver cross-core buffer: alloc from transferOp buffer "
             "operand.");
      }
    }
  }

  // Collect alloc/mark for the OTHER side if not yet found.
  // Some transfer ops (e.g. fixpipe) have both a local input (cc) and a
  // cross-core output (ub). The receiver side's buffer is the cross-core one.
  // Walk all allocs in the group to find any remaining unassigned buffer.
  SmallVector<Operation *> allocs;
  for (Operation *op : ops) {
    if (isa<memref::AllocOp>(op))
      allocs.push_back(op);
  }

  // Fill missing side from remaining allocs (prefer allocs with marks)
  for (auto *allocOp : allocs) {
    if (allocOp == info.senderBuf.allocOp ||
        allocOp == info.receiverBuf.allocOp)
      continue;
    Operation *mark = findMarkForAlloc(allocOp);
    if (!info.senderBuf.allocOp) {
      info.senderBuf.allocOp = allocOp;
      info.senderBuf.markOp = mark;
    } else if (!info.receiverBuf.allocOp) {
      info.receiverBuf.allocOp = allocOp;
      info.receiverBuf.markOp = mark;
    }
  }

  LDBG("Sender buffer: " << (info.senderBuf.allocOp ? "alloc" : "none") << " + "
                         << (info.senderBuf.markOp ? "mark" : "none") << ".");
  LDBG("Receiver buffer: " << (info.receiverBuf.allocOp ? "alloc" : "none")
                           << " + "
                           << (info.receiverBuf.markOp ? "mark" : "none")
                           << ".");
  return 0;
}

/// Collect llvm.load volatile and llvm.store volatile ops by transfer_id
static int collectLoadStoreOpsByTransferId(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &loadStoreByTid) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(mlir::CVPipeline::kTransferId)) {
      return;
    }
    int tid = getTransferId(op);
    if (tid < 0) {
      return;
    }
    if (isa<mlir::LLVM::LoadOp>(op) || isa<mlir::LLVM::StoreOp>(op)) {
      loadStoreByTid[tid].push_back(op);
    }
  });
  LDBG("Collected load/store ops for " << loadStoreByTid.size()
                                       << " transfer groups.");
  return 0;
}

/// Tag load/store ops with crossDeps (producer=store, consumer=load)
static int tagLoadStoreOpsWithCrossDeps(
    DenseMap<int, SmallVector<Operation *>> &loadStoreByTid) {
  for (auto &p : loadStoreByTid) {
    int tid = p.first;
    for (auto *op : p.second) {
      MLIRContext *ctx = op->getContext();
      OpBuilder builder(ctx);
      if (auto storeOp = dyn_cast<mlir::LLVM::StoreOp>(op)) {
        // producer: crossDeps = {tid, 1}
        // Tag the defining op of the store's second operand (ptr), not the
        // store itself
        Value ptr = storeOp.getOperand(1);
        if (auto *ptrDefOp = ptr.getDefiningOp()) {
          ptrDefOp->setAttr(
              mlir::CVPipeline::kCrossCoreDeps,
              builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                    builder.getI32IntegerAttr(1)}));
          LDBG("Tagged ptr-defining-op with crossDeps={tid=" << tid << ", 1}.");
        }
      } else if (auto loadOp = dyn_cast<mlir::LLVM::LoadOp>(op)) {
        // consumer: crossDeps = {tid, 0}
        // Tag the load op itself
        op->setAttr(mlir::CVPipeline::kCrossCoreDeps,
                    builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                          builder.getI32IntegerAttr(0)}));
        LDBG("Tagged llvm.load volatile with crossDeps={tid=" << tid
                                                              << ", 0}.");
      }
    }
  }
  return 0;
}

/// Collect extra sync ops (parent has no main_loop), paired by flag
static int collectExtraSync(const SmallVector<Operation *> &ops,
                            int originalFlag, ExtraSyncInfo &info) {
  SmallVector<Operation *> extraSets;
  SmallVector<Operation *> extraWaits;

  for (Operation *op : ops) {
    if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
      continue;
    }

    bool hasMainLoop = parentOpHasMainLoopAttr(op);
    LDBG("sync op: flag=" << getFlagFromSyncOp(op)
                          << ", block_id=" << getBlockId(op)
                          << ", parentHasMainLoop=" << hasMainLoop << ".");

    if (!hasMainLoop) {
      if (isa<hivm::SyncBlockSetOp>(op)) {
        extraSets.push_back(op);
      } else if (isa<hivm::SyncBlockWaitOp>(op)) {
        extraWaits.push_back(op);
      }
    }
  }

  // Match by flag
  for (auto *setOp : extraSets) {
    if (getFlagFromSyncOp(setOp) != originalFlag) {
      continue;
    }
    for (auto *waitOp : extraWaits) {
      if (getFlagFromSyncOp(waitOp) != originalFlag) {
        continue;
      }
      info.setOp = setOp;
      info.waitOp = waitOp;
      LDBG("Extra sync pair: set(flag="
           << originalFlag << ", block_id=" << getBlockId(setOp)
           << "), wait(flag=" << originalFlag
           << ", block_id=" << getBlockId(waitOp) << ".");
      return 0;
    }
  }

  // Fallback: use first available pair if exact match not found
  if (!extraSets.empty() && !extraWaits.empty()) {
    info.setOp = extraSets.front();
    info.waitOp = extraWaits.front();
  }

  return 0;
}

/// Collect in-loop sender/receiver chains. Tag+dataflow rules:
/// - group buffer: alloc carrying the transfer_id
/// - sender: writes a group buffer (Write effect or result-less)
/// - receiver head: first op reading a group buffer as a value
/// - no op-type matching; receiver trailing ops found at wrap time
static int collectTransferChains(const SmallVector<Operation *> &ops,
                                 int originalFlag, TransferChainInfo &info) {
  // Group buffers: allocs carrying this transfer_id, one per core (sender and
  // receiver sides each own one physical buffer)
  SmallVector<Value> groupBuffers;
  for (Operation *op : ops) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      groupBuffers.push_back(allocOp.getResult());
    }
  }
  auto isGroupBuffer = [&](Value v) {
    return llvm::is_contained(groupBuffers, v);
  };

  for (Operation *op : ops) {
    if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) ||
        !op->getBlock()) {
      continue;
    }
    // Buffers and annotations are data, not transfer behavior
    if (isa<memref::AllocOp>(op) || isa<annotation::MarkOp>(op)) {
      continue;
    }
    if (!parentOpHasMainLoopAttr(op)) {
      continue;
    }

    // First operand backed by a group buffer
    Value bufferOperand;
    for (Value operand : op->getOperands()) {
      if (isGroupBuffer(operand)) {
        bufferOperand = operand;
        break;
      }
    }
    if (!bufferOperand) {
      continue;
    }

    Block *block = op->getBlock();

    if (op->getNumResults() == 0 || hasWriteEffectOn(op, bufferOperand)) {
      // Writes the cross-core buffer → sender
      if (info.sender.transferOp) {
        continue;
      }
      info.sender.transferOp = op;
      info.sender.bufferOperand = bufferOperand;
      info.sender.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.sender.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      LDBG("Sender chain (tag-driven): " << op->getName()
                                         << ", flag=" << originalFlag << ".");
    } else if (!info.receiver.transferOp && op->getNumResults() > 0) {
      // Reads the cross-core buffer as a value → receiver head
      info.receiver.transferOp = op;
      info.receiver.bufferOperand = bufferOperand;
      info.receiver.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.receiver.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      info.receiver.toTensorOp = findChainTensorTerminal(op);
      LDBG("Receiver chain (tag-driven): "
           << op->getName() << ", flag=" << originalFlag << ", toTensorOp="
           << (info.receiver.toTensorOp ? "found" : "none") << ".");
    }
  }

  return 0;
}

/// Build TransferGroupInfo for a single transfer_id
static int buildTransferGroupData(int tid, const SmallVector<Operation *> &ops,
                                  TransferGroupInfo &info) {
  info.tid = tid;

  LDBG("Building group tid=" << tid << ", ops=" << ops.size() << ".");

  // 1. Determine original flag
  for (Operation *op : ops) {
    if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
      int f = getFlagFromSyncOp(op);
      if (f >= 0) {
        info.originalFlag = f;
        break;
      }
    }
  }

  // 2. Collect extra sync (parent has no main_loop)
  ExtraSyncInfo extraInfo;
  if (collectExtraSync(ops, info.originalFlag, extraInfo)) {
    return -1;
  }
  info.extraSyncSetOp = extraInfo.setOp;
  info.extraSyncWaitOp = extraInfo.waitOp;
  if (extraInfo.setOp && extraInfo.waitOp) {
    LDBG("Extra sync: set(block_id=" << getBlockId(extraInfo.setOp)
                                     << "), wait(block_id="
                                     << getBlockId(extraInfo.waitOp) << ".");
  } else {
    LDBG("Extra sync: not found.");
  }

  // 3. Collect transfer chain (parent has main_loop)
  TransferChainInfo chainInfo;
  if (collectTransferChains(ops, info.originalFlag, chainInfo)) {
    return -1;
  }
  info.senderChain = chainInfo.sender;
  info.receiverChain = chainInfo.receiver;

  // 3.5 Chain completeness: each side needs both its wait and set syncs.
  // Wrapping half a handshake (producer alternates flags while the consumer
  // does not) deadlocks; drop the group's chains entirely so the transfer
  // stays single-buffered instead.
  auto chainComplete = [](const TransferOpChain &chain) {
    return !chain.transferOp || (chain.waitOp && chain.setOp);
  };
  if (!chainComplete(info.senderChain) || !chainComplete(info.receiverChain)) {
    LDBG("Group tid=" << tid
                      << " has an incomplete chain (wait/set missing), keeping"
                      << " it single-buffered.");
    info.senderChain = TransferOpChain();
    info.receiverChain = TransferOpChain();
  }

  // 4. Direction from the executing core's scope:
  // - sender in VECTOR scope: V→C; in CUBE scope: C→V
  // - no sender: derive from receiver scope instead
  if (info.senderChain.transferOp) {
    info.isCtoV = !isInsideVectorScope(info.senderChain.transferOp);
  } else if (info.receiverChain.transferOp) {
    info.isCtoV = isInsideVectorScope(info.receiverChain.transferOp);
  }

  // 5. Collect buffer alloc/mark pairs from transfer ops
  //    Must run after transfer chain collection to identify the correct
  //    cross-core buffer (ub/cbuf) from each transfer op's operands,
  //    ignoring local buffers (e.g. cc on CUBE side).
  if (collectBufferAllocs(ops, info)) {
    return -1;
  }

  // 6. Output flag is assigned in collectTransferGroupData, shared by groups
  // with the same (originalFlag, direction).

  if (info.senderChain.transferOp || info.receiverChain.transferOp) {
    LDBG("Direction: " << (info.isCtoV ? "C→V" : "V→C")
                       << ", flag=" << info.originalFlag << ".");
  }

  return 0;
}

/// Collect TransferGroupInfo for all transfer groups
static int collectTransferGroupData(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &opsByTid,
    FlagIdManager &flagIdMgr, DenseMap<int, TransferGroupInfo> &groups) {
  for (auto &p : opsByTid) {
    TransferGroupInfo info;
    if (buildTransferGroupData(p.first, p.second, info)) {
      continue;
    }
    if (info.senderChain.transferOp || info.receiverChain.transferOp) {
      groups[p.first] = info;
    }
  }

  // Output flag reuse: groups with same (originalFlag, direction) share an
  // output flag. Assign one new flag per distinct key; std::map iterates in
  // sorted key order for determinism.
  std::map<std::pair<int, bool>, int> outputFlagByKey;
  for (auto &p : groups) {
    outputFlagByKey.try_emplace(
        std::make_pair(p.second.originalFlag, p.second.isCtoV),
        FlagIdManager::INVALID_FLAG_ID);
  }
  for (auto &entry : outputFlagByKey) {
    for (int attempt = 0; attempt < kMaxFlagAttempts; ++attempt) {
      int64_t pf = flagIdMgr.acquireId();
      if (pf == FlagIdManager::INVALID_FLAG_ID || pf != entry.first.first) {
        entry.second = static_cast<int>(pf);
        break;
      }
    }
  }
  for (auto &p : groups) {
    auto &g = p.second;
    g.outputFlag = outputFlagByKey[std::make_pair(g.originalFlag, g.isCtoV)];
    LDBG("Group tid=" << g.tid << " gets outputFlag=" << g.outputFlag
                      << " (shared originalFlag=" << g.originalFlag
                      << ", dir=" << (g.isCtoV ? "C→V" : "V→C") << ").");
  }

  return 0;
}

// ============================================================================
// Step 2: Create output buffers
// ============================================================================

static constexpr int kMaxTcbSearch = 100;

static int allocateNewTcbId(int startFrom, std::set<int> &usedTcbIds) {
  for (int id = startFrom; id < kMaxTcbSearch; ++id) {
    if (!usedTcbIds.count(id)) {
      usedTcbIds.insert(id);
      return id;
    }
  }
  return -1;
}

/// Create an output buffer for an input/output buffer pair
static int createOutputBufferPair(Operation *inputAllocOp, int tid, int tcbId,
                                  Value &inputBuffer, Value &outputBuffer,
                                  OpBuilder &builder, bool isSender) {
  if (!inputAllocOp) {
    return -1;
  }

  Location loc = builder.getUnknownLoc();

  inputBuffer = inputAllocOp->getResult(0);
  auto memRefType = dyn_cast<MemRefType>(inputBuffer.getType());
  if (!memRefType) {
    return -1;
  }

  int origBlockId = getBlockId(inputAllocOp);
  int outputBlockId = origBlockId;

  builder.setInsertionPointAfter(inputAllocOp);
  auto outputAlloc = builder.create<memref::AllocOp>(loc, memRefType);
  outputAlloc->setAttr(mlir::CVPipeline::kBlockId,
                       builder.getI32IntegerAttr(outputBlockId));
  outputAlloc->setAttr(mlir::CVPipeline::kTransferId,
                       builder.getI32IntegerAttr(tid));
  outputBuffer = outputAlloc.getResult();

  // NOTE: output alloc carries no ssbuffer.crossCoreDeps — alloc is a
  // buffer-creation op, not a behavior op. Producer tag lives on the
  // fixpipe/copy clone inside scf.if (set in wrapTransferOpWithScfIf*),
  // and consumer tag lives on the scf.if wrapper itself (set in
  // wrapReceiverChainWithScfIf). Do NOT re-introduce crossDeps here.

  auto outputMark = builder.create<annotation::MarkOp>(loc, outputBuffer);
  outputMark->setAttr("effects", builder.getStrArrayAttr({"write", "read"}));
  outputMark->setAttr(mlir::CVPipeline::kBlockId,
                      builder.getI32IntegerAttr(outputBlockId));
  outputMark->setAttr(mlir::CVPipeline::kTransferId,
                      builder.getI32IntegerAttr(tid));
  outputMark->setAttr(
      "hivm.tightly_coupled_buffer",
      hivm::HIVMTightlyCoupledBufferAttr::get(builder.getContext(), tcbId));
  LDBG("Created " << (isSender ? "sender" : "receiver")
                  << " output buffer: block_id=" << outputBlockId
                  << ", tcb_id=" << tcbId << ".");
  return 0;
}

static constexpr unsigned kBits32 = 32;

static int attachSsbufferTags(Operation *op, int blockId, int transferId) {
  MLIRContext *ctx = op->getContext();
  op->setAttr(mlir::CVPipeline::kBlockId,
              IntegerAttr::get(IntegerType::get(ctx, kBits32), blockId));
  op->setAttr(mlir::CVPipeline::kTransferId,
              IntegerAttr::get(IntegerType::get(ctx, kBits32), transferId));
  op->setAttr("ssbuffer.analyze_flag_id", UnitAttr::get(ctx));
  return 0;
}

static hivm::SyncBlockSetOp createOutputSyncSetOp(Operation *origSetOp,
                                                  int outputFlag, int tid,
                                                  OpBuilder &builder) {
  auto setOp = cast<hivm::SyncBlockSetOp>(origSetOp);
  builder.setInsertionPointAfter(origSetOp);
  auto newSetOp = builder.create<hivm::SyncBlockSetOp>(
      setOp.getLoc(), setOp.getTcoreType(), setOp.getTpipe(), setOp.getPipe(),
      builder.getI64IntegerAttr(outputFlag));
  attachSsbufferTags(newSetOp.getOperation(), getBlockId(setOp), tid);
  return newSetOp;
}

static hivm::SyncBlockWaitOp createOutputSyncWaitOp(Operation *origWaitOp,
                                                    int outputFlag, int tid,
                                                    OpBuilder &builder) {
  auto waitOp = cast<hivm::SyncBlockWaitOp>(origWaitOp);
  builder.setInsertionPointAfter(origWaitOp);
  auto newWaitOp = builder.create<hivm::SyncBlockWaitOp>(
      waitOp.getLoc(), waitOp.getTcoreType(), waitOp.getTpipe(),
      waitOp.getPipe(), builder.getI64IntegerAttr(outputFlag));
  attachSsbufferTags(newWaitOp.getOperation(), getBlockId(waitOp), tid);
  return newWaitOp;
}

/// Create output buffer for a single transfer group, with output flag sync ops
static int createOutputBufferForGroup(TransferGroupInfo &g,
                                      OpBuilder &builder) {
  if (createOutputBufferPair(g.senderBuf.allocOp, g.tid, g.tcbId,
                             g.senderInputBuffer, g.senderOutputBuffer, builder,
                             true)) {
    return -1;
  }
  if (createOutputBufferPair(g.receiverBuf.allocOp, g.tid, g.tcbId,
                             g.receiverInputBuffer, g.receiverOutputBuffer,
                             builder, false)) {
    return -1;
  }
  // Insert output sync set at extra_sync position
  if (g.extraSyncSetOp) {
    createOutputSyncSetOp(g.extraSyncSetOp, g.outputFlag, g.tid, builder);
    LDBG("Created output sync set with flag=" << g.outputFlag << " at block_id="
                                              << getBlockId(g.extraSyncSetOp)
                                              << " (sender scope).");
  }

  // Insert output sync wait at extra_sync position
  Operation *outputWaitInsertOp =
      g.extraSyncWaitOp ? g.extraSyncWaitOp : g.receiverChain.waitOp;
  if (outputWaitInsertOp) {
    createOutputSyncWaitOp(outputWaitInsertOp, g.outputFlag, g.tid, builder);
    LDBG("Created output sync wait with flag="
         << g.outputFlag << " at block_id=" << getBlockId(outputWaitInsertOp)
         << " (receiver scope).");
  }
  return 0;
}

/// Create output buffers for all transfer groups
static int createOutputBuffers(DenseMap<int, TransferGroupInfo> &groups,
                               ModuleOp module) {
  OpBuilder builder(module.getContext());
  std::set<int> usedTcbIds;

  // Collect existing tcb ids
  module.walk([&](Operation *op) {
    if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            "hivm.tightly_coupled_buffer")) {
      auto id = tcbAttr.getId();
      if (id.has_value()) {
        LDBG("Found mark op with tcb_id=" << id.value() << ".");
        usedTcbIds.insert(id.value());
      }
    }
  });

  LDBG("=== Step 2: Creating output buffers ===.");
  {
    std::string ids;
    llvm::raw_string_ostream os(ids);
    for (int id : usedTcbIds)
      os << id << " ";
    LDBG("Collected existing tcb_ids: " << ids << ".");
  }

  int maxExistingTcbId = usedTcbIds.empty() ? 0 : *usedTcbIds.rbegin();
  LDBG("Max existing tcb_id: " << maxExistingTcbId << ".");

  int nextTcbId = maxExistingTcbId + 1;

  for (auto &p : groups) {
    TransferGroupInfo &g = p.second;
    LDBG("Group tid=" << g.tid << " (" << (g.isCtoV ? "C→V" : "V→C") << ").");

    g.tcbId = allocateNewTcbId(nextTcbId, usedTcbIds);
    LDBG("Allocated tcb_id=" << g.tcbId << ".");

    nextTcbId = g.tcbId + 1;

    createOutputBufferForGroup(g, builder);
  }
  return 0;
}

/// Tag consumer-side alloc and transferOp with crossDeps marks
static int addConsumerCrossDepsTags(TransferGroupInfo &g, ModuleOp module) {
  auto &consumerBuf = g.receiverBuf;
  auto &consumerChain = g.receiverChain;

  OpBuilder builder(module.getContext());

  if (consumerBuf.allocOp) {
    consumerBuf.allocOp->setAttr(
        mlir::CVPipeline::kCrossCoreDeps,
        builder.getArrayAttr(
            {builder.getI32IntegerAttr(g.tid), builder.getI32IntegerAttr(1)}));
  }
  if (consumerChain.transferOp) {
    consumerChain.transferOp->setAttr(
        mlir::CVPipeline::kCrossCoreDeps,
        builder.getArrayAttr(
            {builder.getI32IntegerAttr(g.tid), builder.getI32IntegerAttr(0)}));
  }
  return 0;
}

// ============================================================================
// Step 3: Add polling control flow
// ============================================================================

/// Set ssbuffer tags on an op
static int setSsbufferTags(Operation *op, OpBuilder &builder, int blockId,
                           int tid) {
  op->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
  op->setAttr(mlir::CVPipeline::kTransferId, builder.getI32IntegerAttr(tid));
  return 0;
}

/// Ensure a WhileOp has an i32 iteration counter (polling: counter % 2 == 0).
/// - reuse an existing one (InnerScope-injected, kIterCounter on the while)
/// - normalize its body-end update: move to body head, re-tag first block id
static Value ensureWhileOpHasCounter(scf::WhileOp whileOp) {
  if (whileOp->hasAttr(CVPipeline::kIterCounter)) {
    Block &after = whileOp.getAfter().front();
    SmallVector<Operation *> updates;
    for (Operation &op : after)
      if (op.hasAttr(CVPipeline::kIterCounter))
        updates.push_back(&op);
    for (Operation *addi : updates) {
      Operation *firstOp = &after.front();
      if (addi == firstOp || addi == firstOp->getNextNode())
        continue; // already at head
      std::optional<int> firstId = getFirstBlockId(&after);
      if (!firstId)
        continue;
      addi->moveBefore(&after, after.begin());
      for (Value operand : addi->getOperands())
        if (Operation *defOp = operand.getDefiningOp())
          if (defOp->getBlock() == &after)
            defOp->moveBefore(addi);
      OpBuilder tagBuilder(whileOp.getContext());
      IntegerAttr blockIdAttr = tagBuilder.getI32IntegerAttr(*firstId);
      addi->setAttr(CVPipeline::kBlockId, blockIdAttr);
      for (Value operand : addi->getOperands())
        if (Operation *defOp = operand.getDefiningOp())
          defOp->setAttr(CVPipeline::kBlockId, blockIdAttr);
    }
    return after.getArgument(after.getNumArguments() - 1);
  }

  OpBuilder builder(whileOp);
  Location loc = whileOp.getLoc();
  auto oldWhile = whileOp;
  Type i32Type = builder.getI32Type();

  // Init counter = 0
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);

  SmallVector<Value> newInits(oldWhile.getInits());
  newInits.push_back(zero);
  SmallVector<Type> newResultTypes(oldWhile.getResultTypes());
  newResultTypes.push_back(i32Type);

  Value counterIterArg;
  Operation *counterOne = nullptr;
  Operation *counterAdd = nullptr;

  // Rebuild via the Builder callback API (matching InnerScope's
  // setupWhileIterArgCounter)
  auto newWhile = builder.create<scf::WhileOp>(
      loc, newResultTypes, newInits,
      [&](OpBuilder &bb, Location bl, ValueRange iterArgs) {
        Block *oldBefore = oldWhile.getBeforeBody();
        unsigned n = oldBefore->getNumArguments();
        IRMapping map;
        for (unsigned i = 0; i < n; ++i)
          map.map(oldBefore->getArgument(i), iterArgs[i]);

        for (Operation &op : oldBefore->without_terminator())
          bb.clone(op, map);

        auto oldCond = cast<scf::ConditionOp>(oldBefore->getTerminator());
        SmallVector<Value> condArgs;
        for (Value a : oldCond.getArgs())
          condArgs.push_back(map.lookupOrDefault(a));
        condArgs.push_back(iterArgs[n]); // counter
        bb.create<scf::ConditionOp>(
            bl, map.lookupOrDefault(oldCond.getCondition()), condArgs);
      },
      [&](OpBuilder &ab, Location al, ValueRange iterArgs) {
        Block *oldAfter = oldWhile.getAfterBody();
        unsigned n = oldAfter->getNumArguments();
        counterIterArg = iterArgs[n];
        // Counter update at body start; prepareLoopPolling builds the
        // polling chain around it (c2+remsi before, c0+cmpi after).
        counterOne = ab.create<arith::ConstantIntOp>(al, 1, kBits32);
        counterAdd = ab.create<arith::AddIOp>(al, counterIterArg,
                                              counterOne->getResult(0));
        Value nextCounter = counterAdd->getResult(0);
        IRMapping map;
        for (unsigned i = 0; i < n; ++i)
          map.map(oldAfter->getArgument(i), iterArgs[i]);

        for (Operation &op : oldAfter->without_terminator())
          ab.clone(op, map);

        auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
        SmallVector<Value> yOps;
        for (Value v : oldYield.getOperands())
          yOps.push_back(map.lookupOrDefault(v));
        yOps.push_back(nextCounter);
        ab.create<scf::YieldOp>(al, yOps);
      });

  // Copy attrs (must include ssbuffer.main_loop) and mark as processed
  for (auto attr : oldWhile->getAttrs())
    newWhile->setAttr(attr.getName(), attr.getValue());
  newWhile->setAttr(CVPipeline::kIterCounter, builder.getUnitAttr());

  // Tag counter-update ops with the first body block_id (CloneOps
  // contiguity, mirrors InnerScope); they carry no id yet, so the scan
  // picks the first old-body op's id.
  Block &afterBody = newWhile.getAfter().front();
  if (std::optional<int> firstBlockId = getFirstBlockId(&afterBody);
      firstBlockId && counterOne && counterAdd) {
    IntegerAttr blockIdAttr = builder.getI32IntegerAttr(*firstBlockId);
    counterOne->setAttr(CVPipeline::kBlockId, blockIdAttr);
    counterAdd->setAttr(CVPipeline::kBlockId, blockIdAttr);
    counterAdd->setAttr(CVPipeline::kIterCounter, builder.getUnitAttr());
  }

  // Replace results (exclude counter result)
  for (unsigned i = 0, e = oldWhile.getNumResults(); i < e; ++i)
    oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
  oldWhile.erase();

  return counterIterArg;
}

/// Create polling condition: (iter / step) % 2 == 0 (true=input, false=output)
static Value createPollingCondition(scf::ForOp forOp, OpBuilder &builder,
                                    int blockId, int tid) {
  Location loc = forOp.getLoc();
  Value iterVar = forOp.getInductionVar();
  Value step = forOp.getStep();

  auto divOp = builder.create<arith::DivSIOp>(loc, iterVar, step);
  setSsbufferTags(divOp.getOperation(), builder, blockId, tid);

  Type counterType = divOp.getResult().getType();
  int bitWidth = counterType.getIntOrFloatBitWidth();
  auto c2Val = builder.create<arith::ConstantIntOp>(loc, 2, bitWidth);
  setSsbufferTags(c2Val.getOperation(), builder, blockId, tid);
  auto remOp =
      builder.create<arith::RemSIOp>(loc, divOp.getResult(), c2Val.getResult());
  setSsbufferTags(remOp.getOperation(), builder, blockId, tid);

  auto c0Val = builder.create<arith::ConstantIntOp>(loc, 0, bitWidth);
  auto cmpOp = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, remOp.getResult(), c0Val.getResult());
  setSsbufferTags(cmpOp.getOperation(), builder, blockId, tid);
  setSsbufferTags(c0Val.getOperation(), builder, blockId, tid);

  return cmpOp.getResult();
}

/// Wrap a sync op (wait/set) in scf.if: then=clone original, else=create
/// alternative
template <typename OpTy>
static Operation *wrapSyncOpWithScfIf(
    Operation *op, Value cond, int outputFlag, OpBuilder &builder,
    std::function<Operation *(OpBuilder &, Location)> createAltFn) {
  static_assert(std::is_same<OpTy, hivm::SyncBlockWaitOp>::value ||
                    std::is_same<OpTy, hivm::SyncBlockSetOp>::value,
                "OpTy must be SyncBlockWaitOp or SyncBlockSetOp");

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);

  Location loc = op->getLoc();
  auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond,
                                        true /* withElseRegion */);
  ifOp->setAttr(mlir::CVPipeline::kBlockId,
                builder.getI32IntegerAttr(getBlockId(op)));
  ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));

  // then branch: clone original op
  auto thenBuilder = ifOp.getThenBodyBuilder();
  Operation *cloned = thenBuilder.clone(*op);

  // else branch: create alternative op
  auto elseBuilder = ifOp.getElseBodyBuilder();
  Operation *altOp = createAltFn(elseBuilder, loc);

  // Copy ssbuffer tags
  int bid = getBlockId(op);
  int tid = getTransferId(op);
  if (bid >= 0) {
    cloned->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(bid));
    altOp->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(bid));
  }
  if (tid >= 0) {
    cloned->setAttr(mlir::CVPipeline::kTransferId,
                    builder.getI32IntegerAttr(tid));
    altOp->setAttr(mlir::CVPipeline::kTransferId,
                   builder.getI32IntegerAttr(tid));
  }
  if (op->hasAttr("ssbuffer.analyze_flag_id")) {
    cloned->setAttr("ssbuffer.analyze_flag_id", builder.getUnitAttr());
    altOp->setAttr("ssbuffer.analyze_flag_id", builder.getUnitAttr());
  }

  op->replaceAllUsesWith(ifOp.getOperation());
  op->erase();
  return ifOp.getOperation();
}

/// Wrap transfer op (with external uses) in scf.if with yield:
/// - then: clone with the original buffer operand
/// - else: clone with bufferOperand remapped to the spare buffer
static Operation *wrapTransferOpWithScfIfYield(Operation *transferOp,
                                               Value cond, Value bufferOperand,
                                               Value outputBuffer, int bid,
                                               int tid, bool isProducer,
                                               OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(transferOp);

  Location loc = transferOp->getLoc();
  auto ifOp = builder.create<scf::IfOp>(loc, transferOp->getResultTypes(), cond,
                                        true /* withElseRegion */);

  // then branch: keep the original buffer operand
  Operation *thenCloned = nullptr;
  {
    auto thenBuilder = ifOp.getThenBodyBuilder();
    thenCloned = thenBuilder.clone(*transferOp);
    thenBuilder.create<scf::YieldOp>(loc, thenCloned->getResults());
  }

  // else branch: use outputBuffer
  Operation *elseCloned = nullptr;
  {
    auto elseBuilder = ifOp.getElseBodyBuilder();
    IRMapping outputMap;
    if (bufferOperand) {
      outputMap.map(bufferOperand, outputBuffer);
    }
    elseCloned = elseBuilder.clone(*transferOp, outputMap);
    elseBuilder.create<scf::YieldOp>(loc, elseCloned->getResults());
  }

  // Producer: tag the cloned transferOps (the actual behavior ops) with
  // [tid, 1]. The ifOp wrapper itself does NOT carry crossDeps because
  // scf.if is the polling-flow control structure, not the data-movement
  // behavior op — the producer role follows the inner fixpipe/copy clones.
  if (isProducer) {
    auto crossDeps = builder.getArrayAttr(
        {builder.getI32IntegerAttr(tid), builder.getI32IntegerAttr(1)});
    thenCloned->setAttr(mlir::CVPipeline::kCrossCoreDeps, crossDeps);
    elseCloned->setAttr(mlir::CVPipeline::kCrossCoreDeps, crossDeps);
  }

  // Tag the ifOp
  ifOp->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(bid));
  ifOp->setAttr(mlir::CVPipeline::kTransferId, builder.getI32IntegerAttr(tid));
  ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));

  // Replace all uses of the original transferOp
  for (auto [oldResult, newResult] :
       llvm::zip_equal(transferOp->getResults(), ifOp->getResults())) {
    oldResult.replaceAllUsesWith(newResult);
  }
  transferOp->erase();
  return ifOp.getOperation();
}

/// Wrap a transfer op (no external uses) in scf.if without yield
static Operation *wrapTransferOpWithScfIfSimple(Operation *transferOp,
                                                Value cond, Value bufferOperand,
                                                Value outputBuffer, int bid,
                                                int tid, bool isProducer,
                                                OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(transferOp);

  Location loc = transferOp->getLoc();
  auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond,
                                        true /* withElseRegion */);

  // then branch: clone directly (keeps the original buffer operand)
  Operation *thenCloned = nullptr;
  {
    auto thenBuilder = ifOp.getThenBodyBuilder();
    thenCloned = thenBuilder.clone(*transferOp);
  }

  // else branch: use outputBuffer
  Operation *elseCloned = nullptr;
  {
    auto elseBuilder = ifOp.getElseBodyBuilder();
    IRMapping outputMap;
    if (bufferOperand) {
      outputMap.map(bufferOperand, outputBuffer);
    }
    elseCloned = elseBuilder.clone(*transferOp, outputMap);
  }

  // Producer: tag the cloned transferOps (the actual behavior ops) with
  // [tid, 1]. Mirror of wrapTransferOpWithScfIfYield: behavior op, not
  // the ifOp wrapper, carries the producer role.
  if (isProducer) {
    auto crossDeps = builder.getArrayAttr(
        {builder.getI32IntegerAttr(tid), builder.getI32IntegerAttr(1)});
    thenCloned->setAttr(mlir::CVPipeline::kCrossCoreDeps, crossDeps);
    elseCloned->setAttr(mlir::CVPipeline::kCrossCoreDeps, crossDeps);
  }

  // Tag the ifOp
  ifOp->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(bid));
  ifOp->setAttr(mlir::CVPipeline::kTransferId, builder.getI32IntegerAttr(tid));
  ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));

  transferOp->erase();
  return ifOp.getOperation();
}

/// Wrap receiver chain (head + trailing ops + tensor boundary) in scf.if
/// returning tensor
static Operation *wrapReceiverChainWithScfIf(Operation *transferOp,
                                             Operation *toTensorOp, Value cond,
                                             Value bufferOperand,
                                             Value outputBuffer, int bid,
                                             int tid, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(transferOp);
  Location loc = transferOp->getLoc();

  // Collect the chain from transferOp to toTensorOp: ops whose result flows
  // into toTensorOp (e.g. memref.memory_space_cast between convert_layout and
  // bufferization.to_tensor for V→C transfers).
  SmallVector<Operation *> trailingOps;
  Value curVal = transferOp->getResult(0);
  while (curVal != toTensorOp->getOperand(0)) {
    bool found = false;
    for (auto &use : curVal.getUses()) {
      Operation *user = use.getOwner();
      if (user->isBeforeInBlock(toTensorOp) || user == toTensorOp) {
        curVal = user->getResult(0);
        if (user != toTensorOp)
          trailingOps.push_back(user);
        found = true;
        break;
      }
    }
    if (!found)
      break;
  }

  auto tensorType = toTensorOp->getResult(0).getType();
  auto ifOp = builder.create<scf::IfOp>(loc, tensorType, cond,
                                        true /* withElseRegion */);

  // then branch: keep the original buffer operand → clone chain + boundary
  {
    auto thenBuilder = ifOp.getThenBodyBuilder();
    IRMapping thenMapper;
    Operation *clonedTransfer = thenBuilder.clone(*transferOp, thenMapper);
    // Strip crossDeps from the cloned transferOp: clone() inherits attrs
    // from the original (which may carry [tid, 0] from upstream tagging),
    // but consumer role is owned by the ifOp wrapper below. Inner clone
    // must stay clean.
    clonedTransfer->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
    Value chainResult = clonedTransfer->getResult(0);
    thenMapper.map(transferOp->getResult(0), chainResult);
    for (Operation *op : trailingOps) {
      Operation *cloned = thenBuilder.clone(*op, thenMapper);
      cloned->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
      thenMapper.map(op->getResult(0), cloned->getResult(0));
    }
    Operation *clonedToTensor = thenBuilder.clone(*toTensorOp, thenMapper);
    clonedToTensor->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
    thenBuilder.create<scf::YieldOp>(loc, clonedToTensor->getResult(0));
  }

  // else branch: use outputBuffer → clone chain + boundary
  {
    auto elseBuilder = ifOp.getElseBodyBuilder();
    IRMapping outputMap;
    if (bufferOperand) {
      outputMap.map(bufferOperand, outputBuffer);
    }
    Operation *clonedTransfer = elseBuilder.clone(*transferOp, outputMap);
    clonedTransfer->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
    Value chainResult = clonedTransfer->getResult(0);
    auto elseMapper = outputMap;
    elseMapper.map(transferOp->getResult(0), chainResult);
    for (Operation *op : trailingOps) {
      Operation *cloned = elseBuilder.clone(*op, elseMapper);
      cloned->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
      elseMapper.map(op->getResult(0), cloned->getResult(0));
    }
    Operation *clonedToTensor = elseBuilder.clone(*toTensorOp, elseMapper);
    clonedToTensor->removeAttr(mlir::CVPipeline::kCrossCoreDeps);
    elseBuilder.create<scf::YieldOp>(loc, clonedToTensor->getResult(0));
  }

  // Tag the wrapper — single source of truth for consumer role
  ifOp->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(bid));
  ifOp->setAttr(mlir::CVPipeline::kTransferId, builder.getI32IntegerAttr(tid));
  ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));
  ifOp->setAttr(mlir::CVPipeline::kCrossCoreDeps,
                builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                      builder.getI32IntegerAttr(0)}));

  // Replace and erase from outermost to innermost to avoid use-after-free
  toTensorOp->getResult(0).replaceAllUsesWith(ifOp.getResult(0));
  toTensorOp->erase();
  for (Operation *op : llvm::reverse(trailingOps))
    op->erase();
  transferOp->erase();

  return ifOp.getOperation();
}

/// Process polling for a sender or receiver transfer chain
static LogicalResult processTransferChain(TransferOpChain &chain, Value cond,
                                          Value outputBuffer, int outputFlag,
                                          bool isProducer, OpBuilder &builder) {
  if (!chain.waitOp) {
    return failure();
  }

  Location loc = chain.waitOp->getLoc();

  // 1. Wrap waitOp in polling if
  chain.waitOp = wrapSyncOpWithScfIf<hivm::SyncBlockWaitOp>(
      chain.waitOp, cond, outputFlag, builder,
      [&](OpBuilder &b, Location l) -> Operation * {
        auto waitOp = cast<hivm::SyncBlockWaitOp>(chain.waitOp);
        return b
            .create<hivm::SyncBlockWaitOp>(l, waitOp.getTcoreType(),
                                           waitOp.getTpipe(), waitOp.getPipe(),
                                           b.getI64IntegerAttr(outputFlag))
            .getOperation();
      });

  // 2. Wrap transferOp in polling if (then=original buffer, else=outputBuffer)
  if (chain.transferOp) {
    int bid = getBlockId(chain.transferOp);
    int tid = getTransferId(chain.transferOp);

    // Receiver with boundary op: wrap the whole chain so scf.if yields tensor
    if (!isProducer && chain.toTensorOp) {
      LDBG("transferOp: " << chain.transferOp->getName()
                          << " (receiver, wrapping to_tensor).");
      chain.transferOp = wrapReceiverChainWithScfIf(
          chain.transferOp, chain.toTensorOp, cond, chain.bufferOperand,
          outputBuffer, bid, tid, builder);
      chain.toTensorOp = nullptr;
    } else {
      bool hasExternalUses = !chain.transferOp->getResults().empty() &&
                             !chain.transferOp->getResult(0).getUses().empty();

      LDBG("transferOp: " << chain.transferOp->getName()
                          << ", hasExternalUses=" << hasExternalUses << ".");

      chain.transferOp =
          hasExternalUses
              ? wrapTransferOpWithScfIfYield(chain.transferOp, cond,
                                             chain.bufferOperand, outputBuffer,
                                             bid, tid, isProducer, builder)
              : wrapTransferOpWithScfIfSimple(chain.transferOp, cond,
                                              chain.bufferOperand, outputBuffer,
                                              bid, tid, isProducer, builder);
    }
  }

  // 3. Wrap setOp in polling if
  if (chain.setOp) {
    chain.setOp = wrapSyncOpWithScfIf<hivm::SyncBlockSetOp>(
        chain.setOp, cond, outputFlag, builder,
        [&](OpBuilder &b, Location l) -> Operation * {
          auto setOp = cast<hivm::SyncBlockSetOp>(chain.setOp);
          return b
              .create<hivm::SyncBlockSetOp>(l, setOp.getTcoreType(),
                                            setOp.getTpipe(), setOp.getPipe(),
                                            b.getI64IntegerAttr(outputFlag))
              .getOperation();
        });
  }
  return success();
}

/// Create polling condition and builder for a loop op (ForOp or WhileOp).
/// Returns the condition Value; `builderOut` is set to the insertion point
/// for subsequent wrapping ops (before the loop terminator).
static Value prepareLoopPolling(Operation *loopOp, Operation *waitOp,
                                OpBuilder &builderOut) {
  int bid = getBlockId(waitOp);
  int tid = getTransferId(waitOp);

  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    // Build the polling cond at body start so it dominates every group's
    // scf.if wrapper (mirrors the while branch). tid comes from the anchor
    // wait; the block id from the first body op (CloneOps contiguity).
    builderOut.setInsertionPointToStart(forOp.getBody());
    std::optional<int> pollBlockId = getFirstBlockId(forOp.getBody());
    if (!pollBlockId)
      pollBlockId = bid;
    OpBuilder condBuilder(builderOut);
    Value cond = createPollingCondition(forOp, condBuilder, *pollBlockId, tid);
    return cond;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    // Guard: only main_loop whiles carry an injected counter; a plain
    // while here has none to poll on — fall back.
    if (!whileOp->hasAttr(CVPipeline::kIterCounter))
      return Value();
    // Polling cond: (counter % 2) == 0. The counter update sits at the
    // body head (created / normalized by ensure), so the chain is built
    // around it: c2+remsi before, c0+cmpi after.
    Block &after = whileOp.getAfter().front();
    Value counter = after.getArgument(after.getNumArguments() - 1);
    // Locate the counter update (top-level only; nested loops own theirs).
    Operation *lastCounterUpdate = nullptr;
    for (Operation &op : after)
      if (op.hasAttr(CVPipeline::kIterCounter))
        lastCounterUpdate = &op;
    bool counterAtHead =
        lastCounterUpdate && (lastCounterUpdate == &after.front() ||
                              lastCounterUpdate == after.front().getNextNode());
    // Insert at body start to dominate the wrapping scf.ifs.
    builderOut.setInsertionPointToStart(&after);
    // Tag the cond ops with the first body block_id (CloneOps contiguity).
    std::optional<int> pollBlockId = getFirstBlockId(&after);
    if (!pollBlockId)
      pollBlockId = bid;
    OpBuilder condBuilder(builderOut);
    Operation *c2Op =
        condBuilder.create<arith::ConstantIntOp>(whileOp.getLoc(), 2, kBits32);
    setSsbufferTags(c2Op, condBuilder, *pollBlockId, tid);
    Operation *remOp = condBuilder.create<arith::RemSIOp>(
        whileOp.getLoc(), counter, c2Op->getResult(0));
    setSsbufferTags(remOp, condBuilder, *pollBlockId, tid);
    // Put c0+cmpi right after the counter update:
    // - +1 lands between remsi and cmpi (validated backend ordering)
    // - otherwise chain stays contiguous at body start: the cond must
    //   dominate the wrapping scf.ifs
    if (counterAtHead)
      condBuilder.setInsertionPointAfter(lastCounterUpdate);
    Operation *c0Op =
        condBuilder.create<arith::ConstantIntOp>(whileOp.getLoc(), 0, kBits32);
    setSsbufferTags(c0Op, condBuilder, *pollBlockId, tid);
    Operation *condOp = condBuilder.create<arith::CmpIOp>(
        whileOp.getLoc(), arith::CmpIPredicate::eq, remOp->getResult(0),
        c0Op->getResult(0));
    setSsbufferTags(condOp, condBuilder, *pollBlockId, tid);
    return condOp->getResult(0);
  }

  // Unexpected loop type: caller falls back with ERRCODE_IGNORED.
  return Value();
}

/// Add polling control flow for all transfer groups
/// Get or create the polling condition shared by all groups rotating buffers
/// on one loop. The parity value ((iter%2)==0) is identical for every group,
/// so one scalar condition chain per loop suffices; anchorWait (the earliest
/// wait in the loop) guarantees the cond dominates every group's scf.if.
static Value getOrCreateLoopCond(Operation *loopOp, Operation *anchorWait,
                                 DenseMap<Operation *, Value> &condCache) {
  auto it = condCache.find(loopOp);
  if (it != condCache.end()) {
    return it->second;
  }
  OpBuilder builder(loopOp->getContext());
  Value cond = prepareLoopPolling(loopOp, anchorWait, builder);
  if (cond) {
    condCache[loopOp] = cond;
    LDBG("Shared polling cond created for loop, tagged by anchor wait.");
  }
  return cond;
}

/// Nearest ForOp/WhileOp ancestor of a sync op (sync may sit in scf.if).
/// A While nested in a For resolves to the While, not the For.
static Operation *resolveLoopOp(Operation *op) {
  return findAncestorOp(op, [](Operation *cur) {
    return isa<scf::ForOp>(cur) || isa<scf::WhileOp>(cur);
  });
}

static LogicalResult
addPollingControlFlow(DenseMap<int, TransferGroupInfo> &groups, int &errCode) {
  // Anchor per loop: the earliest wait across all groups — the shared cond
  // must dominate every group's scf.if wrappers.
  DenseMap<Operation *, Operation *> loopAnchor;
  for (auto &[tid, group] : groups) {
    auto track = [&](Operation *waitOp) {
      if (!waitOp) {
        return;
      }
      Operation *loop = resolveLoopOp(waitOp);
      Operation *anchor = loopAnchor.lookup(loop);
      if (!anchor || (waitOp->getBlock() == anchor->getBlock() &&
                      waitOp->isBeforeInBlock(anchor))) {
        loopAnchor[loop] = waitOp;
      }
    };
    track(group.senderChain.waitOp);
    track(group.receiverChain.waitOp);
  }

  DenseMap<Operation *, Value> condCache;
  for (auto &[tid, group] : groups) {
    // Get sender's loop op (ForOp or WhileOp)
    Operation *senderWaitParent = resolveLoopOp(group.senderChain.waitOp);

    // Shared polling condition for the sender loop
    OpBuilder senderBuilder(senderWaitParent->getContext());
    Value senderCond = getOrCreateLoopCond(
        senderWaitParent, loopAnchor[senderWaitParent], condCache);
    if (!senderCond) {
      LDBG("FALLBACK: unexpected sender loop op, tid="
           << tid << ", op=" << *senderWaitParent
           << ", rc=" << CVPipeline::ERRCODE_IGNORED << ".");
      errCode = CVPipeline::ERRCODE_IGNORED;
      return failure();
    }

    // Process sender chain (isProducer=true)
    if (failed(processTransferChain(group.senderChain, senderCond,
                                    group.senderOutputBuffer, group.outputFlag,
                                    true, senderBuilder))) {
      errCode = CVPipeline::ERRCODE_FAILED;
      return failure();
    }

    // Process receiver chain (may use different loop op) (isProducer=false)
    if (group.receiverChain.waitOp) {
      Operation *receiverWaitParent = resolveLoopOp(group.receiverChain.waitOp);

      if (receiverWaitParent == senderWaitParent) {
        // Use the same cond and builder
        if (failed(processTransferChain(
                group.receiverChain, senderCond, group.receiverOutputBuffer,
                group.outputFlag, false, senderBuilder))) {
          errCode = CVPipeline::ERRCODE_FAILED;
          return failure();
        }
      } else {
        // Receiver uses a different loop op, share its cond too
        OpBuilder receiverBuilder(receiverWaitParent->getContext());
        Value receiverCond = getOrCreateLoopCond(
            receiverWaitParent, loopAnchor[receiverWaitParent], condCache);
        if (!receiverCond) {
          LDBG("FALLBACK: unexpected receiver loop op, tid="
               << tid << ", op=" << *receiverWaitParent
               << ", rc=" << CVPipeline::ERRCODE_IGNORED << ".");
          errCode = CVPipeline::ERRCODE_IGNORED;
          return failure();
        }
        if (failed(processTransferChain(
                group.receiverChain, receiverCond, group.receiverOutputBuffer,
                group.outputFlag, false, receiverBuilder))) {
          errCode = CVPipeline::ERRCODE_FAILED;
          return failure();
        }
      }
    }
  }
  return success();
}

// ============================================================================
// Preprocessing: inject iteration counter into WhileOps with main_loop
// ============================================================================

/// Inject an i32 iteration counter loop-carried variable into every WhileOp
/// that has main_loop and contains transfer_id ops. Must run BEFORE Step 1 so
/// subsequent data collection sees the already-modified IR.
static void preInjectWhileOpToggles(ModuleOp module) {
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp whileOp) {
    if (!CVPipeline::isMainLoopOp(whileOp))
      return;
    bool hasTransferOps = false;
    whileOp.walk([&](Operation *op) {
      if (op->hasAttr(mlir::CVPipeline::kTransferId)) {
        hasTransferOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasTransferOps)
      whileOps.push_back(whileOp);
  });

  for (auto whileOp : whileOps)
    ensureWhileOpHasCounter(whileOp);

  LDBG("Preprocessed " << whileOps.size()
                       << " WhileOps with toggle injection.");
}

// ============================================================================
// Pass entry point
// ============================================================================

void AddMultiBufferOuterScopePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("============================================================");
  LDBG("Enter AddMultiBufferOuterScope pass.");
  LDBG("============================================================");

  // Determine buffer mode early; only inject toggle for double-buffer
  int interCoreBufNum = BufferCountManager(module).getBufferCountByType(
      BufferCountManager::DepType::InterCore);
  bool isDoubleBuf = (interCoreBufNum > 1);
  LDBG("[BufferCount] interCoreBufNum=" << interCoreBufNum
                                        << " doubleBuf=" << isDoubleBuf << ".");

  // Preprocessing: inject iteration counter into WhileOps before data
  // collection (only needed for double-buffer polling)
  if (isDoubleBuf) {
    preInjectWhileOpToggles(module);
  }

  // Step 1: Collect transfer group information
  LDBG("[Step 1/3] Start: transfer group collection.");
  FlagIdManager flagIdMgr(module);
  DenseMap<int, SmallVector<Operation *>> opsByTid;
  collectOpsByTransferId(module, opsByTid);
  DenseMap<int, TransferGroupInfo> groups;
  if (collectTransferGroupData(module, opsByTid, flagIdMgr, groups)) {
    LDBG("FALLBACK: Step 1/3 failed, no valid transfer groups found, rc="
         << CVPipeline::ERRCODE_FAILED << ".");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  LDBG("[Step 1/3] Done: " << groups.size() << " transfer groups.");

  // Flag ID budget:
  // 1. Usable ids 0..MAX_FLAG_ID (14); kReservedPipeFlagId (15) is reserved.
  // 2. Final max id = largest output flag acquired in Step 1.
  // 3. Exceed budget -> keep single-buffer mode.
  // 4. Input flag > kReservedPipeFlagId -> fallback rc=2.
  std::set<int> usedFlags;
  module.walk([&](Operation *op) {
    if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
      int f = getFlagFromSyncOp(op);
      if (f >= 0)
        usedFlags.insert(f);
    }
  });
  // Input flags:
  // 1. Flag id > kReservedPipeFlagId not producible by this pass.
  // 2. kReservedPipeFlagId (pipe) itself is allowed.
  bool inputOverBudget = false;
  for (int f : usedFlags) {
    if (f > kReservedPipeFlagId)
      inputOverBudget = true;
  }
  if (inputOverBudget) {
    LDBG("FALLBACK: FlagBudget, input flag id > "
         << kReservedPipeFlagId << ", rc=" << CVPipeline::ERRCODE_IGNORED
         << ".");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
  if (isDoubleBuf) {
    int maxOutputFlag = -1;
    for (auto &p : groups) {
      if (p.second.outputFlag > maxOutputFlag)
        maxOutputFlag = p.second.outputFlag;
    }
    LDBG("[FlagBudget] maxOutputFlag=" << maxOutputFlag
                                       << " (usable flag ids 0.."
                                       << FlagIdManager::MAX_FLAG_ID << ").");
    if (maxOutputFlag > FlagIdManager::MAX_FLAG_ID) {
      LDBG("FALLBACK: FlagBudget, estimated flag id "
           << maxOutputFlag << " exceeds usable range (0.."
           << FlagIdManager::MAX_FLAG_ID
           << "), fallback to single-buffer mode.");
      isDoubleBuf = false;
    }
  }

  if (isDoubleBuf) {
    // Tag llvm.load/store volatile ops with crossDeps
    DenseMap<int, SmallVector<Operation *>> loadStoreByTid;
    collectLoadStoreOpsByTransferId(module, loadStoreByTid);
    tagLoadStoreOpsWithCrossDeps(loadStoreByTid);
  }

  if (isDoubleBuf) {
    LDBG("[Step 2/3] Start: output buffer creation.");
    if (createOutputBuffers(groups, module)) {
      LDBG("FALLBACK: Step 2/3 failed, output buffer creation failed, rc="
           << CVPipeline::ERRCODE_FAILED << ".");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    LDBG("[Step 2/3] Done.");

    LDBG("[Step 3/3] Start: polling control flow.");
    int pollingRc = CVPipeline::ERRCODE_FAILED;
    if (failed(addPollingControlFlow(groups, pollingRc))) {
      LDBG("FALLBACK: Step 3/3 failed, polling control flow failed, rc="
           << pollingRc << ".");
      CVPipeline::setFallbackAttr(module, pollingRc);
      return;
    }
    LDBG("[Step 3/3] Done.");
  } else {
    LDBG("[Step 2-3] Skipped (single-buffer mode).");
  }

  LDBG("============================================================");
  LDBG("Exit AddMultiBufferOuterScope pass.");
  LDBG("============================================================");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass() {
  return std::make_unique<AddMultiBufferOuterScopePass>();
}

void AddMultiBufferOuterScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<mlir::annotation::AnnotationDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect>();
}

void registerAddMultiBufferOuterScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddMultiBufferOuterScopePass();
  });
}

} // namespace triton
} // namespace mlir
