#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferOuterScope";
#define LDBG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace triton;
using namespace hivm;

namespace mlir {
namespace triton {

// Maximum number of flag allocation attempts per transfer group
static constexpr int kMaxFlagAttempts = 16;

// --- Attribute helpers ---

static int getFlagFromSyncOp(Operation *op)
{
    if (auto attr = op->getAttrOfType<IntegerAttr>("flag_id")) { return attr.getInt(); }
    if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id")) { return attr.getInt(); }
    if (auto attr = op->getAttrOfType<IntegerAttr>("flag")) { return attr.getInt(); }
    return -1;
}

static int getBlockId(Operation *op)
{
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id"))
        return attr.getInt();
    return -1;
}

static int getTransferId(Operation *op)
{
    if (auto attr = op->getAttrOfType<IntegerAttr>("ssbuffer.transfer_id"))
        return attr.getInt();
    return -1;
}

// --- Address space helpers ---

static bool isInVectorScope(Operation *op)
{
    auto scopeOp = op->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) { return false; }
    if (auto tcoreAttr = scopeOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type"))
        return tcoreAttr.getTcoretype() == TCoreType::VECTOR;
    return false;
}

// --- main_loop attribute helpers ---

/// Check if forOp (or its terminator) has ssbuffer.main_loop attribute
static bool forOpHasMainLoopAttr(scf::ForOp forOp)
{
    if (forOp->hasAttr("ssbuffer.main_loop")) {
        return true;
    }
    Operation *terminator = forOp.getBody()->getTerminator();
    return terminator && terminator->hasAttr("ssbuffer.main_loop");
}

/// Check if a sync op's direct parent has ssbuffer.main_loop attribute
static bool parentOpHasMainLoopAttr(Operation *syncOp)
{
    if (!syncOp) { return false; }
    Operation *parent = syncOp->getParentOp();
    if (!parent) { return false; }
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        return forOpHasMainLoopAttr(forOp);
    }
    return false;
}

// --- Operation search helpers ---

/// Find sync op with a specific flag, searching forward or backward in a block
static Operation *findSyncOpWithFlag(Block *block, Operation *start, int flag, bool forward, bool wantWait)
{
    if (!block) { return nullptr; }
    auto it = start->getIterator();
    if (forward) {
        for (auto e = block->end(); it != e; ++it) {
            Operation *op = &*it;
            if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) { continue; }
            if (getFlagFromSyncOp(op) != flag) { continue; }
            if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) { return op; }
            if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) { return op; }
        }
    } else {
        if (it == block->begin()) { return nullptr; }
        do {
            --it;
            Operation *op = &*it;
            if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) { continue; }
            if (getFlagFromSyncOp(op) != flag) { continue; }
            if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) { return op; }
            if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) { return op; }
        } while (it != block->begin());
    }
    return nullptr;
}

/// Find the to_tensor op after a given op in the same block
static Operation *findToTensorAfter(Block *block, Operation *start)
{
    if (!block) { return nullptr; }
    auto it = start->getIterator();
    for (auto e = block->end(); it != e; ++it) {
        if (isa<bufferization::ToTensorOp>(&*it)) { return &*it; }
    }
    return nullptr;
}

// ============================================================================
// Step 1: Collect transfer group info
// ============================================================================

/// Collect all ops with ssbuffer.transfer_id, grouped by transfer_id
static int collectOpsByTransferId(ModuleOp module,
    DenseMap<int, SmallVector<Operation *>> &opsByTid)
{
    module.walk([&](Operation *op) {
        if (!op->hasAttr("ssbuffer.transfer_id")) { return; }
        int tid = getTransferId(op);
        if (tid >= 0) { opsByTid[tid].push_back(op); }
    });
    LDBG("Collected " << opsByTid.size() << " transfer groups");

    for (auto &p : opsByTid) {
        LDBG("  tid=" << p.first << " has " << p.second.size() << " ops");
        DenseMap<int, int> blockIdCount;
        for (auto *op : p.second) {
            int bid = getBlockId(op);
            blockIdCount[bid]++;
        }
        for (auto &bp : blockIdCount) {
            LDBG("    block_id=" << bp.first << ": " << bp.second << " ops");
        }
    }
    return 0;
}

/// Collect alloc/mark pairs (independent of block_id and main_loop)
static int collectBufferAllocs(const SmallVector<Operation *> &ops, BufferAllocInfo &info)
{
    SmallVector<Operation *> allocs;
    SmallVector<Operation *> marks;

    for (Operation *op : ops) {
        if (isa<memref::AllocOp>(op)) {
            allocs.push_back(op);
        } else if (isa<annotation::MarkOp>(op)) {
            marks.push_back(op);
        }
    }

    LDBG("collectBufferAllocs: allocs=" << allocs.size() << ", marks=" << marks.size());

    // Pair in order: sender first, receiver second
    if (!allocs.empty()) { info.sender.allocOp = allocs[0]; }
    if (allocs.size() > 1) { info.receiver.allocOp = allocs[1]; }
    if (!marks.empty()) { info.sender.markOp = marks[0]; }
    if (marks.size() > 1) { info.receiver.markOp = marks[1]; }

    return 0;
}

/// Collect extra sync ops (parent has no main_loop), paired by flag
static int collectExtraSync(const SmallVector<Operation *> &ops, int originalFlag, ExtraSyncInfo &info)
{
    SmallVector<Operation *> extraSets;
    SmallVector<Operation *> extraWaits;

    for (Operation *op : ops) {
        if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) { continue; }

        bool hasMainLoop = parentOpHasMainLoopAttr(op);
        LDBG("sync op: flag=" << getFlagFromSyncOp(op) << ", block_id=" << getBlockId(op)
                     << ", parentHasMainLoop=" << hasMainLoop);

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
        if (getFlagFromSyncOp(setOp) != originalFlag) { continue; }
        for (auto *waitOp : extraWaits) {
            if (getFlagFromSyncOp(waitOp) != originalFlag) { continue; }
            info.setOp = setOp;
            info.waitOp = waitOp;
            LDBG("Extra sync pair: set(flag=" << originalFlag
                         << ", block_id=" << getBlockId(setOp)
                         << "), wait(flag=" << originalFlag
                         << ", block_id=" << getBlockId(waitOp));
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

/// Collect transfer chain ops (parent has main_loop)
static int collectTransferChains(const SmallVector<Operation *> &ops,
    int originalFlag, TransferChainInfo &info)
{
    for (Operation *op : ops) {
        if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) || !op->getBlock()) { continue; }
        if (!parentOpHasMainLoopAttr(op)) { continue; }

        Block *block = op->getBlock();

        if (isa<hivm::FixpipeOp>(op)) {
            info.sender.transferOp = op;
            info.sender.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.sender.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Sender chain (CUBE): fixpipe, flag=" << originalFlag);
        } else if (isa<hivm::CopyOp>(op)) {
            info.sender.transferOp = op;
            info.sender.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.sender.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Sender chain (VECTOR): hir.copy, flag=" << originalFlag);
        } else if (isa<memref::MemorySpaceCastOp>(op) && isInVectorScope(op)) {
            info.receiver.transferOp = op;
            info.receiver.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.receiver.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            info.receiver.toTensorOp = findToTensorAfter(block, op);
            LDBG("Receiver chain (VECTOR): memory_space_cast, flag=" << originalFlag);
        } else if (isa<hivm::ConvertLayoutOp>(op)) {
            info.receiver.transferOp = op;
            info.receiver.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
            info.receiver.setOp = findSyncOpWithFlag(block, op, originalFlag, true, false);
            LDBG("Receiver chain (CUBE): convert_layout, flag=" << originalFlag);
        }
    }

    return 0;
}

/// Build TransferGroupInfo for a single transfer_id
static int buildTransferGroupData(int tid, const SmallVector<Operation *> &ops,
    FlagIdManager &flagIdMgr, TransferGroupInfo &info)
{
    info.tid = tid;

    LDBG("Building group tid=" << tid << ", ops=" << ops.size());

    // 1. Collect buffer alloc/mark pairs
    BufferAllocInfo bufInfo;
    if (collectBufferAllocs(ops, bufInfo)) { return -1; }
    info.senderBuf = bufInfo.sender;
    info.receiverBuf = bufInfo.receiver;
    LDBG("Sender buffer: " << (info.senderBuf.allocOp ? "alloc" : "none")
                 << " + " << (info.senderBuf.markOp ? "mark" : "none"));
    LDBG("Receiver buffer: " << (info.receiverBuf.allocOp ? "alloc" : "none")
                 << " + " << (info.receiverBuf.markOp ? "mark" : "none"));

    // 2. Determine original flag
    for (Operation *op : ops) {
        if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
            int f = getFlagFromSyncOp(op);
            if (f >= 0) {
                info.originalFlag = f;
                break;
            }
        }
    }

    // 3. Collect extra sync (parent has no main_loop)
    ExtraSyncInfo extraInfo;
    if (collectExtraSync(ops, info.originalFlag, extraInfo)) { return -1; }
    info.extraSyncSetOp = extraInfo.setOp;
    info.extraSyncWaitOp = extraInfo.waitOp;
    if (extraInfo.setOp && extraInfo.waitOp) {
        LDBG("Extra sync: set(block_id=" << getBlockId(extraInfo.setOp)
                     << "), wait(block_id=" << getBlockId(extraInfo.waitOp));
    } else {
        LDBG("Extra sync: not found");
    }

    // 4. Collect transfer chain (parent has main_loop)
    TransferChainInfo chainInfo;
    if (collectTransferChains(ops, info.originalFlag, chainInfo)) { return -1; }
    info.senderChain = chainInfo.sender;
    info.receiverChain = chainInfo.receiver;

    // 5. Determine direction
    if (info.senderChain.transferOp) {
        if (isa<hivm::FixpipeOp>(info.senderChain.transferOp)) {
            info.isCtoV = true;
        } else if (isa<hivm::CopyOp>(info.senderChain.transferOp)) {
            info.isCtoV = false;
        }
    }

    // For C→V transfer, sender uses receiver's buffer (the second alloc)
    if (info.isCtoV && info.senderBuf.allocOp && info.receiverBuf.allocOp) {
        LDBG("C→V transfer: swapping sender/receiver buffers");
        std::swap(info.senderBuf, info.receiverBuf);
    }

    // 6. Acquire output flag
    for (int attempt = 0; attempt < kMaxFlagAttempts; ++attempt) {
        int64_t pf = flagIdMgr.acquireId(nullptr);
        if (pf == FlagIdManager::INVALID_FLAG_ID) { break; }
        if (pf != info.originalFlag) {
            info.outputFlag = static_cast<int>(pf);
            break;
        }
    }

    if (info.senderChain.transferOp || info.receiverChain.transferOp) {
        LDBG("Direction: " << (info.isCtoV ? "C→V" : "V→C")
                     << ", flag=" << info.originalFlag << ", outputFlag=" << info.outputFlag);
    }

    return 0;
}

/// Collect TransferGroupInfo for all transfer groups
static int collectTransferGroupData(
    ModuleOp module,
    DenseMap<int, SmallVector<Operation *>> &opsByTid,
    FlagIdManager &flagIdMgr, DenseMap<int, TransferGroupInfo> &groups)
{
    for (auto &p : opsByTid) {
        TransferGroupInfo info;
        if (buildTransferGroupData(p.first, p.second, flagIdMgr, info)) { continue; }
        if ((info.senderChain.transferOp || info.receiverChain.transferOp)
            && info.outputFlag >= 0) {
            groups[p.first] = info;
        }
    }

    // Output flag reuse: groups with same (originalFlag, direction) share an output flag
    std::map<std::pair<int, bool>, int> outputFlagByKey;
    for (auto &p : groups) {
        auto &g = p.second;
        auto key = std::make_pair(g.originalFlag, g.isCtoV);
        auto it = outputFlagByKey.find(key);
        if (it != outputFlagByKey.end()) {
            g.outputFlag = it->second;
            LDBG("Group tid=" << g.tid << " reuses outputFlag=" << g.outputFlag
                 << " (shared originalFlag=" << g.originalFlag << ")");
        } else {
            outputFlagByKey[key] = g.outputFlag;
            LDBG("Group tid=" << g.tid << " gets new shared outputFlag=" << g.outputFlag
                 << " for originalFlag=" << g.originalFlag);
        }
    }

    LDBG("=== Step 1 Summary ===");
    LDBG("Transfer groups: " << groups.size());
    for (auto &p : groups) {
        LDBG("Group tid=" << p.first
                     << ", dir=" << (p.second.isCtoV ? "C→V" : "V→C")
                     << ", flag=" << p.second.originalFlag
                     << ", outputFlag=" << p.second.outputFlag);
        if (p.second.senderChain.transferOp)
            LDBG("  Sender: " << p.second.senderChain.transferOp->getName().getStringRef());
        if (p.second.receiverChain.transferOp)
            LDBG("  Receiver: " << p.second.receiverChain.transferOp->getName().getStringRef());
    }

    return 0;
}

// ============================================================================
// Step 2: Create output buffers
// ============================================================================

static constexpr int kMaxTcbSearch = 100;

static int allocateNewTcbId(int startFrom, std::set<int> &usedTcbIds)
{
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
    OpBuilder &builder, bool isSender)
{
    if (!inputAllocOp) { return -1; }

    Location loc = builder.getUnknownLoc();

    inputBuffer = inputAllocOp->getResult(0);
    auto memRefType = dyn_cast<MemRefType>(inputBuffer.getType());
    if (!memRefType) { return -1; }

    int origBlockId = getBlockId(inputAllocOp);
    int outputBlockId = origBlockId;

    builder.setInsertionPointAfter(inputAllocOp);
    auto outputAlloc = builder.create<memref::AllocOp>(loc, memRefType);
    outputAlloc->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(outputBlockId));
    outputAlloc->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    outputBuffer = outputAlloc.getResult();

    if (!isSender) {
        outputAlloc->setAttr("ssbuffer.crossDeps", builder.getArrayAttr({
            builder.getI32IntegerAttr(tid),
            builder.getI32IntegerAttr(1)
        }));
    }

    auto outputMark = builder.create<annotation::MarkOp>(loc, outputBuffer);
    outputMark->setAttr("effects", builder.getStrArrayAttr({"write", "read"}));
    outputMark->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(outputBlockId));
    outputMark->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    outputMark->setAttr("hivm.tightly_coupled_buffer", hivm::HIVMTightlyCoupledBufferAttr::get(builder.getContext(), tcbId));
    LDBG("Created " << (isSender ? "sender" : "receiver")
                 << " output buffer: block_id=" << outputBlockId << ", tcb_id=" << tcbId);
    return 0;
}

static constexpr unsigned kBits32 = 32;

static int attachSsbufferTags(Operation *op, int blockId, int transferId)
{
    MLIRContext* ctx = op->getContext();
    op->setAttr("ssbuffer.block_id", IntegerAttr::get(IntegerType::get(ctx, kBits32), blockId));
    op->setAttr("ssbuffer.transfer_id", IntegerAttr::get(IntegerType::get(ctx, kBits32), transferId));
    return 0;
}

static hivm::SyncBlockSetOp createOutputSyncSetOp(Operation *origSetOp, int outputFlag, int tid, OpBuilder &builder)
{
    auto setOp = cast<hivm::SyncBlockSetOp>(origSetOp);
    builder.setInsertionPointAfter(origSetOp);
    auto newSetOp = builder.create<hivm::SyncBlockSetOp>(
        setOp.getLoc(), setOp.getTcoreType(), setOp.getTpipe(), setOp.getPipe(),
        builder.getI64IntegerAttr(outputFlag));
    attachSsbufferTags(newSetOp.getOperation(), getBlockId(setOp), tid);
    return newSetOp;
}

static hivm::SyncBlockWaitOp createOutputSyncWaitOp(Operation *origWaitOp, int outputFlag, int tid, OpBuilder &builder)
{
    auto waitOp = cast<hivm::SyncBlockWaitOp>(origWaitOp);
    builder.setInsertionPointAfter(origWaitOp);
    auto newWaitOp = builder.create<hivm::SyncBlockWaitOp>(
        waitOp.getLoc(), waitOp.getTcoreType(), waitOp.getTpipe(), waitOp.getPipe(),
        builder.getI64IntegerAttr(outputFlag));
    attachSsbufferTags(newWaitOp.getOperation(), getBlockId(waitOp), tid);
    return newWaitOp;
}

/// Create output buffer for a single transfer group, with output flag sync ops
static int createOutputBufferForGroup(TransferGroupInfo &g, OpBuilder &builder)
{
    if (createOutputBufferPair(g.senderBuf.allocOp, g.tid, g.tcbId,
                               g.senderInputBuffer, g.senderOutputBuffer, builder, true)) {
        return -1;
    }
    if (createOutputBufferPair(g.receiverBuf.allocOp, g.tid, g.tcbId,
                               g.receiverInputBuffer, g.receiverOutputBuffer, builder, false)) {
        return -1;
    }
    // Insert output sync set at extra_sync position
    if (g.extraSyncSetOp) {
        createOutputSyncSetOp(g.extraSyncSetOp, g.outputFlag, g.tid, builder);
        LDBG("Created output sync set with flag=" << g.outputFlag
                     << " at block_id=" << getBlockId(g.extraSyncSetOp) << " (sender scope)");
    }

    // Insert output sync wait at extra_sync position
    Operation *outputWaitInsertOp = g.extraSyncWaitOp ? g.extraSyncWaitOp : g.receiverChain.waitOp;
    if (outputWaitInsertOp) {
        createOutputSyncWaitOp(outputWaitInsertOp, g.outputFlag, g.tid, builder);
        LDBG("Created output sync wait with flag=" << g.outputFlag
                     << " at block_id=" << getBlockId(outputWaitInsertOp) << " (receiver scope)");
    }
    return 0;
}

/// Create output buffers for all transfer groups
static int createOutputBuffers(DenseMap<int, TransferGroupInfo> &groups, ModuleOp module)
{
    OpBuilder builder(module.getContext());
    std::set<int> usedTcbIds;

    // Collect existing tcb ids
    module.walk([&](Operation *op) {
        if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>("hivm.tightly_coupled_buffer")) {
            auto id = tcbAttr.getId();
            if (id.has_value()) {
                LDBG("Found mark op with tcb_id=" << id.value());
                usedTcbIds.insert(id.value());
            }
        }
    });

    LDBG("=== Step 2: Creating output buffers ===");
    {
        std::string ids;
        llvm::raw_string_ostream os(ids);
        for (int id : usedTcbIds) os << id << " ";
        LDBG("Collected existing tcb_ids: " << ids);
    }

    int maxExistingTcbId = usedTcbIds.empty() ? 0 : *usedTcbIds.rbegin();
    LDBG("Max existing tcb_id: " << maxExistingTcbId);

    int nextTcbId = maxExistingTcbId + 1;

    for (auto &p : groups) {
        TransferGroupInfo &g = p.second;
        LDBG("Group tid=" << g.tid << " (" << (g.isCtoV ? "C→V" : "V→C") << ")");

        g.tcbId = allocateNewTcbId(nextTcbId, usedTcbIds);
        LDBG("Allocated tcb_id=" << g.tcbId);

        nextTcbId = g.tcbId + 1;

        createOutputBufferForGroup(g, builder);
    }
    return 0;
}

/// Tag consumer-side alloc and transferOp with crossDeps marks
static int addConsumerCrossDepsTags(TransferGroupInfo &g, ModuleOp module)
{
    bool consumerIsVector = g.isCtoV;
    auto &consumerBuf = consumerIsVector ? g.receiverBuf : g.senderBuf;
    auto &consumerChain = consumerIsVector ? g.receiverChain : g.senderChain;

    OpBuilder builder(module.getContext());

    if (consumerBuf.allocOp) {
        consumerBuf.allocOp->setAttr("ssbuffer.crossDeps", builder.getArrayAttr({
            builder.getI32IntegerAttr(g.tid),
            builder.getI32IntegerAttr(1)
        }));
    }
    if (consumerChain.transferOp) {
        consumerChain.transferOp->setAttr("ssbuffer.crossDeps", builder.getArrayAttr({
            builder.getI32IntegerAttr(g.tid),
            builder.getI32IntegerAttr(0)
        }));
    }
    return 0;
}

// ============================================================================
// Step 3: Add polling control flow
// ============================================================================

/// Set ssbuffer tags on an op
static int setSsbufferTags(Operation *op, OpBuilder &builder, int blockId, int tid)
{
    op->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(blockId));
    op->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    return 0;
}

/// Create polling condition: (iter / step) % 2 == 0 (true=input, false=output)
static Value createPollingCondition(scf::ForOp forOp, OpBuilder &builder, int blockId, int tid)
{
    Location loc = forOp.getLoc();
    Value iterVar = forOp.getInductionVar();
    Value step = forOp.getStep();

    auto divOp = builder.create<arith::DivSIOp>(loc, iterVar, step);
    setSsbufferTags(divOp.getOperation(), builder, blockId, tid);

    Type counterType = divOp.getResult().getType();
    int bitWidth = counterType.getIntOrFloatBitWidth();
    auto c2Val = builder.create<arith::ConstantIntOp>(loc, 2, bitWidth);
    setSsbufferTags(c2Val.getOperation(), builder, blockId, tid);
    auto remOp = builder.create<arith::RemSIOp>(loc, divOp.getResult(), c2Val.getResult());
    setSsbufferTags(remOp.getOperation(), builder, blockId, tid);

    auto c0Val = builder.create<arith::ConstantIntOp>(loc, 0, bitWidth);
    auto cmpOp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remOp.getResult(), c0Val.getResult());
    setSsbufferTags(cmpOp.getOperation(), builder, blockId, tid);
    setSsbufferTags(c0Val.getOperation(), builder, blockId, tid);

    return cmpOp.getResult();
}

/// Wrap a sync op (wait/set) in scf.if: then=clone original, else=create alternative
template <typename OpTy>
static Operation *wrapSyncOpWithScfIf(Operation *op, Value cond, int outputFlag,
    OpBuilder &builder, std::function<Operation*(OpBuilder&, Location)> createAltFn)
{
    static_assert(std::is_same<OpTy, hivm::SyncBlockWaitOp>::value ||
                  std::is_same<OpTy, hivm::SyncBlockSetOp>::value,
                  "OpTy must be SyncBlockWaitOp or SyncBlockSetOp");

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);

    Location loc = op->getLoc();
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, true /* withElseRegion */);
    ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(getBlockId(op)));
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
        cloned->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(bid));
        altOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(bid));
    }
    if (tid >= 0) {
        cloned->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
        altOp->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    }

    op->replaceAllUsesWith(ifOp.getOperation());
    op->erase();
    return ifOp.getOperation();
}

/// Wrap a transfer op (with external uses) in scf.if with yield
static Operation *wrapTransferOpWithScfIfYield(Operation *transferOp, Value cond,
    Value inputBuffer, Value outputBuffer,
    int bid, int tid, bool isProducer, OpBuilder &builder)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(transferOp);

    Location loc = transferOp->getLoc();
    auto ifOp = builder.create<scf::IfOp>(loc, transferOp->getResultTypes(), cond, true /* withElseRegion */);

    // then branch: use inputBuffer
    {
        auto thenBuilder = ifOp.getThenBodyBuilder();
        IRMapping inputMap;
        if (transferOp->getNumOperands() > 0) {
            inputMap.map(transferOp->getOperand(transferOp->getNumOperands() - 1), inputBuffer);
        }
        Operation *cloned = thenBuilder.clone(*transferOp, inputMap);
        thenBuilder.create<scf::YieldOp>(loc, cloned->getResults());
    }

    // else branch: use outputBuffer
    {
        auto elseBuilder = ifOp.getElseBodyBuilder();
        IRMapping outputMap;
        if (transferOp->getNumOperands() > 0) {
            outputMap.map(transferOp->getOperand(transferOp->getNumOperands() - 1), outputBuffer);
        }
        Operation *cloned = elseBuilder.clone(*transferOp, outputMap);
        elseBuilder.create<scf::YieldOp>(loc, cloned->getResults());
    }

    // Tag the ifOp
    ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(bid));
    ifOp->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));
    if (!isProducer) {
        ifOp->setAttr("ssbuffer.crossDeps", builder.getArrayAttr({
            builder.getI32IntegerAttr(tid),
            builder.getI32IntegerAttr(0)
        }));
    }

    // Replace all uses of the original transferOp
    for (auto [oldResult, newResult] : llvm::zip_equal(transferOp->getResults(), ifOp->getResults())) {
        oldResult.replaceAllUsesWith(newResult);
    }
    transferOp->erase();
    return ifOp.getOperation();
}

/// Wrap a transfer op (no external uses) in scf.if without yield
static Operation *wrapTransferOpWithScfIfSimple(Operation *transferOp, Value cond,
    Value inputBuffer, Value outputBuffer, int bid, int tid, bool isProducer, OpBuilder &builder)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(transferOp);

    Location loc = transferOp->getLoc();
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, true /* withElseRegion */);

    // then branch: clone directly
    {
        auto thenBuilder = ifOp.getThenBodyBuilder();
        thenBuilder.clone(*transferOp);
    }

    // else branch: use outputBuffer
    {
        auto elseBuilder = ifOp.getElseBodyBuilder();
        IRMapping outputMap;
        if (transferOp->getNumOperands() > 0) {
            outputMap.map(transferOp->getOperand(transferOp->getNumOperands() - 1), outputBuffer);
        }
        elseBuilder.clone(*transferOp, outputMap);
    }

    // Tag the ifOp
    ifOp->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(bid));
    ifOp->setAttr("ssbuffer.transfer_id", builder.getI32IntegerAttr(tid));
    ifOp->setAttr("ssbuffer.cross_buffer", builder.getI32IntegerAttr(1));
    if (!isProducer) {
        ifOp->setAttr("ssbuffer.crossDeps", builder.getArrayAttr({
            builder.getI32IntegerAttr(tid),
            builder.getI32IntegerAttr(0)
        }));
    }

    transferOp->erase();
    return ifOp.getOperation();
}

/// Process polling for a sender or receiver transfer chain
static int processTransferChain(TransferOpChain &chain, Value cond,
    Value inputBuffer, Value outputBuffer,
    int outputFlag, bool isProducer, OpBuilder &builder)
{
    if (!chain.waitOp) { return -1; }

    Location loc = chain.waitOp->getLoc();

    // 1. Wrap waitOp in polling if
    chain.waitOp = wrapSyncOpWithScfIf<hivm::SyncBlockWaitOp>(
        chain.waitOp, cond, outputFlag, builder,
        [&](OpBuilder &b, Location l) -> Operation* {
            auto waitOp = cast<hivm::SyncBlockWaitOp>(chain.waitOp);
            return b.create<hivm::SyncBlockWaitOp>(l, waitOp.getTcoreType(),
                                                    waitOp.getTpipe(), waitOp.getPipe(),
                                                    b.getI64IntegerAttr(outputFlag)).getOperation();
        });

    // 2. Wrap transferOp in polling if (then=use inputBuffer, else=use outputBuffer)
    if (chain.transferOp) {
        int bid = getBlockId(chain.transferOp);
        int tid = getTransferId(chain.transferOp);
        bool hasExternalUses = !chain.transferOp->getResults().empty() &&
                               !chain.transferOp->getResult(0).getUses().empty();

        LDBG("transferOp: " << chain.transferOp->getName()
                     << ", hasExternalUses=" << hasExternalUses);

        chain.transferOp = hasExternalUses
            ? wrapTransferOpWithScfIfYield(chain.transferOp, cond, inputBuffer, outputBuffer, bid, tid, isProducer, builder)
            : wrapTransferOpWithScfIfSimple(chain.transferOp, cond, inputBuffer, outputBuffer, bid, tid, isProducer, builder);
    }

    // 3. Wrap setOp in polling if
    if (chain.setOp) {
        chain.setOp = wrapSyncOpWithScfIf<hivm::SyncBlockSetOp>(
            chain.setOp, cond, outputFlag, builder,
            [&](OpBuilder &b, Location l) -> Operation* {
                auto setOp = cast<hivm::SyncBlockSetOp>(chain.setOp);
                return b.create<hivm::SyncBlockSetOp>(l, setOp.getTcoreType(),
                                                      setOp.getTpipe(), setOp.getPipe(),
                                                      b.getI64IntegerAttr(outputFlag)).getOperation();
            });
    }
    return 0;
}

/// Add polling control flow for all transfer groups
static int addPollingControlFlow(DenseMap<int, TransferGroupInfo> &groups)
{
    for (auto &p : groups) {
        TransferGroupInfo &g = p.second;

        // Get sender's scf.for
        Operation *senderWaitParent = g.senderChain.waitOp->getParentOp();
        scf::ForOp senderForOp = cast<scf::ForOp>(senderWaitParent);

        int senderBid = getBlockId(g.senderChain.waitOp);
        int senderTid = getTransferId(g.senderChain.waitOp);

        // Insert polling condition at sender waitOp's position
        OpBuilder senderCondBuilderForInsert(senderForOp.getBody(), Block::iterator(g.senderChain.waitOp));
        Value senderCond = createPollingCondition(senderForOp, senderCondBuilderForInsert, senderBid, senderTid);
        OpBuilder senderBuilder(senderForOp.getBody()->getTerminator());

        // Process sender chain (isProducer=true)
        if (processTransferChain(g.senderChain, senderCond,
                                 g.senderInputBuffer, g.senderOutputBuffer,
                                 g.outputFlag, true, senderBuilder) != 0) {
            return -1;
        }

        // Process receiver chain (may use different scf.for) (isProducer=false)
        if (g.receiverChain.waitOp) {
            Operation *receiverWaitParent = g.receiverChain.waitOp->getParentOp();

            if (receiverWaitParent == senderWaitParent) {
                // Use the same cond
                if (processTransferChain(g.receiverChain, senderCond,
                                         g.receiverInputBuffer, g.receiverOutputBuffer,
                                         g.outputFlag, false, senderBuilder) != 0) {
                    return -1;
                }
            } else {
                // Receiver uses a different scf.for, create new cond
                scf::ForOp receiverForOp = cast<scf::ForOp>(receiverWaitParent);
                int receiverBid = getBlockId(g.receiverChain.waitOp);
                int receiverTid = getTransferId(g.receiverChain.waitOp);
                OpBuilder receiverCondBuilderForInsert(receiverForOp.getBody(), Block::iterator(g.receiverChain.waitOp));
                Value receiverCond = createPollingCondition(receiverForOp, receiverCondBuilderForInsert, receiverBid, receiverTid);
                OpBuilder receiverBuilder(receiverForOp.getBody()->getTerminator());
                if (processTransferChain(g.receiverChain, receiverCond,
                                         g.receiverInputBuffer, g.receiverOutputBuffer,
                                         g.outputFlag, false, receiverBuilder) != 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

// ============================================================================
// Pass entry point
// ============================================================================

void AddMultiBufferOuterScopePass::runOnOperation()
{
    ModuleOp module = getOperation();
    LDBG("============================================================");
    LDBG("[AddMultiBufferOuterScope] ENTER");
    LDBG("============================================================");

    // Step 1: Collect transfer group information
    LDBG("[Step 1/3] Start: transfer group collection");
    FlagIdManager flagIdMgr(module);
    DenseMap<int, SmallVector<Operation *>> opsByTid;
    collectOpsByTransferId(module, opsByTid);
    DenseMap<int, TransferGroupInfo> groups;
    if (collectTransferGroupData(module, opsByTid, flagIdMgr, groups)) {
        LDBG("[Step 1/3] FAILED: no valid transfer groups found");
        signalPassFailure();
        return;
    }
    LDBG("[Step 1/3] Done: " << groups.size() << " transfer groups");

    int interCoreBufNum = BufferCountManager::getInstance()
        .getBufferCountByType(BufferCountManager::DepType::InterCore);
    bool isDoubleBuf = (interCoreBufNum > 1);
    LDBG("[BufferCount] interCoreBufNum=" << interCoreBufNum << " doubleBuf=" << isDoubleBuf);

    // Tag consumer-side alloc and transferOp with crossDeps (both modes)
    for (auto &p : groups)
        addConsumerCrossDepsTags(p.second, module);

    if (isDoubleBuf) {
        LDBG("[Step 2/3] Start: output buffer creation");
        if (createOutputBuffers(groups, module)) {
            LDBG("[Step 2/3] FAILED: output buffer creation failed");
            signalPassFailure();
            return;
        }
        LDBG("[Step 2/3] Done");

        LDBG("[Step 3/3] Start: polling control flow");
        if (addPollingControlFlow(groups)) {
            LDBG("[Step 3/3] FAILED: polling control flow failed");
            signalPassFailure();
            return;
        }
        LDBG("[Step 3/3] Done");
    } else {
        LDBG("[Step 2-3] Skipped (single-buffer mode)");
    }

    LDBG("============================================================");
    LDBG("[AddMultiBufferOuterScope] EXIT successfully");
    LDBG("============================================================");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass()
{
    return std::make_unique<AddMultiBufferOuterScopePass>();
}

void AddMultiBufferOuterScopePass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<mlir::annotation::AnnotationDialect,
                    mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect,
                    mlir::hivm::HIVMDialect,
                    mlir::scope::ScopeDialect>();
}

void registerAddMultiBufferOuterScopePasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createAddMultiBufferOuterScopePass(); });
}

} // namespace triton
} // namespace mlir
