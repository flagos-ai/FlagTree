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

// Memory side-effect analysis: tracks per-location read/write effects across
// ops to identify implicit ordering constraints (RAW, WAR, WAW) and expose
// them as scheduling edges (memDefs/memUsers, execBefore/execAfter).
//
// Each memory location is a MemSlot recording its last writer and pending
// readers. Two-level alias resolution determines which slots an effect touches:
//   1. Exact: value V found directly in valueToSlot.
//   2. AA query: AliasAnalysis asked whether a slot’s memref may-alias V.
//      Root BlockArguments reject MayAlias (rejectMayAlias=true) to avoid
//      false positives from conservative AA on function parameters.
//
// Unknown ops (no SideEffect interface) act as full barriers: they depend on
// all prior writers/readers and become the sole writer for every slot.

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "memory-effects-tracker";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::CVPipeline;

namespace {

bool isDefinedInside(Value v, Operation *op)
{
    if (!v || !op) {
        return false;
    }

    if (auto arg = dyn_cast<BlockArgument>(v)) {
        Operation *parent = arg.getOwner()->getParentOp();
        return parent && op->isAncestor(parent);
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        return false;
    }

    return op->isProperAncestor(defOp);
}

} // namespace

MemoryDependenceGraph::MemoryDependenceGraph(Operation *root, AliasAnalysis &aa) : root(root), aa(aa)
{
    if (!root) {
        return;
    }
    analyzeOp(root);
    slots.clear();
    valueToSlot.clear();
}

ArrayRef<Operation *> MemoryDependenceGraph::getMemDefs(Operation *op) const
{
    auto it = memDefs.find(op);
    if (it == memDefs.end()) {
        return {};
    }
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getMemUsers(Operation *op) const
{
    auto it = memUsers.find(op);
    if (it == memUsers.end())
        return {};
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getExecBefore(Operation *op) const
{
    auto it = execBefore.find(op);
    if (it == execBefore.end())
        return {};
    return it->second;
}

ArrayRef<Operation *> MemoryDependenceGraph::getExecAfter(Operation *op) const
{
    auto it = execAfter.find(op);
    if (it == execAfter.end())
        return {};
    return it->second;
}

void MemoryDependenceGraph::analyzeOp(Operation *op)
{
    if (!op) {
        return;
    }

    // The graph root is treated as a pure container: we don't query its own
    // effects against itself, only recurse into its regions.
    if (op == root) {
        analyzeRegionsOf(op);
        return;
    }

    LOG_DEBUG("Analyzing op " << *op << "\n");
    // Step 1: collect outside memory effects
    bool unknown = false;
    SmallVector<MemoryEffects::EffectInstance> effects = collectOuterEffects(op, unknown);

    // Step 2: collect defs(WAR) and preds(WAR, WAW, RAW) dependency.
    SmallVector<Operation *> defs;
    SmallVector<Operation *> preds;
    collectPreds(effects, unknown, defs, preds);
    LOG_DEBUG("Defs: \n");
    for (auto def: defs) {
        LOG_DEBUG(*def << "\n");
    }
    LOG_DEBUG("Preds: \n");
    for (auto pred: preds) {
        LOG_DEBUG(*pred << "\n");
    }

    // Step 3: extract edges from defs and preds to the graph.
    recordEdges(op, defs, preds);

    // Step 4: recursively search regions.
    if (op->getNumRegions() > 0) {
        analyzeRegionsOf(op);
    }

    // Step 5: store effects into slots.
    applyEffects(op, effects, unknown);
}

void MemoryDependenceGraph::analyzeRegionsOf(Operation *op)
{
    // Whether to isolated slots.
    // For example: scf.for should keeps slots; funOp should use new slots.
    const bool isolated = op != root && op->hasTrait<OpTrait::IsIsolatedFromAbove>();

    for (Region &region : op->getRegions()) {
        // Snapshot is used for quickly restoration.
        Snapshot snap = takeSnapshot();
        if (isolated) {
            slots.clear();
            valueToSlot.clear();
        }

        for (Block &block : region) {
            for (BlockArgument arg : block.getArguments()) {
                if (isa<BaseMemRefType>(arg.getType())) {
                    auto *slot = getOrCreateSlot(arg);
                }
            }
            for (Operation &inner : block) {
                analyzeOp(&inner);
            }
        }

        restoreSnapshot(std::move(snap));
    }
}

SmallVector<MemoryEffects::EffectInstance> MemoryDependenceGraph::collectOuterEffects(Operation *op, bool &unknown)
{
    unknown = false;

    std::optional<SmallVector<MemoryEffects::EffectInstance>> raw = getEffectsRecursively(op);
    if (!raw) {
        unknown = true;
        return {};
    }

    SmallVector<MemoryEffects::EffectInstance> filtered;
    filtered.reserve(raw->size());
    for (auto &e : *raw) {
        if (isDefinedInside(e.getValue(), op)) {
            continue;
        }
        filtered.push_back(e);
    }
    return filtered;
}

AliasResult MemoryDependenceGraph::queryAlias(Value lhs, Value rhs)
{
    auto isFuncEntryArg = [] (const Value &val) -> bool {
        auto arg = llvm::dyn_cast<BlockArgument>(val);
        return arg && arg.getOwner()->isEntryBlock();
    };
    auto getSource = [] (Value val) -> Value {
        while (auto viewLike = val.getDefiningOp<ViewLikeOpInterface>()) {
            val = viewLike.getViewSource();
        }
        return val;
    };
    if (isFuncEntryArg(getSource(lhs)) && isFuncEntryArg(getSource(rhs))) {
        return lhs == rhs ? AliasResult::MustAlias : AliasResult::NoAlias;
    }
    return aa.alias(lhs, rhs);
}

SmallVector<MemoryDependenceGraph::MemSlot *> MemoryDependenceGraph::findAliasSlots(Value v)
{
    SmallVector<MemSlot *> result;

    if (!v) { // unknown location: conservatively alias with every slot
        result.reserve(slots.size());
        for (auto &slot : slots) {
            result.push_back(slot.get());
        }
        return result;
    }

    SmallPtrSet<MemSlot *, INIT_SIZE> seen;

    auto it = valueToSlot.find(v);
    if (it != valueToSlot.end()) {
        result.push_back(it->second);
        seen.insert(it->second);
    }

    for (auto &slot : slots) {
        MemSlot *raw = slot.get();
        if (seen.contains(raw) || !raw->memref) {
            continue;
        }
        auto aliasResult = queryAlias(raw->memref, v);
        if (aliasResult != AliasResult::NoAlias) {
            result.push_back(raw);
            seen.insert(raw);
        }
    }

    return result;
}

MemoryDependenceGraph::MemSlot *MemoryDependenceGraph::getOrCreateSlot(Value v)
{
    if (!v) {
        return nullptr;
    }
    auto it = valueToSlot.find(v);
    if (it != valueToSlot.end()) {
        return it->second;
    }
    auto slot = std::make_unique<MemSlot>(v);
    MemSlot *raw = slot.get();
    slots.push_back(std::move(slot));
    valueToSlot[v] = raw;
    return raw;
}

ArrayRef<MemoryDependenceGraph::MemSlot *> MemoryDependenceGraph::resolveAliasSlots(
    Value v, DenseMap<Value, SmallVector<MemSlot *>> &cache)
{
    auto it = cache.find(v);
    if (it != cache.end()) {
        return it->second;
    }
    auto [ins, _] = cache.try_emplace(v, findAliasSlots(v));
    return ins->second;
}

void MemoryDependenceGraph::collectPreds(ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown,
                                         SmallVectorImpl<Operation *> &defsOut,
                                         SmallVectorImpl<Operation *> &predsOut)
{
    llvm::SmallSetVector<Operation *, INIT_SIZE> defs;
    llvm::SmallSetVector<Operation *, INIT_SIZE> preds;

    // defs collects lastWriter only on reads (RAW); preds collects all ordering deps (RAW/WAR/WAW).
    auto addFromSlot = [&](MemSlot *slot, bool isWriteLike) {
        if (slot->dataSource && !isWriteLike) {
            defs.insert(slot->dataSource);
        }
        if (slot->lastWriter) {
            preds.insert(slot->lastWriter); // XAW: we write/read what lastWriter wrote
        }
        if (isWriteLike) {
            for (Operation *r : slot->pendingReads) {
                preds.insert(r); // WAR: we will overwrite data still needed by readers
            }
        }
    };

    // The unknown op is considered as a barrier.
    if (unknown) {
        for (auto &slot : slots) {
            addFromSlot(slot.get(), true);
        }
        predsOut.assign(preds.begin(), preds.end());
        return;
    }

    DenseMap<Value, SmallVector<MemSlot *>> cache;
    for (const auto &e : effects) {
        Value v = e.getValue();
        if (isa<MemoryEffects::Read>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(v, cache)) {
                addFromSlot(s, false);
            }
        } else if (isa<MemoryEffects::Write>(e.getEffect()) || isa<MemoryEffects::Free>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(v, cache)) {
                addFromSlot(s, true);
            }
        }
    }

    defsOut.assign(defs.begin(), defs.end());
    predsOut.assign(preds.begin(), preds.end());
}

void MemoryDependenceGraph::applyEffects(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown)
{
    // Conservative strategy. Unknown op is to treat as barrier.
    if (unknown) {
        for (auto &slot : slots) {
            slot->lastWriter = op;
            slot->pendingReads.clear();
        }
        return;
    }

    for (auto result: op->getOpResults()) {
        if (isa<BaseMemRefType>(result.getType())) {
            if (MemSlot *s = getOrCreateSlot(result)) {
                s->dataSource = op;
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        }
    }

    DenseMap<Value, SmallVector<MemSlot *>> cache;

    for (const auto &e : effects) {
        // Reads first: prevents a self-aliasing op from clearing its own pending read.
        if (isa<MemoryEffects::Read>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(e.getValue(), cache)) {
                s->pendingReads.insert(op);
            }
        }
    }

    for (const auto &e : effects) {
        Value v = e.getValue();
        if (isa<MemoryEffects::Allocate>(e.getEffect())) {
            if (MemSlot *s = getOrCreateSlot(v)) {
                s->dataSource = op;
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Write>(e.getEffect())) {
            for (MemSlot *s : resolveAliasSlots(v, cache)) {
                // annotation::MarkOp, no data produced, is marked with Mem::Write
                if (!isa<annotation::MarkOp>(op)) {
                    s->dataSource = op;
                }
                s->lastWriter = op;
                s->pendingReads.clear();
            }
        } else if (isa<MemoryEffects::Free>(e.getEffect())) {
            SmallPtrSet<MemSlot *, INIT_SIZE> toRemove;
            if (!v) {
                LOG_DEBUG("Free of unknown value: conservatively drop all slots.");
                for (auto &slot : slots) {
                    toRemove.insert(slot.get());
                }
            } else {
                auto aliasSlots = findAliasSlots(v);
                for (auto *slot : aliasSlots) {
                    toRemove.insert(slot);
                }
            }
            llvm::erase_if(slots, [&](std::unique_ptr<MemSlot> &sp) {
                if (toRemove.contains(sp.get())) {
                    if (sp->memref) {
                        valueToSlot.erase(sp->memref);
                    }
                    return true;
                }
                return false;
            });
            cache.clear();
        }
    }
}

MemoryDependenceGraph::Snapshot MemoryDependenceGraph::takeSnapshot() const
{
    Snapshot snap;
    snap.states.reserve(slots.size());
    for (const auto &slot : slots) {
        snap.states.push_back(*slot);
    }
    return snap;
}

void MemoryDependenceGraph::restoreSnapshot(Snapshot &&snap)
{
    slots.clear();
    valueToSlot.clear();
    slots.reserve(snap.states.size());
    for (auto &s : snap.states) {
        auto slot = std::make_unique<MemSlot>(std::move(s));
        if (slot->memref) {
            valueToSlot[slot->memref] = slot.get();
        }
        slots.push_back(std::move(slot));
    }
}

void MemoryDependenceGraph::recordEdges(Operation *op, ArrayRef<Operation *> defs, ArrayRef<Operation *> preds)
{
    if (!defs.empty()) {
        auto &defList = memDefs[op];
        defList.assign(defs.begin(), defs.end());
        for (Operation *p : defs) {
            memUsers[p].push_back(op);
        }
    }

    if (!preds.empty()) {
        auto &execBeforeList = execBefore[op];
        execBeforeList.assign(preds.begin(), preds.end());
        for (Operation *p : preds) {
            execAfter[p].push_back(op);
        }
    }
}
