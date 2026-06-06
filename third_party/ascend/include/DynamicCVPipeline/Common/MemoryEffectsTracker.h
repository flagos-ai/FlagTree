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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace CVPipeline {
constexpr int INIT_SIZE = 4;
class MemoryDependenceGraph {
public:
    MemoryDependenceGraph(Operation *root, AliasAnalysis &aa);

    ArrayRef<Operation *> getMemDefs(Operation *op) const;
    ArrayRef<Operation *> getMemUsers(Operation *op) const;

    ArrayRef<Operation *> getExecBefore(Operation *op) const;
    ArrayRef<Operation *> getExecAfter(Operation *op) const;

private:
    struct MemSlot {
        Value memref;
        Operation *lastWriter = nullptr;
        Operation *dataSource = nullptr;
        SmallPtrSet<Operation *, INIT_SIZE> pendingReads;
        explicit MemSlot(Value v) : memref(v) {}
    };

    struct Snapshot {
        SmallVector<MemSlot> states;
    };

    void analyzeOp(Operation *op);
    void analyzeRegionsOf(Operation *op);

    SmallVector<MemoryEffects::EffectInstance> collectOuterEffects(Operation *op, bool &unknown);

    SmallVector<MemSlot *> findAliasSlots(Value v);
    ArrayRef<MemSlot *> resolveAliasSlots(Value v,
                                          DenseMap<Value, SmallVector<MemSlot *>> &cache);
    MemSlot *getOrCreateSlot(Value v);

    void collectPreds(ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown,
                      SmallVectorImpl<Operation *> &defsOut,
                      SmallVectorImpl<Operation *> &predsOut);

    void applyEffects(Operation *op, ArrayRef<MemoryEffects::EffectInstance> effects, bool unknown);

    Snapshot takeSnapshot() const;
    void restoreSnapshot(Snapshot &&snap);

    void recordEdges(Operation *op, ArrayRef<Operation *> defs, ArrayRef<Operation *> preds);
    AliasResult queryAlias(Value lhs, Value rhs);

    Operation *root;
    AliasAnalysis &aa;

    // Data dependency
    DenseMap<Operation *, SmallVector<Operation *>> memDefs;
    DenseMap<Operation *, SmallVector<Operation *>> memUsers;

    // Execute order
    DenseMap<Operation *, SmallVector<Operation *>> execBefore;
    DenseMap<Operation *, SmallVector<Operation *>> execAfter;

    SmallVector<std::unique_ptr<MemSlot>> slots;
    DenseMap<Value, MemSlot *> valueToSlot;
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_MEMORY_EFFECTS_TRACKER_H
