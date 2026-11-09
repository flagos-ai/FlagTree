#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

#include "Common.h"

static constexpr const char *DEBUG_TYPE = "SplitIfByBlockIdScalarClosure";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace CVPipeline;
using namespace SplitIf;

void ScalarClosure::collectOuterDependency(Operation *op) {
  op->walk([this](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      collectScalarClosure(operand);
    }
  });
};

void ScalarClosure::collectScalarClosure(Value val) {
  if (!val || isa<TensorType>(val.getType())) {
    return;
  }
  Operation *defOp = val.getDefiningOp();
  if (!defOp || llvm::isa<memref::AllocOp, memref::AllocaOp>(defOp)) {
    return;
  }

  auto *defBlock = defOp->getBlock();
  if (!defBlock || (defBlock != block &&
                    // to prevent memref dependencies in main loop
                    (!includeParent || defBlock != parentBlock))) {
    return;
  }

  // hard to determine - if scf ops have scalar/memref result, should we clone
  // the op as well? Perhaps walk into the subregions and clone the
  // corresponding ops would be better. But for now let's just clone them
  collectOuterDependency(defOp);
  scalarOps.insert(defOp);
}

void ScalarClosure::collect() {
  if (block == nullptr) {
    return;
  }
  for (Operation *op : ops) {
    collectOuterDependency(op);
  }
}

ScalarClosure::ScalarClosure(BlockGroup &group, ArrayRef<Operation *> ops)
    : ops(ops) {
  if (group.ops.empty()) {
    return;
  }
  block = group.ops.front()->getBlock();
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    return;
  }
  parentBlock = parentOp->getBlock();
}

ScalarClosure::ScalarClosure(Block *block, ArrayRef<Operation *> ops,
                             bool includeParent)
    : block(block), ops(ops), includeParent(includeParent) {
  if (!block)
    return;
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    return;
  }
  parentBlock = parentOp->getBlock();
}

bool ScalarClosure::isBefore(Operation *a, Operation *b) {
  if (a->getBlock() == parentBlock || b->getBlock() == parentBlock) {
    a = parentBlock->findAncestorOpInBlock(*a);
    b = parentBlock->findAncestorOpInBlock(*b);
  }
  return a->isBeforeInBlock(b);
}

std::pair<SmallVector<Operation *>, IRMapping>
ScalarClosure::capture(int blockId) {
  if (scalarOps.empty()) {
    return {};
  }

  IRMapping mapper;
  SmallVector<Operation *> opsToClone(scalarOps.begin(), scalarOps.end());

  // sort the collected ops so that they are in the same dominance order as in
  // mlir
  llvm::sort(opsToClone,
             [this](Operation *a, Operation *b) { return isBefore(a, b); });

  Operation *firstOp = nullptr;
  for (auto *op : ops) {
    if (op->getBlock() != block) {
      continue;
    }
    if (firstOp == nullptr || op->isBeforeInBlock(firstOp)) {
      firstOp = op;
    }
  }
  OpBuilder builder(firstOp);
  builder.setInsertionPoint(firstOp);

  SmallVector<Operation *> newOps;

  LDBG("Collecting scalar closures for blockId: " << blockId);
  for (Operation *scalarOp : opsToClone) {
    if (llvm::is_contained(ops, scalarOp)) {
      // the current operation have the same blockId, but will not be cloned
      // hence we need to move insertion point
      builder.setInsertionPointAfter(scalarOp);
      continue;
    }

    LDBG("Clone: " << *scalarOp);

    Operation *cloned = builder.clone(*scalarOp, mapper);
    cloned->setAttr(CVPipeline::kBlockId,
                    Builder(scalarOp->getContext()).getI32IntegerAttr(blockId));
    newOps.push_back(cloned);
  }

  return {newOps, mapper};
}
