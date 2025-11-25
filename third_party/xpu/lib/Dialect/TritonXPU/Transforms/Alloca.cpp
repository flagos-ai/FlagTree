//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUALLOCA
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUAllocaPass
    : public impl::TritonXPUAllocaBase<TritonXPUAllocaPass> {

public:
  using impl::TritonXPUAllocaBase<TritonXPUAllocaPass>::TritonXPUAllocaBase;

  TritonXPUAllocaPass() = default;
  TritonXPUAllocaPass(unsigned bufferSize, unsigned coreNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    SetVector<Operation *> visitedOps;
    m.walk([&](triton::xpu::LoadOp loadOp) {
      auto loc = loadOp.getLoc();
      OpBuilder builder(loadOp);
      auto resType = loadOp.getResult().getType();
      auto gmPtrType = loadOp.getPtr().getType();
      auto lmPtrType = addrspaceCast(gmPtrType, 0);
      auto size =
          mlir::isa<RankedTensorType>(gmPtrType)
              ? product(mlir::cast<RankedTensorType>(gmPtrType).getShape())
              : 1;
      if (auto gm2lmOp =
              dyn_cast<triton::xpu::GM2LMOp>(loadOp.getPtr().getDefiningOp())) {
        if (!visitedOps.count(gm2lmOp)) {
          visitedOps.insert(gm2lmOp);
          auto offsetState = static_cast<OffsetState>(gm2lmOp.getOffsetState());
          size = offsetState == OffsetState::DiscreteSame
                     ? std::min(static_cast<int64_t>(coreNum), size)
                     : size;
          auto allocaOp =
              builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

          auto operandSegmentSizesAttr =
              gm2lmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
          SmallVector<int32_t> operandSegmentSizes(
              operandSegmentSizesAttr.asArrayRef());
          ++(operandSegmentSizes.back()); // 0: ptr, 1: mask, 2: len, 3: bufPtr
          gm2lmOp->setAttr("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr(operandSegmentSizes));

          gm2lmOp->insertOperands(gm2lmOp->getNumOperands(), {allocaOp});

          allocaOp->moveBefore(gm2lmOp);
        }
      } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(
                     loadOp.getPtr().getDefiningOp())) {
        if (!visitedOps.count(gm2lmOp)) {
          visitedOps.insert(gm2lmOp);
          auto offsetState = static_cast<OffsetState>(gm2lmOp.getOffsetState());
          size = offsetState == OffsetState::DiscreteSame
                     ? std::min(static_cast<int64_t>(coreNum), size)
                     : size;
          auto allocaOp =
              builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

          auto operandSegmentSizesAttr =
              gm2lmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
          SmallVector<int32_t> operandSegmentSizes(
              operandSegmentSizesAttr.asArrayRef());
          ++(operandSegmentSizes.back()); // 0: ptr, 1: mask, 2: len, 3: bufPtr
          gm2lmOp->setAttr("operandSegmentSizes",
                           builder.getDenseI32ArrayAttr(operandSegmentSizes));

          gm2lmOp->insertOperands(gm2lmOp->getNumOperands(), {allocaOp});

          allocaOp->moveBefore(gm2lmOp);
        }
      } else {
        llvm_unreachable("Only support GM2LM as definingOp of load ptr");
      }
    });

    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto loc = storeOp.getLoc();
      OpBuilder builder(storeOp);
      auto resType = storeOp.getValue().getType();
      auto gmPtrType = storeOp.getPtr().getType();
      auto lmPtrType = addrspaceCast(gmPtrType, 0);
      auto size =
          mlir::isa<RankedTensorType>(gmPtrType)
              ? product(mlir::cast<RankedTensorType>(gmPtrType).getShape())
              : 1;
      if (auto lm2gmOp =
              dyn_cast<triton::xpu::LM2GMOp>(storeOp->getNextNode())) {
        auto allocaOp =
            builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

        auto operandSegmentSizesAttr =
            lm2gmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        SmallVector<int, 4> operandSegmentSizes(
            operandSegmentSizesAttr.asArrayRef());
        ++operandSegmentSizes[3]; // 0: ptr, 1: value, 2: len, 3: bufPtr
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->insertOperands(lm2gmOp->getNumOperands(), {allocaOp});
        // remove value from lm2gm
        --operandSegmentSizes[1];
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->eraseOperands(1);

        allocaOp->moveBefore(storeOp);
        storeOp->setOperand(0, allocaOp);
      } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(
                     storeOp->getNextNode())) {
        auto allocaOp =
            builder.create<triton::xpu::AllocaOp>(loc, lmPtrType, size);

        auto operandSegmentSizesAttr =
            lm2gmOp->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        SmallVector<int, 4> operandSegmentSizes(
            operandSegmentSizesAttr.asArrayRef());
        ++operandSegmentSizes[4]; // 0: ptr, 1: value, 2: mask, 3: len, 4:
                                  // bufPtr
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->insertOperands(lm2gmOp->getNumOperands(), {allocaOp});
        // remove value from lm2gm
        --operandSegmentSizes[1];
        lm2gmOp->setAttr("operandSegmentSizes",
                         builder.getDenseI32ArrayAttr(operandSegmentSizes));
        lm2gmOp->eraseOperands(1);

        allocaOp->moveBefore(storeOp);
        storeOp->setOperand(0, allocaOp);
      } else {
        llvm_unreachable("Only support LM2GM as next node of store");
      }
    });

    // Move Alloca in the Front of FuncOp Body
    m.walk([&](triton::xpu::AllocaOp allocaOp) {
      // 1.Find FuncOp
      Operation *ancestorOp = allocaOp;
      while (!isa<triton::FuncOp>(ancestorOp)) {
        Block *block = ancestorOp->getBlock();
        ancestorOp = block->getParentOp();
      }
      // 2. Move alloca in the Front of the First Op in the FuncOp Body
      Operation *firstOp =
          &(*(cast<triton::FuncOp>(ancestorOp).getBody().front().begin()));
      allocaOp->moveBefore(firstOp);
    });

    // Eliminate Redundant Load-Store Pairs(TODO: Create a New pass for this)
    SmallVector<Operation *> loadStoreOps;
    m.walk([&](triton::xpu::LoadOp loadOp) {
      auto res = loadOp.getResult();
      if (res.hasOneUse() &&
          (loadOp.getStride() == 1 || loadOp.getStride() == -1) &&
          !loadOp.getIsDiscrete()) {
        for (auto user : res.getUsers()) {
          if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(user)) {
            if (auto lm2gmOp =
                    dyn_cast<triton::xpu::LM2GMOp>(storeOp->getNextNode())) {
              if (auto gmlmOp = dyn_cast<triton::xpu::GM2LMOp>(
                      loadOp.getPtr().getDefiningOp())) {
                if (gmlmOp.getPtr().getType() == lm2gmOp.getPtr().getType()) {
                  lm2gmOp->setOperand(lm2gmOp->getNumOperands() - 1,
                                      loadOp.getPtr());
                  loadStoreOps.push_back(storeOp);
                  loadStoreOps.push_back(loadOp);
                }
              }
            } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(
                           storeOp->getNextNode())) {
              if (auto gmlmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(
                      loadOp.getPtr().getDefiningOp())) {
                if (gmlmOp.getPtr().getType() == lm2gmOp.getPtr().getType()) {
                  lm2gmOp->setOperand(lm2gmOp->getNumOperands() - 1,
                                      loadOp.getPtr());
                  loadStoreOps.push_back(storeOp);
                  loadStoreOps.push_back(loadOp);
                }
              }
            }
          }
        }
      }
    });
    for (auto op : loadStoreOps) {
      op->erase();
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
