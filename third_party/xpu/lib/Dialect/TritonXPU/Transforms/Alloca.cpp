//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "triton/Tools/Sys/GetEnv.hpp"

#include <climits>

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

    // Attach an LM scratch buffer to each vector<->scalar boundary op. The
    // scalar side fixes both the element type and the element count: pack
    // writes that many scalars and reads them back as whole vectors, unpack
    // does the reverse, so a single buffer shaped like the scalar side covers
    // both views.
    DenseSet<Operation *> boundaryAllocas;
    auto attachBoundaryBuffer = [&](Operation *op, Value scalarSide) {
      if (op->getNumOperands() > 1)
        return; // bufPtr already attached
      auto scalarTy = mlir::dyn_cast<RankedTensorType>(scalarSide.getType());
      if (!scalarTy)
        return;
      OpBuilder builder(op);
      auto ptrTy = triton::PointerType::get(scalarTy.getElementType(), 0);
      auto lmPtrType = RankedTensorType::get(scalarTy.getShape(), ptrTy,
                                             scalarTy.getEncoding());
      auto allocaOp = builder.create<triton::xpu::AllocaOp>(
          op->getLoc(), lmPtrType, product(scalarTy.getShape()));
      op->insertOperands(op->getNumOperands(), {allocaOp});
      allocaOp->moveBefore(op);
      boundaryAllocas.insert(allocaOp);
    };
    m.walk([&](triton::xpu::PackOp packOp) {
      attachBoundaryBuffer(packOp, packOp.getSrc());
    });
    m.walk([&](triton::xpu::UnpackOp unpackOp) {
      attachBoundaryBuffer(unpackOp, unpackOp.getResult());
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
          (loadOp.getStride() == 1 || loadOp.getStride() == INT32_MIN) &&
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

    // Report-only LM accounting. There is no LM capacity check anywhere in this
    // pipeline -- XTDK decides whether the allocas fit -- so this never
    // rejects anything, it only makes the footprint visible. The boundary
    // scratch is broken out so a regression in it is attributable.
    int64_t totalBytes = 0;
    int64_t scratchBytes = 0;
    unsigned numAllocas = 0;
    m.walk([&](triton::xpu::AllocaOp allocaOp) {
      auto ty = allocaOp.getResult().getType();
      int64_t elems = getTotalElemsPerThread(ty);
      int64_t bytes = elems * triton::getPointeeBitWidth(ty) / 8;
      bytes = (bytes + 63) / 64 * 64; // LM allocas are 64-byte aligned
      totalBytes += bytes;
      ++numAllocas;
      if (boundaryAllocas.contains(allocaOp))
        scratchBytes += bytes;
    });
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "[Alloca] allocas=" << numAllocas << " lmBytes=" << totalBytes
       << " boundaryScratchBytes=" << scratchBytes
       << " (64B-aligned, per core, before memory-inplace reuse; no LM "
          "capacity check exists in this pipeline)";
    m->emitRemark(msg);
    if (mlir::triton::tools::getBoolEnv("TRITONXPU_LM_REPORT"))
      llvm::errs() << "[Alloca][lm] " << msg << "\n";
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
