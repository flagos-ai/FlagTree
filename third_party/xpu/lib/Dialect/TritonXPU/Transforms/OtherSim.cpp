//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUOTHERSIM
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUOtherSim
    : public impl::TritonXPUOtherSimBase<TritonXPUOtherSim> {

public:
  using impl::TritonXPUOtherSimBase<TritonXPUOtherSim>::TritonXPUOtherSimBase;

  TritonXPUOtherSim() = default;
  TritonXPUOtherSim(unsigned bufferSize, unsigned coreNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  int64_t getNumCol(Type type) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(type))
      return tensorTy.getShape().back();
    else
      return 1;
  }

  int64_t getNumUnroll(Type type) {
    int64_t numUnroll = numUnrollPerCore * coreNum;
    if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
      auto clusterEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
      numUnroll = numUnrollPerCore * clusterEncoding.getCoresPerGroup().back();
    }
    return numUnroll;
  }

  triton::xpu::ClusterLayoutAttr
  createEncoding(MLIRContext *context, triton::xpu::ClusterLayoutAttr &encoding,
                 int64_t iterNum) const {
    auto sizePerCore = encoding.getSizePerCore().vec();
    sizePerCore[sizePerCore.size() - 1] =
        ceil<int64_t>(sizePerCore.back(), iterNum);
    auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
        context, sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder());
    return newEncoding;
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    MLIRContext *context = &getContext();
    m.walk([&](triton::xpu::ReduceOp redOp) {
      // Set numUnrollPerCore=1 When coreDealMultiRows
      RankedTensorType operandType = redOp.getInputTypes()[0];
      auto shape = operandType.getShape();
      auto layout =
          cast<triton::xpu::ClusterLayoutAttr>(operandType.getEncoding());
      unsigned rowsPerCore = layout.getSizePerCore()[0];
      numUnrollPerCore =
          (shape.size() == 2 && rowsPerCore > 1) ? 1 : numUnrollPerCore;
    });

    m.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto loc = gm2lmOp.getLoc();
      OpBuilder builder(gm2lmOp);
      if (auto other = gm2lmOp.getOther()) {
        Value extractOther = other;
        unsigned iterNum = 1;
        if (auto otherTy = dyn_cast<RankedTensorType>(other.getType())) {
          int64_t numUnroll = getNumUnroll(otherTy);
          int64_t numCol = getNumCol(otherTy);
          iterNum = ceil<int64_t>(numCol, numUnroll);
          auto shape = otherTy.getShape().vec();
          shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(otherTy.getEncoding());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          auto newOtherTensorTy = RankedTensorType::get(
              shape, otherTy.getElementType(), newClusterEncoding);
          extractOther = builder.create<triton::xpu::ExtractSliceOp>(
              loc, newOtherTensorTy, other);
        }

        // Create ForOp for UnrollControl
        auto low = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0));
        auto upper = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(iterNum));
        auto step = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1));
        auto forLoopOp = builder.create<scf::ForOp>(loc, low, upper, step);
        builder.setInsertionPointToStart(forLoopOp.getBody());
        Value idx = builder.create<mlir::arith::IndexCastOp>(
            loc, builder.getI32Type(), forLoopOp.getInductionVar());

        // Store Other to LM
        auto allocaOp =
            gm2lmOp.getBufPtr().getDefiningOp<triton::xpu::AllocaOp>();
        auto otherSimOp = builder.create<triton::xpu::StoreOp>(
            loc, allocaOp, extractOther, Value(), idx, -1, false,
            Dtype::UNKNOWN, MemorySyncMode::SYNC);
      }
    });

    m.walk([&](triton::xpu::GM2LMMaskOp gm2lmOp) {
      auto loc = gm2lmOp.getLoc();
      OpBuilder builder(gm2lmOp);
      if (auto other = gm2lmOp.getOther()) {
        Value extractOther = other;
        unsigned iterNum = 1;
        if (auto otherTy = dyn_cast<RankedTensorType>(other.getType())) {
          int64_t numUnroll = getNumUnroll(otherTy);
          int64_t numCol = getNumCol(otherTy);
          iterNum = ceil<int64_t>(numCol, numUnroll);
          auto shape = otherTy.getShape().vec();
          shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(otherTy.getEncoding());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          auto newOtherTensorTy = RankedTensorType::get(
              shape, otherTy.getElementType(), newClusterEncoding);
          extractOther = builder.create<triton::xpu::ExtractSliceOp>(
              loc, newOtherTensorTy, other);
        }

        // Create ForOp for UnrollControl
        auto low = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0));
        auto upper = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(iterNum));
        auto step = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(1));
        auto forLoopOp = builder.create<scf::ForOp>(loc, low, upper, step);
        builder.setInsertionPointToStart(forLoopOp.getBody());
        Value idx = builder.create<mlir::arith::IndexCastOp>(
            loc, builder.getI32Type(), forLoopOp.getInductionVar());

        // Store Other to LM
        auto allocaOp =
            gm2lmOp.getBufPtr().getDefiningOp<triton::xpu::AllocaOp>();
        auto otherSimOp = builder.create<triton::xpu::StoreOp>(
            loc, allocaOp, extractOther, Value(), idx, -1, false,
            Dtype::UNKNOWN, MemorySyncMode::SYNC);
      }
    });
  }

private:
  unsigned numUnrollPerCore = 2;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
