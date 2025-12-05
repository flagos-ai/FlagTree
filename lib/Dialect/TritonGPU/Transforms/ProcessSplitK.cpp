#include <iostream>
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPROCESSSPLITK
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct ProcessSplitKPass
    : public impl::TritonGPUProcessSplitKBase<ProcessSplitKPass> {
  
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    moduleOp.walk([&](Operation *curr) {
      if (auto op = dyn_cast<triton::SplitKExtractSliceOp>(curr)) {
        IRRewriter builder(op);
        builder.setInsertionPoint(op);
        auto loc = op.getLoc();

        auto ty = cast<RankedTensorType>(op.getType());
        auto src = op.getSrc();

        triton::gpu::ConvertLayoutOp convertLayoutOp;
        if (isa<triton::gpu::ConvertLayoutOp>(src.getDefiningOp())) {
          convertLayoutOp = src.getDefiningOp<triton::gpu::ConvertLayoutOp>();
          src = convertLayoutOp.getSrc();
        }

        auto localLoadOp = src.getDefiningOp<triton::gpu::LocalLoadOp>();
        auto loadValue = localLoadOp.getSrc();
        auto wait = localLoadOp.getToken();
        auto memdesc = cast<triton::gpu::MemDescType>(loadValue.getType());
        
        auto eleType = memdesc.getElementType();
        auto encoding = memdesc.getEncoding();
        auto memSpace = memdesc.getMemorySpace();
        auto mutableMemory = memdesc.getMutableMemory();
        auto allocShape = memdesc.getAllocShape();

        auto res = op.getResult();

        auto offsets = op.getOffsets();
        auto sizes = op.getSizes();
        auto strides = op.getStrides();

        // auto staticOffsets = op.getStaticOffsets();
        // auto staticSizes = op.getStaticSizes();
        // auto staticStrides = op.getStaticStrides();

        SmallVector<int64_t> shape;
        for (const auto &s : sizes) {
          shape.push_back(s);
        }
        auto subSliceDescType = triton::gpu::MemDescType::get(
          shape, 
          memdesc.getElementType(),
          memdesc.getEncoding(),
          memdesc.getMemorySpace(),
          memdesc.getMutableMemory(),
          memdesc.getAllocShape()
        );
        
        // SmallVector<int32_t> newOffsets;
        // auto offsetsLength = staticOffsets.size();
        // for (int i = 0; i < offsetsLength; i++) {
        //   newOffsets.push_back(staticOffsets[i]);
        // }

        auto splitkSubslice = builder.create<triton::gpu::SplitKSubsliceOp>(loc, subSliceDescType, loadValue, offsets);
        auto newLocalLoadOp = builder.create<triton::gpu::LocalLoadOp>(loc, op.getType(), splitkSubslice, wait);
        replaceUsesAndPropagateType(builder, op, newLocalLoadOp.getResult());

        op.erase();
        if (convertLayoutOp) {
          convertLayoutOp.erase();
        }
        localLoadOp.erase();

        // auto context = ty.getContext();
        // auto ctaLayout = triton::gpu::getCTALayout(ty.getEncoding());
        // auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(context);
        // auto tensorMemorySpace = triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
        // auto memdescType = triton::gpu::MemDescType::get(
        //   ty.getShape(), ty.getElementType(), ty.getEncoding(), sharedMemorySpace
        // );
      }
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir