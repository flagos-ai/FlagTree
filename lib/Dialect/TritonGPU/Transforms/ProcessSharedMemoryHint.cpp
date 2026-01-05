#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <iostream>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPROCESSSHAREDMEMORYHINT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// flagtree hints # @hint: shared_memory
struct ProcessSharedMemoryHintPass
    : public impl::TritonGPUProcessSharedMemoryHintBase<
          ProcessSharedMemoryHintPass> {
  Value createIncrementModuloForLoad(OpBuilder &builder, Location loc,
                                     Value counter, Value modulus, Value zero,
                                     Value one) {
    Value addOne = builder.create<arith::AddIOp>(loc, counter, one);
    Value outOfRangeCond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, addOne, modulus);

    return builder.create<arith::SelectOp>(loc, outOfRangeCond, zero, addOne);
  }

  TypedValue<triton::gpu::MemDescType>
  createSingleBufferViewForLoad(OpBuilder &builder, Value alloc, Value idx) {
    assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
           "Expected MemDescType");
    auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
    SmallVector<int64_t> shape;
    assert(allocDescType.getShape().size() > 1 &&
           "Expected multi-dimensional memdesc (e.g., Nx...) for subview");
    shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
                 allocDescType.getShape().end());
    auto viewDescType = triton::gpu::MemDescType::get(
        shape, allocDescType.getElementType(), allocDescType.getEncoding(),
        allocDescType.getMemorySpace(), allocDescType.getMutableMemory(),
        /*allocShape=*/allocDescType.getAllocShape());
    return builder.create<triton::gpu::MemDescIndexOp>(
        alloc.getLoc(), viewDescType, alloc, idx);
  }

  void createAsyncCopyForLoad(triton::LoadOp loadOp, Value alloc,
                              Value insertIdx, Value extractIdx) {
    OpBuilder builder(loadOp);
    Value zero = builder.create<arith::ConstantIntOp>(loadOp.getLoc(), 0, 32);

    builder.setInsertionPoint(loadOp);
    Location loc = loadOp.getLoc();
    Value src = loadOp.getPtr();
    Value mask = loadOp.getMask();
    Value other = loadOp.getOther();
    triton::gpu::MemDescType allocTy =
        cast<triton::gpu::MemDescType>(alloc.getType());

    // Create async copy
    Value view = createSingleBufferViewForLoad(builder, alloc, insertIdx);
    Operation *copy = builder.create<triton::gpu::AsyncCopyGlobalToLocalOp>(
        loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
        loadOp.getIsVolatile());
    Operation *commmit = builder.create<triton::gpu::AsyncCommitGroupOp>(
        loc, copy->getResult(0));

    // Create wait and local load
    auto wait =
        builder.create<triton::gpu::AsyncWaitOp>(loc, commmit->getResult(0), 0);
    auto viewLoad = createSingleBufferViewForLoad(builder, alloc, extractIdx);

    replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad,
                             wait.getResult());
    loadOp->erase();
  }

  Value createAllocForLoad(triton::LoadOp &loadOp) {
    OpBuilder builder(loadOp);

    auto ty = cast<RankedTensorType>(loadOp.getType());
    SmallVector<int64_t> bufferShape(ty.getShape().begin(),
                                     ty.getShape().end());
    bufferShape.insert(bufferShape.begin(), 1); // Use 1 replace distance

    auto context = ty.getContext();
    auto ctaLayout = triton::gpu::getCTALayout(ty.getEncoding());
    auto order = triton::gpu::getOrder(ty);
    auto sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
        context, 1, 1, 1, order, ctaLayout);
    auto sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(context);
    auto memdescType = triton::gpu::MemDescType::get(
        bufferShape, ty.getElementType(), sharedEnc, sharedMemorySpace, true);

    Value alloc =
        builder.create<triton::gpu::LocalAllocOp>(loadOp.getLoc(), memdescType);
    return alloc;
  }

  bool
  canUseAsyncCopyForLoad(triton::LoadOp &op,
                         triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    bool canUseAsyncCopyFlag = false;
    if (!isa<RankedTensorType>(op.getType())) {
      canUseAsyncCopyFlag = op.getType().getIntOrFloatBitWidth() >= 32;
    } else {
      triton::gpu::SharedEncodingTrait sharedEncoding =
          triton::getSharedEncoding(cast<RankedTensorType>(op.getType()));
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      canUseAsyncCopyFlag =
          triton::canBeConvertedToAsyncLoad(op, axisInfoAnalysis);
      int copyVecBytes = triton::getCopyVecBytes(
          cast<RankedTensorType>(op.getType()), sharedEncoding);

      canUseAsyncCopyFlag &= copyVecBytes >= 4;
    }

    return canUseAsyncCopyFlag;
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    moduleOp.walk([&](Operation *curr) {
      // Only handle LoadOp
      if (auto op = dyn_cast<triton::LoadOp>(curr)) {
        if (!op.getFlagtreeHints().empty() &&
            op.getFlagtreeHints().compare("shared_memory") == 0) {
          if (canUseAsyncCopyForLoad(op, axisInfoAnalysis)) {
            auto alloc = createAllocForLoad(op);
            assert(alloc && "Failed to create alloc for the async load.");

            IRRewriter builder(op.getContext());
            builder.setInsertionPoint(op);
            Location loc = op.getLoc();

            // Create two new counters to index into the allocs.
            Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
            Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
            Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
            Value insertIdx = minusOne;
            Value extractIdx = minusOne;

            // Create two counters for the insert and extract indices to avoid
            // creating long liverange.
            int numBuffers = 1;
            Value numBuffersVal =
                builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
            insertIdx = createIncrementModuloForLoad(builder, loc, insertIdx,
                                                     numBuffersVal, zero, one);
            extractIdx = createIncrementModuloForLoad(builder, loc, extractIdx,
                                                      numBuffersVal, zero, one);

            createAsyncCopyForLoad(op, alloc, insertIdx, extractIdx);
          } else {
            std::cout << "Error: Asynchronous copying to shared memory is not "
                         "possible "
                         "due to being less than four bytes."
                      << std::endl;
          }
        }
      }
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
