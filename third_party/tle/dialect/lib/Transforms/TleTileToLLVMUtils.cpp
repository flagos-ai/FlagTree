#include "TleTileToLLVMUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
// Get the order of dimensions for CTA tile, from encoding if available,
// otherwise reverse order.
SmallVector<unsigned> getCTATileOrder(RankedTensorType type) {
  if (auto blockedLayout =
          dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())) {
    auto order = blockedLayout.getOrder();
    return SmallVector<unsigned>(order.begin(), order.end());
  }

  unsigned rank = type.getRank();
  SmallVector<unsigned> order;
  order.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    order.push_back(rank - 1 - i);
  return order;
}
// Convert a linear index to multi-dimensional coordinates according to the
// given order.
SmallVector<unsigned> delinearize(unsigned linearIndex,
                                  ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order) {
  SmallVector<unsigned> result(shape.size(), 0);
  unsigned idx = linearIndex;
  for (size_t i = 0; i < order.size(); ++i) {
    unsigned dim = order[i];
    result[dim] = idx % shape[dim];
    idx /= shape[dim];
  }
  return result;
}
// Convert multi-dimensional coordinates to a linear index according to the
// given order.
unsigned linearize(ArrayRef<unsigned> coords, ArrayRef<unsigned> shape,
                   ArrayRef<unsigned> order) {
  unsigned result = 0;
  unsigned stride = 1;
  for (size_t i = 0; i < order.size(); ++i) {
    unsigned dim = order[i];
    result += coords[dim] * stride;
    stride *= shape[dim];
  }
  return result;
}
// Get the shape (number of elements) per CTA tile for each dimension.
SmallVector<unsigned> getShapePerCTATile(RankedTensorType type) {
  auto encoding = type.getEncoding();
  if (!encoding)
    llvm_unreachable("tile op requires tensor with encoding");

  auto shape = type.getShape();
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    auto sizePerThread = blocked.getSizePerThread();
    auto threadsPerWarp = blocked.getThreadsPerWarp();
    auto warpsPerCTA = blocked.getWarpsPerCTA();

    SmallVector<unsigned> result;
    result.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      result.push_back(static_cast<unsigned>(sizePerThread[i]) *
                       static_cast<unsigned>(threadsPerWarp[i]) *
                       static_cast<unsigned>(warpsPerCTA[i]));
    }
    return result;
  }

  llvm_unreachable("tile op only supports BlockedEncoding");
}

// Map the linear thread index (according to BlockedEncoding rules) to the
// multi-dimensional "global data start index" in the tensor.
SmallVector<Value> computeThreadOffsets(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        RankedTensorType tensorType) {
  auto bl = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sizePerThread = bl.getSizePerThread();
  auto threadsPerWarp = bl.getThreadsPerWarp();
  auto warpsPerCTA = bl.getWarpsPerCTA();
  auto order = bl.getOrder();
  int rank = tensorType.getRank();

  auto i32Ty = rewriter.getIntegerType(32);
  Value threadId = rewriter.create<NVVM::ThreadIdXOp>(loc, i32Ty);

  unsigned warpSizeVal = 1;
  for (auto t : threadsPerWarp)
    warpSizeVal *= t;
  // Compute the total number of threads in a warp (product of threadsPerWarp).
  Value warpSizeV = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)warpSizeVal));

  Value laneId = rewriter.create<LLVM::URemOp>(loc, i32Ty, threadId, warpSizeV);
  Value warpId = rewriter.create<LLVM::UDivOp>(loc, i32Ty, threadId, warpSizeV);
  // Compute the thread's lane index in each dimension within the warp.
  SmallVector<Value> laneInDim(rank);
  {
    Value rem = laneId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = order[i];
      unsigned count = threadsPerWarp[dim];
      Value cv = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)count));
      laneInDim[dim] = rewriter.create<LLVM::URemOp>(loc, i32Ty, rem, cv);
      rem = rewriter.create<LLVM::UDivOp>(loc, i32Ty, rem, cv);
    }
  }
  // Compute the warp's index in each dimension within the CTA.
  SmallVector<Value> warpInDim(rank);
  {
    Value rem = warpId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = order[i];
      unsigned count = warpsPerCTA[dim];
      Value cv = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)count));
      warpInDim[dim] = rewriter.create<LLVM::URemOp>(loc, i32Ty, rem, cv);
      rem = rewriter.create<LLVM::UDivOp>(loc, i32Ty, rem, cv);
    }
  }

  SmallVector<Value> threadOffsets(rank);
  for (int d = 0; d < rank; ++d) {
    Value tpw = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)threadsPerWarp[d]));
    Value spt = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr((int32_t)sizePerThread[d]));
    // warpID * threadsPerWarp gives the starting thread index of this warp in
    // the dimension.
    Value warpContrib =
        rewriter.create<LLVM::MulOp>(loc, i32Ty, warpInDim[d], tpw);
    // Add the lane index within the warp to get the global thread index in the
    // dimension.
    Value threadCoord =
        rewriter.create<LLVM::AddOp>(loc, i32Ty, warpContrib, laneInDim[d]);
    // Multiply the global thread index by the number of elements per thread to
    // get the starting element index for this thread.
    threadOffsets[d] =
        rewriter.create<LLVM::MulOp>(loc, i32Ty, threadCoord, spt);
  }

  return threadOffsets;
}

} // namespace mlir::triton::tle
