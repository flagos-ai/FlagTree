#ifdef __TLE__

#include "TileOpUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::musa_tle {

namespace ttg = mlir::triton::gpu;

SmallVector<unsigned> getCTATileOrder(RankedTensorType type) {
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())) {
    auto order = blocked.getOrder();
    return SmallVector<unsigned>(order.begin(), order.end());
  }

  unsigned rank = type.getRank();
  SmallVector<unsigned> order;
  order.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    order.push_back(rank - 1 - i);
  return order;
}

SmallVector<unsigned> delinearize(unsigned linearIndex,
                                  ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order) {
  SmallVector<unsigned> result(shape.size(), 0);
  unsigned idx = linearIndex;
  for (unsigned dim : order) {
    result[dim] = idx % shape[dim];
    idx /= shape[dim];
  }
  return result;
}

unsigned linearize(ArrayRef<unsigned> coords, ArrayRef<unsigned> shape,
                   ArrayRef<unsigned> order) {
  unsigned result = 0;
  unsigned stride = 1;
  for (unsigned dim : order) {
    result += coords[dim] * stride;
    stride *= shape[dim];
  }
  return result;
}

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

SmallVector<Value> computeThreadOffsets(Location loc, OpBuilder &rewriter,
                                        RankedTensorType tensorType) {
  auto blocked = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  auto sizePerThread = blocked.getSizePerThread();
  auto threadsPerWarp = blocked.getThreadsPerWarp();
  auto warpsPerCTA = blocked.getWarpsPerCTA();
  auto order = blocked.getOrder();
  int rank = tensorType.getRank();

  auto i32Ty = rewriter.getIntegerType(32);
  Value threadId = getThreadId(rewriter, loc);

  unsigned warpSize = 1;
  for (unsigned threads : threadsPerWarp)
    warpSize *= threads;
  Value warpSizeV = LLVM::ConstantOp::create(
      rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(warpSize));

  Value laneId =
      LLVM::URemOp::create(rewriter, loc, i32Ty, threadId, warpSizeV);
  Value warpId =
      LLVM::UDivOp::create(rewriter, loc, i32Ty, threadId, warpSizeV);

  SmallVector<Value> laneInDim(rank);
  {
    Value rem = laneId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = order[i];
      Value count = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(threadsPerWarp[dim]));
      laneInDim[dim] = LLVM::URemOp::create(rewriter, loc, i32Ty, rem, count);
      rem = LLVM::UDivOp::create(rewriter, loc, i32Ty, rem, count);
    }
  }

  SmallVector<Value> warpInDim(rank);
  {
    Value rem = warpId;
    for (int i = 0; i < rank; ++i) {
      unsigned dim = order[i];
      Value count = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(warpsPerCTA[dim]));
      warpInDim[dim] = LLVM::URemOp::create(rewriter, loc, i32Ty, rem, count);
      rem = LLVM::UDivOp::create(rewriter, loc, i32Ty, rem, count);
    }
  }

  SmallVector<Value> threadOffsets(rank);
  for (int d = 0; d < rank; ++d) {
    Value threads = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(threadsPerWarp[d]));
    Value elems = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(sizePerThread[d]));
    Value warpContrib =
        LLVM::MulOp::create(rewriter, loc, i32Ty, warpInDim[d], threads);
    Value threadCoord =
        LLVM::AddOp::create(rewriter, loc, i32Ty, warpContrib, laneInDim[d]);
    threadOffsets[d] =
        LLVM::MulOp::create(rewriter, loc, i32Ty, threadCoord, elems);
  }

  return threadOffsets;
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
