#include "TleTileToLLVMUtils.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;

SmallVector<unsigned> getCTATileOrder(RankedTensorType type) {
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())) {
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

unsigned linearize(ArrayRef<unsigned> coords,
                   ArrayRef<unsigned> shape,
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

} // namespace mlir::triton::tle
