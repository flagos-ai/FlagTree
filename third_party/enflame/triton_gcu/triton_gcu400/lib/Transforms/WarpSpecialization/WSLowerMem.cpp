/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <unordered_set>

#include "CodePartitionUtility.h"

#include "mlir/IR/Dominance.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-lower-mem"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gcu {

/*
static Value createBufferView(OpBuilderWithAsyncTaskIds &builder, Value alloc,
                              Value idx) {
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
      allocDescType.getAllocShape());
  return builder.create<triton::gpu::MemDescIndexOp>(alloc.getLoc(),
                                                     viewDescType, alloc, idx);
}

static int getloadOpsize(tt::LoadOp &loadOp) {
  auto tensorType = getTensorType(loadOp.getType());
  int loadSize = product(tensorType.getShape());
  return loadSize * tensorType.getElementType().getIntOrFloatBitWidth() / 8;
}
*/

Value getBufferForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                Type loadType, Value buffer, Value bufferIdx,
                                bool mutableMem) {
  auto context = buffer.getContext();
  auto tensorType = getTensorType(loadType);
  assert(tensorType);

  // Get shape, layout and type of a slice
  auto sliceShape = tensorType.getShape();
  auto elemType = tensorType.getElementType();
  auto sharedLayout =
      dyn_cast<ttg::MemDescType>(buffer.getType()).getEncoding();
  auto sliceType = RankedTensorType::get(sliceShape, elemType, sharedLayout);

  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType subviewTy =
      ttg::MemDescType::get(sliceType.getShape(), sliceType.getElementType(),
                            sliceType.getEncoding(), sharedMemorySpace,
                            /*mutableMemOry=*/mutableMem);

  return builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      buffer.getLoc(), subviewTy, buffer, bufferIdx);
}

Operation *optimizeLoadOps(OpBuilderWithAsyncTaskIds &builder,
                           SmallVector<tt::LoadOp> &loadOps,
                           SmallVector<Value> &buffers, Value bufferIdx,
                           Value bufferIdxExtract, Operation *headProducer,
                           Operation *headConsumer) {
  // Compute the total size of the loads.
  // int sizeInBytes = 0;
  // for (auto &loadOp : loadOps) {
  //   sizeInBytes += getloadOpsize(loadOp);
  // }

  // For each of the following ops, we will operate on a subview of each value
  // according to the pipeline stage.
  builder.setAsyncTaskIdsFromOp(headProducer);
  Operation *copy = nullptr;
  for (auto [loadOp, buffer] : zip(loadOps, buffers)) {
    builder.setInsertionPoint(loadOp);
    auto pipelineBuffer = getBufferForPipelineStage(builder, loadOp.getType(),
                                                    buffer, bufferIdx, true);
    auto mem = builder.createWithAsyncTaskIds<tt::LoadOp>(
        loadOp.getLoc(), loadOp.getPtr(), loadOp.getMask(), loadOp.getOther(),
        loadOp.getBoundaryCheck(), loadOp.getPadding(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    copy = builder.createWithAsyncTaskIds<ttg::LocalStoreOp>(
        loadOp.getLoc(), mem.getResult(), pipelineBuffer);
  }

  // Convert all the consumers to local_load
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.setInsertionPoint(headConsumer);
  for (auto [loadOp, buffer] : zip(loadOps, buffers)) {
    builder.setInsertionPoint(loadOp);
    auto pipelineBuffer = getBufferForPipelineStage(
        builder, loadOp.getType(), buffer, bufferIdxExtract, false);
    auto sharedLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
        headConsumer->getLoc(), loadOp.getType(), pipelineBuffer);

    loadOp.getResult().replaceAllUsesWith(sharedLoad.getResult());
    loadOp.erase();
  }
  return copy;
}

} // namespace mlir::triton::gcu
