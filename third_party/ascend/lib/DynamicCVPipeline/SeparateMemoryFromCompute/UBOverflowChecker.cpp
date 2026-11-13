/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/UBOverflowChecker.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace triton;

#define DEBUG_TYPE "UBOverflow"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static annotation::MarkOp findMarkOp(memref::AllocOp allocOp) {
  for (auto *user : allocOp.getResult().getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user))
      return markOp;
  }
  return nullptr;
}

int triton::getAlignUnit(Type elementType) {
  unsigned width = elementType.getIntOrFloatBitWidth();
  if (width == 0 || width > UBConstants::ALIGN_UNIT_BITS)
    return 1;
  return static_cast<int>(UBConstants::ALIGN_UNIT_BITS / width);
}

SmallVector<BufferInfo> triton::collectBuffers(ModuleOp module) {
  SmallVector<BufferInfo> buffers;

  module.walk([&](scope::ScopeOp scopeOp) {
    bool isCube = false;
    bool isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector)))
      return WalkResult::advance();
    if (!isVector)
      return WalkResult::advance();

    scopeOp.walk([&](memref::AllocOp allocOp) {
      BufferInfo buf;
      buf.allocOp = allocOp;
      buf.forOp = allocOp->getParentOfType<scf::ForOp>();

      annotation::MarkOp markOp = findMarkOp(allocOp);
      if (markOp) {
        if (auto attr = markOp->getAttrOfType<IntegerAttr>(
                hivm::MultiBufferAttr::name)) {
          buf.kind = BufferInfo::Kind::Annot;
          buf.markOp = markOp;
          buf.multiBufferCount = std::max<int64_t>(attr.getInt(), 1);
          buf.fromHint = markOp->hasAttr(CVPipeline::kGMLoadHintAttr);
          if (buf.fromHint)
            markOp->removeAttr(CVPipeline::kGMLoadHintAttr);
        }
      }

      buffers.push_back(buf);
    });

    return WalkResult::advance();
  });

  LOG_DEBUG("collected " << buffers.size() << " buffers");
  return buffers;
}

SmallVector<TensorInfo> triton::collectTensorEmpties(ModuleOp module) {
  SmallVector<TensorInfo> tensors;

  module.walk([&](scope::ScopeOp scopeOp) {
    bool isCube = false;
    bool isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector)) || !isVector)
      return WalkResult::advance();

    scopeOp.walk([&](tensor::EmptyOp emptyOp) {
      if (emptyOp->getParentOfType<scope::ScopeOp>() != scopeOp)
        return;

      auto tensorType = emptyOp.getType();
      if (!tensorType.hasStaticShape() || tensorType.getRank() == 0 ||
          llvm::any_of(tensorType.getShape(),
                       [](int64_t dim) { return dim == 0; })) {
        LOG_DEBUG("unsupported tensor.empty shape, skipping " << emptyOp);
        return;
      }

      TensorInfo tensor;
      tensor.emptyOp = emptyOp;
      LOG_DEBUG("tensor.empty: " << emptyOp);
      tensors.push_back(tensor);
    });

    return WalkResult::advance();
  });

  LOG_DEBUG("collected " << tensors.size() << " tensor.empty ops");
  return tensors;
}

static void computeShapedSize(ShapedType shapedType, int64_t &originalSize,
                              int64_t &reducedSize, int64_t &alignedSize) {
  ArrayRef<int64_t> shape = shapedType.getShape();
  Type elementType = shapedType.getElementType();
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();

  if (!shapedType.hasStaticShape() || shape.empty() || bitWidth == 0 ||
      llvm::any_of(shape, [](int64_t dim) { return dim == 0; })) {
    LOG_DEBUG("unsupported shape, skipping size computation");
    return;
  }

  int64_t numElements = 1;
  for (auto dim : shape) {
    numElements *= dim;
  }
  originalSize = numElements * bitWidth;

  // TileAndBindSubBlock splits dim0 across two sub-blocks (one-to-two):
  // reducedDim0 = ceil(dim0 / K_SUB_BLOCK_DIM), where K_SUB_BLOCK_DIM = 2.
  int64_t reducedDim0 = (shape[0] + UBConstants::K_SUB_BLOCK_DIM - 1) /
                        UBConstants::K_SUB_BLOCK_DIM;
  int64_t reducedElements = numElements / shape[0] * reducedDim0;
  reducedSize = reducedElements * bitWidth;

  int64_t lastDim = shape.size() == 1 ? reducedDim0 : shape.back();
  int64_t alignUnit = getAlignUnit(elementType);
  if (lastDim % alignUnit == 0) {
    alignedSize = reducedSize;
    return;
  }

  int64_t alignedLastDim = (lastDim + alignUnit - 1) / alignUnit * alignUnit;
  alignedSize = (reducedSize * alignedLastDim + lastDim - 1) / lastDim;
  alignedSize = llvm::alignTo(alignedSize, UBConstants::ALIGN_UNIT_BITS);
}

void triton::computeBufferSize(BufferInfo &buf) {
  auto memrefType = mlir::cast<MemRefType>(buf.allocOp.getResult().getType());
  computeShapedSize(memrefType, buf.originalSize, buf.reducedSize,
                    buf.alignedSize);
}

void triton::computeTensorSize(TensorInfo &tensor) {
  computeShapedSize(tensor.emptyOp.getType(), tensor.originalSize,
                    tensor.reducedSize, tensor.alignedSize);
}

UBEstimateResult triton::checkUBOverflow(ArrayRef<BufferInfo> buffers,
                                         ArrayRef<TensorInfo> tensors) {
  UBEstimateResult result;
  for (const auto &buf : buffers) {
    if (buf.alignedSize > 0)
      result.totalBits += buf.alignedSize * buf.multiBufferCount;
  }
  for (const auto &tensor : tensors) {
    if (tensor.alignedSize > 0)
      result.totalBits += tensor.alignedSize;
  }

  LOG_DEBUG("UB = " << result.totalBits << " bits");
  return result;
}

static SmallVector<size_t>
collectPruneCandidates(ArrayRef<BufferInfo> buffers) {
  SmallVector<size_t> candidates;
  for (size_t index = 0; index < buffers.size(); ++index) {
    const auto &buf = buffers[index];
    if (buf.kind == BufferInfo::Kind::Annot && buf.markOp &&
        buf.alignedSize > 0 && !buf.fromHint)
      candidates.push_back(index);
  }
  return candidates;
}

static bool shouldPrune(const UBEstimateResult &result,
                        ArrayRef<size_t> candidates) {
  LOG_DEBUG("initial UB = " << result.totalBits << " bits, max = "
                            << UBConstants::UB_SPACE_SIZE_BITS << " bits");

  if (result.totalBits <= UBConstants::UB_SPACE_SIZE_BITS) {
    LOG_DEBUG("safe, no pruning needed");
    return false;
  }

  if (candidates.empty()) {
    LOG_DEBUG("no computable multi_buffer marks to prune");
    return false;
  }

  LOG_DEBUG("collected " << candidates.size() << " annot marks");
  return true;
}

LogicalResult triton::pruneMultiBufferMarks(ModuleOp module) {
  // Step 1: collect buffers and tensors once, then compute their static sizes.
  auto buffers = collectBuffers(module);
  for (auto &buf : buffers)
    computeBufferSize(buf);

  auto tensors = collectTensorEmpties(module);
  for (auto &tensor : tensors)
    computeTensorSize(tensor);

  // Step 2: compute the initial UB usage and collect pruning candidates.
  auto result = checkUBOverflow(buffers, tensors);
  auto candidates = collectPruneCandidates(buffers);
  if (!shouldPrune(result, candidates))
    return success();

  // Step 3: sort eligible non-hint marks by buffer size, largest first.
  llvm::sort(candidates, [&](size_t lhs, size_t rhs) {
    return buffers[lhs].alignedSize > buffers[rhs].alignedSize;
  });

  // Step 4: remove marks one by one, update BufferInfo, and fully re-estimate.
  int deleted = 0;
  for (size_t index : candidates) {
    auto &buf = buffers[index];
    buf.markOp->removeAttr(hivm::MultiBufferAttr::name);
    buf.kind = BufferInfo::Kind::Unannot;
    buf.multiBufferCount = 1;
    ++deleted;

    result = checkUBOverflow(buffers, tensors);
    LOG_DEBUG("after deleting mark #" << deleted << " (size " << buf.alignedSize
                                      << " bits): UB = " << result.totalBits
                                      << " bits");

    if (result.totalBits <= UBConstants::UB_SPACE_SIZE_BITS)
      break;
  }

  return success();
}

void UBOverflowCheckerPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG("Enter UBOverflowChecker pass");

  if (failed(pruneMultiBufferMarks(module))) {
    LOG_DEBUG("pruneMultiBufferMarks failed (non-fatal)");
  }

  LOG_DEBUG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUBOverflowCheckerPass() {
  return std::make_unique<UBOverflowCheckerPass>();
}

void registerUBOverflowCheckerPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createUBOverflowCheckerPass();
  });
}

} // namespace triton
} // namespace mlir
