/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
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
#include "GCUTritonGPUConversion.h"
#include "Utils/TritonVersionCompat.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#ifdef ENABLE_TRITON_DISTRIBUTED
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#endif

#ifdef ENABLE_TLE
#include "tle/dialect/include/IR/Dialect.h"
#endif

using namespace mlir;
using namespace mlir::triton::gpu;

// Build a BlockedEncodingAttr using the caller-supplied order and reduce-axis
// frequency map.  When the supplied order's rank doesn't match the tensor rank,
// we rebuild an order for this rank using the frequency map: dims that appear
// in axisFreq are placed at the back (lower priority), sorted by frequency
// ascending; the remaining dims use reverse-iota (higher priority).
triton::gpu::BlockedEncodingAttr
mlir::getBlockedEncodingWithOrder(
    MLIRContext *context, ArrayRef<int64_t> shape,
    ArrayRef<unsigned> order,
    const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq,
    int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  SmallVector<unsigned> effectiveOrder;

  if (static_cast<int>(order.size()) == rank) {
    effectiveOrder.assign(order.begin(), order.end());
  } else {
    SmallVector<unsigned> nonReduceDims;
    SmallVector<std::pair<unsigned, unsigned>> reduceDimsWithFreq;
    for (int i = rank - 1; i >= 0; --i) {
      unsigned dim = static_cast<unsigned>(i);
      auto it = axisFreq.find(dim);
      if (it != axisFreq.end())
        reduceDimsWithFreq.push_back({dim, it->second});
      else
        nonReduceDims.push_back(dim);
    }
    llvm::sort(reduceDimsWithFreq,
               [](const std::pair<unsigned, unsigned> &a,
                  const std::pair<unsigned, unsigned> &b) {
                 if (a.second != b.second)
                   return a.second < b.second;
                 return a.first > b.first;
               });
    effectiveOrder.append(nonReduceDims.begin(), nonReduceDims.end());
    for (auto &kv : reduceDimsWithFreq)
      effectiveOrder.push_back(kv.first);
  }

  SmallVector<unsigned> sizePerThread(rank, 1);
  return triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                               effectiveOrder, numWarps,
                                               threadsPerWarp, numCTAs);
}

// Compute dot-op-compatible warpsPerCTA using the same algorithm as
// TritonDotPattern: shapePerWarp[M] = 8, shapePerWarp[N] = 128.
// This ensures non-dot tensors (loads, stores, elementwise) use the same
// warp distribution as dot operands, minimizing convert_layout ops.
SmallVector<unsigned>
mlir::computeDotWarpsPerCTA(const triton::gpu::BlockedEncodingAttr &encoding,
                            ArrayRef<int64_t> shape, unsigned numWarps) {
  unsigned rank = shape.size();
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(encoding, shape);
  SmallVector<unsigned> shapePerWarp(rank, 1);
  if (rank >= 2) {
    shapePerWarp[rank - 1] = 128; // N dimension
    shapePerWarp[rank - 2] = 8;   // M dimension
  }
  SmallVector<unsigned> warpsPerCTA(rank, 1);
  SmallVector<unsigned> order(encoding.getOrder());
  unsigned remainingNumWarps = numWarps;
  for (unsigned d = 0; d < rank; ++d) {
    unsigned i = order[d];
    warpsPerCTA[i] = std::clamp<unsigned>(
        static_cast<unsigned>(shapePerCTA[i]) / shapePerWarp[i], 1,
        remainingNumWarps);
    remainingNumWarps /= warpsPerCTA[i];
  }
  if (remainingNumWarps > 1 && rank >= 2) {
    warpsPerCTA[rank - 1] *= 2;
    remainingNumWarps /= 2;
  }
  warpsPerCTA[order[rank - 1]] *= remainingNumWarps;
  return warpsPerCTA;
}

//===----------------------------------------------------------------------===//
// GCUTritonGPUTypeConverter
//===----------------------------------------------------------------------===//

GCUTritonGPUTypeConverter::GCUTritonGPUTypeConverter(
    MLIRContext *context, int numWarps, int threadsPerWarp, int numCTAs,
    ArrayRef<unsigned> defaultOrder,
    const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq,
    bool hasDotOp)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs), defaultOrder(defaultOrder.begin(), defaultOrder.end()),
      axisFreq(axisFreq), hasDotOp(hasDotOp) {
  addConversion([](Type type) { return type; });

  addConversion(
      [this](RankedTensorType tensorType) -> RankedTensorType {
#ifdef ENABLE_TLE
        return convertRankedTensorType(tensorType, this->numWarps);
#else
        if (tensorType.getEncoding())
          return tensorType;
        ArrayRef<int64_t> shape = tensorType.getShape();
        auto encoding = getBlockedEncodingWithOrder(
            this->context, shape, this->defaultOrder, this->axisFreq,
            this->numWarps, this->threadsPerWarp, this->numCTAs);
        if (this->hasDotOp) {
          SmallVector<unsigned> warpsPerCTA =
              computeDotWarpsPerCTA(encoding, shape, this->numWarps);
          encoding = BlockedEncodingAttr::get(
              this->context, encoding.getSizePerThread(),
              encoding.getThreadsPerWarp(), warpsPerCTA, encoding.getOrder(),
              triton_gcu::compat::getCGALayout(encoding));
        }
        return tensorType.cloneWithEncoding(encoding);
#endif
      });

  addConversion(
      [this](triton::PointerType ptrType) -> triton::PointerType {
        auto pointeeTensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType());
        if (!pointeeTensorType)
          return ptrType;
        auto convertedTensorType = convertType(pointeeTensorType);
        return triton::PointerType::get(convertedTensorType,
                                        ptrType.getAddressSpace());
      });

#ifdef ENABLE_TLE
  addConversion([this](Value value) -> std::optional<Type> {
    Type type = value.getType();
    int valueNumWarps = getNumWarps(value);
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return convertRankedTensorType(tensorType, valueNumWarps);

    if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
      auto pointeeTensorType =
          dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      if (pointeeTensorType)
        return triton::PointerType::get(
            convertRankedTensorType(pointeeTensorType, valueNumWarps),
            ptrType.getAddressSpace());
    }

    return std::nullopt;
  });
#endif

  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                              ValueRange inputs, Location loc) {
    auto cast = triton::gpu::ConvertLayoutOp::create(builder, loc, tensorType,
                                                     inputs);
    return cast.getResult();
  });
}

#ifdef ENABLE_TLE
int GCUTritonGPUTypeConverter::getNumWarps(Value value) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (Block *owner = blockArg.getOwner()) {
      if (Region *region = owner->getParent()) {
        if (region->getParentOp())
          return triton::gpu::lookupNumWarps(region);
      }
    }
  }
  if (Operation *op = value.getDefiningOp()) {
    if (std::optional<int> contextualNumWarps =
            triton::gpu::maybeLookupNumWarps(op))
      return *contextualNumWarps;
  }
  return numWarps;
}

RankedTensorType
GCUTritonGPUTypeConverter::convertRankedTensorType(RankedTensorType tensorType,
                                                    int contextualNumWarps) const {
  if (tensorType.getEncoding())
    return tensorType;
  ArrayRef<int64_t> shape = tensorType.getShape();
  auto encoding = getBlockedEncodingWithOrder(
      context, shape, defaultOrder, axisFreq, contextualNumWarps,
      threadsPerWarp, numCTAs);
  return tensorType.cloneWithEncoding(encoding);
}
#endif

//===----------------------------------------------------------------------===//
// GCUTritonGPUConversionTarget
//===----------------------------------------------------------------------===//

GCUTritonGPUConversionTarget::GCUTritonGPUConversionTarget(
    MLIRContext &context, GCUTritonGPUTypeConverter &typeConverter)
    : ConversionTarget(context) {
  addLegalDialect<triton::gpu::TritonGPUDialect>();

  addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
               scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect, ub::UBDialect,
                             tensor::TensorDialect>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });

#ifdef ENABLE_TRITON_DISTRIBUTED
  addDynamicallyLegalDialect<triton::distributed::DistributedDialect,
                             triton::simt::SIMTDialect>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });
#endif

  addDynamicallyLegalOp<triton::DotOp>([](triton::DotOp dotOp) -> bool {
    Attribute aEncoding =
        cast<RankedTensorType>(dotOp.getA().getType()).getEncoding();
    Attribute bEncoding =
        cast<RankedTensorType>(dotOp.getB().getType()).getEncoding();
    if (aEncoding && isa<triton::gpu::DotOperandEncodingAttr>(aEncoding) &&
        bEncoding && isa<triton::gpu::DotOperandEncodingAttr>(bEncoding))
      return true;
    return false;
  });

  addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) -> bool {
    for (auto arg : funcOp.getArguments()) {
      if (auto tensor = dyn_cast<RankedTensorType>(arg.getType())) {
        if (!tensor.getEncoding())
          return false;
      }
    }
    return true;
  });

#ifdef ENABLE_TLE
  addDynamicallyLegalDialect<triton::tle::TleDialect>([&](Operation *op) {
    bool hasLegalRegions = true;
    for (auto &region : op->getRegions()) {
      hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
    }
    return hasLegalRegions && typeConverter.isLegal(op);
  });
#endif
}

bool GCUTritonGPUConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {
  bool hasLegalRegions = true;
  for (auto &region : op->getRegions())
    hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
  return hasLegalRegions && typeConverter.isLegal(op);
}
