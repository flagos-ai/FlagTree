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

#include <algorithm>
#include <numeric>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gpu;

// Build a BlockedEncodingAttr using the caller-supplied order and reduce-axis
// frequency map.  When the supplied order's rank doesn't match the tensor rank,
// we rebuild an order for this rank using the frequency map: dims that appear
// in axisFreq are placed at the back (lower priority), sorted by frequency
// ascending; the remaining dims use reverse-iota (higher priority).
triton::gpu::BlockedEncodingAttr mlir::getBlockedEncodingWithOrder(
    MLIRContext *context, ArrayRef<int64_t> shape, ArrayRef<unsigned> order,
    const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq, int numWarps,
    int threadsPerWarp, int numCTAs) {
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
    llvm::sort(reduceDimsWithFreq, [](const std::pair<unsigned, unsigned> &a,
                                      const std::pair<unsigned, unsigned> &b) {
      if (a.second != b.second)
        return a.second < b.second;
      return a.first > b.first;
    });
    effectiveOrder.append(nonReduceDims.begin(), nonReduceDims.end());
    for (auto &[dim, freq] : reduceDimsWithFreq)
      effectiveOrder.push_back(dim);
  }

  SmallVector<unsigned> sizePerThread(rank, 1);
  return triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                               effectiveOrder, numWarps,
                                               threadsPerWarp, numCTAs);
}

//===----------------------------------------------------------------------===//
// GCUTritonGPUTypeConverter
//===----------------------------------------------------------------------===//

GCUTritonGPUTypeConverter::GCUTritonGPUTypeConverter(
    MLIRContext *context, int numWarps, int threadsPerWarp, int numCTAs,
    ArrayRef<unsigned> defaultOrder,
    const llvm::SmallDenseMap<unsigned, unsigned> &axisFreq)
    : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
      numCTAs(numCTAs) {
  addConversion([](Type type) { return type; });

  addConversion([this, defaultOrder = SmallVector<unsigned>(defaultOrder),
                 axisFreq = llvm::SmallDenseMap<unsigned, unsigned>(axisFreq)](
                    RankedTensorType tensorType) -> RankedTensorType {
    if (tensorType.getEncoding())
      return tensorType;
    ArrayRef<int64_t> shape = tensorType.getShape();
    auto encoding = getBlockedEncodingWithOrder(
        this->context, shape, defaultOrder, axisFreq, this->numWarps,
        this->threadsPerWarp, this->numCTAs);
    return tensorType.cloneWithEncoding(encoding);
  });

  addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
    auto pointeeTensorType =
        dyn_cast<RankedTensorType>(ptrType.getPointeeType());
    if (!pointeeTensorType)
      return ptrType;
    auto convertedTensorType = convertType(pointeeTensorType);
    return triton::PointerType::get(convertedTensorType,
                                    ptrType.getAddressSpace());
  });

  addTargetMaterialization([](OpBuilder &builder, RankedTensorType tensorType,
                              ValueRange inputs, Location loc) {
    auto cast =
        triton::gpu::ConvertLayoutOp::create(builder, loc, tensorType, inputs);
    return cast.getResult();
  });
}

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
                             scf::SCFDialect, ub::UBDialect>(
      [&](Operation *op) { return isDynamicallyLegal(op, typeConverter); });

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
}

bool GCUTritonGPUConversionTarget::isDynamicallyLegal(
    Operation *op, const TypeConverter &typeConverter) {
  bool hasLegalRegions = true;
  for (auto &region : op->getRegions())
    hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
  return hasLegalRegions && typeConverter.isLegal(op);
}
