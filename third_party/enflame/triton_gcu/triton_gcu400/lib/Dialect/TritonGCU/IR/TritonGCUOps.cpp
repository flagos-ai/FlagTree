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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

#define GET_OP_CLASSES
#include "Dialect/TritonGCU/IR/TritonGCUOps.cpp.inc"

namespace mlir {
namespace triton {
namespace gcu {

LogicalResult LoadOp::verify() {
  if (getOffsets().size() != getShape().size() ||
      getOffsets().size() != getStrides().size() ||
      getOffsets().size() != static_cast<unsigned>(getType().getRank()))
    return emitOpError() << "shape/strides/offsets mismatch with result rank";
  if (getPtr().getType().getElementType() != getType().getElementType())
    return emitOpError() << "pointer element type mismatch";
  if (getDefaultValue() &&
      getDefaultValue().getType() != getType().getElementType())
    return emitOpError() << "default element type mismatch";
  if (getOrderHint().size() > getShape().size())
    return emitOpError() << "order_hint rank mismatch with result rank";
  return success();
}

LogicalResult StoreOp::verify() {
  if (getOffsets().size() != getShape().size() ||
      getOffsets().size() != getStrides().size() ||
      getOffsets().size() !=
          static_cast<unsigned>(getValue().getType().getRank()))
    return emitOpError() << "shape/strides/offsets mismatch with value rank";
  if (getPtr().getType().getElementType() !=
      getValue().getType().getElementType())
    return emitOpError() << "pointer element type mismatch";
  if (getOrderHint().size() > getShape().size())
    return emitOpError() << "order_hint rank mismatch with result rank";
  return success();
}

// -- WarpGroupDotOp --
mlir::LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  auto aEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc =
      cast<triton::gpu::TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    auto checkDotOpEncoding = [&](Attribute enc, unsigned opIdx) {
      if (mlir::isa<triton::gpu::NVMMASharedEncodingAttr>(enc))
        return mlir::success();
      Dialect &dialect = enc.getDialect();
      auto interface = cast<DialectInferLayoutInterface>(&dialect);
      return interface->inferDotOpEncoding(enc, opIdx, retEnc, location);
    };
    if (checkDotOpEncoding(aEnc, 0).failed())
      return mlir::failure();
    if (checkDotOpEncoding(bEnc, 1).failed())
      return mlir::failure();
  }
  return mlir::success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<mlir::triton::gpu::MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a,
                         mlir::triton::gpu::SharedMemory::get());
  if (isa<mlir::triton::gpu::MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b,
                         mlir::triton::gpu::SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = aTensorTy.getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  return aShape[aShape.size() - 1] == bShape[bShape.size() - 2];
}

LogicalResult WarpGroupDotOp::verify() {
  auto aTy = cast<triton::gpu::TensorOrMemDesc>(getA().getType());
  auto bTy = cast<triton::gpu::TensorOrMemDesc>(getB().getType());
  auto cTy = cast<RankedTensorType>(getC().getType());
  auto dTy = cast<RankedTensorType>(getD().getType());

  const auto &aShape = aTy.getShape();
  const auto &bShape = bTy.getShape();
  const auto &cShape = cTy.getShape();
  const auto &dShape = dTy.getShape();

  const int64_t aRank = static_cast<int64_t>(aShape.size());
  const int64_t bRank = static_cast<int64_t>(bShape.size());
  const int64_t cRank = static_cast<int64_t>(cShape.size());
  const int64_t dRank = static_cast<int64_t>(dShape.size());

  if (aRank < 2 || bRank < 2 || cRank < 2 || dRank < 2)
    return emitOpError() << "expects A/B/C/D rank >= 2 for matrix multiply";

  if (aRank != bRank || aRank != cRank || aRank != dRank)
    return emitOpError() << "expects A/B/C/D to have the same rank, but got A:"
                         << aRank << ", B:" << bRank << ", C:" << cRank
                         << ", D:" << dRank;

  auto dimCompatible = [](int64_t lhs, int64_t rhs) {
    return lhs == rhs || ShapedType::isDynamic(lhs) ||
           ShapedType::isDynamic(rhs);
  };

  for (int64_t i = 0; i < aRank - 2; ++i) {
    if (!dimCompatible(aShape[i], bShape[i]) ||
        !dimCompatible(aShape[i], cShape[i]) ||
        !dimCompatible(aShape[i], dShape[i])) {
      return emitOpError() << "batch dimensions of A/B/C/D must match at dim "
                           << i << ", but got A:" << aShape[i]
                           << ", B:" << bShape[i] << ", C:" << cShape[i]
                           << ", D:" << dShape[i];
    }
  }

  const int64_t m = aShape[aRank - 2];
  const int64_t kA = aShape[aRank - 1];
  const int64_t kB = bShape[bRank - 2];
  const int64_t n = bShape[bRank - 1];
  const int64_t cM = cShape[cRank - 2];
  const int64_t cN = cShape[cRank - 1];
  const int64_t dM = dShape[dRank - 2];
  const int64_t dN = dShape[dRank - 1];

  if (!dimCompatible(kA, kB))
    return emitOpError() << "expects A(..., M, K) and B(..., K, N), but got K="
                         << kA << " and " << kB;

  if (!dimCompatible(m, cM) || !dimCompatible(m, dM) || !dimCompatible(n, cN) ||
      !dimCompatible(n, dN)) {
    return emitOpError()
           << "expects C/D shape (..., M, N) from A(..., M, K) * B(..., K, N), "
           << "but got A M=" << m << ", B N=" << n << ", C tail=(" << cM << ", "
           << cN << "), D tail=(" << dM << ", " << dN << ")";
  }

  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

} // namespace gcu
} // namespace triton
} // namespace mlir
