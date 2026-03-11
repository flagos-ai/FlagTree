#include "tle/dialect/include/Conversion/TleToLLVM/RemotePointerUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"

namespace {

using namespace mlir;
constexpr llvm::StringLiteral kRemoteShardCarrierAttr =
    "tle.remote_shard_id_carrier";

Value peelRemoteMetadataCarrier(Value ptr) {
  llvm::DenseSet<Value> visited;
  Value current = ptr;
  while (current && visited.insert(current).second) {
    Operation *def = current.getDefiningOp();
    if (!def)
      break;
    if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
      current = convert.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      current = bcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      current = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(def)) {
      current = reshape.getSrc();
      continue;
    }
    break;
  }
  return current;
}

bool isTlePointerProducer(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == "tle.local_pointers" || name == "tle.remote_pointers";
}

} // namespace

namespace mlir::triton::tle {

bool isTlePointerValue(Value ptr) {
  Value carrier = peelRemoteMetadataCarrier(ptr);
  return isTlePointerProducer(carrier.getDefiningOp());
}

RemotePointerInfo
getRemotePointerInfoFromValue(Value ptr, ConversionPatternRewriter &rewriter) {
  RemotePointerInfo info;
  info.basePtr = ptr;
  info.vectorHintPtr = ptr;

  Value carrier = peelRemoteMetadataCarrier(ptr);
  info.vectorHintPtr = carrier;
  Operation *defOp = carrier.getDefiningOp();
  if (!defOp)
    return info;

  // Dedicated remote op path: recover vectorization hint from the source
  // pointer and derive shard id from the shard operand directly.
  if (defOp->getName().getStringRef() == "tle.remote_pointers") {
    if (defOp->getNumOperands() >= 1)
      info.vectorHintPtr = defOp->getOperand(0);
    if (defOp->getNumOperands() >= 2) {
      Value shard = defOp->getOperand(1);
      APInt shardConst;
      if (matchPattern(shard, m_ConstantInt(&shardConst))) {
        info.constCTAId = static_cast<int32_t>(shardConst.getSExtValue());
      } else {
        Value remappedShard = rewriter.getRemappedValue(shard);
        info.dynamicCTAId = remappedShard ? remappedShard : shard;
      }
    }
  }

  auto ctaAttr = defOp->getAttrOfType<IntegerAttr>("tle.remote_cta_id");
  if (ctaAttr)
    info.constCTAId = static_cast<int32_t>(ctaAttr.getInt());

  if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(defOp);
      addPtrOp && addPtrOp->hasAttr(kRemoteShardCarrierAttr)) {
    // Keep the remote carrier pointer as the lowering source and strip the
    // synthetic shard offset during remote memory op lowering.
    info.basePtr = ptr;
    info.vectorHintPtr = addPtrOp.getPtr();
    info.stripShardOffsetFromPtr = true;
    Value shardOffset = addPtrOp.getOffset();
    APInt shardConst;
    if (matchPattern(shardOffset, m_ConstantInt(&shardConst))) {
      info.constCTAId = static_cast<int32_t>(shardConst.getSExtValue());
    } else {
      if (auto splatOp = shardOffset.getDefiningOp<triton::SplatOp>()) {
        APInt splatConst;
        if (matchPattern(splatOp.getSrc(), m_ConstantInt(&splatConst))) {
          info.constCTAId = static_cast<int32_t>(splatConst.getSExtValue());
        }
      }
      DenseElementsAttr denseConst;
      if (!info.constCTAId &&
          matchPattern(shardOffset, m_Constant(&denseConst)) &&
          denseConst.isSplat() && denseConst.getElementType().isInteger(32)) {
        info.constCTAId = static_cast<int32_t>(
            denseConst.getSplatValue<APInt>().getSExtValue());
      } else if (!info.constCTAId) {
        info.dynamicCTAId = rewriter.getRemappedValue(shardOffset);
      }
    }
  }

  return info;
}

unsigned inferTlePointerVectorSize(Value ptr,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy || tensorTy.getRank() == 0)
    return 1;
  if (!tensorTy.getEncoding())
    return 1;

  auto *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
  if (!axisInfo || axisInfo->getRank() == 0)
    return 1;

  SmallVector<unsigned> order;
  Attribute encoding = tensorTy.getEncoding();
  if (auto dist = dyn_cast<triton::gpu::DistributedEncodingTrait>(encoding)) {
    order = triton::gpu::getOrder(dist, tensorTy.getShape());
  } else if (auto shared =
                 dyn_cast<triton::gpu::SharedEncodingTrait>(encoding)) {
    order = triton::gpu::getOrder(shared, tensorTy.getShape());
  } else {
    order = triton::gpu::getOrder(tensorTy);
  }
  auto contigPerThread = triton::gpu::getContigPerThread(tensorTy);
  if (contigPerThread.empty())
    return 1;

  auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  unsigned elemBytes = std::max<unsigned>(1, pointeeBitWidth / 8);
  unsigned maxByType = std::max<unsigned>(1, 128 / pointeeBitWidth);
  unsigned elemsPerThread = std::max<unsigned>(
      1, static_cast<unsigned>(
             triton::gpu::getTotalElemsPerThread(ptr.getType())));
  unsigned best = 1;
  // Local/remote pointer tensors can carry an encoding order whose leading
  // axis does not correspond to the flattened row-major contiguous dimension.
  // Probe all axes and keep the best legal vector width.
  for (unsigned axis = 0; axis < static_cast<unsigned>(axisInfo->getRank()) &&
                          axis < contigPerThread.size();
       ++axis) {
    unsigned contiguity = std::max<unsigned>(
        1,
        std::min<unsigned>(std::max<int64_t>(1, axisInfo->getContiguity(axis)),
                           contigPerThread[axis]));
    unsigned divisibility =
        std::max<int64_t>(1, axisInfo->getDivisibility(axis));
    unsigned alignment = std::min<unsigned>(
        contiguity, std::max<unsigned>(1, divisibility / elemBytes));
    unsigned candidate = std::max<unsigned>(
        1, std::min<unsigned>(std::min<unsigned>(maxByType, alignment),
                              elemsPerThread));
    best = std::max(best, candidate);
  }
  return best;
}

unsigned inferTlePointerLayoutVectorHint(Value ptr) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy || tensorTy.getRank() == 0)
    return 1;
  if (!tensorTy.getEncoding())
    return 1;

  auto dist =
      dyn_cast<triton::gpu::DistributedEncodingTrait>(tensorTy.getEncoding());
  if (!dist)
    return 1;
  auto order = triton::gpu::getOrder(dist, tensorTy.getShape());
  auto contigPerThread = triton::gpu::getContigPerThread(tensorTy);
  if (order.empty() || contigPerThread.empty())
    return 1;

  unsigned pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  unsigned maxByType = std::max<unsigned>(1, 128 / pointeeBitWidth);
  unsigned elemsPerThread = std::max<unsigned>(
      1, static_cast<unsigned>(
             triton::gpu::getTotalElemsPerThread(ptr.getType())));
  unsigned best = 1;
  for (unsigned axis : order) {
    if (axis >= contigPerThread.size())
      continue;
    unsigned candidate = std::max<unsigned>(
        1,
        std::min<unsigned>(std::min<unsigned>(maxByType, contigPerThread[axis]),
                           elemsPerThread));
    best = std::max(best, candidate);
  }
  return best;
}

} // namespace mlir::triton::tle
