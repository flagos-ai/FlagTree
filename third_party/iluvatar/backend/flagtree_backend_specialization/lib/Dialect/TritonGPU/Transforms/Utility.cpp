#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Analysis/AxisInfo.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "ttg-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

using namespace triton;

unsigned getNumElementsPerThread(Operation *op, SmallVector<unsigned> order,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  Value val = getMemAccessPtr(op);
  auto ty = cast<RankedTensorType>(val.getType());
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(val);
  unsigned elemNumBits = getElementBitWidth(ty);
  unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
  unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
  unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
  unsigned maxContig =
      std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
  unsigned alignment = std::min(maxMultiple, maxContig);
  // For int64, we have to use this
  unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
  if (elemNumBits <= 32)
    currPerThread = std::min(alignment, 32 / elemNumBits);
  LDBG("elemNumBytes: " << elemNumBytes
                        << ", divisibility: " << maxMultipleBytes
                        << ", contig: " << valInfo.getContiguity(order[0])
                        << ", alignment: " << alignment);
  return currPerThread;
}

static std::optional<Attribute> inferDstEncoding(triton::ReduceOp op,
                                                 Attribute encoding) {
  return triton::gpu::SliceEncodingAttr::get(op->getContext(), op.getAxis(),
                                             encoding, op.getNoWarpReduce());
}

static std::optional<Attribute> inferDstEncoding(triton::ExpandDimsOp op,
                                                 Attribute encoding) {
  auto sliceEncoding = mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);
  if (!sliceEncoding)
    return std::nullopt;
  if (op.getAxis() != sliceEncoding.getDim())
    return std::nullopt;
  return sliceEncoding.getParent();
}

static std::optional<Attribute> inferDstEncoding(JoinOp op, Attribute srcEnc) {
  Attribute dstEnc;
  if (srcEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferJoinOpEncoding(srcEnc, dstEnc,
                                /*loc=*/std::nullopt)
          .succeeded()) {
    return dstEnc;
  }
  return std::nullopt;
}

static std::optional<Attribute> inferDstEncoding(SplitOp op, Attribute srcEnc) {
  Attribute dstEnc;
  if (srcEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferSplitOpEncoding(srcEnc, dstEnc,
                                 /*loc=*/std::nullopt)
          .succeeded()) {
    return dstEnc;
  }
  return std::nullopt;
}

static std::optional<Attribute> inferSrcEncoding(triton::ReduceOp op,
                                                 Attribute encoding) {
  auto sliceEncoding = mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);
  if (!sliceEncoding)
    return std::nullopt;
  if (op.getAxis() != sliceEncoding.getDim())
    return std::nullopt;
  return sliceEncoding.getParent();
}

static std::optional<Attribute> inferSrcEncoding(triton::ExpandDimsOp op,
                                                 Attribute encoding) {
  return triton::gpu::SliceEncodingAttr::get(op->getContext(), op.getAxis(),
                                             encoding, false);
  // FIXME: Shall we support noWarpReduce filed for ExpandDimsOp?
}

static std::optional<Attribute> inferSrcEncoding(JoinOp op, Attribute dstEnc) {
  // Split is the inverse of join.
  Attribute srcEnc;
  if (dstEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferSplitOpEncoding(dstEnc, srcEnc, /*loc=*/std::nullopt)
          .succeeded()) {
    return srcEnc;
  }
  return std::nullopt;
}

static std::optional<Attribute> inferSrcEncoding(SplitOp op, Attribute dstEnc) {
  // Join is the inverse of split.
  Attribute srcEnc;
  if (dstEnc.getDialect()
          .getRegisteredInterface<DialectInferLayoutInterface>()
          ->inferJoinOpEncoding(dstEnc, srcEnc, /*loc=*/std::nullopt)
          .succeeded()) {
    return srcEnc;
  }
  return std::nullopt;
}

static std::optional<Attribute>
inferTransOpDstEncoding(Attribute srcEnc, ArrayRef<int32_t> order) {
  // Simply forward to the existing inferTransOpEncoding function.
  Attribute retEncoding;
  if (succeeded(
          srcEnc.getDialect()
              .getRegisteredInterface<triton::DialectInferLayoutInterface>()
              ->inferTransOpEncoding(srcEnc, order, retEncoding))) {
    return retEncoding;
  }
  return std::nullopt;
}

static std::optional<Attribute> inferDstEncoding(triton::TransOp op,
                                                 Attribute encoding) {
  return inferTransOpDstEncoding(encoding, op.getOrder());
}

static std::optional<Attribute> inferSrcEncoding(triton::TransOp op,
                                                 Attribute encoding) {
  // We want to solve for srcEnc in
  //   transpose(srcEnc, order) -> dstEnc.
  // Given the identity
  //   transpose(transpose(x, order), inverse(order)) == x,
  // we can see this is equivalent to
  //   transpose(dstEnc, inverse(order)) -> srcEnc.
  return inferTransOpDstEncoding(encoding,
                                 triton::inversePermutation(op.getOrder()));
}

static std::optional<Attribute>
inferReshapeOpDstEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                          ArrayRef<int64_t> dstShape, bool allowReorder) {
  // We don't do anything smart to allow-reorder reshapes here.  They are
  // handled in OptimizeThreadLocality.
  if (allowReorder)
    return std::nullopt;

  Attribute dstEnc;
  if (succeeded(
          srcEnc.getDialect()
              .getRegisteredInterface<triton::DialectInferLayoutInterface>()
              ->inferReshapeOpNoReorderEncoding(
                  srcShape, srcEnc, dstShape, dstEnc, /*loc=*/std::nullopt))) {
    return dstEnc;
  }
  return std::nullopt;
}

static std::optional<Attribute> inferDstEncoding(triton::ReshapeOp op,
                                                 Attribute encoding) {
  return inferReshapeOpDstEncoding(op.getSrc().getType().getShape(), encoding,
                                   op.getType().getShape(),
                                   op.getAllowReorder());
}

static std::optional<Attribute> inferSrcEncoding(triton::ReshapeOp op,
                                                 Attribute encoding) {
  // The encoding of x given the encoding of y in `reshape(x) -> y` is the same
  // as the encoding of x given the encoding of y in `reshape(y) -> x`.  It's an
  // invariant of inferReshapeOpNoReorderEncoding that it's symmetric in this
  // way.
  return inferReshapeOpDstEncoding(op.getType().getShape(), encoding,
                                   op.getSrc().getType().getShape(),
                                   op.getAllowReorder());
}

std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding) {
  if (isa<triton::ScanOp>(op)) {
    // Scan only supports blocked encoding at the moment.
    if (!isa<triton::gpu::BlockedEncodingAttr>(encoding))
      return std::nullopt;
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<scf::WhileOp, scf::YieldOp, scf::ConditionOp>(op)) {
    return encoding;
  }

  if (auto reduceOp = dyn_cast<triton::ReduceOp>(op))
    return inferSrcEncoding(reduceOp, encoding);
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
    return inferSrcEncoding(expand, encoding);
  if (auto join = dyn_cast<triton::JoinOp>(op))
    return inferSrcEncoding(join, encoding);
  if (auto split = dyn_cast<triton::SplitOp>(op))
    return inferSrcEncoding(split, encoding);
  if (auto trans = dyn_cast<triton::TransOp>(op))
    return inferSrcEncoding(trans, encoding);
  if (auto reshape = dyn_cast<triton::ReshapeOp>(op))
    return inferSrcEncoding(reshape, encoding);
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return encoding;

  return std::nullopt;
}

std::optional<Attribute> inferDstEncoding(Operation *op, Attribute encoding) {
  if (isa<triton::ScanOp>(op)) {
    if (!isa<triton::gpu::BlockedEncodingAttr>(encoding))
      return std::nullopt;
  }
  if (op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::SameLoadStoreOperandsAndResultEncoding>() ||
      op->hasTrait<mlir::OpTrait::Elementwise>() ||
      isa<scf::WhileOp, scf::ForOp, scf::YieldOp, scf::ConditionOp>(op))
    return encoding;
  if (auto reduceOp = dyn_cast<triton::ReduceOp>(op))
    return inferDstEncoding(reduceOp, encoding);
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op))
    return inferDstEncoding(expand, encoding);
  if (auto join = dyn_cast<triton::JoinOp>(op))
    return inferDstEncoding(join, encoding);
  if (auto split = dyn_cast<triton::SplitOp>(op))
    return inferDstEncoding(split, encoding);
  if (auto trans = dyn_cast<triton::TransOp>(op))
    return inferDstEncoding(trans, encoding);
  if (auto reshape = dyn_cast<triton::ReshapeOp>(op))
    return inferDstEncoding(reshape, encoding);

  return std::nullopt;
}

} // namespace mlir
