#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "triton/Analysis/AxisInfo.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "ttg-utility"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

using namespace triton;

SmallVector<unsigned, 3> mmaVersionToInstrShape(int version,
                                                const ArrayRef<int64_t> &shape,
                                                TensorOrMemDesc type,
                                                int numWarps) {
  if (version == 1)
    return {16, 16};
  else if (version == 2) {
    auto rank = shape.size();
    SmallVector<unsigned, 3> ret(rank, 1);
    ret[rank - 1] = 8;
    ret[rank - 2] = 16;
    return ret;
  } else if (version == 3) {
    unsigned k = 256 / type.getElementTypeBitWidth();
    if (shape[0] % 64 != 0 || shape[1] % 8 != 0) {
      assert(false && "type not supported");
      return {0, 0, 0};
    }
    auto eltType = type.getElementType();
    SmallVector<unsigned> validN;

    // MMAv3 with larger instruction shape is preferred.
    if (eltType.isFloat8E5M2() || eltType.isFloat8E4M3FN() ||
        eltType.isFloat8E4M3FNUZ() || eltType.isF16() || eltType.isBF16() ||
        eltType.isF32()) {
      validN.assign({256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                     168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,
                     80,  72,  64,  56,  48,  40,  32,  24,  16,  8});
    }

    if (eltType.isInteger(8)) {
      validN.assign({224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32,
                     24, 16, 8});
    }

    unsigned m = 16;
    unsigned mWarps = std::max<unsigned>(shape[0] / m, 1);
    unsigned nWarps = std::max<unsigned>(numWarps / mWarps, 1);
    unsigned maxN = std::max<unsigned>(shape[1] / nWarps, 8);
    for (auto n : validN) {
      if (shape[1] % n == 0 && n <= maxN) {
        return {m, n, k};
      }
    }

    assert(false && "type not supported");
    return {0, 0, 0};
  } else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

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

// Check if the convert will be a no-op in codegen.
static bool isFreeConvert(Operation *op) {
  auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!convertOp)
    return false;
  return isMmaToMmaShortcut(convertOp.getSrc().getType(), convertOp.getType());
}

LogicalResult
getConvertBackwardSlice(Value root, SetVector<Value> &slice,
                        Attribute rootEncoding,
                        DenseMap<Value, Attribute> &layout,
                        std::function<bool(Operation *)> stopPropagation) {
  DenseSet<std::pair<Value, Attribute>> seen;
  SmallVector<std::pair<Value, Attribute>> queue;

  auto enqueue = [&](Value operand, Attribute encoding) {
    auto x = std::make_pair(operand, encoding);
    if (!seen.insert(x).second) {
      return; // Already enqueued, skip
    }
    queue.push_back(x);
  };
  enqueue(root, rootEncoding);

  while (!queue.empty()) {
    auto [currentValue, encoding] = queue.back();
    queue.pop_back();
    if (!isa<RankedTensorType>(currentValue.getType()))
      continue;
#ifndef __ILUVATAR__
    // Skip propagating through for op results for now.
    // TODO: enable this based on needs.
    if (currentValue.getDefiningOp<scf::ForOp>())
      return failure();
#endif
    slice.insert(currentValue);
    if (layout.find(currentValue) != layout.end()) {
      if (layout[currentValue] != encoding)
        return failure();
    }
    layout[currentValue] = encoding;

    if (auto ifOp = currentValue.getDefiningOp<scf::IfOp>()) {
      auto results = ifOp.getResults();
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();

      auto thenValue = ifOp.thenYield().getOperand(argIdx);
      auto elseValue = ifOp.elseYield().getOperand(argIdx);

      enqueue(thenValue, encoding);
      enqueue(elseValue, encoding);

      continue;
    }
#ifdef __ILUVATAR__
    if (auto forOp = currentValue.getDefiningOp<scf::ForOp>()) {
      if (auto blkEncoding =
              dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding)) {
        if (blkEncoding.getLoadType() != 1 || !blkEncoding.getSmeMask())
          return failure();
      } else {
        return failure();
      }
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();
      Value yieldOperand = forOp.getBody()->getTerminator()->getOperand(argIdx);
      enqueue(yieldOperand, encoding);
      continue;
    }
#endif

    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (Value result : definingOp->getResults()) {
        if (result == currentValue || !isa<RankedTensorType>(result.getType()))
          continue;
        enqueue(result, encoding);
      }
      if (!isFreeConvert(definingOp) &&
          canFoldIntoConversion(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
      for (Value operand : definingOp->getOperands()) {
        auto srcEncoding = inferSrcEncoding(definingOp, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(operand, *srcEncoding);
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
      Value yieldOperand = forOp.getBody()->getTerminator()->getOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      enqueue(initOperand->get(), encoding);
      enqueue(yieldOperand, encoding);
      continue;
    }
    // TODO: add support for WhileOp and other region types.
    return failure();
  }
  return success();
}

} // namespace mlir
