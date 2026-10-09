#include "triton/Conversion/CommonIRToTTGIR/Passes.h"
#include "triton/Conversion/CommonIRToTTGIR/TensorOps.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"

#include <limits>
#include <numeric>

namespace mlir::triton::metax {

#define GEN_PASS_DEF_COMMONIRLOWERTENSORTILES
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

// Preserve the logical ops while TTIR -> TTGIR assigns their external layouts.
// Physical register selection is handled by the separate lowering pass below.
template <typename Op> struct TensorTileTypePattern : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto fullTy = cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    auto resultTy = cast<RankedTensorType>(op.getResult().getType())
                        .cloneWithEncoding(fullTy.getEncoding());
    rewriter.template replaceOpWithNewOp<Op>(
        op, TypeRange{resultTy}, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct TensorTilePlan {
  Operation *op;
  RankedTensorType fullType;
  RankedTensorType tileType;
  DenseI64ArrayAttr ctaIndices;
  DenseI64ArrayAttr elementIndices;
};

// The native MACA view lowering sorts the selected register slots in its MMA
// ABI order. Check their logical coordinates too: inferred shape/cardinality
// alone does not prove a same-lane rectangular slice, especially for colMajor.
static bool preservesMmaCoordinates(RankedTensorType fullTy,
                                    RankedTensorType tileTy,
                                    ArrayRef<int64_t> tileIndex) {
  auto mma = cast<ttg::MACAMmaEncodingAttr>(fullTy.getEncoding());
  auto full = ttg::toLinearLayout(fullTy);
  auto part = ttg::toLinearLayout(tileTy);
  auto reg = StringAttr::get(fullTy.getContext(), "register");
  for (auto &[dim, bases] : full.getBases())
    if (dim != reg && bases != part.getBases().lookup(dim))
      return false;

  auto unit = ttg::getShapePerCTATile(fullTy);
  int64_t repN = fullTy.getDimSize(1) / unit[1];
  int64_t elemM = mma.getElementsMNK()[0], elemN = mma.getElementsMNK()[1];
  int64_t beginM = tileIndex[0] * tileTy.getDimSize(0) / unit[0];
  int64_t beginN = tileIndex[1] * tileTy.getDimSize(1) / unit[1];
  int64_t countM = tileTy.getDimSize(0) / unit[0];
  int64_t countN = tileTy.getDimSize(1) / unit[1];
  auto coordinates = [&](const LinearLayout &layout, int32_t slot) {
    SmallVector<std::pair<StringAttr, int32_t>> inputs;
    for (auto dim : layout.getInDimNames())
      inputs.emplace_back(dim, dim == reg ? slot : 0);
    return layout.apply(inputs);
  };
  int32_t selected = 0;
  for (int32_t slot = 0; slot < full.getInDimSize(reg); ++slot) {
    // Matches emitSubOffsetForMmaLayoutMACA's physical register order.
    int64_t m = slot / (4 * elemM * elemN * repN);
    int64_t n = (slot / (4 * elemN * (mma.getColMajor() ? 1 : elemM))) % repN;
    if (m < beginM || m >= beginM + countM || n < beginN ||
        n >= beginN + countN)
      continue;
    if (selected >= part.getInDimSize(reg))
      return false;
    auto src = coordinates(full, slot);
    auto dst = coordinates(part, selected++);
    for (auto [s, d] : llvm::zip(src, dst)) {
      unsigned axis = s.first.getValue() == "dim0" ? 0 : 1;
      if (s.first != d.first ||
          s.second != d.second + tileIndex[axis] * tileTy.getDimSize(axis))
        return false;
    }
  }
  return selected == part.getInDimSize(reg);
}

// Select complete CTA-sized register replicas, retaining the encoding.
static FailureOr<TensorTilePlan>
planForEncoding(Operation *op, RankedTensorType fullTy, RankedTensorType tileTy,
                ArrayRef<int64_t> tileIndex, Attribute encoding) {
  SmallVector<unsigned> order, elements;
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding)) {
    order = llvm::to_vector(blocked.getOrder());
    elements = llvm::to_vector(blocked.getSizePerThread());
  } else if (auto mma = dyn_cast<ttg::MACAMmaEncodingAttr>(encoding)) {
    if (fullTy.getRank() != 2 || mma.getVersionMajor() != 2 ||
        mma.getIsATrans() || mma.getIsBTrans())
      return failure();
    order = mma.getColMajor() ? SmallVector<unsigned>{0, 1}
                              : SmallVector<unsigned>{1, 0};
    elements = {mma.getElementsMNK()[0], mma.getElementsMNK()[1]};
  } else {
    return failure();
  }
  fullTy = fullTy.cloneWithEncoding(encoding);
  tileTy = tileTy.cloneWithEncoding(encoding);
  auto shapePerCTA = ttg::getShapePerCTATile(fullTy);
  SmallVector<int64_t> replicas, selectedReplicas;
  int64_t selectedCount = 1;
  for (int d = 0; d < fullTy.getRank(); ++d) {
    auto unit = shapePerCTA[d];
    if (!unit || fullTy.getDimSize(d) % unit || tileTy.getDimSize(d) % unit)
      return failure();
    replicas.push_back(fullTy.getDimSize(d) / unit);
    selectedReplicas.push_back(tileTy.getDimSize(d) / unit);
    selectedCount *= selectedReplicas.back();
  }

  SmallVector<int64_t> ctaIndices;
  for (int64_t i = 0; i < selectedCount; ++i) {
    int64_t remaining = i, linear = 0, stride = 1;
    for (unsigned d : order) {
      int64_t coordinate =
          tileIndex[d] * selectedReplicas[d] + remaining % selectedReplicas[d];
      remaining /= selectedReplicas[d];
      linear += coordinate * stride;
      stride *= replicas[d];
    }
    ctaIndices.push_back(linear);
  }
  unsigned elementsPerReplica = 1;
  for (unsigned size : elements)
    elementsPerReplica *= size;
  SmallVector<int64_t> elementIndices(elementsPerReplica);
  std::iota(elementIndices.begin(), elementIndices.end(), 0);

  auto ctaAttr = DenseI64ArrayAttr::get(op->getContext(), ctaIndices);
  auto elementAttr = DenseI64ArrayAttr::get(op->getContext(), elementIndices);
  ttg::ExtractTensorOp::Properties properties;
  properties.ctaIdx = ctaAttr;
  properties.elemIdx = elementAttr;
  Block block;
  Value source = block.addArgument(fullTy, op->getLoc());
  SmallVector<Type> inferredTypes;
  if (failed(ttg::ExtractTensorOp::inferReturnTypes(
          op->getContext(), op->getLoc(), ValueRange{source}, DictionaryAttr(),
          OpaqueProperties(&properties), RegionRange(), inferredTypes)) ||
      inferredTypes.size() != 1)
    return failure();
  auto inferredTy = dyn_cast<RankedTensorType>(inferredTypes.front());
  // Native type inference can normalize order on dimensions with only one
  // register. Accept that spelling only if it preserves the physical mapping.
  if (!inferredTy || inferredTy.getShape() != tileTy.getShape() ||
      inferredTy.getElementType() != tileTy.getElementType() ||
      ttg::toLinearLayout(inferredTy) != ttg::toLinearLayout(tileTy))
    return failure();
  if (isa<ttg::MACAMmaEncodingAttr>(encoding) &&
      !preservesMmaCoordinates(fullTy, tileTy, tileIndex))
    return failure();
  return TensorTilePlan{op, fullTy, inferredTy, ctaAttr, elementAttr};
}

static FailureOr<SmallVector<int64_t>> getTileIndex(Operation *op,
                                                    bool diagnose) {
  auto reject = [&](StringRef message) -> LogicalResult {
    if (diagnose)
      op->emitError(message);
    return failure();
  };
  bool isExtract = isa<tile::ExtractTileOp>(op);
  auto fullTy = cast<RankedTensorType>(op->getOperand(0).getType());
  auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
  auto tileTy = isExtract ? resultTy
                          : cast<RankedTensorType>(op->getOperand(1).getType());
  if (fullTy.getRank() == 0 || fullTy.getRank() != tileTy.getRank() ||
      fullTy.getElementType() != tileTy.getElementType() ||
      (!isExtract && resultTy != fullTy))
    return reject("incompatible tensor tile types");
  if (ttg::lookupNumCTAs(op) != 1)
    return reject("tensor tile lowering currently requires one CTA");

  APInt index;
  if (!matchPattern(op->getOperand(isExtract ? 1 : 2), m_ConstantInt(&index)))
    return reject("tensor tile lowering currently requires a constant index");
  if (index.getBitWidth() > 64 || index.isNegative())
    return reject("tensor tile index is out of bounds");

  SmallVector<int64_t> grid;
  uint64_t fullElements = 1;
  int64_t totalTiles = 1;
  for (auto [fullDim, tileDim] :
       llvm::zip(fullTy.getShape(), tileTy.getShape())) {
    if (fullDim <= 0 || tileDim <= 0 || fullDim % tileDim)
      return reject(
          "source shape must be divisible by positive tile dimensions");
    auto next = llvm::checkedMulUnsigned(fullElements, uint64_t(fullDim));
    if (!next || *next > std::numeric_limits<int32_t>::max())
      return reject("tensor shape exceeds the register indexing range");
    fullElements = *next;
    grid.push_back(fullDim / tileDim);
    totalTiles *= grid.back();
  }
  int64_t remaining = index.getSExtValue();
  if (remaining >= totalTiles)
    return reject("tensor tile index is out of bounds");
  SmallVector<int64_t> tileIndex(grid.size());
  for (int d = grid.size() - 1; d >= 0; --d) {
    tileIndex[d] = remaining % grid[d];
    remaining /= grid[d];
  }
  return tileIndex;
}

static FailureOr<TensorTilePlan> planTile(Operation *op) {
  auto tileIndex = getTileIndex(op, /*diagnose=*/true);
  if (failed(tileIndex))
    return failure();
  auto fullTy = cast<RankedTensorType>(op->getOperand(0).getType());
  auto tileTy = cast<RankedTensorType>(isa<tile::ExtractTileOp>(op)
                                           ? op->getResult(0).getType()
                                           : op->getOperand(1).getType());
  Attribute encoding = fullTy.getEncoding();
  if (!isa_and_nonnull<ttg::BlockedEncodingAttr, ttg::MACAMmaEncodingAttr>(
          encoding) ||
      !isa_and_nonnull<ttg::BlockedEncodingAttr, ttg::MACAMmaEncodingAttr>(
          tileTy.getEncoding()))
    return op->emitError(
        "tensor tile lowering requires blocked or MACA MMA layouts");
  auto module = op->getParentOfType<ModuleOp>();
  if (isa<ttg::MACAMmaEncodingAttr>(encoding))
    if (auto direct = planForEncoding(op, fullTy, tileTy, *tileIndex, encoding);
        succeeded(direct))
      return direct;

  auto tryEncoding =
      [&](ttg::BlockedEncodingAttr candidate) -> FailureOr<TensorTilePlan> {
    if (auto direct =
            planForEncoding(op, fullTy, tileTy, *tileIndex, candidate);
        succeeded(direct))
      return direct;
    // Native extract inference takes its order from the register bases. When
    // an axis has no register replicas this can differ from the blocked order,
    // also changing lane/warp placement. Re-plan with that order explicitly;
    // the surrounding conversions preserve the original logical tensor.
    auto order = ttg::getOrder(fullTy.cloneWithEncoding(candidate));
    if (llvm::equal(order, candidate.getOrder()))
      return failure();
    auto normalized = ttg::BlockedEncodingAttr::get(
        op->getContext(), candidate.getSizePerThread(),
        candidate.getThreadsPerWarp(), candidate.getWarpsPerCTA(), order,
        candidate.getCTALayout());
    return planForEncoding(op, fullTy, tileTy, *tileIndex, normalized);
  };
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(encoding))
    if (auto direct = tryEncoding(blocked); succeeded(direct))
      return direct;

  // As in Gluon's blocked slice legalization, fit the thread distribution to
  // the logical tile. Keep both public tensor types via convert_layout.
  auto order =
      isa<ttg::BlockedEncodingAttr>(encoding)
          ? llvm::to_vector(cast<ttg::BlockedEncodingAttr>(encoding).getOrder())
          : ttg::getOrder(fullTy);
  auto carrier = ttg::BlockedEncodingAttr::get(
      op->getContext(), tileTy.getShape(),
      SmallVector<unsigned>(tileTy.getRank(), 1), order,
      ttg::lookupNumWarps(op), ttg::TritonGPUDialect::getThreadsPerWarp(module),
      1);
  if (auto converted = tryEncoding(carrier); succeeded(converted))
    return converted;
  return op->emitError(
             "cannot represent tensor tile with a blocked register slice")
         << ": source " << fullTy << ", tile " << tileTy.getShape();
}

static Value convertLayout(OpBuilder &builder, Location loc, Value value,
                           RankedTensorType type) {
  if (value.getType() == type)
    return value;
  return ttg::ConvertLayoutOp::create(builder, loc, type, value);
}

struct CommonIRLowerTensorTilesPass
    : impl::CommonIRLowerTensorTilesBase<CommonIRLowerTensorTilesPass> {
  void runOnOperation() override {
    SmallVector<TensorTilePlan> plans;
    WalkResult result = getOperation().walk([&](Operation *op) {
      if (!isa<tile::ExtractTileOp, tile::InsertTileOp>(op))
        return WalkResult::advance();
      auto plan = planTile(op);
      if (failed(plan))
        return WalkResult::interrupt();
      plans.push_back(*plan);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    for (const auto &plan : plans) {
      Operation *op = plan.op;
      OpBuilder builder(op);
      Location loc = op->getLoc();
      Value full =
          convertLayout(builder, loc, op->getOperand(0), plan.fullType);
      Value replacement;
      if (isa<tile::ExtractTileOp>(op)) {
        replacement = builder.create<ttg::ExtractTensorOp>(
            loc, plan.tileType, full, plan.ctaIndices, plan.elementIndices);
      } else {
        Value update =
            convertLayout(builder, loc, op->getOperand(1), plan.tileType);
        replacement = builder.create<ttg::InsertTensorOp>(
            loc, plan.fullType, full, update, plan.ctaIndices,
            plan.elementIndices);
      }
      replacement =
          convertLayout(builder, loc, replacement,
                        cast<RankedTensorType>(op->getResult(0).getType()));
      op->getResult(0).replaceAllUsesWith(replacement);
      op->erase();
    }
  }
};

} // namespace

bool canLowerTensorTileInEncoding(Operation *op, Attribute encoding) {
  if (!isa<tile::ExtractTileOp, tile::InsertTileOp>(op))
    return false;
  auto index = getTileIndex(op, /*diagnose=*/false);
  if (failed(index))
    return false;
  auto fullTy = cast<RankedTensorType>(op->getOperand(0).getType());
  auto tileTy = cast<RankedTensorType>(isa<tile::ExtractTileOp>(op)
                                           ? op->getResult(0).getType()
                                           : op->getOperand(1).getType());
  auto plan = planForEncoding(op, fullTy, tileTy, *index, encoding);
  return succeeded(plan) && plan->tileType.getEncoding() == encoding;
}

void populateCommonIRTensorPatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target) {
  patterns.add<TensorTileTypePattern<tile::ExtractTileOp>,
               TensorTileTypePattern<tile::InsertTileOp>>(
      typeConverter, patterns.getContext());
  target.addDynamicallyLegalOp<tile::ExtractTileOp, tile::InsertTileOp>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
}

} // namespace mlir::triton::metax
