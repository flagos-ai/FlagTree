#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "MapElementwiseExpansion.h"
#include "TritonToTileIR/Passes.h"
#include "TritonToTileIR/TritonToTileIRPass.h"
#include "TritonToTileIR/Utils.h"
#include "Utils/Utils.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "triton/Analysis/AxisInfo.h"
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_set>

// MLIR pass TableGen uses per-pass macros (GEN_PASS_DEF_*).
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONTOCUDATILE
#include "TritonToTileIR/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::bridge_utils;

// We can safely assume that the pointer and strides in TMA descriptors are
// divisible by 16. (Sizes can do not have this divisibility requirement.)
static constexpr int64_t kTMAAlignment = 16;
// TMA hardware limit for stride.
constexpr unsigned long long kMaxStride = (1LL << 40) - 1;
// TMA hardware limit for shape.
constexpr unsigned long long kMaxShape = (1LL << 32) - 1;

//
// CudaTileConversion
//
class CudaTileConversionTarget : public ConversionTarget {
public:
  CudaTileConversionTarget(MLIRContext &context,
                           CudaTileTypeConverter &typeConverter)
      : ConversionTarget(context) {

    addLegalDialect<cuda_tile::CudaTileDialect>();
    addIllegalDialect<scf::SCFDialect, cf::ControlFlowDialect,
                      mlir::gpu::GPUDialect, triton::TritonDialect,
                      ub::UBDialect>();

    addLegalOp<ub::PoisonOp>();
    // barrierOp will be removed in AutoGenMemoryTokenPass
    addLegalOp<mlir::gpu::BarrierOp>();

    // TODO: support these arith/math ops in cuda_tile
    addLegalOp<arith::IndexCastOp>();

    // TODO: remove these ops
    addLegalOp<UnrealizedConversionCastOp>();
  }
};

template <typename OpTy, typename DestOpTy>
static LogicalResult rewriteReshapeLike(const TypeConverter *typeConverter,
                                        OpTy op, typename OpTy::Adaptor adaptor,
                                        ConversionPatternRewriter &rewriter) {
  RankedTensorType sourceTensorType = op.getSrc().getType();
  RankedTensorType resultTensorType = op.getResult().getType();

  // If source and result types are matching, those are no-ops.
  if (sourceTensorType == resultTensorType) {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }

  auto newResultType = typeConverter->convertType(resultTensorType);
  assert(newResultType && "failed to convert tensor type");
  rewriter.replaceOpWithNewOp<DestOpTy>(op, newResultType, adaptor.getSrc());
  return success();
}

static DenseIntOrFPElementsAttr
convertArithAttrToCudaTileAttr(const TypedAttr &attr,
                               const ShapedType &shapeType) {
  auto shape = shapeType.getShape();
  if (auto arithDense = dyn_cast<DenseElementsAttr>(attr))
    return cast<DenseIntOrFPElementsAttr>(DenseElementsAttr::getFromRawBuffer(
        shapeType, arithDense.getRawData()));
  return cast<DenseIntOrFPElementsAttr>(
      DenseElementsAttr::get(shapeType, {cast<Attribute>(attr)}));
}

class ConvertAbsFOp : public OpConversionPattern<math::AbsFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::AbsFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    cuda_tile::TileType newResultTensorType = cast<cuda_tile::TileType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    auto eltype = newResultTensorType.getElementType();
    if (eltype.getIntOrFloatBitWidth() < 16) {
      // f8 and f4 not directly supported, upcast to fp16 and downcast after
      cuda_tile::TileType f16TensorType = cuda_tile::TileType::get(
          newResultTensorType.getShape(), rewriter.getF16Type());
      auto upcast = cuda_tile::FToFOp::create(rewriter, loc, f16TensorType,
                                              adaptor.getOperand());
      auto absop = cuda_tile::AbsFOp::create(rewriter, loc, upcast);
      rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(op, newResultTensorType,
                                                     absop);
      return success();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::AbsFOp>(op, adaptor.getOperand());
    return success();
  }
};

class ConvertConstantOp : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    auto loc = op.getLoc();
    auto tensorType = converter->convertType(op.getValueAttr().getType());
    if (!isa<ShapedType>(tensorType))
      return rewriter.notifyMatchFailure(loc,
                                         "typeConversion of current op failed");

    DenseIntOrFPElementsAttr tensorAttr = convertArithAttrToCudaTileAttr(
        adaptor.getValueAttr(), cast<ShapedType>(tensorType));

    SmallVector<Type> retTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), retTypes)))
      return rewriter.notifyMatchFailure(loc,
                                         "typeConversion of current op failed");

    rewriter.replaceOpWithNewOp<cuda_tile::ConstantOp>(op, retTypes,
                                                       tensorAttr);
    return success();
  }
};

class ConvertSelectOp : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto intTy = dyn_cast<IntegerType>(op.getCondition().getType())) {
      int rank = cast<ShapedType>(adaptor.getTrueValue().getType()).getRank();
      SmallVector<int64_t> reshapedView(rank, 1);
      auto newCondType1 =
          cuda_tile::TileType::get(reshapedView, op.getCondition().getType());

      auto reshapeOp = cuda_tile::ReshapeOp::create(
          rewriter, op.getLoc(), newCondType1, adaptor.getCondition());

      auto newShape =
          cast<ShapedType>(adaptor.getTrueValue().getType()).getShape();
      auto newCondType2 =
          cuda_tile::TileType::get(newShape, op.getCondition().getType());

      auto broadcastOp = cuda_tile::BroadcastOp::create(
          rewriter, op.getLoc(), newCondType2, reshapeOp.getResult());

      rewriter.replaceOpWithNewOp<cuda_tile::SelectOp>(
          op, broadcastOp.getResult(), adaptor.getTrueValue(),
          adaptor.getFalseValue());
      return success();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::SelectOp>(op, adaptor.getCondition(),
                                                     adaptor.getTrueValue(),
                                                     adaptor.getFalseValue());
    return success();
  }
};

class ConvertReturnOp : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cuda_tile::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class ConvertPrintOp : public OpConversionPattern<triton::PrintOp> {
public:
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto prefix = op.getPrefix().str();
    auto args = adaptor.getArgs();
    std::string newPrefix = prefix;
    for (Value arg : args) {
      auto tileType = cast<cuda_tile::TileType>(arg.getType());
      Type elType = tileType.getElementType();
      if (isa<IntegerType>(elType)) {
        newPrefix += "%i";
      } else if (isa<FloatType>(elType)) {
        newPrefix += "%.5f";
      } else if (isa<cuda_tile::PointerType>(elType)) {
        newPrefix += "%p";
      } else {
        llvm::report_fatal_error("unsupported type");
      }
    }
    newPrefix += "\n";

    cuda_tile::PrintTkoOp::create(rewriter, op.getLoc(), newPrefix, args,
                                  /*token=*/Value());
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLoadOp : public OpConversionPattern<triton::LoadOp> {
public:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;
  ConvertLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                DenseMap<Operation *, int> &numStagesMap, int computeCapability,
                std::optional<int> numStages)
      : OpConversionPattern<triton::LoadOp>(typeConverter, context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, typename triton::LoadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto ctx = rewriter.getContext();

    auto sem = cuda_tile::MemoryOrderingSemanticsAttr::get(
        ctx, cuda_tile::MemoryOrderingSemantics::WEAK);
    // There is no per op num_stages support in triton, and usually the
    // global-scope num_stages is meant to tune for the large load op which are
    // critical to performance. For small load op, we skip the num_stages hint.
    auto tileType = cast<cuda_tile::TileType>(retType);
    int64_t tileSizeInBytes =
        tileType.getNumElements() *
        (tileType.getElementType().getIntOrFloatBitWidth() / 8);
    std::optional<cuda_tile::OptimizationHintsAttr> optHint;
    if (tileSizeInBytes <= 256) {
      optHint = std::nullopt;
    } else {
      optHint = mlir::triton::utils::convertNumStagesToOptHint(
          op, ctx, numStagesMap, computeCapability, numStages);
    }

    auto newLoadOp = cuda_tile::LoadPtrTkoOp::create(
        rewriter, op.getLoc(), retType, cuda_tile::TokenType::get(ctx), sem,
        /*memoryScope=*/nullptr, adaptor.getPtr(), adaptor.getMask(),
        adaptor.getOther(), /*token=*/nullptr, optHint.value_or(nullptr));
    rewriter.replaceOp(op, newLoadOp.getResult());
    return success();
  }
};

class ConvertStoreOp : public OpConversionPattern<triton::StoreOp> {
public:
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;
  ConvertStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                 DenseMap<Operation *, int> &numStagesMap,
                 int computeCapability, std::optional<int> numStages)
      : OpConversionPattern<triton::StoreOp>(typeConverter, context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, typename triton::StoreOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto sem = cuda_tile::MemoryOrderingSemanticsAttr::get(
        ctx, cuda_tile::MemoryOrderingSemantics::WEAK);

    // There is no per op num_stages support in triton, and usually the
    // global-scope num_stages is meant to tune for the large store op which are
    // critical to performance. For small store op, we skip the num_stages hint.
    auto tileType = cast<cuda_tile::TileType>(adaptor.getValue().getType());
    int64_t tileSizeInBytes =
        tileType.getNumElements() *
        (tileType.getElementType().getIntOrFloatBitWidth() / 8);
    std::optional<cuda_tile::OptimizationHintsAttr> optHint;
    if (tileSizeInBytes <= 256) {
      optHint = std::nullopt;
    } else {
      optHint = mlir::triton::utils::convertNumStagesToOptHint(
          op, ctx, numStagesMap, computeCapability, numStages);
    }

    cuda_tile::StorePtrTkoOp::create(
        rewriter, op.getLoc(), cuda_tile::TokenType::get(ctx), sem,
        /*memoryScope=*/nullptr, adaptor.getPtr(), adaptor.getValue(),
        adaptor.getMask(), /*token=*/nullptr, optHint.value_or(nullptr));
    rewriter.eraseOp(op);
    return success();
  }
};

// Helper function to create target operations (FuncOp or EntryOp)
template <typename TargetOp>
void createTargetOp(ConversionPatternRewriter &rewriter, triton::FuncOp op,
                    FunctionType newFunctionType,
                    TypeConverter::SignatureConversion &result,
                    int computeCapability, int numCTAInCGA,
                    int simtNumWarpsInCTA, int occupancy) {
  MLIRContext *ctx = rewriter.getContext();

  auto newKernelName = op.getSymName();

  TargetOp newOp;
  if constexpr (std::is_same_v<TargetOp, cuda_tile::EntryOp>) {
    llvm::StringRef archPrefix = "sm_";
    auto numCTAAttr = rewriter.getNamedAttr(
        "num_cta_in_cga", rewriter.getI32IntegerAttr(numCTAInCGA));
    auto occupancyAttr = rewriter.getNamedAttr(
        "occupancy", rewriter.getI32IntegerAttr(occupancy));
    auto hintEntry = rewriter.getNamedAttr(
        (archPrefix + llvm::Twine(computeCapability)).str(),
        DictionaryAttr::get(ctx, {numCTAAttr, occupancyAttr}));
    cuda_tile::OptimizationHintsAttr optHint =
        cuda_tile::OptimizationHintsAttr::get(
            ctx, DictionaryAttr::get(ctx, {hintEntry}));
    newOp = cuda_tile::EntryOp::create(
        rewriter, op.getLoc(), StringAttr::get(ctx, newKernelName),
        newFunctionType, op.getArgAttrsAttr(), op.getResAttrsAttr(), optHint);
  } else {
    static_assert(std::is_same_v<TargetOp, void>, "unsupported TargetOp type");
  }

  for (unsigned i = 0, e = newOp.getNumArguments(); i < e; ++i) {
    if (newOp.getArgAttr(i, "tt.divisibility")) {
      newOp.removeArgAttr(i, "tt.divisibility");
    }
  }

  Block *oldBody = &op.getFunctionBody().front();
  (void)rewriter.applySignatureConversion(oldBody, result);
  rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                              newOp.getBody().end());
  rewriter.replaceOp(op, newOp);
}

class ConvertFuncOp : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;
  int computeCapability;
  int numCTAInCGA;
  int simtNumWarpsInCTA;
  int occupancy;
  ConvertFuncOp(TypeConverter &typeConverter, MLIRContext *context,
                int computeCapability, int numCTAInCGA, int simtNumWarpsInCTA,
                int occupancy)
      : OpConversionPattern(typeConverter, context),
        computeCapability(computeCapability), numCTAInCGA(numCTAInCGA),
        simtNumWarpsInCTA(simtNumWarpsInCTA), occupancy(occupancy) {};

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    FunctionType type = dyn_cast<FunctionType>(op.getFunctionType());

    TypeConverter::SignatureConversion result(type.getNumInputs());

    // Special treat for host tma descriptor:
    // We need to convert triton::TensorDescType to TileType<int> instead of
    // PartitionViewType, because cuda_tile does not allow view type in
    // signatures. Here we convert triton::TensorDescType to integer type, later
    // type converter will convert it to TileType<int> in the
    // convertSignatureArgs API.
    SmallVector<Type> modifiedInputs = llvm::to_vector(type.getInputs());
    for (size_t i = 0; i < modifiedInputs.size(); ++i) {
      if (auto tensorDescType =
              dyn_cast<triton::TensorDescType>(modifiedInputs[i])) {
        auto pointeeType = tensorDescType.getBlockType().getElementType();
        if (auto intTy = dyn_cast<mlir::IntegerType>(pointeeType)) {
          pointeeType =
              mlir::IntegerType::get(rewriter.getContext(), intTy.getWidth(),
                                     mlir::IntegerType::Signless);
        }
        modifiedInputs[i] = rewriter.getI32Type();
        if (i + 1 < modifiedInputs.size())
          modifiedInputs[i + 1] = triton::PointerType::get(pointeeType, 1);
      }
    }

    SmallVector<Type, 1> newResults;
    if (failed(typeConverter->convertSignatureArgs(modifiedInputs, result)) ||
        failed(typeConverter->convertTypes(type.getResults(), newResults)) ||
        failed(rewriter.convertRegionTypes(&op.getFunctionBody(),
                                           *typeConverter, &result)))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "typeConversion of FuncOp failed");

    auto context = rewriter.getContext();
    auto newFunctionType =
        FunctionType::get(context, result.getConvertedTypes(), newResults);

    createTargetOp<cuda_tile::EntryOp>(rewriter, op, newFunctionType, result,
                                       computeCapability, numCTAInCGA,
                                       simtNumWarpsInCTA, occupancy);
    return success();
    return success();
  }
};

class ConvertBitcastOp : public OpConversionPattern<triton::BitcastOp> {
public:
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tyCvt = getTypeConverter();
    auto ty = tyCvt->convertType(op.getResult().getType());
    auto resTyScalar = getElementTypeOrSelf(ty);
    auto srcTyScalar = getElementTypeOrSelf(adaptor.getSrc().getType());
    if (isa<cuda_tile::PointerType>(resTyScalar) &&
        isa<cuda_tile::PointerType>(srcTyScalar)) {
      rewriter.replaceOpWithNewOp<cuda_tile::PtrToPtrOp>(op, ty,
                                                         adaptor.getSrc());
      return success();
    }
    if (!isa<IntegerType, FloatType>(resTyScalar))
      return rewriter.notifyMatchFailure(
          op, "bitcast supports only integer or float types");
    rewriter.replaceOpWithNewOp<cuda_tile::BitcastOp>(op, ty, adaptor.getSrc());
    return success();
  }
};

class ConvertBroadCastOp : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteReshapeLike<triton::BroadcastOp, cuda_tile::BroadcastOp>(
        getTypeConverter(), op, adaptor, rewriter);
  }
};

class ConvertReshapeOp : public OpConversionPattern<triton::ReshapeOp> {
public:
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Investigate allow_reorder and efficient_layout since we
    // do not map these flags to cuda_tile.
    return rewriteReshapeLike<triton::ReshapeOp, cuda_tile::ReshapeOp>(
        getTypeConverter(), op, adaptor, rewriter);
  }
};

class ConvertDescriptorLoadOp
    : public OpConversionPattern<triton::DescriptorLoadOp> {
public:
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;
  ConvertDescriptorLoadOp(TypeConverter &typeConverter, MLIRContext *context,
                          DenseMap<Operation *, int> &numStagesMap,
                          int computeCapability, std::optional<int> numStages)
      : OpConversionPattern<triton::DescriptorLoadOp>(typeConverter, context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto ctx = rewriter.getContext();
    auto view = adaptor.getDesc();

    auto optHint = mlir::triton::utils::convertNumStagesToOptHint(
        op, ctx, numStagesMap, computeCapability, numStages);

    auto originalIndices = adaptor.getIndices();

    SmallVector<Value> indices;
    SmallVector<int64_t> viewShapeVec;
    Type viewTy = view.getType();

    // Handle PartitionViewType
    if (auto partTy = dyn_cast<cuda_tile::PartitionViewType>(viewTy)) {
      // For PartitionViewType, we need to divide indices by tile size
      auto tileSizes = partTy.getTileShape();
      for (size_t i = 0; i < originalIndices.size(); i++) {
        Value indicesWithBlockSize = originalIndices[i];
        cuda_tile::TileType constType =
            cuda_tile::TileType::get({}, rewriter.getI32Type());
        auto tileSizeAttr =
            DenseIntElementsAttr::get(constType, {tileSizes[i]});
        Value tileSizeOp = cuda_tile::ConstantOp::create(
            rewriter, op.getLoc(), constType, tileSizeAttr);
        indices.push_back(cuda_tile::DivIOp::create(
            rewriter, op.getLoc(), indicesWithBlockSize, tileSizeOp,
            cuda_tile::Signedness::Signed));
      }
      for (size_t i = 0; i < tileSizes.size(); i++)
        viewShapeVec.push_back(tileSizes[i]);
    } else {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "expect a partition view type");
    }

    auto tileShape = op.getResult().getType().getShape();
    auto elemTy = op.getResult().getType().getElementType();
    auto viewTileTy = cuda_tile::TileType::get(ctx, viewShapeVec, elemTy);

    auto memOrder = cuda_tile::MemoryOrderingSemanticsAttr::get(
        ctx, cuda_tile::MemoryOrderingSemantics::WEAK);
    auto LoadViewOp = cuda_tile::LoadViewTkoOp::create(
        rewriter, op.getLoc(), viewTileTy, cuda_tile::TokenType::get(ctx),
        /*memory_ordering_semantics=*/memOrder,
        /*scope=*/nullptr, view, indices, /*token=*/nullptr,
        optHint.value_or(nullptr));

    if (viewShapeVec.size() != tileShape.size()) {
      auto tileTy = cuda_tile::TileType::get(ctx, tileShape, elemTy);
      auto reshapeOp = cuda_tile::ReshapeOp::create(
          rewriter, op.getLoc(), tileTy, LoadViewOp.getTile());
      rewriter.replaceOp(op, reshapeOp.getResult());
      return success();
    }

    rewriter.replaceOp(op, LoadViewOp.getTile());
    return success();
  }
};

class ConvertDescriptorStoreOp
    : public OpConversionPattern<triton::DescriptorStoreOp> {
public:
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;
  ConvertDescriptorStoreOp(TypeConverter &typeConverter, MLIRContext *context,
                           DenseMap<Operation *, int> &numStagesMap,
                           int computeCapability, std::optional<int> numStages)
      : OpConversionPattern<triton::DescriptorStoreOp>(typeConverter, context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = rewriter.getContext();
    auto view = adaptor.getDesc();

    auto originalIndices = adaptor.getIndices();
    SmallVector<Value> indices;
    Type viewTy = view.getType();

    // Handle PartitionViewType
    if (auto partTy = dyn_cast<cuda_tile::PartitionViewType>(viewTy)) {
      // For PartitionViewType, we need to divide indices by tile size
      auto tileSizes = partTy.getTileShape();
      for (size_t i = 0; i < originalIndices.size(); i++) {
        Value idxWithBlockSize = originalIndices[i];
        auto tileSize = tileSizes[i];
        cuda_tile::TileType constType =
            cuda_tile::TileType::get({}, rewriter.getI32Type());
        auto tileSizeAttr = DenseIntElementsAttr::get(constType, {tileSize});
        Value tileSizeOp = cuda_tile::ConstantOp::create(
            rewriter, op.getLoc(), constType, tileSizeAttr);
        indices.push_back(cuda_tile::DivIOp::create(
            rewriter, op.getLoc(), idxWithBlockSize, tileSizeOp,
            cuda_tile::Signedness::Signed));
      }
    } else {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "expect a partition view type");
    }

    auto optHint = mlir::triton::utils::convertNumStagesToOptHint(
        op, ctx, numStagesMap, computeCapability, numStages);

    auto src = adaptor.getSrc();
    auto StoreViewOp = cuda_tile::StoreViewTkoOp::create(
        rewriter, op.getLoc(), cuda_tile::TokenType::get(ctx),
        cuda_tile::MemoryOrderingSemantics::WEAK, /*scope=*/nullptr, src, view,
        indices, /*token=*/nullptr, optHint.value_or(nullptr));

    rewriter.eraseOp(op);
    return success();
  }
};

/// Convert an expand dims to a reshape by adding a new dimension (1) at a given
/// position.
class ConvertExpandDimsOp : public OpConversionPattern<triton::ExpandDimsOp> {
public:
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteReshapeLike<triton::ExpandDimsOp, cuda_tile::ReshapeOp>(
        getTypeConverter(), op, adaptor, rewriter);
  }
};

class ConvertExternElementwiseOp
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto symbol = op.getSymbol();

    auto getTileType = [](Value v) {
      return cast<cuda_tile::TileType>(v.getType());
    };
    auto rmNearestEven = cuda_tile::RoundingModeAttr::get(
        rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN);

    auto splatFloat = [&](cuda_tile::TileType tileType, double value) -> Value {
      auto elementType = cast<FloatType>(tileType.getElementType());
      auto scalarAttr = rewriter.getFloatAttr(elementType, value);
      auto denseAttr = cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, scalarAttr));
      return cuda_tile::ConstantOp::create(rewriter, loc, tileType, denseAttr);
    };
    auto splatInt = [&](cuda_tile::TileType tileType, int64_t value) -> Value {
      auto elementType = cast<IntegerType>(tileType.getElementType());
      auto scalarAttr = rewriter.getIntegerAttr(elementType, value);
      auto denseAttr = cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(tileType, scalarAttr));
      return cuda_tile::ConstantOp::create(rewriter, loc, tileType, denseAttr);
    };

    auto addf = [&](Value a, Value b) -> Value {
      return cuda_tile::AddFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto subf = [&](Value a, Value b) -> Value {
      return cuda_tile::SubFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto mulf = [&](Value a, Value b) -> Value {
      return cuda_tile::MulFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto divf = [&](Value a, Value b) -> Value {
      return cuda_tile::DivFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto floor = [&](Value a) -> Value {
      return cuda_tile::FloorOp::create(rewriter, loc, a);
    };
    auto sqrt = [&](Value a) -> Value {
      return cuda_tile::SqrtOp::create(rewriter, loc, a, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto exp = [&](Value a) -> Value {
      return cuda_tile::ExpOp::create(rewriter, loc, a);
    };
    auto cmpf = [&](cuda_tile::ComparisonPredicate pred,
                    cuda_tile::ComparisonOrdering ord, Value a,
                    Value b) -> Value {
      return cuda_tile::CmpFOp::create(rewriter, loc, pred, ord, a, b);
    };
    // TODO: other math func support(use extern_eltwise or impl math func)
    if (symbol == "__nv_acosf" || symbol == "__nv_acos") {
      // acos(x) = atan2(sqrt(1 - x^2), x)
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Value one = splatFloat(xType, 1.0);
      Value xx = mulf(x, x);
      Value y = sqrt(subf(one, xx));
      // cuda_tile.atan2 uses the conventional atan2(y, x) argument order.
      rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, y, x);
      return success();
    } else if (symbol == "__nv_atanf" || symbol == "__nv_atan") {
      // atan(x) = atan2(x, 1)
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Value one = splatFloat(xType, 1.0);
      // cuda_tile.atan2 uses the conventional atan2(y, x) argument order.
      rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, x, one);
      return success();
    } else if (symbol == "__nv_asinf" || symbol == "__nv_asin") {
      // asin(x) = atan2(x, sqrt(1 - x^2))
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Value one = splatFloat(xType, 1.0);
      Value xx = mulf(x, x);
      Value denom = sqrt(subf(one, xx));
      // cuda_tile.atan2 uses the conventional atan2(y, x) argument order.
      rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, x, denom);
      return success();
    } else if (symbol == "__nv_rintf" || symbol == "__nv_rint" ||
               symbol == "__nv_nearbyintf" || symbol == "__nv_nearbyint") {
      // Round-to-nearest-even (ties to even).
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Value f = floor(x);
      Value d = subf(x, f); // d in [0, 1) for finite x

      Value half = splatFloat(xType, 0.5);
      Value one = splatFloat(xType, 1.0);
      Value fPlusOne = addf(f, one);

      Value gtHalf = cmpf(cuda_tile::ComparisonPredicate::GREATER_THAN,
                          cuda_tile::ComparisonOrdering::ORDERED, d, half);
      Value ltHalf = cmpf(cuda_tile::ComparisonPredicate::LESS_THAN,
                          cuda_tile::ComparisonOrdering::ORDERED, d, half);

      // Even check without int conversion: f is even iff (f * 0.5) is integer.
      Value halfF = mulf(f, half);
      Value halfFFloor = floor(halfF);
      Value fIsEven =
          cmpf(cuda_tile::ComparisonPredicate::EQUAL,
               cuda_tile::ComparisonOrdering::ORDERED, halfF, halfFFloor);

      Value tie =
          cuda_tile::SelectOp::create(rewriter, loc, fIsEven, f, fPlusOne);
      Value resLt = cuda_tile::SelectOp::create(rewriter, loc, ltHalf, f, tie);
      Value res =
          cuda_tile::SelectOp::create(rewriter, loc, gtHalf, fPlusOne, resLt);
      rewriter.replaceOp(op, res);
      return success();
    } else if (symbol == "__nv_isnanf" || symbol == "__nv_isnand") {
      // isnan(x) = unordered (x != x)
      Value x = adaptor.getSrcs()[0];
      auto resType = cast<cuda_tile::TileType>(
          getTypeConverter()->convertType(op.getResult().getType()));
      Value cond = cmpf(cuda_tile::ComparisonPredicate::NOT_EQUAL,
                        cuda_tile::ComparisonOrdering::UNORDERED, x, x);
      Value one = splatInt(resType, 1);
      Value zero = splatInt(resType, 0);
      Value res =
          cuda_tile::SelectOp::create(rewriter, loc, resType, cond, one, zero);
      rewriter.replaceOp(op, res);
      return success();
    } else if (symbol == "__nv_isinff" || symbol == "__nv_isinfd") {
      // IEEE isinf classification via bitcast.
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Type floatElemType = xType.getElementType();
      auto resType = cast<cuda_tile::TileType>(
          getTypeConverter()->convertType(op.getResult().getType()));

      cuda_tile::TileType bitsType;
      int64_t signMask = 0;
      int64_t infBits = 0;
      if (floatElemType.isF32()) {
        auto shape = llvm::to_vector(xType.getShape());
        bitsType = cuda_tile::TileType::get(shape, rewriter.getI32Type());
        signMask = 0x7fffffffLL;
        infBits = 0x7f800000LL;
      } else if (floatElemType.isF64()) {
        auto shape = llvm::to_vector(xType.getShape());
        bitsType = cuda_tile::TileType::get(shape, rewriter.getI64Type());
        signMask = 0x7fffffffffffffffLL;
        infBits = 0x7ff0000000000000LL;
      } else {
        return rewriter.notifyMatchFailure(
            op, llvm::Twine("unsupported type for ") + symbol +
                    ": expected f32/f64 input");
      }

      Value bits = cuda_tile::BitcastOp::create(rewriter, loc, bitsType, x);
      Value mask = splatInt(bitsType, signMask);
      Value absBits = cuda_tile::AndIOp::create(rewriter, loc, bits, mask);
      Value inf = splatInt(bitsType, infBits);
      Value cond = cuda_tile::CmpIOp::create(
          rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL, absBits, inf,
          cuda_tile::Signedness::Unsigned);

      Value one = splatInt(resType, 1);
      Value zero = splatInt(resType, 0);
      Value res =
          cuda_tile::SelectOp::create(rewriter, loc, resType, cond, one, zero);
      rewriter.replaceOp(op, res);
      return success();
    } else if (symbol == "__nv_finitef" || symbol == "__nv_isfinited") {
      // IEEE isfinite classification via bitcast.
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Type floatElemType = xType.getElementType();
      auto resType = cast<cuda_tile::TileType>(
          getTypeConverter()->convertType(op.getResult().getType()));

      cuda_tile::TileType bitsType;
      int64_t signMask = 0;
      int64_t infBits = 0;
      if (symbol == "__nv_finitef") {
        // finitef is only defined for fp32 in libdevice.
        if (!floatElemType.isF32())
          return rewriter.notifyMatchFailure(op,
                                             "__nv_finitef expects f32 input");
        auto shape = llvm::to_vector(xType.getShape());
        bitsType = cuda_tile::TileType::get(shape, rewriter.getI32Type());
        signMask = 0x7fffffffLL;
        infBits = 0x7f800000LL;
      } else {
        // __nv_isfinited is fp64.
        if (!floatElemType.isF64())
          return rewriter.notifyMatchFailure(
              op, "__nv_isfinited expects f64 input");
        auto shape = llvm::to_vector(xType.getShape());
        bitsType = cuda_tile::TileType::get(shape, rewriter.getI64Type());
        signMask = 0x7fffffffffffffffLL;
        infBits = 0x7ff0000000000000LL;
      }

      Value bits = cuda_tile::BitcastOp::create(rewriter, loc, bitsType, x);
      Value mask = splatInt(bitsType, signMask);
      Value absBits = cuda_tile::AndIOp::create(rewriter, loc, bits, mask);
      Value inf = splatInt(bitsType, infBits);
      Value cond = cuda_tile::CmpIOp::create(
          rewriter, loc, cuda_tile::ComparisonPredicate::LESS_THAN, absBits,
          inf, cuda_tile::Signedness::Unsigned);

      Value one = splatInt(resType, 1);
      Value zero = splatInt(resType, 0);
      Value res =
          cuda_tile::SelectOp::create(rewriter, loc, resType, cond, one, zero);
      rewriter.replaceOp(op, res);
      return success();
    } else if (symbol == "__nv_erff" || symbol == "__nv_erf") {
      // High-accuracy approximation (max error ~1.5e-7):
      // https://stackoverflow.com/a/4578056
      //
      // t = 1 / (1 + 0.5 * |x|)
      // tau = t * exp(-x^2 - 1.26551223 + t*(1.00002368 + t*(0.37409196 + ...
      // ))) erf(x) = sign(x) * (1 - tau)
      Value x = adaptor.getSrcs()[0];
      auto xType = getTileType(x);
      Type floatElemType = xType.getElementType();
      if (!floatElemType.isF32() && !floatElemType.isF64()) {
        return rewriter.notifyMatchFailure(
            op, llvm::Twine("unsupported type for ") + symbol +
                    ": expected f32/f64 input");
      }

      Value zero = splatFloat(xType, 0.0);
      Value one = splatFloat(xType, 1.0);
      Value half = splatFloat(xType, 0.5);

      Value ax = cuda_tile::AbsFOp::create(rewriter, loc, x);
      Value t = divf(one, addf(one, mulf(half, ax)));

      // Horner form for the nested polynomial.
      Value p = splatFloat(xType, 0.17087277);
      p = addf(splatFloat(xType, -0.82215223), mulf(t, p));
      p = addf(splatFloat(xType, 1.48851587), mulf(t, p));
      p = addf(splatFloat(xType, -1.13520398), mulf(t, p));
      p = addf(splatFloat(xType, 0.27886807), mulf(t, p));
      p = addf(splatFloat(xType, -0.18628806), mulf(t, p));
      p = addf(splatFloat(xType, 0.09678418), mulf(t, p));
      p = addf(splatFloat(xType, 0.37409196), mulf(t, p));
      p = addf(splatFloat(xType, 1.00002368), mulf(t, p));

      Value negAx2 = subf(zero, mulf(ax, ax));
      Value expArg =
          addf(addf(negAx2, splatFloat(xType, -1.26551223)), mulf(t, p));
      Value tau = mulf(t, exp(expArg));
      Value erfAbs = subf(one, tau);

      Value isNeg = cmpf(cuda_tile::ComparisonPredicate::LESS_THAN,
                         cuda_tile::ComparisonOrdering::ORDERED, x, zero);
      Value negErfAbs = subf(zero, erfAbs);
      Value res =
          cuda_tile::SelectOp::create(rewriter, loc, isNeg, negErfAbs, erfAbs);
      rewriter.replaceOp(op, res);
      return success();
    } else if (symbol == "__nv_ceil" || symbol == "__nv_ceilf") {
      rewriter.replaceOpWithNewOp<cuda_tile::CeilOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_pow" || symbol == "__nv_powf") {
      rewriter.replaceOpWithNewOp<cuda_tile::PowOp>(op, adaptor.getSrcs()[0],
                                                    adaptor.getSrcs()[1]);
      return success();
    } else if (symbol == "__nv_cos" || symbol == "__nv_cosf") {
      rewriter.replaceOpWithNewOp<cuda_tile::CosOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_sin" || symbol == "__nv_sinf") {
      rewriter.replaceOpWithNewOp<cuda_tile::SinOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_tan" || symbol == "__nv_tanf") {
      rewriter.replaceOpWithNewOp<cuda_tile::TanOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_exp" || symbol == "__nv_expf") {
      rewriter.replaceOpWithNewOp<cuda_tile::ExpOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_fast_expf") {
      rewriter.replaceOpWithNewOp<cuda_tile::ExpOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_exp2" || symbol == "__nv_exp2f") {
      rewriter.replaceOpWithNewOp<cuda_tile::Exp2Op>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_log2f" || symbol == "__nv_log2") {
      rewriter.replaceOpWithNewOp<cuda_tile::Log2Op>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_rsqrtf" || symbol == "__nv_rsqrt") {
      rewriter.replaceOpWithNewOp<cuda_tile::RsqrtOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_sqrtf") {
      rewriter.replaceOpWithNewOp<cuda_tile::SqrtOp>(
          op, adaptor.getSrcs()[0], cuda_tile::RoundingMode::APPROX);
      return success();
    } else if (symbol == "__nv_sqrt") {
      // Prefer IEEE-style rounding for fp64 sqrt.
      Value x = adaptor.getSrcs()[0];
      rewriter.replaceOp(op, sqrt(x));
      return success();
    } else if (symbol == "__nv_floorf" || symbol == "__nv_floor") {
      rewriter.replaceOpWithNewOp<cuda_tile::FloorOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_tanhf" || symbol == "__nv_tanh") {
      rewriter.replaceOpWithNewOp<cuda_tile::TanHOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_fast_tanhf") {
      rewriter.replaceOpWithNewOp<cuda_tile::TanHOp>(op, adaptor.getSrcs()[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, llvm::Twine("unsupported extern_elementwise symbol for stable "
                        "cuda_tile lowering: ") +
                symbol +
                ". Please add a rewrite in ConvertExternElementwiseOp.");
  }
};

class ConvertErfOp : public OpConversionPattern<math::ErfOp> {
public:
  using OpConversionPattern<math::ErfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::ErfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = adaptor.getOperand();
    auto xType = cast<cuda_tile::TileType>(x.getType());
    Type floatElemType = xType.getElementType();
    if (!floatElemType.isF32() && !floatElemType.isF64()) {
      return rewriter.notifyMatchFailure(
          op, "math.erf expects f32/f64 input for TileIR lowering");
    }

    auto rmNearestEven = cuda_tile::RoundingModeAttr::get(
        rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN);
    auto splatFloat = [&](double value) -> Value {
      auto elementType = cast<FloatType>(xType.getElementType());
      auto scalarAttr = rewriter.getFloatAttr(elementType, value);
      auto denseAttr = cast<DenseIntOrFPElementsAttr>(
          DenseElementsAttr::get(xType, scalarAttr));
      return cuda_tile::ConstantOp::create(rewriter, loc, xType, denseAttr);
    };
    auto addf = [&](Value a, Value b) -> Value {
      return cuda_tile::AddFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto subf = [&](Value a, Value b) -> Value {
      return cuda_tile::SubFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto mulf = [&](Value a, Value b) -> Value {
      return cuda_tile::MulFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };
    auto divf = [&](Value a, Value b) -> Value {
      return cuda_tile::DivFOp::create(rewriter, loc, a, b, rmNearestEven,
                                       /*flush_to_zero=*/nullptr);
    };

    Value zero = splatFloat(0.0);
    Value one = splatFloat(1.0);
    Value half = splatFloat(0.5);

    Value ax = cuda_tile::AbsFOp::create(rewriter, loc, x);
    Value t = divf(one, addf(one, mulf(half, ax)));

    Value p = splatFloat(0.17087277);
    p = addf(splatFloat(-0.82215223), mulf(t, p));
    p = addf(splatFloat(1.48851587), mulf(t, p));
    p = addf(splatFloat(-1.13520398), mulf(t, p));
    p = addf(splatFloat(0.27886807), mulf(t, p));
    p = addf(splatFloat(-0.18628806), mulf(t, p));
    p = addf(splatFloat(0.09678418), mulf(t, p));
    p = addf(splatFloat(0.37409196), mulf(t, p));
    p = addf(splatFloat(1.00002368), mulf(t, p));

    Value negAx2 = subf(zero, mulf(ax, ax));
    Value expArg = addf(addf(negAx2, splatFloat(-1.26551223)), mulf(t, p));
    Value tau = mulf(t, cuda_tile::ExpOp::create(rewriter, loc, expArg));
    Value erfAbs = subf(one, tau);

    Value isNeg = cuda_tile::CmpFOp::create(
        rewriter, loc, cuda_tile::ComparisonPredicate::LESS_THAN,
        cuda_tile::ComparisonOrdering::ORDERED, x, zero);
    Value negErfAbs = subf(zero, erfAbs);
    Value res =
        cuda_tile::SelectOp::create(rewriter, loc, isNeg, negErfAbs, erfAbs);
    rewriter.replaceOp(op, res);
    return success();
  }
};

class ConvertCatOp : public OpConversionPattern<triton::CatOp> {
public:
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = cast<ShapedType>(
        getTypeConverter()->convertType(op.getResult().getType()));

    // This should always be true since SameTypeOperands trait is enforced for
    // triton::CatOp
    auto lhsTy = adaptor.getLhs().getType();
    auto rhsTy = adaptor.getRhs().getType();
    assert(lhsTy == rhsTy && "Operands must have identical types");

    // Add singleton dimension to operand type to match result rank
    auto reshapeToMatchResultRank = [&](Value operand) -> Value {
      auto operandTy = cast<ShapedType>(operand.getType());
      if (operandTy.getRank() == resTy.getRank())
        return operand;
      auto operandShape = llvm::to_vector(operandTy.getShape());
      operandShape.resize(resTy.getRank(), 1);
      auto newTy =
          operandTy.cloneWith(operandShape, operandTy.getElementType());
      return cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), newTy,
                                          operand);
    };

    Value lhs = reshapeToMatchResultRank(adaptor.getLhs());
    Value rhs = reshapeToMatchResultRank(adaptor.getRhs());

    // Determine concatenation axis (last dimension by default)
    int64_t concatDim = resTy.getRank() - 1;
    auto lhsShape = cast<ShapedType>(lhs.getType()).getShape();
    for (int64_t i = 0; i < resTy.getRank(); i++)
      if (lhsShape[i] != resTy.getShape()[i]) {
        concatDim = i;
        break;
      }

    rewriter.replaceOpWithNewOp<cuda_tile::CatOp>(
        op, resTy, lhs, rhs, rewriter.getI64IntegerAttr(concatDim));

    return success();
  }
};

// Join the tensors in a new minor dimension.
class ConvertJoinOp : public OpConversionPattern<triton::JoinOp> {
public:
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Step1. Create a new minor dimenion using reshape.
    cuda_tile::TileType lhsType =
        cast<cuda_tile::TileType>(adaptor.getLhs().getType());
    cuda_tile::TileType rhsType =
        cast<cuda_tile::TileType>(adaptor.getRhs().getType());
    assert(lhsType == rhsType &&
           "expect triton cat to have same operand type for lhs and rhs");

    SmallVector<int64_t> reshapedView = llvm::to_vector(lhsType.getShape());
    int64_t concatDim = reshapedView.size();
    reshapedView.push_back(1);
    cuda_tile::TileType newLhsType =
        cuda_tile::TileType::get(reshapedView, lhsType.getElementType());
    auto viewOnLhs = cuda_tile::ReshapeOp::create(rewriter, loc, newLhsType,
                                                  adaptor.getLhs());
    auto viewOnRhs = cuda_tile::ReshapeOp::create(rewriter, loc, newLhsType,
                                                  adaptor.getRhs());
    reshapedView[concatDim] += reshapedView[concatDim];
    cuda_tile::TileType newResType =
        cuda_tile::TileType::get(reshapedView, lhsType.getElementType());

    // Step2. Concat along the new minor dimension.
    rewriter.replaceOpWithNewOp<cuda_tile::CatOp>(
        op, newResType, viewOnLhs, viewOnRhs,
        rewriter.getI64IntegerAttr(concatDim));
    return success();
  }
};

class ConvertGetProgramIdOp
    : public OpConversionPattern<triton::GetProgramIdOp> {
public:
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t axis = op.getAxisAsInt();
    Operation *newOp =
        cuda_tile::GetTileBlockIdOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, newOp->getResult(axis));
    return success();
  }
};

// Helper function for common functionality between ReduceOp and ScanOp
LogicalResult convertAggregationOp(Operation *op,
                                   ConversionPatternRewriter &rewriter,
                                   const TypeConverter *typeConverter,
                                   Operation *newOp) {
  SmallVector<Type> newResultTypes;
  auto resConversion =
      typeConverter->convertTypes(op->getResults().getType(), newResultTypes);
  assert(succeeded(resConversion) && "failed to convert tensor type");

  Block *oldBody = &op->getRegion(0).front();

  Block *body = new Block();
  SmallVector<Type> currentOperandsTy;
  SmallVector<Location> locs;
  // We use pair for better readability:
  // [current_operand[i], prev_operand[i], current_operand[i + 1],
  // prev_operand[i + 1]] while triton is: current_operand[i],
  // current_operand[i + 1], prev_operand[i], prev_operand[i + 1]]
  size_t numberOfArgs = oldBody->getNumArguments();
  size_t half = numberOfArgs / 2;

  for (int i = 0, e = numberOfArgs / 2; i < e; i++) {
    BlockArgument currentBlkOperand = oldBody->getArgument(i);
    locs.push_back(currentBlkOperand.getLoc());
    BlockArgument preBlkOperand = oldBody->getArgument(i + half);
    locs.push_back(preBlkOperand.getLoc());
    currentOperandsTy.push_back(
        cuda_tile::TileType::get({}, currentBlkOperand.getType()));
    currentOperandsTy.push_back(
        cuda_tile::TileType::get({}, preBlkOperand.getType()));
  }
  body->addArguments(currentOperandsTy, locs);

  SmallVector<Value> blockOperands;
  for (int i = 0; i < numberOfArgs / 2; i++)
    blockOperands.push_back(body->getArgument(i * 2));

  for (int i = 0; i < numberOfArgs / 2; i++)
    blockOperands.push_back(body->getArgument(i * 2 + 1));

  rewriter.inlineBlockBefore(&op->getRegion(0).front(), body, body->end(),
                             blockOperands);
  newOp->getRegion(0).push_back(body);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

class ConvertReduceOp : public OpConversionPattern<triton::ReduceOp> {
public:
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    auto resConversion = getTypeConverter()->convertTypes(
        op.getResults().getType(), newResultTypes);
    assert(succeeded(resConversion) && "failed to convert tensor type");
    Block *oldBody = &op.getRegion().front();
    if (llvm::hasSingleElement(*oldBody)) {
      // Fast path: This reduction is a no-op. It contains just the terminator.
      auto terminator = cast<triton::ReduceReturnOp>(oldBody->getTerminator());
      SmallVector<Value> repls;
      for (auto it : llvm::enumerate(terminator.getResult())) {
        // The returned value must be one of the bbargs. Find out which one,
        // then replace the reduction op result with the respective operand.
        Value v = it.value();
        auto bbArg = cast<BlockArgument>(v);
        unsigned operandIdx = bbArg.getArgNumber() / 2;
        repls.push_back(cuda_tile::ReshapeOp::create(
            rewriter, op.getLoc(), newResultTypes[it.index()],
            adaptor.getSrcs()[operandIdx]));
      }
      rewriter.replaceOp(op, repls);
      return success();
    }

    auto identitiesOrFailure =
        getIdentitiesFromCombineOp(op.getCombineOp(), newResultTypes, rewriter);
    if (failed(identitiesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to setup valid identities for combineOp");
    ArrayAttr identities = *identitiesOrFailure;

    auto newReduceOp = cuda_tile::ReduceOp::create(
        rewriter, op.getLoc(), newResultTypes, adaptor.getOperands(),
        adaptor.getAxis(), identities);

    return convertAggregationOp(op, rewriter, getTypeConverter(), newReduceOp);
  }
};

class ConvertScanOp : public OpConversionPattern<triton::ScanOp> {
public:
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    auto resConversion = getTypeConverter()->convertTypes(
        op.getResults().getType(), newResultTypes);
    assert(succeeded(resConversion) && "failed to convert tensor type");

    auto identitiesOrFailure =
        getIdentitiesFromCombineOp(op.getCombineOp(), newResultTypes, rewriter);
    if (failed(identitiesOrFailure))
      return rewriter.notifyMatchFailure(
          op, "failed to setup valid identities for combineOp");
    ArrayAttr identities = *identitiesOrFailure;

    auto newScanOp = cuda_tile::ScanOp::create(
        rewriter, op.getLoc(), newResultTypes, adaptor.getOperands(),
        adaptor.getAxis(), adaptor.getReverse(), identities);

    return convertAggregationOp(op, rewriter, getTypeConverter(), newScanOp);
  }
};

class ConvertScanReturnOp : public OpConversionPattern<triton::ScanReturnOp> {
public:
  using OpConversionPattern<triton::ScanReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentOp = op->getParentOp();
    if (parentOp && isa<cuda_tile::ScanOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(op,
                                                      adaptor.getOperands());
      return success();
    }
    return failure();
  }
};

class ConvertGetNumProgramsOp
    : public OpConversionPattern<triton::GetNumProgramsOp> {
public:
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t axis = op.getAxisAsInt();
    Operation *newOp =
        cuda_tile::GetNumTileBlocksOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, newOp->getResult(axis));
    return success();
  }
};

class ConvertReduceReturnOp
    : public OpConversionPattern<triton::ReduceReturnOp> {
public:
  using OpConversionPattern<triton::ReduceReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentOp = op->getParentOp();
    if (parentOp && isa<cuda_tile::ReduceOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(op,
                                                      adaptor.getOperands());
      return success();
    }
    return failure();
  }
};

class ConvertIfOp : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();
    auto newIfOp = cuda_tile::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                           adaptor.getCondition());
    rewriter.inlineRegionBefore(op.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());
    rewriter.replaceOp(op, newIfOp.getResults());
    return success();
  }
};

// clang-format off
// We will rewrite scf.while op into cuda_tile.loop op

// for example:
// ---------------------------------------------------------
// scf.while
// %results = scf.while (<while_args>) : type(<while_args>) -> type(results) { // type(<while_args>) != type(results)
//     ... // `before` region code
//     %cond = ...
//     <condition_args> = ...
//     scf.condition (%cond) <condition_args> : type(condition_args)  // type(condition_args) == type(results)
// } do {
//     ^bb0(after_args):  // `after_args` come from `condition_args` and type(condition_args) == type(after_args)
//     ... // `after` region code
//     scf.yield <yield_vals> : type(yield_vals)  // type(yield_vals) == type(while_args)
// }
// ---------------------------------------------------------

// will be rewritten into:

// ---------------------------------------------------------
// %results = cuda_tile.loop iter_values(<while_args>) -> type(results) { // type(<while_args>) != type(results)
//     ... // `before` region code
//     %cond = ...
//     <condition_args> = ...
//     cuda_tile.if %cond {
//         ... // `after` region code
//         cuda_tile.continue <yield_vals>
//     }
//     cuda_tile.break <condition_args>
// }
// ---------------------------------------------------------

// clang-format on
class ConvertWhileOp : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp whileOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *beforeBlock = &adaptor.getBefore().front();
    Block *afterBlock = &adaptor.getAfter().front();

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(whileOp->getResultTypes(),
                                           resultTypes)))
      return failure();

    SmallVector<Type> inputTypes;
    for (auto type : whileOp.getOperandTypes()) {
      Type convertedType = typeConverter->convertType(type);
      inputTypes.push_back(convertedType);
    }

    rewriter.setInsertionPoint(whileOp);

    SmallVector<Location> locs(inputTypes.size(), whileOp.getLoc());
    auto newLoopOp = cuda_tile::LoopOp::create(
        rewriter, whileOp.getLoc(), resultTypes, adaptor.getOperands());

    Block *newLoopBlock = rewriter.createBlock(
        &newLoopOp.getRegion(), /*insertPt=*/{}, inputTypes, locs);

    rewriter.inlineBlockBefore(beforeBlock, newLoopBlock, newLoopBlock->end(),
                               newLoopBlock->getArguments());
    auto conditionOp = &newLoopBlock->back();
    auto castI1Op = rewriter.getRemappedValue(conditionOp->getOperand(0));

    auto ifOp = cuda_tile::IfOp::create(rewriter, whileOp.getLoc(), TypeRange(),
                                        castI1Op);
    rewriter.createBlock(&ifOp.getThenRegion());
    rewriter.createBlock(&ifOp.getElseRegion());

    rewriter.setInsertionPointToStart(ifOp.getThenBlock());

    SmallVector<Value> afterBlockArgs;
    for (size_t i = 1; i < conditionOp->getOperands().size(); i++) {
      afterBlockArgs.push_back(conditionOp->getOperands()[i]);
    }

    rewriter.inlineBlockBefore(afterBlock, ifOp.getThenBlock(),
                               ifOp.getThenBlock()->end(), afterBlockArgs);
    auto yieldOp = &ifOp.getThenBlock()->back();
    SmallVector<Value> yieldOpCastArgs;
    if (failed(rewriter.getRemappedValues(yieldOp->getOperands(),
                                          yieldOpCastArgs)))
      return failure();

    cuda_tile::ContinueOp::create(rewriter, whileOp.getLoc(), yieldOpCastArgs);

    SmallVector<Value> afterBlockCastArgs;
    if (failed(rewriter.getRemappedValues(afterBlockArgs, afterBlockCastArgs)))
      return failure();

    rewriter.setInsertionPointToEnd(ifOp.getElseBlock());
    cuda_tile::BreakOp::create(rewriter, whileOp.getLoc(), afterBlockCastArgs);

    rewriter.setInsertionPointToEnd(newLoopBlock);
    cuda_tile::BreakOp::create(rewriter, whileOp.getLoc(), afterBlockCastArgs);

    rewriter.eraseOp(conditionOp);
    rewriter.eraseOp(yieldOp);
    rewriter.replaceOp(whileOp, newLoopOp.getResults());
    return success();
  }
};

class ConvertForOp : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    BlockArgument indVar = op.getBody()->getArgument(0);
    if (isa<IndexType>(indVar.getType())) {
      return rewriter.notifyMatchFailure(
          loc, "index type is not supported in cuda tile");
    }
    Block *origBody = op.getBody();

    auto newForOp = cuda_tile::ForOp::create(
        rewriter, loc, adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs(),
        [&](OpBuilder builder, Location loc, Value indVar,
            ValueRange initArgs) {
          // Don't build the body here, we'll inline it right after.
        });

    // Apply a signature conversion on the for loop body.
    cuda_tile::TileType indVarTy =
        cuda_tile::TileType::get({}, indVar.getType());
    TypeConverter::SignatureConversion sigConversion(
        origBody->getNumArguments());
    sigConversion.addInputs(0, indVarTy);
    if (failed(typeConverter->convertSignatureArgs(
            llvm::drop_begin(TypeRange(origBody->getArgumentTypes())),
            sigConversion, /*origInputOffset=*/1)))
      return failure();
    Block *body = rewriter.applySignatureConversion(origBody, sigConversion);

    rewriter.inlineRegionBefore(op.getRegion(), newForOp.getRegion(),
                                newForOp.getRegion().begin());
    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

struct ConvertCmpIOp : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the arith comparison predicate
    auto arithPredicate = op.getPredicate();

    // Infer signedness and comparison predicate from arith predicate
    cuda_tile::Signedness signedness = cuda_tile::Signedness::Signed;
    cuda_tile::ComparisonPredicate comparisonPredicate;

    switch (arithPredicate) {
    case arith::CmpIPredicate::eq:
      comparisonPredicate = cuda_tile::ComparisonPredicate::EQUAL;
      break;
    case arith::CmpIPredicate::ne:
      comparisonPredicate = cuda_tile::ComparisonPredicate::NOT_EQUAL;
      break;
    case arith::CmpIPredicate::slt:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN;
      break;
    case arith::CmpIPredicate::sle:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
      break;
    case arith::CmpIPredicate::sgt:
      comparisonPredicate = cuda_tile::ComparisonPredicate::GREATER_THAN;
      break;
    case arith::CmpIPredicate::sge:
      comparisonPredicate =
          cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      break;
    case arith::CmpIPredicate::ult:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN;
      signedness = cuda_tile::Signedness::Unsigned;
      break;
    case arith::CmpIPredicate::ule:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
      signedness = cuda_tile::Signedness::Unsigned;
      break;
    case arith::CmpIPredicate::ugt:
      comparisonPredicate = cuda_tile::ComparisonPredicate::GREATER_THAN;
      signedness = cuda_tile::Signedness::Unsigned;
      break;
    case arith::CmpIPredicate::uge:
      comparisonPredicate =
          cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      signedness = cuda_tile::Signedness::Unsigned;
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported arith::CmpIOp predicate" +
                  stringifyCmpIPredicate(arithPredicate));
    }

    // Upcast to i16 if necessary.
    Location loc = op.getLoc();
    Value lhs = upCastOrSelf(rewriter, loc, adaptor.getLhs(),
                             signedness == cuda_tile::Signedness::Unsigned
                                 ? Signedness::Unsigned
                                 : Signedness::Signed,
                             IntegerUpCast::To_I16);
    Value rhs = upCastOrSelf(rewriter, loc, adaptor.getRhs(),
                             signedness == cuda_tile::Signedness::Unsigned
                                 ? Signedness::Unsigned
                                 : Signedness::Signed,
                             IntegerUpCast::To_I16);

    // Replace the op with cuda_tile.cmpi.
    rewriter.replaceOpWithNewOp<cuda_tile::CmpIOp>(op, comparisonPredicate, lhs,
                                                   rhs, signedness);
    return success();
  }
};

struct ConvertCmpFOp : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern<arith::CmpFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the arith comparison predicate
    auto arithPredicate = op.getPredicate();

    // Infer comparison predicate and ordering from arith predicate
    cuda_tile::ComparisonPredicate comparisonPredicate;
    cuda_tile::ComparisonOrdering comparisonOrdering =
        cuda_tile::ComparisonOrdering::ORDERED;

    switch (arithPredicate) {
    case arith::CmpFPredicate::OEQ:
      comparisonPredicate = cuda_tile::ComparisonPredicate::EQUAL;
      break;
    case arith::CmpFPredicate::ONE:
      comparisonPredicate = cuda_tile::ComparisonPredicate::NOT_EQUAL;
      break;
    case arith::CmpFPredicate::OLT:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN;
      break;
    case arith::CmpFPredicate::OLE:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
      break;
    case arith::CmpFPredicate::OGT:
      comparisonPredicate = cuda_tile::ComparisonPredicate::GREATER_THAN;
      break;
    case arith::CmpFPredicate::OGE:
      comparisonPredicate =
          cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      break;
    case arith::CmpFPredicate::UEQ:
      comparisonPredicate = cuda_tile::ComparisonPredicate::EQUAL;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    case arith::CmpFPredicate::UNE:
      comparisonPredicate = cuda_tile::ComparisonPredicate::NOT_EQUAL;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    case arith::CmpFPredicate::ULT:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    case arith::CmpFPredicate::ULE:
      comparisonPredicate = cuda_tile::ComparisonPredicate::LESS_THAN_OR_EQUAL;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    case arith::CmpFPredicate::UGT:
      comparisonPredicate = cuda_tile::ComparisonPredicate::GREATER_THAN;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    case arith::CmpFPredicate::UGE:
      comparisonPredicate =
          cuda_tile::ComparisonPredicate::GREATER_THAN_OR_EQUAL;
      comparisonOrdering = cuda_tile::ComparisonOrdering::UNORDERED;
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported arith::CmpFOp predicate" +
                  stringifyCmpFPredicate(arithPredicate));
    }

    // Replace the op with cuda_tile.cmpf.
    rewriter.replaceOpWithNewOp<cuda_tile::CmpFOp>(
        op, comparisonPredicate, comparisonOrdering, adaptor.getLhs(),
        adaptor.getRhs());
    return success();
  }
};

class ConvertYieldOp : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentOp = op->getParentOp();
    // Only ForOp is currently supported as parent operation.
    if (parentOp && isa<cuda_tile::ForOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<cuda_tile::ContinueOp>(op,
                                                         adaptor.getOperands());
      return success();
    }
    if (parentOp && isa<cuda_tile::LoopOp>(parentOp)) {
      rewriter.replaceOpWithNewOp<cuda_tile::ContinueOp>(op,
                                                         adaptor.getOperands());
      return success();
    }
    rewriter.replaceOpWithNewOp<cuda_tile::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Simple pattern to convert a tt.splat ty -> tensor<XxYxZxTy> by first
/// reshaping and then broadcasting.
class ConvertSplatOp : public OpConversionPattern<triton::SplatOp> {
public:
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    assert(newResultType && "unable to convert type");
    cuda_tile::TileType newResultTensorType =
        cast<cuda_tile::TileType>(newResultType);

    SmallVector<int64_t> rank1Shape(newResultTensorType.getRank(), 1);
    cuda_tile::TileType rank1Ty = cuda_tile::TileType::get(
        rank1Shape, newResultTensorType.getElementType());
    auto replaceOp = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(),
                                                  rank1Ty, adaptor.getSrc());
    rewriter.replaceOpWithNewOp<cuda_tile::BroadcastOp>(op, newResultTensorType,
                                                        replaceOp);
    return success();
  }
};

class ConvertUnsplatOp : public OpConversionPattern<triton::UnsplatOp> {
public:
  using OpConversionPattern<triton::UnsplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type reTileTy = getTypeConverter()->convertType(op.getResult().getType());
    auto reshapeOp = cuda_tile::ReshapeOp::create(rewriter, op.getLoc(),
                                                  reTileTy, adaptor.getSrc());
    rewriter.replaceOp(op, reshapeOp.getResult());
    return success();
  }
};

class ConvertMaximumFOp : public OpConversionPattern<arith::MaximumFOp> {
public:
  using OpConversionPattern<arith::MaximumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MaximumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    assert(resTy && "expect valid type");
    rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(
        op, resTy, adaptor.getLhs(), adaptor.getRhs(),
        /*nan=*/rewriter.getUnitAttr(),
        /*flush_to_zero=*/nullptr);
    return success();
  }
};

class ConvertMinimumFOp : public OpConversionPattern<arith::MinimumFOp> {
public:
  using OpConversionPattern<arith::MinimumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MinimumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    assert(resTy && "expect valid type");
    rewriter.replaceOpWithNewOp<cuda_tile::MinFOp>(
        op, resTy, adaptor.getLhs(), adaptor.getRhs(),
        /*nan_modifier=*/rewriter.getUnitAttr(),
        /*flush_to_zero_modifier=*/nullptr);
    return success();
  }
};

class ConvertMakeRangeOp : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    int64_t start = op.getStart();
    auto converter = this->getTypeConverter();
    Type retTy = converter->convertType(op.getResult().getType());
    if (!retTy)
      return rewriter.notifyMatchFailure(
          op.getLoc(), "typeConversion of make_range op failed");

    Location loc = op.getLoc();
    auto iotaOp = cuda_tile::IotaOp::create(rewriter, loc, retTy);
    if (start == 0) {
      rewriter.replaceOp(op, iotaOp);
      return success();
    }
    ShapedType retTyAsShape = cast<ShapedType>(retTy);
    IntegerAttr attr =
        rewriter.getIntegerAttr(retTyAsShape.getElementType(), start);
    DenseIntOrFPElementsAttr denseAttr = cast<DenseIntOrFPElementsAttr>(
        DenseElementsAttr::get(retTyAsShape, attr));
    cuda_tile::ConstantOp cstOp =
        cuda_tile::ConstantOp::create(rewriter, loc, retTy, denseAttr);
    cuda_tile::AddIOp offsetOp =
        cuda_tile::AddIOp::create(rewriter, loc, iotaOp, cstOp);
    rewriter.replaceOp(op, offsetOp);
    return success();
  }
};

Value wrapIntoScalarTile(OpBuilder &rewriter, Value v, unsigned attachAlignment,
                         int64_t attachRange) {
  auto ctx = v.getType().getContext();
  auto loc = v.getLoc();
  if (v.getType().isInteger(32))
    v = arith::ExtSIOp::create(rewriter, loc, rewriter.getI64Type(), v)
            .getResult();
  auto elemType = v.getType();
  auto scalarTileTy = cuda_tile::TileType::get(ctx, /*shape=*/{}, elemType);
  auto scalarTile =
      UnrealizedConversionCastOp::create(rewriter, loc, scalarTileTy, v)
          .getResult(0);
  scalarTile = cuda_tile::AssumeOp::create(
                   rewriter, loc, scalarTile,
                   cuda_tile::BoundedAttr::get(ctx, 0, attachRange))
                   .getResult();

  // we can always assume the stride are divisible by 16
  // because openai has already make it into host tma api.
  if (!attachAlignment)
    return scalarTile;
  return cuda_tile::AssumeOp::create(
             rewriter, loc, scalarTile,
             cuda_tile::DivByAttr::get(ctx, attachAlignment, std::nullopt,
                                       std::nullopt))
      .getResult();
};

/// Lowering of tt.make_tensor_desc to cuda_tile.make_tensor_view.
///
/// Triton currently assumes that the pointer, sizes and strides are
/// compatible with the TMA requirements of the target architecture.
/// See commit message: https://github.com/triton-lang/triton/pull/6753
/// "This does not implement: Interop for unsupported tensor descriptors on
/// devices which support tensor descriptors."
///
/// This means that we can safely assume that the pointer and strides are
/// divisible by 16. (Sizes do not have this divisibility requirement.)
/// Using a pointer or strides that are not divisible by 16 will result in
/// undefined behavior.
/// This lowering attaches the divisibility hints to the pointer and strides.
///
/// It is safe to assume that shapes are in the range [0, 2^32 - 1] and
/// strides are in the range [0, 2^40 - 1]
/// This lowering attaches the range hints to the shapes and strides.
class ConvertMakeTensorDescOp
    : public OpConversionPattern<triton::MakeTensorDescOp> {
public:
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  ConvertMakeTensorDescOp(MLIRContext *context)
      : OpConversionPattern<triton::MakeTensorDescOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op->getContext();
    auto ptrTy = op.getBase().getType();

    size_t rank = op.getShape().size();
    if (rank == 0) {
      return rewriter.notifyMatchFailure(
          loc, "Device TMA descriptor with rank 0 is not supported.");
    }

    SmallVector<int64_t> globalShape(rank, cuda_tile::TensorViewType::kDynamic);
    SmallVector<int64_t> globalStride(rank,
                                      cuda_tile::TensorViewType::kDynamic);
    // we can always assume the stride is 1
    // because openai has assume this.
    globalStride[rank - 1] = 1;

    auto elemTy = ptrTy.getPointeeType();
    unsigned elem_bytes = elemTy.getIntOrFloatBitWidth() / 8;
    unsigned align_byte = kTMAAlignment / elem_bytes;

    auto getConstInt = [&](Value stride) -> std::optional<int> {
      if (auto constOp =
              dyn_cast_or_null<arith::ConstantOp>(stride.getDefiningOp())) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue()))
          return intAttr.getInt();
      }
      return std::nullopt;
    };

    SmallVector<Value> wrappedDynShapes;
    // It is safe to assume that shapes are in the range [0, 2^32 - 1] which is
    // the TMA hardware limit for shape. This ensures that if users explicitly
    // want to use TMA for this operation, the shape parameters will satisfy TMA
    // descriptor encoding requirements.
    for (auto v : op.getShape())
      wrappedDynShapes.push_back(
          wrapIntoScalarTile(rewriter, v, /*attachAlignment=*/0, kMaxShape));
    // Strides are required to be divisible by 16.
    SmallVector<Value> wrappedDynStrides;
    auto strides = op.getStrides();
    for (size_t i = 0; i < rank; ++i) {
      auto stride = strides[i];
      auto constValOpt = getConstInt(stride);

      if (i == rank - 1) {
        // Last stride must be 1
        if (!constValOpt.has_value() || constValOpt.value() != 1)
          return rewriter.notifyMatchFailure(
              loc, "the last stride is expected to be constexpr 1");
      } else {
        // Other strides should be divisible by 16-bytes
        if (constValOpt.has_value() &&
            (constValOpt.value() % align_byte != 0)) {
          op.emitWarning("the stride is expected to be divisible by 16-bytes, "
                         "may result in error");
          // It is safe to assume that shapes are in the range [0, 2^40 - 1]
          // which is the TMA hardware limit for stride. This ensures that if
          // users explicitly want to use TMA for this operation, the stride
          // parameters will satisfy TMA descriptor encoding requirements.
          wrappedDynStrides.push_back(wrapIntoScalarTile(
              rewriter, stride, /*attachAlignment=*/0, kMaxStride));
        } else
          wrappedDynStrides.push_back(wrapIntoScalarTile(
              rewriter, stride, /*attachAlignment=*/align_byte, kMaxStride));
      }
    }

    auto tensorViewTy =
        cuda_tile::TensorViewType::get(ctx, elemTy, globalShape, globalStride);

    auto tileIRPtrType = cuda_tile::PointerType::get(elemTy);
    auto ptrTypeWrapper = cuda_tile::TileType::get(ctx, {}, tileIRPtrType);
    auto ptrOp = UnrealizedConversionCastOp::create(
                     rewriter, loc, ptrTypeWrapper, op.getBase())
                     .getResult(0);
    // Pointer is required to be divisible by 16.
    auto ptrWithDivBy = cuda_tile::AssumeOp::create(
                            rewriter, loc, ptrOp,
                            cuda_tile::DivByAttr::get(
                                ctx, kTMAAlignment, std::nullopt, std::nullopt))
                            .getResult();

    auto makeTensorViewOp = cuda_tile::MakeTensorViewOp::create(
        rewriter, loc, tensorViewTy, ptrWithDivBy, wrappedDynShapes,
        wrappedDynStrides);

    SmallVector<int32_t> dimMap(rank);
    std::iota(dimMap.begin(), dimMap.end(), 0);

    auto tileShape = op.getTensorShape();
    SmallVector<int32_t> arrayOfi32Shape;
    for (auto i64Shape : tileShape) {
      arrayOfi32Shape.push_back(i64Shape);
    }

    auto tilePartViewTy = cuda_tile::PartitionViewType::get(
        ctx, rewriter.getDenseI32ArrayAttr(arrayOfi32Shape), tensorViewTy,
        dimMap,
        cuda_tile::PaddingValueAttr::get(ctx, cuda_tile::PaddingValue::zero));

    auto partViewOp = cuda_tile::MakePartitionViewOp::create(
        rewriter, loc, tilePartViewTy, makeTensorViewOp);

    rewriter.replaceOp(op, partViewOp);
    return success();
  }
};

class ConvertMaxNumFOp : public OpConversionPattern<arith::MaxNumFOp> {
public:
  using OpConversionPattern<arith::MaxNumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    assert(resTy && "expect valid type");
    rewriter.replaceOpWithNewOp<cuda_tile::MaxFOp>(
        op, resTy, adaptor.getLhs(), adaptor.getRhs(),
        /*nan_modifier=*/nullptr,
        /*flush_to_zero_modifier=*/nullptr);
    return success();
  }
};

class ConvertMinNumFOp : public OpConversionPattern<arith::MinNumFOp> {
public:
  using OpConversionPattern<arith::MinNumFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    assert(resTy && "expect valid type");
    rewriter.replaceOpWithNewOp<cuda_tile::MinFOp>(
        op, resTy, adaptor.getLhs(), adaptor.getRhs(),
        /*nan_modifier=*/nullptr,
        /*flush_to_zero_modifier=*/nullptr);
    return success();
  }
};

class ConvertDotOp : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    SmallVector<Type> retTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), retTypes)))
      return failure();

    // Aux functions
    auto isF32 = [](Value operand) {
      return cast<RankedTensorType>(operand.getType()).getElementType().isF32();
    };
    auto zeroLike = [&](Value value) -> Value {
      auto shapedType = cast<ShapedType>(value.getType());
      Type elemType = shapedType.getElementType();

      TypedAttr zeroAttr;
      if (auto floatTy = dyn_cast<FloatType>(elemType)) {
        zeroAttr = rewriter.getFloatAttr(floatTy, 0.0);
      } else {
        llvm_unreachable("unsupported element type for zeroLike");
      }

      auto c0Attr = DenseElementsAttr::get(shapedType, zeroAttr);
      return cuda_tile::ConstantOp::create(
          rewriter, op.getLoc(), value.getType(),
          cast<DenseIntOrFPElementsAttr>(c0Attr));
    };
    auto convertFloat = [&](Value value, FloatType dstFloatTy) -> Value {
      auto srcTy = dyn_cast<cuda_tile::TileType>(value.getType());
      auto dstTy = cuda_tile::TileType::get(srcTy.getShape(), dstFloatTy);
      return cuda_tile::FToFOp::create(rewriter, op.getLoc(), dstTy, value);
    };
    auto sub = [&](Value a, Value b) -> Value {
      return cuda_tile::SubFOp::create(
          rewriter, op.getLoc(), a, b,
          cuda_tile::RoundingModeAttr::get(
              rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN),
          /*FlushToZeroModifier=*/nullptr);
    };
    auto add = [&](Value a, Value b) -> Value {
      return cuda_tile::AddFOp::create(
          rewriter, op.getLoc(), a, b,
          cuda_tile::RoundingModeAttr::get(
              rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN),
          /*FlushToZeroModifier=*/nullptr);
    };
    auto splitF32 = [&](Value input, unsigned N,
                        FloatType dstFloatTy) -> llvm::SmallVector<Value, 3> {
      llvm::SmallVector<Value, 3> splitInputs;

      FloatType f32Ty = rewriter.getF32Type();
      for (unsigned i = 0; i < N; ++i) {
        Value dstValue = convertFloat(input, dstFloatTy);
        if (i != N - 1) {
          Value dstF32Value = convertFloat(dstValue, f32Ty);
          input = sub(input, dstF32Value);
        }
        splitInputs.push_back(dstValue);
      }
      return splitInputs;
    };
    auto mma = [&](Value a, Value b, Value c) -> Value {
      return cuda_tile::MmaFOp::create(rewriter, op->getLoc(), a, b, c);
    };
    auto replaceNansWithZeros = [&](Value value) -> Value {
      // triton use arith::CmpFPredicate::UNO here, we use an ordered equal to
      // replace it
      auto nans = cuda_tile::CmpFOp::create(
          rewriter, op.getLoc(), cuda_tile::ComparisonPredicate::EQUAL,
          cuda_tile::ComparisonOrdering::ORDERED, value, value);
      auto zero = zeroLike(value);
      return cuda_tile::SelectOp::create(rewriter, op.getLoc(), nans, value,
                                         zero);
    };

    // Non-IEEE mode, mixed precision
    if (isF32(op.getA()) && isF32(op.getB()) &&
        op.getInputPrecision() != InputPrecision::IEEE) {
      FloatType computeTy;  // mma compute type
      unsigned nSplits = 0; // number of splits for lhs and rhs
      switch (op.getInputPrecision()) {
      case InputPrecision::TF32: {
        computeTy = rewriter.getTF32Type();
        nSplits = 1;
        break;
      }
      case InputPrecision::TF32x3: {
        computeTy = rewriter.getTF32Type();
        nSplits = 2;
        break;
      }
      case InputPrecision::BF16x3: {
        computeTy = rewriter.getBF16Type();
        nSplits = 2;
        break;
      }
      case InputPrecision::BF16x6: {
        computeTy = rewriter.getBF16Type();
        nSplits = 3;
        break;
      }
      default:
        return rewriter.notifyMatchFailure(op, "unsupported input precision");
      }

      const auto lhs_parts = splitF32(adaptor.getA(), nSplits, computeTy);
      const auto rhs_parts = splitF32(adaptor.getB(), nSplits, computeTy);
      const unsigned hi = 0, mid = 1, lo = 2;

      if (nSplits == 1) {
        // for TF32 mode, only one mma is needed
        auto result = mma(lhs_parts[hi], rhs_parts[hi], adaptor.getC());
        rewriter.replaceOp(op, result);
        return success();
      } else {
        // for other mixed precision modes, multiple mmas are needed
        auto result = zeroLike(adaptor.getC());
        if (nSplits > 2) {
          result = mma(lhs_parts[mid], rhs_parts[mid], result);
          result = mma(lhs_parts[lo], rhs_parts[hi], result);
          result = mma(lhs_parts[hi], rhs_parts[lo], result);
        }
        if (nSplits > 1) {
          result = mma(lhs_parts[mid], rhs_parts[hi], result);
          result = mma(lhs_parts[hi], rhs_parts[mid], result);
          result = replaceNansWithZeros(result);
        }
        result = mma(lhs_parts[hi], rhs_parts[hi], result);
        result = add(result, adaptor.getC());
        rewriter.replaceOp(op, result);
        return success();
      }
    }

    // IEEE mode, directly lower to mma
    auto opElType =
        cast<cuda_tile::TileType>(adaptor.getA().getType()).getElementType();
    if (opElType.isInteger(8)) {
      // To lower IMMA, we must distinguish between signed and unsigned at the
      // operation level. Triton IR is signless, and there are no attributes for
      // us to recover this information. Hence, here, for integer type, we
      // default to signed.
      rewriter.replaceOpWithNewOp<cuda_tile::MmaIOp>(
          op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
          cuda_tile::Signedness::Signed, cuda_tile::Signedness::Signed);
    } else if (opElType.isFloat()) {
      rewriter.replaceOpWithNewOp<cuda_tile::MmaFOp>(
          op, adaptor.getA(), adaptor.getB(), adaptor.getC());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported operand types of mma op");
    }
    return success();
  }
};

class ConvertTransOp : public OpConversionPattern<triton::TransOp> {
public:
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    assert(resTy && "expect valid type");
    // We need to replace the attribute, so we cannot use ConvertGenericOp.
    rewriter.replaceOpWithNewOp<cuda_tile::PermuteOp>(
        op, resTy, adaptor.getSrc(), op.getOrder());
    return success();
  }
};

class ConvertAssertOp : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cuda_tile::AssertOp>(op, adaptor.getCondition(),
                                                     op.getMessage());
    return success();
  }
};

class ConvertRsqrtOp : public OpConversionPattern<math::RsqrtOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(math::RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "typeConversion of rsqrt op failed");
    rewriter.replaceOpWithNewOp<cuda_tile::RsqrtOp>(op, resTy,
                                                    adaptor.getOperand());
    return success();
  }
};

static cuda_tile::AtomicRMWMode
convertAtomicModeToCudaTile(triton::RMWOp rmwOp) {
  switch (rmwOp) {
  case triton::RMWOp::AND:
    return cuda_tile::AtomicRMWMode::AND;
  case triton::RMWOp::OR:
    return cuda_tile::AtomicRMWMode::OR;
  case triton::RMWOp::XOR:
    return cuda_tile::AtomicRMWMode::XOR;
  case triton::RMWOp::ADD:
    return cuda_tile::AtomicRMWMode::ADD;
  case triton::RMWOp::FADD:
    return cuda_tile::AtomicRMWMode::ADDF;
  case triton::RMWOp::MAX:
    return cuda_tile::AtomicRMWMode::MAX;
  case triton::RMWOp::MIN:
    return cuda_tile::AtomicRMWMode::MIN;
  case triton::RMWOp::UMAX:
    return cuda_tile::AtomicRMWMode::UMAX;
  case triton::RMWOp::UMIN:
    return cuda_tile::AtomicRMWMode::UMIN;
  case triton::RMWOp::XCHG:
    return cuda_tile::AtomicRMWMode::XCHG;
  default:
    llvm_unreachable("unknown RMW mode");
  }
}

static cuda_tile::MemoryOrderingSemantics
convertMemorySemToCudaTile(triton::MemSemantic sem) {
  switch (sem) {
  case triton::MemSemantic::RELAXED:
    return cuda_tile::MemoryOrderingSemantics::RELAXED;
  case triton::MemSemantic::ACQUIRE:
    return cuda_tile::MemoryOrderingSemantics::ACQUIRE;
  case triton::MemSemantic::RELEASE:
    return cuda_tile::MemoryOrderingSemantics::RELEASE;
  case triton::MemSemantic::ACQUIRE_RELEASE:
    return cuda_tile::MemoryOrderingSemantics::ACQ_REL;
  default:
    llvm_unreachable("unknown memory sem mode");
  }
}

static cuda_tile::MemoryScope
convertMemoryScopeToCudaTile(triton::MemSyncScope scope) {
  switch (scope) {
  case triton::MemSyncScope::GPU:
    return cuda_tile::MemoryScope::DEVICE;
  // We do not expose CTA use TL_BLK instead.
  case triton::MemSyncScope::CTA:
    return cuda_tile::MemoryScope::TL_BLK;
  case triton::MemSyncScope::SYSTEM:
    return cuda_tile::MemoryScope::SYS;
  default:
    llvm_unreachable("unknown memory scope mode");
  }
}

static cuda_tile::RoundingMode
convertRoundingModeToCudaTile(triton::RoundingMode rounding) {
  switch (rounding) {
  case triton::RoundingMode::RTZ:
    return cuda_tile::RoundingMode::ZERO;
  case triton::RoundingMode::RTNE:
    return cuda_tile::RoundingMode::NEAREST_EVEN;
  default:
    llvm_unreachable("unknown rounding mode");
  }
}

class ConvertAtomicRMWOp : public OpConversionPattern<triton::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto scalarTy = getElementTypeOrSelf(retType);
    auto mode = convertAtomicModeToCudaTile(op.getAtomicRmwOp());
    auto newRMWOp = cuda_tile::AtomicRMWTkoOp::create(
        rewriter, op.getLoc(), retType,
        cuda_tile::TokenType::get(rewriter.getContext()),
        convertMemorySemToCudaTile(op.getSem()),
        convertMemoryScopeToCudaTile(op.getScope()), adaptor.getPtr(), mode,
        adaptor.getVal(), adaptor.getMask(), /*token=*/nullptr);
    rewriter.replaceOp(op, newRMWOp.getResult());
    return success();
  }
};

class ConvertAtomicCASOp : public OpConversionPattern<triton::AtomicCASOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto newCASOp = cuda_tile::AtomicCASTkoOp::create(
        rewriter, op.getLoc(), retType,
        cuda_tile::TokenType::get(rewriter.getContext()),
        convertMemorySemToCudaTile(op.getSem()),
        convertMemoryScopeToCudaTile(op.getScope()), adaptor.getPtr(),
        adaptor.getCmp(), adaptor.getVal(), /*mask=*/Value(),
        /*token=*/nullptr);
    rewriter.replaceOp(op, newCASOp.getResult());
    return success();
  }
};

// Clamp operation clamps a value x between min and max bounds:
// clamp(x, min, max) = min(max(x, min), max)
//
// Examples:
// For x = -3 with bounds [0,2]:
//   max(-3, 0) = 0   // First clamp to lower bound
//   min(0, 2) = 0    // Then clamp to upper bound
//
// For x = 5 with bounds [0,2]:
//   max(5, 0) = 5    // First clamp to lower bound
//   min(5, 2) = 2    // Then clamp to upper bound
//
// The operation can either propagate NaN values (ALL) or not (NONE)
class ConvertClampFOp : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto converter = getTypeConverter();
    Type xTy = converter->convertType(op.getX().getType());
    if (!xTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert operand types of clampf");

    auto nanModifier = op.getPropagateNan() == PropagateNan::ALL
                           ? rewriter.getUnitAttr()
                           : nullptr;

    auto v = cuda_tile::MaxFOp::create(rewriter, loc, xTy, adaptor.getX(),
                                       adaptor.getMin(), nanModifier,
                                       /*flush_to_zero_modifier=*/nullptr);
    rewriter.replaceOpWithNewOp<cuda_tile::MinFOp>(
        op, v, adaptor.getMax(), nanModifier,
        /*flush_to_zero_modifier=*/nullptr);
    return success();
  }
};

class ConvertSplitOp : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types.
    assert(op.getResult(0).getType() == op.getResult(1).getType() &&
           "result type mismatch");
    Type resultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Split the last dimension with two cuda_tile.extract.
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    cuda_tile::TileType srcType = cast<cuda_tile::TileType>(src.getType());
    cuda_tile::TileType constType =
        cuda_tile::TileType::get({}, rewriter.getI32Type());
    auto c0Attr = DenseIntElementsAttr::get(constType, {0});
    Value c0 = cuda_tile::ConstantOp::create(rewriter, loc, constType, c0Attr);
    auto c1Attr = DenseIntElementsAttr::get(constType, {1});
    Value c1 = cuda_tile::ConstantOp::create(rewriter, loc, constType, c1Attr);
    SmallVector<Value> indices0(srcType.getRank() - 1, c0);
    SmallVector<Value> indices1(srcType.getRank() - 1, c0);
    indices0.push_back(c0);
    indices1.push_back(c1);
    SmallVector<int64_t> extractedShape =
        llvm::to_vector(llvm::drop_end(srcType.getShape()));
    extractedShape.push_back(1);
    auto extractedType =
        cuda_tile::TileType::get(extractedShape, srcType.getElementType());
    Value extract0 = cuda_tile::ExtractOp::create(rewriter, loc, extractedType,
                                                  src, indices0);
    Value extract1 = cuda_tile::ExtractOp::create(rewriter, loc, extractedType,
                                                  src, indices1);

    // Drop the last dimension.
    SmallVector<Value> repls;
    repls.push_back(
        cuda_tile::ReshapeOp::create(rewriter, loc, resultType, extract0));
    repls.push_back(
        cuda_tile::ReshapeOp::create(rewriter, loc, resultType, extract1));
    rewriter.replaceOp(op, repls);
    return success();
  }
};

class ConvertFpToFpOp : public OpConversionPattern<triton::FpToFpOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    SmallVector<Type> retTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), retTypes)))
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "typeConversion for FpToFpOp failed");

    auto roundingMode = op.getRounding();
    auto cudaTileRoundingMode = cuda_tile::RoundingMode::NEAREST_EVEN;
    if (roundingMode.has_value())
      cudaTileRoundingMode =
          convertRoundingModeToCudaTile(roundingMode.value());
    auto cudaTileRoundingModeAttr = cuda_tile::RoundingModeAttr::get(
        rewriter.getContext(), cudaTileRoundingMode);
    rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(
        op, retTypes, adaptor.getSrc(), cudaTileRoundingModeAttr);
    return success();
  }
};

void populateTTirToCudaTileConversionPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, bool approx, bool flushToZero,
    DenseMap<Operation *, int> &numStagesMap, int computeCapability,
    int numCTAInCGA, int simtNumWarpsInCTA, int occupancy,
    std::optional<int> numStages) {
  MLIRContext *context = patterns.getContext();
  // clang-format off
  patterns.add<
    // Arith operations
    ConvertGenericOp<arith::AddFOp, cuda_tile::AddFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::AddIOp, cuda_tile::AddIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::AndIOp, cuda_tile::AndIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::BitcastOp, cuda_tile::BitcastOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::CeilDivSIOp, cuda_tile::DivIOp, Signedness::Signed, IntegerUpCast::None>,
    ConvertGenericOp<arith::CeilDivUIOp, cuda_tile::DivIOp, Signedness::Unsigned, IntegerUpCast::None>,
    ConvertGenericOp<arith::DivFOp, cuda_tile::DivFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::DivSIOp, cuda_tile::DivIOp, Signedness::Signed, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::DivUIOp, cuda_tile::DivIOp, Signedness::Unsigned, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::ExtFOp, cuda_tile::FToFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::ExtSIOp, cuda_tile::ExtIOp, Signedness::Signed, IntegerUpCast::None>,
    ConvertGenericOp<arith::ExtUIOp, cuda_tile::ExtIOp, Signedness::Unsigned, IntegerUpCast::None>,
    ConvertGenericOp<arith::FloorDivSIOp, cuda_tile::DivIOp, Signedness::Signed, IntegerUpCast::None>,
    ConvertGenericOp<arith::FPToSIOp, cuda_tile::FToIOp, Signedness::Signed, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::FPToUIOp, cuda_tile::FToIOp, Signedness::Unsigned, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::MaxSIOp, cuda_tile::MaxIOp, Signedness::Signed, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::MaxUIOp, cuda_tile::MaxIOp, Signedness::Unsigned, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::MinSIOp, cuda_tile::MinIOp, Signedness::Signed, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::MinUIOp, cuda_tile::MinIOp, Signedness::Unsigned, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::MulFOp, cuda_tile::MulFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::MulIOp, cuda_tile::MulIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::NegFOp, cuda_tile::NegFOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::OrIOp, cuda_tile::OrIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::RemFOp, cuda_tile::RemFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::RemSIOp, cuda_tile::RemIOp, Signedness::Signed, IntegerUpCast::None>,
    ConvertGenericOp<arith::RemUIOp, cuda_tile::RemIOp, Signedness::Unsigned, IntegerUpCast::None>,
    ConvertGenericOp<arith::ShLIOp, cuda_tile::ShLIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::ShRSIOp, cuda_tile::ShRIOp, Signedness::Signed, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::ShRUIOp, cuda_tile::ShRIOp, Signedness::Unsigned, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::SIToFPOp, cuda_tile::IToFOp, Signedness::Signed, IntegerUpCast::None>,
    ConvertGenericOp<arith::SubFOp, cuda_tile::SubFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::SubIOp, cuda_tile::SubIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<arith::TruncFOp, cuda_tile::FToFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::TruncIOp, cuda_tile::TruncIOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<arith::UIToFPOp, cuda_tile::IToFOp, Signedness::Unsigned, IntegerUpCast::None>,
    ConvertGenericOp<arith::XOrIOp, cuda_tile::XOrIOp, Signedness::None, IntegerUpCast::To_I16>,

    // Math operations
    ConvertGenericOp<math::AbsIOp, cuda_tile::AbsIOp, Signedness::None, IntegerUpCast::To_I16>,
    ConvertGenericOp<math::CeilOp, cuda_tile::CeilOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::CosOp, cuda_tile::CosOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::CoshOp, cuda_tile::CosHOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::Exp2Op, cuda_tile::Exp2Op, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::ExpOp, cuda_tile::ExpOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::FloorOp, cuda_tile::FloorOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::FmaOp, cuda_tile::FmaOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::Log2Op, cuda_tile::Log2Op, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::PowFOp, cuda_tile::PowOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::SinOp, cuda_tile::SinOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::SinhOp, cuda_tile::SinHOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::SqrtOp, cuda_tile::SqrtOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::TanOp, cuda_tile::TanOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<math::TanhOp, cuda_tile::TanHOp, Signedness::None, IntegerUpCast::None>,

    // Triton operations
    ConvertGenericOp<triton::AddPtrOp, cuda_tile::OffsetOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<triton::IntToPtrOp, cuda_tile::IntToPtrOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<triton::MulhiUIOp, cuda_tile::MulhiIOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<triton::PtrToIntOp, cuda_tile::PtrToIntOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<triton::PreciseDivFOp, cuda_tile::DivFOp, Signedness::None, IntegerUpCast::None>,
    ConvertGenericOp<triton::PreciseSqrtOp, cuda_tile::SqrtOp, Signedness::None, IntegerUpCast::None>
>(typeConverter, context, approx, flushToZero);

  patterns.add<ConvertFuncOp>(typeConverter, context, computeCapability, numCTAInCGA, simtNumWarpsInCTA, occupancy);
  patterns.add<
    ConvertGenericOp<math::LogOp, cuda_tile::LogOp, Signedness::None, IntegerUpCast::None>
  >(typeConverter, context);

  patterns.add<
    ConvertAbsFOp,
    ConvertAssertOp,
    ConvertAtomicCASOp,
    ConvertAtomicRMWOp,
    ConvertBitcastOp,
    ConvertBroadCastOp,
    ConvertCatOp,
    ConvertClampFOp,
    ConvertCmpFOp,
    ConvertCmpIOp,
    ConvertConstantOp,
    ConvertDotOp,
    ConvertErfOp,
    ConvertExpandDimsOp,
    ConvertExternElementwiseOp,
    ConvertForOp,
    ConvertFpToFpOp,
    ConvertGetNumProgramsOp,
    ConvertGetProgramIdOp,
    ConvertJoinOp,
    ConvertMakeRangeOp,
    ConvertMaxNumFOp,
    ConvertMaximumFOp,
    ConvertMinNumFOp,
    ConvertMinimumFOp,
    ConvertPrintOp,
    ConvertReduceOp,
    ConvertReduceReturnOp,
    ConvertReshapeOp,
    ConvertReturnOp,
    ConvertRsqrtOp,
    ConvertScanOp,
    ConvertScanReturnOp,
    ConvertSelectOp,
    ConvertSplatOp,
    ConvertSplitOp,
    ConvertTransOp,
    ConvertIfOp,
    ConvertUnsplatOp,
    ConvertWhileOp,
    ConvertYieldOp
>(typeConverter, context);

  patterns.add<ConvertLoadOp, ConvertStoreOp>(typeConverter, context, numStagesMap, computeCapability, numStages);

    patterns.add<ConvertDescriptorLoadOp, ConvertDescriptorStoreOp>(
        typeConverter, context, numStagesMap, computeCapability, numStages);
  patterns.add<ConvertMakeTensorDescOp>(context);
  // clang-format on
}

/// Convert the given tt.constancy attribute.
static Value convertConstAttr(OpBuilder &b, Value v, Location loc,
                              const AxisInfo &info) {
  auto type = cast<cuda_tile::TileType>(v.getType());
  if (type.getRank() == 0)
    return v;
  assert(type.getRank() == info.getRank() && "rank mismatch");
  auto pred = cuda_tile::SameElementsAttr::get(
      b.getContext(), b.getDenseI64ArrayAttr(info.getConstancy()));
  return cuda_tile::AssumeOp::create(b, loc, v, pred);
}

/// Insert a cuda_tile.assume op based on the divisibility / contiguity of the
/// given Triton axis attributes.
static Value convertDivByAndContAttr(OpBuilder &b, Value v, Location loc,
                                     const AxisInfo &info) {
  auto type = cast<cuda_tile::TileType>(v.getType());

  // Find the dimension with the largest divisibility.
  int64_t divisor = 1;
  std::optional<int64_t> every = std::nullopt;
  std::optional<int64_t> along = std::nullopt;
  for (int64_t i = 0, e = info.getRank(); i < e; ++i) {
    if (info.getDivisibility(i) > divisor ||
        (info.getDivisibility(i) == divisor &&
         info.getContiguity(i) > every.value_or(1))) {
      divisor = info.getDivisibility(i);
      every = info.getContiguity(i);
      along = i;
    }
  }

  if (type.getRank() == 0) {
    // Rank 0 (scalar): drop contiguity.
    every = std::nullopt;
    along = std::nullopt;
  }

  cuda_tile::DivByAttr pred =
      cuda_tile::DivByAttr::get(b.getContext(), divisor, every, along);
  return cuda_tile::AssumeOp::create(b, loc, v, pred);
}

/// Helper struct that stores the Triton axis information for a given SSA
/// value, which was injected by the user. The AxisInfo object stores not only
/// the injected information. That's because divisibility and contiguity in
/// Triton can be set independently, whereas they always come as a pair in
/// cuda_tile. (And must be set together in cuda_tile.)
struct Assumption {
  Assumption(Value value, const AxisInfo &info, bool hasDivByAttr,
             bool hasContAttr, bool hasConstrAttr)
      : value(value), info(info), hasDivByAttr(hasDivByAttr),
        hasContAttr(hasContAttr), hasConstAttr(hasConstrAttr) {}
  Value value;
  AxisInfo info;
  bool hasDivByAttr;
  bool hasContAttr;
  bool hasConstAttr;
};

/// Create a cuda_tile.assume op for the given assumption.
static void assumeAxisAttributes(RewriterBase &rewriter,
                                 TypeConverter &converter,
                                 const Assumption &assumption) {
  assert((assumption.hasDivByAttr || assumption.hasContAttr ||
          assumption.hasConstAttr) &&
         "no attributes found to forward");
  OpBuilder::InsertionGuard g(rewriter);
  Value v = assumption.value;
  Location loc = v.getLoc();
  if (auto bbArg = dyn_cast<BlockArgument>(v)) {
    rewriter.setInsertionPointToStart(bbArg.getOwner());
  } else {
    rewriter.setInsertionPointAfter(v.getDefiningOp());
  }

  // Insert an unrealized_conversion_cast to the respective cuda_tile type.
  Type convertedType = converter.convertType(v.getType());
  assert(convertedType && "could not convert type");
  auto tileType = dyn_cast<cuda_tile::TileType>(convertedType);
  assert(tileType && "axis attribute not supported on non-tensor type");
  if (!isa<IntegerType, cuda_tile::PointerType>(tileType.getElementType()))
    return;
  Value val =
      UnrealizedConversionCastOp::create(rewriter, loc, convertedType, v)
          .getResult(0);
  Operation *firstCast = val.getDefiningOp();

  // Create cuda_tile.assume op.
  if (assumption.hasConstAttr)
    val = convertConstAttr(rewriter, val, loc, assumption.info);
  if (assumption.hasDivByAttr || assumption.hasContAttr)
    val = convertDivByAndContAttr(rewriter, val, loc, assumption.info);

  // Insert an unrealized_conversion_cast back to the original type.
  val = UnrealizedConversionCastOp::create(rewriter, loc, v.getType(), val)
            .getResult(0);
  rewriter.replaceAllUsesExcept(v, val, firstCast);
}

static void getNumStages(Operation *op,
                         DenseMap<Operation *, int> &numStagesMap) {
  op->walk([&](Operation *op) {
    if (isa<triton::DescriptorLoadOp, triton::DescriptorStoreOp, triton::LoadOp,
            triton::StoreOp>(op)) {
      auto numStages = mlir::triton::utils::getNumStagesFromParentForOp(op);
      if (numStages.has_value()) {
        numStagesMap.insert({op, numStages.value()});
      }
    }
    return WalkResult::advance();
  });
}

static void
checkDivisibilityForDescriptorOps(mlir::ModuleOp op,
                                  ModuleAxisInfoAnalysis &axisInfo) {
  auto checkDivisibility = [&](Operation *op, mlir::ValueRange indices,
                               Value desc) -> void {
    auto tensorDescType = dyn_cast<triton::TensorDescType>(desc.getType());
    if (!tensorDescType)
      return;
    auto tileSizes = tensorDescType.getBlockType().getShape();
    for (size_t i = 0; i < indices.size(); i++) {
      auto *info = axisInfo.getAxisInfo(indices[i]);
      if (!info) {
        op->emitWarning()
            << "divisibility info not found for offset in dimension " << i
            << " (offset: " << indices[i] << "), "
            << "but tile size requires divisibility by " << tileSizes[i]
            << " in tileir 13.1. Please add tl.assume(offset % block_shape == "
               "0) if the offset is always divisible by the block shape.";
      }
      auto divisibility = info->getDivisibility(0);
      if (divisibility % tileSizes[i] != 0) {
        op->emitWarning()
            << "divisibility at dimension " << i << ": offset divisibility is "
            << divisibility << " but tile size requires divisibility by "
            << tileSizes[i]
            << " in tileir 13.1. Please add tl.assume(offset % block_shape == "
               "0) if the offset is always divisible by the block shape.";
      }
    }
  };

  op->walk([&](Operation *op) {
    if (auto descriptorStoreOp = dyn_cast<triton::DescriptorStoreOp>(op)) {
      checkDivisibility(op, descriptorStoreOp.getIndices(),
                        descriptorStoreOp.getDesc());
    } else if (auto descriptorLoadOp = dyn_cast<triton::DescriptorLoadOp>(op)) {
      checkDivisibility(op, descriptorLoadOp.getIndices(),
                        descriptorLoadOp.getDesc());
    }
  });
}

static void convertTmaDescriptorOps(Operation *op, TypeConverter &converter) {
  IRRewriter rewriter(op->getContext());
  auto ctx = op->getContext();
  auto loc = op->getLoc();
  op->walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
        Value tensorDesc = funcOp.getArgument(i);
        if (isa<triton::TensorDescType>(tensorDesc.getType())) {
          // [tensordesc, ptr, shape, stride]
          // 'i' is the tensordesc type
          int argIdx = i;
          rewriter.setInsertionPointToStart(&funcOp.getBody().front());
          auto tensorDescType =
              cast<triton::TensorDescType>(tensorDesc.getType());
          auto descBlock = tensorDescType.getBlockType();
          auto rank = descBlock.getRank();
          if (rank == 0) {
            op->emitError("Host TMA descriptor with rank 0 is not supported.");
            return WalkResult::interrupt();
          }

          auto pointeeType = descBlock.getElementType();
          if (auto intTy = dyn_cast<mlir::IntegerType>(pointeeType)) {
            pointeeType = mlir::IntegerType::get(ctx, intTy.getWidth(),
                                                 mlir::IntegerType::Signless);
          }
          unsigned elem_bytes = pointeeType.getIntOrFloatBitWidth() / 8;
          unsigned align_byte = kTMAAlignment / elem_bytes;

          // 'i + 1' is the pointer of the global tensor
          auto ptrArg = funcOp.getArgument(i + 1);
          i = i + 1;
          SmallVector<Value> shape;
          for (size_t j = 0; j < rank; j++)
            shape.push_back(funcOp.getArgument(i + j + 1));
          i = i + rank;
          SmallVector<Value> stride;
          for (size_t j = 0; j + 1 < rank; j++)
            stride.push_back(funcOp.getArgument(i + j + 1));
          i = i + rank - 1;

          SmallVector<Value> wrappedDynShapes;
          // It is safe to assume that shapes are in the range [0, 2^32 - 1]
          // which is the TMA hardware limit for shape. This ensures that if
          // users explicitly want to use TMA for this operation, the shape
          // parameters will satisfy TMA descriptor encoding requirements.
          for (auto v : shape)
            wrappedDynShapes.push_back(wrapIntoScalarTile(
                rewriter, v, /*attachAlignment=*/0, kMaxShape));

          SmallVector<Value> wrappedDynStrides;
          // It is safe to assume that strides are in the range [0, 2^40 - 1]
          // which is the TMA hardware limit for stride. This ensures that if
          // users explicitly want to use TMA for this operation, the stride
          // parameters will satisfy TMA descriptor encoding requirements.
          for (int i = 0; i < stride.size(); i++)
            wrappedDynStrides.push_back(
                wrapIntoScalarTile(rewriter, stride[i],
                                   /*attachAlignment=*/align_byte, kMaxStride));

          auto tileIRPtrType = cuda_tile::PointerType::get(pointeeType);
          auto ptrTypeWrapper =
              cuda_tile::TileType::get(ctx, {}, tileIRPtrType);
          auto ptrOp = UnrealizedConversionCastOp::create(
                           rewriter, loc, ptrTypeWrapper, ptrArg)
                           .getResult(0);

          // we can always assume the pointer is divisible by 16
          // because openai has already make it into host tma api.
          auto ptrWithDivBy =
              cuda_tile::AssumeOp::create(
                  rewriter, loc, ptrOp,
                  cuda_tile::DivByAttr::get(ctx, kTMAAlignment, std::nullopt,
                                            std::nullopt))
                  .getResult();

          SmallVector<int64_t> globalShape(rank,
                                           cuda_tile::TensorViewType::kDynamic);
          SmallVector<int64_t> globalStride(
              rank, cuda_tile::TensorViewType::kDynamic);
          // we can always assume the stride is 1
          // because openai has assume this.
          globalStride[globalShape.size() - 1] = 1;

          auto tensorViewTy = cuda_tile::TensorViewType::get(
              ctx, tileIRPtrType.getPointeeType(), globalShape, globalStride);
          auto makeTensorViewOp = cuda_tile::MakeTensorViewOp::create(
              rewriter, loc, tensorViewTy, ptrWithDivBy, wrappedDynShapes,
              wrappedDynStrides);

          auto tileShape = descBlock.getShape();
          SmallVector<int32_t> arrayOfi32Shape;
          for (auto i64Shape : tileShape) {
            arrayOfi32Shape.push_back(i64Shape);
          }

          SmallVector<int32_t> dimMap(rank);
          std::iota(dimMap.begin(), dimMap.end(), 0);

          auto tilePartViewTy = cuda_tile::PartitionViewType::get(
              ctx, rewriter.getDenseI32ArrayAttr(arrayOfi32Shape), tensorViewTy,
              dimMap,
              cuda_tile::PaddingValueAttr::get(ctx,
                                               cuda_tile::PaddingValue::zero));

          auto partViewOp = cuda_tile::MakePartitionViewOp::create(
              rewriter, loc, tilePartViewTy, makeTensorViewOp);

          auto castBackToTensorDescriptorOp =
              UnrealizedConversionCastOp::create(rewriter, loc, tensorDescType,
                                                 partViewOp.getResult());

          rewriter.replaceAllUsesWith(
              tensorDesc, castBackToTensorDescriptorOp.getResult(0));
        }
      }
    }
    return WalkResult::advance();
  });
}

/// Convert attributes that are related to the axis analysis.
static void convertAxisAttributes(mlir::ModuleOp op,
                                  ModuleAxisInfoAnalysis &axisInfo,
                                  TypeConverter &converter) {
  SmallVector<Assumption> assumptions;

  // Find all tt.divisibility, tt.contiguity, tt.constancy attributes. For each
  // such value, do not read the value directly, but query the Triton AxisInfo.
  op->walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      // Convert attributes that are attached to function block arguments.
      for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
        auto divByAttr = funcOp.getArgAttr(i, "tt.divisibility");
        auto contAttr = funcOp.getArgAttr(i, "tt.contiguity");
        auto constAttr = funcOp.getArgAttr(i, "tt.constancy");
        if (divByAttr || contAttr || constAttr) {
          Value v = funcOp.getArgument(i);
          assumptions.emplace_back(
              v, *axisInfo.getAxisInfo(v), static_cast<bool>(divByAttr),
              static_cast<bool>(contAttr), static_cast<bool>(constAttr));
        }
      }
      return WalkResult::advance();
    }

    // Convert attributes that are attached to operations.
    auto divByAttr = op->getDiscardableAttr("tt.divisibility");
    auto contAttr = op->getDiscardableAttr("tt.contiguity");
    auto constAttr = op->getDiscardableAttr("tt.constancy");
    if (divByAttr || contAttr || constAttr) {
      assert(op->getNumResults() == 1 && "expected op with single result");
      Value v = op->getResult(0);
      assumptions.emplace_back(
          v, *axisInfo.getAxisInfo(v), static_cast<bool>(divByAttr),
          static_cast<bool>(contAttr), static_cast<bool>(constAttr));
    }
    return WalkResult::advance();
  });

  // Now materialize all assumptions as cuda_tile.assume ops. This is not done
  // during the above loop because modifying IR invalidates the axis analysis.
  IRRewriter rewriter(op->getContext());
  for (const auto &assumption : assumptions)
    assumeAxisAttributes(rewriter, converter, assumption);
}

// map_elementwise pre-processing functions are in MapElementwiseExpansion.cpp.
// expandMapElementwiseOps() is called from runOnOperation() below.

struct ConvertTritonToCudaTile
    : public ::mlir::triton::impl::ConvertTritonToCudaTileBase<
          ConvertTritonToCudaTile> {
public:
  // Map from load/store operations to num_stages from its parent ForOp.
  DenseMap<Operation *, int> numStagesMap;
  // Value of per-kernel num_stages.
  std::optional<int> numStages;

  ConvertTritonToCudaTile() = default;
  ConvertTritonToCudaTile(bool approxModifier, bool flushToZeroModifier,
                          int computeCapability, int numCTAInCGA,
                          int simtNumWarpsInCTA, int occupancy,
                          std::optional<int> numStages) {
    this->approxModifier = approxModifier;
    this->flushToZeroModifier = flushToZeroModifier;
    this->computeCapability = computeCapability;
    this->numCTAInCGA = numCTAInCGA;
    this->simtNumWarpsInCTA = simtNumWarpsInCTA;
    this->occupancy = occupancy;
    this->numStages = numStages;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod_buildin = getOperation();
    CudaTileTypeConverter typeConverter;

    // Insert cuda tile module directly.
    OpBuilder builder(context);
    auto mod = cuda_tile::ModuleOp::create(builder, mod_buildin.getLoc(),
                                           "cuda_tile_module");
    auto &region = mod.getBodyRegion();
    region.getBlocks().clear();
    IRMapping mapping;
    mod_buildin.getBodyRegion().cloneInto(&region, mapping);
    auto &block = mod_buildin.getBodyRegion().front();
    block.clear();
    block.push_front(mod);

    // Insert Host TMA descriptor ops.
    convertTmaDescriptorOps(mod.getOperation(), typeConverter);

    ModuleAxisInfoAnalysis axisInfo(mod_buildin);

    // Check divisibility for all indices in descriptor load and store ops.
    checkDivisibilityForDescriptorOps(mod_buildin, axisInfo);

    // Convert all axis attributes.
    convertAxisAttributes(mod_buildin, axisInfo, typeConverter);

    // Get num_stages for load/store ops.
    getNumStages(mod.getOperation(), this->numStagesMap);

    // Pre-processing: expand tt.map_elementwise into tensor-level ops.
    // This must run before dialect conversion so the expanded arith/math ops
    // get picked up by ConvertGenericOp patterns.
    if (failed(mlir::triton::expandMapElementwiseOps(mod.getOperation())))
      return signalPassFailure();

    // Dialect conversion: Convert all operations.
    CudaTileConversionTarget target(*context, typeConverter);
    RewritePatternSet patterns(context);
    populateTTirToCudaTileConversionPatternsAndLegality(
        typeConverter, patterns, target, this->approxModifier,
        this->flushToZeroModifier, this->numStagesMap, this->computeCapability,
        this->numCTAInCGA, this->simtNumWarpsInCTA, this->occupancy,
        this->numStages);

    ConversionConfig config = ConversionConfig();
    config.buildMaterializations = false;
    // use full conversion here to allow only know operations since cuda_tile
    // doesn't allow other dialect's ops
    if (failed(applyFullConversion(mod, target, std::move(patterns), config)))
      return signalPassFailure();

    // Try to reconcile as many unrealized_conversion_cast ops as possible.
    SmallVector<UnrealizedConversionCastOp> castOps, remainingCastOps;
    mod->walk([&](UnrealizedConversionCastOp op) { castOps.push_back(op); });
    reconcileUnrealizedCasts(castOps, &remainingCastOps);

    // Required to clean up any remaining unrealized casts and ensure IR
    // validity after dialect conversion. Without this, subsequent passes may
    // fail due to invalid IR structure or unreconciled casts.
    {
      RewritePatternSet patterns(context);
      if (failed(applyPatternsGreedily(mod, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToCudaTilePass() {
  return std::make_unique<ConvertTritonToCudaTile>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToCudaTilePass(bool approx, bool ftz,
                                                int capability, int num_ctas,
                                                int simt_num_warps,
                                                int occupancy,
                                                std::optional<int> num_stages) {
  return std::make_unique<ConvertTritonToCudaTile>(
      approx, ftz, capability, num_ctas, simt_num_warps, occupancy, num_stages);
}
