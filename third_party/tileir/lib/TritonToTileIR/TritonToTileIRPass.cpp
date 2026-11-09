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
#include <string>
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
    auto numWorkerWarpsAttr = rewriter.getNamedAttr(
        stringifyHintKey(cuda_tile::HintKey::NumWorkerWarpsPerCTA),
        rewriter.getI32IntegerAttr(simtNumWarpsInCTA));
    auto occupancyAttr = rewriter.getNamedAttr(
        "occupancy", rewriter.getI32IntegerAttr(occupancy));
    auto hintEntry = rewriter.getNamedAttr(
        (archPrefix + llvm::Twine(computeCapability)).str(),
        DictionaryAttr::get(ctx,
                            {numCTAAttr, numWorkerWarpsAttr, occupancyAttr}));
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
    for (auto &inType : modifiedInputs) {
      if (isa<triton::TensorDescType>(inType))
        inType = rewriter.getI32Type();
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

static Value traceUnrealizedConversionCast(Value value) {
  Value current = value;
  // Follow the chain of unrealized_conversion_cast operations
  while (auto castOp = current.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      current = castOp.getInputs()[0];
    } else {
      break;
    }
  }
  return current;
}

// This marker exists only between host descriptor reconstruction and read
// conversion. The descriptor ABI and every memory view remain f32.
static constexpr llvm::StringLiteral kRoundHostDescriptor =
    "tileir.round_f32_to_tf32";

static Value roundHostDescriptorRead(ConversionPatternRewriter &rewriter,
                                     Operation *read, Value value) {
  if (!read->hasAttr(kRoundHostDescriptor))
    return value;
  auto loc = read->getLoc();
  auto f32Ty = cast<cuda_tile::TileType>(value.getType());
  assert(f32Ty.getElementType().isF32());
  auto tf32Ty = cuda_tile::TileType::get(
      f32Ty.getShape(), FloatTF32Type::get(rewriter.getContext()));
  auto rn = cuda_tile::RoundingModeAttr::get(
      rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN);
  Value tf32 = cuda_tile::FToFOp::create(rewriter, loc, tf32Ty, value, rn);
  Value rounded = cuda_tile::FToFOp::create(rewriter, loc, f32Ty, tf32, rn);

  // OAIT's descriptor contract preserves all NaN/Inf source bits, including
  // NaN payloads. Native ftof promises the NaN value class, not its payload.
  // Preserve special values with an integer select; all finite rounding above
  // is performed by native ftof, with no integer rounding approximation.
  auto bitsTy =
      cuda_tile::TileType::get(f32Ty.getShape(), rewriter.getI32Type());
  Value bits = cuda_tile::BitcastOp::create(rewriter, loc, bitsTy, value);
  Value expMask = cuda_tile::ConstantOp::create(
      rewriter, loc, bitsTy, DenseIntElementsAttr::get(bitsTy, {0x7f800000}));
  Value exponent = cuda_tile::AndIOp::create(rewriter, loc, bits, expMask);
  Value special = cuda_tile::CmpIOp::create(
      rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL, exponent, expMask,
      cuda_tile::Signedness::Unsigned);
  Value roundedBits =
      cuda_tile::BitcastOp::create(rewriter, loc, bitsTy, rounded);
  Value resultBits = cuda_tile::SelectOp::create(rewriter, loc, bitsTy, special,
                                                 bits, roundedBits);
  return cuda_tile::BitcastOp::create(rewriter, loc, f32Ty, resultBits);
}

class ConvertDescriptorGatherOp
    : public OpConversionPattern<triton::DescriptorGatherOp> {
public:
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;

  ConvertDescriptorGatherOp(TypeConverter &typeConverter, MLIRContext *context,
                            DenseMap<Operation *, int> &numStagesMap,
                            int computeCapability, std::optional<int> numStages)
      : OpConversionPattern<triton::DescriptorGatherOp>(typeConverter, context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op->getContext();

    // 1. Get the descriptor and trace through conversion casts
    Value view = adaptor.getDesc();
    view = traceUnrealizedConversionCast(view);
    Type viewTy = view.getType();

    // 2. Extract the underlying tensor view
    Value srcTensorView;

    // If view is already a TensorViewType, use it directly
    if (isa<cuda_tile::TensorViewType>(viewTy)) {
      srcTensorView = view;
    }
    // If view is a PartitionViewType or StridedViewType, extract the tensor
    // view
    else if (isa<cuda_tile::PartitionViewType, cuda_tile::StridedViewType>(
                 viewTy)) {
      if (auto *viewOp = view.getDefiningOp()) {
        srcTensorView = viewOp->getOperand(0);
        // Trace through unrealized_conversion_cast to get the actual tensor
        // view
        srcTensorView = traceUnrealizedConversionCast(srcTensorView);
      } else {
        return rewriter.notifyMatchFailure(
            op, "tensor view for DescriptorGatherOp not found");
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "expect a tensor view, partition view, or strided view type for "
              "DescriptorGatherOp");
    }

    // Verify we have a valid TensorViewType
    if (!isa<cuda_tile::TensorViewType>(srcTensorView.getType())) {
      return rewriter.notifyMatchFailure(
          op,
          "srcTensorView is not a TensorViewType after tracing conversions");
    }

    // 3. Get gather parameters
    Value xOffsets = adaptor.getXOffsets(); // 1D tile of indices
    Value yOffset = adaptor.getYOffset();   // scalar offset

    // 4. Get result type information
    auto reType = op.getResult().getType();
    auto reTileShape = reType.getShape(); // [BLOCK_X, BLOCK_Y]
    auto reElemTy = reType.getElementType();
    auto reTileTy = cuda_tile::TileType::get(ctx, reTileShape, reElemTy);

    // 5. Create GatherScatterView
    // sparse_dim = 0 (gather along first dimension, which is the row dimension)
    int64_t sparseDim = 0;

    // Preserve the source descriptor's padding for gathered reads as well.
    auto paddingValueAttr =
        cuda_tile::PaddingValueAttr::get(ctx, cuda_tile::PaddingValue::zero);
    if (auto stridedTy = dyn_cast<cuda_tile::StridedViewType>(viewTy))
      paddingValueAttr = stridedTy.getPaddingValue();
    else if (auto partitionTy = dyn_cast<cuda_tile::PartitionViewType>(viewTy))
      paddingValueAttr = partitionTy.getPaddingValue();

    // Convert int64_t shape to int32_t for DenseI32ArrayAttr
    SmallVector<int32_t> tileShapeI32;
    for (auto dim : reTileShape)
      tileShapeI32.push_back(static_cast<int32_t>(dim));

    auto gsViewType = cuda_tile::GatherScatterViewType::get(
        ctx,
        /*tile_shape=*/rewriter.getDenseI32ArrayAttr(tileShapeI32),
        /*tensor_view=*/
        cast<cuda_tile::TensorViewType>(srcTensorView.getType()),
        /*sparse_dim=*/sparseDim,
        /*padding_value=*/paddingValueAttr);

    auto gsViewOp = cuda_tile::MakeGatherScatterViewOp::create(
        rewriter, loc, gsViewType, srcTensorView);

    // 6. Build coordinates for load_view_tko
    // For gather: first coord is the 1D index tensor (xOffsets),
    //             second coord is the scalar offset (yOffset)
    SmallVector<Value> coords = {xOffsets, yOffset};

    // 7. Get optimization hints
    auto optHint = mlir::triton::utils::convertNumStagesToOptHint(
        op, ctx, numStagesMap, computeCapability, numStages);

    // 8. Use load_view_tko to perform the gather load
    auto memOrder = cuda_tile::MemoryOrderingSemanticsAttr::get(
        ctx, cuda_tile::MemoryOrderingSemantics::WEAK);

    auto loadOp = cuda_tile::LoadViewTkoOp::create(
        rewriter, loc, reTileTy, cuda_tile::TokenType::get(ctx),
        /*memory_ordering_semantics=*/memOrder,
        /*scope=*/nullptr, gsViewOp.getResult(), coords,
        /*token=*/nullptr, optHint.value_or(nullptr));

    // 9. Replace the original op with the loaded tile
    rewriter.replaceOp(op,
                       roundHostDescriptorRead(rewriter, op, loadOp.getTile()));
    return success();
  }
};

class ConvertDescriptorScatterOp
    : public OpConversionPattern<triton::DescriptorScatterOp> {
public:
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  const DenseMap<Operation *, int> &numStagesMap;
  int computeCapability;
  std::optional<int> numStages;

  ConvertDescriptorScatterOp(TypeConverter &typeConverter, MLIRContext *context,
                             DenseMap<Operation *, int> &numStagesMap,
                             int computeCapability,
                             std::optional<int> numStages)
      : OpConversionPattern<triton::DescriptorScatterOp>(typeConverter,
                                                         context),
        numStagesMap(numStagesMap), computeCapability(computeCapability),
        numStages(numStages) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op->getContext();

    // 1. Get the descriptor and trace through conversion casts
    Value view = adaptor.getDesc();
    view = traceUnrealizedConversionCast(view);
    Type viewTy = view.getType();

    // 2. Extract the underlying tensor view
    Value dstTensorView;

    // If view is already a TensorViewType, use it directly
    if (isa<cuda_tile::TensorViewType>(viewTy)) {
      dstTensorView = view;
    }
    // If view is a PartitionViewType or StridedViewType, extract the tensor
    // view
    else if (isa<cuda_tile::PartitionViewType, cuda_tile::StridedViewType>(
                 viewTy)) {
      if (auto *viewOp = view.getDefiningOp()) {
        dstTensorView = viewOp->getOperand(0);
        // Trace through unrealized_conversion_cast to get the actual tensor
        // view
        dstTensorView = traceUnrealizedConversionCast(dstTensorView);
      } else {
        return rewriter.notifyMatchFailure(
            op, "tensor view for DescriptorScatterOp not found");
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "expect a tensor view, partition view, or strided view type for "
              "DescriptorScatterOp");
    }

    // Verify we have a valid TensorViewType
    if (!isa<cuda_tile::TensorViewType>(dstTensorView.getType())) {
      return rewriter.notifyMatchFailure(
          op,
          "dstTensorView is not a TensorViewType after tracing conversions");
    }

    // 3. Get scatter parameters
    Value srcTile = adaptor.getSrc();       // tile to scatter
    Value xOffsets = adaptor.getXOffsets(); // 1D tile of indices
    Value yOffset = adaptor.getYOffset();   // scalar offset

    // 4. Get source tile type information
    auto srcTileType = cast<cuda_tile::TileType>(srcTile.getType());
    auto srcTileShape = srcTileType.getShape(); // [BLOCK_X, BLOCK_Y]
    auto srcElemTy = srcTileType.getElementType();

    // 5. Create GatherScatterView
    // sparse_dim = 0 (scatter along first dimension, which is the row
    // dimension)
    int64_t sparseDim = 0;

    // Convert int64_t shape to int32_t for DenseI32ArrayAttr
    SmallVector<int32_t> tileShapeI32;
    for (auto dim : srcTileShape)
      tileShapeI32.push_back(static_cast<int32_t>(dim));

    auto gsViewType = cuda_tile::GatherScatterViewType::get(
        ctx,
        /*tile_shape=*/rewriter.getDenseI32ArrayAttr(tileShapeI32),
        /*tensor_view=*/
        cast<cuda_tile::TensorViewType>(dstTensorView.getType()),
        /*sparse_dim=*/sparseDim,
        /*padding_value=*/nullptr);

    auto gsViewOp = cuda_tile::MakeGatherScatterViewOp::create(
        rewriter, loc, gsViewType, dstTensorView);

    // 6. Build coordinates for store_view_tko
    // For scatter: first coord is the 1D index tensor (xOffsets),
    //              second coord is the scalar offset (yOffset)
    SmallVector<Value> coords = {xOffsets, yOffset};

    // 7. Get optimization hints
    auto optHint = mlir::triton::utils::convertNumStagesToOptHint(
        op, ctx, numStagesMap, computeCapability, numStages);

    // 8. Use store_view_tko to perform the scatter store
    auto memOrder = cuda_tile::MemoryOrderingSemanticsAttr::get(
        ctx, cuda_tile::MemoryOrderingSemantics::WEAK);

    auto storeOp = cuda_tile::StoreViewTkoOp::create(
        rewriter, loc, cuda_tile::TokenType::get(ctx),
        /*memory_ordering_semantics=*/memOrder,
        /*scope=*/nullptr, srcTile, gsViewOp.getResult(), coords,
        /*token=*/nullptr, optHint.value_or(nullptr));

    // 9. Erase the original op
    rewriter.eraseOp(op);
    return success();
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
    // Bypass framework-inserted unrealized_conversion_cast so we read the
    // padding_value produced by ConvertMakeTensorDescOp (which reflects the
    // triton-level padding option) rather than the type-converter default.
    view = traceUnrealizedConversionCast(view);

    auto optHint = mlir::triton::utils::convertNumStagesToOptHint(
        op, ctx, numStagesMap, computeCapability, numStages);

    auto originalIndices = adaptor.getIndices();

    SmallVector<Value> indices;
    SmallVector<int64_t> viewShapeVec;
    Type viewTy = view.getType();

    // Handle both PartitionViewType and StridedViewType
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
    } else if (auto stridedTy = dyn_cast<cuda_tile::StridedViewType>(viewTy)) {
      // For StridedViewType with traversal stride = 1, use indices directly
      // Add assume div_by for TMA alignment requirement on leading dimension.
      // TMA requires 16-byte alignment, convert to element count.
      // Note: tl.make_tensor_descriptor enforces strides[-1] == 1 (row-major),
      // so leading dim is always the last dim.
      // See:
      // triton/python/triton/language/semantic.py::make_tensor_descriptor()
      auto elemTy = op.getResult().getType().getElementType();
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      unsigned alignElements = kTMAAlignment / elemBytes;
      for (size_t i = 0; i < originalIndices.size(); i++) {
        Value idx = originalIndices[i];
        // TMA requires the leading dimension coordinate (last dim for
        // row-major) to be 16-byte aligned (alignElements in element count)
        if (i == originalIndices.size() - 1) {
          idx = cuda_tile::AssumeOp::create(
                    rewriter, op.getLoc(), idx,
                    cuda_tile::DivByAttr::get(ctx, alignElements, std::nullopt,
                                              std::nullopt))
                    .getResult();
        }
        indices.push_back(idx);
      }
      auto viewTileFromStridedView =
          cast<cuda_tile::TileType>(stridedTy.getViewTileType());
      viewShapeVec.assign(viewTileFromStridedView.getShape().begin(),
                          viewTileFromStridedView.getShape().end());
    } else {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "expect a partition view type or strided view type");
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

    Value loaded = roundHostDescriptorRead(rewriter, op, LoadViewOp.getTile());
    if (viewShapeVec.size() != tileShape.size()) {
      auto tileTy = cuda_tile::TileType::get(ctx, tileShape, elemTy);
      auto reshapeOp =
          cuda_tile::ReshapeOp::create(rewriter, op.getLoc(), tileTy, loaded);
      rewriter.replaceOp(op, reshapeOp.getResult());
      return success();
    }

    rewriter.replaceOp(op, loaded);
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
    // Bypass framework-inserted unrealized_conversion_cast so we read the
    // padding_value produced by ConvertMakeTensorDescOp (which reflects the
    // triton-level padding option) rather than the type-converter default.
    view = traceUnrealizedConversionCast(view);

    auto originalIndices = adaptor.getIndices();
    SmallVector<Value> indices;
    Type viewTy = view.getType();

    // Handle both PartitionViewType and StridedViewType
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
    } else if (dyn_cast<cuda_tile::StridedViewType>(viewTy)) {
      // For StridedViewType with traversal stride = 1, use indices directly
      // Add assume div_by for TMA alignment requirement on leading dimension.
      // TMA requires 16-byte alignment, convert to element count.
      // Note: tl.make_tensor_descriptor enforces strides[-1] == 1 (row-major),
      // so leading dim is always the last dim.
      // See:
      // triton/python/triton/language/semantic.py::make_tensor_descriptor()
      auto srcTileType = cast<cuda_tile::TileType>(adaptor.getSrc().getType());
      auto elemTy = srcTileType.getElementType();
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      unsigned alignElements = kTMAAlignment / elemBytes;
      for (size_t i = 0; i < originalIndices.size(); i++) {
        Value idx = originalIndices[i];
        // TMA requires the leading dimension coordinate (last dim for
        // row-major) to be 16-byte aligned (alignElements in element count)
        if (i == originalIndices.size() - 1) {
          idx = cuda_tile::AssumeOp::create(
                    rewriter, op.getLoc(), idx,
                    cuda_tile::DivByAttr::get(ctx, alignElements, std::nullopt,
                                              std::nullopt))
                    .getResult();
        }
        indices.push_back(idx);
      }
    } else {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "expect a partition view type or strided view type");
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

// Descriptor reductions do not read a padded tile or return old values. Rebuild
// just this write consumer's view without load padding, preserving its backing
// tensor and element coordinates; keep the descriptor's read view unchanged.
class ConvertDescriptorReduceOp
    : public OpConversionPattern<triton::DescriptorReduceOp> {
public:
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto originalElemTy =
        op.getDesc().getType().getBlockType().getElementType();
    auto src = adaptor.getSrc();
    auto srcTy = cast<cuda_tile::TileType>(src.getType());
    auto elemTy = srcTy.getElementType();
    bool floating = isa<FloatType>(elemTy);
    if (floating ? !(elemTy.isF16() || elemTy.isBF16() || elemTy.isF32() ||
                     elemTy.isF64())
                 : !(elemTy.isInteger(32) || elemTy.isInteger(64)))
      return rewriter.notifyMatchFailure(
          op, "unsupported descriptor reduction type");
    if (floating && op.getKind() != triton::DescriptorReduceKind::ADD)
      return rewriter.notifyMatchFailure(
          op, "public TileIR descriptor floating-point reductions support add "
              "only");

    bool isUnsigned = originalElemTy.isUnsignedInteger();
    cuda_tile::AtomicRMWMode mode;
    switch (op.getKind()) {
    case triton::DescriptorReduceKind::ADD:
      mode = floating ? cuda_tile::AtomicRMWMode::ADDF
                      : cuda_tile::AtomicRMWMode::ADD;
      break;
    case triton::DescriptorReduceKind::MIN:
      mode = isUnsigned ? cuda_tile::AtomicRMWMode::UMIN
                        : cuda_tile::AtomicRMWMode::MIN;
      break;
    case triton::DescriptorReduceKind::MAX:
      mode = isUnsigned ? cuda_tile::AtomicRMWMode::UMAX
                        : cuda_tile::AtomicRMWMode::MAX;
      break;
    case triton::DescriptorReduceKind::AND:
      mode = cuda_tile::AtomicRMWMode::AND;
      break;
    case triton::DescriptorReduceKind::OR:
      mode = cuda_tile::AtomicRMWMode::OR;
      break;
    case triton::DescriptorReduceKind::XOR:
      mode = cuda_tile::AtomicRMWMode::XOR;
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "unsupported descriptor reduction kind");
    }

    Value view = traceUnrealizedConversionCast(adaptor.getDesc());
    Value tensorView;
    cuda_tile::TensorViewType tensorViewTy;
    SmallVector<int32_t> dimMap;
    cuda_tile::TileType viewTileTy;
    if (auto ty = dyn_cast<cuda_tile::StridedViewType>(view.getType())) {
      if (llvm::any_of(ty.getTraversalStrides().asArrayRef(),
                       [](int32_t v) { return v != 1; }))
        return rewriter.notifyMatchFailure(
            op, "descriptor traversal strides must be one");
      auto make = view.getDefiningOp<cuda_tile::MakeStridedViewOp>();
      if (!make)
        return rewriter.notifyMatchFailure(
            op, "descriptor backing tensor view is unavailable");
      tensorView = make.getTensorView();
      tensorViewTy = ty.getTensorView();
      dimMap.assign(ty.getDimMap().begin(), ty.getDimMap().end());
      viewTileTy = cast<cuda_tile::TileType>(ty.getViewTileType());
    } else if (auto ty =
                   dyn_cast<cuda_tile::PartitionViewType>(view.getType())) {
      auto make = view.getDefiningOp<cuda_tile::MakePartitionViewOp>();
      if (!make)
        return rewriter.notifyMatchFailure(
            op, "descriptor backing tensor view is unavailable");
      tensorView = make.getTensorView();
      tensorViewTy = ty.getTensorView();
      dimMap.assign(ty.getDimMap().begin(), ty.getDimMap().end());
      viewTileTy = cast<cuda_tile::TileType>(ty.getViewTileType());
    } else {
      return rewriter.notifyMatchFailure(
          op, "expected a strided or partition descriptor view");
    }
    if (viewTileTy != srcTy || adaptor.getIndices().size() != srcTy.getRank())
      return rewriter.notifyMatchFailure(
          op, "descriptor reduction view/value shape or type mismatch");

    SmallVector<int32_t> tileShape(srcTy.getShape().begin(),
                                   srcTy.getShape().end());
    SmallVector<int32_t> traversal(tileShape.size(), 1);
    // Triton offsets are element coordinates. A unit-traversal strided view
    // also represents partition-backed descriptors without lossy division by
    // block size or additional divisibility assumptions.
    auto writeViewTy = cuda_tile::StridedViewType::get(
        ctx, rewriter.getDenseI32ArrayAttr(tileShape),
        rewriter.getDenseI32ArrayAttr(traversal), tensorViewTy, dimMap,
        /*padding_value=*/nullptr);
    Value writeView = cuda_tile::MakeStridedViewOp::create(
        rewriter, loc, writeViewTy, tensorView);
    SmallVector<Value> indices(adaptor.getIndices());
    unsigned alignElements =
        kTMAAlignment / (elemTy.getIntOrFloatBitWidth() / 8);
    indices.back() = cuda_tile::AssumeOp::create(
        rewriter, loc, indices.back(),
        cuda_tile::DivByAttr::get(ctx, alignElements, std::nullopt,
                                  std::nullopt));
    cuda_tile::AtomicRedViewTkoOp::create(
        rewriter, loc, cuda_tile::TokenType::get(ctx),
        cuda_tile::MemoryOrderingSemantics::RELAXED,
        cuda_tile::MemoryScope::DEVICE, writeView, indices, mode, src,
        /*token=*/nullptr);
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
    // libdevice's explicit-rounding arithmetic entry points encode the
    // IEEE-754 rounding mode in the symbol suffix. Each entry maps bit-exactly
    // onto the corresponding native op with a rounding_mode attribute; the
    // f/d name prefix only selects f32/f64, which the operand types already
    // carry. Unrecognized bases with a rounding suffix (e.g. conversions like
    // __nv_double2float_rd) fall through to the symbol chain below.
    std::optional<cuda_tile::RoundingMode> suffixMode;
    StringRef base = symbol;
    if (base.consume_front("__nv_")) {
      // Parse a copy so short or unrecognized names retain their original
      // symbol when falling through to the external-call lowering.
      auto [opName, rounding] = base.rsplit('_');
      base = opName;
      if (rounding == "rn")
        suffixMode = cuda_tile::RoundingMode::NEAREST_EVEN;
      else if (rounding == "rz")
        suffixMode = cuda_tile::RoundingMode::ZERO;
      else if (rounding == "rd")
        suffixMode = cuda_tile::RoundingMode::NEGATIVE_INF;
      else if (rounding == "ru")
        suffixMode = cuda_tile::RoundingMode::POSITIVE_INF;
    }
    if (suffixMode) {
      auto rm =
          cuda_tile::RoundingModeAttr::get(rewriter.getContext(), *suffixMode);
      auto srcs = adaptor.getSrcs();
      if (base == "fadd" || base == "dadd") {
        rewriter.replaceOpWithNewOp<cuda_tile::AddFOp>(
            op, srcs[0], srcs[1], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "fsub" || base == "dsub") {
        rewriter.replaceOpWithNewOp<cuda_tile::SubFOp>(
            op, srcs[0], srcs[1], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "fmul" || base == "dmul") {
        rewriter.replaceOpWithNewOp<cuda_tile::MulFOp>(
            op, srcs[0], srcs[1], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "fdiv" || base == "ddiv") {
        rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(
            op, srcs[0], srcs[1], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "fsqrt" || base == "dsqrt") {
        rewriter.replaceOpWithNewOp<cuda_tile::SqrtOp>(
            op, srcs[0], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "frcp" || base == "drcp") {
        // rcp(x) = 1 / x with the requested rounding.
        Value one = splatFloat(getTileType(srcs[0]), 1.0);
        rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(
            op, one, srcs[0], rm, /*flush_to_zero=*/nullptr);
        return success();
      }
      if (base == "fmaf" || base == "fma") {
        rewriter.replaceOpWithNewOp<cuda_tile::FmaOp>(
            op, srcs[0], srcs[1], srcs[2], rm,
            /*flush_to_zero=*/nullptr);
        return success();
      }
    }

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
    } else if (symbol == "__nv_atan2f" || symbol == "__nv_atan2") {
      // Both libdevice and cuda_tile use the conventional atan2(y, x) order.
      rewriter.replaceOpWithNewOp<cuda_tile::Atan2Op>(op, adaptor.getSrcs()[0],
                                                      adaptor.getSrcs()[1]);
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
    } else if (symbol == "__nv_float_as_int" ||
               symbol == "__nv_float_as_uint" ||
               symbol == "__nv_int_as_float" ||
               symbol == "__nv_uint_as_float" ||
               symbol == "__nv_double_as_longlong" ||
               symbol == "__nv_longlong_as_double") {
      // Bit-preserving reinterpretation.
      auto resType = getTypeConverter()->convertType(op.getResult().getType());
      rewriter.replaceOpWithNewOp<cuda_tile::BitcastOp>(op, resType,
                                                        adaptor.getSrcs()[0]);
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
    } else if (symbol == "__nv_fabsf" || symbol == "__nv_fabs") {
      rewriter.replaceOpWithNewOp<cuda_tile::AbsFOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_abs" || symbol == "__nv_llabs") {
      rewriter.replaceOpWithNewOp<cuda_tile::AbsIOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_pow" || symbol == "__nv_powf") {
      rewriter.replaceOpWithNewOp<cuda_tile::FPowFOp>(op, adaptor.getSrcs()[0],
                                                      adaptor.getSrcs()[1]);
      return success();
    } else if (symbol == "__nv_fmod" || symbol == "__nv_fmodf") {
      rewriter.replaceOpWithNewOp<cuda_tile::RemFOp>(op, adaptor.getSrcs()[0],
                                                     adaptor.getSrcs()[1]);
      return success();
    } else if (symbol == "__nv_fmaf" || symbol == "__nv_fma") {
      rewriter.replaceOpWithNewOp<cuda_tile::FmaOp>(
          op, adaptor.getSrcs()[0], adaptor.getSrcs()[1], adaptor.getSrcs()[2],
          rmNearestEven, /*flush_to_zero=*/nullptr);
      return success();
    } else if (symbol == "__nv_fast_fdividef") {
      // Fast approximate f32 division.
      rewriter.replaceOpWithNewOp<cuda_tile::DivFOp>(
          op, adaptor.getSrcs()[0], adaptor.getSrcs()[1],
          cuda_tile::RoundingModeAttr::get(rewriter.getContext(),
                                           cuda_tile::RoundingMode::APPROX),
          /*flush_to_zero=*/nullptr);
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
    } else if (symbol == "__nv_cosh" || symbol == "__nv_coshf") {
      rewriter.replaceOpWithNewOp<cuda_tile::CosHOp>(op, adaptor.getSrcs()[0]);
      return success();
    } else if (symbol == "__nv_sinh" || symbol == "__nv_sinhf") {
      rewriter.replaceOpWithNewOp<cuda_tile::SinHOp>(op, adaptor.getSrcs()[0]);
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
    } else if (symbol == "__nv_logf" || symbol == "__nv_log") {
      rewriter.replaceOpWithNewOp<cuda_tile::LogOp>(op, adaptor.getSrcs()[0]);
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

    auto descType = cast<triton::TensorDescType>(op.getResult().getType());
    auto tileShape = descType.getBlockType().getShape();
    SmallVector<int32_t> arrayOfi32Shape;
    SmallVector<int32_t> traversalStrides;
    for (auto i64Shape : tileShape) {
      arrayOfi32Shape.push_back(static_cast<int32_t>(i64Shape));
      traversalStrides.push_back(1); // Set traversal stride to 1
    }

    cuda_tile::PaddingValue paddingValue;
    switch (op.getPadding()) {
    case triton::PaddingOption::PAD_ZERO:
      paddingValue = cuda_tile::PaddingValue::zero;
      break;
    case triton::PaddingOption::PAD_NAN:
      paddingValue = cuda_tile::PaddingValue::nan;
      break;
    }

    auto tileStridedViewTy = cuda_tile::StridedViewType::get(
        ctx, rewriter.getDenseI32ArrayAttr(arrayOfi32Shape),
        rewriter.getDenseI32ArrayAttr(traversalStrides), tensorViewTy, dimMap,
        cuda_tile::PaddingValueAttr::get(ctx, paddingValue));

    auto stridedViewOp = cuda_tile::MakeStridedViewOp::create(
        rewriter, loc, tileStridedViewTy, makeTensorViewOp);

    rewriter.replaceOp(op, stridedViewOp);
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
      bool fastAcc = op.getMaxNumImpreciseAcc() > 0;
      auto mma = rewriter.replaceOpWithNewOp<cuda_tile::MmaFOp>(
          op, adaptor.getA(), adaptor.getB(), adaptor.getC());
      if (fastAcc)
        mma.setFastAccAttr(rewriter.getUnitAttr());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported operand types of mma op");
    }
    return success();
  }
};

class ConvertDotScaledOp : public OpConversionPattern<triton::DotScaledOp> {
  using OpConversionPattern::OpConversionPattern;

  // Check if element type argument equals with tile element type.
  bool checkInputElemType(Type type, triton::ScaleDotElemType elemType) const {
    switch (elemType) {
    case triton::ScaleDotElemType::E4M3:
      return isa<Float8E4M3FNType>(type);
    case triton::ScaleDotElemType::E5M2:
      return isa<Float8E5M2Type>(type);
    case triton::ScaleDotElemType::E2M1:
      return type.isInteger(8);
    case triton::ScaleDotElemType::BF16:
      return isa<BFloat16Type>(type);
    case triton::ScaleDotElemType::FP16:
      return isa<Float16Type>(type);
    default:
      return false;
    }
  }

  bool isF4Type(triton::ScaleDotElemType elemType) const {
    return elemType == triton::ScaleDotElemType::E2M1;
  }

  bool isF8Type(triton::ScaleDotElemType elemType) const {
    return elemType == triton::ScaleDotElemType::E4M3 ||
           elemType == triton::ScaleDotElemType::E5M2;
  }

  // Swap the last two dimensions of a tile.
  // e.g., [M, K] -> [K, M] or [B, M, K] -> [B, K, M].
  Value permuteLastTwoDims(Value value, ConversionPatternRewriter &rewriter,
                           Location loc) const {
    auto tileType = dyn_cast<cuda_tile::TileType>(value.getType());
    if (!tileType)
      return value;

    auto valueShape = tileType.getShape();
    auto elemType = tileType.getElementType();
    int rank = valueShape.size();
    assert(rank >= 2 && "tile rank must be at least 2");

    // Permute order of dimensions.
    SmallVector<int32_t> permuteDims;
    for (int i = 0; i < rank - 2; ++i)
      permuteDims.push_back(i);
    permuteDims.push_back(rank - 1);
    permuteDims.push_back(rank - 2);

    // Swap the last 2 dimensions of the input tile with permute op.
    auto permuteAttr = rewriter.getDenseI32ArrayAttr(permuteDims);
    return cuda_tile::PermuteOp::create(rewriter, loc, value, permuteAttr)
        .getResult();
  }

  // TTIR uses i8 to represent packed f4 values,
  // use reshape + unpack + reshape to convert it to fp4 tile.
  // Only one dimension (packDim) can have a different size.
  // e.g., convert tile<128x64xi8> to tile<128x128xf4E2M1FN> for packDim = 1.
  // Add permute ops if necessary as reshape expects logically row-major input.
  Value convertI8ToF4Tile(Value i8Value, triton::ScaleDotElemType elemType,
                          ConversionPatternRewriter &rewriter, Location loc,
                          int packDim) const {
    if (!isF4Type(elemType))
      return i8Value;
    auto i8TileType = dyn_cast<cuda_tile::TileType>(i8Value.getType());
    if (!i8TileType)
      return i8Value;

    // Check if the element type is i8, which indicates packed fp4 values.
    if (!i8TileType.getElementType().isInteger(8))
      return i8Value;
    assert(packDim >= 0 && packDim < static_cast<int>(i8TileType.getRank()) &&
           "packDim must be within the range of i8Shape");

    // Permute the tile before and after the reshape ops,
    // if the packDim is not the last dimension.
    int tileRank = i8TileType.getRank();
    bool isLastDimPacked = (packDim == tileRank - 1);
    if (!isLastDimPacked) {
      i8Value = permuteLastTwoDims(i8Value, rewriter, loc);
      // update packDim and i8TileType after permute.
      packDim = tileRank - 1;
      i8TileType = dyn_cast<cuda_tile::TileType>(i8Value.getType());
    }

    // Reshape the tile to 1D for unpacking.
    int tileSizeInBytes = i8TileType.getNumElements();
    Type i8ElemType = i8TileType.getElementType();
    auto reshapeType = cuda_tile::TileType::get({tileSizeInBytes}, i8ElemType);
    auto i8Reshape =
        cuda_tile::ReshapeOp::create(rewriter, loc, reshapeType, i8Value);

    auto ctx = rewriter.getContext();
    Type f4ElemType;
    if (elemType == triton::ScaleDotElemType::E2M1)
      f4ElemType = Float4E2M1FNType::get(ctx);
    else
      return i8Value;

    // Unpack the i8 tile to f4 tile.
    int f4TileSize = tileSizeInBytes * 2;
    auto unpackType = cuda_tile::TileType::get({f4TileSize}, f4ElemType);
    auto f4Unpack =
        cuda_tile::UnpackOp::create(rewriter, loc, unpackType, i8Reshape);

    // Reshape the tile back to ND shape after unpacking.
    auto i8Shape = i8TileType.getShape();
    SmallVector<int64_t> f4FinalShape(i8Shape.begin(), i8Shape.end());
    f4FinalShape[packDim] *= 2;
    auto f4TileType = cuda_tile::TileType::get(f4FinalShape, f4ElemType);
    Value f4Reshape =
        cuda_tile::ReshapeOp::create(rewriter, loc, f4TileType, f4Unpack)
            .getResult();

    if (!isLastDimPacked)
      f4Reshape = permuteLastTwoDims(f4Reshape, rewriter, loc);
    return f4Reshape;
  }

  // Convert Triton scale type to TileIR scale type.
  Value convertTritonScaleType(Value scaleValue,
                               ConversionPatternRewriter &rewriter,
                               Location loc) const {
    auto scaleTileType = dyn_cast<cuda_tile::TileType>(scaleValue.getType());
    if (!scaleTileType)
      return scaleValue;

    // If element type is i8, convert it to f8e8m0 tile.
    if (scaleTileType.getElementType().isInteger(8)) {
      auto ctx = rewriter.getContext();
      Type f8e8m0ElemType = Float8E8M0FNUType::get(ctx);
      Type f8e8m0TileType =
          cuda_tile::TileType::get(scaleTileType.getShape(), f8e8m0ElemType);

      // Convert i8 tile scale to f8e8m0 tile using bitcast.
      return cuda_tile::BitcastOp::create(rewriter, loc, f8e8m0TileType,
                                          scaleValue);
    }
    // Otherwise, return the original scale.
    return scaleValue;
  }

  // Check if the scale is TileType with the given rank.
  bool checkScaleTileType(Value scale, int rank) const {
    auto scaleTileType = dyn_cast<cuda_tile::TileType>(scale.getType());
    if (!scaleTileType)
      return false;
    int scaleRank = scaleTileType.getRank();
    return scaleRank == rank;
  }

  // Both scale values are explicit, including a native unit scale if needed.
  void convertToMmaFScaled(triton::DotScaledOp op, OpAdaptor adaptor, Value sfa,
                           Value sfb, ConversionPatternRewriter &rewriter,
                           SmallVector<Type> &retTypes) const {
    auto loc = op.getLoc();
    auto aElemType = op.getAElemType();
    auto bElemType = op.getBElemType();

    auto sfaTileType = dyn_cast<cuda_tile::TileType>(sfa.getType());
    auto sfbTileType = dyn_cast<cuda_tile::TileType>(sfb.getType());
    assert(sfaTileType && sfbTileType &&
           "scale operands sfa and sfb must be TileType");

    int sfaRank = sfaTileType.getRank();
    int sfbRank = sfbTileType.getRank();
    assert(sfaRank >= 2 && sfbRank >= 2 &&
           "scale operands sfa and sfb must be at least 2D");

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    auto aType = dyn_cast<cuda_tile::TileType>(a.getType());
    auto bType = dyn_cast<cuda_tile::TileType>(b.getType());
    int aRank = aType.getRank();
    int bRank = bType.getRank();

    bool lhsKPack = op.getLhsKPack();
    bool rhsKPack = op.getRhsKPack();
    // Note: lhsKPack and rhsKPack can be different for mixed precision (e.g.,
    // fp8 x fp4). For fp8, k_pack must be true. For fp4, k_pack can be true or
    // false.

    // Convert i8 tile to f4 tile if element type is f4,
    // with a double size of packing dimension in result tile.
    if (isF4Type(aElemType)) {
      int lhsPackDim = lhsKPack ? aRank - 1 : aRank - 2;
      a = convertI8ToF4Tile(a, aElemType, rewriter, loc, lhsPackDim);
    }

    if (isF4Type(bElemType)) {
      int rhsPackDim = rhsKPack ? bRank - 2 : bRank - 1;
      b = convertI8ToF4Tile(b, bElemType, rewriter, loc, rhsPackDim);
    }

    // Convert scale type from i8 to f8e8m0 if they are i8 tile.
    // TTIR does not define f8e8m0 type natively and uses i8 type instead,
    // but TileIR dialect exposes a f8e8m0 type.
    //
    // Supported scale factor types:
    // ┌─────────────┬────────────────────────┬──────────────────────────┐
    // │ dtype       │ TTIR scale factor type │ TileIR scale factor type │
    // ├─────────────┼────────────────────────┼──────────────────────────┤
    // │ mxfp8/mxfp4 │ i8                     │ f8e8m0                   │
    // │ nvfp4       │ f8e4m3                 │ f8e4m3                   │
    // └─────────────┴────────────────────────┴──────────────────────────┘
    sfa = convertTritonScaleType(sfa, rewriter, loc);
    sfb = convertTritonScaleType(sfb, rewriter, loc);

    // Transpose the last 2 dimensions of sfb.
    // TTIR converts sfb in shape of [..., N, K//x], but TileIR wants K in
    // dim[-2].
    sfb = permuteLastTwoDims(sfb, rewriter, loc);

    rewriter.replaceOpWithNewOp<cuda_tile::MmaFScaledOp>(op, retTypes[0], a, b,
                                                         c, sfa, sfb);
  }

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = this->getTypeConverter();
    SmallVector<Type> retTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), retTypes)))
      return rewriter.notifyMatchFailure(
          op, "typeConversion for DotScaledOp failed");

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    auto aType = dyn_cast<cuda_tile::TileType>(a.getType());
    auto bType = dyn_cast<cuda_tile::TileType>(b.getType());
    if (!aType || !bType)
      return rewriter.notifyMatchFailure(op,
                                         "operands a and b must be TileType");

    int aRank = aType.getRank();
    int bRank = bType.getRank();
    if (aRank < 2 || bRank < 2)
      return rewriter.notifyMatchFailure(
          op, "operands a and b must be at least 2D");

    auto aElemType = op.getAElemType();
    auto bElemType = op.getBElemType();
    if (!checkInputElemType(aType.getElementType(), aElemType))
      return rewriter.notifyMatchFailure(
          op, "aElemType is not aligned with input element type");
    if (!checkInputElemType(bType.getElementType(), bElemType))
      return rewriter.notifyMatchFailure(
          op, "bElemType is not aligned with input element type");

    Value sfa = adaptor.getAScale();
    Value sfb = adaptor.getBScale();
    if (sfa && !checkScaleTileType(sfa, aRank))
      return rewriter.notifyMatchFailure(
          op, "sfa must be TileType with rank equal to aRank");
    if (sfb && !checkScaleTileType(sfb, bRank))
      return rewriter.notifyMatchFailure(
          op, "sfb must be TileType with rank equal to bRank");

    if (aElemType != bElemType)
      return rewriter.notifyMatchFailure(
          op,
          "public native scaled MMA requires matching operand element types");
    if (!isF4Type(aElemType) && !isF8Type(aElemType))
      return rewriter.notifyMatchFailure(
          op, "public native scaled MMA requires FP4 or FP8 operands");
    if (!sfa || !sfb) {
      // A missing scale means exactly one, not an input-type conversion.
      // Only same-format FP8 with an existing E8M0 byte scale is covered.
      if ((!sfa && !sfb) || !isF8Type(aElemType))
        return rewriter.notifyMatchFailure(
            op, "unit scale completion requires exactly one FP8 scale");
      auto existingScaleType =
          cast<cuda_tile::TileType>((sfa ? sfa : sfb).getType());
      if (!existingScaleType.getElementType().isInteger(8))
        return rewriter.notifyMatchFailure(
            op, "FP8 unit scale completion requires an E8M0 byte scale");
      if (aRank != bRank || (aRank != 2 && aRank != 3))
        return rewriter.notifyMatchFailure(
            op, "FP8 unit scale completion requires rank 2 or 3");
      int64_t k = aType.getShape().back();
      if (k <= 0 || k % 32 != 0 || k != bType.getShape()[bRank - 2])
        return rewriter.notifyMatchFailure(
            op,
            "FP8 unit scale completion requires matching K divisible by 32");

      // Triton scale shapes are [..., M, K/32] and [..., N, K/32].
      // The existing conversion transposes RHS scales for native mmaf_scaled.
      SmallVector<int64_t> scaleShape((sfa ? bType : aType).getShape());
      if (sfa)
        std::swap(scaleShape[aRank - 2], scaleShape.back());
      scaleShape.back() /= 32;
      auto unitType =
          cuda_tile::TileType::get(scaleShape, rewriter.getI8Type());
      // E8M0 has bias 127 and no sign/mantissa: byte 0x7f is exactly 2^0.
      // Bitcast below preserves that encoding; no floating multiply is added.
      Value unit = cuda_tile::ConstantOp::create(
          rewriter, op.getLoc(), unitType,
          DenseIntElementsAttr::get(unitType, ArrayRef<APInt>{APInt(8, 127)}));
      if (sfa)
        sfb = unit;
      else
        sfa = unit;
    }
    convertToMmaFScaled(op, adaptor, sfa, sfb, rewriter, retTypes);

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

class ConvertGatherOp : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<cuda_tile::TileType>(adaptor.getSrc().getType());
    auto indexType =
        dyn_cast<cuda_tile::TileType>(adaptor.getIndices().getType());
    auto resultType = dyn_cast_or_null<cuda_tile::TileType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!sourceType || !indexType || !resultType)
      return rewriter.notifyMatchFailure(op, "expected converted tile types");

    unsigned axis = op.getAxis();
    int64_t axisSize = sourceType.getShape()[axis];
    if (axisSize <= 0 || axisSize > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(op,
                                         "gather axis exceeds i32 indexing");

    Location loc = op.getLoc();
    Value source = adaptor.getSrc();
    auto originalResultType = resultType;
    // Transport floating-point values as bits so source bitwise expressions
    // cannot become floating-point arithmetic while moving gathered slices.
    // Public bitcast has no i4 tile operand, so retain sub-byte values
    // directly.
    if (auto floatType = dyn_cast<FloatType>(sourceType.getElementType());
        floatType && floatType.getWidth() >= 8) {
      Type bitsType = rewriter.getIntegerType(floatType.getWidth());
      sourceType = cuda_tile::TileType::get(sourceType.getShape(), bitsType);
      resultType = cuda_tile::TileType::get(resultType.getShape(), bitsType);
      source = cuda_tile::BitcastOp::create(rewriter, loc, sourceType, source);
    }
    auto restoreResult = [&](Value value) -> Value {
      if (resultType != originalResultType)
        return cuda_tile::BitcastOp::create(rewriter, loc, originalResultType,
                                            value);
      return value;
    };
    auto scalarI32 = cuda_tile::TileType::get({}, rewriter.getI32Type());
    auto constant = [&](int64_t value) -> Value {
      return cuda_tile::ConstantOp::create(
          rewriter, loc, scalarI32,
          DenseIntElementsAttr::get(scalarI32, {static_cast<int32_t>(value)}));
    };
    Value zero = constant(0);
    SmallVector<int64_t> sliceShape(sourceType.getShape());
    sliceShape[axis] = 1;
    auto sliceType =
        cuda_tile::TileType::get(sliceShape, sourceType.getElementType());
    SmallVector<Value> offsets(sourceType.getRank(), zero);
    Value first =
        cuda_tile::ExtractOp::create(rewriter, loc, sliceType, source, offsets);
    Value initial =
        cuda_tile::BroadcastOp::create(rewriter, loc, resultType, first);
    if (axisSize == 1) {
      rewriter.replaceOp(op, restoreResult(initial));
      return success();
    }

    // Compare indices at their original width, extending narrow types first.
    // Narrowing the induction variable could wrap and select a later slice.
    Value indices = adaptor.getIndices();
    unsigned indexWidth =
        cast<IntegerType>(indexType.getElementType()).getWidth();
    if (indexWidth < 32) {
      indexType =
          cuda_tile::TileType::get(indexType.getShape(), rewriter.getI32Type());
      indices = cuda_tile::ExtIOp::create(rewriter, loc, indexType, indices,
                                          cuda_tile::Signedness::Unsigned);
    }
    auto indexScalarType =
        cuda_tile::TileType::get({}, indexType.getElementType());
    auto indexSingletonType =
        cuda_tile::TileType::get(SmallVector<int64_t>(sourceType.getRank(), 1),
                                 indexType.getElementType());
    Value one = constant(1), end = constant(axisSize);

    // Select whole slices of the computed source tile. No source reload,
    // arithmetic on source values, or floating-point reduction is required.
    // A loop keeps the generated code bounded when the gather axis is large.
    auto loop = cuda_tile::LoopOp::create(rewriter, loc, TypeRange{resultType},
                                          ValueRange{one, initial});
    Block *body = rewriter.createBlock(
        &loop.getRegion(), {}, TypeRange{scalarI32, resultType}, {loc, loc});
    Value k = body->getArgument(0), accumulated = body->getArgument(1);
    Value stop = cuda_tile::CmpIOp::create(
        rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL, k, end,
        cuda_tile::Signedness::Unsigned);
    auto ifOp = cuda_tile::IfOp::create(rewriter, loc, TypeRange{}, stop);
    rewriter.createBlock(&ifOp.getThenRegion());
    cuda_tile::BreakOp::create(rewriter, loc, ValueRange{accumulated});
    rewriter.createBlock(&ifOp.getElseRegion());
    cuda_tile::YieldOp::create(rewriter, loc, ValueRange{});
    rewriter.setInsertionPointAfter(ifOp);

    Value indexK = k;
    if (indexWidth > 32)
      indexK = cuda_tile::ExtIOp::create(rewriter, loc, indexScalarType, k,
                                         cuda_tile::Signedness::Unsigned);
    Value shapedK =
        cuda_tile::ReshapeOp::create(rewriter, loc, indexSingletonType, indexK);
    Value broadcastK =
        cuda_tile::BroadcastOp::create(rewriter, loc, indexType, shapedK);
    Value selected = cuda_tile::CmpIOp::create(
        rewriter, loc, cuda_tile::ComparisonPredicate::EQUAL, indices,
        broadcastK, cuda_tile::Signedness::Unsigned);
    offsets[axis] = k;
    Value slice =
        cuda_tile::ExtractOp::create(rewriter, loc, sliceType, source, offsets);
    Value values =
        cuda_tile::BroadcastOp::create(rewriter, loc, resultType, slice);
    Value updated = cuda_tile::SelectOp::create(rewriter, loc, selected, values,
                                                accumulated);
    Value next = cuda_tile::AddIOp::create(rewriter, loc, k, one);
    cuda_tile::ContinueOp::create(rewriter, loc, ValueRange{next, updated});
    rewriter.setInsertionPointAfter(loop);
    rewriter.replaceOp(op, restoreResult(loop.getResult(0)));
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

// Only whitespace is normalized. Preserve boundaries between PTX word tokens,
// so e.g. "cvt .rn" cannot become "cvt.rn". Comments and extra instructions
// remain part of the string and cannot match one of the supported forms.
static std::string normalizeNativeInlineAsm(StringRef text) {
  auto isWord = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '$';
  };
  std::string result;
  bool space = false;
  for (char c : text) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
        c == '\v') {
      space = true;
      continue;
    }
    if (space && !result.empty() && isWord(result.back()) && isWord(c))
      result += ' ';
    result += c;
    space = false;
  }
  return result;
}

// These are the complete pure, pack=1 forms used by triton_kernels. Keep
// unsupported asm illegal; the public dialect has no generic inline-asm op.
class ConvertKnownNumericInlineAsmToNativeOp
    : public OpConversionPattern<triton::ElementwiseInlineAsmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getPure() || op.getPackedElement() != 1 || op.getNumResults() != 1)
      return failure();

    std::string text = normalizeNativeInlineAsm(op.getAsmString());
    std::string constraints = normalizeNativeInlineAsm(op.getConstraints());
    bool fp4 = text == "{.reg .b8 r;cvt.rn.satfinite.e2m1x2.f32 r,$1,$2;"
                       "mov.b32 $0,{r,r,r,r};}" &&
               constraints == "=r,f,f";
    bool exp2 = text == "ex2.approx.ftz.f32 $0,$1;" && constraints == "=r,r";
    bool tf32 = text == "cvt.rn.tf32.f32 $0,$1;" && constraints == "=r,r";
    bool xorsign = text == "{max.NaN.xorsign.abs.f32 $0,$1,$2;}" &&
                   constraints == "=r,r,r";
    if (!fp4 && !exp2 && !tf32 && !xorsign)
      return failure();
    if (adaptor.getArgs().size() != (fp4 || xorsign ? 2 : 1))
      return failure();

    auto resultTy = dyn_cast_or_null<cuda_tile::TileType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    if (!resultTy || (fp4 ? !resultTy.getElementType().isSignlessInteger(8)
                          : !resultTy.getElementType().isF32()))
      return failure();
    for (Value arg : adaptor.getArgs()) {
      auto argTy = dyn_cast<cuda_tile::TileType>(arg.getType());
      if (!argTy || !argTy.getElementType().isF32() ||
          argTy.getShape() != resultTy.getShape())
        return failure();
    }

    auto loc = op.getLoc();
    auto rn = cuda_tile::RoundingModeAttr::get(
        rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN);
    Value a = adaptor.getArgs()[0];
    if (exp2) {
      // PTX .ftz applies to both subnormal inputs and outputs. This modifier
      // must remain true even when the pass's general FTZ option is false.
      rewriter.replaceOpWithNewOp<cuda_tile::Exp2Op>(
          op, resultTy, a, /*flush_to_zero=*/rewriter.getUnitAttr());
      return success();
    }
    if (tf32) {
      auto tf32Ty = cuda_tile::TileType::get(
          resultTy.getShape(), FloatTF32Type::get(rewriter.getContext()));
      Value rounded = cuda_tile::FToFOp::create(rewriter, loc, tf32Ty, a, rn);
      // Preserve Inf/NaN as specified by native ftof. Unlike a host descriptor
      // read, PTX cvt does not promise to retain the source NaN payload bits.
      rewriter.replaceOpWithNewOp<cuda_tile::FToFOp>(op, resultTy, rounded, rn);
      return success();
    }
    Value b = adaptor.getArgs()[1];
    if (fp4) {
      int64_t count = resultTy.getNumElements();
      if (count <= 0 || count > std::numeric_limits<int64_t>::max() / 2)
        return failure();
      auto columnTy =
          cuda_tile::TileType::get({count, 1}, rewriter.getF32Type());
      Value lo = cuda_tile::ReshapeOp::create(rewriter, loc, columnTy, b);
      Value hi = cuda_tile::ReshapeOp::create(rewriter, loc, columnTy, a);
      auto pairTy = cuda_tile::TileType::get({count, 2}, rewriter.getF32Type());
      // PTX's second input is the low nibble; native pack puts the first
      // element in the low nibble. Interleave [b, a] for each output byte.
      Value pairs = cuda_tile::CatOp::create(rewriter, loc, pairTy, lo, hi,
                                             rewriter.getI64IntegerAttr(1));
      auto flatTy =
          cuda_tile::TileType::get({2 * count}, rewriter.getF32Type());
      Value flat = cuda_tile::ReshapeOp::create(rewriter, loc, flatTy, pairs);
      auto f4Ty = cuda_tile::TileType::get(
          {2 * count}, Float4E2M1FNType::get(rewriter.getContext()));
      // f4E2M1FN conversion natively saturates finite overflow and Inf to
      // signed max, and NaN to positive max. There is no separate ftof
      // saturation attribute in the public 13.4 dialect.
      Value f4 = cuda_tile::FToFOp::create(rewriter, loc, f4Ty, flat, rn);
      auto byteTy = cuda_tile::TileType::get({count}, rewriter.getI8Type());
      Value bytes = cuda_tile::PackOp::create(rewriter, loc, byteTy, f4);
      // mov.b32 repeats the packed byte, but the exact matched i8 result
      // observes only its low byte. An i32 result must not use this lowering.
      rewriter.replaceOpWithNewOp<cuda_tile::ReshapeOp>(op, resultTy, bytes);
      return success();
    }

    // The maximum is native floating-point maxf. Integer operations only
    // preserve PTX's exact sign-bit and canonical-NaN selection semantics.
    auto bitsTy =
        cuda_tile::TileType::get(resultTy.getShape(), rewriter.getI32Type());
    auto constant = [&](int64_t value) -> Value {
      return cuda_tile::ConstantOp::create(
          rewriter, loc, bitsTy,
          DenseIntElementsAttr::get(bitsTy, ArrayRef<APInt>{APInt(32, value)}));
    };
    Value signMask = constant(0x80000000);
    Value absMask = constant(0x7fffffff);
    Value inf = constant(0x7f800000);
    Value aBits = cuda_tile::BitcastOp::create(rewriter, loc, bitsTy, a);
    Value bBits = cuda_tile::BitcastOp::create(rewriter, loc, bitsTy, b);
    Value xorBits = cuda_tile::XOrIOp::create(rewriter, loc, aBits, bBits);
    Value sign = cuda_tile::AndIOp::create(rewriter, loc, xorBits, signMask);
    Value aAbsBits = cuda_tile::AndIOp::create(rewriter, loc, aBits, absMask);
    Value bAbsBits = cuda_tile::AndIOp::create(rewriter, loc, bBits, absMask);
    Value aAbs =
        cuda_tile::BitcastOp::create(rewriter, loc, resultTy, aAbsBits);
    Value bAbs =
        cuda_tile::BitcastOp::create(rewriter, loc, resultTy, bAbsBits);
    Value magnitude = cuda_tile::MaxFOp::create(
        rewriter, loc, resultTy, aAbs, bAbs,
        /*propagate_nan=*/rewriter.getUnitAttr(), /*flush_to_zero=*/nullptr);
    Value magnitudeBits =
        cuda_tile::BitcastOp::create(rewriter, loc, bitsTy, magnitude);
    Value absMagnitudeBits =
        cuda_tile::AndIOp::create(rewriter, loc, magnitudeBits, absMask);
    Value isNan = cuda_tile::CmpIOp::create(
        rewriter, loc, cuda_tile::ComparisonPredicate::GREATER_THAN,
        absMagnitudeBits, inf, cuda_tile::Signedness::Unsigned);
    Value signedBits =
        cuda_tile::OrIOp::create(rewriter, loc, magnitudeBits, sign);
    // Ignore xorsign when max returns NaN. Preserve native canonical NaN,
    // including its sign; signed zero otherwise comes from the input XOR.
    Value resultBits = cuda_tile::SelectOp::create(rewriter, loc, bitsTy, isNan,
                                                   magnitudeBits, signedBits);
    rewriter.replaceOpWithNewOp<cuda_tile::BitcastOp>(op, resultTy, resultBits);
    return success();
  }
};

// Match only the packed FP4-to-half conversion used by triton_kernels. The
// caller subsequently extracts the low/high halfwords from this i32 result.
class ConvertPackedFp4UpcastInlineAsmToNativeOp
    : public OpConversionPattern<triton::ElementwiseInlineAsmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getPure() || op.getPackedElement() != 1 ||
        op.getNumResults() != 1 || adaptor.getArgs().size() != 1 ||
        normalizeNativeInlineAsm(op.getConstraints()) != "=r,r" ||
        normalizeNativeInlineAsm(op.getAsmString()) !=
            "{.reg .b8 in_8;.reg .f16x2 out;cvt.u8.u32 in_8,$1;"
            "cvt.rn.f16x2.e2m1x2 out,in_8;mov.b32 $0,out;}")
      return failure();

    auto resultTy = dyn_cast_or_null<cuda_tile::TileType>(
        getTypeConverter()->convertType(op->getResult(0).getType()));
    Value input = adaptor.getArgs()[0];
    auto inputTy = dyn_cast<cuda_tile::TileType>(input.getType());
    if (!resultTy || !resultTy.getElementType().isSignlessInteger(32) ||
        !inputTy || !inputTy.getElementType().isSignlessInteger(8) ||
        inputTy.getShape() != resultTy.getShape())
      return failure();
    int64_t count = resultTy.getNumElements();
    if (count <= 0 || count > std::numeric_limits<int64_t>::max() / 4)
      return failure();

    auto loc = op.getLoc();
    auto byteTy = cuda_tile::TileType::get({count}, rewriter.getI8Type());
    Value bytes = cuda_tile::ReshapeOp::create(rewriter, loc, byteTy, input);
    auto f4Ty = cuda_tile::TileType::get(
        {2 * count}, Float4E2M1FNType::get(rewriter.getContext()));
    // Native unpack orders each pair [low nibble, high nibble]. All FP4
    // values, including both signed zeros, are exactly representable in f16.
    Value f4 = cuda_tile::UnpackOp::create(rewriter, loc, f4Ty, bytes);
    auto halfTy = cuda_tile::TileType::get({2 * count}, rewriter.getF16Type());
    auto rn = cuda_tile::RoundingModeAttr::get(
        rewriter.getContext(), cuda_tile::RoundingMode::NEAREST_EVEN);
    Value halves = cuda_tile::FToFOp::create(rewriter, loc, halfTy, f4, rn);

    // Pack accepts only an i8 result; unpack then reinterprets each adjacent
    // pair of halfwords as i32. It preserves low/high ordering and every bit,
    // matching cvt.f16x2 followed by mov.b32, without a numerical int cast.
    auto halfBytesTy =
        cuda_tile::TileType::get({4 * count}, rewriter.getI8Type());
    Value halfBytes =
        cuda_tile::PackOp::create(rewriter, loc, halfBytesTy, halves);
    auto wordTy = cuda_tile::TileType::get({count}, rewriter.getI32Type());
    Value words = cuda_tile::UnpackOp::create(rewriter, loc, wordTy, halfBytes);
    rewriter.replaceOpWithNewOp<cuda_tile::ReshapeOp>(op, resultTy, words);
    return success();
  }
};

static constexpr llvm::StringLiteral kGdcWaitHelperAsm =
    "griddepcontrol.wait; // dummy $0";
static constexpr llvm::StringLiteral kGdcLaunchDependentsHelperAsm =
    "griddepcontrol.launch_dependents; // dummy $0";

// Keep the frontend aligned with upstream Triton: GDC first appears as Triton
// inline asm rather than backend-specific ops. Recognize the GDC inline asm
// form that can be losslessly reconstructed as native cuda_tile GDC ops so
// AutoGenMemoryToken can reconstruct token ordering later.
// TODO: emit cuda_tile ops directly from tt.extra.cuda
bool isCanonicalizableGdcInlineAsmOp(triton::ElementwiseInlineAsmOp op,
                                     llvm::StringLiteral asmString) {
  if (op.getAsmString() != asmString || !op.getArgs().empty() ||
      op.getNumResults() != 1 || op.getConstraints() != "=r" ||
      op.getPackedElement() != 1 || op.getPure() ||
      !op->getResult(0).use_empty())
    return false;

  auto resultTy = dyn_cast<IntegerType>(op->getResult(0).getType());
  return resultTy && resultTy.isSignlessInteger(32);
}

/// Recognize GDC inline asm patterns and convert to native TKO ops.
/// GDC operations participate in the AutoGenMemoryToken token chain.
class ConvertGdcInlineAsmToNativeOp
    : public OpConversionPattern<triton::ElementwiseInlineAsmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tokenTy = cuda_tile::TokenType::get(rewriter.getContext());

    if (isCanonicalizableGdcInlineAsmOp(op, kGdcWaitHelperAsm)) {
      rewriter.replaceOpWithNewOp<cuda_tile::GdcWaitTkoOp>(op, tokenTy,
                                                           /*token=*/Value());
      return success();
    }
    if (isCanonicalizableGdcInlineAsmOp(op, kGdcLaunchDependentsHelperAsm)) {
      rewriter.replaceOpWithNewOp<cuda_tile::GdcLaunchDependentsTkoOp>(
          op, tokenTy, /*token=*/Value());
      return success();
    }
    return failure(); // Not the supported upstream GDC helper form.
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
    ConvertGenericOp<math::PowFOp, cuda_tile::FPowFOp, Signedness::None, IntegerUpCast::None>,
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
    ConvertExpandDimsOp,
    ConvertExternElementwiseOp,
    ConvertForOp,
    ConvertFpToFpOp,
    ConvertGatherOp,
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

  patterns.add<ConvertGdcInlineAsmToNativeOp>(typeConverter, context, /*benefit=*/2);
  patterns.add<ConvertPackedFp4UpcastInlineAsmToNativeOp>(
      typeConverter, context, /*benefit=*/2);
  patterns.add<ConvertKnownNumericInlineAsmToNativeOp>(typeConverter, context,
                                                       /*benefit=*/2);

  patterns.add<ConvertLoadOp, ConvertStoreOp>(typeConverter, context, numStagesMap, computeCapability, numStages);

    patterns.add<ConvertDescriptorLoadOp, ConvertDescriptorStoreOp,
                 ConvertDescriptorGatherOp, ConvertDescriptorScatterOp>(
        typeConverter, context, numStagesMap, computeCapability, numStages);
  patterns.add<ConvertDotScaledOp>(typeConverter, context);
  patterns.add<ConvertDescriptorReduceOp>(typeConverter, context);
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

static LogicalResult convertTmaDescriptorOps(Operation *op,
                                             TypeConverter &converter) {
  IRRewriter rewriter(op->getContext());
  auto ctx = op->getContext();
  auto loc = op->getLoc();
  WalkResult result = op->walk([&](Operation *op) {
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
        Value tensorDesc = funcOp.getArgument(i);
        if (isa<triton::TensorDescType>(tensorDesc.getType())) {
          // [tensordesc, ptr, shape, stride]
          // 'i' is the tensordesc type
          int argIdx = i;
          rewriter.setInsertionPointToStart(&funcOp.getBody().front());
          auto padding = cuda_tile::PaddingValue::zero;
          if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(
                  i, "tileir.padding_nan")) {
            if (attr.getInt() != 0)
              padding = cuda_tile::PaddingValue::nan;
            funcOp.removeArgAttr(i, "tileir.padding_nan");
          }
          auto tensorDescType =
              cast<triton::TensorDescType>(tensorDesc.getType());
          auto descBlock = tensorDescType.getBlockType();
          if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(
                  argIdx, kRoundHostDescriptor)) {
            if (attr.getInt() != 0) {
              if (!descBlock.getElementType().isF32()) {
                funcOp.emitError(
                    "round_f32_to_tf32 requires an f32 descriptor");
                return WalkResult::interrupt();
              }
              // make_ttir inlines ordinary descriptor helpers before this pass.
              // Nested loop/if captures still directly use this SSA descriptor.
              // View-valued loop arguments/results are illegal in PUBLIC Tile
              // IR; do not silently lose the property on an unhandled forward.
              for (Operation *user : tensorDesc.getUsers()) {
                if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(
                        user)) {
                  user->setAttr(kRoundHostDescriptor, rewriter.getUnitAttr());
                } else if (!isa<triton::DescriptorStoreOp,
                                triton::DescriptorScatterOp,
                                triton::DescriptorReduceOp>(user)) {
                  user->emitError("cannot forward a rounding host descriptor; "
                                  "inline helpers and capture the descriptor "
                                  "directly in control-flow regions");
                  return WalkResult::interrupt();
                }
              }
            }
            funcOp.removeArgAttr(argIdx, kRoundHostDescriptor);
          }
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
          for (size_t j = 0; j < rank; j++)
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
          for (int i = 0; i < stride.size() - 1; i++)
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
          SmallVector<int32_t> traversalStrides;
          for (auto i64Shape : tileShape) {
            arrayOfi32Shape.push_back(i64Shape);
            traversalStrides.push_back(1); // Set traversal stride to 1
          }

          SmallVector<int32_t> dimMap(rank);
          std::iota(dimMap.begin(), dimMap.end(), 0);

          auto tileStridedViewTy = cuda_tile::StridedViewType::get(
              ctx, rewriter.getDenseI32ArrayAttr(arrayOfi32Shape),
              rewriter.getDenseI32ArrayAttr(traversalStrides), tensorViewTy,
              dimMap, cuda_tile::PaddingValueAttr::get(ctx, padding));

          auto stridedViewOp = cuda_tile::MakeStridedViewOp::create(
              rewriter, loc, tileStridedViewTy, makeTensorViewOp);

          auto castBackToTensorDescriptorOp =
              UnrealizedConversionCastOp::create(rewriter, loc, tensorDescType,
                                                 stridedViewOp.getResult());

          rewriter.replaceAllUsesWith(
              tensorDesc, castBackToTensorDescriptorOp.getResult(0));
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
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
    if (failed(convertTmaDescriptorOps(mod.getOperation(), typeConverter)))
      return signalPassFailure();

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
