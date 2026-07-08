#include "TritonToTileIR/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-tileir-utils"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace bridge_utils {
namespace {

enum class IdentityValue {
  NONE,
  ZERO,
  ONE,
  POS_INF,
  NEG_INF,
  MAX_SIGNED,
  MIN_SIGNED,
  MAX_UNSIGNED,
  MIN_UNSIGNED
};

// Helper function to convert IdentityValue to string for debugging
static const char *identityValueToString(IdentityValue value) {
  switch (value) {
  case IdentityValue::NONE:
    return "NONE";
  case IdentityValue::ZERO:
    return "ZERO";
  case IdentityValue::ONE:
    return "ONE";
  case IdentityValue::POS_INF:
    return "POS_INF";
  case IdentityValue::NEG_INF:
    return "NEG_INF";
  case IdentityValue::MAX_SIGNED:
    return "MAX_SIGNED";
  case IdentityValue::MIN_SIGNED:
    return "MIN_SIGNED";
  case IdentityValue::MAX_UNSIGNED:
    return "MAX_UNSIGNED";
  case IdentityValue::MIN_UNSIGNED:
    return "MIN_UNSIGNED";
  }
  return "UNKNOWN";
}

Attribute getIdentitiesAttr(MLIRContext *context,
                            ConversionPatternRewriter &rewriter, Type eltType,
                            IdentityValue value) {
  if (auto floatTy = dyn_cast<FloatType>(eltType)) {
    const auto &semantics = floatTy.getFloatSemantics();
    switch (value) {
    case IdentityValue::ZERO:
      return FloatAttr::get(floatTy, APFloat::getZero(semantics));
    case IdentityValue::ONE:
      return FloatAttr::get(floatTy, APFloat::getOne(semantics));
    case IdentityValue::NEG_INF:
      return FloatAttr::get(floatTy,
                            APFloat::getInf(semantics, /*negative=*/true));
    case IdentityValue::POS_INF:
      return FloatAttr::get(floatTy,
                            APFloat::getInf(semantics, /*negative=*/false));
    default:
      llvm_unreachable("unexpected identity value for float type");
    }
  }

  if (auto intTy = dyn_cast<IntegerType>(eltType)) {
    auto width = intTy.getWidth();
    switch (value) {
    case IdentityValue::ZERO:
      return IntegerAttr::get(intTy, APInt::getZero(width));
    case IdentityValue::ONE:
      return IntegerAttr::get(intTy, APInt(width, 1));
    case IdentityValue::MAX_SIGNED:
      return IntegerAttr::get(intTy, APInt::getSignedMaxValue(width));
    case IdentityValue::MIN_SIGNED:
      return IntegerAttr::get(intTy, APInt::getSignedMinValue(width));
    case IdentityValue::MAX_UNSIGNED:
      return IntegerAttr::get(intTy, APInt::getMaxValue(width));
    case IdentityValue::MIN_UNSIGNED:
      return IntegerAttr::get(intTy, APInt::getMinValue(width));
    default:
      llvm_unreachable("unexpected identity value for integer type");
    }
  }

  assert(false && "unexpected data type for reduceOp.");
}

bool isI8OrI1ElementTensor(Type type) {
  auto tensorTy = dyn_cast<cuda_tile::TileType>(type);
  if (!tensorTy)
    return false;
  auto elmTy = tensorTy.getElementType();
  if (auto intTy = dyn_cast<IntegerType>(elmTy)) {
    auto bitWidth = intTy.getIntOrFloatBitWidth();
    return bitWidth == 8 || bitWidth == 1;
  }
  return false;
}

} // namespace

// Helper function to find operations that consume both block arguments
static SmallVector<Operation *> findConsumingOperations(Value inputOperand,
                                                        Value identityOperand) {
  SmallVector<Operation *> consumingOps;

  // Collect all operations that use the input operand
  SmallPtrSet<Operation *, 8> inputUsers;
  for (auto &use : inputOperand.getUses()) {
    inputUsers.insert(use.getOwner());
  }

  // Check which of these also use the identity operand
  for (auto &use : identityOperand.getUses()) {
    Operation *op = use.getOwner();
    if (inputUsers.contains(op)) {
      // Verify the operation actually uses both values as operands
      bool usesInput = llvm::any_of(op->getOperands(), [&](Value operand) {
        return operand == inputOperand;
      });
      bool usesIdentity = llvm::any_of(op->getOperands(), [&](Value operand) {
        return operand == identityOperand;
      });

      if (usesInput && usesIdentity) {
        consumingOps.push_back(op);
      }
    }
  }

  return consumingOps;
}

// Helper function to analyze operations and get consistent identity
static FailureOr<IdentityValue>
analyzeConsistentIdentity(ArrayRef<Operation *> consumingOps,
                          size_t pairIndex) {
  if (consumingOps.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Warning: No consumer found for Arguments in pair "
               << pairIndex << ".\n");
    return IdentityValue::NONE;
  }

  // Helper to analyze operation and determine reduction type
  auto analyzeOp = [&](Operation *op) -> IdentityValue {
    // Integer comparison operations
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      switch (cmp.getPredicate()) {
      case arith::CmpIPredicate::sgt:
        return IdentityValue::MIN_SIGNED;
      case arith::CmpIPredicate::slt:
        return IdentityValue::MAX_SIGNED;
      case arith::CmpIPredicate::ugt:
        return IdentityValue::MIN_UNSIGNED;
      case arith::CmpIPredicate::ult:
        return IdentityValue::MAX_UNSIGNED;
      default:
        break;
      }
    }

    // Float comparison operations
    if (auto cmp = dyn_cast<arith::CmpFOp>(op)) {
      switch (cmp.getPredicate()) {
      case arith::CmpFPredicate::OGT:
        return IdentityValue::NEG_INF;
      case arith::CmpFPredicate::OLT:
        return IdentityValue::POS_INF;
      default:
        break;
      }
    }

    // Arithmetic operations
    if (isa<arith::AddFOp, arith::AddIOp>(op))
      return IdentityValue::ZERO;
    if (isa<arith::MulFOp, arith::MulIOp>(op))
      return IdentityValue::ONE;
    if (isa<arith::XOrIOp>(op))
      return IdentityValue::ZERO;
    if (isa<arith::AndIOp>(op))
      // Bitwise AND identity requires all bits set to 1, not just value 1
      return IdentityValue::MAX_UNSIGNED;
    if (isa<arith::OrIOp>(op))
      return IdentityValue::ZERO;

    // Min/Max operations
    if (isa<arith::MinNumFOp, arith::MinimumFOp>(op))
      return IdentityValue::POS_INF;
    if (isa<arith::MaxNumFOp, arith::MaximumFOp>(op))
      return IdentityValue::NEG_INF;
    if (isa<arith::MinSIOp>(op))
      return IdentityValue::MAX_SIGNED;
    if (isa<arith::MinUIOp>(op))
      return IdentityValue::MAX_UNSIGNED;
    if (isa<arith::MaxSIOp>(op))
      return IdentityValue::MIN_SIGNED;
    if (isa<arith::MaxUIOp>(op))
      return IdentityValue::MIN_UNSIGNED;

    assert(!isa<triton::CallOp>(op) && "CallOp need to be inlined");
    return IdentityValue::NONE;
  };

  // Analyze all consuming operations to get their identities
  SmallVector<IdentityValue> identityValues;
  for (Operation *op : consumingOps) {
    IdentityValue identity = analyzeOp(op);
    if (identity != IdentityValue::NONE) {
      identityValues.push_back(identity);
    }
  }

  if (identityValues.empty()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Warning: No valid identity values found for Arguments in pair "
        << pairIndex << ".\n");
    return IdentityValue::NONE;
  }

  // Check if all identities are the same (with early exit optimization)
  IdentityValue firstIdentity = identityValues[0];
  for (size_t j = 1; j < identityValues.size(); ++j) {
    if (identityValues[j] != firstIdentity) {
      llvm::errs() << "Error: Arguments pair in combineOp are consumed by "
                      "operations with different identities: "
                   << "first=" << identityValueToString(firstIdentity)
                   << ", found=" << identityValueToString(identityValues[j])
                   << ", please check if combineOp satisfies associative and "
                      "commutative laws.\n";
      return failure();
    }
  }

  return firstIdentity;
}

FailureOr<ArrayAttr>
getIdentitiesFromCombineOp(Region &combineOp, ArrayRef<Type> retType,
                           ConversionPatternRewriter &rewriter) {
  SmallVector<Attribute> attributes;
  MLIRContext *context = combineOp.getContext();
  // Here, it tries to deduce the correct identity, but even if it fails,
  // the backend can still calculate the correct result.
  // It's hard to code a general logic to cover all complicate region
  // calculations. Hence, backend should ensure ReduceOp to be identity
  // insensitive in power of 2 cases. Details refer to:
  Block &block = combineOp.front();
  auto blockArgs = block.getArguments();
  size_t numReturns = blockArgs.size() / 2;

#ifndef NDEBUG
  // Validate that we have an even number of arguments
  assert(blockArgs.size() % 2 == 0 &&
         "Combine operation must have an even number of block arguments");
  // Number of returns should be half of all operands
  assert(retType.size() == numReturns &&
         "Number of return types must be half of block arguments");
  // Validate the block arguments types with the retType
  // First half of blockArgs are input operands, second half are identities
  for (size_t i = 0; i < numReturns; i++) {
    // Check input operand type
    assert(blockArgs[i].getType() ==
               dyn_cast<ShapedType>(retType[i]).getElementType() &&
           "block argument type mismatch for input operand");
    // Check identity type
    assert(blockArgs[i + numReturns].getType() ==
               dyn_cast<ShapedType>(retType[i]).getElementType() &&
           "block argument type mismatch for identity");
  }
#endif // NDEBUG

  // Process each pair of arguments
  for (size_t i = 0; i < numReturns; i++) {
    Value inputOperand = blockArgs[i];
    Value identityOperand = blockArgs[i + numReturns];
    IdentityValue identityValue = IdentityValue::ZERO;

    // Find operations and analyze their identities
    SmallVector<Operation *> consumingOps =
        findConsumingOperations(inputOperand, identityOperand);
    FailureOr<IdentityValue> identityValueOr =
        analyzeConsistentIdentity(consumingOps, i);

    if (failed(identityValueOr)) {
      // Identity consistency error - propagate failure
      return failure();
    }

    IdentityValue result = *identityValueOr;
    if (result == IdentityValue::NONE) {
      // Use dummy identity for cases with no valid identity value
      LLVM_DEBUG(llvm::dbgs()
                 << "Using dummy identity (ZERO) for pair " << i << ".\n");
      identityValue = IdentityValue::ZERO;
    } else {
      identityValue = result;
    }
    attributes.push_back(getIdentitiesAttr(
        context, rewriter, dyn_cast<ShapedType>(retType[i]).getElementType(),
        identityValue));
  }

  return ArrayAttr::get(context, attributes);
}

bool canMapToCudaTile(triton::FuncOp op, CudaTileTypeConverter &typeConverter) {
  // kernel in cuda tile do not return any result.
  if (op.getNumResults() > 0 && op.getSymVisibility() == "public")
    return false;
  // The operation is legal if we cannot convert a type to cuda tile.
  SmallVector<Type> newTypes;
  if (failed(typeConverter.convertTypes(op.getFunctionType().getInputs(),
                                        newTypes)))
    return false;
  if (failed(typeConverter.convertTypes(op.getFunctionType().getResults(),
                                        newTypes)))
    return false;
  return true;
}

/// Upcast input (expected to be a cuda tile tensor) to i16 from i1 or i8,
/// otherwise just return the input.
Value upCastOrSelf(OpBuilder &builder, Location loc, Value input,
                   Signedness signedness, IntegerUpCast integerUpCast) {
  auto type = dyn_cast<cuda_tile::TileType>(input.getType());
  // Cast not needed.
  if (integerUpCast == IntegerUpCast::None || !isI8OrI1ElementTensor(type))
    return input;
  assert(integerUpCast == IntegerUpCast::To_I16 &&
         "only upcast to i16 is supported");
  auto tensorTy =
      cuda_tile::TileType::get(type.getShape(), builder.getIntegerType(16));
  auto signednessAttr = signedness == Signedness::Unsigned
                            ? cuda_tile::Signedness::Unsigned
                            : cuda_tile::Signedness::Signed;
  return cuda_tile::ExtIOp::create(builder, loc, tensorTy, input,
                                   signednessAttr);
}

/// Downcast the result of `createOp` back to i1 or i8.
Value downCastOrSelf(
    OpBuilder &builder, Location loc, ArrayRef<Value> operands, Type retType,
    llvm::function_ref<Value(OpBuilder &, Location, Type, ArrayRef<Value>)>
        createOp,
    IntegerUpCast integerUpCast) {
  if (integerUpCast == IntegerUpCast::None || !isI8OrI1ElementTensor(retType))
    return createOp(builder, loc, retType, operands);

  auto tensorTy =
      cuda_tile::TileType::get(cast<cuda_tile::TileType>(retType).getShape(),
                               builder.getIntegerType(16));
  auto newOp = createOp(builder, loc, tensorTy, operands);
  return cuda_tile::TruncIOp::create(builder, loc, retType, newOp);
}

LogicalResult matchAndRewriteGenericOpImpl(
    Operation *op, ValueRange operands, const TypeConverter *converter,
    ConversionPatternRewriter &rewriter,
    llvm::function_ref<Value(OpBuilder &, Location, Type, ArrayRef<Value>)>
        createOp,
    Signedness signedness, IntegerUpCast integerUpCast) {
  if (!op->hasTrait<OpTrait::OneResult>())
    return rewriter.notifyMatchFailure(op, "expect single result operation");

  auto loc = op->getLoc();
  SmallVector<Value> newOperands;
  for (auto input : operands)
    newOperands.push_back(
        upCastOrSelf(rewriter, loc, input, signedness, integerUpCast));

  Type retType = converter->convertType(op->getResult(0).getType());
  auto newOp = downCastOrSelf(rewriter, loc, newOperands, retType, createOp,
                              integerUpCast);
  rewriter.replaceOp(op, newOp);
  return success();
}

CudaTileTypeConverter::CudaTileTypeConverter() {
  // in python api level, we use 0 as a placeholder for tensordesc type
  // so we need to convert it to i32 type
  addConversion([](triton::TensorDescType type) {
    MLIRContext *ctx = type.getContext();
    auto descBlock = type.getBlockType();
    auto tileShape = descBlock.getShape();
    auto rank = tileShape.size();

    SmallVector<int64_t> globalShape(rank, cuda_tile::TensorViewType::kDynamic);
    SmallVector<int64_t> globalStride(rank,
                                      cuda_tile::TensorViewType::kDynamic);
    globalStride[rank - 1] = 1;

    auto pointeeType = descBlock.getElementType();
    if (auto intTy = dyn_cast<mlir::IntegerType>(pointeeType)) {
      pointeeType = mlir::IntegerType::get(ctx, intTy.getWidth(),
                                           mlir::IntegerType::Signless);
    }

    auto tensorViewTy = cuda_tile::TensorViewType::get(
        ctx, pointeeType, globalShape, globalStride);

    SmallVector<int32_t> dimMap(rank);
    std::iota(dimMap.begin(), dimMap.end(), 0);

    SmallVector<int32_t> arrayOfi32Shape;
    for (auto i64Shape : tileShape) {
      arrayOfi32Shape.push_back(i64Shape);
    }
    auto shapeAttr = DenseI32ArrayAttr::get(ctx, arrayOfi32Shape);

    return cuda_tile::PartitionViewType::get(
        ctx, shapeAttr, tensorViewTy, dimMap,
        cuda_tile::PaddingValueAttr::get(ctx, cuda_tile::PaddingValue::zero));
  });
  addConversion([](cuda_tile::TensorViewType type) { return type; });
  addConversion([](cuda_tile::PartitionViewType type) { return type; });
  addConversion([](cuda_tile::TokenType type) { return type; });
  addConversion(
      [](FloatType type) { return cuda_tile::TileType::get({}, type); });
  addConversion(
      [](IntegerType type) { return cuda_tile::TileType::get({}, type); });
  // Convert a pointer type into a zero-ranked tensor type, where the element
  // type is a CUDA pointer type.
  addConversion([this](triton::PointerType ptrType) -> Type {
    auto pointeeType = ptrType.getPointeeType();
    // Do not crash on cuda tile verifier if we get a ptr<tensor>.
    if (isa<RankedTensorType>(pointeeType))
      return Type();
    return cuda_tile::TileType::get({},
                                    cuda_tile::PointerType::get(pointeeType));
  });
  // Convert a ranked tensor type to a CUDA tensor type. There are two
  // possible conversions: 1.	When the element type is a pointer type: Extract
  // the pointer type element from the zero-ranked tensor type produced by the
  // type converter, then repack it into a new tensor while adjusting the
  // shape accordingly.
  // 2. When the element type is an integer or a floating point scalar type,
  // pack it into a CUDA tensor type adjusting the shape accordingly.
  addConversion([this](RankedTensorType tensorType) {
    auto elemType = tensorType.getElementType();
    if (isa<triton::PointerType>(elemType)) {
      cuda_tile::TileType tensorTy =
          cast<cuda_tile::TileType>(convertType(elemType));
      return cuda_tile::TileType::get(tensorType.getShape(),
                                      tensorTy.getElementType());
    }
    return cuda_tile::TileType::get(tensorType.getShape(), elemType);
  });
}

} // namespace bridge_utils
} // namespace mlir
