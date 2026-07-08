#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Transform/Passes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

// MLIR pass TableGen now uses per-pass macros (GEN_PASS_DEF_*).
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_REWRITEASSUMEWITHCUDATILE
#include "Transform/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

// clang-format off
// Match pattern:
// %a = ... i32
// %rem = arith.remsi %a, %c8_i32 : i32
// %eq = arith.cmpi eq, %rem, %c0_i32 : i32
// llvm.intr.assume %eq : i1
// ->
// %tile_a = buildin.unrealized_conversion_cast %a : i32 -> tile<i32>
// %assume_a = assume div_by<8 : i64>, %tile_a : tile<i32>
// replace %a with %assume_a
//
// Or match pattern for ptr types:
// %ptr = ... tt.ptr<i32>
// %ptr_int = tt.ptr_to_int %ptr : !tt.ptr<i32> -> i64
// %rem = arith.remsi %ptr_int, %c16_i64 : i64
// %eq = arith.cmpi eq, %rem, %c0_i64 : i64
// llvm.intr.assume %eq : i1
// ->
// %cuda_ptr = buildin.unrealized_conversion_cast %ptr : !tt.ptr<i32> -> tile<ptr<i32>>
// %assume_cuda_ptr = assume div_by<16 : i64>, %cuda_ptr : tile<ptr<i32>>
// %tt_ptr = buildin.unrealized_conversion_cast %assume_cuda_ptr : tile<ptr<i32>> -> tt.ptr<i32>
// replace %ptr with %tt_ptr
// clang-format on
LogicalResult RewriteArithAssumeImpl(LLVM::AssumeOp assumeOp,
                                     PatternRewriter &rewriter) {
  auto loc = assumeOp.getLoc();
  Value condValue = assumeOp.getCond();

  // Step 1: Check if the condition is from a arith.cmpi eq operation
  auto cmpOp = condValue.getDefiningOp<arith::CmpIOp>();
  if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::eq)
    return failure();

  // Step 2: Get the operands of cmpi
  Value remResult = cmpOp.getLhs();
  Value zeroConstant = cmpOp.getRhs();

  // Step 3: Check if zeroConstant is a constant 0
  auto zeroDefOp = zeroConstant.getDefiningOp<arith::ConstantOp>();
  if (!zeroDefOp)
    return failure();

  // Check if the constant value is 0
  auto zeroAttr = dyn_cast<IntegerAttr>(zeroDefOp.getValue());
  if (!zeroAttr || !zeroAttr.getValue().isZero())
    return failure();

  // Step 4: Check if remResult is from a arith.remsi operation
  auto remOp = remResult.getDefiningOp<arith::RemSIOp>();
  if (!remOp)
    return failure();

  // Step 5: Get the operands of remsi
  Value intOrPtrToInt = remOp.getLhs();
  Value divisorConstant = remOp.getRhs();

  // Step 6: Check if divisorConstant is a constant
  auto divisorOp = divisorConstant.getDefiningOp<arith::ConstantOp>();
  if (!divisorOp)
    return failure();

  // Get the divisor value
  auto divisorAttr = dyn_cast<IntegerAttr>(divisorOp.getValue());
  if (!divisorAttr)
    return failure();
  int64_t divisor = divisorAttr.getValue().getSExtValue();

  auto definingOp = intOrPtrToInt.getDefiningOp();
  if (definingOp)
    definingOp->setAttr("tt.divisibility", divisorAttr);
  // There are two cases:
  // Case 1: intOrPtrToInt is a scalar integer value directly
  // Case 2: intOrPtrToInt is a result of tt.ptr_to_int operation
  auto ptrToIntOp = intOrPtrToInt.getDefiningOp<triton::PtrToIntOp>();
  if (ptrToIntOp) {
    Value ttPtr = ptrToIntOp.getOperand();
    auto ttPtrType = dyn_cast<triton::PointerType>(ttPtr.getType());
    if (!ttPtrType)
      return failure();

    Type pointeeType = ttPtrType.getPointeeType();
    Type cudaTilePtrType =
        cuda_tile::TileType::get({}, cuda_tile::PointerType::get(pointeeType));
    auto divByAttr = cuda_tile::DivByAttr::get(rewriter.getContext(), divisor,
                                               std::nullopt, std::nullopt);

    auto ttptr2cudaPtrOp = UnrealizedConversionCastOp::create(
        rewriter, loc, cudaTilePtrType, ttPtr);
    auto cudaTilePtr = ttptr2cudaPtrOp.getResult(0);
    auto assumeCudaTileOp =
        cuda_tile::AssumeOp::create(rewriter, loc, cudaTilePtr, divByAttr);
    Value newTtPtr =
        UnrealizedConversionCastOp::create(rewriter, loc, ttPtr.getType(),
                                           assumeCudaTileOp.getResult())
            .getResult(0);

    newTtPtr.getDefiningOp()->setAttr("tt.divisibility", divisorAttr);
    // Don't replace uses in the cast tt.ptr to cuda_tile.ptr operation and
    // those beyond dominance.
    DominanceInfo domInfo(assumeOp);
    ttPtr.replaceUsesWithIf(newTtPtr, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      if (user == ttptr2cudaPtrOp.getOperation())
        return false;
      if (domInfo.dominates(assumeOp, user))
        return true;
      return false;
    });
    return success();
  } else {
    // Handle integer case
    auto intType = dyn_cast<IntegerType>(intOrPtrToInt.getType());
    if (!isa<IntegerType>(intOrPtrToInt.getType()))
      return failure();

    // Create cuda_tile.div_by attribute
    auto divByAttr = cuda_tile::DivByAttr::get(rewriter.getContext(), divisor,
                                               std::nullopt, std::nullopt);

    Type cudaTileIntType =
        cuda_tile::TileType::get({}, intOrPtrToInt.getType());
    auto int2tileIntOp = UnrealizedConversionCastOp::create(
        rewriter, loc, cudaTileIntType, intOrPtrToInt);
    auto cudaTileInt = int2tileIntOp.getResult(0);
    auto assumeCudaTileOp =
        cuda_tile::AssumeOp::create(rewriter, loc, cudaTileInt, divByAttr);
    Value assumedInt = UnrealizedConversionCastOp::create(
                           rewriter, loc, intOrPtrToInt.getType(),
                           assumeCudaTileOp.getResult())
                           .getResult(0);

    assumedInt.getDefiningOp()->setAttr("tt.divisibility", divisorAttr);
    // Don't replace uses in the cast tt.ptr to cuda_tile.ptr operation and
    // those beyond dominance.
    DominanceInfo domInfo(assumeOp);
    intOrPtrToInt.replaceUsesWithIf(assumedInt, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      if (user == int2tileIntOp.getOperation())
        return false;
      if (domInfo.dominates(assumeOp, user))
        return true;
      return false;
    });
    return success();
  }
}

class CudaTileTensorAssumePattern : public OpRewritePattern<LLVM::AssumeOp> {
public:
  CudaTileTensorAssumePattern(MLIRContext *context)
      : OpRewritePattern<LLVM::AssumeOp>(context) {}

  LogicalResult matchAndRewrite(LLVM::AssumeOp assumeOp,
                                PatternRewriter &rewriter) const override {
    if (succeeded(RewriteArithAssumeImpl(assumeOp, rewriter))) {
      rewriter.eraseOp(assumeOp);
      return success();
    }

    rewriter.eraseOp(assumeOp);
    return failure();
  }
};

// Pass to rewrite llvm.intr.assume to cuda_tile.assume
class RewriteAssumeWithCudaTilePass
    : public ::mlir::triton::impl::RewriteAssumeWithCudaTileBase<
          RewriteAssumeWithCudaTilePass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // Create rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<CudaTileTensorAssumePattern>(context);

    // Apply rewrite patterns
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createRewriteAssumeWithCudaTilePass() {
  return std::make_unique<RewriteAssumeWithCudaTilePass>();
}
