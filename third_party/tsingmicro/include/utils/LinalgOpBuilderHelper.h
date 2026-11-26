#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

// buildLinalgElementwise with OneResult and custom result type
template <typename OpT>
static Value buildLinalgElementwise(OpBuilder &rewriter, Location loc,
                                    RankedTensorType resultType,
                                    ValueRange inputs) {
  static_assert(OpT::template hasTrait<mlir::OpTrait::OneResult>() &&
                    OpT::template hasTrait<mlir::OpTrait::Elementwise>(),
                "OpT must have OneResult and Elementwise traits");
  auto inputType = cast<RankedTensorType>(inputs[0].getType());
  auto rank = inputType.getRank();
  auto identityMap =
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  SmallVector<AffineMap> indexingMaps(inputs.size() + 1, identityMap);
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);

  auto output = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        resultType.getElementType());

  auto linalgBuilder = [&](OpBuilder &nestedBuilder, Location nestedloc,
                           ValueRange iterArgs) {
    Value opRes = OpT::create(nestedBuilder, nestedloc,
                              iterArgs.back().getType(), iterArgs.drop_back());
    linalg::YieldOp::create(nestedBuilder, nestedloc, opRes);
  };

  return linalg::GenericOp::create(rewriter, loc, resultType, inputs,
                                   ValueRange{output}, indexingMaps,
                                   iteratorTypes, linalgBuilder)
      .getResult(0);
}

// buildLinalgElementwise with OneResult and SameOperandsAndResultType
template <typename OpT>
static Value buildLinalgElementwise(OpBuilder &rewriter, Location loc,
                                    ValueRange inputs) {
  static_assert(
      OpT::template hasTrait<mlir::OpTrait::OneResult>() &&
          OpT::template hasTrait<mlir::OpTrait::Elementwise>() &&
          OpT::template hasTrait<mlir::OpTrait::SameOperandsAndResultType>(),
      "OpT must have OneResult, Elementwise and SameOperandsAndResultType "
      "traits");
  auto inputType = cast<RankedTensorType>(inputs[0].getType());
  return buildLinalgElementwise<OpT>(rewriter, loc, inputType, inputs);
}
