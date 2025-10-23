#ifndef TRITON_ADAPTER_TRITONOPCONVERTER_H
#define TRITON_ADAPTER_TRITONOPCONVERTER_H

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

/*
Convert `tt.precise_div` operation to `arith.divf` operation.
tensor_x / tensor_y

```ttir
  %11 = tt.precise_divf %7, %10 : tensor<100xf32>
```

converts to:

```mlir
  %11 = arith.divf %7, %10 : tensor<100xf32>
```
*/
struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
public:
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

/*
 * Move tt.bitcast to a previous location if tt.bitcast is not directly applied
 * on function arguments
 */
class BitcastCanonicalizer : public OpRewritePattern<triton::BitcastOp> {
public:
  using OpRewritePattern<triton::BitcastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::BitcastOp bitcastOp,
                                PatternRewriter &rewriter) const override;
};

template <typename MathOp>
class ScalarMathCanonicalizer : public OpRewritePattern<MathOp> {
public:
  using OpRewritePattern<MathOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer expects single scalar output.");
    }
    if (!op->getResult(0).getType().isIntOrIndexOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles scalar load scene.");
    }
    if (auto linalgOp = op->template getParentOfType<triton::ReduceOp>()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles op not within tt.reduce.");
    }
    if (auto linalgOp = op->template getParentOfType<triton::ScanOp>()) {
      return rewriter.notifyMatchFailure(
          op, "ScalarMathCanonicalizer handles op not within tt.scan.");
    }
    auto loc = op.getLoc();
    llvm::SmallVector<Value> inputs;
    for (auto input : op->getOperands()) {
      auto blkTy = RankedTensorType::get({(int64_t)1}, input.getType());
      auto inputSplat = rewriter.create<triton::SplatOp>(loc, blkTy, input);
      inputs.push_back(inputSplat.getResult());
    }
    auto blkOp = rewriter.create<MathOp>(loc, inputs);
    Value offset =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto extractOp =
        rewriter.create<tensor::ExtractOp>(loc, blkOp.getResult(), offset);
    rewriter.replaceOp(op, extractOp);
    return success();
  }
};

/*
 * Rewrite tt.make_tensor_ptr with non-contiguous order to
 * tt.make_tensor_ptr + tt.load + tt.trans.
 */
class MakeTensorPtrCanonicalizer
    : public OpRewritePattern<triton::MakeTensorPtrOp> {
public:
  using OpRewritePattern<triton::MakeTensorPtrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override;
};

class ReduceSingleCanonicalizer : public OpRewritePattern<triton::ReduceOp> {
public:
  using OpRewritePattern<triton::ReduceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override;
};

class DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
public:
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplatConverter : public OpConversionPattern<triton::SplatOp> {
public:
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
public:
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
public:
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class ClampFConverter : public OpConversionPattern<triton::ClampFOp> {
public:
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
public:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename OpTy>
class ReductionOpBaseConverter : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpTy op,
                                typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() && "Expected reduction axis is within operand's rank");

    auto reductionOps = this->getRedOps(op);
    if (reductionOps.size() == 1) {
      return this->convertToTargetOp(op, adaptor, rewriter);
    }
    return this->convertToTargetOpExtended(op, adaptor, rewriter);
  }

protected:
  llvm::SmallVector<Operation *> getRedOps(OpTy redOp) const {
    auto redBody = redOp.getBody();
    return llvm::map_to_vector(redBody->without_terminator(),
                              [](Operation &op) { return &op; });
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr = llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
                    .Case([&](arith::AddFOp) {
                      return rewriter.getFloatAttr(constantType, 0.f);
                    })
                    .Case([&](arith::AddIOp) {
                      return rewriter.getIntegerAttr(constantType, 0);
                    })
                    .Case([&](arith::MulFOp) {
                      return rewriter.getFloatAttr(constantType, 1.f);
                    })
                    .template Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
                      return rewriter.getFloatAttr(
                          constantType, -std::numeric_limits<float>::infinity());
                    })
                    .template Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
                      return rewriter.getFloatAttr(
                          constantType, std::numeric_limits<float>::infinity());
                    })
                    .Case([&](arith::MinSIOp) {
                      return rewriter.getIntegerAttr(constantType,
                                                    llvm::maxIntN(bitWidth));
                    })
                    .Case([&](arith::MinUIOp) {
                      return rewriter.getIntegerAttr(constantType,
                                                    llvm::maxUIntN(bitWidth));
                    })
                    .Case([&](arith::MaxSIOp) {
                      return rewriter.getIntegerAttr(constantType,
                                                    llvm::minIntN(bitWidth));
                    })
                    .Case([&](arith::MaxUIOp) {
                      return rewriter.getIntegerAttr(constantType, 0);
                    })
                    .Case([&](arith::OrIOp) {
                      return rewriter.getIntegerAttr(constantType, 0);
                    })
                    .Case([&](arith::AndIOp) {
                      return rewriter.getIntegerAttr(constantType, 1);
                    })
                    .Case([&](arith::XOrIOp) {
                      return rewriter.getIntegerAttr(constantType, 0);
                    })
                    .Default([](Operation *op) {
                      op->dump();
                      llvm_unreachable("Reduction op not supported yet");
                      return nullptr;
                    });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType, attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return isa<FloatType>(elemType) &&
          elemType.getIntOrFloatBitWidth() <
              Float32Type::get(elemType.getContext()).getWidth() &&
          (isa<arith::AddFOp>(redOp) || isa<arith::MulFOp>(redOp));
  }

  Value getRedElement(
    Value lhs, Value rhs, const Location loc, Operation *redOp, OpBuilder &b,
    const bool convertLhsToF32Precision) const {
  return llvm::TypeSwitch<Operation *, Value>(redOp)
      .template Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
        if (convertLhsToF32Precision) {
          lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                        lhs);
        }
        return b.create<decltype(redOp)>(loc, lhs, rhs);
      })
      .template Case<arith::AddIOp, arith::MaximumFOp, arith::MaxNumFOp,
            arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp>(
          [&](auto redOp) { return b.create<decltype(redOp)>(loc, lhs, rhs); })
      .Default([](Operation *op) {
        op->dump();
        llvm_unreachable("Reduction op not yet supported");
        return nullptr;
      });
  }

  virtual bool isReductionOpSupported(Operation *redOp) const = 0;

  virtual LogicalResult
  convertToTargetOp(OpTy op, typename OpTy::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const = 0;

  virtual LogicalResult
  convertToTargetOpExtended(OpTy op, typename OpTy::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const = 0;
};

class ReduceConverter : public ReductionOpBaseConverter<triton::ReduceOp> {
public:
  explicit ReduceConverter(MLIRContext *context)
      : ReductionOpBaseConverter<triton::ReduceOp>(context) {}

  using ReductionOpBaseConverter<triton::ReduceOp>::ReductionOpBaseConverter;

protected:
  bool isReductionOpSupported(Operation *redOp) const override;

  LogicalResult
  convertToTargetOp(triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ReduceOp op, typename triton::ReduceOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;

};

class ScanConverter : public ReductionOpBaseConverter<triton::ScanOp> {
public:
  explicit ScanConverter(MLIRContext *context)
      : ReductionOpBaseConverter<triton::ScanOp>(context) {}

  using ReductionOpBaseConverter<triton::ScanOp>::ReductionOpBaseConverter;

protected:
  bool isReductionOpSupported(Operation *redOp) const override;

  LogicalResult
  convertToTargetOp(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;

};

class ExternElementwiseClOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class JoinConverter : public OpConversionPattern<triton::JoinOp> {
public:
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class SplitConverter : public OpConversionPattern<triton::SplitOp> {
public:
  using OpConversionPattern<triton::SplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class CatConverter : public OpConversionPattern<triton::CatOp> {
public:
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class GatherConverter : public OpConversionPattern<triton::GatherOp> {
private:
  static constexpr llvm::StringRef gatherFuncNameBase = "triton_gather";

public:
  using OpConversionPattern<triton::GatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class YieldConverter : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename LoopOpTy, typename =
    std::enable_if_t<std::is_same_v<LoopOpTy, scf::ForOp> ||
                     std::is_same_v<LoopOpTy, scf::WhileOp>>>
class LoopConverter : public OpConversionPattern<LoopOpTy> {
public:
  using OpConversionPattern<LoopOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOpTy op, typename OpConversionPattern<LoopOpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, BlockData> known;

    op->removeAttr("UnhandledLoopOp");
    BlockDataParser::rewriteLoopOp(op, rewriter, known);
    return success();
  }
};

class AdvanceConverter : public OpConversionPattern<triton::AdvanceOp> {
public:
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class MakeTensorPtrConverter
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
public:
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;
  explicit MakeTensorPtrConverter(MLIRContext *context)
      : OpConversionPattern<triton::MakeTensorPtrOp>(context) {}

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TransposeConverter : public OpConversionPattern<triton::TransOp> {
public:
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
public:
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonMulhiuiConverter : public OpConversionPattern<triton::MulhiUIOp> {
public:
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class TritonPreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
public:
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DeviceAssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

private:
  static constexpr llvm::StringRef printFuncNameBase = "triton_assert";
  static constexpr llvm::StringRef msgAttrName = "msg";

public:
  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class DevicePrintConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

private:
  static constexpr llvm::StringRef printFuncNameBase = "triton_print";
  static constexpr llvm::StringRef prefixAttrName = "prefix";
  static constexpr llvm::StringRef hexAttrName = "hex";

public:
  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};


struct SortOpConverter : public OpConversionPattern<triton::SortOp> {
    using OpConversionPattern<triton::SortOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(triton::SortOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};


struct DotScaledConverter : public OpConversionPattern<triton::DotScaledOp> {
    using OpConversionPattern<triton::DotScaledOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class PtrToIntConverter : public OpConversionPattern<triton::PtrToIntOp> {
public:
  using OpConversionPattern<triton::PtrToIntOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // end of namespace TTOpConverters

#endif
