/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_TRITONOPCONVERTER_H
#define TRITON_ADAPTER_TRITONOPCONVERTER_H

#include "TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-to-linalg"

namespace TTOpConverters {
using namespace mlir;
using namespace triton;

static constexpr unsigned kFuncNameCap = 128;

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
Convert `tt.fp_to_fp` operation with RTNE (default) rounding mode to
`arith.truncf` or `arith.extf` operation.

For default RTNE rounding:
- downcast: tt.fp_to_fp -> arith.truncf
- upcast: tt.fp_to_fp -> arith.extf
- same-bitwidth conversions between distinct floating-point formats:
  tt.fp_to_fp -> arith.extf (to f32) -> arith.truncf

Only an operation whose complete input and result MLIR types are identical can
be eliminated as an identity conversion.

Note: Non-RTNE rounding modes (e.g., RTZ) are handled by TritonToHFusion pass.
*/
struct FpToFpCanonicalizer : public OpRewritePattern<triton::FpToFpOp> {
public:
  using OpRewritePattern<triton::FpToFpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::FpToFpOp op,
                                PatternRewriter &rewriter) const override;
};

class SelectCanonicalizer : public OpRewritePattern<arith::SelectOp> {
public:
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override;
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
    if (auto linalgOp =
            op->template getParentOfType<triton::MapElementwiseOp>()) {
      return rewriter.notifyMatchFailure(
          op,
          "ScalarMathCanonicalizer handles op not within tt.map_elementwise.");
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

class UnsplatConverter : public OpConversionPattern<triton::UnsplatOp> {
public:
  using OpConversionPattern<triton::UnsplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
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

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction axis is within operand's rank");

    auto realReductionOps = this->getRealReductionOps(op);
    if (realReductionOps.size() == 1) {
      return this->convertToTargetOp(op, adaptor, rewriter);
    }
    return this->convertToTargetOpExtended(op, adaptor, rewriter);
  }

protected:
  llvm::SmallVector<Operation *> getReductionOps(OpTy reductionOp) const {
    auto reductionBody = reductionOp.getBody();
    return llvm::map_to_vector(reductionBody->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  llvm::SmallVector<Operation *> getRealReductionOps(OpTy reductionOp) const {
    auto *body = reductionOp.getBody();
    auto *terminator = body->getTerminator();

    // Compute the backward slice from yielded values: only ops that
    // actually contribute to the computation of the reduction result
    // are considered.
    llvm::DenseSet<Operation *> liveOps;
    llvm::SmallVector<Value> worklist(terminator->getOperands());
    while (!worklist.empty()) {
      Value val = worklist.pop_back_val();
      if (auto *defOp = val.getDefiningOp()) {
        if (defOp->getBlock() == body && liveOps.insert(defOp).second) {
          for (auto operand : defOp->getOperands())
            worklist.push_back(operand);
        }
      }
    }

    // Count only live ops, excluding type conversions that serve as
    // precision promotion or demotion (e.g. bf16 -> f32 -> bf16).
    llvm::SmallVector<Operation *> realOps;
    for (Operation &bodyOp : body->without_terminator()) {
      if (!liveOps.contains(&bodyOp))
        continue;
      if (isa<arith::ExtFOp, arith::TruncFOp, arith::BitcastOp>(&bodyOp))
        continue;
      realOps.push_back(&bodyOp);
    }
    return realOps;
  }

  arith::ConstantOp
  getMultiOpReductionBaseConstOp(ConversionPatternRewriter &rewriter, OpTy op,
                                 Location loc, Type constantType) const {
    // for multiop reduce of 1 element result is defined as exactly this element
    // for multiop reduce of tensor with N elements default value
    // is not involved in result computation
    auto reductionOps = this->getReductionOps(op);
    assert(reductionOps.size() == 1);
    auto reductionOp = reductionOps.front();

    assert(constantType.isIntOrFloat());

    if (constantType.isInteger()) {
      return rewriter.create<arith::ConstantOp>(
          loc, constantType, rewriter.getIntegerAttr(constantType, 0));
    }
    return rewriter.create<arith::ConstantOp>(
        loc, constantType, rewriter.getFloatAttr(constantType, 0.f));
  }

  arith::ConstantOp getReductionBaseConstOp(ConversionPatternRewriter &rewriter,
                                            Operation *reductionOp,
                                            Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(reductionOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case([&](arith::MulIOp) {
              return rewriter.getIntegerAttr(constantType, 1);
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

    return rewriter.create<arith::ConstantOp>(reductionOp->getLoc(),
                                              constantType, attr);
  }

  bool requiresF32Conversion(const Type elemType,
                             Operation *reductionOp) const {
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() <
               Float32Type::get(elemType.getContext())
                   .getIntOrFloatBitWidth() &&
           (isa<arith::AddFOp>(reductionOp) || isa<arith::MulFOp>(reductionOp));
  }

  Value getReductionElement(Value lhs, Value rhs, const Location loc,
                            Operation *reductionOp, OpBuilder &b,
                            const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(reductionOp)
        .template Case<arith::AddFOp, arith::MulFOp>([&](auto reductionOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<decltype(reductionOp)>(loc, lhs, rhs);
        })
        .template Case<arith::AddIOp, arith::MulIOp, arith::MaximumFOp,
                       arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp,
                       arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
                       arith::MaxUIOp, arith::AndIOp, arith::OrIOp,
                       arith::XOrIOp>([&](auto reductionOp) {
          return b.create<decltype(reductionOp)>(loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  virtual bool isReductionOpSupported(Operation *reductionOp) const = 0;

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
  bool isReductionOpSupported(Operation *reductionOp) const override;

  static bool isMultiReductionOpSupported(Operation *reductionOp);

  Value cloneReduceOps(OpBuilder &builder, Value in, Value out, Value opIns,
                       Value opOuts, triton::ReduceOp op) const;

  void
  checkIsNotCallOp(const llvm::SmallVector<Operation *> &reductionOps) const;

  bool isSCFOpReduce(const llvm::SmallVector<Operation *> &reductionOps) const;

  bool
  isMultiOpReduce(const llvm::SmallVector<Operation *> &reductionOps) const;

  Value computeReduceResultWithCompileFlag(
      OpBuilder &opBuilder, Location loc, Value lhs, Value rhs, Value source,
      Value initTensor, triton::ReduceOp reductionOp,
      bool compileOn91095Flag = false) const;

  LogicalResult
  convertToTargetOp(triton::ReduceOp op,
                    typename triton::ReduceOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ReduceOp op,
                            typename triton::ReduceOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;
};

class ScanConverter : public ReductionOpBaseConverter<triton::ScanOp> {
public:
  explicit ScanConverter(MLIRContext *context)
      : ReductionOpBaseConverter<triton::ScanOp>(context) {}

  using ReductionOpBaseConverter<triton::ScanOp>::ReductionOpBaseConverter;

protected:
  bool isReductionOpSupported(Operation *reductionOp) const override;

  LogicalResult
  convertToTargetOp(triton::ScanOp op, typename triton::ScanOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override;

  LogicalResult
  convertToTargetOpExtended(triton::ScanOp op,
                            typename triton::ScanOp::Adaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override;
};

/*
 * Decompose tt.map_elementwise region body into tensor-level Named Ops.
 * Each scalar op (arith, math, scf.if, etc.) is promoted to its tensor
 * counterpart, producing a chain of independent Named Ops.
 */
class MapElementwiseDecomposeConverter
    : public OpConversionPattern<triton::MapElementwiseOp> {
public:
  using OpConversionPattern<triton::MapElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  // Promote a single scalar op to tensor-level and record the result in
  // valueMap.  Returns the tensor result values (one per op result).
  SmallVector<Value> promoteOp(Operation *op,
                               llvm::DenseMap<Value, Value> &valueMap,
                               OpBuilder &builder, Location loc,
                               ArrayRef<int64_t> tensorShape) const;

  // Promote ops in a region body (excluding terminator), return the yielded
  // tensor values from the terminator.
  SmallVector<Value> promoteRegionBody(Region &region,
                                       llvm::DenseMap<Value, Value> &valueMap,
                                       OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> tensorShape) const;
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

// These predicates select only scalar !tt.ptr<T> transports. A
// tensor<...x!tt.ptr<T>> follows the separate tensor-pointer lowering.
bool hasScalarPointerResult(scf::IfOp op);
bool isScalarPointerSelect(arith::SelectOp op);

// Marks an scf.if temporarily rebuilt by IfConverter. Its scalar-pointer
// results are represented as complete i64 addresses, so only its own yields
// require the matching pointer-to-address conversion.
inline constexpr llvm::StringLiteral kScalarPointerCarrierBoundaryAttr =
    "ScalarPointerCarrierBoundary";

// Rebuild an scf.if with scalar-pointer results so the boundary carries
// complete i64 addresses and reconstructs memrefs only after the join.
// The original branch regions are moved into the new operation, preserving
// side effects and allowing the conversion driver to rewrite each scf.yield
// operand in place.
//
// Example:
//   %base = scf.if %cond -> !tt.ptr<f32> {
//     scf.yield %lhs : !tt.ptr<f32>
//   } else {
//     scf.yield %rhs : !tt.ptr<f32>
//   }
//   %ptr = tt.make_tensor_ptr %base, ...
// becomes an scf.if returning i64 plus one hivm.pointer_cast after the if.
class IfConverter : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Convert a scalar-pointer select into a select over complete integer addresses
// and reconstruct one memref after the selection. This handles both BlockPtr
// bases and ordinary scalar pointers without asking the backend to merge two
// memory objects.
//
// Example:
//   %base = arith.select %cond, %lhs, %rhs : !tt.ptr<f32>
//   %ptr = tt.make_tensor_ptr %base, ...
// becomes:
//   %lhs_addr = memref.extract_aligned_pointer_as_index %lhs
//   %rhs_addr = memref.extract_aligned_pointer_as_index %rhs
//   %selected_addr = arith.select %cond, %lhs_addr, %rhs_addr : i64
//   %base = hivm.pointer_cast %selected_addr : i64 to memref<?xf32>
class PointerSelectConverter : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Convert the yields of an IfConverter-created scf.if to its carrier result
// types. In particular, a yielded scalar pointer becomes its complete i64
// address. Yields belonging to ordinary ifs or loops are intentionally left to
// their owning conversions.
class YieldConverter : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename LoopOpTy,
          typename = std::enable_if_t<std::is_same_v<LoopOpTy, scf::ForOp> ||
                                      std::is_same_v<LoopOpTy, scf::WhileOp>>>
class LoopConverter : public OpConversionPattern<LoopOpTy> {
public:
  using OpConversionPattern<LoopOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOpTy op,
                  typename OpConversionPattern<LoopOpTy>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // CFO-expanded descriptor loops already carry pointer-free policy values
    // and remain structurally unchanged. This legacy BlockData rewrite is only
    // valid for explicitly marked legacy loops.
    if (!op->hasAttr("UnhandledLoopOp"))
      return failure();
    llvm::SmallDenseMap<Value, BlockData> known;

    rewriter.modifyOpInPlace(op, [&]() { op->removeAttr("UnhandledLoopOp"); });
    return BlockDataParser::rewriteLoopOp(op, rewriter, known, {});
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

struct DotConverter : public OpConversionPattern<triton::ascend::DotOp> {
  using OpConversionPattern<triton::ascend::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct FlipOpConverter : public OpConversionPattern<triton::ascend::FlipOp> {
  using OpConversionPattern<triton::ascend::FlipOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::FlipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

  static constexpr StringRef baseFuncName = "triton_flip";
};

struct SortOpConverter : public OpConversionPattern<triton::ascend::SortOp> {
  using OpConversionPattern<triton::ascend::SortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::SortOp op, OpAdaptor adaptor,
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

class IntToPtrConverter : public OpConversionPattern<triton::IntToPtrOp> {
public:
  using OpConversionPattern<triton::IntToPtrOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class IndexPutConverter
    : public OpConversionPattern<triton::ascend::IndexPutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::IndexPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_index_put";
};

class GatherOutToUbConverter
    : public OpConversionPattern<triton::ascend::GatherOutToUbOp> {
public:
  using OpConversionPattern<
      triton::ascend::GatherOutToUbOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::GatherOutToUbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton__gather_out_to_ub";
};

class ScatterUbToOutConverter
    : public OpConversionPattern<triton::ascend::ScatterUbToOutOp> {
public:
  using OpConversionPattern<
      triton::ascend::ScatterUbToOutOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::ScatterUbToOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_scatter_ub_to_out";
};

class IndirectLoadConverter
    : public OpConversionPattern<triton::ascend::IndirectLoadOp> {
public:
  using OpConversionPattern<
      triton::ascend::IndirectLoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::IndirectLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_indirect_load";
};

class StrideLoadConverter
    : public OpConversionPattern<triton::ascend::StrideLoadOp> {
public:
  using OpConversionPattern<triton::ascend::StrideLoadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::StrideLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_stride_load";
};

class StrideStoreConverter
    : public OpConversionPattern<triton::ascend::StrideStoreOp> {
public:
  using OpConversionPattern<triton::ascend::StrideStoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::StrideStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_stride_store";
};

class IndirectStoreConverter
    : public OpConversionPattern<triton::ascend::IndirectStoreOp> {
public:
  using OpConversionPattern<
      triton::ascend::IndirectStoreOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ascend::IndirectStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  static constexpr llvm::StringRef funcNameBase = "triton_indirect_store";
};

class IndexSelectSimdConverter
    : public OpConversionPattern<triton::ascend::IndexSelectSimdOp> {
public:
  explicit IndexSelectSimdConverter(MLIRContext *context);
  using OpConversionPattern<
      triton::ascend::IndexSelectSimdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ascend::IndexSelectSimdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

class HistogramConverter : public OpConversionPattern<triton::HistogramOp> {
public:
  using OpConversionPattern<triton::HistogramOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // end of namespace TTOpConverters

#endif
