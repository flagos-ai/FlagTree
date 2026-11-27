#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Conversion/UnstructuredToMK/UnstructuredToMK.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "utils/TypeConvertor.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#define DEBUG_TYPE "unstructured-to-memref"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_UNSTRUCTUREDTOMK
#include "triton-shared/Conversion/UnstructuredToMK/Passes.h.inc"

namespace {

struct ScalarAtomicRMWOpConverter
    : public OpConversionPattern<tts::IndexedAtomicRMWOp> {
  using OpConversionPattern<tts::IndexedAtomicRMWOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::IndexedAtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getValue();
    if (isa<RankedTensorType>(value.getType())) {
      return failure();
    }

    auto loc = op->getLoc();

    // Calculate the ptr from the offset
    auto ptr = adaptor.getPtr();
    auto offset = adaptor.getOffset();
    auto index = arith::IndexCastOp::create(rewriter, loc,
                                            rewriter.getIndexType(), offset);
    auto rankedMemref = memref::ReinterpretCastOp::create(
        rewriter, loc, ptr, getAsOpFoldResult(index),
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto inputTensorType =
        RankedTensorType::get(SmallVector<int64_t>(1, 1), value.getType());

    auto empty =
        tensor::EmptyOp::create(rewriter, loc, inputTensorType.getShape(),
                                inputTensorType.getElementType());
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto valueTensor = tensor::InsertOp::create(rewriter, loc, inputTensorType,
                                                value, empty, ValueRange{zero});

    auto init =
        tensor::EmptyOp::create(rewriter, loc, inputTensorType.getShape(),
                                inputTensorType.getElementType());
    if (op.getMask()) {
      // If there is a mask, we need to check it before performing the
      // atomic RMW operation.
      auto mask = op.getMask();
      auto ifOp = scf::IfOp::create(
          rewriter, loc, mask,
          [&](OpBuilder &b, Location loc) {
            // TODO: Support other types of inputs and outputs: f32.
            auto atomic = mk::AtomicRMWOp::create(
                              rewriter, loc, inputTensorType, rankedMemref,
                              valueTensor, init, op.getAtomicRmwOpAttr(),
                              op.getSemAttr(), op.getScopeAttr())
                              ->getResult(0);

            auto resultValue = tensor::ExtractOp::create(rewriter, loc, atomic,
                                                         ValueRange{zero});

            scf::YieldOp::create(b, loc, resultValue.getResult());
          },
          [&](OpBuilder &b, Location loc) {
            // else branch
            Value zero =
                arith::ConstantOp::create(b, loc, b.getZeroAttr(op.getType()));
            scf::YieldOp::create(b, loc, zero);
          });

      rewriter.replaceOp(op, ifOp);
    } else {

      auto atomic =
          mk::AtomicRMWOp::create(rewriter, loc, inputTensorType, rankedMemref,
                                  valueTensor, init, op.getAtomicRmwOpAttr(),
                                  op.getSemAttr(), op.getScopeAttr())
              ->getResult(0);

      auto resultValue =
          tensor::ExtractOp::create(rewriter, loc, atomic, ValueRange{zero});

      rewriter.replaceOp(op, resultValue.getResult());
    }

    return success();
  }
};

struct ScalarAtomicCASOpConverter
    : public OpConversionPattern<tts::AtomicCASOp> {
  using OpConversionPattern<tts::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tts::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getOffset())
      return failure();

    if (!op.getType().isIntOrIndexOrFloat()) {
      return failure();
    }

    auto loc = op->getLoc();
    auto ptr = adaptor.getPtr();
    auto cmp = adaptor.getCmp();
    auto value = adaptor.getValue();

    if (isa<RankedTensorType>(value.getType())) {
      return failure();
    }

    // Calculate the ptr from the offset
    auto offset = adaptor.getOffset();
    auto index = arith::IndexCastOp::create(rewriter, loc,
                                            rewriter.getIndexType(), offset);
    auto rankedMemref = memref::ReinterpretCastOp::create(
        rewriter, loc, ptr, getAsOpFoldResult(index),
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*sizes*/,
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)} /*strides*/);

    auto inputTensorType =
        RankedTensorType::get(SmallVector<int64_t>(1, 1), value.getType());

    auto empty =
        tensor::EmptyOp::create(rewriter, loc, inputTensorType.getShape(),
                                inputTensorType.getElementType());
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto valueTensor = tensor::InsertOp::create(rewriter, loc, inputTensorType,
                                                value, empty, ValueRange{zero});
    auto cmpTensor = tensor::InsertOp::create(rewriter, loc, inputTensorType,
                                              cmp, empty, ValueRange{zero});

    auto init =
        tensor::EmptyOp::create(rewriter, loc, inputTensorType.getShape(),
                                inputTensorType.getElementType());

    // TODO: Support other types of inputs and outputs: f32.
    auto atomic = mk::AtomicCASOp::create(
                      rewriter, loc, inputTensorType, rankedMemref, cmpTensor,
                      valueTensor, init, op.getSemAttr(), op.getScopeAttr())
                      ->getResult(0);

    auto resultValue =
        tensor::ExtractOp::create(rewriter, loc, atomic, ValueRange{zero});

    rewriter.replaceOp(op, resultValue.getResult());

    return success();
  }
};

template <typename TTS_AtomicOp>
struct IndexedAtomicOpConverter : public OpConversionPattern<TTS_AtomicOp> {
  using OpConversionPattern<TTS_AtomicOp>::OpConversionPattern;
  using OpAdaptor = typename TTS_AtomicOp::Adaptor;

  LogicalResult
  matchAndRewrite(TTS_AtomicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType) {
      return failure();
    }

    SmallVector<Value> inputs(op->getOperands().begin() + 1,
                              op->getOperands().end());

    SmallVector<Value> outputs = {tensor::EmptyOp::create(
        rewriter, op->getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType())};
    assert(op->getResultTypes().size() == 1);

    auto scalarResultType =
        cast<TensorType>(op->getResultTypes().front()).getElementType();

    // NOTE: linalg.generic cannot nested with linalg.generic (mk.atomic will
    // generate linalg.generic inside), so we need to use scf.for to build the
    // loop
    auto shape = resultTensorType.getShape();
    auto loc = op->getLoc();
    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<Value> lbs, ubs, steps;
    for (auto [i, size] : enumerate(shape)) {
      auto sizeValue = arith::ConstantIndexOp::create(rewriter, loc, size);
      lbs.push_back(zero);
      ubs.push_back(sizeValue);
      steps.push_back(one);
    }
    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps, outputs,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange indices,
            ValueRange iterArgs) {
          SmallVector<Value> regionInputs(op.getNumOperands());
          regionInputs.front() = adaptor.getPtr();
          std::transform(inputs.begin(), inputs.end(), regionInputs.begin() + 1,
                         [&](auto val) {
                           return tensor::ExtractOp::create(rewriter, loc, val,
                                                            indices);
                         });
          auto *scalarOp = nestedBuilder.create(
              loc, op->getName().getIdentifier(), regionInputs,
              scalarResultType, op->getAttrs());

          auto outValTensor = tensor::InsertOp::create(
              nestedBuilder, loc, scalarOp->getResult(0), iterArgs[0], indices);
          return SmallVector<Value>{outValTensor};
        });

    rewriter.replaceOp(op, loopNest.results);

    return success();
  }
};

class UnstructuredToMKPass
    : public ::impl::UnstructuredToMKBase<UnstructuredToMKPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    ttx::TritonTilingExtDialect, mk::MagicKernelDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, memref::MemRefDialect,
        ttx::TritonTilingExtDialect, mk::MagicKernelDialect>();

    target.addIllegalOp<tts::GatherOp, tts::ScatterOp, tts::IndexedAtomicRMWOp,
                        tts::AtomicCASOp>();

    PtrToUnrankedMemrefConverter typeConverter;

    patterns.add<ScalarAtomicRMWOpConverter,
                 IndexedAtomicOpConverter<tts::IndexedAtomicRMWOp>,
                 ScalarAtomicCASOpConverter,
                 IndexedAtomicOpConverter<tts::AtomicCASOp>>(
        typeConverter, patterns.getContext());

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createUnstructuredToMKPass() {
  return std::make_unique<UnstructuredToMKPass>();
}
