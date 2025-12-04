//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/AnalysisStructured/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cassert>
#include <optional>

#define DEBUG_TYPE "triton-to-structured"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToStructured/Passes.h.inc"

namespace {

class TritonToStructuredPass
    : public TritonToStructuredBase<TritonToStructuredPass> {

  static TupleType getStructuredStateTupleType(MLIRContext *context, Type t) {
    SmallVector<Type> tupleTypes{t};
    auto [offsetTypes, strideTypes] =
        *tts::GetStructuredStateOp::getOffsetAndStrideTypes(context, t);
    tupleTypes.append(offsetTypes);
    tupleTypes.append(strideTypes);
    return TupleType::get(context, tupleTypes);
  }

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  LogicalResult convertToPointerTupleWithOffsetsAndStrides() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());

    auto context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->1 type conversion here, where a triton pointer type
    // maps to a tuple of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...} type.
    //
    // Case 1: Unstructured pointers (tensor<!tt.ptr<type>>)
    converter.addConversion([context](RankedTensorType tensorType,
                                      SmallVectorImpl<Type> &types)
                                -> std::optional<LogicalResult> {
      // Important note:
      // We only care about tensor of index / int (in addition to pointer
      // type) because only values of int and index type can potentially be
      // part of a pointer arithmetic sequence.
      // FIXME: Integer type may not always involved in pointer arithmetic
      if (!isa<triton::PointerType>(tensorType.getElementType()) &&
          !tensorType.getElementType().isIntOrIndex()) {
        // There's a subtle difference between returning failure() and
        // std::nullopt. From the documentation:
        //
        // If std::nullopt is returned, the converter is allowed to try another
        // conversion function to perform the conversion.
        //
        // Say we have type tensor<4x256xbf16> which is a RankedTensorType. Even
        // though this RankedTensorType matches the converter that handles the
        // tuple conversion, we want to keep this type as is because the inner
        // type isn't a pointer.
        //
        // By returning failure(), the TypeConverters will stop trying the
        // remaining converters. In our case, the last type converter which
        // simply returns the same type is skipped. And because the conversion
        // for this type has failed, the whole conversion process is also
        // skipped.
        //
        // Relevant links to the implementation:
        //
        // https://github.com/llvm/llvm-project/blob/cb5dc1faa8b3702e0d03426ee5dfc5e1b903ec47/mlir/lib/Transforms/Utils/DialectConversion.cpp#L2958
        // https://github.com/llvm/llvm-project/blob/cb5dc1faa8b3702e0d03426ee5dfc5e1b903ec47/mlir/lib/Transforms/Utils/DialectConversion.cpp#L3033
        return std::nullopt;
      }
      types =
          SmallVector<Type>{getStructuredStateTupleType(context, tensorType)};
      return success();
    });

    // Case 2: Block pointers (!tt.ptr<tensor<type>> or !tt.ptr<type>)
    converter.addConversion([context](triton::PointerType ptrType,
                                      SmallVectorImpl<Type> &types)
                                -> std::optional<LogicalResult> {
      types = SmallVector<Type>{getStructuredStateTupleType(context, ptrType)};
      return success();
    });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert the tuple type back to
    // the original triton pointer type. These are used when there are ops that
    // still need to use the original pointer type. For instance, we convert the
    // result of tt.addptr from tt.ptr type to a tuple, but the original ptr
    // result is still being used by another tt.load or tt.store.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs, Location loc) {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };

    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // Compute the target materialization, given a value with the pointer type,
    // convert that value to a tuple type.
    converter.addTargetMaterialization([](OpBuilder &builder,
                                          TypeRange resultTypes,
                                          ValueRange inputs,
                                          Location loc) -> SmallVector<Value> {
      return builder
          .create<UnrealizedConversionCastOp>(loc, resultTypes, inputs.front())
          ->getResults();
    });

    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createCanonicalizerPass());
    if (failed(runPipeline(pm, getOperation()))) {
      return failure();
    }

    return success();
  }

  LogicalResult decomposePointerTuple() {
    auto moduleOp = getOperation();

    auto context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });

    // We are doing a 1->N type conversion here, where a pointer tuple type
    // maps to a sequence of {pointer, offset_0, offset_1,..., stride_0,
    // stride_1,...}
    converter.addConversion(
        [context](TupleType tupleType, SmallVectorImpl<Type> &types)
            -> std::optional<LogicalResult> {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    // Hooks to compute the correct materialization, "argument" and "source"
    // materialization are used when we need to convert a series of {pointer,
    // offset_0, offset_1,..., stride_0, stride_1,...} type back to the "pointer
    // tuple type".
    //
    // Because we actually want to get rid of the tuple type, return `inputs[0]`
    // which corresponds to a "triton pointer type". This approach will work as
    // intended because the ops that currently take "pointer tuple type" are
    // `unrealized_conversion_cast` ops which will get removed below during
    // reconcile-unrealized-conversion-casts.
    auto materialize = [](OpBuilder &builder, Type resultType,
                          ValueRange inputs,
                          Location loc) { return inputs[0]; };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);

    // For each value of "pointer tuple type" that gets decomposed into a
    // sequence of {pointer, offset_0, offset_1,..., stride_0, stride_1,...},
    // create a `tts.get_structured_state` op that serves as a placeholder.
    // The return values for this op will be used as the init-args for scf.for.
    // At the end of pointer analysis, we will use the PtrState to create the
    // correct offsets, strides, and remove these ops.
    converter.addTargetMaterialization([](OpBuilder &builder,
                                          TypeRange resultTypes,
                                          ValueRange inputs, Location loc) {
      auto placeholder = builder.create<tts::GetStructuredStateOp>(
          loc, inputs.front().getDefiningOp()->getOperand(0));
      assert(llvm::equal(placeholder.getResultTypes(), resultTypes));
      return placeholder.getResults();
    });

    RewritePatternSet patterns(&getContext());
    scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);
    if (failed(applyPartialOneToNConversion(getOperation(), converter,
                                            std::move(patterns)))) {
      return failure();
    }

    // Note:
    // Be careful not to run canonicalization here, because the
    // tts.get_structured_state ops created above are just placeholders and
    // don't have any effects. Canonicalization will remove them altogether.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    return success();
  }

  // Prepass that inserts `tts.get_structured_state` ops. These ops are used as
  // placeholders to make passing structured pointer state into scf.for loop's
  // init args easier, especially with multiple levels of loops.
  //
  // Background:
  //
  // PtrAnalysis computes a PtrState for every operand (or triton value)
  // involved in a sequence of pointer arithmetic; some examples include: triton
  // pointer, offsets (which could be a tensor of indices or just a simple index
  // value).
  //
  // If a triton value is updated and returned in a scf.for op, it means
  // that we have to carry its offsets and strides in the scf.for's iterargs.
  //
  // Previously, we have to manually rewrite the loops to include the
  // relevant information from a PtrState which was rather involved and
  // error-prone; this was also hard to scale up to multiple level of loops
  // because there are several book-keeping data structures that we have to
  // maintain.
  //
  // With the introduction of the prepass that inserts
  // `tts.get_structured_state`. The return values of these ops, which include a
  // triton value with its original result type and its corresponding offsets
  // and strides, will be used as "placeholders" into the scf.for's init-args.
  // We leverage standard MLIR infrastructure 1->N conversion to perform this
  // rewrite, which helps simplify the logic significantly.
  //
  // After PtrAnalysis finishes, the return values of these
  // `tts.get_structured_state` ops will be remapped to the correct
  // initialization of the value's offsets and strides through the value's
  // computed PtrState.
  //
  // Implementation details:
  // In essence, what we really want to do in the prepass is, for every value
  // of triton-pointer-like type (tt.ptr or tensor<tt.ptr<>>) and tensor of
  // indices (tensor<i32>) which might be used in a sequence of pointer
  // arithmetic, we want to create an op `tts.get_structured_state` that takes
  // in the original triton value and returns a series of values:
  //
  // {triton_value, offset_0, offset_1, ..., stride_0, stride_1,...}
  //
  // Applying the above conversion will also mean that any structural ops such
  // as scf.for and scf.yield that originally takes the triton pointer will
  // then take {triton_value, offset_0, offset_1, ..., stride_0, stride_1,...}.
  //
  // The 1->N type conversion is a perfect fit for this transformation.
  // Unfortunately, we cannot do this is one pass, because the current 1->N
  // type conversion implementation for scf.for ops doesn't provide us with a
  // way to detect that a type conversion is recursive. So a triton_value type
  // that gets converted to a {triton_value, offset_0, offset_1, ..., stride_0,
  // stride_1,...} will recursively trigger other conversions.
  //
  // To fix this issue, we have to first convert triton_value to
  // tuple<triton_value, offset_0, offset_1, ..., stride_0, stride_1,...>.
  // Finally, we decompose these tuples into the desired sequence.
  //
  // Note that even though the type conversion happens for every integer tensor
  // appearing in loops' iter-args, this conversion is reversible. If the
  // integer tensor isn't used in a pointer arithmetic sequence,
  // canonicalization will remove all the `tts.get_structured_state` ops and
  // revert the IR back to its original form.
  LogicalResult runTritonToStructuredPrepass() {
    if (failed(convertToPointerTupleWithOffsetsAndStrides())) {
      return failure();
    }

    return decomposePointerTuple();
  }

  void runOnOperation() override {
    if (!skipPrepass && failed(runTritonToStructuredPrepass())) {
      signalPassFailure();
      return;
    }

    if (runPrepassOnly) {
      return;
    }

    auto moduleOp = getOperation();
    mlir::tts::PtrAnalysis ptrAnalysis;
    ptrAnalysis.initializeMaybeStructuredArgs(moduleOp);

    if (failed(ptrAnalysis.rewriteOp(moduleOp, useUnsafeMask))) {
      moduleOp->emitWarning("PtrAnalysis failed");
    }

    // Now that all the PtrStates have been populated, we can wire up the states
    // with the tts.get_structured_state ops inserted in the prepass.
    moduleOp.walk([&ptrAnalysis](tts::GetStructuredStateOp op) {
      if (failed(ptrAnalysis.rewriteGetStructuredStateOp(op))) {
        op.emitWarning("Rewriting GetStructuredStateOp failed.");
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToStructuredPass() {
  return std::make_unique<TritonToStructuredPass>();
}
