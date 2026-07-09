#include "RPUPlan/IR/Dialect.h"
#include "RPUPlanModel.h"
#include "RPUTTIRPatternMatcher.h"
#include "RPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace rpu {
namespace {

DictionaryAttr
mapStringIntToDictAttr(MLIRContext *ctx,
                       const std::map<std::string, int64_t> &map) {
  Builder builder(ctx);
  NamedAttrList attrs;
  for (const auto &[key, value] : map)
    attrs.append(builder.getNamedAttr(key, builder.getI64IntegerAttr(value)));
  return attrs.getDictionary(ctx);
}

Attribute jsonValueToAttr(MLIRContext *ctx, const llvm::json::Value &value);

DictionaryAttr jsonObjectToDictAttr(MLIRContext *ctx,
                                    const llvm::json::Object &object) {
  Builder builder(ctx);
  NamedAttrList attrs;
  for (const auto &[key, value] : object) {
    Attribute attr = jsonValueToAttr(ctx, value);
    if (!attr)
      return nullptr;
    attrs.append(builder.getNamedAttr(key, attr));
  }
  return attrs.getDictionary(ctx);
}

Attribute jsonValueToAttr(MLIRContext *ctx, const llvm::json::Value &value) {
  Builder builder(ctx);
  if (auto stringValue = value.getAsString())
    return builder.getStringAttr(*stringValue);
  if (auto integerValue = value.getAsInteger())
    return builder.getI64IntegerAttr(*integerValue);
  if (auto boolValue = value.getAsBoolean())
    return builder.getBoolAttr(*boolValue);
  if (auto arrayValue = value.getAsArray()) {
    SmallVector<Attribute> attrs;
    for (const llvm::json::Value &entry : *arrayValue) {
      Attribute attr = jsonValueToAttr(ctx, entry);
      if (!attr)
        return nullptr;
      attrs.push_back(attr);
    }
    return ArrayAttr::get(ctx, attrs);
  }
  if (auto objectValue = value.getAsObject())
    return jsonObjectToDictAttr(ctx, *objectValue);
  return nullptr;
}

ArrayAttr signatureParamsToArrayAttr(MLIRContext *ctx,
                                     ArrayRef<llvm::json::Object> params) {
  SmallVector<Attribute> attrs;
  for (const llvm::json::Object &param : params) {
    DictionaryAttr attr = jsonObjectToDictAttr(ctx, param);
    if (!attr)
      return nullptr;
    attrs.push_back(attr);
  }
  return ArrayAttr::get(ctx, attrs);
}

DictionaryAttr signatureToDictAttr(MLIRContext *ctx, const RPUPlan &plan) {
  Builder builder(ctx);
  ArrayAttr params = signatureParamsToArrayAttr(ctx, plan.signatureParams);
  if (!params)
    return nullptr;
  NamedAttrList attrs;
  attrs.append(builder.getNamedAttr("params", params));
  attrs.append(builder.getNamedAttr("return_type",
                                    builder.getStringAttr(plan.returnType)));
  return attrs.getDictionary(ctx);
}

ArrayAttr requiredDslFeaturesToArrayAttr(MLIRContext *ctx,
                                         ArrayRef<std::string> features) {
  Builder builder(ctx);
  SmallVector<Attribute> attrs;
  for (StringRef feature : features)
    attrs.push_back(builder.getStringAttr(feature));
  return ArrayAttr::get(ctx, attrs);
}

FailureOr<plan::KernelOp> createRPUPlanKernelOp(ModuleOp module,
                                                triton::FuncOp source,
                                                const RPUPlan &plan) {
  MLIRContext *ctx = module.getContext();

  std::string planSymbol = (Twine(plan.kernelName) + "_plan").str();
  if (SymbolTable::lookupSymbolIn(module.getOperation(), planSymbol)) {
    module.emitError() << "RPUPlan symbol already exists: " << planSymbol;
    return failure();
  }

  DictionaryAttr signature = signatureToDictAttr(ctx, plan);
  DictionaryAttr layout = jsonObjectToDictAttr(ctx, plan.layout);
  DictionaryAttr mask = jsonObjectToDictAttr(ctx, plan.mask);
  DictionaryAttr emission = jsonObjectToDictAttr(ctx, plan.emission);
  if (!signature || !layout || !mask || !emission) {
    module.emitError(
        "failed to convert RPUPlan JSON payload to MLIR attributes");
    return failure();
  }

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());
  auto op = builder.create<plan::KernelOp>(
      source.getLoc(), builder.getStringAttr(planSymbol),
      builder.getStringAttr(plan.kernelName),
      FlatSymbolRefAttr::get(ctx, source.getSymName()),
      builder.getI32IntegerAttr(plan.version),
      builder.getStringAttr(plan.pattern), signature,
      mapStringIntToDictAttr(ctx, plan.shape),
      mapStringIntToDictAttr(ctx, plan.args), layout, mask,
      requiredDslFeaturesToArrayAttr(ctx, plan.requiredDslFeatures), emission);
  return op;
}

std::string serializePlanTraceToJson(std::optional<StringRef> selected,
                                     StringRef functionName,
                                     StringRef functionLocation,
                                     ArrayRef<TraceAttempt> attempts) {
  llvm::json::Array attemptObjects;
  for (const TraceAttempt &attempt : attempts) {
    attemptObjects.push_back(llvm::json::Object{
        {"anchor",
         llvm::json::Object{
             {"kind", attempt.anchor.kind},
             {"location", attempt.anchor.location},
             {"op", attempt.anchor.op},
         }},
        {"location", attempt.anchor.location},
        {"pattern", attempt.pattern},
        {"reason", attempt.reason},
        {"status", attempt.status},
    });
  }

  llvm::json::Object root;
  root["attempts"] = std::move(attemptObjects);
  root["function"] = llvm::json::Object{
      {"location", functionLocation.str()},
      {"name", functionName.str()},
  };
  root["matched"] = selected.has_value();
  root["selected"] = selected ? llvm::json::Value(selected->str())
                              : llvm::json::Value(nullptr);
  root["version"] = 1;
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
}

void setPlanTraceAttr(ModuleOp module, std::optional<StringRef> selected,
                      StringRef functionName, StringRef functionLocation,
                      ArrayRef<TraceAttempt> attempts) {
  module->setAttr(
      "rpu.plan.trace",
      StringAttr::get(module.getContext(),
                      serializePlanTraceToJson(selected, functionName,
                                               functionLocation, attempts)));
}

LogicalResult emitPlanOpJsonAndTrace(ModuleOp module, triton::FuncOp source,
                                     const RPUPlan &plan, StringRef selected,
                                     StringRef functionName,
                                     StringRef functionLocation,
                                     ArrayRef<TraceAttempt> attempts) {
  setPlanTraceAttr(module, selected, functionName, functionLocation, attempts);

  // JSON is a stable artifact exported from rpu_plan.kernel, not a recognizer
  // output path. Keep op creation before JSON export so the dialect op remains
  // the internal source of truth.
  FailureOr<plan::KernelOp> op = createRPUPlanKernelOp(module, source, plan);
  if (failed(op))
    return failure();

  std::optional<std::string> planJson = serializeRPUPlanKernelOpToJson(*op);
  if (!planJson) {
    module.emitError("failed to export rpu_plan.kernel to rpu.plan.json");
    return failure();
  }

  module->setAttr("rpu.plan.json",
                  StringAttr::get(module.getContext(), *planJson));
  return success();
}

RPUPlan makeAddPlan(triton::FuncOp func, const AddOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "add";
  plan.shape = {{"n", operands.n}, {"logical_n", operands.logicalN}};
  plan.args = {
      {"out", operands.out},
      {"lhs", operands.lhs},
      {"rhs", operands.rhs},
  };
  plan.emission = llvm::json::Object{
      {"kind", "add"},
      {"n", operands.n},
      {"logical_n", operands.logicalN},
      {"masked", operands.masked},
      {"out", operands.out},
      {"lhs", operands.lhs},
      {"rhs", operands.rhs},
  };

  FunctionType functionType = func.getFunctionType();
  for (auto [index, type] : llvm::enumerate(functionType.getInputs())) {
    (void)type;
    plan.signatureParams.push_back(llvm::json::Object{
        {"index", static_cast<int64_t>(index)},
        {"name", ("arg" + Twine(index)).str()},
        {"kind", "ptr"},
        {"element_type", "f16"},
    });
  }

  if (operands.masked) {
    plan.layout = llvm::json::Object{
        {"memory", "tensor"},
        {"access", "masked_tile_view"},
        {"order", "row_major"},
    };
    plan.mask = llvm::json::Object{
        {"masked", true},
        {"logical_n", operands.logicalN},
        {"block_n", operands.n},
    };
    plan.requiredDslFeatures = {"rpu.make_tensor", "rpu.local_tile", "ctx.load",
                                "tile.add", "ctx.store"};
  } else if (operands.n <= kAddMaxContiguousNVec) {
    plan.layout = llvm::json::Object{
        {"memory", "contiguous_vector"},
        {"access", "linear"},
    };
    plan.mask = llvm::json::Object{{"masked", false}};
    plan.requiredDslFeatures = {"ctx.load_contig", "tile.add",
                                "ctx.store_contig"};
  } else {
    plan.layout = llvm::json::Object{
        {"memory", "tensor"},
        {"access", "chunked_tile_view"},
        {"order", "row_major"},
        {"max_rows_per_frame", kAddMaxTensorRows},
    };
    plan.mask = llvm::json::Object{{"masked", false}};
    plan.requiredDslFeatures = {"rpu.make_tensor", "rpu.local_tile",
                                "ctx.tile_frame",  "ctx.load",
                                "tile.add",        "ctx.store"};
  }

  return plan;
}

void addF16PtrSignature(RPUPlan &plan, triton::FuncOp func) {
  FunctionType functionType = func.getFunctionType();
  for (auto [index, type] : llvm::enumerate(functionType.getInputs())) {
    (void)type;
    plan.signatureParams.push_back(llvm::json::Object{
        {"index", static_cast<int64_t>(index)},
        {"name", ("arg" + Twine(index)).str()},
        {"kind", "ptr"},
        {"element_type", "f16"},
    });
  }
}

RPUPlan makeGemmPlan(triton::FuncOp func, const GemmOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "gemm";
  plan.shape = {
      {"m", operands.m},
      {"n", operands.n},
      {"k", operands.k},
  };
  plan.args = {
      {"out", operands.out},
      {"lhs", operands.lhs},
      {"rhs", operands.rhs},
  };
  plan.layout = llvm::json::Object{
      {"memory", "array2d"},
      {"access", "matrix_tile"},
      {"order", "row_major"},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma",
                              "ctx.store"};
  plan.emission = llvm::json::Object{
      {"kind", "gemm"},      {"m", operands.m},     {"n", operands.n},
      {"k", operands.k},     {"out", operands.out}, {"lhs", operands.lhs},
      {"rhs", operands.rhs},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeSoftmaxPlan(triton::FuncOp func, const SoftmaxOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "softmax";
  plan.shape = {{"n", operands.n}};
  plan.args = {
      {"out", operands.out},
      {"input", operands.input},
  };
  plan.layout = llvm::json::Object{
      {"memory", "contiguous_vector"},
      {"access", "linear"},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"ctx.load_contig", "ctx.reduce_max_all",
                              "rpu.exp",         "ctx.reduce_sum_all",
                              "rpu.reciprocal",  "ctx.store_contig"};
  plan.emission = llvm::json::Object{
      {"kind", "softmax"},
      {"n", operands.n},
      {"out", operands.out},
      {"input", operands.input},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeSqrtPlan(triton::FuncOp func, const SqrtOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "sqrt";
  plan.shape = {{"n", operands.n}};
  plan.args = {
      {"out", operands.out},
      {"input", operands.input},
  };
  plan.layout = llvm::json::Object{
      {"memory", "contiguous_vector"},
      {"access", "linear"},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"ctx.load_contig", "rpu.sqrt",
                              "ctx.store_contig"};
  plan.emission = llvm::json::Object{
      {"kind", "sqrt"},
      {"n", operands.n},
      {"out", operands.out},
      {"input", operands.input},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeReduceSumAllPlan(triton::FuncOp func,
                             const ReduceSumAllOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "reduce_sum_all";
  plan.shape = {{"n", operands.n}};
  plan.args = {
      {"out", operands.out},
      {"input", operands.input},
  };
  plan.layout = llvm::json::Object{
      {"memory", "contiguous_vector"},
      {"access", "linear"},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"ctx.load_contig", "ctx.reduce_sum_all",
                              "ctx.full", "ctx.store_contig"};
  plan.emission = llvm::json::Object{
      {"kind", "reduce_sum_all"},
      {"n", operands.n},
      {"out", operands.out},
      {"input", operands.input},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeResNetBlockPlan(triton::FuncOp func,
                            const ResNetBlockOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "resnet_block";
  plan.shape = {
      {"m", operands.m},
      {"channels", operands.channels},
      {"hidden", operands.hidden},
  };
  plan.args = {
      {"out", operands.out},
      {"x", operands.x},
      {"w1", operands.w1},
      {"w2", operands.w2},
  };
  plan.layout = llvm::json::Object{
      {"memory", "array2d"},
      {"access", "matrix_tile"},
      {"order", "row_major"},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"rpu.Array", "ctx.load", "ctx.zeros",
                              "ctx.mma",   "tile.add", "rpu.max_binop",
                              "ctx.store"};
  plan.emission = llvm::json::Object{
      {"kind", "resnet_block"},
      {"m", operands.m},
      {"channels", operands.channels},
      {"hidden", operands.hidden},
      {"out", operands.out},
      {"x", operands.x},
      {"w1", operands.w1},
      {"w2", operands.w2},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeConvKxKPlan(triton::FuncOp func, const ConvKxKOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "convkxk";
  plan.shape = {
      {"kernel_size", operands.kernelSize},
      {"m", operands.m},
      {"in_channels", operands.inChannels},
      {"out_channels", operands.outChannels},
      {"input_width", operands.inputWidth},
  };
  plan.args = {
      {"out", operands.out},
      {"input", operands.input},
      {"weight", operands.weight},
  };
  plan.layout = llvm::json::Object{
      {"memory", "array2d"},
      {"access", "row_window"},
      {"order", "row_major"},
      {"window",
       llvm::json::Object{
           {"kernel_size", operands.kernelSize},
           {"input_width", operands.inputWidth},
           {"stride", llvm::json::Array{1, 1}},
           {"padding", llvm::json::Array{0, 0}},
       }},
      {"tile",
       llvm::json::Object{
           {"m", operands.m},
           {"n", operands.outChannels},
       }},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"rpu.Array", "ctx.load", "ctx.zeros", "ctx.mma",
                              "ctx.store"};
  plan.emission = llvm::json::Object{
      {"kind", "convkxk"},
      {"kernel_size", operands.kernelSize},
      {"m", operands.m},
      {"in_channels", operands.inChannels},
      {"out_channels", operands.outChannels},
      {"input_width", operands.inputWidth},
      {"out", operands.out},
      {"input", operands.input},
      {"weight", operands.weight},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

RPUPlan makeResNet50BottleneckPlan(triton::FuncOp func,
                                   const ResNet50BottleneckOperands &operands) {
  RPUPlan plan;
  plan.kernelName = func.getSymName().str();
  plan.pattern = "resnet50_bottleneck";
  plan.shape = {
      {"kernel_size", operands.kernelSize}, {"m", operands.m},
      {"channels", operands.channels},      {"bottleneck", operands.bottleneck},
      {"input_width", operands.inputWidth},
  };
  plan.args = {
      {"out", operands.out}, {"input", operands.input}, {"w1", operands.w1},
      {"w2", operands.w2},   {"w3", operands.w3},
  };
  plan.layout = llvm::json::Object{
      {"memory", "array2d"},
      {"access", "bottleneck_row_window"},
      {"order", "row_major"},
      {"window",
       llvm::json::Object{
           {"kernel_size", operands.kernelSize},
           {"input_width", operands.inputWidth},
           {"stride", llvm::json::Array{1, 1}},
           {"padding", llvm::json::Array{0, 0}},
       }},
      {"tile",
       llvm::json::Object{
           {"m", operands.m},
           {"n", operands.channels},
       }},
  };
  plan.mask = llvm::json::Object{{"masked", false}};
  plan.requiredDslFeatures = {"rpu.Array", "ctx.load", "ctx.zeros",
                              "ctx.mma",   "tile.add", "rpu.max_binop",
                              "ctx.store"};
  plan.emission = llvm::json::Object{
      {"kind", "resnet50_bottleneck"},
      {"kernel_size", operands.kernelSize},
      {"m", operands.m},
      {"channels", operands.channels},
      {"bottleneck", operands.bottleneck},
      {"input_width", operands.inputWidth},
      {"out", operands.out},
      {"input", operands.input},
      {"w1", operands.w1},
      {"w2", operands.w2},
      {"w3", operands.w3},
  };
  addF16PtrSignature(plan, func);
  return plan;
}

class RecognizeRPUPlanPass
    : public PassWrapper<RecognizeRPUPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecognizeRPUPlanPass)

  StringRef getArgument() const final { return "rpu-recognize-plan"; }
  StringRef getDescription() const final {
    return "recognize RPU-supported TTIR patterns into RPUPlan";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<plan::RPUPlanDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    SmallVector<TraceAttempt, 6> attempts;
    FailureOr<triton::FuncOp> maybeFunc = getSinglePublicFunc(module);
    if (failed(maybeFunc)) {
      TraceAnchor anchor = moduleAnchor(module);
      appendAllFailedAttempts(attempts, kSinglePublicFuncReason, anchor);
      setPlanTraceAttr(module, std::nullopt, "<module>", anchor.location,
                       attempts);
      module.emitError(
          "Unsupported RPU TTIR pattern. add lowering: C++ recognizer "
          "expects exactly one public tt.func; GEMM lowering: C++ "
          "recognizer expects exactly one public tt.func; softmax lowering: "
          "C++ recognizer expects exactly one public tt.func; resnet block "
          "lowering: C++ recognizer expects exactly one public tt.func; KxK "
          "conv lowering: C++ recognizer expects exactly one public tt.func; "
          "ResNet50 bottleneck lowering: C++ recognizer expects exactly one "
          "public tt.func");
      signalPassFailure();
      return;
    }

    std::string functionName = maybeFunc->getSymName().str();
    std::string functionLocation = stringifyLocation(maybeFunc->getLoc());

    if (FailureOr<AddOperands> operands = recognizeAdd(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeAddPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kAddPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan, kAddPattern,
                                        functionName, functionLocation,
                                        attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kAddPattern, kAddFailureReason,
                                     addFailureAnchor(*maybeFunc)));

    if (FailureOr<GemmOperands> operands = recognizeGemm(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeGemmPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kGemmPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan, kGemmPattern,
                                        functionName, functionLocation,
                                        attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kGemmPattern, kGemmFailureReason,
                                     gemmFailureAnchor(*maybeFunc)));

    if (FailureOr<SoftmaxOperands> operands = recognizeSoftmax(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeSoftmaxPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kSoftmaxPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan,
                                        kSoftmaxPattern, functionName,
                                        functionLocation, attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kSoftmaxPattern, kSoftmaxFailureReason,
                                     softmaxFailureAnchor(*maybeFunc)));

    if (FailureOr<SqrtOperands> operands = recognizeSqrt(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeSqrtPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kSqrtPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan, kSqrtPattern,
                                        functionName, functionLocation,
                                        attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kSqrtPattern, kSqrtFailureReason,
                                     sqrtFailureAnchor(*maybeFunc)));

    if (FailureOr<ReduceSumAllOperands> operands =
            recognizeReduceSumAll(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeReduceSumAllPlan(*maybeFunc, *operands);
      attempts.push_back(
          matchedAttempt(kReduceSumAllPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan,
                                        kReduceSumAllPattern, functionName,
                                        functionLocation, attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kReduceSumAllPattern,
                                     kReduceSumAllFailureReason,
                                     reduceSumAllFailureAnchor(*maybeFunc)));

    if (FailureOr<ResNetBlockOperands> operands =
            recognizeResNetBlock(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeResNetBlockPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kResNetBlockPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan,
                                        kResNetBlockPattern, functionName,
                                        functionLocation, attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kResNetBlockPattern,
                                     kResNetBlockFailureReason,
                                     resNetBlockFailureAnchor(*maybeFunc)));

    if (FailureOr<ConvKxKOperands> operands = recognizeConvKxK(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeConvKxKPlan(*maybeFunc, *operands);
      attempts.push_back(matchedAttempt(kConvKxKPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(module, *maybeFunc, plan,
                                        kConvKxKPattern, functionName,
                                        functionLocation, attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(kConvKxKPattern, kConvKxKFailureReason,
                                     convKxKFailureAnchor(*maybeFunc)));

    if (FailureOr<ResNet50BottleneckOperands> operands =
            recognizeResNet50Bottleneck(*maybeFunc);
        succeeded(operands)) {
      RPUPlan plan = makeResNet50BottleneckPlan(*maybeFunc, *operands);
      attempts.push_back(
          matchedAttempt(kResNet50BottleneckPattern, operands->anchor));
      if (failed(emitPlanOpJsonAndTrace(
              module, *maybeFunc, plan, kResNet50BottleneckPattern,
              functionName, functionLocation, attempts))) {
        signalPassFailure();
        return;
      }
      return;
    }
    attempts.push_back(failedAttempt(
        kResNet50BottleneckPattern, kResNet50BottleneckFailureReason,
        resNet50BottleneckFailureAnchor(*maybeFunc)));

    setPlanTraceAttr(module, std::nullopt, functionName, functionLocation,
                     attempts);
    module.emitError(
        "Unsupported RPU TTIR pattern. add lowering: C++ recognizer did not "
        "match supported vector add; GEMM lowering: C++ recognizer did not "
        "match supported dot kernel; softmax lowering: C++ recognizer did not "
        "match supported vector softmax; resnet block lowering: C++ "
        "recognizer did not match supported residual block; KxK conv "
        "lowering: C++ recognizer did not match supported row-window "
        "convolution; ResNet50 bottleneck lowering: C++ recognizer did not "
        "match supported bottleneck block");
    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createRecognizeRPUPlanPass() {
  return std::make_unique<RecognizeRPUPlanPass>();
}

} // namespace rpu
} // namespace mlir
