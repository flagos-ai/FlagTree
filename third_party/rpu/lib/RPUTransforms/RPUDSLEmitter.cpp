#include "RPUDSLEmitter.h"
#include "RPUDSLSourceBuilder.h"
#include "RPUPlan/IR/Dialect.h"
#include "RPUPlanModel.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace rpu {

std::vector<std::string> directRPUDSLSupportedPatterns() {
  return {"add",
          "gemm",
          "softmax",
          "sqrt",
          "reduce_sum_all",
          "convkxk",
          "resnet_block",
          "resnet50_bottleneck"};
}

bool isDirectRPUDSLSupportedPattern(llvm::StringRef pattern) {
  for (const std::string &supported : directRPUDSLSupportedPatterns()) {
    if (pattern == supported)
      return true;
  }
  return false;
}

static FailureOr<int64_t> getIntegerField(plan::KernelOp op,
                                          const llvm::json::Object &object,
                                          llvm::StringRef name) {
  std::optional<int64_t> value = object.getInteger(name);
  if (!value) {
    op.emitError("direct .rc emission requires integer emission field ")
        << name;
    return failure();
  }
  return *value;
}

static FailureOr<bool> getBooleanField(plan::KernelOp op,
                                       const llvm::json::Object &object,
                                       llvm::StringRef name) {
  std::optional<bool> value = object.getBoolean(name);
  if (!value) {
    op.emitError("direct .rc emission requires boolean emission field ")
        << name;
    return failure();
  }
  return *value;
}

static FailureOr<std::string> formatSignature(plan::KernelOp op,
                                              const RPUPlan &plan) {
  llvm::SmallVector<std::string> params;
  for (const llvm::json::Object &param : plan.signatureParams) {
    std::optional<int64_t> index = param.getInteger("index");
    std::optional<llvm::StringRef> name = param.getString("name");
    std::optional<llvm::StringRef> kind = param.getString("kind");
    std::optional<llvm::StringRef> elementType =
        param.getString("element_type");
    if (!index || !name || !kind || !elementType || *kind != "ptr" ||
        *elementType != "f16") {
      op.emitError("direct .rc emission only supports f16 pointer params");
      return failure();
    }
    params.push_back(llvm::formatv("half* {0}", *name).str());
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  llvm::interleaveComma(params, os);
  return os.str();
}

static FailureOr<std::string> emitAddBody(plan::KernelOp op,
                                          const RPUPlan &plan) {
  FailureOr<int64_t> n = getIntegerField(op, plan.emission, "n");
  FailureOr<int64_t> logicalN = getIntegerField(op, plan.emission, "logical_n");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  FailureOr<int64_t> lhs = getIntegerField(op, plan.emission, "lhs");
  FailureOr<int64_t> rhs = getIntegerField(op, plan.emission, "rhs");
  FailureOr<bool> masked = getBooleanField(op, plan.emission, "masked");
  if (failed(n) || failed(logicalN) || failed(out) || failed(lhs) ||
      failed(rhs) || failed(masked))
    return failure();

  return buildAddBody(*n, *logicalN, *out, *lhs, *rhs, *masked);
}

static FailureOr<std::string> emitGemmBody(plan::KernelOp op,
                                           const RPUPlan &plan) {
  FailureOr<int64_t> m = getIntegerField(op, plan.emission, "m");
  FailureOr<int64_t> n = getIntegerField(op, plan.emission, "n");
  FailureOr<int64_t> k = getIntegerField(op, plan.emission, "k");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  FailureOr<int64_t> lhs = getIntegerField(op, plan.emission, "lhs");
  FailureOr<int64_t> rhs = getIntegerField(op, plan.emission, "rhs");
  if (failed(m) || failed(n) || failed(k) || failed(out) || failed(lhs) ||
      failed(rhs))
    return failure();

  return buildGemmBody(*m, *n, *k, *out, *lhs, *rhs);
}

static FailureOr<std::string> emitSoftmaxBody(plan::KernelOp op,
                                              const RPUPlan &plan) {
  FailureOr<int64_t> n = getIntegerField(op, plan.emission, "n");
  FailureOr<int64_t> input = getIntegerField(op, plan.emission, "input");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  if (failed(n) || failed(input) || failed(out))
    return failure();

  return buildSoftmaxBody(*n, *input, *out);
}

static FailureOr<std::string> emitSqrtBody(plan::KernelOp op,
                                           const RPUPlan &plan) {
  FailureOr<int64_t> n = getIntegerField(op, plan.emission, "n");
  FailureOr<int64_t> input = getIntegerField(op, plan.emission, "input");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  if (failed(n) || failed(input) || failed(out))
    return failure();

  return buildSqrtBody(*n, *input, *out);
}

static FailureOr<std::string> emitReduceSumAllBody(plan::KernelOp op,
                                                   const RPUPlan &plan) {
  FailureOr<int64_t> n = getIntegerField(op, plan.emission, "n");
  FailureOr<int64_t> input = getIntegerField(op, plan.emission, "input");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  if (failed(n) || failed(input) || failed(out))
    return failure();

  return buildReduceSumAllBody(*n, *input, *out);
}

static FailureOr<std::string> emitConvKxKBody(plan::KernelOp op,
                                              const RPUPlan &plan) {
  FailureOr<int64_t> kernelSize =
      getIntegerField(op, plan.emission, "kernel_size");
  FailureOr<int64_t> m = getIntegerField(op, plan.emission, "m");
  FailureOr<int64_t> inChannels =
      getIntegerField(op, plan.emission, "in_channels");
  FailureOr<int64_t> outChannels =
      getIntegerField(op, plan.emission, "out_channels");
  FailureOr<int64_t> inputWidth =
      getIntegerField(op, plan.emission, "input_width");
  FailureOr<int64_t> input = getIntegerField(op, plan.emission, "input");
  FailureOr<int64_t> weight = getIntegerField(op, plan.emission, "weight");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  if (failed(kernelSize) || failed(m) || failed(inChannels) ||
      failed(outChannels) || failed(inputWidth) || failed(input) ||
      failed(weight) || failed(out))
    return failure();

  return buildConvKxKBody(*kernelSize, *m, *inChannels, *outChannels,
                          *inputWidth, *input, *weight, *out);
}

static FailureOr<std::string> emitResNetBlockBody(plan::KernelOp op,
                                                  const RPUPlan &plan) {
  FailureOr<int64_t> m = getIntegerField(op, plan.emission, "m");
  FailureOr<int64_t> channels = getIntegerField(op, plan.emission, "channels");
  FailureOr<int64_t> hidden = getIntegerField(op, plan.emission, "hidden");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  FailureOr<int64_t> x = getIntegerField(op, plan.emission, "x");
  FailureOr<int64_t> w1 = getIntegerField(op, plan.emission, "w1");
  FailureOr<int64_t> w2 = getIntegerField(op, plan.emission, "w2");
  if (failed(m) || failed(channels) || failed(hidden) || failed(out) ||
      failed(x) || failed(w1) || failed(w2))
    return failure();

  return buildResNetBlockBody(*m, *channels, *hidden, *out, *x, *w1, *w2);
}

static FailureOr<std::string> emitResNet50BottleneckBody(plan::KernelOp op,
                                                         const RPUPlan &plan) {
  FailureOr<int64_t> kernelSize =
      getIntegerField(op, plan.emission, "kernel_size");
  FailureOr<int64_t> m = getIntegerField(op, plan.emission, "m");
  FailureOr<int64_t> channels = getIntegerField(op, plan.emission, "channels");
  FailureOr<int64_t> bottleneck =
      getIntegerField(op, plan.emission, "bottleneck");
  FailureOr<int64_t> inputWidth =
      getIntegerField(op, plan.emission, "input_width");
  FailureOr<int64_t> out = getIntegerField(op, plan.emission, "out");
  FailureOr<int64_t> input = getIntegerField(op, plan.emission, "input");
  FailureOr<int64_t> w1 = getIntegerField(op, plan.emission, "w1");
  FailureOr<int64_t> w2 = getIntegerField(op, plan.emission, "w2");
  FailureOr<int64_t> w3 = getIntegerField(op, plan.emission, "w3");
  if (failed(kernelSize) || failed(m) || failed(channels) ||
      failed(bottleneck) || failed(inputWidth) || failed(out) ||
      failed(input) || failed(w1) || failed(w2) || failed(w3))
    return failure();

  return buildResNet50BottleneckBody(*kernelSize, *m, *channels, *bottleneck,
                                     *inputWidth, *out, *input, *w1, *w2, *w3);
}

FailureOr<RPUPlanKernelSummary>
getRPUPlanKernelSummaryFromKernelOp(plan::KernelOp op) {
  return RPUPlanKernelSummary{op.getKernelName().str(), op.getPattern().str()};
}

FailureOr<RPUPlanKernelSummary>
getRPUPlanKernelSummaryFromModule(ModuleOp module) {
  llvm::SmallVector<plan::KernelOp> plans;
  module.walk([&](plan::KernelOp op) { plans.push_back(op); });
  if (plans.empty()) {
    module.emitError(
        "RPU rpu_plan.kernel summary requires one rpu_plan.kernel, found none");
    return failure();
  }
  if (plans.size() > 1) {
    module.emitError("RPU rpu_plan.kernel summary requires one "
                     "rpu_plan.kernel, found multiple");
    return failure();
  }
  return getRPUPlanKernelSummaryFromKernelOp(plans.front());
}

FailureOr<RPUDSLEmissionResult> emitRPUDSLFromKernelOp(plan::KernelOp op) {
  std::optional<RPUPlan> converted = rpuPlanFromKernelOp(op);
  if (!converted) {
    op.emitError(
        "failed to convert rpu_plan.kernel to RPUPlan DTO for .rc emission");
    return failure();
  }
  RPUPlan plan = std::move(*converted);

  if (!isDirectRPUDSLSupportedPattern(plan.pattern)) {
    op.emitError("RPU direct .rc emission does not support pattern '")
        << plan.pattern << "'";
    return failure();
  }

  FailureOr<std::string> args = formatSignature(op, plan);
  if (failed(args))
    return failure();

  if (plan.pattern == "add") {
    FailureOr<std::string> body = emitAddBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "gemm") {
    FailureOr<std::string> body = emitGemmBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "softmax") {
    FailureOr<std::string> body = emitSoftmaxBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "sqrt") {
    FailureOr<std::string> body = emitSqrtBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "reduce_sum_all") {
    FailureOr<std::string> body = emitReduceSumAllBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "convkxk") {
    FailureOr<std::string> body = emitConvKxKBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "resnet_block") {
    FailureOr<std::string> body = emitResNetBlockBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }
  if (plan.pattern == "resnet50_bottleneck") {
    FailureOr<std::string> body = emitResNet50BottleneckBody(op, plan);
    if (failed(body))
      return failure();
    return RPUDSLEmissionResult{
        plan.kernelName, plan.pattern,
        buildRPUDSLProgram(plan.kernelName, *args, *body)};
  }

  op.emitError("RPU direct .rc emission does not support pattern '")
      << plan.pattern << "'";
  return failure();
}

FailureOr<RPUDSLEmissionResult> emitRPUDSLFromModule(ModuleOp module) {
  llvm::SmallVector<plan::KernelOp> plans;
  module.walk([&](plan::KernelOp op) { plans.push_back(op); });
  if (plans.empty()) {
    module.emitError(
        "RPU direct .rc emission requires one rpu_plan.kernel, found none");
    return failure();
  }
  if (plans.size() > 1) {
    module.emitError(
        "RPU direct .rc emission requires one rpu_plan.kernel, found multiple");
    return failure();
  }
  return emitRPUDSLFromKernelOp(plans.front());
}

} // namespace rpu
} // namespace mlir
