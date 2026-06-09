#include "RPUExecutableDirectLowering.h"

#include "RPUTTIRPatternMatcher.h"

#include "RPU/IR/Dialect.h"
#include "RPU/IR/ExecutableKind.h"
#include "RPUExecutableDirectElementwise1D.h"
#include "RPUExecutableElementwise16ValueMapLowering.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace mlir {
namespace rpu {

namespace {

LogicalResult emitUnsupportedDirectExecutablePattern(ModuleOp module);

} // namespace

LogicalResult lowerSupportedTTIRToRPUExecutable(ModuleOp module) {
  FailureOr<triton::FuncOp> maybeFunc = getSinglePublicFunc(module);
  if (failed(maybeFunc)) {
    module.emitError(
        "RPU direct executable recognition expects exactly one public tt.func");
    return failure();
  }

  RewritePatternSet patterns(module.getContext());
  populateSupportedTTIRToRPUExecutablePatterns(patterns);
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    module.emitError("failed to apply direct TTIR to executable RPU patterns");
    return failure();
  }

  if (module.getOps<triton::FuncOp>().empty())
    return success();

  return emitUnsupportedDirectExecutablePattern(module);
}

namespace {

static bool isPositiveMultipleOf16(int64_t value) {
  return value > 0 && value % 16 == 0;
}

static bool isSupportedDirectConvKxKKernelSize(int64_t kernelSize) {
  return kernelSize == 3 || kernelSize == 5 || kernelSize == 7 ||
         kernelSize == 9;
}

static bool shouldLowerAddAsGenericExecutable(AddOperands operands) {
  return operands.n > 0 && operands.n <= 128;
}

static void eraseModuleBodyExceptDirectFunc(ModuleOp module,
                                            triton::FuncOp root,
                                            PatternRewriter &rewriter) {
  llvm::SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations()) {
    if (&op != root.getOperation())
      ops.push_back(&op);
  }
  for (Operation *op : ops)
    rewriter.eraseOp(op);
}

static void prepareDirectExecutableModuleForDirectLowering(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    std::string &kernelName) {
  kernelName = func.getSymName().str();
  MLIRContext *context = module.getContext();
  context->getOrLoadDialect<triton::TritonDialect>();
  context->getOrLoadDialect<exec::RPUDialect>();
  eraseModuleBodyExceptDirectFunc(module, func, rewriter);
}

static exec::ExecutableAddBodyBuildSpec
buildExecutableAddBodySpec(const AddOperands &source) {
  exec::ExecutableAddBodyBuildSpec body;
  body.n = source.n;
  body.logicalN = source.logicalN;
  body.masked = source.masked;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.lhsArgIndex = static_cast<unsigned>(source.lhs);
  body.rhsArgIndex = static_cast<unsigned>(source.rhs);
  return body;
}

static StringRef getExecutableAddKind(const AddOperands &source) {
  return shouldLowerAddAsGenericExecutable(source)
             ? llvm::StringRef("generic")
             : llvm::StringRef(kAddPattern);
}

static FailureOr<bool> lowerCanonicalElementwise16ValueMapToExecutableOps(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const Elementwise16ValueMapLoweringRequest &request) {
  Elementwise16ValueMapLoweringPlan plan;
  FailureOr<bool> matched =
      buildElementwise16ValueMapLoweringPlan(request, plan);
  if (failed(matched))
    return failure();
  if (!*matched)
    return false;

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, "generic",
          /*numArgs=*/plan.inputArgIndices.size() + 1,
          "direct Elementwise16 value-map lowering requires executable kernel "
          "scaffold");
  if (failed(scaffold))
    return failure();

  if (failed(materializeElementwise16ValueMapLoweringPlan(
          rewriter, loc, scaffold->kernel, plan,
          "direct Elementwise16 value-map lowering requires IR-owned body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module))) {
    module.emitError("failed to verify direct Elementwise16 value-map "
                     "executable RPU module");
    return failure();
  }
  return true;
}

static LogicalResult lowerMatchedElementwise1DToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const Elementwise1DExecutableRequest &request) {
  if (request.outputArgIndex != 0)
    return module.emitError(
        "RPU executable Elementwise1D direct lowering requires out=0");
  if (request.inputArgIndices.empty() || request.ops.empty())
    return module.emitError(
        "RPU executable Elementwise1D direct lowering requires inputs and ops");
  if (!isPositiveMultipleOf16(request.n) || request.n > 256)
    return module.emitError("RPU executable Elementwise1D direct lowering "
                            "requires 0 < n <= 256 and 16-aligned n");
  if (request.masked) {
    if (request.logicalN <= 0 || request.logicalN > request.n)
      return module.emitError("RPU executable Elementwise1D direct lowering "
                              "requires 0 < logical_n <= n");
  } else if (request.logicalN != request.n) {
    return module.emitError(
        "RPU executable Elementwise1D direct lowering requires logical_n == n");
  }
  for (size_t i = 0, e = request.inputArgIndices.size(); i < e; ++i) {
    if (request.inputArgIndices[i] != static_cast<unsigned>(i + 1))
      return module.emitError("RPU executable Elementwise1D direct lowering "
                              "requires ordered input args");
  }

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();

  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, "generic",
          /*numArgs=*/request.inputArgIndices.size() + 1,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  exec::ExecutableCompactElementwise1DBuildSpec spec;
  spec.n = request.n;
  spec.logicalN = request.logicalN;
  spec.masked = request.masked;
  spec.outputArgIndex = request.outputArgIndex;
  spec.inputArgIndices = request.inputArgIndices;
  spec.ops = request.ops;

  if (failed(exec::buildExecutableCompactElementwise1DBody(
          rewriter, loc, kernel, spec,
          "direct Elementwise1D lowering requires IR-owned compact elementwise "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Elementwise1D executable RPU module");
  return success();
}

static LogicalResult lowerMatchedAddToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableAddBodyBuildSpec &body, StringRef executableKind) {
  if (body.outputArgIndex != 0 || body.lhsArgIndex != 1 ||
      body.rhsArgIndex != 2)
    return module.emitError(
        "RPU executable Add direct lowering requires out=0 lhs=1 rhs=2");
  if (!isPositiveMultipleOf16(body.n))
    return module.emitError(
        "RPU executable Add direct lowering requires positive 16-aligned n");
  if (body.masked) {
    if (body.logicalN <= 0 || body.logicalN > body.n)
      return module.emitError(
          "RPU executable Add direct lowering requires 0 < logical_n <= n");
  } else if (body.logicalN != body.n) {
    return module.emitError(
        "RPU executable Add direct lowering requires logical_n == n");
  }

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();

  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, executableKind,
          /*numArgs=*/3,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableAddBody(
          rewriter, loc, kernel, body,
          "direct Add lowering requires IR-owned Add body materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Add executable RPU module");
  return success();
}

static exec::ExecutableGemmBodyBuildSpec
buildExecutableGemmBodySpec(const GemmOperands &source) {
  exec::ExecutableGemmBodyBuildSpec body;
  body.m = source.m;
  body.n = source.n;
  body.k = source.k;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.lhsArgIndex = static_cast<unsigned>(source.lhs);
  body.rhsArgIndex = static_cast<unsigned>(source.rhs);
  return body;
}

static LogicalResult
lowerMatchedGemmToExecutable(PatternRewriter &rewriter, ModuleOp module,
                             triton::FuncOp func,
                             const exec::ExecutableGemmBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.lhsArgIndex != 1 ||
      body.rhsArgIndex != 2)
    return module.emitError(
        "RPU executable GEMM direct lowering requires out=0 lhs=1 rhs=2");
  if (!isPositiveMultipleOf16(body.m) || !isPositiveMultipleOf16(body.n) ||
      !isPositiveMultipleOf16(body.k))
    return module.emitError("RPU executable GEMM direct lowering requires "
                            "positive 16-aligned m/n/k");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();

  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kGemmPattern, /*numArgs=*/3,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableGemmBody(
          rewriter, loc, kernel, body,
          "direct GEMM lowering requires IR-owned GEMM body materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct GEMM executable RPU module");
  return success();
}

static exec::ExecutableSoftmaxBodyBuildSpec
buildExecutableSoftmaxBodySpec(const SoftmaxOperands &source) {
  exec::ExecutableSoftmaxBodyBuildSpec body;
  body.nvec = source.n / 16;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  return body;
}

static LogicalResult
validateExecutableSoftmaxOperands(ModuleOp module,
                                  const SoftmaxOperands &source) {
  if (source.out != 0 || source.input != 1)
    return module.emitError(
        "RPU executable Softmax direct lowering requires out=0 input=1");
  if (!isPositiveMultipleOf16(source.n))
    return module.emitError("RPU executable Softmax direct lowering requires "
                            "positive 16-aligned n");
  return success();
}

static LogicalResult lowerMatchedSoftmaxToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableSoftmaxBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1)
    return module.emitError(
        "RPU executable Softmax direct lowering requires out=0 input=1");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();

  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kSoftmaxPattern, /*numArgs=*/2,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableSoftmaxBody(
          rewriter, loc, kernel, body,
          "direct Softmax lowering requires IR-owned Softmax body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Softmax executable RPU module");
  return success();
}

static exec::ExecutableSqrtBodyBuildSpec
buildExecutableSqrtBodySpec(const SqrtOperands &source) {
  exec::ExecutableSqrtBodyBuildSpec body;
  body.nvec = source.n / 16;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  return body;
}

static LogicalResult
validateExecutableSqrtOperands(ModuleOp module, const SqrtOperands &source) {
  if (source.out != 0 || source.input != 1)
    return module.emitError(
        "RPU executable Sqrt direct lowering requires out=0 input=1");
  if (!isPositiveMultipleOf16(source.n))
    return module.emitError(
        "RPU executable Sqrt direct lowering requires positive 16-aligned n");
  return success();
}

static LogicalResult
lowerMatchedSqrtToExecutable(PatternRewriter &rewriter, ModuleOp module,
                             triton::FuncOp func,
                             const exec::ExecutableSqrtBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1)
    return module.emitError(
        "RPU executable Sqrt direct lowering requires out=0 input=1");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kSqrtPattern, /*numArgs=*/2,
          "direct Sqrt lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableSqrtBody(
          rewriter, loc, kernel, body,
          "direct Sqrt lowering requires IR-owned Sqrt body materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Sqrt executable RPU module");
  return success();
}

static exec::ExecutableReduceSumAllBodyBuildSpec
buildExecutableReduceSumAllBodySpec(const ReduceSumAllOperands &source) {
  exec::ExecutableReduceSumAllBodyBuildSpec body;
  body.nvec = source.n / 16;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  return body;
}

static LogicalResult
validateExecutableReduceSumAllOperands(ModuleOp module,
                                       const ReduceSumAllOperands &source) {
  if (source.out != 0 || source.input != 1)
    return module.emitError(
        "RPU executable ReduceSumAll direct lowering requires out=0 input=1");
  if (!isPositiveMultipleOf16(source.n))
    return module.emitError(
        "RPU executable ReduceSumAll direct lowering requires positive "
        "16-aligned n");
  return success();
}

static LogicalResult lowerMatchedReduceSumAllToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableReduceSumAllBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1)
    return module.emitError(
        "RPU executable ReduceSumAll direct lowering requires out=0 input=1");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kReduceSumAllPattern,
          /*numArgs=*/2,
          "direct ReduceSumAll lowering requires IR-owned executable kernel "
          "scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableReduceSumAllBody(
          rewriter, loc, kernel, body,
          "direct ReduceSumAll lowering requires IR-owned ReduceSumAll body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct ReduceSumAll executable RPU module");
  return success();
}

static exec::ExecutableReluBodyBuildSpec
buildExecutableReluBodySpec(const ReluOperands &source) {
  exec::ExecutableReluBodyBuildSpec body;
  body.nvec = source.n / 16;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  return body;
}

static LogicalResult
validateExecutableReluOperands(ModuleOp module, const ReluOperands &source) {
  if (source.out != 0 || source.input != 1)
    return module.emitError(
        "RPU executable Relu direct lowering requires out=0 input=1");
  if (!isPositiveMultipleOf16(source.n))
    return module.emitError(
        "RPU executable Relu direct lowering requires positive 16-aligned n");
  return success();
}

static LogicalResult
lowerMatchedReluToExecutable(PatternRewriter &rewriter, ModuleOp module,
                             triton::FuncOp func,
                             const exec::ExecutableReluBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1)
    return module.emitError(
        "RPU executable Relu direct lowering requires out=0 input=1");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kReluPattern, /*numArgs=*/2,
          "direct Relu lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableReluBody(
          rewriter, loc, kernel, body,
          "direct Relu lowering requires IR-owned Relu body materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Relu executable RPU module");
  return success();
}

static exec::ExecutableMaximumBodyBuildSpec
buildExecutableMaximumBodySpec(const MaximumOperands &source) {
  exec::ExecutableMaximumBodyBuildSpec body;
  body.n = source.n;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.lhsArgIndex = static_cast<unsigned>(source.lhs);
  body.rhsArgIndex = static_cast<unsigned>(source.rhs);
  return body;
}

static LogicalResult
validateExecutableMaximumOperands(ModuleOp module,
                                  const MaximumOperands &source) {
  if (source.out != 0 || source.lhs != 1 || source.rhs != 2)
    return module.emitError(
        "RPU executable Maximum direct lowering requires out=0 lhs=1 rhs=2");
  if (!isPositiveMultipleOf16(source.n))
    return module.emitError("RPU executable Maximum direct lowering requires "
                            "positive 16-aligned n");
  return success();
}

static LogicalResult lowerMatchedMaximumToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableMaximumBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.lhsArgIndex != 1 ||
      body.rhsArgIndex != 2)
    return module.emitError(
        "RPU executable Maximum direct lowering requires out=0 lhs=1 rhs=2");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kMaximumPattern, /*numArgs=*/3,
          "direct Maximum lowering requires IR-owned executable kernel "
          "scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableMaximumBody(
          rewriter, loc, kernel, body,
          "direct Maximum lowering requires IR-owned Maximum body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Maximum executable RPU module");
  return success();
}

static exec::ExecutableReduceSumAxisBodyBuildSpec
buildExecutableReduceSumAxisBodySpec(const ReduceSumAxisOperands &source) {
  exec::ExecutableReduceSumAxisBodyBuildSpec body;
  body.rows = source.rows;
  body.cols = source.cols;
  body.axis = source.axis;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  return body;
}

static LogicalResult
validateExecutableReduceSumAxisOperands(ModuleOp module,
                                        const ReduceSumAxisOperands &source) {
  if (source.out != 0 || source.input != 1)
    return module.emitError(
        "RPU executable ReduceSumAxis direct lowering requires out=0 input=1");
  if (source.rows <= 0 || source.cols <= 0 || source.rows % 16 != 0 ||
      source.cols % 16 != 0)
    return module.emitError(
        "RPU executable ReduceSumAxis direct lowering requires positive "
        "16-aligned rows and cols");
  if (source.axis != 0 && source.axis != 1)
    return module.emitError(
        "RPU executable ReduceSumAxis direct lowering requires axis in {0, 1}");
  return success();
}

static LogicalResult lowerMatchedReduceSumAxisToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableReduceSumAxisBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1)
    return module.emitError(
        "RPU executable ReduceSumAxis direct lowering requires out=0 input=1");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  llvm::StringRef pattern =
      body.axis == 0 ? kReduceSumAxis0Pattern : kReduceSumAxis1Pattern;
  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, pattern, /*numArgs=*/2,
          "direct ReduceSumAxis lowering requires IR-owned executable kernel "
          "scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableReduceSumAxisBody(
          rewriter, loc, kernel, body,
          "direct ReduceSumAxis lowering requires IR-owned body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct ReduceSumAxis executable RPU module");
  return success();
}

static exec::ExecutableBroadcastAddBodyBuildSpec
buildExecutableBroadcastAddBodySpec(const BroadcastAddOperands &source) {
  exec::ExecutableBroadcastAddBodyBuildSpec body;
  body.rows = source.rows;
  body.cols = source.cols;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.lhsArgIndex = static_cast<unsigned>(source.lhs);
  body.rhsArgIndex = static_cast<unsigned>(source.rhs);
  return body;
}

static LogicalResult
validateExecutableBroadcastAddOperands(ModuleOp module,
                                       const BroadcastAddOperands &source) {
  if (source.out != 0 || source.lhs != 1 || source.rhs != 2)
    return module.emitError("RPU executable BroadcastAdd direct lowering "
                            "requires out=0 lhs=1 rhs=2");
  if (source.rows <= 0 || source.cols <= 0 || source.rows % 16 != 0 ||
      source.cols % 16 != 0)
    return module.emitError(
        "RPU executable BroadcastAdd direct lowering requires positive "
        "16-aligned rows and cols");
  return success();
}

static LogicalResult lowerMatchedBroadcastAddToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableBroadcastAddBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.lhsArgIndex != 1 ||
      body.rhsArgIndex != 2)
    return module.emitError("RPU executable BroadcastAdd direct lowering "
                            "requires out=0 lhs=1 rhs=2");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  MLIRContext *context = module.getContext();
  Location loc = UnknownLoc::get(context);

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kBroadcastAddPattern,
          /*numArgs=*/3,
          "direct BroadcastAdd lowering requires IR-owned executable kernel "
          "scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableBroadcastAddBody(
          rewriter, loc, kernel, body,
          "direct BroadcastAdd lowering requires IR-owned body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct BroadcastAdd executable RPU module");
  return success();
}

static exec::ExecutableConvKxKBodyBuildSpec
buildExecutableConvKxKBodySpec(const ConvKxKOperands &source) {
  exec::ExecutableConvKxKBodyBuildSpec body;
  body.kernelSize = source.kernelSize;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  body.weightArgIndex = static_cast<unsigned>(source.weight);
  return body;
}

static LogicalResult
validateExecutableConvKxKOperands(ModuleOp module,
                                  const ConvKxKOperands &source) {
  if (source.out != 0 || source.input != 1 || source.weight != 2)
    return module.emitError("RPU executable ConvKxK direct lowering requires "
                            "out=0 input=1 weight=2");
  if (!isSupportedDirectConvKxKKernelSize(source.kernelSize))
    return module.emitError("RPU executable ConvKxK direct lowering requires "
                            "supported kernel_size");
  if (source.m != 16 || source.inChannels != 16 || source.outChannels != 16 ||
      source.inputWidth != 16)
    return module.emitError(
        "RPU executable ConvKxK direct lowering requires m/in_channels/"
        "out_channels/input_width all 16");
  return success();
}

static LogicalResult lowerMatchedConvKxKToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableConvKxKBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1 ||
      body.weightArgIndex != 2)
    return module.emitError("RPU executable ConvKxK direct lowering requires "
                            "out=0 input=1 weight=2");
  if (!isSupportedDirectConvKxKKernelSize(body.kernelSize))
    return module.emitError("RPU executable ConvKxK direct lowering requires "
                            "supported kernel_size");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  Location loc = UnknownLoc::get(module.getContext());

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kConvKxKPattern, /*numArgs=*/3,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableConvKxKBody(
          rewriter, loc, kernel, body,
          "direct ConvKxK lowering requires IR-owned ConvKxK body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct ConvKxK executable RPU module");
  return success();
}

static exec::ExecutableResNetBlockBodyBuildSpec
buildExecutableResNetBlockBodySpec(const ResNetBlockOperands &source) {
  exec::ExecutableResNetBlockBodyBuildSpec body;
  body.hidden = source.hidden;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.xArgIndex = static_cast<unsigned>(source.x);
  body.w1ArgIndex = static_cast<unsigned>(source.w1);
  body.w2ArgIndex = static_cast<unsigned>(source.w2);
  return body;
}

static LogicalResult
validateExecutableResNetBlockOperands(ModuleOp module,
                                      const ResNetBlockOperands &source) {
  if (source.out != 0 || source.x != 1 || source.w1 != 2 || source.w2 != 3)
    return module.emitError(
        "RPU executable residual direct lowering requires out=0 x=1 w1=2 w2=3");
  if (source.m != 16 || source.channels != 16 ||
      (source.hidden != 16 && source.hidden != 32))
    return module.emitError(
        "RPU executable residual direct lowering requires m/channels=16 and "
        "hidden in {16,32}");
  return success();
}

static LogicalResult lowerMatchedResNetBlockToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableResNetBlockBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.xArgIndex != 1 || body.w1ArgIndex != 2 ||
      body.w2ArgIndex != 3)
    return module.emitError(
        "RPU executable residual direct lowering requires out=0 x=1 w1=2 w2=3");
  if (body.hidden != 16 && body.hidden != 32)
    return module.emitError(
        "RPU executable residual direct lowering requires hidden in {16,32}");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  Location loc = UnknownLoc::get(module.getContext());

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kResNetBlockPattern,
          /*numArgs=*/4,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableResNetBlockBody(
          rewriter, loc, kernel, body,
          "direct residual lowering requires IR-owned residual body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct residual executable RPU module");
  return success();
}

static exec::ExecutableResNet50BottleneckBodyBuildSpec
buildExecutableResNet50BottleneckBodySpec(
    const ResNet50BottleneckOperands &source) {
  exec::ExecutableResNet50BottleneckBodyBuildSpec body;
  body.bottleneck = source.bottleneck;
  body.outputArgIndex = static_cast<unsigned>(source.out);
  body.inputArgIndex = static_cast<unsigned>(source.input);
  body.w1ArgIndex = static_cast<unsigned>(source.w1);
  body.w2ArgIndex = static_cast<unsigned>(source.w2);
  body.w3ArgIndex = static_cast<unsigned>(source.w3);
  return body;
}

static LogicalResult validateExecutableResNet50BottleneckOperands(
    ModuleOp module, const ResNet50BottleneckOperands &source) {
  if (source.out != 0 || source.input != 1 || source.w1 != 2 ||
      source.w2 != 3 || source.w3 != 4)
    return module.emitError("RPU executable bottleneck direct lowering "
                            "requires out=0 input=1 w1=2 w2=3 w3=4");
  if (source.kernelSize != 3 || source.m != 16 || source.channels != 16 ||
      source.inputWidth != 16 ||
      (source.bottleneck != 16 && source.bottleneck != 32))
    return module.emitError(
        "RPU executable bottleneck direct lowering requires kernel_size=3, "
        "m/channels/input_width=16, and bottleneck in {16,32}");
  return success();
}

static LogicalResult lowerMatchedResNet50BottleneckToExecutable(
    PatternRewriter &rewriter, ModuleOp module, triton::FuncOp func,
    const exec::ExecutableResNet50BottleneckBodyBuildSpec &body) {
  if (body.outputArgIndex != 0 || body.inputArgIndex != 1 ||
      body.w1ArgIndex != 2 || body.w2ArgIndex != 3 || body.w3ArgIndex != 4)
    return module.emitError(
        "RPU executable bottleneck direct lowering requires "
        "out=0 input=1 w1=2 w2=3 w3=4");
  if (body.bottleneck != 16 && body.bottleneck != 32)
    return module.emitError("RPU executable bottleneck direct lowering "
                            "requires bottleneck in {16,32}");

  std::string kernelName;
  prepareDirectExecutableModuleForDirectLowering(rewriter, module, func,
                                                 kernelName);
  Location loc = UnknownLoc::get(module.getContext());

  FailureOr<exec::ExecutableKernelScaffold> scaffold =
      exec::buildExecutableF16PointerKernelScaffold(
          rewriter, module, loc, kernelName, kResNet50BottleneckPattern,
          /*numArgs=*/5,
          "direct lowering requires IR-owned executable kernel scaffold");
  if (failed(scaffold))
    return failure();
  exec::KernelOp kernel = scaffold->kernel;

  if (failed(exec::buildExecutableResNet50BottleneckBody(
          rewriter, loc, kernel, body,
          "direct bottleneck lowering requires IR-owned bottleneck body "
          "materialization")))
    return failure();

  rewriter.eraseOp(func);

  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct bottleneck executable RPU module");
  return success();
}

LogicalResult emitUnsupportedDirectExecutablePattern(ModuleOp module) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "Unsupported RPU direct executable pattern. ";
  os << "Elementwise1D lowering: C++ recognizer "
     << kElementwise1DFailureReason;
  os << "; Add lowering: C++ recognizer " << kAddFailureReason;
  os << "; GEMM lowering: C++ recognizer " << kGemmFailureReason;
  os << "; Softmax lowering: C++ recognizer " << kSoftmaxFailureReason;
  os << "; Sqrt lowering: C++ recognizer " << kSqrtFailureReason;
  os << "; ReduceSumAll lowering: C++ recognizer "
     << kReduceSumAllFailureReason;
  os << "; Relu lowering: C++ recognizer " << kReluFailureReason;
  os << "; Maximum lowering: C++ recognizer " << kMaximumFailureReason;
  os << "; ReduceSumAxis0 lowering: C++ recognizer "
     << kReduceSumAxis0FailureReason;
  os << "; ReduceSumAxis1 lowering: C++ recognizer "
     << kReduceSumAxis1FailureReason;
  os << "; BroadcastAdd lowering: C++ recognizer "
     << kBroadcastAddFailureReason;
  os << "; ConvKxK lowering: C++ recognizer " << kConvKxKFailureReason;
  os << "; residual block lowering: C++ recognizer "
     << kResNetBlockFailureReason;
  os << "; ResNet50 bottleneck lowering: C++ recognizer "
     << kResNet50BottleneckFailureReason;
  module.emitError(os.str());
  return failure();
}

class Elementwise16ValueMapToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    FailureOr<Elementwise16ValueMapLoweringRequest> valueMapRequest =
        recognizeElementwise16ValueMapRequest(func);
    if (failed(valueMapRequest))
      return failure();

    FailureOr<bool> loweredElementwise16 =
        lowerCanonicalElementwise16ValueMapToExecutableOps(
            rewriter, module, func, *valueMapRequest);
    if (failed(loweredElementwise16))
      return failure();
    if (*loweredElementwise16)
      return success();
    return failure();
  }
};

class GenericElementwise1DToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    FailureOr<Elementwise1DExecutableRequest> request =
        recognizeElementwise1DExecutableRequest(func);
    if (failed(request))
      return failure();

    return lowerMatchedElementwise1DToExecutable(rewriter, module, func,
                                                 *request);
  }
};

class AddToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<AddOperands> operands = recognizeAdd(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    exec::ExecutableAddBodyBuildSpec body =
        buildExecutableAddBodySpec(*operands);
    StringRef executableKind = getExecutableAddKind(*operands);
    return lowerMatchedAddToExecutable(rewriter, module, func, body,
                                       executableKind);
  }
};

class GemmToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<GemmOperands> operands = recognizeGemm(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    exec::ExecutableGemmBodyBuildSpec body =
        buildExecutableGemmBodySpec(*operands);
    return lowerMatchedGemmToExecutable(rewriter, module, func, body);
  }
};

class SoftmaxToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<SoftmaxOperands> operands = recognizeSoftmax(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableSoftmaxOperands(module, *operands)))
      return failure();
    exec::ExecutableSoftmaxBodyBuildSpec body =
        buildExecutableSoftmaxBodySpec(*operands);
    return lowerMatchedSoftmaxToExecutable(rewriter, module, func, body);
  }
};

class SqrtToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<SqrtOperands> operands = recognizeSqrt(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableSqrtOperands(module, *operands)))
      return failure();
    exec::ExecutableSqrtBodyBuildSpec body =
        buildExecutableSqrtBodySpec(*operands);
    return lowerMatchedSqrtToExecutable(rewriter, module, func, body);
  }
};

class ReduceSumAllToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ReduceSumAllOperands> operands = recognizeReduceSumAll(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableReduceSumAllOperands(module, *operands)))
      return failure();
    exec::ExecutableReduceSumAllBodyBuildSpec body =
        buildExecutableReduceSumAllBodySpec(*operands);
    return lowerMatchedReduceSumAllToExecutable(rewriter, module, func, body);
  }
};

class ReluToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ReluOperands> operands = recognizeRelu(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableReluOperands(module, *operands)))
      return failure();
    exec::ExecutableReluBodyBuildSpec body =
        buildExecutableReluBodySpec(*operands);
    return lowerMatchedReluToExecutable(rewriter, module, func, body);
  }
};

class MaximumToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<MaximumOperands> operands = recognizeMaximum(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableMaximumOperands(module, *operands)))
      return failure();
    exec::ExecutableMaximumBodyBuildSpec body =
        buildExecutableMaximumBodySpec(*operands);
    return lowerMatchedMaximumToExecutable(rewriter, module, func, body);
  }
};

class ReduceSumAxis0ToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ReduceSumAxisOperands> operands =
        recognizeReduceSumAxis(func, /*axis=*/0);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableReduceSumAxisOperands(module, *operands)))
      return failure();
    exec::ExecutableReduceSumAxisBodyBuildSpec body =
        buildExecutableReduceSumAxisBodySpec(*operands);
    return lowerMatchedReduceSumAxisToExecutable(rewriter, module, func, body);
  }
};

class ReduceSumAxis1ToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ReduceSumAxisOperands> operands =
        recognizeReduceSumAxis(func, /*axis=*/1);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableReduceSumAxisOperands(module, *operands)))
      return failure();
    exec::ExecutableReduceSumAxisBodyBuildSpec body =
        buildExecutableReduceSumAxisBodySpec(*operands);
    return lowerMatchedReduceSumAxisToExecutable(rewriter, module, func, body);
  }
};

class BroadcastAddToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<BroadcastAddOperands> operands = recognizeBroadcastAdd(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableBroadcastAddOperands(module, *operands)))
      return failure();
    exec::ExecutableBroadcastAddBodyBuildSpec body =
        buildExecutableBroadcastAddBodySpec(*operands);
    return lowerMatchedBroadcastAddToExecutable(rewriter, module, func, body);
  }
};

class ConvKxKToRPUExecutablePattern : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ConvKxKOperands> operands = recognizeConvKxK(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableConvKxKOperands(module, *operands)))
      return failure();
    exec::ExecutableConvKxKBodyBuildSpec body =
        buildExecutableConvKxKBodySpec(*operands);
    return lowerMatchedConvKxKToExecutable(rewriter, module, func, body);
  }
};

class ResNetBlockToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ResNetBlockOperands> operands = recognizeResNetBlock(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableResNetBlockOperands(module, *operands)))
      return failure();
    exec::ExecutableResNetBlockBodyBuildSpec body =
        buildExecutableResNetBlockBodySpec(*operands);
    return lowerMatchedResNetBlockToExecutable(rewriter, module, func, body);
  }
};

class ResNet50BottleneckToRPUExecutablePattern
    : public OpRewritePattern<triton::FuncOp> {
public:
  using OpRewritePattern<triton::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FuncOp func,
                                PatternRewriter &rewriter) const override {
    FailureOr<ResNet50BottleneckOperands> operands =
        recognizeResNet50Bottleneck(func);
    if (failed(operands))
      return failure();

    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    if (failed(validateExecutableResNet50BottleneckOperands(module, *operands)))
      return failure();
    exec::ExecutableResNet50BottleneckBodyBuildSpec body =
        buildExecutableResNet50BottleneckBodySpec(*operands);
    return lowerMatchedResNet50BottleneckToExecutable(rewriter, module, func,
                                                      body);
  }
};

} // namespace

void populateSupportedTTIRToRPUExecutablePatterns(RewritePatternSet &patterns) {
  patterns.add<Elementwise16ValueMapToRPUExecutablePattern>(
      patterns.getContext(), PatternBenefit(3));
  patterns.add<GenericElementwise1DToRPUExecutablePattern>(
      patterns.getContext(), PatternBenefit(2));
  patterns.add<AddToRPUExecutablePattern>(patterns.getContext(),
                                          PatternBenefit(1));
  patterns.add<GemmToRPUExecutablePattern>(patterns.getContext(),
                                           PatternBenefit(1));
  patterns.add<SoftmaxToRPUExecutablePattern>(patterns.getContext(),
                                              PatternBenefit(1));
  patterns.add<SqrtToRPUExecutablePattern>(patterns.getContext(),
                                           PatternBenefit(1));
  patterns.add<ReduceSumAllToRPUExecutablePattern>(patterns.getContext(),
                                                   PatternBenefit(1));
  patterns.add<ReluToRPUExecutablePattern>(patterns.getContext(),
                                           PatternBenefit(1));
  patterns.add<MaximumToRPUExecutablePattern>(patterns.getContext(),
                                              PatternBenefit(1));
  patterns.add<ReduceSumAxis0ToRPUExecutablePattern>(patterns.getContext(),
                                                     PatternBenefit(1));
  patterns.add<ReduceSumAxis1ToRPUExecutablePattern>(patterns.getContext(),
                                                     PatternBenefit(1));
  patterns.add<BroadcastAddToRPUExecutablePattern>(patterns.getContext(),
                                                   PatternBenefit(1));
  patterns.add<ConvKxKToRPUExecutablePattern>(patterns.getContext(),
                                              PatternBenefit(1));
  patterns.add<ResNetBlockToRPUExecutablePattern>(patterns.getContext(),
                                                  PatternBenefit(1));
  patterns.add<ResNet50BottleneckToRPUExecutablePattern>(patterns.getContext(),
                                                         PatternBenefit(1));
}

} // namespace rpu
} // namespace mlir
