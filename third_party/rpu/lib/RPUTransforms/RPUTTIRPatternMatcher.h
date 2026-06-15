#pragma once

#include "RPUExecutableElementwise16ValueMapLowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace rpu {

constexpr int64_t kAddMaxContiguousNVec = 128;
constexpr int64_t kAddMaxTensorRows = 1024;

struct TraceAnchor {
  std::string kind;
  std::string op;
  std::string location;
};

struct AddOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  int64_t out = -1;
  int64_t lhs = -1;
  int64_t rhs = -1;
};

struct GemmOperands {
  TraceAnchor anchor;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t out = -1;
  int64_t lhs = -1;
  int64_t rhs = -1;
};

struct SoftmaxOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t out = -1;
  int64_t input = -1;
};

struct SqrtOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t out = -1;
  int64_t input = -1;
};

struct ReduceSumAllOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t out = -1;
  int64_t input = -1;
};

struct ReluOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t out = -1;
  int64_t input = -1;
};

struct MaximumOperands {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t out = -1;
  int64_t lhs = -1;
  int64_t rhs = -1;
};

struct ReduceSumAxisOperands {
  TraceAnchor anchor;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t axis = 0;
  int64_t out = -1;
  int64_t input = -1;
};

struct BroadcastAddOperands {
  TraceAnchor anchor;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t out = -1;
  int64_t lhs = -1;
  int64_t rhs = -1;
};

struct ConvKxKOperands {
  TraceAnchor anchor;
  int64_t kernelSize = 0;
  int64_t m = 0;
  int64_t inChannels = 0;
  int64_t outChannels = 0;
  int64_t inputWidth = 0;
  int64_t out = -1;
  int64_t input = -1;
  int64_t weight = -1;
};

struct ResNetBlockOperands {
  TraceAnchor anchor;
  int64_t m = 0;
  int64_t channels = 0;
  int64_t hidden = 0;
  int64_t out = -1;
  int64_t x = -1;
  int64_t w1 = -1;
  int64_t w2 = -1;
};

struct ResNet50BottleneckOperands {
  TraceAnchor anchor;
  int64_t kernelSize = 0;
  int64_t m = 0;
  int64_t channels = 0;
  int64_t bottleneck = 0;
  int64_t inputWidth = 0;
  int64_t out = -1;
  int64_t input = -1;
  int64_t w1 = -1;
  int64_t w2 = -1;
  int64_t w3 = -1;
};

struct TraceAttempt {
  std::string pattern;
  std::string status;
  std::string reason;
  TraceAnchor anchor;
};

constexpr StringRef kAddPattern = "add";
constexpr StringRef kGemmPattern = "gemm";
constexpr StringRef kSoftmaxPattern = "softmax";
constexpr StringRef kSqrtPattern = "sqrt";
constexpr StringRef kReduceSumAllPattern = "reduce_sum_all";
constexpr StringRef kReluPattern = "relu";
constexpr StringRef kMaximumPattern = "maximum";
constexpr StringRef kReduceSumAxis0Pattern = "reduce_sum_axis0";
constexpr StringRef kReduceSumAxis1Pattern = "reduce_sum_axis1";
constexpr StringRef kBroadcastAddPattern = "broadcast_add";
constexpr StringRef kResNetBlockPattern = "resnet_block";
constexpr StringRef kConvKxKPattern = "convkxk";
constexpr StringRef kResNet50BottleneckPattern = "resnet50_bottleneck";

constexpr StringRef kSinglePublicFuncReason =
    "expects exactly one public tt.func";
constexpr StringRef kAddFailureReason = "did not match supported vector add";
constexpr StringRef kGemmFailureReason = "did not match supported dot kernel";
constexpr StringRef kSoftmaxFailureReason =
    "did not match supported vector softmax";
constexpr StringRef kSqrtFailureReason = "did not match supported vector sqrt";
constexpr StringRef kReduceSumAllFailureReason =
    "did not match supported vector reduce_sum_all";
constexpr StringRef kReluFailureReason = "did not match supported vector relu";
constexpr StringRef kMaximumFailureReason =
    "did not match supported vector maximum";
constexpr StringRef kReduceSumAxis0FailureReason =
    "did not match supported reduce_sum along axis 0";
constexpr StringRef kReduceSumAxis1FailureReason =
    "did not match supported reduce_sum along axis 1";
constexpr StringRef kBroadcastAddFailureReason =
    "did not match supported [N,C]+[N] broadcast add";
constexpr StringRef kResNetBlockFailureReason =
    "did not match supported residual block";
constexpr StringRef kConvKxKFailureReason =
    "did not match supported row-window convolution";
constexpr StringRef kResNet50BottleneckFailureReason =
    "did not match supported bottleneck block";

std::string stringifyLocation(Location loc);
TraceAnchor moduleAnchor(ModuleOp module);

TraceAttempt failedAttempt(StringRef pattern, StringRef reason,
                           const TraceAnchor &anchor);
TraceAttempt matchedAttempt(StringRef pattern, const TraceAnchor &anchor);
void appendAllFailedAttempts(SmallVectorImpl<TraceAttempt> &attempts,
                             StringRef reason, const TraceAnchor &anchor);

TraceAnchor addFailureAnchor(triton::FuncOp func);
TraceAnchor gemmFailureAnchor(triton::FuncOp func);
TraceAnchor softmaxFailureAnchor(triton::FuncOp func);
TraceAnchor sqrtFailureAnchor(triton::FuncOp func);
TraceAnchor reduceSumAllFailureAnchor(triton::FuncOp func);
TraceAnchor reluFailureAnchor(triton::FuncOp func);
TraceAnchor maximumFailureAnchor(triton::FuncOp func);
TraceAnchor reduceSumAxisFailureAnchor(triton::FuncOp func);
TraceAnchor broadcastAddFailureAnchor(triton::FuncOp func);
TraceAnchor convKxKFailureAnchor(triton::FuncOp func);
TraceAnchor resNetBlockFailureAnchor(triton::FuncOp func);
TraceAnchor resNet50BottleneckFailureAnchor(triton::FuncOp func);

FailureOr<triton::FuncOp> getSinglePublicFunc(ModuleOp module);
FailureOr<AddOperands> recognizeAdd(triton::FuncOp func);
FailureOr<GemmOperands> recognizeGemm(triton::FuncOp func);
FailureOr<SoftmaxOperands> recognizeSoftmax(triton::FuncOp func);
FailureOr<SqrtOperands> recognizeSqrt(triton::FuncOp func);
FailureOr<ReduceSumAllOperands> recognizeReduceSumAll(triton::FuncOp func);
FailureOr<ReluOperands> recognizeRelu(triton::FuncOp func);
FailureOr<MaximumOperands> recognizeMaximum(triton::FuncOp func);
FailureOr<ReduceSumAxisOperands> recognizeReduceSumAxis(triton::FuncOp func,
                                                        int64_t axis);
FailureOr<BroadcastAddOperands> recognizeBroadcastAdd(triton::FuncOp func);
FailureOr<ConvKxKOperands> recognizeConvKxK(triton::FuncOp func);
FailureOr<ResNetBlockOperands> recognizeResNetBlock(triton::FuncOp func);
FailureOr<ResNet50BottleneckOperands>
recognizeResNet50Bottleneck(triton::FuncOp func);
FailureOr<Elementwise16ValueMapLoweringRequest>
recognizeElementwise16ValueMapRequest(triton::FuncOp func);

} // namespace rpu
} // namespace mlir
