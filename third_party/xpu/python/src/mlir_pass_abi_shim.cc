// The 202606 trust LLVM package ships MLIR headers that declare
// mlir::Pass::initializeOptions with mlir::LogicalResult in the callback type,
// while libMLIRPass.a exports the otherwise identical llvm::LogicalResult ABI.
// Provide a local symbol alias so libtriton.so can be imported when building
// the XPU backend against that package.

namespace flagtree_xpu_abi {
struct LogicalResult {
  bool isSuccess;
};

struct StringRef {
  const char *data;
  unsigned long length;
};

struct FunctionRef {
  void *callback;
  void *callable;
};

struct LogicalResult (*getRewritePatternMatchAndRewrite(void *pattern))(
    void *pattern, void *op, void *rewriter) {
  auto *vtable = *reinterpret_cast<void ***>(pattern);
  return reinterpret_cast<LogicalResult (*)(void *, void *, void *)>(vtable[2]);
}

bool isFailure(LogicalResult result) { return result.isSuccess; }
} // namespace flagtree_xpu_abi

extern "C" flagtree_xpu_abi::LogicalResult oldInitializeOptions(
    void *pass, flagtree_xpu_abi::StringRef options,
    flagtree_xpu_abi::FunctionRef
        errorHandler) asm("_ZN4mlir4Pass17initializeOptionsEN4llvm9StringRefENS"
                          "1_12function_refIFNS1_"
                          "13LogicalResultERKNS1_5TwineEEEE");

extern "C" flagtree_xpu_abi::LogicalResult newInitializeOptions(
    void *pass, flagtree_xpu_abi::StringRef options,
    flagtree_xpu_abi::FunctionRef
        errorHandler) asm("_ZN4mlir4Pass17initializeOptionsEN4llvm9StringRefENS"
                          "1_12function_refIFNS_"
                          "13LogicalResultERKNS1_5TwineEEEE");

extern "C" flagtree_xpu_abi::LogicalResult
newInitializeOptions(void *pass, flagtree_xpu_abi::StringRef options,
                     flagtree_xpu_abi::FunctionRef errorHandler) {
  return oldInitializeOptions(pass, options, errorHandler);
}

extern "C" flagtree_xpu_abi::LogicalResult
rewritePatternRewrite(void *pattern, void *op, void *rewriter) asm(
    "_ZNK4mlir14RewritePattern7rewriteEPNS_9OperationERNS_15PatternRewriterE");

extern "C" flagtree_xpu_abi::LogicalResult
rewritePatternRewrite(void *pattern, void *op, void *rewriter) {
  return flagtree_xpu_abi::getRewritePatternMatchAndRewrite(pattern)(
      pattern, op, rewriter);
}

// Compatibility symbols for prebuilt XPU objects compiled against the original
// Triton 3.6 LoadOp builders (before flagtree_hints was added).
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using llvm::ArrayRef;
using mlir::OpBuilder;
using mlir::OperationState;
using mlir::Value;
using mlir::triton::CacheModifier;
using mlir::triton::EvictionPolicy;
using mlir::triton::LoadOp;
using mlir::triton::PaddingOption;

extern "C" void
loadBuildCompat1(OpBuilder &, OperationState &, Value, CacheModifier,
                 EvictionPolicy,
                 bool) asm("_ZN4mlir6triton6LoadOp5buildERNS_9OpBuilderERNS_"
                           "14OperationStateENS_"
                           "5ValueENS0_13CacheModifierENS0_14EvictionPolicyEb");
extern "C" void loadBuildCompat1(OpBuilder &builder, OperationState &state,
                                 Value ptr, CacheModifier cache,
                                 EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, cache, evict, isVolatile,
                mlir::StringAttr{});
}

extern "C" void loadBuildCompat2(
    OpBuilder &, OperationState &, Value, ArrayRef<int32_t>,
    std::optional<PaddingOption>, CacheModifier, EvictionPolicy,
    bool) asm("_ZN4mlir6triton6LoadOp5buildERNS_9OpBuilderERNS_"
              "14OperationStateENS_"
              "5ValueEN4llvm8ArrayRefIiEESt8optionalINS0_13PaddingOptionEENS0_"
              "13CacheModifierENS0_14EvictionPolicyEb");
extern "C" void loadBuildCompat2(OpBuilder &builder, OperationState &state,
                                 Value ptr, ArrayRef<int32_t> boundaryCheck,
                                 std::optional<PaddingOption> padding,
                                 CacheModifier cache, EvictionPolicy evict,
                                 bool isVolatile) {
  LoadOp::build(builder, state, ptr, boundaryCheck, padding, cache, evict,
                isVolatile, mlir::StringAttr{});
}

extern "C" void loadBuildCompat3(
    OpBuilder &, OperationState &, Value, Value, CacheModifier, EvictionPolicy,
    bool) asm("_ZN4mlir6triton6LoadOp5buildERNS_9OpBuilderERNS_"
              "14OperationStateENS_"
              "5ValueES5_NS0_13CacheModifierENS0_14EvictionPolicyEb");
extern "C" void loadBuildCompat3(OpBuilder &builder, OperationState &state,
                                 Value ptr, Value mask, CacheModifier cache,
                                 EvictionPolicy evict, bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, cache, evict, isVolatile,
                mlir::StringAttr{});
}

extern "C" void loadBuildCompat4(
    OpBuilder &, OperationState &, Value, Value, Value, CacheModifier,
    EvictionPolicy,
    bool) asm("_ZN4mlir6triton6LoadOp5buildERNS_9OpBuilderERNS_"
              "14OperationStateENS_"
              "5ValueES5_S5_NS0_13CacheModifierENS0_14EvictionPolicyEb");
extern "C" void loadBuildCompat4(OpBuilder &builder, OperationState &state,
                                 Value ptr, Value mask, Value other,
                                 CacheModifier cache, EvictionPolicy evict,
                                 bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, other, cache, evict, isVolatile,
                mlir::StringAttr{});
}

extern "C" void
loadBuildCompat5(OpBuilder &, OperationState &, Value, Value, Value,
                 ArrayRef<int32_t>, std::optional<PaddingOption>, CacheModifier,
                 EvictionPolicy,
                 bool) asm("_ZN4mlir6triton6LoadOp5buildERNS_9OpBuilderERNS_"
                           "14OperationStateENS_"
                           "5ValueES5_S5_N4llvm8ArrayRefIiEESt8optionalINS0_"
                           "13PaddingOptionEENS0_"
                           "13CacheModifierENS0_14EvictionPolicyEb");
extern "C" void loadBuildCompat5(OpBuilder &builder, OperationState &state,
                                 Value ptr, Value mask, Value other,
                                 ArrayRef<int32_t> boundaryCheck,
                                 std::optional<PaddingOption> padding,
                                 CacheModifier cache, EvictionPolicy evict,
                                 bool isVolatile) {
  LoadOp::build(builder, state, ptr, mask, other, boundaryCheck, padding, cache,
                evict, isVolatile, mlir::StringAttr{});
}
