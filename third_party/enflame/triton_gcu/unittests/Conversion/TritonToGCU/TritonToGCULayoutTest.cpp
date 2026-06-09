// Copyright 2025 Enflame. All Rights Reserved.
#include <string>

#include "gtest/gtest.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Conversion/Passes.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

using namespace mlir;

namespace ttg = triton::gpu;

#if defined(TEST_GCU300) || defined(TEST_GCU400)

namespace {

#ifdef TEST_GCU400
constexpr const char *kTarget = "gcu:gcu400";
#else
constexpr const char *kTarget = "gcu:gcu300";
#endif

class TritonToGCULayoutTest : public ::testing::Test {
protected:
  MLIRContext ctx;

  void SetUp() override {
    ctx.loadDialect<triton::TritonDialect, ttg::TritonGPUDialect,
                    gpu::GPUDialect, func::FuncDialect, arith::ArithDialect,
                    scf::SCFDialect, cf::ControlFlowDialect,
                    triton::gcu::TritonGCUDialect>();
    ctx.allowUnregisteredDialects();
  }

  OwningOpRef<ModuleOp> parseModule(StringRef source, bool verify = true) {
    if (verify)
      return parseSourceString<ModuleOp>(source, &ctx);
    ParserConfig config(&ctx, /*verifyAfterParse=*/false);
    return parseSourceString<ModuleOp>(source, config);
  }

  LogicalResult runConvertTritonToGCU(ModuleOp module) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    pm.enableVerifier(false);
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertTritonToGCUPass());
    return pm.run(module);
  }
};

// tt.trans with blocked→blocked encoding triggers TTTransOpLowering
// shared-staging path (lines 2640-2679 gcu300 / 2205-2249 gcu400).
TEST_F(TritonToGCULayoutTest, TransBlocked) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>) -> (),
      sym_name = "trans_blocked", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>):
      %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
      %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
      %c32 = "arith.constant"() <{value = 32 : index}> : () -> index
      %zero_f32 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %ptr = "tt.ptr_to_int"(%arg0) : (!tt.ptr<f32>) -> i64
      %mptr = "triton_gcu.int2ptr"(%ptr) : (i64) -> !triton_gcu.ptr<f32>
      %t = "triton_gcu.load"(%mptr, %c32, %c32, %c1, %c1, %c0, %c0, %zero_f32) <{operandSegmentSizes = array<i32: 1, 2, 2, 2, 1>, order_hint = array<i32: 1, 0>}> : (!triton_gcu.ptr<f32>, index, index, index, index, index, index, f32) -> tensor<32x32xf32, #blocked>
      %tr = "tt.trans"(%t) <{order = array<i32: 1, 0>}> : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked1>
      "triton_gcu.store"(%tr, %mptr, %c32, %c32, %c1, %c1, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>, order_hint = array<i32: 1, 0>}> : (tensor<32x32xf32, #blocked1>, !triton_gcu.ptr<f32>, index, index, index, index, index, index) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTritonToGCU(*module);
  EXPECT_TRUE(succeeded(result));
}

// ttg.convert_layout with same src/dst encoding triggers the noop path
// (lines 2710-2713 gcu300 / 2264-2272 gcu400).
TEST_F(TritonToGCULayoutTest, ConvertLayoutIdentity) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>) -> (),
      sym_name = "cvt_identity", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>):
      %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
      %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
      %c32 = "arith.constant"() <{value = 32 : index}> : () -> index
      %zero_f32 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %ptr = "tt.ptr_to_int"(%arg0) : (!tt.ptr<f32>) -> i64
      %mptr = "triton_gcu.int2ptr"(%ptr) : (i64) -> !triton_gcu.ptr<f32>
      %t = "triton_gcu.load"(%mptr, %c32, %c32, %c1, %c1, %c0, %c0, %zero_f32) <{operandSegmentSizes = array<i32: 1, 2, 2, 2, 1>, order_hint = array<i32: 1, 0>}> : (!triton_gcu.ptr<f32>, index, index, index, index, index, index, f32) -> tensor<32x32xf32, #blocked>
      %cvt = "ttg.convert_layout"(%t) : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked>
      "triton_gcu.store"(%cvt, %mptr, %c32, %c32, %c1, %c1, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>, order_hint = array<i32: 1, 0>}> : (tensor<32x32xf32, #blocked>, !triton_gcu.ptr<f32>, index, index, index, index, index, index) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTritonToGCU(*module);
  EXPECT_TRUE(succeeded(result));
}

// tt.broadcast with different src/dst layouts triggers the cross-layout
// shared-staging path (lines 2043-2090 gcu300 / 1601-1701 gcu400).
// The tt.broadcast verifier normally rejects different encodings, so
// we disable parse-time verification to exercise this code path.
TEST_F(TritonToGCULayoutTest, BroadcastCrossLayout) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>) -> (),
      sym_name = "broadcast_cross", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>):
      %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
      %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
      %c32 = "arith.constant"() <{value = 32 : index}> : () -> index
      %zero_f32 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %ptr = "tt.ptr_to_int"(%arg0) : (!tt.ptr<f32>) -> i64
      %mptr = "triton_gcu.int2ptr"(%ptr) : (i64) -> !triton_gcu.ptr<f32>
      %t = "triton_gcu.load"(%mptr, %c1, %c32, %c1, %c1, %c0, %c0, %zero_f32) <{operandSegmentSizes = array<i32: 1, 2, 2, 2, 1>, order_hint = array<i32: 1, 0>}> : (!triton_gcu.ptr<f32>, index, index, index, index, index, index, f32) -> tensor<1x32xf32, #blocked>
      %bc = "tt.broadcast"(%t) : (tensor<1x32xf32, #blocked>) -> tensor<32x32xf32, #blocked1>
      "triton_gcu.store"(%bc, %mptr, %c32, %c32, %c1, %c1, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>, order_hint = array<i32: 1, 0>}> : (tensor<32x32xf32, #blocked1>, !triton_gcu.ptr<f32>, index, index, index, index, index, index) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir, /*verify=*/false);
  ASSERT_TRUE(module);
  auto result = runConvertTritonToGCU(*module);
  (void)result;
}

// tt.expand_dims with per-thread element mismatch triggers the shared
// round-trip path (lines 2177-2199 gcu300 / 1734-1757 gcu400).
// The input must use SliceEncodingAttr whose parent has sizePerThread > 1
// in the expanded dimension, so srcNumElems != dstNumElems after insert.
TEST_F(TritonToGCULayoutTest, ExpandDimsSharedRoundTrip) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>) -> (),
      sym_name = "expand_dims_kernel", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>):
      %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
      %c1 = "arith.constant"() <{value = 1 : index}> : () -> index
      %c32 = "arith.constant"() <{value = 32 : index}> : () -> index
      %zero_f32 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %ptr = "tt.ptr_to_int"(%arg0) : (!tt.ptr<f32>) -> i64
      %mptr = "triton_gcu.int2ptr"(%ptr) : (i64) -> !triton_gcu.ptr<f32>
      %t = "triton_gcu.load"(%mptr, %c32, %c1, %c0, %zero_f32) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>, order_hint = array<i32: 0>}> : (!triton_gcu.ptr<f32>, index, index, index, f32) -> tensor<32xf32, #slice>
      %expanded = "tt.expand_dims"(%t) <{axis = 1 : i32}> : (tensor<32xf32, #slice>) -> tensor<32x1xf32, #blocked>
      "triton_gcu.store"(%expanded, %mptr, %c32, %c1, %c1, %c1, %c0, %c0) <{operandSegmentSizes = array<i32: 1, 1, 2, 2, 2>, order_hint = array<i32: 1, 0>}> : (tensor<32x1xf32, #blocked>, !triton_gcu.ptr<f32>, index, index, index, index, index, index) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTritonToGCU(*module);
  EXPECT_TRUE(succeeded(result));
}

} // namespace

#endif // TEST_GCU300 || TEST_GCU400
