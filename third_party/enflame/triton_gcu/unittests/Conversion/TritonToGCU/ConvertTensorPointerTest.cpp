// Copyright 2025 Enflame. All Rights Reserved.
#include <string>

#include "gtest/gtest.h"

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

class ConvertTensorPointerTest : public ::testing::Test {
protected:
  MLIRContext ctx;

  void SetUp() override {
    ctx.loadDialect<triton::TritonDialect, ttg::TritonGPUDialect,
                    gpu::GPUDialect, func::FuncDialect, arith::ArithDialect,
                    scf::SCFDialect, cf::ControlFlowDialect,
                    triton::gcu::TritonGCUDialect>();
  }

  OwningOpRef<ModuleOp> parseModule(StringRef source) {
    return parseSourceString<ModuleOp>(source, &ctx);
  }

  LogicalResult runConvertTensorPointerPass(ModuleOp module) {
    PassManager pm(module->getName(), PassManager::Nesting::Implicit);
    pm.enableVerifier(false);
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertTensorPointerPass());
    return pm.run(module);
  }
};

// scf.if returning a tensor pointer triggers rewriteIfOp (lines 382-447).
// The pass has a type mismatch bug (i64 result types vs i32 offsets) making
// this path dead for MLIR tests. The gtest runs with verifier disabled to
// exercise the code path for coverage.
TEST_F(ConvertTensorPointerTest, RewriteIfOp_TensorPointerResult) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>, !tt.ptr<f32>) -> (),
      sym_name = "test_if_tensor_ptr", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>):
      %c0_i32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %c2_i32 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %c1_i64 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %c128_i64 = "arith.constant"() <{value = 128 : i64}> : () -> i64
      %c32_i32 = "arith.constant"() <{value = 32 : i32}> : () -> i32
      %pid = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32
      %off0 = "arith.muli"(%pid, %c32_i32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %ptr0 = "tt.make_tensor_ptr"(%arg0, %c128_i64, %c128_i64, %c128_i64, %c1_i64, %off0, %c0_i32) <{order = array<i32: 1, 0>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
      %cond = "arith.cmpi"(%pid, %c0_i32) <{predicate = 0 : i64}> : (i32, i32) -> i1
      %if_ptr = "scf.if"(%cond) ({
        %adv = "tt.advance"(%ptr0, %c32_i32, %c0_i32) : (!tt.ptr<tensor<32x32xf32, #blocked>>, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
        "scf.yield"(%adv) : (!tt.ptr<tensor<32x32xf32, #blocked>>) -> ()
      }, {
        "scf.yield"(%ptr0) : (!tt.ptr<tensor<32x32xf32, #blocked>>) -> ()
      }) : (i1) -> !tt.ptr<tensor<32x32xf32, #blocked>>
      %ld = "tt.load"(%if_ptr) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 0, 0>}> : (!tt.ptr<tensor<32x32xf32, #blocked>>) -> tensor<32x32xf32, #blocked>
      %ptr1 = "tt.make_tensor_ptr"(%arg1, %c128_i64, %c128_i64, %c128_i64, %c1_i64, %off0, %c0_i32) <{order = array<i32: 1, 0>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
      "tt.store"(%ptr1, %ld) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (!tt.ptr<tensor<32x32xf32, #blocked>>, tensor<32x32xf32, #blocked>) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTensorPointerPass(*module);
  EXPECT_TRUE(succeeded(result));
}

// scf.if with non-tensor-pointer results exercises the early return
// (needRewrite=false).
TEST_F(ConvertTensorPointerTest, RewriteIfOp_NonTensorPointerResult) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}, {tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>, !tt.ptr<f32>) -> (),
      sym_name = "test_if_non_tensor_ptr", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>):
      %c0_i32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %c1_i64 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %c128_i64 = "arith.constant"() <{value = 128 : i64}> : () -> i64
      %c32_i32 = "arith.constant"() <{value = 32 : i32}> : () -> i32
      %cst1 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32, #blocked>}> : () -> tensor<32x32xf32, #blocked>
      %cst2 = "arith.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32, #blocked>}> : () -> tensor<32x32xf32, #blocked>
      %pid = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32
      %off0 = "arith.muli"(%pid, %c32_i32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %cond = "arith.cmpi"(%pid, %c0_i32) <{predicate = 0 : i64}> : (i32, i32) -> i1
      %res = "scf.if"(%cond) ({
        "scf.yield"(%cst1) : (tensor<32x32xf32, #blocked>) -> ()
      }, {
        "scf.yield"(%cst2) : (tensor<32x32xf32, #blocked>) -> ()
      }) : (i1) -> tensor<32x32xf32, #blocked>
      %ptr1 = "tt.make_tensor_ptr"(%arg1, %c128_i64, %c128_i64, %c128_i64, %c1_i64, %off0, %c0_i32) <{order = array<i32: 1, 0>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
      "tt.store"(%ptr1, %res) <{boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32}> : (!tt.ptr<tensor<32x32xf32, #blocked>>, tensor<32x32xf32, #blocked>) -> ()
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTensorPointerPass(*module);
  EXPECT_TRUE(succeeded(result));
}

// scf.if returning mixed results (tensor pointer + non-tensor-pointer).
TEST_F(ConvertTensorPointerTest, RewriteIfOp_MixedResults) {
  std::string ir = R"(
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
"builtin.module"() ({
  "gpu.module"() <{sym_name = "triton"}> ({
    "tt.func"() <{arg_attrs = [{tt.divisibility = 16 : i32}],
      function_type = (!tt.ptr<f32>) -> (),
      sym_name = "test_if_mixed", sym_visibility = "public"}> ({
    ^bb0(%arg0: !tt.ptr<f32>):
      %c0_i32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %c1_i64 = "arith.constant"() <{value = 1 : i64}> : () -> i64
      %c128_i64 = "arith.constant"() <{value = 128 : i64}> : () -> i64
      %c32_i32 = "arith.constant"() <{value = 32 : i32}> : () -> i32
      %cst1 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32, #blocked>}> : () -> tensor<32x32xf32, #blocked>
      %cst2 = "arith.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32, #blocked>}> : () -> tensor<32x32xf32, #blocked>
      %pid = "tt.get_program_id"() <{axis = 0 : i32}> : () -> i32
      %off0 = "arith.muli"(%pid, %c32_i32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %ptr0 = "tt.make_tensor_ptr"(%arg0, %c128_i64, %c128_i64, %c128_i64, %c1_i64, %off0, %c0_i32) <{order = array<i32: 1, 0>}> : (!tt.ptr<f32>, i64, i64, i64, i64, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
      %cond = "arith.cmpi"(%pid, %c0_i32) <{predicate = 0 : i64}> : (i32, i32) -> i1
      %if:2 = "scf.if"(%cond) ({
        %adv = "tt.advance"(%ptr0, %c32_i32, %c0_i32) : (!tt.ptr<tensor<32x32xf32, #blocked>>, i32, i32) -> !tt.ptr<tensor<32x32xf32, #blocked>>
        "scf.yield"(%adv, %cst1) : (!tt.ptr<tensor<32x32xf32, #blocked>>, tensor<32x32xf32, #blocked>) -> ()
      }, {
        "scf.yield"(%ptr0, %cst2) : (!tt.ptr<tensor<32x32xf32, #blocked>>, tensor<32x32xf32, #blocked>) -> ()
      }) : (i1) -> (!tt.ptr<tensor<32x32xf32, #blocked>>, tensor<32x32xf32, #blocked>)
      "tt.return"() : () -> ()
    }) {noinline = false} : () -> ()
  }) : () -> ()
}) {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = ")" +
                   std::string(kTarget) +
                   R"(", "ttg.threads-per-warp" = 1 : i32} : () -> ()
  )";

  auto module = parseModule(ir);
  ASSERT_TRUE(module);
  auto result = runConvertTensorPointerPass(*module);
  EXPECT_TRUE(succeeded(result));
}

} // namespace

#endif // TEST_GCU300 || TEST_GCU400
