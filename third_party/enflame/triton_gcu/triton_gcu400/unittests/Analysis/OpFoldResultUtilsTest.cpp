/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>

#include "gtest/gtest.h"

#include "Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gcu;

class OpFoldResultUtilsTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    triton::TritonDialect, triton::gpu::TritonGPUDialect>();
    ctx.getDiagEngine().registerHandler([](Diagnostic &) { return success(); });

    builder = std::make_unique<OpBuilder>(&ctx);
    module = ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());
  }

  void TearDown() override {
    if (module)
      module.erase();
  }

  Location loc() { return builder->getUnknownLoc(); }

  Value makeIndexValue(int64_t val) {
    return builder->create<arith::ConstantIndexOp>(loc(), val).getResult();
  }

  OpFoldResult makeIndexAttr(int64_t val) { return builder->getIndexAttr(val); }

  func::FuncOp makeFuncWithIndexArg(StringRef name) {
    auto funcType = builder->getFunctionType({builder->getIndexType()}, {});
    auto funcOp = builder->create<func::FuncOp>(loc(), name, funcType);
    auto &entryBlock = funcOp.getBody().emplaceBlock();
    entryBlock.addArgument(builder->getIndexType(), loc());
    builder->setInsertionPointToStart(&entryBlock);
    return funcOp;
  }

  void resetInsertionToModule() {
    builder->setInsertionPointToEnd(module.getBody());
  }
};

// ---------- getScalarValue ----------

TEST_F(OpFoldResultUtilsTest, SplatOp) {
  auto scalarVal =
      builder->create<arith::ConstantOp>(loc(), builder->getF32FloatAttr(1.0));
  auto tensorTy = RankedTensorType::get({16}, builder->getF32Type());
  auto splat = builder->create<triton::SplatOp>(loc(), tensorTy, scalarVal);

  auto result = getScalarValue(*builder, loc(), splat.getResult());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(static_cast<bool>(*result));
  EXPECT_FALSE(isa<ShapedType>(result->getType()));
}

TEST_F(OpFoldResultUtilsTest, SIToFPAfterSplat) {
  auto intScalar =
      builder->create<arith::ConstantOp>(loc(), builder->getI32IntegerAttr(0));
  auto intTensorTy = RankedTensorType::get({16}, builder->getI32Type());
  auto splat = builder->create<triton::SplatOp>(loc(), intTensorTy, intScalar);
  auto fpTensorTy = RankedTensorType::get({16}, builder->getF32Type());
  auto sitofp = builder->create<arith::SIToFPOp>(loc(), fpTensorTy, splat);

  auto result = getScalarValue(*builder, loc(), sitofp.getResult());
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(static_cast<bool>(*result));
  EXPECT_TRUE(isa<FloatType>(result->getType()));
}

TEST_F(OpFoldResultUtilsTest, NonSplatConstant) {
  auto tensorTy = RankedTensorType::get({2}, builder->getI32Type());
  SmallVector<int32_t> values = {1, 2};
  auto attr = DenseElementsAttr::get(tensorTy, llvm::ArrayRef(values));
  auto constOp = builder->create<arith::ConstantOp>(loc(), attr);

  auto result = getScalarValue(*builder, loc(), constOp.getResult());
  bool usable = result.has_value() && static_cast<bool>(*result);
  EXPECT_FALSE(usable);
}

TEST_F(OpFoldResultUtilsTest, UnsupportedOp) {
  auto tensorTy = RankedTensorType::get({16}, builder->getF32Type());
  auto splatAttr = DenseElementsAttr::get(tensorTy, {0.0f});
  auto c1 = builder->create<arith::ConstantOp>(loc(), splatAttr);
  auto c2 = builder->create<arith::ConstantOp>(loc(), splatAttr);
  auto addOp = builder->create<arith::AddFOp>(loc(), c1, c2);

  auto result = getScalarValue(*builder, loc(), addOp.getResult());
  bool usable = result.has_value() && static_cast<bool>(*result);
  EXPECT_FALSE(usable);
}

// ---------- subOFRs ----------

TEST_F(OpFoldResultUtilsTest, SubOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_sub");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(10);
  OpFoldResult rhs(dynamicRhs);

  auto result = subOFRs(*builder, loc(), lhs, rhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- mulOFRValue ----------

TEST_F(OpFoldResultUtilsTest, MulOFRValue_RhsZero) {
  auto lhsVal = makeIndexValue(7);
  auto rhsZero = makeIndexValue(0);

  auto result = mulOFRValue(*builder, loc(), OpFoldResult(lhsVal), rhsZero);
  EXPECT_TRUE(llvm::isa<Value>(result));
}

TEST_F(OpFoldResultUtilsTest, MulOFRValue_RhsOne) {
  auto rhsOne = makeIndexValue(1);
  OpFoldResult lhs = makeIndexAttr(42);

  auto result = mulOFRValue(*builder, loc(), lhs, rhsOne);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 42);
}

TEST_F(OpFoldResultUtilsTest, MulOFRValue_BothConst) {
  OpFoldResult lhs = makeIndexAttr(6);
  auto rhsConst = makeIndexValue(7);

  auto result = mulOFRValue(*builder, loc(), lhs, rhsConst);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 42);
}

TEST_F(OpFoldResultUtilsTest, MulOFRValue_LhsConstRhsDynamic) {
  auto funcOp = makeFuncWithIndexArg("test_mul");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(3);

  auto result = mulOFRValue(*builder, loc(), lhs, dynamicRhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- remOFRs ----------

TEST_F(OpFoldResultUtilsTest, RemOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_rem");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(5);

  auto result =
      remOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, RemOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_rem2");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(10);

  auto result = remOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- divOFRs ----------

TEST_F(OpFoldResultUtilsTest, DivOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_div");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(4);

  auto result =
      divOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, DivOFRs_DynamicLhsConstRhs) {
  auto funcOp = makeFuncWithIndexArg("test_div2");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult rhs = makeIndexAttr(3);

  auto result = divOFRs(*builder, loc(), OpFoldResult(dynamicLhs), rhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, DivOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_div3");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(12);

  auto result = divOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, DivOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(20);
  OpFoldResult rhs = makeIndexAttr(4);

  auto result = divOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 5);
}

// ---------- getIntAttr ----------

TEST_F(OpFoldResultUtilsTest, GetIntAttr_FromAttribute) {
  OpFoldResult ofr = makeIndexAttr(42);
  auto result = getIntAttr(ofr);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

TEST_F(OpFoldResultUtilsTest, GetIntAttr_FromValue) {
  Value v = makeIndexValue(7);
  OpFoldResult ofr(v);
  auto result = getIntAttr(ofr);
  EXPECT_FALSE(result.has_value());
}

// ---------- getValue ----------

TEST_F(OpFoldResultUtilsTest, GetValue_FromAttribute) {
  OpFoldResult ofr = makeIndexAttr(10);
  Value result = getValue(*builder, loc(), ofr);
  EXPECT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(isa<IndexType>(result.getType()));
}

TEST_F(OpFoldResultUtilsTest, GetValue_FromValue) {
  Value v = makeIndexValue(5);
  OpFoldResult ofr(v);
  Value result = getValue(*builder, loc(), ofr);
  EXPECT_EQ(result, v);
}

// ---------- getValues ----------

TEST_F(OpFoldResultUtilsTest, GetValues_Mixed) {
  Value v = makeIndexValue(3);
  SmallVector<OpFoldResult> ofrs = {makeIndexAttr(1), OpFoldResult(v),
                                    makeIndexAttr(2)};
  auto results = getValues(*builder, loc(), ofrs);
  ASSERT_EQ(results.size(), 3u);
  EXPECT_TRUE(isa<IndexType>(results[0].getType()));
  EXPECT_EQ(results[1], v);
  EXPECT_TRUE(isa<IndexType>(results[2].getType()));
}

// ---------- getScalarValue: splat DenseElementsAttr ----------

TEST_F(OpFoldResultUtilsTest, SplatDenseConstant) {
  auto tensorTy = RankedTensorType::get({16}, builder->getF32Type());
  auto splatAttr = DenseElementsAttr::get(tensorTy, {3.14f});
  auto constOp = builder->create<arith::ConstantOp>(loc(), splatAttr);

  auto result = getScalarValue(*builder, loc(), constOp.getResult());
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(static_cast<bool>(*result));
  EXPECT_TRUE(isa<FloatType>(result->getType()));
}

// ---------- addOFRs ----------

TEST_F(OpFoldResultUtilsTest, AddOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(3);
  OpFoldResult rhs = makeIndexAttr(7);

  auto result = addOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 10);
}

TEST_F(OpFoldResultUtilsTest, AddOFRs_LhsDynamic_RhsZero) {
  auto funcOp = makeFuncWithIndexArg("test_add_rhs0");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult rhs = makeIndexAttr(0);

  auto result = addOFRs(*builder, loc(), OpFoldResult(dynamicLhs), rhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  EXPECT_EQ(llvm::cast<Value>(result), dynamicLhs);
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, AddOFRs_LhsZero_RhsDynamic) {
  auto funcOp = makeFuncWithIndexArg("test_add_lhs0");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(0);

  auto result = addOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  EXPECT_EQ(llvm::cast<Value>(result), dynamicRhs);
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, AddOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_add_mixed");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(5);

  auto result = addOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, AddOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_add_dyn");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(3);

  auto result =
      addOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- subOFRs ----------

TEST_F(OpFoldResultUtilsTest, SubOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(10);
  OpFoldResult rhs = makeIndexAttr(3);

  auto result = subOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 7);
}

TEST_F(OpFoldResultUtilsTest, SubOFRs_RhsZeroShortcut) {
  auto funcOp = makeFuncWithIndexArg("test_sub_rhs0");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult rhs = makeIndexAttr(0);

  auto result = subOFRs(*builder, loc(), OpFoldResult(dynamicLhs), rhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  EXPECT_EQ(llvm::cast<Value>(result), dynamicLhs);
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, SubOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_sub_dyn");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(2);

  auto result =
      subOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- mulOFRValue additional ----------

TEST_F(OpFoldResultUtilsTest, MulOFRValue_LhsAttrZero) {
  auto funcOp = makeFuncWithIndexArg("test_mul_lhs0");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(0);

  auto result = mulOFRValue(*builder, loc(), lhs, dynamicRhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 0);
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, MulOFRValue_LhsAttrOne) {
  auto funcOp = makeFuncWithIndexArg("test_mul_lhs1");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(1);

  auto result = mulOFRValue(*builder, loc(), lhs, dynamicRhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  EXPECT_EQ(llvm::cast<Value>(result), dynamicRhs);
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, MulOFRValue_DynamicLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_mul_dyn");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  Value dynamicRhs = makeIndexValue(5);

  auto result =
      mulOFRValue(*builder, loc(), OpFoldResult(dynamicLhs), dynamicRhs);
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- minOFRs ----------

TEST_F(OpFoldResultUtilsTest, MinOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(5);
  OpFoldResult rhs = makeIndexAttr(3);

  auto result = minOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 3);
}

TEST_F(OpFoldResultUtilsTest, MinOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_min_dyn");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(10);

  auto result =
      minOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, MinOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_min_mixed");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(7);

  auto result = minOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- maxOFRs ----------

TEST_F(OpFoldResultUtilsTest, MaxOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(5);
  OpFoldResult rhs = makeIndexAttr(3);

  auto result = maxOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 5);
}

TEST_F(OpFoldResultUtilsTest, MaxOFRs_DynamicBoth) {
  auto funcOp = makeFuncWithIndexArg("test_max_dyn");
  Value dynamicLhs = funcOp.getBody().front().getArgument(0);
  auto rhsVal = makeIndexValue(10);

  auto result =
      maxOFRs(*builder, loc(), OpFoldResult(dynamicLhs), OpFoldResult(rhsVal));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

TEST_F(OpFoldResultUtilsTest, MaxOFRs_ConstLhsDynamicRhs) {
  auto funcOp = makeFuncWithIndexArg("test_max_mixed");
  Value dynamicRhs = funcOp.getBody().front().getArgument(0);
  OpFoldResult lhs = makeIndexAttr(7);

  auto result = maxOFRs(*builder, loc(), lhs, OpFoldResult(dynamicRhs));
  EXPECT_TRUE(llvm::isa<Value>(result));
  resetInsertionToModule();
}

// ---------- remOFRs ----------

TEST_F(OpFoldResultUtilsTest, RemOFRs_BothConst) {
  OpFoldResult lhs = makeIndexAttr(10);
  OpFoldResult rhs = makeIndexAttr(3);

  auto result = remOFRs(*builder, loc(), lhs, rhs);
  auto intVal = getIntAttr(result);
  ASSERT_TRUE(intVal.has_value());
  EXPECT_EQ(intVal.value(), 1);
}
