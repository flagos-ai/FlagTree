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
#include <string>

#include "gtest/gtest.h"

#ifdef TEST_GCU400

#include "Conversion/TritonToGCU/ReduceScanCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gcu;

class ReduceScanCommonTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;
  int funcCounter = 0;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    vector::VectorDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect>();
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

  triton::ReduceOp createReduceWithCombine(
      Type elemType,
      std::function<Value(OpBuilder &, Location, Value, Value)> bodyFn) {
    auto tensorType = RankedTensorType::get({4, 128}, elemType);
    auto resultType = RankedTensorType::get({4}, elemType);
    std::string name = "fn_" + std::to_string(funcCounter++);
    auto funcType = builder->getFunctionType({tensorType}, {resultType});
    auto funcOp = builder->create<func::FuncOp>(loc(), name, funcType);
    auto &entry = funcOp.getBody().emplaceBlock();
    entry.addArgument(tensorType, loc());

    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&entry);

    auto reduceOp = builder->create<triton::ReduceOp>(
        loc(), TypeRange{resultType}, ValueRange{entry.getArgument(0)},
        builder->getI32IntegerAttr(1));

    auto &region = reduceOp.getCombineOp();
    auto &block = region.emplaceBlock();
    block.addArgument(elemType, loc());
    block.addArgument(elemType, loc());
    builder->setInsertionPointToStart(&block);
    Value result =
        bodyFn(*builder, loc(), block.getArgument(0), block.getArgument(1));
    builder->create<triton::ReduceReturnOp>(loc(), ValueRange{result});
    return reduceOp;
  }
};

// ---------- inferIdentityAttrs tests ----------

TEST_F(ReduceScanCommonTest, InferIdentity_ADD_Int) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::AddIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getInt(), 0);
}

TEST_F(ReduceScanCommonTest, InferIdentity_ADD_Float) {
  auto op = createReduceWithCombine(
      builder->getF32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::AddFOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<FloatAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_DOUBLE_EQ(attr.getValueAsDouble(), 0.0);
}

TEST_F(ReduceScanCommonTest, InferIdentity_MAXSI) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MaxSIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), APInt::getSignedMinValue(32));
}

TEST_F(ReduceScanCommonTest, InferIdentity_MINSI) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MinSIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), APInt::getSignedMaxValue(32));
}

TEST_F(ReduceScanCommonTest, InferIdentity_MAXNUMF) {
  auto op = createReduceWithCombine(
      builder->getF32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MaxNumFOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<FloatAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_TRUE(attr.getValue().isInfinity());
  EXPECT_TRUE(attr.getValue().isNegative());
}

TEST_F(ReduceScanCommonTest, InferIdentity_MINNUMF) {
  auto op = createReduceWithCombine(
      builder->getF32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MinNumFOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<FloatAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_TRUE(attr.getValue().isInfinity());
  EXPECT_FALSE(attr.getValue().isNegative());
}

TEST_F(ReduceScanCommonTest, InferIdentity_AND) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::AndIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), APInt::getAllOnes(32));
}

TEST_F(ReduceScanCommonTest, InferIdentity_OR) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::OrIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getInt(), 0);
}

TEST_F(ReduceScanCommonTest, InferIdentity_XOR) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::XOrIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getInt(), 0);
}

TEST_F(ReduceScanCommonTest, InferIdentity_MAXUI) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MaxUIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getInt(), 0);
}

TEST_F(ReduceScanCommonTest, InferIdentity_MINUI) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MinUIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  auto result = desc.inferIdentityAttrs(*builder);
  ASSERT_TRUE(succeeded(result));
  auto attr = dyn_cast<IntegerAttr>(result.value()[0]);
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getValue(), APInt::getAllOnes(32));
}

TEST_F(ReduceScanCommonTest, InferIdentity_MUL_Fails) {
  auto op = createReduceWithCombine(
      builder->getI32Type(), [](OpBuilder &b, Location l, Value a, Value c) {
        return b.create<arith::MulIOp>(l, a, c).getResult();
      });
  CombineOpDesc desc(op);
  EXPECT_TRUE(failed(desc.inferIdentityAttrs(*builder)));
}

// ---------- applyVectorizedCombine with captured boolean constant ----------

TEST_F(ReduceScanCommonTest, VectorizedCombine_CapturedBoolConst) {
  auto tensorType = RankedTensorType::get({4, 128}, builder->getF32Type());
  auto resultType = RankedTensorType::get({4}, builder->getF32Type());
  auto funcType = builder->getFunctionType({tensorType}, {resultType});
  auto funcOp = builder->create<func::FuncOp>(loc(), "test_vec_bool", funcType);
  auto &entry = funcOp.getBody().emplaceBlock();
  entry.addArgument(tensorType, loc());

  OpBuilder::InsertionGuard outerGuard(*builder);
  builder->setInsertionPointToStart(&entry);

  auto trueCst =
      builder->create<arith::ConstantOp>(loc(), builder->getBoolAttr(true));

  auto reduceOp = builder->create<triton::ReduceOp>(
      loc(), TypeRange{resultType}, ValueRange{entry.getArgument(0)},
      builder->getI32IntegerAttr(1));
  {
    auto &region = reduceOp.getCombineOp();
    auto &block = region.emplaceBlock();
    block.addArgument(builder->getF32Type(), loc());
    block.addArgument(builder->getF32Type(), loc());
    OpBuilder::InsertionGuard innerGuard(*builder);
    builder->setInsertionPointToStart(&block);
    auto sel = builder->create<arith::SelectOp>(
        loc(), trueCst.getResult(), block.getArgument(0), block.getArgument(1));
    builder->create<triton::ReduceReturnOp>(loc(), ValueRange{sel.getResult()});
  }

  builder->setInsertionPointAfter(reduceOp);
  CombineOpDesc desc(reduceOp);

  unsigned vectorLength = 4;
  auto vecType = VectorType::get({static_cast<int64_t>(vectorLength)},
                                 builder->getF32Type());
  SmallVector<float> zeros(vectorLength, 0.0f);
  auto zeroAttr = DenseElementsAttr::get(vecType, llvm::ArrayRef(zeros));
  auto v1 = builder->create<arith::ConstantOp>(loc(), zeroAttr);
  auto v2 = builder->create<arith::ConstantOp>(loc(), zeroAttr);

  auto results = desc.applyVectorizedCombine(*builder, loc(),
                                             ValueRange{v1, v2}, vectorLength);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(isa<VectorType>(results[0].getType()));
}

#else

TEST(ReduceScanCommonTest, GCU300Placeholder) { SUCCEED(); }

#endif
