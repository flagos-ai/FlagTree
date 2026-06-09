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

#ifdef TEST_GCU400

#include "Conversion/TritonToGCU/Vectorize.h"

#include "Dialect/GCU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

class VectorizeTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect,
                    vector::VectorDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect, mlir::gcu::GCUDialect>();
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
};

// ---------- SelectVectorizeHandler i8 condition branch (lines 62-70)
// ----------

TEST_F(VectorizeTest, SelectHandler_I8Condition) {
  auto funcType = builder->getFunctionType({}, {});
  auto funcOp =
      builder->create<func::FuncOp>(loc(), "test_select_i8", funcType);
  auto &entry = funcOp.getBody().emplaceBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&entry);

  auto f32Ty = builder->getF32Type();
  auto i1Ty = builder->getI1Type();
  unsigned vectorLength = 4;

  auto condCst = builder->create<arith::ConstantOp>(
      loc(), builder->getIntegerAttr(i1Ty, 1));
  auto trueCst = builder->create<arith::ConstantOp>(
      loc(), builder->getFloatAttr(f32Ty, 1.0));
  auto falseCst = builder->create<arith::ConstantOp>(
      loc(), builder->getFloatAttr(f32Ty, 0.0));
  auto selectOp = builder->create<arith::SelectOp>(
      loc(), condCst.getResult(), trueCst.getResult(), falseCst.getResult());

  auto vecF32 = VectorType::get({static_cast<int64_t>(vectorLength)}, f32Ty);
  auto vecI8 = VectorType::get({static_cast<int64_t>(vectorLength)},
                               builder->getIntegerType(8));

  SmallVector<float> ones(vectorLength, 1.0f);
  SmallVector<float> zeros(vectorLength, 0.0f);
  auto trueVec = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(ones)));
  auto falseVec = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(zeros)));

  SmallVector<int8_t> i8ones(vectorLength, 1);
  auto condVecI8 = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecI8, llvm::ArrayRef(i8ones)));

  IRMapping mapper;
  mapper.map(condCst.getResult(), condVecI8.getResult());
  mapper.map(trueCst.getResult(), trueVec.getResult());
  mapper.map(falseCst.getResult(), falseVec.getResult());

  VectorizeContext context{*builder, loc(), mapper, vectorLength};
  SelectVectorizeHandler handler;
  auto *result = handler.rewriteOp(context, selectOp.getOperation());

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(isa<arith::SelectOp>(result));
}

// ---------- ExternElementwiseVectorizeHandler (lines 78-108) ----------

TEST_F(VectorizeTest, ExternElementwise_NvLibDevice) {
  auto f32Ty = builder->getF32Type();
  auto tensorTy = RankedTensorType::get({128}, f32Ty);
  auto funcType = builder->getFunctionType({tensorTy}, {tensorTy});
  auto funcOp =
      builder->create<func::FuncOp>(loc(), "test_extern_nv", funcType);
  auto &entry = funcOp.getBody().emplaceBlock();
  entry.addArgument(tensorTy, loc());
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&entry);

  auto externOp = builder->create<triton::ExternElementwiseOp>(
      loc(), tensorTy, ValueRange{entry.getArgument(0)},
      /*libname=*/"libdevice", /*libpath=*/"",
      /*symbol=*/"__nv_expf", /*pure=*/true);

  unsigned vectorLength = 4;
  auto vecF32 = VectorType::get({static_cast<int64_t>(vectorLength)}, f32Ty);

  SmallVector<float> zeros(vectorLength, 0.0f);
  auto inputVec = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(zeros)));

  IRMapping mapper;
  mapper.map(entry.getArgument(0), inputVec.getResult());

  VectorizeContext context{*builder, loc(), mapper, vectorLength};
  ExternElementwiseVectorizeHandler handler;
  auto *result = handler.rewriteOp(context, externOp.getOperation());

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(isa<VectorType>(result->getResult(0).getType()));
  auto resTy = cast<VectorType>(result->getResult(0).getType());
  EXPECT_EQ(resTy.getShape()[0], static_cast<int64_t>(vectorLength));

  handler.finalizeMappedResults(context, externOp.getOperation(), result);
  auto mapped = mapper.lookupOrNull(externOp.getResult());
  EXPECT_TRUE(mapped != nullptr);
}

TEST_F(VectorizeTest, ExternElementwise_MixedPrecision) {
  auto f32Ty = builder->getF32Type();
  auto tensorTy = RankedTensorType::get({128}, f32Ty);
  auto funcType = builder->getFunctionType({tensorTy, tensorTy}, {tensorTy});
  auto funcOp =
      builder->create<func::FuncOp>(loc(), "test_extern_mixed", funcType);
  auto &entry = funcOp.getBody().emplaceBlock();
  entry.addArgument(tensorTy, loc());
  entry.addArgument(tensorTy, loc());
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&entry);

  auto externOp = builder->create<triton::ExternElementwiseOp>(
      loc(), tensorTy, ValueRange{entry.getArgument(0), entry.getArgument(1)},
      /*libname=*/"libgcu", /*libpath=*/"",
      /*symbol=*/"__gcu_wadd_f32_bf16", /*pure=*/true);

  unsigned vectorLength = 4;
  auto vecF32 = VectorType::get({static_cast<int64_t>(vectorLength)}, f32Ty);

  SmallVector<float> zeros(vectorLength, 0.0f);
  auto vec0 = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(zeros)));
  auto vec1 = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(zeros)));

  IRMapping mapper;
  mapper.map(entry.getArgument(0), vec0.getResult());
  mapper.map(entry.getArgument(1), vec1.getResult());

  VectorizeContext context{*builder, loc(), mapper, vectorLength};
  ExternElementwiseVectorizeHandler handler;
  auto *result = handler.rewriteOp(context, externOp.getOperation());

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(isa<VectorType>(result->getResult(0).getType()));
}

// ---------- VectorizeEngine::vectorize (covers lines 115-124) ----------

TEST_F(VectorizeTest, Engine_DefaultHandlerVectorize) {
  auto f32Ty = builder->getF32Type();
  auto funcType = builder->getFunctionType({}, {});
  auto funcOp = builder->create<func::FuncOp>(loc(), "test_engine", funcType);
  auto &entry = funcOp.getBody().emplaceBlock();
  OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&entry);

  auto cst = builder->create<arith::ConstantOp>(
      loc(), builder->getFloatAttr(f32Ty, 1.0));
  auto neg = builder->create<arith::NegFOp>(loc(), cst.getResult());

  unsigned vectorLength = 4;
  auto vecF32 = VectorType::get({static_cast<int64_t>(vectorLength)}, f32Ty);
  SmallVector<float> ones(vectorLength, 1.0f);
  auto vecCst = builder->create<arith::ConstantOp>(
      loc(), DenseElementsAttr::get(vecF32, llvm::ArrayRef(ones)));

  IRMapping mapper;
  mapper.map(cst.getResult(), vecCst.getResult());

  VectorizeContext context{*builder, loc(), mapper, vectorLength};
  VectorizeEngine engine;
  auto *result = engine.vectorize(neg.getOperation(), context);

  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(isa<arith::NegFOp>(result));
}

#else
TEST(VectorizeTest, GCU300Placeholder) { SUCCEED(); }
#endif
