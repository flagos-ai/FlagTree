// Copyright 2025 Enflame. All Rights Reserved.
#include <memory>

#include "gtest/gtest.h"

#include "ConstantUtil.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::triton::gcu;

class ConstantUtilTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect>();
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

// ---------- createConstantZero ----------

TEST_F(ConstantUtilTest, Zero_I32) {
  auto v = createConstantZero(*builder, loc(), builder->getI32Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(isa<IntegerType>(v.getType()));
}

TEST_F(ConstantUtilTest, Zero_I64) {
  auto v = createConstantZero(*builder, loc(), builder->getI64Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(isa<IntegerType>(v.getType()));
}

TEST_F(ConstantUtilTest, Zero_F32) {
  auto v = createConstantZero(*builder, loc(), builder->getF32Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF32());
}

TEST_F(ConstantUtilTest, Zero_F16) {
  auto v = createConstantZero(*builder, loc(), builder->getF16Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF16());
}

TEST_F(ConstantUtilTest, Zero_BF16) {
  auto v = createConstantZero(*builder, loc(), builder->getBF16Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isBF16());
}

TEST_F(ConstantUtilTest, Zero_F64) {
  auto v = createConstantZero(*builder, loc(), builder->getF64Type());
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF64());
}

TEST_F(ConstantUtilTest, Zero_Float8E4M3B11FNUZ) {
  auto v =
      createConstantZero(*builder, loc(), Float8E4M3B11FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Zero_Float8E4M3FNUZ) {
  auto v = createConstantZero(*builder, loc(), Float8E4M3FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Zero_Float8E5M2FNUZ) {
  auto v = createConstantZero(*builder, loc(), Float8E5M2FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Zero_Float8E4M3FN) {
  auto v = createConstantZero(*builder, loc(), Float8E4M3FNType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Zero_Float8E5M2) {
  auto v = createConstantZero(*builder, loc(), Float8E5M2Type::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

// ---------- createConstantInf ----------

TEST_F(ConstantUtilTest, Inf_F32) {
  auto v = createConstantInf(*builder, loc(), builder->getF32Type(), false);
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF32());
}

TEST_F(ConstantUtilTest, Inf_F16) {
  auto v = createConstantInf(*builder, loc(), builder->getF16Type(), false);
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF16());
}

TEST_F(ConstantUtilTest, Inf_BF16) {
  auto v = createConstantInf(*builder, loc(), builder->getBF16Type(), false);
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isBF16());
}

TEST_F(ConstantUtilTest, Inf_F64) {
  auto v = createConstantInf(*builder, loc(), builder->getF64Type(), false);
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF64());
}

TEST_F(ConstantUtilTest, NegInf_F32) {
  auto v = createConstantInf(*builder, loc(), builder->getF32Type(), true);
  ASSERT_TRUE(static_cast<bool>(v));
  EXPECT_TRUE(v.getType().isF32());
}

TEST_F(ConstantUtilTest, Inf_Float8E4M3B11FNUZ) {
  auto v = createConstantInf(*builder, loc(), Float8E4M3B11FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Inf_Float8E4M3FNUZ) {
  auto v = createConstantInf(*builder, loc(), Float8E4M3FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Inf_Float8E5M2FNUZ) {
  auto v = createConstantInf(*builder, loc(), Float8E5M2FNUZType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Inf_Float8E4M3FN) {
  auto v = createConstantInf(*builder, loc(), Float8E4M3FNType::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}

TEST_F(ConstantUtilTest, Inf_Float8E5M2) {
  auto v = createConstantInf(*builder, loc(), Float8E5M2Type::get(&ctx));
  ASSERT_TRUE(static_cast<bool>(v));
}
