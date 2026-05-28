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

#include "Analysis/AxisInfoEx.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gcu;

class AxisInfoExTest : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    scf::SCFDialect, LLVM::LLVMDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect,
                    triton::gcu::TritonGCUDialect>();

    builder = std::make_unique<OpBuilder>(&ctx);
    module = ModuleOp::create(loc());
    builder->setInsertionPointToStart(module.getBody());
  }

  Location loc() { return builder->getUnknownLoc(); }

  func::FuncOp createFunc(StringRef name, ArrayRef<Type> argTypes) {
    auto funcType = FunctionType::get(&ctx, argTypes, {});
    auto funcOp = builder->create<func::FuncOp>(loc(), name, funcType);
    funcOp.addEntryBlock();
    builder->setInsertionPointToStart(&funcOp.getBody().front());
    builder->create<func::ReturnOp>(loc());
    builder->setInsertionPointToStart(&funcOp.getBody().front());
    return funcOp;
  }

  RankedTensorType i32TensorTy(int64_t size) {
    return RankedTensorType::get({size}, builder->getI32Type());
  }

  Value createDenseConstant(RankedTensorType ty, int64_t val) {
    auto attr =
        DenseElementsAttr::get(ty, IntegerAttr::get(ty.getElementType(), val));
    return builder->create<arith::ConstantOp>(loc(), attr);
  }

  std::unique_ptr<ModuleAxisInfoExAnalysis> runAnalysis() {
    return std::make_unique<ModuleAxisInfoExAnalysis>(module);
  }
};

// ======================================================================
// DivOp Case 1: lhs / 1 → preserves lhs info
// ======================================================================

TEST_F(AxisInfoExTest, DivByOne) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_div_one", {tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(16));
  funcOp.setArgAttr(0, "tt.contiguity", builder->getI64IntegerAttr(16));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst1 = createDenseConstant(tensorTy, 1);
  auto divOp = builder->create<arith::DivSIOp>(loc(), arg0, cst1);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(divOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 16);
  EXPECT_EQ(info->getContinualSize(0), 16);
  EXPECT_EQ(info->getContinualInterval(0), 1);
}

// ======================================================================
// DivOp Case 1b: 0 / rhs → preserves lhs (zero) info
// ======================================================================

TEST_F(AxisInfoExTest, DivZeroByAny) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_div_zero", {tensorTy});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst0 = createDenseConstant(tensorTy, 0);
  Value arg0 = funcOp.getArgument(0);
  auto divOp = builder->create<arith::DivSIOp>(loc(), cst0, arg0);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(divOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 0);
}

// ======================================================================
// DivOp Case 2: constant / constant
// ======================================================================

TEST_F(AxisInfoExTest, DivConstByConst) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_div_const", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto divOp = builder->create<arith::DivSIOp>(loc(), cst10, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(divOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 3);
}

// ======================================================================
// DivOp Case 3: strided lhs / constant rhs (covers gcu300 uncovered)
// ======================================================================

TEST_F(AxisInfoExTest, DivStridedByConst) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_div_strided", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  auto rangeOp = builder->create<triton::MakeRangeOp>(loc(), tensorTy,
                                                      /*start=*/0, /*end=*/16);
  Value cst4 = createDenseConstant(tensorTy, 4);
  auto divOp = builder->create<arith::DivSIOp>(loc(), rangeOp, cst4);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(divOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualInterval(0), 0);
}

// ======================================================================
// RemOp Case 1: lhs % 1 → constant 0
// ======================================================================

TEST_F(AxisInfoExTest, RemModOne) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_rem_one", {tensorTy});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst1 = createDenseConstant(tensorTy, 1);
  auto remOp = builder->create<arith::RemSIOp>(loc(), arg0, cst1);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(remOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 0);
  EXPECT_EQ(info->getContinualSize(0), 16);
  EXPECT_EQ(info->getContinualInterval(0), 0);
}

// ======================================================================
// RemOp Case 2: constant % constant
// ======================================================================

TEST_F(AxisInfoExTest, RemConstByConst) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_rem_const", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto remOp = builder->create<arith::RemSIOp>(loc(), cst10, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(remOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

// ======================================================================
// RemOp Case 4: strided contiguous lhs, constant rhs
// ======================================================================

TEST_F(AxisInfoExTest, RemStridedContiguous) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_rem_strided", {tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(16));
  funcOp.setArgAttr(0, "tt.contiguity", builder->getI64IntegerAttr(8));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst4 = createDenseConstant(tensorTy, 4);
  auto remOp = builder->create<arith::RemSIOp>(loc(), arg0, cst4);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(remOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualInterval(0), 1 % 4);
}

// ======================================================================
// RemOp Case 5: strided constant lhs, constant rhs
// ======================================================================

TEST_F(AxisInfoExTest, RemStridedConstant) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_rem_strcst", {tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(4));
  funcOp.setArgAttr(0, "tt.constancy", builder->getI64IntegerAttr(8));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto remOp = builder->create<arith::RemSIOp>(loc(), arg0, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(remOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualSize(0), 8);
  EXPECT_EQ(info->getContinualInterval(0), 0);
}

// ======================================================================
// CmpOp: both constants with various predicates
// ======================================================================

TEST_F(AxisInfoExTest, CmpIBothConstantsSlt) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_slt", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::slt,
                                              cst3, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsEq) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_eq", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst5 = createDenseConstant(tensorTy, 5);
  Value cst5b = createDenseConstant(tensorTy, 5);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::eq,
                                              cst5, cst5b);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsNe) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_ne", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::ne,
                                              cst3, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsSle) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_sle", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst3b = createDenseConstant(tensorTy, 3);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::sle,
                                              cst3, cst3b);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsSgt) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_sgt", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::sgt,
                                              cst10, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsSge) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_sge", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst10b = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::sge,
                                              cst10, cst10b);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsUlt) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_ult", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::ult,
                                              cst3, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsUle) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_ule", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::ule,
                                              cst3, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsUgt) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_ugt", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::ugt,
                                              cst10, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

TEST_F(AxisInfoExTest, CmpIBothConstantsUge) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cmpi_uge", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst10 = createDenseConstant(tensorTy, 10);
  Value cst10b = createDenseConstant(tensorTy, 10);
  auto cmpOp = builder->create<arith::CmpIOp>(loc(), arith::CmpIPredicate::uge,
                                              cst10, cst10b);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(cmpOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 1);
}

// ======================================================================
// SelectOp: constant false condition picks rhs
// ======================================================================

TEST_F(AxisInfoExTest, SelectConstantFalse) {
  auto tensorTy = i32TensorTy(16);
  auto i1TensorTy = RankedTensorType::get({16}, builder->getI1Type());
  auto funcOp = createFunc("test_select_false", {tensorTy, tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(16));
  funcOp.setArgAttr(1, "tt.divisibility", builder->getI64IntegerAttr(8));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  auto falseAttr =
      DenseElementsAttr::get(i1TensorTy, builder->getBoolAttr(false));
  Value cond = builder->create<arith::ConstantOp>(loc(), falseAttr);
  Value arg0 = funcOp.getArgument(0);
  Value arg1 = funcOp.getArgument(1);
  auto selectOp = builder->create<arith::SelectOp>(loc(), cond, arg0, arg1);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(selectOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 8);
}

// ======================================================================
// SelectOp: constant true condition picks lhs
// ======================================================================

TEST_F(AxisInfoExTest, SelectConstantTrue) {
  auto tensorTy = i32TensorTy(16);
  auto i1TensorTy = RankedTensorType::get({16}, builder->getI1Type());
  auto funcOp = createFunc("test_select_true", {tensorTy, tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(16));
  funcOp.setArgAttr(1, "tt.divisibility", builder->getI64IntegerAttr(8));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  auto trueAttr =
      DenseElementsAttr::get(i1TensorTy, builder->getBoolAttr(true));
  Value cond = builder->create<arith::ConstantOp>(loc(), trueAttr);
  Value arg0 = funcOp.getArgument(0);
  Value arg1 = funcOp.getArgument(1);
  auto selectOp = builder->create<arith::SelectOp>(loc(), cond, arg0, arg1);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(selectOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 16);
}

// ======================================================================
// LogicalOp: AndI / OrI / XOrI with both constants
// ======================================================================

TEST_F(AxisInfoExTest, LogicalAndConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_andi", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst6 = createDenseConstant(tensorTy, 6);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto andOp = builder->create<arith::AndIOp>(loc(), cst6, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(andOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 6 & 3);
}

TEST_F(AxisInfoExTest, LogicalOrConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_ori", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst6 = createDenseConstant(tensorTy, 6);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto orOp = builder->create<arith::OrIOp>(loc(), cst6, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(orOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 6 | 3);
}

TEST_F(AxisInfoExTest, LogicalXorConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_xori", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst6 = createDenseConstant(tensorTy, 6);
  Value cst3 = createDenseConstant(tensorTy, 3);
  auto xorOp = builder->create<arith::XOrIOp>(loc(), cst6, cst3);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(xorOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 6 ^ 3);
}

// ======================================================================
// ShLI: constant shift → continual size/interval propagation
// ======================================================================

TEST_F(AxisInfoExTest, ShLIConstantShift) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_shli", {tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(4));
  funcOp.setArgAttr(0, "tt.contiguity", builder->getI64IntegerAttr(16));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst2 = createDenseConstant(tensorTy, 2);
  auto shlOp = builder->create<arith::ShLIOp>(loc(), arg0, cst2);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(shlOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 16);
  EXPECT_EQ(info->getContinualSize(0), 16);
  EXPECT_EQ(info->getContinualInterval(0), 4);
}

// ======================================================================
// ShLI: both constants
// ======================================================================

TEST_F(AxisInfoExTest, ShLIBothConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_shli_const", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst3 = createDenseConstant(tensorTy, 3);
  Value cst2 = createDenseConstant(tensorTy, 2);
  auto shlOp = builder->create<arith::ShLIOp>(loc(), cst3, cst2);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(shlOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 3 << 2);
}

// ======================================================================
// ShRUI: shift by 0 preserves info
// ======================================================================

TEST_F(AxisInfoExTest, ShRUIShiftZero) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_shrui_zero", {tensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(8));
  funcOp.setArgAttr(0, "tt.contiguity", builder->getI64IntegerAttr(16));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value cst0 = createDenseConstant(tensorTy, 0);
  auto shrOp = builder->create<arith::ShRUIOp>(loc(), arg0, cst0);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(shrOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualSize(0), 16);
  EXPECT_EQ(info->getContinualInterval(0), 1);
}

// ======================================================================
// ShRUI: both constants
// ======================================================================

TEST_F(AxisInfoExTest, ShRUIBothConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_shrui_const", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst12 = createDenseConstant(tensorTy, 12);
  Value cst2 = createDenseConstant(tensorTy, 2);
  auto shrOp = builder->create<arith::ShRUIOp>(loc(), cst12, cst2);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(shrOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 12 >> 2);
}

// ======================================================================
// MaxSI / MinSI: both constants
// ======================================================================

TEST_F(AxisInfoExTest, MaxSIBothConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_maxsi", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst5 = createDenseConstant(tensorTy, 5);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto maxOp = builder->create<arith::MaxSIOp>(loc(), cst5, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(maxOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 10);
}

TEST_F(AxisInfoExTest, MinSIBothConstants) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_minsi", {});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value cst5 = createDenseConstant(tensorTy, 5);
  Value cst10 = createDenseConstant(tensorTy, 10);
  auto minOp = builder->create<arith::MinSIOp>(loc(), cst5, cst10);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(minOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_TRUE(info->getConstantValue().has_value());
  EXPECT_EQ(info->getConstantValue().value(), 5);
}

// ======================================================================
// CastOp: ExtSIOp passes through axis info
// ======================================================================

TEST_F(AxisInfoExTest, CastOpPassthrough) {
  auto i16TensorTy = RankedTensorType::get({16}, builder->getI16Type());
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_cast", {i16TensorTy});
  funcOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(8));

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  auto extOp = builder->create<arith::ExtSIOp>(loc(), tensorTy, arg0);

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(extOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 8);
}

// ======================================================================
// visitOperation: tt.contiguity attribute override
// ======================================================================

TEST_F(AxisInfoExTest, ContiguityAttributeOverride) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_contiguity_attr", {tensorTy, tensorTy});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value arg1 = funcOp.getArgument(1);
  auto addOp = builder->create<arith::AddIOp>(loc(), arg0, arg1);

  auto attrTy = RankedTensorType::get({1}, builder->getI32Type());
  addOp->setDiscardableAttr(
      "tt.contiguity",
      DenseElementsAttr::get(attrTy, builder->getI32IntegerAttr(8)));

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(addOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualSize(0), 8);
  EXPECT_EQ(info->getContinualInterval(0), 1);
}

// ======================================================================
// visitOperation: tt.constancy attribute override
// ======================================================================

TEST_F(AxisInfoExTest, ConstancyAttributeOverride) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_constancy_attr", {tensorTy, tensorTy});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value arg1 = funcOp.getArgument(1);
  auto addOp = builder->create<arith::AddIOp>(loc(), arg0, arg1);

  auto attrTy = RankedTensorType::get({1}, builder->getI32Type());
  addOp->setDiscardableAttr(
      "tt.constancy",
      DenseElementsAttr::get(attrTy, builder->getI32IntegerAttr(4)));

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(addOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getContinualSize(0), 4);
  EXPECT_EQ(info->getContinualInterval(0), 0);
}

// ======================================================================
// initPessimisticStateFromFunc: DenseElementsAttr for tt.divisibility
// ======================================================================

TEST_F(AxisInfoExTest, DenseElementsAttrFuncArg) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_dense_attr", {tensorTy});

  auto attrTy = RankedTensorType::get({1}, builder->getI32Type());
  funcOp.setArgAttr(
      0, "tt.divisibility",
      DenseElementsAttr::get(attrTy, builder->getI32IntegerAttr(32)));

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(funcOp.getArgument(0));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 32);
}

// ======================================================================
// getPessimisticValueState: discardable attrs on defining op
// ======================================================================

TEST_F(AxisInfoExTest, DiscardableAttrOnDefiningOp) {
  auto tensorTy = i32TensorTy(16);
  auto funcOp = createFunc("test_disc_attr", {tensorTy, tensorTy});

  builder->setInsertionPoint(funcOp.getBody().front().getTerminator());
  Value arg0 = funcOp.getArgument(0);
  Value arg1 = funcOp.getArgument(1);
  auto addOp = builder->create<arith::AddIOp>(loc(), arg0, arg1);

  auto attrTy = RankedTensorType::get({1}, builder->getI32Type());
  addOp->setDiscardableAttr(
      "tt.divisibility",
      DenseElementsAttr::get(attrTy, builder->getI32IntegerAttr(64)));
  addOp->setDiscardableAttr(
      "tt.constancy",
      DenseElementsAttr::get(attrTy, builder->getI32IntegerAttr(4)));

  auto analysis = runAnalysis();
  auto *info = analysis->getAxisInfoEx(addOp.getResult());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->getDivisibility(0), 64);
  EXPECT_EQ(info->getContinualInterval(0), 0);
}

// ======================================================================
// update(): call graph propagation of divisibility
// ======================================================================

TEST_F(AxisInfoExTest, UpdateCallGraphDivisibility) {
  auto i32Ty = builder->getI32Type();

  auto calleeType = FunctionType::get(&ctx, {i32Ty}, {});
  auto calleeFuncOp =
      builder->create<func::FuncOp>(loc(), "callee_div", calleeType);
  calleeFuncOp.addEntryBlock();
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&calleeFuncOp.getBody().front());
    builder->create<func::ReturnOp>(loc());
  }

  auto callerType = FunctionType::get(&ctx, {i32Ty}, {});
  auto callerFuncOp =
      builder->create<func::FuncOp>(loc(), "caller_div", callerType);
  callerFuncOp.addEntryBlock();
  callerFuncOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(16));
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&callerFuncOp.getBody().front());
    Value callerArg = callerFuncOp.getArgument(0);
    builder->create<func::CallOp>(loc(), calleeFuncOp, ValueRange{callerArg});
    builder->create<func::ReturnOp>(loc());
  }

  auto analysis = runAnalysis();

  auto attr = calleeFuncOp.getArgAttrOfType<IntegerAttr>(0, "tt.divisibility");
  ASSERT_TRUE(attr != nullptr);
  EXPECT_EQ(attr.getInt(), 16);
}

// ======================================================================
// update(): call graph propagation of constancy
// ======================================================================

TEST_F(AxisInfoExTest, UpdateCallGraphConstancy) {
  auto i32Ty = builder->getI32Type();

  auto calleeType = FunctionType::get(&ctx, {i32Ty}, {});
  auto calleeFuncOp =
      builder->create<func::FuncOp>(loc(), "callee_cst", calleeType);
  calleeFuncOp.addEntryBlock();
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&calleeFuncOp.getBody().front());
    builder->create<func::ReturnOp>(loc());
  }

  auto callerType = FunctionType::get(&ctx, {i32Ty}, {});
  auto callerFuncOp =
      builder->create<func::FuncOp>(loc(), "caller_cst", callerType);
  callerFuncOp.addEntryBlock();
  callerFuncOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(8));
  callerFuncOp.setArgAttr(0, "tt.constancy", builder->getI64IntegerAttr(4));
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&callerFuncOp.getBody().front());
    Value callerArg = callerFuncOp.getArgument(0);
    builder->create<func::CallOp>(loc(), calleeFuncOp, ValueRange{callerArg});
    builder->create<func::ReturnOp>(loc());
  }

  auto analysis = runAnalysis();

  auto constAttr =
      calleeFuncOp.getArgAttrOfType<IntegerAttr>(0, "tt.constancy");
  ASSERT_TRUE(constAttr != nullptr);
  EXPECT_EQ(constAttr.getInt(), 4);
}

// ======================================================================
// update(): call graph propagation of contiguity
// ======================================================================

TEST_F(AxisInfoExTest, UpdateCallGraphContiguity) {
  auto i32Ty = builder->getI32Type();

  auto calleeType = FunctionType::get(&ctx, {i32Ty}, {});
  auto calleeFuncOp =
      builder->create<func::FuncOp>(loc(), "callee_cont", calleeType);
  calleeFuncOp.addEntryBlock();
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&calleeFuncOp.getBody().front());
    builder->create<func::ReturnOp>(loc());
  }

  auto callerType = FunctionType::get(&ctx, {i32Ty}, {});
  auto callerFuncOp =
      builder->create<func::FuncOp>(loc(), "caller_cont", callerType);
  callerFuncOp.addEntryBlock();
  callerFuncOp.setArgAttr(0, "tt.divisibility", builder->getI64IntegerAttr(8));
  callerFuncOp.setArgAttr(0, "tt.contiguity", builder->getI64IntegerAttr(8));
  {
    OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&callerFuncOp.getBody().front());
    Value callerArg = callerFuncOp.getArgument(0);
    builder->create<func::CallOp>(loc(), calleeFuncOp, ValueRange{callerArg});
    builder->create<func::ReturnOp>(loc());
  }

  auto analysis = runAnalysis();

  auto contAttr =
      calleeFuncOp.getArgAttrOfType<IntegerAttr>(0, "tt.contiguity");
  ASSERT_TRUE(contAttr != nullptr);
  EXPECT_EQ(contAttr.getInt(), 8);
}
