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

#include "Analysis/MaskAnalysis.h"
#include "Analysis/OpFoldResultUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

using namespace mlir;
using namespace mlir::triton::gcu;

class MaskAnalysisTestBase : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<OpBuilder> builder;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    scf::SCFDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect,
                    triton::gcu::TritonGCUDialect>();

    builder = std::make_unique<OpBuilder>(&ctx);
    auto l = builder->getUnknownLoc();

    module = ModuleOp::create(l);
    builder->setInsertionPointToStart(module.getBody());
    gpuModule = builder->create<gpu::GPUModuleOp>(l, "triton");
    builder->setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  Location loc() { return builder->getUnknownLoc(); }

  func::FuncOp makeFuncWithArgs(StringRef name, ArrayRef<Type> argTypes) {
    auto funcType = FunctionType::get(&ctx, argTypes, {});
    auto funcOp = builder->create<func::FuncOp>(loc(), name, funcType);
    funcOp.addEntryBlock();
    builder->setInsertionPointToStart(&funcOp.getBody().front());
    builder->create<func::ReturnOp>(loc());
    builder->setInsertionPointToStart(&funcOp.getBody().front());
    return funcOp;
  }

  MaskState makeScalarState(int64_t scalarVal) {
    MaskState s;
    s.scalar = builder->getIndexAttr(scalarVal);
    return s;
  }

  MaskState makeScalarStateWithDims(int64_t scalarVal,
                                    ArrayRef<int64_t> dimSizes) {
    MaskState s;
    s.scalar = builder->getIndexAttr(scalarVal);
    for (auto d : dimSizes)
      s.dims.push_back(builder->getIndexAttr(d));
    return s;
  }

  MaskState makeRangeState(int64_t startVal, int64_t endVal,
                           ArrayRef<int64_t> dimSizes) {
    MaskState s;
    s.start = builder->getIndexAttr(startVal);
    s.end = builder->getIndexAttr(endVal);
    for (auto d : dimSizes)
      s.dims.push_back(builder->getIndexAttr(d));
    return s;
  }
};

// ======================================================================
// MaskAnalysis::parse dispatching to parseDot  (gcu400 lines 174-177)
// parseDot body (gcu400 lines 464-476)
// ======================================================================

TEST_F(MaskAnalysisTestBase, ParseDot) {
  auto f32Type = builder->getF32Type();
  auto mat16 = RankedTensorType::get({16, 16}, f32Type);

  auto funcOp = makeFuncWithArgs("test_parse_dot", {mat16, mat16, mat16});
  Block &entry = funcOp.getBody().front();

  Value a = entry.getArgument(0);
  Value b = entry.getArgument(1);
  Value c = entry.getArgument(2);

  builder->setInsertionPoint(entry.getTerminator());
  auto dotOp = builder->create<triton::DotOp>(
      loc(), mat16, a, b, c,
      triton::InputPrecisionAttr::get(&ctx, triton::InputPrecision::IEEE),
      builder->getI32IntegerAttr(0));

  llvm::SmallDenseMap<Value, MaskState> knownMasks;
  knownMasks[c] = makeRangeState(0, 128, {16, 16});

  MaskState state;
#ifdef TEST_GCU400
  bool ok = MaskAnalysis::parse(*builder, loc(), dotOp.getResult(), state,
                                knownMasks);
  EXPECT_TRUE(ok);
#else
  MaskAnalysis::parse(*builder, loc(), dotOp.getResult(), state, knownMasks);
#endif

  EXPECT_FALSE(state.isEmpty());
  ASSERT_TRUE(getIntAttr(state.start).has_value());
  EXPECT_EQ(getIntAttr(state.start).value(), 0);
  ASSERT_TRUE(getIntAttr(state.end).has_value());
  EXPECT_EQ(getIntAttr(state.end).value(), 128);
  ASSERT_EQ(state.getRank(), 2);
  EXPECT_EQ(getIntAttr(state.dims[0]).value(), 16);
  EXPECT_EQ(getIntAttr(state.dims[1]).value(), 16);
}

// ======================================================================
// MaskAnalysis::parse dispatching to parseRemsi  (gcu400 lines 178-181)
// parseRemsi body (gcu400 lines 478-492)
// Also covers MaskState::addStates scalar+scalar path (lines 76-79)
// ======================================================================

TEST_F(MaskAnalysisTestBase, ParseRemsi) {
  auto i32Type = builder->getI32Type();
  auto tensorType = RankedTensorType::get({1024}, i32Type);

  auto funcOp = makeFuncWithArgs("test_parse_remsi", {tensorType, tensorType});
  Block &entry = funcOp.getBody().front();

  Value lhsArg = entry.getArgument(0);
  Value rhsArg = entry.getArgument(1);

  builder->setInsertionPoint(entry.getTerminator());
  auto remOp = builder->create<arith::RemSIOp>(loc(), lhsArg, rhsArg);

  llvm::SmallDenseMap<Value, MaskState> knownMasks;
  knownMasks[lhsArg] = makeScalarStateWithDims(10, {1024});
  knownMasks[rhsArg] = makeScalarStateWithDims(3, {1024});

  MaskState state;
#ifdef TEST_GCU400
  bool ok = MaskAnalysis::parse(*builder, loc(), remOp.getResult(), state,
                                knownMasks);
  EXPECT_TRUE(ok);
#else
  MaskAnalysis::parse(*builder, loc(), remOp.getResult(), state, knownMasks);
#endif

  EXPECT_FALSE(state.isEmpty());
  ASSERT_TRUE(getIntAttr(state.scalar).has_value());
  EXPECT_EQ(getIntAttr(state.scalar).value(), 13); // addOFRs(10, 3) = 13
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.dims[0]).value(), 1024);
}

// ======================================================================
// MaskAnalysis::parse dispatching to parseSelect  (gcu400 lines 182-185)
// parseSelect body (gcu400 lines 494-503)
// Also covers MaskState::setStates (gcu400 lines 101-113)
// ======================================================================

TEST_F(MaskAnalysisTestBase, ParseSelect) {
  auto i32Type = builder->getI32Type();
  auto i1Type = builder->getI1Type();
  auto tensorType = RankedTensorType::get({1024}, i32Type);
  auto boolTensorType = RankedTensorType::get({1024}, i1Type);

  auto funcOp = makeFuncWithArgs("test_parse_select",
                                 {boolTensorType, tensorType, tensorType});
  Block &entry = funcOp.getBody().front();

  Value cond = entry.getArgument(0);
  Value trueVal = entry.getArgument(1);
  Value falseVal = entry.getArgument(2);

  builder->setInsertionPoint(entry.getTerminator());
  auto selectOp =
      builder->create<arith::SelectOp>(loc(), cond, trueVal, falseVal);

  llvm::SmallDenseMap<Value, MaskState> knownMasks;
  knownMasks[trueVal] = makeRangeState(0, 512, {1024});
  knownMasks[falseVal] = makeRangeState(0, 256, {1024});

  MaskState state;
#ifdef TEST_GCU400
  bool ok = MaskAnalysis::parse(*builder, loc(), selectOp.getResult(), state,
                                knownMasks);
  EXPECT_TRUE(ok);
#else
  MaskAnalysis::parse(*builder, loc(), selectOp.getResult(), state, knownMasks);
#endif

  EXPECT_FALSE(state.isEmpty());
  ASSERT_TRUE(getIntAttr(state.start).has_value());
  EXPECT_EQ(getIntAttr(state.start).value(), 0);
  ASSERT_TRUE(getIntAttr(state.end).has_value());
  EXPECT_EQ(getIntAttr(state.end).value(), 512);
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.dims[0]).value(), 1024);
}

// ======================================================================
// MaskAnalysis::parse dispatching to parseReduce  (gcu400 lines 186-189)
// parseReduce body (gcu400 lines 505-526)
// Covers all branches: start, end, scalar, dim-skipping on reduce axis
// ======================================================================

TEST_F(MaskAnalysisTestBase, ParseReduce) {
  auto f32Type = builder->getF32Type();
  auto src2dType = RankedTensorType::get({16, 32}, f32Type);

  auto funcOp = makeFuncWithArgs("test_parse_reduce", {src2dType});
  Block &entry = funcOp.getBody().front();
  Value srcArg = entry.getArgument(0);

  builder->setInsertionPoint(entry.getTerminator());

  auto result1dType = RankedTensorType::get({32}, f32Type);
  OperationState reduceState(loc(), triton::ReduceOp::getOperationName());
  reduceState.addOperands(srcArg);
  reduceState.addAttribute("axis", builder->getI32IntegerAttr(0));
  reduceState.addTypes(result1dType);
  reduceState.addRegion();
  Operation *reduceOpRaw = builder->create(reduceState);
  auto reduceOp = cast<triton::ReduceOp>(reduceOpRaw);

  auto &combinerRegion = reduceOp.getCombineOp();
  Block *combiner = new Block();
  combinerRegion.push_back(combiner);
  combiner->addArgument(f32Type, loc());
  combiner->addArgument(f32Type, loc());
  OpBuilder combinerBuilder(combiner, combiner->begin());
  auto addF = combinerBuilder.create<arith::AddFOp>(
      loc(), combiner->getArgument(0), combiner->getArgument(1));
  combinerBuilder.create<triton::ReduceReturnOp>(loc(), ValueRange{addF});

  llvm::SmallDenseMap<Value, MaskState> knownMasks;
  MaskState srcMaskState;
  srcMaskState.start = builder->getIndexAttr(0);
  srcMaskState.end = builder->getIndexAttr(128);
  srcMaskState.scalar = builder->getIndexAttr(42);
  srcMaskState.dims.push_back(builder->getIndexAttr(16));
  srcMaskState.dims.push_back(builder->getIndexAttr(32));
  knownMasks[srcArg] = srcMaskState;

  MaskState state;
#ifdef TEST_GCU400
  bool ok = MaskAnalysis::parse(*builder, loc(), reduceOp->getResult(0), state,
                                knownMasks);
  EXPECT_TRUE(ok);
#else
  MaskAnalysis::parse(*builder, loc(), reduceOp->getResult(0), state,
                      knownMasks);
#endif

  EXPECT_FALSE(state.isEmpty());
  ASSERT_TRUE(getIntAttr(state.start).has_value());
  EXPECT_EQ(getIntAttr(state.start).value(), 0);
  ASSERT_TRUE(getIntAttr(state.end).has_value());
  EXPECT_EQ(getIntAttr(state.end).value(), 128);
  ASSERT_TRUE(getIntAttr(state.scalar).has_value());
  EXPECT_EQ(getIntAttr(state.scalar).value(), 42);
  // axis=0 removes dim[0]=16, keeps dim[1]=32
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.dims[0]).value(), 32);
}

// ======================================================================
// MaskAnalysis::parse dispatching to parseLoad  (gcu400 lines 190-193)
// parseLoad body (gcu400 lines 528-538)
// Also covers MaskState::setStates (gcu400 lines 101-113)
// ======================================================================

TEST_F(MaskAnalysisTestBase, ParseLoad) {
  auto f32Type = builder->getF32Type();
  auto ptrType = triton::PointerType::get(f32Type, 1);
  auto tensorPtrType = RankedTensorType::get({1024}, ptrType);

  auto funcOp = makeFuncWithArgs("test_parse_load", {tensorPtrType});
  Block &entry = funcOp.getBody().front();
  Value ptrArg = entry.getArgument(0);

  builder->setInsertionPoint(entry.getTerminator());
  auto loadOp = builder->create<triton::LoadOp>(
      loc(), ptrArg, /*mask=*/Value(), /*other=*/Value(),
      triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  llvm::SmallDenseMap<Value, MaskState> knownMasks;
  knownMasks[ptrArg] = makeRangeState(5, 100, {1024});

  MaskState state;
#ifdef TEST_GCU400
  bool ok = MaskAnalysis::parse(*builder, loc(), loadOp.getResult(), state,
                                knownMasks);
  EXPECT_TRUE(ok);
#else
  MaskAnalysis::parse(*builder, loc(), loadOp.getResult(), state, knownMasks);
#endif

  EXPECT_FALSE(state.isEmpty());
  ASSERT_TRUE(getIntAttr(state.start).has_value());
  EXPECT_EQ(getIntAttr(state.start).value(), 5);
  ASSERT_TRUE(getIntAttr(state.end).has_value());
  EXPECT_EQ(getIntAttr(state.end).value(), 100);
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.dims[0]).value(), 1024);
}

// ======================================================================
// MaskAnalysis::parse else fallback  (gcu400 lines 202-204)
// Returns false for unrecognized ops on gcu400.
// ======================================================================

#ifdef TEST_GCU400
TEST_F(MaskAnalysisTestBase, ParseUnrecognizedOp) {
  auto i32Type = builder->getI32Type();
  auto tensorType = RankedTensorType::get({1024}, i32Type);

  auto funcOp = makeFuncWithArgs("test_parse_unrec", {tensorType, tensorType});
  Block &entry = funcOp.getBody().front();

  Value lhsArg = entry.getArgument(0);
  Value rhsArg = entry.getArgument(1);

  builder->setInsertionPoint(entry.getTerminator());
  auto maxOp = builder->create<arith::MaxSIOp>(loc(), lhsArg, rhsArg);

  llvm::SmallDenseMap<Value, MaskState> knownMasks;

  MaskState state;
  bool ok = MaskAnalysis::parse(*builder, loc(), maxOp.getResult(), state,
                                knownMasks);
  EXPECT_FALSE(ok);
}
#endif

// ======================================================================
// MaskState::setStates — scalar-only path (no start/end, no dims)
// Verifies lines 107-108 (scalar branch) without start/end/dims
// ======================================================================

TEST_F(MaskAnalysisTestBase, SetStatesScalarOnly) {
  MaskState src;
  src.scalar = builder->getIndexAttr(99);

  MaskState dst;
  dst.setStates(*builder, loc(), src);

  EXPECT_TRUE(getIntAttr(dst.scalar).has_value());
  EXPECT_EQ(getIntAttr(dst.scalar).value(), 99);
  EXPECT_FALSE(dst.start);
  EXPECT_FALSE(dst.end);
  EXPECT_EQ(dst.getRank(), 0);
}

// ======================================================================
// MaskState::addStates — one scalar, one with start/end
// Covers the addStateScalar path (lines 82-85)
// ======================================================================

TEST_F(MaskAnalysisTestBase, AddStatesOneScalar) {
  MaskState lhs;
  lhs.scalar = builder->getIndexAttr(10);

  MaskState rhs;
  rhs.start = builder->getIndexAttr(0);
  rhs.end = builder->getIndexAttr(128);
  rhs.dims.push_back(builder->getIndexAttr(1024));

  MaskState result;
  result.addStates(*builder, loc(), lhs, rhs);

  ASSERT_TRUE(getIntAttr(result.start).has_value());
  EXPECT_EQ(getIntAttr(result.start).value(), 10); // 0 + 10
  ASSERT_TRUE(getIntAttr(result.end).has_value());
  EXPECT_EQ(getIntAttr(result.end).value(), 138); // 128 + 10
  ASSERT_EQ(result.getRank(), 1);
  EXPECT_EQ(getIntAttr(result.dims[0]).value(), 1024);
}

// ======================================================================
// MaskState::minStates
// Covers lines 88-99
// ======================================================================

TEST_F(MaskAnalysisTestBase, MinStates) {
  MaskState lhs;
  lhs.dims.push_back(builder->getIndexAttr(64));
  lhs.dims.push_back(builder->getIndexAttr(128));

  MaskState rhs;
  rhs.dims.push_back(builder->getIndexAttr(32));
  rhs.dims.push_back(builder->getIndexAttr(256));

  MaskState result;
  result.minStates(*builder, loc(), lhs, rhs);

  ASSERT_EQ(result.getRank(), 2);
  EXPECT_EQ(getIntAttr(result.dims[0]).value(), 32);
  EXPECT_EQ(getIntAttr(result.dims[1]).value(), 128);
}
