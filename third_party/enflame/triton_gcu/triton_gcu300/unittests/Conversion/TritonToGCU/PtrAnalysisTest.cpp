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

#include "Analysis/OpFoldResultUtils.h"
#include "Analysis/PtrAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::gcu;

// LLVM/MLIR is built with -fno-rtti while gtest uses RTTI.
// We cannot subclass PatternRewriter (would need typeinfo).
// IRRewriter and PatternRewriter share the same base (RewriterBase) and
// neither adds virtual methods, so they have identical vtable layouts.
// With RTTI off, reinterpret_cast is safe and matches LLVM convention.
static PatternRewriter &asPatternRewriter(IRRewriter &rw) {
  return reinterpret_cast<PatternRewriter &>(rw);
}

// Shared test fixture: loads all dialects needed by PtrAnalysis and creates a
// module -> gpu.module skeleton.
class PtrAnalysisTestBase : public ::testing::Test {
protected:
  MLIRContext ctx;
  std::unique_ptr<IRRewriter> irRewriter;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  void SetUp() override {
    ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    scf::SCFDialect, cf::ControlFlowDialect,
                    triton::TritonDialect, triton::gpu::TritonGPUDialect,
                    triton::gcu::TritonGCUDialect>();

    irRewriter = std::make_unique<IRRewriter>(&ctx);
    auto l = irRewriter->getUnknownLoc();

    module = ModuleOp::create(l);
    irRewriter->setInsertionPointToStart(module.getBody());
    gpuModule = irRewriter->create<gpu::GPUModuleOp>(l, "triton");
    irRewriter->setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  Location loc() { return irRewriter->getUnknownLoc(); }

  PatternRewriter &rewriter() { return asPatternRewriter(*irRewriter); }

  // Create a PtrState with constant offsets/strides/sizes (rank-1).
  PtrState makeState(int64_t offset, int64_t stride, int64_t size) {
    PtrState s;
    s.offsets.push_back(irRewriter->getIndexAttr(offset));
    s.strides.push_back(irRewriter->getIndexAttr(stride));
    s.sizes.push_back(irRewriter->getIndexAttr(size));
    return s;
  }

  func::FuncOp makeFuncWithArgs(StringRef name, ArrayRef<Type> argTypes) {
    auto funcType = FunctionType::get(&ctx, argTypes, {});
    auto funcOp = irRewriter->create<func::FuncOp>(loc(), name, funcType);
    funcOp.addEntryBlock();
    irRewriter->setInsertionPointToStart(&funcOp.getBody().front());
    irRewriter->create<func::ReturnOp>(loc());
    irRewriter->setInsertionPointToStart(&funcOp.getBody().front());
    return funcOp;
  }
};

// ======================================================================
// PtrState::remState  (lines 160-173)
// ======================================================================

class PtrStateRemTest : public PtrAnalysisTestBase {};

TEST_F(PtrStateRemTest, ConstantAttributes) {
  PtrState lhs = makeState(10, 2, 1024);
  PtrState rhs = makeState(3, 1, 1024);

  PtrState result;
  result.remState(*irRewriter, loc(), lhs, rhs);

  ASSERT_EQ(result.getRank(), 1);
  auto offsetAttr = getIntAttr(result.offsets[0]);
  ASSERT_TRUE(offsetAttr.has_value());
  EXPECT_EQ(offsetAttr.value(), 1); // 10 % 3
  auto strideAttr = getIntAttr(result.strides[0]);
  ASSERT_TRUE(strideAttr.has_value());
  EXPECT_EQ(strideAttr.value(), 2); // copied from lhs
  auto sizeAttr = getIntAttr(result.sizes[0]);
  ASSERT_TRUE(sizeAttr.has_value());
  EXPECT_EQ(sizeAttr.value(), 1024);
}

TEST_F(PtrStateRemTest, MultiDim) {
  PtrState lhs;
  lhs.offsets.push_back(irRewriter->getIndexAttr(20));
  lhs.offsets.push_back(irRewriter->getIndexAttr(15));
  lhs.strides.push_back(irRewriter->getIndexAttr(4));
  lhs.strides.push_back(irRewriter->getIndexAttr(1));
  lhs.sizes.push_back(irRewriter->getIndexAttr(128));
  lhs.sizes.push_back(irRewriter->getIndexAttr(64));

  PtrState rhs;
  rhs.offsets.push_back(irRewriter->getIndexAttr(7));
  rhs.offsets.push_back(irRewriter->getIndexAttr(4));
  rhs.strides.push_back(irRewriter->getIndexAttr(1));
  rhs.strides.push_back(irRewriter->getIndexAttr(1));
  rhs.sizes.push_back(irRewriter->getIndexAttr(128));
  rhs.sizes.push_back(irRewriter->getIndexAttr(64));

  PtrState result;
  result.remState(*irRewriter, loc(), lhs, rhs);

  ASSERT_EQ(result.getRank(), 2);
  EXPECT_EQ(getIntAttr(result.offsets[0]).value(), 20 % 7); // 6
  EXPECT_EQ(getIntAttr(result.offsets[1]).value(), 15 % 4); // 3
}

// ======================================================================
// PtrState::divState  (lines 176-189)
// ======================================================================

class PtrStateDivTest : public PtrAnalysisTestBase {};

TEST_F(PtrStateDivTest, ConstantAttributes) {
  PtrState lhs = makeState(10, 2, 1024);
  PtrState rhs = makeState(3, 1, 1024);

  PtrState result;
  result.divState(*irRewriter, loc(), lhs, rhs);

  ASSERT_EQ(result.getRank(), 1);
  auto offsetAttr = getIntAttr(result.offsets[0]);
  ASSERT_TRUE(offsetAttr.has_value());
  EXPECT_EQ(offsetAttr.value(), 3); // 10 / 3
  auto strideAttr = getIntAttr(result.strides[0]);
  ASSERT_TRUE(strideAttr.has_value());
  EXPECT_EQ(strideAttr.value(), 2); // copied from lhs
}

TEST_F(PtrStateDivTest, MultiDim) {
  PtrState lhs;
  lhs.offsets.push_back(irRewriter->getIndexAttr(21));
  lhs.offsets.push_back(irRewriter->getIndexAttr(100));
  lhs.strides.push_back(irRewriter->getIndexAttr(4));
  lhs.strides.push_back(irRewriter->getIndexAttr(1));
  lhs.sizes.push_back(irRewriter->getIndexAttr(128));
  lhs.sizes.push_back(irRewriter->getIndexAttr(64));

  PtrState rhs;
  rhs.offsets.push_back(irRewriter->getIndexAttr(7));
  rhs.offsets.push_back(irRewriter->getIndexAttr(10));
  rhs.strides.push_back(irRewriter->getIndexAttr(1));
  rhs.strides.push_back(irRewriter->getIndexAttr(1));
  rhs.sizes.push_back(irRewriter->getIndexAttr(128));
  rhs.sizes.push_back(irRewriter->getIndexAttr(64));

  PtrState result;
  result.divState(*irRewriter, loc(), lhs, rhs);

  ASSERT_EQ(result.getRank(), 2);
  EXPECT_EQ(getIntAttr(result.offsets[0]).value(), 21 / 7);   // 3
  EXPECT_EQ(getIntAttr(result.offsets[1]).value(), 100 / 10); // 10
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for RemSIOp  (lines 347-350)
// and visitOperandRem body (lines 481-491)
// ======================================================================

class VisitOperandTest : public PtrAnalysisTestBase {};

TEST_F(VisitOperandTest, RemSIOp) {
  auto i32Tensor = RankedTensorType::get({1024}, irRewriter->getI32Type());
  auto funcOp = makeFuncWithArgs("test_rem", {i32Tensor, i32Tensor});
  Block &entry = funcOp.getBody().front();

  Value lhsArg = entry.getArgument(0);
  Value rhsArg = entry.getArgument(1);

  irRewriter->setInsertionPoint(entry.getTerminator());
  auto remOp = irRewriter->create<arith::RemSIOp>(loc(), lhsArg, rhsArg);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[lhsArg] = makeState(10, 1, 1024);
  knownPtrs[rhsArg] = makeState(3, 1, 1024);

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), remOp.getResult(),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  ASSERT_EQ(state.getRank(), 1);
  auto offsetAttr = getIntAttr(state.offsets[0]);
  ASSERT_TRUE(offsetAttr.has_value());
  EXPECT_EQ(offsetAttr.value(), 1); // 10 % 3
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for DivSIOp  (lines 351-354)
// and visitOperandDiv body (lines 493-503)
// ======================================================================

TEST_F(VisitOperandTest, DivSIOp) {
  auto i32Tensor = RankedTensorType::get({1024}, irRewriter->getI32Type());
  auto funcOp = makeFuncWithArgs("test_div", {i32Tensor, i32Tensor});
  Block &entry = funcOp.getBody().front();

  Value lhsArg = entry.getArgument(0);
  Value rhsArg = entry.getArgument(1);

  irRewriter->setInsertionPoint(entry.getTerminator());
  auto divOp = irRewriter->create<arith::DivSIOp>(loc(), lhsArg, rhsArg);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[lhsArg] = makeState(10, 1, 1024);
  knownPtrs[rhsArg] = makeState(3, 1, 1024);

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), divOp.getResult(),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  ASSERT_EQ(state.getRank(), 1);
  auto offsetAttr = getIntAttr(state.offsets[0]);
  ASSERT_TRUE(offsetAttr.has_value());
  EXPECT_EQ(offsetAttr.value(), 3); // 10 / 3
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for SelectOp  (lines 355-358)
// and visitOperandSelect body (lines 505-520)
// ======================================================================

TEST_F(VisitOperandTest, SelectOp) {
  auto i32Tensor = RankedTensorType::get({1024}, irRewriter->getI32Type());
  auto i1Tensor = RankedTensorType::get({1024}, irRewriter->getI1Type());
  auto funcOp =
      makeFuncWithArgs("test_select", {i1Tensor, i32Tensor, i32Tensor});
  Block &entry = funcOp.getBody().front();

  Value cond = entry.getArgument(0);
  Value trueVal = entry.getArgument(1);
  Value falseVal = entry.getArgument(2);

  irRewriter->setInsertionPoint(entry.getTerminator());
  auto selectOp =
      irRewriter->create<arith::SelectOp>(loc(), cond, trueVal, falseVal);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[trueVal] = makeState(42, 1, 1024);
  knownPtrs[falseVal] = makeState(99, 1, 1024);

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), selectOp.getResult(),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  ASSERT_EQ(state.getRank(), 1);
  // visitOperandSelect always uses trueState
  auto offsetAttr = getIntAttr(state.offsets[0]);
  ASSERT_TRUE(offsetAttr.has_value());
  EXPECT_EQ(offsetAttr.value(), 42);
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for DotOp  (lines 395-398)
// and visitOperandDot body (lines 696-705)
// ======================================================================

TEST_F(VisitOperandTest, DotOp) {
  auto f32Type = irRewriter->getF32Type();
  auto mat16 = RankedTensorType::get({16, 16}, f32Type);

  auto funcOp = makeFuncWithArgs("test_dot", {mat16, mat16, mat16});
  Block &entry = funcOp.getBody().front();

  Value a = entry.getArgument(0);
  Value b = entry.getArgument(1);
  Value c = entry.getArgument(2);

  irRewriter->setInsertionPoint(entry.getTerminator());
  auto dotOp = irRewriter->create<triton::DotOp>(
      loc(), mat16, a, b, c,
      triton::InputPrecisionAttr::get(&ctx, triton::InputPrecision::IEEE),
      irRewriter->getI32IntegerAttr(0));

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[c] = makeState(7, 1, 16);

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), dotOp.getResult(),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.offsets[0]).value(), 7);
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for ReduceOp  (lines 399-402)
// and visitOperandReduce body (lines 707-727)
// ======================================================================

TEST_F(VisitOperandTest, ReduceOp) {
  auto f32Type = irRewriter->getF32Type();
  auto src2dType = RankedTensorType::get({16, 32}, f32Type);

  auto funcOp = makeFuncWithArgs("test_reduce", {src2dType});
  Block &entry = funcOp.getBody().front();
  Value srcArg = entry.getArgument(0);

  irRewriter->setInsertionPoint(entry.getTerminator());

  // Build ReduceOp via OperationState for full control.
  auto result1dType = RankedTensorType::get({32}, f32Type);
  OperationState reduceState(loc(), triton::ReduceOp::getOperationName());
  reduceState.addOperands(srcArg);
  reduceState.addAttribute("axis", irRewriter->getI32IntegerAttr(0));
  reduceState.addTypes(result1dType);
  reduceState.addRegion();
  Operation *reduceOpRaw = irRewriter->create(reduceState);
  auto reduceOp = cast<triton::ReduceOp>(reduceOpRaw);

  // Populate the combiner region.
  auto &combinerRegion = reduceOp.getCombineOp();
  Block *combiner = new Block();
  combinerRegion.push_back(combiner);
  combiner->addArgument(f32Type, loc());
  combiner->addArgument(f32Type, loc());
  OpBuilder combinerBuilder(combiner, combiner->begin());
  auto addF = combinerBuilder.create<arith::AddFOp>(
      loc(), combiner->getArgument(0), combiner->getArgument(1));
  combinerBuilder.create<triton::ReduceReturnOp>(loc(), ValueRange{addF});

  // Pre-populate knownPtrs for the source (rank-2).
  PtrState srcState;
  srcState.offsets.push_back(irRewriter->getIndexAttr(5));
  srcState.offsets.push_back(irRewriter->getIndexAttr(10));
  srcState.strides.push_back(irRewriter->getIndexAttr(32));
  srcState.strides.push_back(irRewriter->getIndexAttr(1));
  srcState.sizes.push_back(irRewriter->getIndexAttr(16));
  srcState.sizes.push_back(irRewriter->getIndexAttr(32));

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[srcArg] = srcState;

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), reduceOp->getResult(0),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  // Reduce on axis=0: removes dim 0, keeps dim 1
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.offsets[0]).value(), 10);
  EXPECT_EQ(getIntAttr(state.strides[0]).value(), 1);
  EXPECT_EQ(getIntAttr(state.sizes[0]).value(), 32);
}

// ======================================================================
// PtrAnalysis::visitOperand dispatcher for LoadOp  (lines 403-406)
// and visitOperandLoad body (lines 729-738)
// ======================================================================

TEST_F(VisitOperandTest, LoadOp) {
  auto f32Type = irRewriter->getF32Type();
  auto ptrType = triton::PointerType::get(f32Type, 1);
  auto tensorPtrType = RankedTensorType::get({1024}, ptrType);

  auto funcOp = makeFuncWithArgs("test_load", {tensorPtrType});
  Block &entry = funcOp.getBody().front();
  Value ptrArg = entry.getArgument(0);

  irRewriter->setInsertionPoint(entry.getTerminator());
  auto loadOp = irRewriter->create<triton::LoadOp>(
      loc(), ptrArg, /*mask=*/Value(), /*other=*/Value(),
      triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
      /*isVolatile=*/false);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  knownPtrs[ptrArg] = makeState(0, 1, 1024);

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), loadOp.getResult(),
                                      state, knownPtrs);
  EXPECT_TRUE(ok);
  ASSERT_EQ(state.getRank(), 1);
  EXPECT_EQ(getIntAttr(state.offsets[0]).value(), 0);
  EXPECT_EQ(getIntAttr(state.strides[0]).value(), 1);
  EXPECT_EQ(getIntAttr(state.sizes[0]).value(), 1024);
}

// ======================================================================
// PtrAnalysis::visitOperand else fallback  (lines 412-414)
// Returns false for unrecognized ops.
// ======================================================================

TEST_F(VisitOperandTest, UnrecognizedOp) {
  auto i32Tensor = RankedTensorType::get({1024}, irRewriter->getI32Type());

  auto funcOp = makeFuncWithArgs("test_unrec", {i32Tensor, i32Tensor});
  Block &entry = funcOp.getBody().front();

  Value lhsArg = entry.getArgument(0);
  Value rhsArg = entry.getArgument(1);

  irRewriter->setInsertionPoint(entry.getTerminator());
  // arith.maxsi is not handled by PtrAnalysis dispatch
  auto maxOp = irRewriter->create<arith::MaxSIOp>(loc(), lhsArg, rhsArg);

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;

  PtrState state;
  bool ok = PtrAnalysis::visitOperand(rewriter(), loc(), maxOp.getResult(),
                                      state, knownPtrs);
  EXPECT_FALSE(ok);
}

// ======================================================================
// rewriteForBlocks (lines 1628-1715) via preProcessEntry
// preProcessEntry walks the module for triton::FuncOp containing
// load/store and calls modifyFuncEntry -> rewriteForBlocks if the
// function has multiple blocks.
// ======================================================================

TEST(PreProcessEntryTest, MultiBlockFunction) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                  scf::SCFDialect, cf::ControlFlowDialect,
                  triton::TritonDialect, triton::gpu::TritonGPUDialect,
                  triton::gcu::TritonGCUDialect>();

  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();

  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  auto f32Type = builder.getF32Type();
  auto ptrType = triton::PointerType::get(f32Type, 1);
  auto tensorPtrType = RankedTensorType::get({1024}, ptrType);
  auto i32Type = builder.getI32Type();

  auto funcType = FunctionType::get(&ctx, {tensorPtrType, i32Type}, {});
  auto ttFunc = builder.create<triton::FuncOp>(loc, "multi_block", funcType);
  ttFunc.setPublic();

  // Block 0: entry — branch to block 1
  Block *entryBlock = ttFunc.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Block 1: contains a tt.load and tt.return
  Block *block1 = builder.createBlock(&ttFunc.getBody());
  block1->addArgument(tensorPtrType, loc);
  block1->addArgument(i32Type, loc);

  builder.setInsertionPointToStart(block1);
  builder.create<triton::LoadOp>(loc, block1->getArgument(0), /*mask=*/Value(),
                                 /*other=*/Value(), triton::CacheModifier::NONE,
                                 triton::EvictionPolicy::NORMAL,
                                 /*isVolatile=*/false);
  builder.create<triton::ReturnOp>(loc);

  // Back to entry: branch to block1
  builder.setInsertionPointToEnd(entryBlock);
  SmallVector<Value> branchArgs;
  for (BlockArgument arg : entryBlock->getArguments())
    branchArgs.push_back(arg);
  builder.create<cf::BranchOp>(loc, block1, branchArgs);

  // Call preProcessEntry — triggers rewriteForBlocks since > 1 block
  Value condition;
  bool result = PtrAnalysis::preProcessEntry(module, condition);
  EXPECT_TRUE(result);

  // rewriteForBlocks clones blocks and creates a new entry with cf.cond_br.
  // Verify: the function now has more blocks than the original 2.
  unsigned blockCount = 0;
  for ([[maybe_unused]] auto &block : ttFunc.getBody())
    ++blockCount;
  EXPECT_GT(blockCount, 2u);

  // The new entry block should contain a cf.cond_br
  Block &newEntry = ttFunc.getBody().front();
  auto terminator = newEntry.getTerminator();
  EXPECT_TRUE(isa<cf::CondBranchOp>(terminator));
}

#else // TEST_GCU300

// PtrAnalysis gcu300 has a different API; these tests target gcu400 only.
TEST(PtrAnalysisTest, Gcu300Placeholder) {
  SUCCEED() << "PtrAnalysis gtest targets gcu400 only";
}

#endif // TEST_GCU400
