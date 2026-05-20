// Copyright 2025 Enflame. All Rights Reserved.
#include <memory>

#include "gtest/gtest.h"

#ifdef TEST_GCU400

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Transforms/WarpSpecialization/CodePartitionUtility.h"
#include "Transforms/WarpSpecialization/Utility.h"

namespace ttgcu = mlir::triton::gcu;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gcu {
bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp);
}

class CodePartitionUtilityTest : public ::testing::Test {
protected:
  void SetUp() override {
    ctx.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
                    tt::TritonDialect, ttg::TritonGPUDialect>();
    builder = std::make_unique<mlir::OpBuilder>(&ctx);
    module = mlir::ModuleOp::create(loc());
    builder->setInsertionPointToStart(module.getBody());
    gpuModule = builder->create<mlir::gpu::GPUModuleOp>(loc(), "triton");
    builder->setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  void TearDown() override { module->erase(); }

  mlir::Location loc() { return builder->getUnknownLoc(); }

  mlir::MLIRContext ctx;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::ModuleOp module;
  mlir::gpu::GPUModuleOp gpuModule;
};

TEST_F(CodePartitionUtilityTest, GetTensorType_PointerType) {
  auto f32Ty = builder->getF32Type();
  auto tensorTy = mlir::RankedTensorType::get({64, 32}, f32Ty);
  auto ptrTy = tt::PointerType::get(tensorTy, 1);

  auto result = ttgcu::getTensorType(ptrTy);
  ASSERT_TRUE(result != nullptr);
  EXPECT_EQ(result, tensorTy);
}

TEST_F(CodePartitionUtilityTest, GetTensorType_DirectTensor) {
  auto f32Ty = builder->getF32Type();
  auto tensorTy = mlir::RankedTensorType::get({64, 32}, f32Ty);

  auto result = ttgcu::getTensorType(tensorTy);
  ASSERT_TRUE(result != nullptr);
  EXPECT_EQ(result, tensorTy);
}

TEST_F(CodePartitionUtilityTest, Enclosing_IfOp) {
  auto funcType = builder->getFunctionType({builder->getI1Type()}, {});
  auto funcOp =
      builder->create<mlir::func::FuncOp>(loc(), "test_enclosing_if", funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto cond = entry->getArgument(0);
  auto ifOp = builder->create<mlir::scf::IfOp>(loc(), cond, false);
  builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto *innerOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 42, 32).getOperation();

  EXPECT_TRUE(ttgcu::enclosing(ifOp, innerOp));

  builder->setInsertionPointAfter(ifOp);
  auto *outerOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 1, 32).getOperation();
  EXPECT_FALSE(ttgcu::enclosing(ifOp, outerOp));
}

TEST_F(CodePartitionUtilityTest, Enclosing_ForOp) {
  auto funcType = builder->getFunctionType({}, {});
  auto funcOp = builder->create<mlir::func::FuncOp>(loc(), "test_enclosing_for",
                                                    funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto lb = builder->create<mlir::arith::ConstantIndexOp>(loc(), 0);
  auto ub = builder->create<mlir::arith::ConstantIndexOp>(loc(), 10);
  auto step = builder->create<mlir::arith::ConstantIndexOp>(loc(), 1);
  auto forOp = builder->create<mlir::scf::ForOp>(loc(), lb, ub, step);
  builder->setInsertionPointToStart(forOp.getBody());
  auto *innerOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 42, 32).getOperation();

  EXPECT_TRUE(ttgcu::enclosing(forOp, innerOp));

  builder->setInsertionPointAfter(forOp);
  auto *outerOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 1, 32).getOperation();
  EXPECT_FALSE(ttgcu::enclosing(forOp, outerOp));
}

TEST_F(CodePartitionUtilityTest, ImmediateEnclosing) {
  auto funcType = builder->getFunctionType({builder->getI1Type()}, {});
  auto funcOp = builder->create<mlir::func::FuncOp>(
      loc(), "test_immediate_enclosing", funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto cond = entry->getArgument(0);
  auto ifOp = builder->create<mlir::scf::IfOp>(loc(), cond, false);
  builder->setInsertionPointToStart(&ifOp.getThenRegion().front());

  auto *directOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 1, 32).getOperation();
  EXPECT_TRUE(ttgcu::immediateEnclosing(ifOp, directOp));

  auto lb = builder->create<mlir::arith::ConstantIndexOp>(loc(), 0);
  auto ub = builder->create<mlir::arith::ConstantIndexOp>(loc(), 10);
  auto step = builder->create<mlir::arith::ConstantIndexOp>(loc(), 1);
  auto innerFor = builder->create<mlir::scf::ForOp>(loc(), lb, ub, step);
  builder->setInsertionPointToStart(innerFor.getBody());
  auto *nestedOp =
      builder->create<mlir::arith::ConstantIntOp>(loc(), 2, 32).getOperation();
  EXPECT_FALSE(ttgcu::immediateEnclosing(ifOp, nestedOp));
}

TEST_F(CodePartitionUtilityTest, GetAccumCnts_IfOp_Branch) {
  auto funcType = builder->getFunctionType({builder->getI1Type()}, {});
  auto funcOp = builder->create<mlir::func::FuncOp>(
      loc(), "test_getaccumcnts_ifop", funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto cond = entry->getArgument(0);
  auto ifOp = builder->create<mlir::scf::IfOp>(loc(), cond, false);
  builder->setInsertionPointToStart(&ifOp.getThenRegion().front());

  auto lb = builder->create<mlir::arith::ConstantIndexOp>(loc(), 0);
  auto ub = builder->create<mlir::arith::ConstantIndexOp>(loc(), 4);
  auto step = builder->create<mlir::arith::ConstantIndexOp>(loc(), 1);
  auto nestedFor = builder->create<mlir::scf::ForOp>(loc(), lb, ub, step);

  mlir::DenseSet<mlir::Operation *> regionsWithChannels;
  regionsWithChannels.insert(nestedFor.getOperation());

  unsigned cnt = ttgcu::getAccumCnts(ifOp.getOperation(), regionsWithChannels);
  EXPECT_EQ(cnt, 1u);
}

TEST_F(CodePartitionUtilityTest, GetAccumCnts_ForOp_Branch) {
  auto funcType = builder->getFunctionType({}, {});
  auto funcOp = builder->create<mlir::func::FuncOp>(
      loc(), "test_getaccumcnts_forop", funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto lb = builder->create<mlir::arith::ConstantIndexOp>(loc(), 0);
  auto ub = builder->create<mlir::arith::ConstantIndexOp>(loc(), 10);
  auto step = builder->create<mlir::arith::ConstantIndexOp>(loc(), 1);
  auto outerFor = builder->create<mlir::scf::ForOp>(loc(), lb, ub, step);
  builder->setInsertionPointToStart(outerFor.getBody());

  auto lb2 = builder->create<mlir::arith::ConstantIndexOp>(loc(), 0);
  auto ub2 = builder->create<mlir::arith::ConstantIndexOp>(loc(), 4);
  auto step2 = builder->create<mlir::arith::ConstantIndexOp>(loc(), 1);
  auto innerFor = builder->create<mlir::scf::ForOp>(loc(), lb2, ub2, step2);

  mlir::DenseSet<mlir::Operation *> regionsWithChannels;
  regionsWithChannels.insert(innerFor.getOperation());

  unsigned cnt =
      ttgcu::getAccumCnts(outerFor.getOperation(), regionsWithChannels);
  EXPECT_EQ(cnt, 1u);
}

TEST_F(CodePartitionUtilityTest, GetBarrierForPipelineStage) {
  auto funcType = builder->getFunctionType({}, {});
  auto funcOp =
      builder->create<mlir::func::FuncOp>(loc(), "test_barrier", funcType);
  auto *entry = funcOp.addEntryBlock();
  builder->setInsertionPointToStart(entry);

  auto sharedMemSpace = ttg::SharedMemorySpaceAttr::get(&ctx);
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&ctx, 1);
  SmallVector<unsigned> order{0};
  auto sharedEnc =
      ttg::SwizzledSharedEncodingAttr::get(&ctx, 1, 1, 1, order, ctaLayout);
  auto barrierAllocTy = ttg::MemDescType::get({4}, builder->getI64Type(),
                                              sharedEnc, sharedMemSpace, true);

  auto barrierAlloc = entry->addArgument(barrierAllocTy, loc());

  auto bufIdx = builder->create<mlir::arith::ConstantIntOp>(loc(), 0, 32);

  ttgcu::OpBuilderWithAsyncTaskIds asyncBuilder(&ctx);
  asyncBuilder.setAsynTaskIdsFromArray({0, 1});
  asyncBuilder.setInsertionPointToEnd(entry);

  auto result =
      ttgcu::getBarrierForPipelineStage(asyncBuilder, barrierAlloc, bufIdx);
  EXPECT_TRUE(result != nullptr);
}

#endif // TEST_GCU400
