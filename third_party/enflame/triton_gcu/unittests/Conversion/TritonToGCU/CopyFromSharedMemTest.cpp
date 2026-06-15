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

#include <map>
#include <utility>

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "Analysis/FirstLastUserAnalysis.h"

#ifdef TEST_GCU400
#include "Conversion/TritonToGCU/Utility.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#endif

using namespace mlir;

namespace {

namespace ttg = triton::gpu;

struct CopyFromSharedMemFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  CopyFromSharedMemFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
    ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect, gpu::GPUDialect, scf::SCFDialect,
                    memref_ext::MemrefExtDialect, gcu::GCUDialect,
                    ttg::TritonGPUDialect>();

    module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  RankedTensorType makeBlockedTensorType(ArrayRef<int64_t> shape, Type elemType,
                                         ArrayRef<unsigned> warpsPerCTA) {
    auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&ctx, shape.size());
    SmallVector<unsigned> sizePerThread(shape.size(), 1);
    SmallVector<unsigned> threadsPerWarp(shape.size(), 1);
    SmallVector<unsigned> order;
    for (int i = shape.size() - 1; i >= 0; --i)
      order.push_back(static_cast<unsigned>(i));
    auto blockedEnc = ttg::BlockedEncodingAttr::get(
        &ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
    return RankedTensorType::get(shape, elemType, blockedEnc);
  }
};

} // namespace

// Branch 1: sync tag, onlyThread0=true → lines 996-1006
TEST(CopyFromSharedMemTest, SyncOnlyThread0) {
  CopyFromSharedMemFixture f;

  auto f32Type = f.builder.getF32Type();
  auto tensorType = f.makeBlockedTensorType({4, 4}, f32Type, {1, 1});
  auto bufferType = MemRefType::get({4, 4}, f32Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));

  auto funcType = f.builder.getFunctionType({bufferType}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_sync_thread0", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
  auto tag = tagPool.trygGetAsyncTagInfo(returnOp);
#endif

  f.builder.setInsertionPoint(returnOp);

  Value buffer = entryBlock->getArgument(0);

  triton::gcu::FirstLastUserAnalysis userAnalysis(f.gpuModule);
  std::map<Operation *, Operation *> replaced2Origin;

  std::pair<Operation *, int> lastTTUser = {nullptr, -1};
  std::pair<Operation *, int> firstTTUser = {nullptr, -1};

  auto result = CopyFromSharedMem(f.builder, tag, tensorType, buffer,
                                  /*onlyThread0=*/true, lastTTUser, firstTTUser,
                                  userAnalysis, replaced2Origin);
  ASSERT_TRUE(result != nullptr);

  bool hasDmaStart = false;
  bool hasDmaWait = false;
  bool hasScfIf = false;
  funcOp.walk([&](memref::DmaStartOp) { hasDmaStart = true; });
  funcOp.walk([&](memref::DmaWaitOp) { hasDmaWait = true; });
  funcOp.walk([&](scf::IfOp) { hasScfIf = true; });

  EXPECT_TRUE(hasDmaStart);
  EXPECT_TRUE(hasDmaWait);
  EXPECT_TRUE(hasScfIf);
}

// Branch 2: sync tag, onlyThread0=false → lines 1007-1014
TEST(CopyFromSharedMemTest, SyncAllThreads) {
  CopyFromSharedMemFixture f;

  auto f32Type = f.builder.getF32Type();
  auto tensorType = f.makeBlockedTensorType({4, 4}, f32Type, {1, 1});
  auto bufferType = MemRefType::get({4, 4}, f32Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));

  auto funcType = f.builder.getFunctionType({bufferType}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_sync_all", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
  auto tag = tagPool.trygGetAsyncTagInfo(returnOp);
#endif

  f.builder.setInsertionPoint(returnOp);

  Value buffer = entryBlock->getArgument(0);

  triton::gcu::FirstLastUserAnalysis userAnalysis(f.gpuModule);
  std::map<Operation *, Operation *> replaced2Origin;

  std::pair<Operation *, int> lastTTUser = {nullptr, -1};
  std::pair<Operation *, int> firstTTUser = {nullptr, -1};

  auto result = CopyFromSharedMem(f.builder, tag, tensorType, buffer,
                                  /*onlyThread0=*/false, lastTTUser,
                                  firstTTUser, userAnalysis, replaced2Origin);
  ASSERT_TRUE(result != nullptr);

  bool hasDmaStart = false;
  bool hasDmaWait = false;
  bool hasScfIf = false;
  funcOp.walk([&](memref::DmaStartOp) { hasDmaStart = true; });
  funcOp.walk([&](memref::DmaWaitOp) { hasDmaWait = true; });
  funcOp.walk([&](scf::IfOp) { hasScfIf = true; });

  EXPECT_TRUE(hasDmaStart);
  EXPECT_TRUE(hasDmaWait);
  EXPECT_FALSE(hasScfIf);
}

// Branch 3: async tag, firstTTUser non-null, onlyThread0=true → lines 1016-1033
TEST(CopyFromSharedMemTest, AsyncOnlyThread0) {
  CopyFromSharedMemFixture f;

  auto f32Type = f.builder.getF32Type();
  auto tensorType = f.makeBlockedTensorType({4, 4}, f32Type, {1, 1});
  auto bufferType = MemRefType::get({4, 4}, f32Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));

  auto funcType = f.builder.getFunctionType({bufferType}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_async_thread0", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
  auto tag = tagPool.trygGetAsyncTagInfo(returnOp);
#endif

  f.builder.setInsertionPoint(returnOp);

  Value buffer = entryBlock->getArgument(0);

  triton::gcu::FirstLastUserAnalysis userAnalysis(f.gpuModule);
  std::map<Operation *, Operation *> replaced2Origin;

  std::pair<Operation *, int> lastTTUser = {nullptr, -1};
  std::pair<Operation *, int> firstTTUser = {returnOp, 0};

  auto result = CopyFromSharedMem(f.builder, tag, tensorType, buffer,
                                  /*onlyThread0=*/true, lastTTUser, firstTTUser,
                                  userAnalysis, replaced2Origin);
  ASSERT_TRUE(result != nullptr);

  int dmaStartCount = 0;
  int dmaWaitCount = 0;
  int scfIfCount = 0;
  funcOp.walk([&](memref::DmaStartOp) { dmaStartCount++; });
  funcOp.walk([&](memref::DmaWaitOp) { dmaWaitCount++; });
  funcOp.walk([&](scf::IfOp) { scfIfCount++; });

  EXPECT_EQ(dmaStartCount, 1);
  EXPECT_EQ(dmaWaitCount, 1);
  EXPECT_EQ(scfIfCount, 2);
}

// Branch 4: async tag, firstTTUser non-null, onlyThread0=false → lines
// 1034-1044
TEST(CopyFromSharedMemTest, AsyncAllThreads) {
  CopyFromSharedMemFixture f;

  auto f32Type = f.builder.getF32Type();
  auto tensorType = f.makeBlockedTensorType({4, 4}, f32Type, {1, 1});
  auto bufferType = MemRefType::get({4, 4}, f32Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));

  auto funcType = f.builder.getFunctionType({bufferType}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_async_all", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
  auto tag = tagPool.trygGetAsyncTagInfo(returnOp);
#endif

  f.builder.setInsertionPoint(returnOp);

  Value buffer = entryBlock->getArgument(0);

  triton::gcu::FirstLastUserAnalysis userAnalysis(f.gpuModule);
  std::map<Operation *, Operation *> replaced2Origin;

  std::pair<Operation *, int> lastTTUser = {nullptr, -1};
  std::pair<Operation *, int> firstTTUser = {returnOp, 0};

  auto result = CopyFromSharedMem(f.builder, tag, tensorType, buffer,
                                  /*onlyThread0=*/false, lastTTUser,
                                  firstTTUser, userAnalysis, replaced2Origin);
  ASSERT_TRUE(result != nullptr);

  int dmaStartCount = 0;
  int dmaWaitCount = 0;
  int scfIfCount = 0;
  funcOp.walk([&](memref::DmaStartOp) { dmaStartCount++; });
  funcOp.walk([&](memref::DmaWaitOp) { dmaWaitCount++; });
  funcOp.walk([&](scf::IfOp) { scfIfCount++; });

  EXPECT_EQ(dmaStartCount, 1);
  EXPECT_EQ(dmaWaitCount, 1);
  EXPECT_EQ(scfIfCount, 0);
}
