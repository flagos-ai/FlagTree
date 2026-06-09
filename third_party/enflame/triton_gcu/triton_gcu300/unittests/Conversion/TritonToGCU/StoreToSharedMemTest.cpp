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

#ifdef TEST_GCU400
#include "Conversion/TritonToGCU/Utility.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#endif

using namespace mlir;

namespace {

namespace ttg = triton::gpu;

struct StoreToSharedMemFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  StoreToSharedMemFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
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

// Covers lines 1081-1088: onlyThread0=true with rank>5 (isNeedMerge=true)
TEST(StoreToSharedMemTest, OnlyThread0HighRankMerge) {
  StoreToSharedMemFixture f;

  auto f32Type = f.builder.getF32Type();
  SmallVector<int64_t> shape = {2, 2, 2, 2, 2, 2};
  SmallVector<unsigned> warpsPerCTA = {1, 1, 1, 1, 1, 1};
  auto tensorType = f.makeBlockedTensorType(shape, f32Type, warpsPerCTA);

  auto privateMemType = MemRefType::get(shape, f32Type);
  auto sharedMemType = MemRefType::get(shape, f32Type, AffineMap{},
                                       f.builder.getI64IntegerAttr(2));

  auto funcType =
      f.builder.getFunctionType({privateMemType, sharedMemType}, {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_store_thread0_merge", funcType);
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
  Value sharedBuffer = entryBlock->getArgument(1);

  storeToSharedMem(f.builder, tag, tensorType, sharedBuffer, buffer,
                   /*onlyThread0=*/true);

  bool hasDesliceStart = false;
  bool hasReinterpretCast = false;
  bool hasDmaWait = false;
  bool hasScfIf = false;
  funcOp.walk([&](memref_ext::DesliceStartOp) { hasDesliceStart = true; });
  funcOp.walk([&](memref::ReinterpretCastOp) { hasReinterpretCast = true; });
  funcOp.walk([&](memref::DmaWaitOp) { hasDmaWait = true; });
  funcOp.walk([&](scf::IfOp) { hasScfIf = true; });

  EXPECT_TRUE(hasDesliceStart);
  EXPECT_TRUE(hasReinterpretCast);
  EXPECT_TRUE(hasDmaWait);
  EXPECT_TRUE(hasScfIf);
}
