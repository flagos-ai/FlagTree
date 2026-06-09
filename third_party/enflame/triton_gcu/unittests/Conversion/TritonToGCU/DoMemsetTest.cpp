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

#include "gtest/gtest.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"

#ifdef TEST_GCU400
#include "Conversion/TritonToGCU/Utility.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#endif

using namespace mlir;

// Verify doMemset creates a memref.reinterpret_cast when the output memref
// rank > 5 and totalNumElems > 128 (the defensive merge path).
TEST(DoMemsetTest, ReinterpretCastForHighRankMemRef) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                  affine::AffineDialect, memref_ext::MemrefExtDialect,
                  gcu::GCUDialect, gpu::GPUDialect>();

  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();

  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  // PrivateTagPool destructor requires a gpu::GPUModuleOp ancestor.
  auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  // Rank 6 (> 5) triggers isNeedMerge = true inside doMemset.
  auto memrefType = MemRefType::get({2, 2, 2, 2, 2, 2}, builder.getF32Type());
  auto funcType =
      builder.getFunctionType({memrefType, builder.getF32Type()}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "test_func", funcType);

  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto returnOp = builder.create<func::ReturnOp>(loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
#endif

  builder.setInsertionPoint(returnOp);

  Value output = entryBlock->getArgument(0);
  Value v = entryBlock->getArgument(1);

  // totalNumElems = 256 enters the (> 128) branch.
  // rank 6 (> 5) sets isNeedMerge = true, triggering ReinterpretCastOp.
#ifdef TEST_GCU400
  doMemset(builder, tag, returnOp, output, v, 256);
#else
  doMemset(builder, tagPool, returnOp, output, v, 256);
#endif

  bool hasReinterpretCast = false;
  funcOp.walk([&](memref::ReinterpretCastOp op) {
    hasReinterpretCast = true;
    auto resultType = op.getType();
    EXPECT_EQ(resultType.getRank(), 1);
    EXPECT_EQ(resultType.getShape()[0], 256);
  });

  EXPECT_TRUE(hasReinterpretCast);
}

// Verify doMemset does NOT create a reinterpret_cast when rank <= 5
// even if totalNumElems > 128 (isNeedMerge = false).
TEST(DoMemsetTest, NoReinterpretCastForLowRankMemRef) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                  affine::AffineDialect, memref_ext::MemrefExtDialect,
                  gcu::GCUDialect, gpu::GPUDialect>();

  OpBuilder builder(&ctx);
  auto loc = builder.getUnknownLoc();

  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
  builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());

  // Rank 5 (<= 5) keeps isNeedMerge = false.
  auto memrefType = MemRefType::get({2, 2, 2, 2, 2}, builder.getF32Type());
  auto funcType =
      builder.getFunctionType({memrefType, builder.getF32Type()}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "test_func2", funcType);

  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  auto returnOp = builder.create<func::ReturnOp>(loc);

#ifdef TEST_GCU400
  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/1, false);
  auto tag = tagPool.tryGetPrivateAsyncTagInfo(returnOp);
#else
  triton::gcu::PrivateDTETagPool tagPool(funcOp);
#endif

  builder.setInsertionPoint(returnOp);

  Value output = entryBlock->getArgument(0);
  Value v = entryBlock->getArgument(1);

#ifdef TEST_GCU400
  doMemset(builder, tag, returnOp, output, v, 256);
#else
  doMemset(builder, tagPool, returnOp, output, v, 256);
#endif

  bool hasReinterpretCast = false;
  funcOp.walk([&](memref::ReinterpretCastOp) { hasReinterpretCast = true; });

  EXPECT_FALSE(hasReinterpretCast);
}
