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

#include <cstdlib>
#include <utility>

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#endif

using namespace mlir;

namespace {

namespace ttg = triton::gpu;

struct ConfigGcuLoadFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  ConfigGcuLoadFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
    ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect, gpu::GPUDialect, scf::SCFDialect,
                    math::MathDialect, memref_ext::MemrefExtDialect,
                    gcu::GCUDialect, triton::gcu::TritonGCUDialect,
                    ttg::TritonGPUDialect>();

    module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }
};

} // namespace

// Covers lines 2038-2091 in gcu300 TritonGCUToGCUUtils.cpp
// (IsShareOutput && bDynamicStride branch of ConfigGcuLoad).
// Covers lines 1600-1631 in gcu400 Utility.cpp (equivalent branch).
TEST(ConfigGcuLoadTest, DynamicStrideShareOutput) {
  ConfigGcuLoadFixture f;

  auto f16Type = f.builder.getF16Type();

  auto resultType = MemRefType::get({32, 64}, f16Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));
  auto gcuPtrType = gcu::PtrType::get(&f.ctx, f16Type);
  auto ttgcuPtrType = triton::gcu::PtrType::get(&f.ctx, f16Type);
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&f.ctx, 2);
  SmallVector<unsigned> order = {1, 0};
  auto sharedEncoding =
      ttg::SwizzledSharedEncodingAttr::get(&f.ctx, 1, 1, 1, order, ctaLayout);
  auto smem = ttg::SharedMemorySpaceAttr::get(&f.ctx);
  auto memdescType =
      ttg::MemDescType::get({32, 64}, f16Type, sharedEncoding, smem, true);

#ifdef TEST_GCU400
  // gcu400: ConfigGcuLoad(rewriter, loc, srcOut, op, ...)
  // Op type: CopyGlobalToLocalOp
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, ttgcuPtrType, memdescType, f16Type}, {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_config_load_dynamic", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/4, false);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value copyPtr = entryBlock->getArgument(2);
  Value dstMem = entryBlock->getArgument(3);
  Value defaultValue = entryBlock->getArgument(4);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};
  SmallVector<Value> offsets = {c0, c0};

  // order_hint=[-1, 0] triggers bDynamicStride=true
  f.builder.create<triton::gcu::CopyGlobalToLocalOp>(
      f.loc, copyPtr, shape, strides, offsets, dstMem, defaultValue,
      ArrayRef<int32_t>{-1, 0});
  auto *copyOp = &(*std::prev(f.builder.getInsertionPoint()));

  auto tag = tagPool.getPrivateSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, srcOut, copyOp, resultType, loadPtr, strides,
                shape, defaultValue, tag, /*IsShareOutput=*/true);
#else
  // gcu300: ConfigGcuLoad(rewriter, loc, pTagPool, srcOut, transOut, op, ...)
  // Op type: AsyncLoadFromGlobalOp
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, memdescType, f16Type},
      {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_config_load_dynamic", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value transOut = entryBlock->getArgument(2);
  Value asyncPtr = entryBlock->getArgument(3);
  Value dstMem = entryBlock->getArgument(4);
  Value defaultValue = entryBlock->getArgument(5);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};
  SmallVector<Value> offsets = {c0, c0};

  // order_hint=[-1, 0] triggers bDynamicStride=true;
  // after bReshape expands rank to 3, the rank-2 optimization doesn't apply.
  auto asyncLoadOp = f.builder.create<triton::gcu::AsyncLoadFromGlobalOp>(
      f.loc, asyncPtr, shape, strides, offsets, dstMem, defaultValue,
      ArrayRef<int32_t>{-1, 0});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, tagPool, srcOut, transOut,
                asyncLoadOp.getOperation(), resultType, loadPtr, strides, shape,
                defaultValue, tag, /*IsShareOutput=*/true);
#endif

  SUCCEED();
}

// Covers lines 1314-1316 in gcu400 Utility.cpp (LoadOp branch of
// getDefaultValue lambda), and lines 1721-1749 (!IsShareOutput &&
// !bDynamicStride && !bStaticTranspose path). Lines 1317-1319
// (CopyGlobalToLocalOp branch) are dead code: getDefaultValue is only reachable
// in the !IsShareOutput path, which casts op to LoadOp at line 1409.
TEST(ConfigGcuLoadTest, NonShareIdentityOrder) {
  ConfigGcuLoadFixture f;

  auto f16Type = f.builder.getF16Type();

  auto resultType = MemRefType::get({32, 64}, f16Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));
  auto gcuPtrType = gcu::PtrType::get(&f.ctx, f16Type);
  auto ttgcuPtrType = triton::gcu::PtrType::get(&f.ctx, f16Type);

  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&f.ctx, 2);
  SmallVector<unsigned> sizePerThread = {1, 1};
  SmallVector<unsigned> threadsPerWarp = {1, 1};
  SmallVector<unsigned> warpsPerCTA = {1, 4};
  SmallVector<unsigned> order = {1, 0};
  auto blockedEnc = ttg::BlockedEncodingAttr::get(
      &f.ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order, ctaLayout);
  auto tensorType = RankedTensorType::get({32, 64}, f16Type, blockedEnc);

#ifdef TEST_GCU400
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, ttgcuPtrType, f16Type}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_nonshare_identity", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/4, false);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value loadOpPtr = entryBlock->getArgument(2);
  Value defaultValue = entryBlock->getArgument(3);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};

  // order_hint={0, 1}: identity, bDynamicStride=false, bStaticTranspose=false
  auto loadOp = f.builder.create<triton::gcu::LoadOp>(
      f.loc, tensorType, loadOpPtr, shape, strides, SmallVector<Value>{c0, c0},
      defaultValue, ArrayRef<int32_t>{0, 1});

  auto tag = tagPool.getPrivateSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, srcOut, loadOp.getOperation(), resultType,
                loadPtr, strides, shape, defaultValue, tag,
                /*IsShareOutput=*/false);
#else
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, f16Type}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_nonshare_identity", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value transOut = entryBlock->getArgument(2);
  Value loadOpPtr = entryBlock->getArgument(3);
  Value defaultValue = entryBlock->getArgument(4);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};

  auto loadOp = f.builder.create<triton::gcu::LoadOp>(
      f.loc, tensorType, loadOpPtr, shape, strides, SmallVector<Value>{c0, c0},
      defaultValue, ArrayRef<int32_t>{0, 1});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, tagPool, srcOut, transOut,
                loadOp.getOperation(), resultType, loadPtr, strides, shape,
                defaultValue, tag, /*IsShareOutput=*/false);
#endif

  SUCCEED();
}

// Covers lines 1355-1361 in gcu400 Utility.cpp: the TRITON_GCU_DEBUG assertion
// inside the bDynamicStride loop. Requires order_hint with -1 AND the env var.
TEST(ConfigGcuLoadTest, DynamicStrideDebugAssert) {
  ::setenv("TRITON_GCU_DEBUG", "1", 1);

  ConfigGcuLoadFixture f;

  auto f16Type = f.builder.getF16Type();

  auto resultType = MemRefType::get({32, 64}, f16Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));
  auto gcuPtrType = gcu::PtrType::get(&f.ctx, f16Type);
  auto ttgcuPtrType = triton::gcu::PtrType::get(&f.ctx, f16Type);
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&f.ctx, 2);
  SmallVector<unsigned> order = {1, 0};
  auto sharedEncoding =
      ttg::SwizzledSharedEncodingAttr::get(&f.ctx, 1, 1, 1, order, ctaLayout);
  auto smem = ttg::SharedMemorySpaceAttr::get(&f.ctx);
  auto memdescType =
      ttg::MemDescType::get({32, 64}, f16Type, sharedEncoding, smem, true);

#ifdef TEST_GCU400
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, ttgcuPtrType, memdescType, f16Type}, {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_dynamic_debug_assert", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateTagPool tagPool(funcOp, /*numWarps=*/4, false);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value copyPtr = entryBlock->getArgument(2);
  Value dstMem = entryBlock->getArgument(3);
  Value defaultValue = entryBlock->getArgument(4);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};

  f.builder.create<triton::gcu::CopyGlobalToLocalOp>(
      f.loc, copyPtr, shape, strides, SmallVector<Value>{c0, c0}, dstMem,
      defaultValue, ArrayRef<int32_t>{-1, 0});
  auto *copyOp = &(*std::prev(f.builder.getInsertionPoint()));

  auto tag = tagPool.getPrivateSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, srcOut, copyOp, resultType, loadPtr, strides,
                shape, defaultValue, tag, /*IsShareOutput=*/true);
#else
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, memdescType, f16Type},
      {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_dynamic_debug_assert", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value transOut = entryBlock->getArgument(2);
  Value asyncPtr = entryBlock->getArgument(3);
  Value dstMem = entryBlock->getArgument(4);
  Value defaultValue = entryBlock->getArgument(5);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};

  auto asyncLoadOp = f.builder.create<triton::gcu::AsyncLoadFromGlobalOp>(
      f.loc, asyncPtr, shape, strides, SmallVector<Value>{c0, c0}, dstMem,
      defaultValue, ArrayRef<int32_t>{-1, 0});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoad(f.builder, f.loc, tagPool, srcOut, transOut,
                asyncLoadOp.getOperation(), resultType, loadPtr, strides, shape,
                defaultValue, tag, /*IsShareOutput=*/true);
#endif

  ::unsetenv("TRITON_GCU_DEBUG");
  SUCCEED();
}
