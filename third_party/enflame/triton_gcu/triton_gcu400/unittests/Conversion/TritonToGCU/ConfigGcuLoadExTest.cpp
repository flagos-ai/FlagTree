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
// ConfigGcuLoadEx only exists in gcu300.
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#endif

using namespace mlir;

#ifdef TEST_GCU400

TEST(ConfigGcuLoadExTest, NotApplicableOnGCU400) { SUCCEED(); }

#else

namespace {

namespace ttg = triton::gpu;

struct ConfigGcuLoadExFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  ConfigGcuLoadExFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
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

// Covers lines 2292-2295 and 2305-2308 in TritonGCUToGCUUtils.cpp:
// the AsyncLoadFromGlobalOp branches of getOrderHint and getDefaultValue
// lambdas inside ConfigGcuLoadEx.
TEST(ConfigGcuLoadExTest, AsyncLoadFromGlobalOpBranches) {
  ConfigGcuLoadExFixture f;

  auto f16Type = f.builder.getF16Type();
  [[maybe_unused]] auto indexType = f.builder.getIndexType();

  // Result memref type: memref<32x64xf16, 2> (shared memory)
  auto resultType = MemRefType::get({32, 64}, f16Type, AffineMap{},
                                    f.builder.getI64IntegerAttr(2));

  // gcu::PtrType for loadPtr (used by gcu.ptr2memref)
  auto gcuPtrType = gcu::PtrType::get(&f.ctx, f16Type);

  // triton_gcu::PtrType for AsyncLoadFromGlobalOp's ptr operand
  auto ttgcuPtrType = triton::gcu::PtrType::get(&f.ctx, f16Type);

  // MemDescType for AsyncLoadFromGlobalOp's dstMem operand
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&f.ctx, 2);
  SmallVector<unsigned> order = {1, 0};
  auto sharedEncoding =
      ttg::SwizzledSharedEncodingAttr::get(&f.ctx, 1, 1, 1, order, ctaLayout);
  auto smem = ttg::SharedMemorySpaceAttr::get(&f.ctx);
  auto memdescType =
      ttg::MemDescType::get({32, 64}, f16Type, sharedEncoding, smem, true);

  // Function type: (gcu.ptr<f16>, memref<32x64xf16,2>, memref<32x64xf16,2>,
  //                  triton_gcu.ptr<f16>, memdesc<32x64xf16>, f16) -> ()
  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, memdescType, f16Type},
      {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_configgculoadex", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value tempBuffer = entryBlock->getArgument(2);
  Value asyncPtr = entryBlock->getArgument(3);
  Value dstMem = entryBlock->getArgument(4);
  Value defaultValue = entryBlock->getArgument(5);

  // Create index constants for shape, strides, offsets
  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};
  SmallVector<Value> offsets = {c0, c0};

  // Create AsyncLoadFromGlobalOp with order_hint=[-1, 0]
  // This triggers bDynamicStride=true inside ConfigGcuLoadEx.
  auto asyncLoadOp = f.builder.create<triton::gcu::AsyncLoadFromGlobalOp>(
      f.loc, asyncPtr, shape, strides, offsets, dstMem, defaultValue,
      ArrayRef<int32_t>{-1, 0});

  // Get a TagInfo from the pool
  auto tag = tagPool.getSyncTagInfo(returnOp);

  // Call ConfigGcuLoadEx with the AsyncLoadFromGlobalOp.
  // IsShareOutput=true to take the simpler path (avoids encoding-dependent
  // getElemsPerThread/getWarpIds).
  ConfigGcuLoadEx(f.builder, f.loc, tagPool, srcOut, tempBuffer,
                  asyncLoadOp.getOperation(), resultType, loadPtr, strides,
                  shape, defaultValue, tag, /*IsShareOutput=*/true);

  // If we get here without crashing, the AsyncLoadFromGlobalOp branches
  // of getOrderHint (lines 2292-2295) and getDefaultValue (lines 2305-2308)
  // have been exercised.
  SUCCEED();
}

// Covers lines 2431-2449 in TritonGCUToGCUUtils.cpp:
// the if (bStaticReshape) block inside ConfigGcuLoadEx.
// bStaticReshape=true requires all order_hint values > 0 (no -1, no 0).
TEST(ConfigGcuLoadExTest, StaticReshapeBranch) {
  ConfigGcuLoadExFixture f;

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

  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, memdescType, f16Type},
      {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_static_reshape", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value tempBuffer = entryBlock->getArgument(2);
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

  // order_hint={2, 1}: all values > 0, so bStaticReshape=true,
  // bDynamicStride=false.
  auto asyncLoadOp = f.builder.create<triton::gcu::AsyncLoadFromGlobalOp>(
      f.loc, asyncPtr, shape, strides, offsets, dstMem, defaultValue,
      ArrayRef<int32_t>{2, 1});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoadEx(f.builder, f.loc, tagPool, srcOut, tempBuffer,
                  asyncLoadOp.getOperation(), resultType, loadPtr, strides,
                  shape, defaultValue, tag, /*IsShareOutput=*/true);

  SUCCEED();
}

// Covers lines 2702-2719 in TritonGCUToGCUUtils.cpp:
// the else branch (neither bDynamicStride nor bStaticTranspose) in
// IsShareOutput inside ConfigGcuLoadEx.
// order_hint={0, 1} is an identity permutation: bDynamicStride=false,
// bStaticReshape=false (0 present), bStaticTranspose=false.
TEST(ConfigGcuLoadExTest, IdentityOrderShareOutput) {
  ConfigGcuLoadExFixture f;

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

  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, memdescType, f16Type},
      {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "test_identity_order", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value tempBuffer = entryBlock->getArgument(2);
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

  auto asyncLoadOp = f.builder.create<triton::gcu::AsyncLoadFromGlobalOp>(
      f.loc, asyncPtr, shape, strides, offsets, dstMem, defaultValue,
      ArrayRef<int32_t>{0, 1});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoadEx(f.builder, f.loc, tagPool, srcOut, tempBuffer,
                  asyncLoadOp.getOperation(), resultType, loadPtr, strides,
                  shape, defaultValue, tag, /*IsShareOutput=*/true);

  SUCCEED();
}

// Covers lines 2869-2902 in TritonGCUToGCUUtils.cpp:
// !IsShareOutput && bStaticTranspose branch of ConfigGcuLoadEx.
// Uses LoadOp (IsShareOutput=false requires LoadOp, not AsyncLoadFromGlobalOp).
TEST(ConfigGcuLoadExTest, NonShareStaticTranspose) {
  ConfigGcuLoadExFixture f;

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

  auto funcType = f.builder.getFunctionType(
      {gcuPtrType, resultType, resultType, ttgcuPtrType, f16Type}, {});
  auto funcOp = f.builder.create<func::FuncOp>(
      f.loc, "test_nonshare_static_transpose", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);
  auto returnOp = f.builder.create<func::ReturnOp>(f.loc);

  triton::gcu::PrivateDTETagPool tagPool(funcOp);

  f.builder.setInsertionPoint(returnOp);

  Value loadPtr = entryBlock->getArgument(0);
  Value srcOut = entryBlock->getArgument(1);
  Value tempBuffer = entryBlock->getArgument(2);
  Value loadOpPtr = entryBlock->getArgument(3);
  Value defaultValue = entryBlock->getArgument(4);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};
  SmallVector<Value> offsets = {c0, c0};

  // order_hint={2, 1}: bStaticReshape=true, after reshape → {1,0,2}, rank=3
  // bStaticTranspose=true (non-identity permutation)
  auto loadOp = f.builder.create<triton::gcu::LoadOp>(
      f.loc, tensorType, loadOpPtr, shape, strides, offsets, defaultValue,
      ArrayRef<int32_t>{2, 1});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoadEx(f.builder, f.loc, tagPool, srcOut, tempBuffer,
                  loadOp.getOperation(), resultType, loadPtr, strides, shape,
                  defaultValue, tag, /*IsShareOutput=*/false);

  SUCCEED();
}

// Covers lines 2903-2928 in TritonGCUToGCUUtils.cpp:
// !IsShareOutput && else (neither bDynamicStride nor bStaticTranspose).
TEST(ConfigGcuLoadExTest, NonShareIdentityOrder) {
  ConfigGcuLoadExFixture f;

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
  Value tempBuffer = entryBlock->getArgument(2);
  Value loadOpPtr = entryBlock->getArgument(3);
  Value defaultValue = entryBlock->getArgument(4);

  auto c32 = f.builder.create<arith::ConstantIndexOp>(f.loc, 32);
  auto c64 = f.builder.create<arith::ConstantIndexOp>(f.loc, 64);
  auto c1 = f.builder.create<arith::ConstantIndexOp>(f.loc, 1);
  auto c0 = f.builder.create<arith::ConstantIndexOp>(f.loc, 0);

  SmallVector<Value> shape = {c32, c64};
  SmallVector<Value> strides = {c64, c1};
  SmallVector<Value> offsets = {c0, c0};

  // order_hint={0, 1}: identity permutation, bStaticReshape=false (0 present),
  // bStaticTranspose=false
  auto loadOp = f.builder.create<triton::gcu::LoadOp>(
      f.loc, tensorType, loadOpPtr, shape, strides, offsets, defaultValue,
      ArrayRef<int32_t>{0, 1});

  auto tag = tagPool.getSyncTagInfo(returnOp);

  ConfigGcuLoadEx(f.builder, f.loc, tagPool, srcOut, tempBuffer,
                  loadOp.getOperation(), resultType, loadPtr, strides, shape,
                  defaultValue, tag, /*IsShareOutput=*/false);

  SUCCEED();
}

#endif
