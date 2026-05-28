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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#ifdef TEST_GCU400
#include "Conversion/Passes.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#endif

using namespace mlir;

namespace ttg = triton::gpu;

#ifdef TEST_GCU400

namespace {

struct TmaCopyFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  OwningOpRef<ModuleOp> module;
  gpu::GPUModuleOp gpuModule;

  TmaCopyFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
    ctx.loadDialect<triton::TritonDialect, ttg::TritonGPUDialect,
                    gpu::GPUDialect, func::FuncDialect, arith::ArithDialect,
                    triton::gcu::TritonGCUDialect>();

    module = ModuleOp::create(loc);
    (*module)->setAttr("ttg.num-ctas", builder.getI32IntegerAttr(1));
    (*module)->setAttr("ttg.num-warps", builder.getI32IntegerAttr(4));
    (*module)->setAttr("ttg.target", builder.getStringAttr("gcu:gcu400"));
    (*module)->setAttr("ttg.threads-per-warp", builder.getI32IntegerAttr(1));

    builder.setInsertionPointToStart((*module).getBody());
    gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  // -- Type helpers ----------------------------------------------------------

  ttg::MemDescType makeMemDescType(Type elemTy,
                                   ArrayRef<int64_t> shape = {32, 64}) {
    auto ctaLayout = ttg::CTAEncodingAttr::getDefault(&ctx, shape.size());
    SmallVector<unsigned> order = (shape.size() == 2)
                                      ? SmallVector<unsigned>{1, 0}
                                      : SmallVector<unsigned>{0};
    auto sharedEnc =
        ttg::SwizzledSharedEncodingAttr::get(&ctx, 1, 1, 1, order, ctaLayout);
    auto smem = ttg::SharedMemorySpaceAttr::get(&ctx);
    return ttg::MemDescType::get(shape, elemTy, sharedEnc, smem,
                                 /*mutableMemory=*/true);
  }

  triton::TensorDescType
  makeTensorDescType(Type elemTy, ArrayRef<int64_t> shape = {32, 64}) {
    return triton::TensorDescType::get(&ctx,
                                       RankedTensorType::get(shape, elemTy));
  }

  // -- IR building helpers ---------------------------------------------------

  struct UnpackFuncResult {
    func::FuncOp funcOp;
    Value base, shape0, shape1, stride0, stride1, padding, offset0, offset1;
  };

  /// Build a function with the standard "unpacked tensor descriptor" signature:
  ///   (ptr<elemTy,1>, i64, i64, i64, i64, i1, i32, i32) -> ()
  /// Returns the funcOp and all block arguments by name.
  UnpackFuncResult createUnpackFunc(StringRef funcName, Type elemTy) {
    auto ptrTy = triton::PointerType::get(elemTy, 1);
    auto i64Ty = builder.getI64Type();
    auto i32Ty = builder.getI32Type();
    auto i1Ty = builder.getI1Type();
    auto funcType = builder.getFunctionType(
        {ptrTy, i64Ty, i64Ty, i64Ty, i64Ty, i1Ty, i32Ty, i32Ty}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, funcName, funcType);
    auto *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    return {funcOp,
            entryBlock->getArgument(0),
            entryBlock->getArgument(1),
            entryBlock->getArgument(2),
            entryBlock->getArgument(3),
            entryBlock->getArgument(4),
            entryBlock->getArgument(5),
            entryBlock->getArgument(6),
            entryBlock->getArgument(7)};
  }

  /// Build an unrealized_conversion_cast that mimics the pattern left by
  /// add_rewrite_tensor_descriptor_to_pointer.
  Value createDescriptorCast(triton::TensorDescType descTy,
                             const UnpackFuncResult &uf) {
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{descTy},
                                            ValueRange{uf.base, uf.shape0,
                                                       uf.shape1, uf.stride0,
                                                       uf.stride1, uf.padding})
        .getResult(0);
  }

  void createTmaCopy(ValueRange operands) {
    OperationState state(loc, "ttg.tma_copy");
    state.addOperands(operands);
    builder.create(state);
  }

  // -- Pass execution --------------------------------------------------------

  LogicalResult runPass() {
    PassManager pm((*module)->getName(), PassManager::Nesting::Implicit);
    pm.enableVerifier(false);
    pm.addNestedPass<gpu::GPUModuleOp>(createTleToTritonGCUPass());
    return pm.run(*module);
  }

  // -- Assertion helpers -----------------------------------------------------

  template <typename OpT> int countOps(func::FuncOp funcOp) {
    int n = 0;
    funcOp.walk([&](OpT) { n++; });
    return n;
  }

  int countTmaCopy(func::FuncOp funcOp) {
    int n = 0;
    funcOp.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "ttg.tma_copy")
        n++;
    });
    return n;
  }
};

} // namespace

// Verify that insertLocalPointerBarriers (Phase 1 of TleToTritonGCUPass)
// inserts a gpu.barrier between a G2L ttg.tma_copy and a subsequent
// L2G ttg.tma_copy sharing the same memdesc.
// Covers TleToTritonGCU.cpp lines 104-116 (collectLocalPointerGroups
// tma_copy handler).
TEST(TmaCopyBarrierTest, BarrierInsertedBetweenG2LAndL2G) {
  TmaCopyFixture f;

  auto f32Ty = f.builder.getF32Type();
  auto tensorDescTy = f.makeTensorDescType(f32Ty);
  auto memdescTy = f.makeMemDescType(f32Ty);

  auto funcType = f.builder.getFunctionType(
      {tensorDescTy, memdescTy, f.builder.getI32Type()}, {});
  auto funcOp =
      f.builder.create<func::FuncOp>(f.loc, "tma_copy_barrier_test", funcType);
  auto *entryBlock = funcOp.addEntryBlock();
  f.builder.setInsertionPointToStart(entryBlock);

  Value tensorDesc = entryBlock->getArgument(0);
  Value memdesc = entryBlock->getArgument(1);
  Value offset = entryBlock->getArgument(2);

  f.createTmaCopy({tensorDesc, memdesc, offset}); // G2L
  f.createTmaCopy({memdesc, tensorDesc, offset}); // L2G
  f.builder.create<func::ReturnOp>(f.loc);

  ASSERT_TRUE(succeeded(f.runPass()));

  EXPECT_EQ(f.countOps<gpu::BarrierOp>(funcOp), 1);

  SmallVector<StringRef> opOrder;
  funcOp.walk([&](Operation *op) {
    auto name = op->getName().getStringRef();
    if (name == "ttg.tma_copy" || name == "gpu.barrier")
      opOrder.push_back(name);
  });
  ASSERT_EQ(opOrder.size(), 3u);
  EXPECT_EQ(opOrder[0], "ttg.tma_copy");
  EXPECT_EQ(opOrder[1], "gpu.barrier");
  EXPECT_EQ(opOrder[2], "ttg.tma_copy");
}

// Verify that ConvertTMACopyOp (Phase 2 of TleToTritonGCUPass)
// lowers a G2L ttg.tma_copy when the tensor_desc operand is produced by
// an unrealized_conversion_cast with (ptr, shape0, shape1, stride0, stride1,
// padding) — matching the pattern from
// add_rewrite_tensor_descriptor_to_pointer. This triggers unpackTmaDescriptor
// to extract base/shapes/strides/padding, then generates tmaGeneratePtr +
// tmaGenerateMask + tmaGenerateOther + tt.load
// + ttg.local_store.
// Covers TleToTritonGCU.cpp:
//   lines 645-655 (unpackTmaDescriptor body)
//   lines 669-707 (tmaGetOffsetRange: MakeRange + ExpandDims + ExtSI + AddI)
//   lines 709-737 (tmaGeneratePtr: splat + offset*stride + broadcast + addptr)
//   lines 739-803 (tmaGenerateMask + tmaGenerateOther)
TEST(TmaCopyBarrierTest, ConvertTMACopyG2LUnpacks) {
  TmaCopyFixture f;

  auto f32Ty = f.builder.getF32Type();
  auto tensorDescTy = f.makeTensorDescType(f32Ty);
  auto memdescTy = f.makeMemDescType(f32Ty);
  auto uf = f.createUnpackFunc("tma_copy_unpack_test", f32Ty);

  Value descVal = f.createDescriptorCast(tensorDescTy, uf);
  auto localAlloc = f.builder.create<ttg::LocalAllocOp>(f.loc, memdescTy);

  f.createTmaCopy({descVal, localAlloc.getResult(), uf.offset0, uf.offset1});
  f.builder.create<func::ReturnOp>(f.loc);

  ASSERT_TRUE(succeeded(f.runPass()));

  EXPECT_EQ(f.countTmaCopy(uf.funcOp), 0)
      << "ttg.tma_copy should have been lowered";

  EXPECT_GE(f.countOps<triton::LoadOp>(uf.funcOp), 1)
      << "Expected tt.load from G2L lowering";
  EXPECT_GE(f.countOps<ttg::LocalStoreOp>(uf.funcOp), 1)
      << "Expected ttg.local_store from G2L lowering";

  // tmaGetOffsetRange produces MakeRange + ExpandDims + ExtSI per dim.
  EXPECT_GE(f.countOps<triton::MakeRangeOp>(uf.funcOp), 2)
      << "Expected MakeRange from tmaGetOffsetRange (at least 2 for rank-2)";
  EXPECT_GE(f.countOps<triton::ExpandDimsOp>(uf.funcOp), 2)
      << "Expected ExpandDims from tmaGetOffsetRange rank-2 path";
  EXPECT_GE(f.countOps<arith::ExtSIOp>(uf.funcOp), 2)
      << "Expected ExtSI (i32->i64) from tmaGetOffsetRange";

  // tmaGeneratePtr: splat(base_ptr) + per-dim (splat(stride) * muli + addptr).
  EXPECT_GE(f.countOps<triton::SplatOp>(uf.funcOp), 1)
      << "Expected tt.splat from tmaGeneratePtr";
  EXPECT_GE(f.countOps<arith::MulIOp>(uf.funcOp), 2)
      << "Expected arith.muli (offset*stride) per dim";
  EXPECT_GE(f.countOps<triton::AddPtrOp>(uf.funcOp), 2)
      << "Expected tt.addptr (ptr + offset) per dim from tmaGeneratePtr";

  // tmaGenerateMask: sge + slt per dim = 4 CmpIOp, 3+ AndIOp.
  EXPECT_GE(f.countOps<arith::CmpIOp>(uf.funcOp), 4)
      << "Expected 4 arith.cmpi (2 per dim: sge + slt) from tmaGenerateMask";
  EXPECT_GE(f.countOps<arith::AndIOp>(uf.funcOp), 3)
      << "Expected at least 3 arith.andi from tmaGenerateMask";

  // tmaGenerateOther: f32 + i1 paddingOption -> NaN/zero select.
  EXPECT_GE(f.countOps<arith::SelectOp>(uf.funcOp), 1)
      << "Expected arith.select from tmaGenerateOther float NaN path";
}

// Verify ConvertTMACopyOp with an integer element type (i32).
// tmaGenerateOther takes the else branch (line 798-800): just zero constant.
// Covers TleToTritonGCU.cpp lines 798-800 (non-float tmaGenerateOther path).
TEST(TmaCopyBarrierTest, ConvertTMACopyG2LIntegerOther) {
  TmaCopyFixture f;

  auto i32Ty = f.builder.getI32Type();
  auto tensorDescTy = f.makeTensorDescType(i32Ty);
  auto memdescTy = f.makeMemDescType(i32Ty);
  auto uf = f.createUnpackFunc("tma_copy_int_other_test", i32Ty);

  Value descVal = f.createDescriptorCast(tensorDescTy, uf);
  auto localAlloc = f.builder.create<ttg::LocalAllocOp>(f.loc, memdescTy);

  f.createTmaCopy({descVal, localAlloc.getResult(), uf.offset0, uf.offset1});
  f.builder.create<func::ReturnOp>(f.loc);

  ASSERT_TRUE(succeeded(f.runPass()));

  EXPECT_EQ(f.countTmaCopy(uf.funcOp), 0)
      << "ttg.tma_copy should have been lowered";
  EXPECT_EQ(f.countOps<arith::SelectOp>(uf.funcOp), 0)
      << "Integer tmaGenerateOther should not produce arith.select";
  EXPECT_GE(f.countOps<triton::LoadOp>(uf.funcOp), 1)
      << "Expected tt.load from G2L lowering";
}

// Verify ConvertTMACopyOp L2G (LocalToGlobal) path: ttg.local_load + tt.store.
// Operand order: (memdesc, tensor_desc_from_cast, offset0, offset1).
// Covers TleToTritonGCU.cpp lines 886-894 (isLocalToGlobal branch).
TEST(TmaCopyBarrierTest, ConvertTMACopyL2G) {
  TmaCopyFixture f;

  auto f32Ty = f.builder.getF32Type();
  auto tensorDescTy = f.makeTensorDescType(f32Ty);
  auto memdescTy = f.makeMemDescType(f32Ty);
  auto uf = f.createUnpackFunc("tma_copy_l2g_test", f32Ty);

  Value descVal = f.createDescriptorCast(tensorDescTy, uf);
  auto localAlloc = f.builder.create<ttg::LocalAllocOp>(f.loc, memdescTy);

  // L2G: (memdesc, tensor_desc, offset0, offset1)
  f.createTmaCopy({localAlloc.getResult(), descVal, uf.offset0, uf.offset1});
  f.builder.create<func::ReturnOp>(f.loc);

  ASSERT_TRUE(succeeded(f.runPass()));

  EXPECT_EQ(f.countTmaCopy(uf.funcOp), 0)
      << "ttg.tma_copy should have been lowered";
  EXPECT_GE(f.countOps<ttg::LocalLoadOp>(uf.funcOp), 1)
      << "Expected ttg.local_load from L2G lowering";
  EXPECT_GE(f.countOps<triton::StoreOp>(uf.funcOp), 1)
      << "Expected tt.store from L2G lowering";
  EXPECT_EQ(f.countOps<arith::SelectOp>(uf.funcOp), 0)
      << "L2G should not produce arith.select";
}

#else // TEST_GCU300

TEST(TmaCopyBarrierTest, NotApplicableOnGCU300) { SUCCEED(); }

#endif
