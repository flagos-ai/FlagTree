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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#ifdef TEST_GCU400
#include "Conversion/TritonToGCU/Utility.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#endif

using namespace mlir;

namespace {

struct MergeContinuousDimsFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  func::FuncOp funcOp;
  Block *entryBlock;
  func::ReturnOp returnOp;

  MergeContinuousDimsFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
    ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect>();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
  }

  // Creates a function with two memref args (shared in addrspace 2, warp
  // in default) and positions the builder inside it.
  // Returns {sharedBuffer, warpOutput} values.
  std::pair<Value, Value> setup(ArrayRef<int64_t> sharedShape,
                                ArrayRef<int64_t> warpShape) {
    auto sharedMemType =
        MemRefType::get(sharedShape, builder.getF32Type(), AffineMap{},
                        builder.getI64IntegerAttr(2));
    auto warpMemType = MemRefType::get(warpShape, builder.getF32Type());
    auto funcType = builder.getFunctionType({sharedMemType, warpMemType}, {});
    funcOp = builder.create<func::FuncOp>(loc, "test_merge", funcType);
    entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    returnOp = builder.create<func::ReturnOp>(loc);
    builder.setInsertionPoint(returnOp);
    return {entryBlock->getArgument(0), entryBlock->getArgument(1)};
  }
};

} // namespace

// When shared and warp shapes differ on every dimension, consecutive equal
// dims do not exist, so merged shapes mirror the originals.
TEST(MergeContinuousDimsTest, AllDimsDifferent) {
  MergeContinuousDimsFixture f;
  // shared: [8, 4], warp: [2, 1] -- both dims differ
  auto [sharedBuffer, warpOutput] = f.setup({8, 4}, {2, 1});

  auto sharedMemType = cast<MemRefType>(sharedBuffer.getType());
  auto warpMemType = cast<MemRefType>(warpOutput.getType());

  SmallVector<Value, 4> offsets;
  auto off0 = f.builder.create<arith::ConstantIntOp>(f.loc, 0, 32);
  auto off1 = f.builder.create<arith::ConstantIntOp>(f.loc, 1, 32);
  offsets.push_back(off0);
  offsets.push_back(off1);

  SmallVector<Value, 4> mergedOffsets;
  Value sharedMemref, warpMemref;

  mergeContinuousDims(f.builder, f.loc, sharedMemref, warpMemref, offsets,
                      mergedOffsets, sharedMemType, warpMemType, sharedBuffer,
                      warpOutput);

  ASSERT_TRUE(sharedMemref != nullptr);
  ASSERT_TRUE(warpMemref != nullptr);

  auto sharedCast = sharedMemref.getDefiningOp<memref::ReinterpretCastOp>();
  auto warpCast = warpMemref.getDefiningOp<memref::ReinterpretCastOp>();
  ASSERT_NE(sharedCast, nullptr);
  ASSERT_NE(warpCast, nullptr);

  // Both dims differ so no merging: merged rank stays 2.
  EXPECT_EQ(sharedCast.getType().getRank(), 2);
  EXPECT_EQ(warpCast.getType().getRank(), 2);
  EXPECT_EQ(sharedCast.getType().getShape()[0], 8);
  EXPECT_EQ(sharedCast.getType().getShape()[1], 4);
  EXPECT_EQ(warpCast.getType().getShape()[0], 2);
  EXPECT_EQ(warpCast.getType().getShape()[1], 1);
}

// When all dimensions are equal, they should be merged into a single dim.
TEST(MergeContinuousDimsTest, AllDimsEqual) {
  MergeContinuousDimsFixture f;
  // shared: [4, 8, 2], warp: [4, 8, 2] -- all dims equal
  auto [sharedBuffer, warpOutput] = f.setup({4, 8, 2}, {4, 8, 2});

  auto sharedMemType = cast<MemRefType>(sharedBuffer.getType());
  auto warpMemType = cast<MemRefType>(warpOutput.getType());

  SmallVector<Value, 4> offsets;
  for (int i = 0; i < 3; ++i)
    offsets.push_back(f.builder.create<arith::ConstantIntOp>(f.loc, 0, 32));

  SmallVector<Value, 4> mergedOffsets;
  Value sharedMemref, warpMemref;

  mergeContinuousDims(f.builder, f.loc, sharedMemref, warpMemref, offsets,
                      mergedOffsets, sharedMemType, warpMemType, sharedBuffer,
                      warpOutput);

  auto sharedCast = sharedMemref.getDefiningOp<memref::ReinterpretCastOp>();
  auto warpCast = warpMemref.getDefiningOp<memref::ReinterpretCastOp>();
  ASSERT_NE(sharedCast, nullptr);
  ASSERT_NE(warpCast, nullptr);

  // All equal dims merged into one: 4*8*2 = 64
  EXPECT_EQ(sharedCast.getType().getRank(), 1);
  EXPECT_EQ(sharedCast.getType().getShape()[0], 64);
  EXPECT_EQ(warpCast.getType().getRank(), 1);
  EXPECT_EQ(warpCast.getType().getShape()[0], 64);
}

// Mixed pattern: first two dims equal, last dim differs.
// Equal prefix dims get merged; the differing dim stays separate.
TEST(MergeContinuousDimsTest, MixedEqualThenDifferent) {
  MergeContinuousDimsFixture f;
  // shared: [4, 8, 16], warp: [4, 8, 2]
  // dims 0,1 equal (merge 4*8=32), dim 2 differs
  auto [sharedBuffer, warpOutput] = f.setup({4, 8, 16}, {4, 8, 2});

  auto sharedMemType = cast<MemRefType>(sharedBuffer.getType());
  auto warpMemType = cast<MemRefType>(warpOutput.getType());

  SmallVector<Value, 4> offsets;
  for (int i = 0; i < 3; ++i)
    offsets.push_back(f.builder.create<arith::ConstantIntOp>(f.loc, i, 32));

  SmallVector<Value, 4> mergedOffsets;
  Value sharedMemref, warpMemref;

  mergeContinuousDims(f.builder, f.loc, sharedMemref, warpMemref, offsets,
                      mergedOffsets, sharedMemType, warpMemType, sharedBuffer,
                      warpOutput);

  auto sharedCast = sharedMemref.getDefiningOp<memref::ReinterpretCastOp>();
  auto warpCast = warpMemref.getDefiningOp<memref::ReinterpretCastOp>();
  ASSERT_NE(sharedCast, nullptr);
  ASSERT_NE(warpCast, nullptr);

  // dims 0,1 merged (32), dim 2 stays: rank 2
  EXPECT_EQ(sharedCast.getType().getRank(), 2);
  EXPECT_EQ(sharedCast.getType().getShape()[0], 32);
  EXPECT_EQ(sharedCast.getType().getShape()[1], 16);
  EXPECT_EQ(warpCast.getType().getRank(), 2);
  EXPECT_EQ(warpCast.getType().getShape()[0], 32);
  EXPECT_EQ(warpCast.getType().getShape()[1], 2);
}

// Pattern: first dim differs, then consecutive equal dims at the end.
TEST(MergeContinuousDimsTest, DifferentThenEqual) {
  MergeContinuousDimsFixture f;
  // shared: [16, 4, 2], warp: [2, 4, 2]
  // dim 0 differs; dims 1,2 equal (merge 4*2=8)
  auto [sharedBuffer, warpOutput] = f.setup({16, 4, 2}, {2, 4, 2});

  auto sharedMemType = cast<MemRefType>(sharedBuffer.getType());
  auto warpMemType = cast<MemRefType>(warpOutput.getType());

  SmallVector<Value, 4> offsets;
  for (int i = 0; i < 3; ++i)
    offsets.push_back(f.builder.create<arith::ConstantIntOp>(f.loc, i, 32));

  SmallVector<Value, 4> mergedOffsets;
  Value sharedMemref, warpMemref;

  mergeContinuousDims(f.builder, f.loc, sharedMemref, warpMemref, offsets,
                      mergedOffsets, sharedMemType, warpMemType, sharedBuffer,
                      warpOutput);

  auto sharedCast = sharedMemref.getDefiningOp<memref::ReinterpretCastOp>();
  auto warpCast = warpMemref.getDefiningOp<memref::ReinterpretCastOp>();
  ASSERT_NE(sharedCast, nullptr);
  ASSERT_NE(warpCast, nullptr);

  // dim 0 differs stays, dims 1+2 merged: rank 2
  EXPECT_EQ(sharedCast.getType().getRank(), 2);
  EXPECT_EQ(sharedCast.getType().getShape()[0], 16);
  EXPECT_EQ(sharedCast.getType().getShape()[1], 8);
  EXPECT_EQ(warpCast.getType().getRank(), 2);
  EXPECT_EQ(warpCast.getType().getShape()[0], 2);
  EXPECT_EQ(warpCast.getType().getShape()[1], 8);
}

// Shared memory is in address space 2; verify the output ReinterpretCast
// preserves the address space.
TEST(MergeContinuousDimsTest, SharedMemorySpacePreserved) {
  MergeContinuousDimsFixture f;
  auto [sharedBuffer, warpOutput] = f.setup({4, 4}, {4, 4});

  auto sharedMemType = cast<MemRefType>(sharedBuffer.getType());
  auto warpMemType = cast<MemRefType>(warpOutput.getType());

  SmallVector<Value, 4> offsets;
  for (int i = 0; i < 2; ++i)
    offsets.push_back(f.builder.create<arith::ConstantIntOp>(f.loc, 0, 32));

  SmallVector<Value, 4> mergedOffsets;
  Value sharedMemref, warpMemref;

  mergeContinuousDims(f.builder, f.loc, sharedMemref, warpMemref, offsets,
                      mergedOffsets, sharedMemType, warpMemType, sharedBuffer,
                      warpOutput);

  auto sharedCast = sharedMemref.getDefiningOp<memref::ReinterpretCastOp>();
  ASSERT_NE(sharedCast, nullptr);

  // Shared memory address space (2) should be preserved.
  auto resultMemSpace = sharedCast.getType().getMemorySpace();
  ASSERT_TRUE(resultMemSpace != nullptr);
  auto intAttr = dyn_cast<IntegerAttr>(resultMemSpace);
  ASSERT_TRUE(intAttr != nullptr);
  EXPECT_EQ(intAttr.getInt(), 2);
}
