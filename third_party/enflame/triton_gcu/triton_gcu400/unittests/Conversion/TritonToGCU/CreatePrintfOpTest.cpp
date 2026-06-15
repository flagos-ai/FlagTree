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

#ifdef TEST_GCU400
#include "Conversion/TritonToGCU/Utility.h"
#else
#include "Conversion/TritonToGCU/TritonGCUToGCU/TritonGCUToGCUUtils.h"
#endif

using namespace mlir;

namespace {

struct PrintfTestFixture {
  MLIRContext ctx;
  OpBuilder builder;
  Location loc;
  ModuleOp module;
  gpu::GPUModuleOp gpuModule;

  PrintfTestFixture() : builder(&ctx), loc(builder.getUnknownLoc()) {
    ctx.loadDialect<arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect, affine::AffineDialect, gpu::GPUDialect,
                    gcu::GCUDialect>();
    module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());
    gpuModule = builder.create<gpu::GPUModuleOp>(loc, "triton");
    builder.setInsertionPointToStart(&gpuModule.getBodyRegion().front());
  }

  // Create a function with a single argument of the given type,
  // set the builder insertion point before the return op.
  std::pair<func::FuncOp, Value> setup(Type argType, StringRef funcName) {
    auto funcType = builder.getFunctionType({argType}, {});
    auto funcOp = builder.create<func::FuncOp>(loc, funcName, funcType);
    auto *entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    builder.create<func::ReturnOp>(loc);
    builder.setInsertionPoint(&entryBlock->front());
    return {funcOp, entryBlock->getArgument(0)};
  }
};

} // namespace

#ifdef TEST_GCU400

// Test: scalar f32 → gpu.printf with "%f " format
TEST(CreatePrintfOpTest, ScalarF32) {
  PrintfTestFixture f;
  auto [funcOp, val] = f.setup(f.builder.getF32Type(), "test_printf_f32");

  createPrintfOp(f.builder, f.loc, "debug", /*hex=*/false, val);

  int printfCount = 0;
  funcOp.walk([&](gpu::PrintfOp op) {
    ++printfCount;
    StringRef fmt = op.getFormat();
    EXPECT_NE(fmt.find("%f"), StringRef::npos)
        << "f32 should use %f format, got: " << fmt.str();
    EXPECT_NE(fmt.find("debug"), StringRef::npos)
        << "prefix should appear in format, got: " << fmt.str();
  });
  EXPECT_EQ(printfCount, 1);
}

// Test: scalar i32 → gpu.printf with "%d " format
TEST(CreatePrintfOpTest, ScalarI32) {
  PrintfTestFixture f;
  auto [funcOp, val] = f.setup(f.builder.getI32Type(), "test_printf_i32");

  createPrintfOp(f.builder, f.loc, "val", /*hex=*/false, val);

  int printfCount = 0;
  funcOp.walk([&](gpu::PrintfOp op) {
    ++printfCount;
    StringRef fmt = op.getFormat();
    EXPECT_NE(fmt.find("%d"), StringRef::npos)
        << "i32 should use %d format, got: " << fmt.str();
  });
  EXPECT_EQ(printfCount, 1);
}

// Test: scalar i32 with hex=true → gpu.printf with "0x%x " format
TEST(CreatePrintfOpTest, ScalarI32Hex) {
  PrintfTestFixture f;
  auto [funcOp, val] = f.setup(f.builder.getI32Type(), "test_printf_i32_hex");

  createPrintfOp(f.builder, f.loc, "addr", /*hex=*/true, val);

  int printfCount = 0;
  funcOp.walk([&](gpu::PrintfOp op) {
    ++printfCount;
    StringRef fmt = op.getFormat();
    EXPECT_NE(fmt.find("0x%x"), StringRef::npos)
        << "hex=true should use 0x%x format, got: " << fmt.str();
  });
  EXPECT_EQ(printfCount, 1);
}

// Test: memref<2x3xf32> → affine loop nest + memref.load + gpu.printf per elem
TEST(CreatePrintfOpTest, MemRef2D) {
  PrintfTestFixture f;
  auto memrefType = MemRefType::get({2, 3}, f.builder.getF32Type());
  auto [funcOp, val] = f.setup(memrefType, "test_printf_memref");

  createPrintfOp(f.builder, f.loc, "mem", /*hex=*/false, val);

  // The memref branch creates an affine loop nest with memref.load + printf.
  int loadCount = 0;
  funcOp.walk([&](memref::LoadOp) { ++loadCount; });
  EXPECT_GE(loadCount, 1) << "MemRef path should generate memref.load";

  int printfCount = 0;
  funcOp.walk([&](gpu::PrintfOp op) {
    ++printfCount;
    StringRef fmt = op.getFormat();
    EXPECT_NE(fmt.find("%f"), StringRef::npos)
        << "loaded f32 element should use %f format";
    // Index format "(idx %d, %d)" should appear for 2D memref
    EXPECT_NE(fmt.find("idx"), StringRef::npos)
        << "memref elements should include idx in format, got: " << fmt.str();
  });
  EXPECT_GE(printfCount, 1);
}

#else // TEST_GCU300

// gcu300 createPrintfOp takes ConversionPatternRewriter& whose constructor is
// private. It cannot be instantiated outside of the dialect conversion
// infrastructure, so these tests are skipped for gcu300.
TEST(CreatePrintfOpTest, ScalarF32) {
  GTEST_SKIP() << "gcu300 createPrintfOp requires ConversionPatternRewriter";
}

TEST(CreatePrintfOpTest, ScalarI32) {
  GTEST_SKIP() << "gcu300 createPrintfOp requires ConversionPatternRewriter";
}

TEST(CreatePrintfOpTest, ScalarI32Hex) {
  GTEST_SKIP() << "gcu300 createPrintfOp requires ConversionPatternRewriter";
}

TEST(CreatePrintfOpTest, MemRef2D) {
  GTEST_SKIP() << "gcu300 createPrintfOp requires ConversionPatternRewriter";
}

#endif
