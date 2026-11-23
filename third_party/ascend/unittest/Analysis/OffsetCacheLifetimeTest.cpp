/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"
#include <cstddef>

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ReleaseProbe {
  Value watched;
  llvm::DenseMap<Value, PtrOffsetInfo> *cache;
  unsigned releases = 0;
  bool staleKey = false;
  bool stalePayload = false;
};
thread_local ReleaseProbe *activeProbe = nullptr;

// Observe deallocation only in this test executable. Do not allocate, print,
// or dereference the watched Value here: its destructor has already run.
void observeRelease(void *pointer) {
  auto *probe = activeProbe;
  if (!probe || pointer != probe->watched.getAsOpaquePointer())
    return;
  ++probe->releases;
  probe->staleKey |= probe->cache->contains(probe->watched);
  for (auto &entry : *probe->cache) {
    auto &info = entry.second;
    probe->stalePayload |=
        info.getPtr() == probe->watched || info.getOffset() == probe->watched;
    for (Value offset : info.getOffsetsRef())
      probe->stalePayload |= offset == probe->watched;
  }
}
} // namespace

// ELF linker wrapping redirects delete calls from the linked MLIR objects;
// the actual allocator and production implementation remain unchanged.
extern "C" void __real__ZdlPv(void *) noexcept;
extern "C" void __real__ZdlPvm(void *, std::size_t) noexcept;
extern "C" void __wrap__ZdlPv(void *pointer) noexcept {
  observeRelease(pointer);
  __real__ZdlPv(pointer);
}
extern "C" void __wrap__ZdlPvm(void *pointer, std::size_t size) noexcept {
  observeRelease(pointer);
  __real__ZdlPvm(pointer, size);
}

TEST(OffsetCacheLifetime, ReleasesOldValuesAfterCacheInvalidation) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @cache_lifetime(%base: !tt.ptr<i32>, %n: i32) {
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %p0 = tt.addptr %base, %n : !tt.ptr<i32>, i32
        %result = scf.for %i = %zero to %n step %one
            iter_args(%p = %p0) -> (!tt.ptr<i32>) : i32 {
          %next = tt.addptr %p, %one : !tt.ptr<i32>, i32
          scf.yield %next : !tt.ptr<i32>
        }
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto loop = *func.getBody().front().getOps<scf::ForOp>().begin();
  Value oldArg = loop.getRegionIterArgs().front();
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  IRRewriter rewriter(&context);
  parse(oldArg, oldArg.getLoc(), rewriter, cache);
  ASSERT_TRUE(cache.contains(oldArg));

  ReleaseProbe probe{oldArg, &cache};
  ASSERT_EQ(activeProbe, nullptr);
  {
    activeProbe = &probe;
    auto resetProbe = llvm::make_scope_exit([] { activeProbe = nullptr; });
    replacePtrArguments(func, cache);
  }

  // Do not silently pass if the linker or MLIR allocation path stops routing
  // the watched deallocation through the wrappers.
  EXPECT_EQ(probe.releases, 1u);
  EXPECT_FALSE(probe.staleKey);
  EXPECT_FALSE(probe.stalePayload);
  EXPECT_TRUE(succeeded(verify(*module)));
}
