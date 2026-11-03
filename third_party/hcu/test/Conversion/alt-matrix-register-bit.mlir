// RUN: triton-opt %s --split-input-file --verify-each | FileCheck %s --check-prefix=PARSE
// RUN: triton-opt %s --split-input-file --allocate-hcugpu-shared-memory --convert-triton-hcugpu-to-llvm="arch=gfx936" --convert-arith-to-llvm --verify-each | FileCheck %s
//
// Start from the existing valid tokenless ALT fixtures. Only the result
// register basis order and function name differ. No private selector hint,
// manual syncedViaAsyncWait attribute, token or compiler test pass is used.
// The four positives relocate the exact physical offset-1 basis to reg3/4;
// reg0/1 remain outside the recipe because producerReg occupies those bits.
// Initialized LDS keeps the fixture self-contained. The deliberately aliased
// output pointers are only a compiler-stage sink: NEVER execute on a GPU.

// PARSE-LABEL: @bf16_reg3(
// CHECK-LABEL: llvm.func @bf16_reg3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-COUNT-4: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 1], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_reg3(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xbf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xbf16, #linear>) -> !ttg.memdesc<16x256xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xbf16, #shared, #smem, mutable> -> tensor<16x256xbf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_reg4(
// CHECK-LABEL: llvm.func @bf16_reg4(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-COUNT-4: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 64], [0, 32], [0, 1]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_reg4(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xbf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xbf16, #linear>) -> !ttg.memdesc<16x256xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xbf16, #shared, #smem, mutable> -> tensor<16x256xbf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f16_reg3(
// CHECK-LABEL: llvm.func @f16_reg3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-COUNT-4: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 1], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f16_reg3(%out: !tt.ptr<f16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xf16, #linear>) -> !ttg.memdesc<16x256xf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xf16, #shared, #smem, mutable> -> tensor<16x256xf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<f16>, #linear>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f16_reg4(
// CHECK-LABEL: llvm.func @f16_reg4(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK-COUNT-4: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 64], [0, 32], [0, 1]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f16_reg4(%out: !tt.ptr<f16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xf16, #linear>) -> !ttg.memdesc<16x256xf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xf16, #shared, #smem, mutable> -> tensor<16x256xf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<f16>, #linear>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_reg0(
// CHECK-LABEL: llvm.func @bf16_reg0(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK: llvm.load {{.*}} : !llvm.ptr<3>
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[0, 1], [2, 0], [1, 0], [0, 32], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_reg0(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xbf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xbf16, #linear>) -> !ttg.memdesc<16x256xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xbf16, #shared, #smem, mutable> -> tensor<16x256xbf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<bf16>, #linear>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f16_reg1(
// CHECK-LABEL: llvm.func @f16_reg1(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK: llvm.load {{.*}} : !llvm.ptr<3>
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.alt
// CHECK: llvm.return

#linear = #ttg.linear<{register = [[1, 0], [0, 1], [2, 0], [0, 32], [0, 64]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [4, 0], [8, 0]], warp = [[0, 128], [0, 0], [0, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f16_reg1(%out: !tt.ptr<f16>) {
    %values = arith.constant dense<1.0> : tensor<16x256xf16, #linear>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<16x256xf16, #linear>) -> !ttg.memdesc<16x256xf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<16x256xf16, #shared, #smem, mutable> -> tensor<16x256xf16, #linear>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #linear>
    tt.store %ptrs, %got : tensor<16x256x!tt.ptr<f16>, #linear>
    tt.return
  }
}
