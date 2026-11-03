// RUN: triton-opt %s --split-input-file --verify-each | FileCheck %s --check-prefix=PARSE
// RUN: triton-opt %s --split-input-file --allocate-hcugpu-shared-memory --convert-triton-hcugpu-to-llvm="arch=gfx936" --convert-arith-to-llvm --verify-each | FileCheck %s
// CPU lowering fixtures only: aliased output sink, NEVER execute on GPU.
// Positive DotB/chart and Linear raw producer dimensions come from actual
// frozen e84/65 kernels. Negative cases retain generic shared reads.
// PARSE-LABEL: @bf16_rank3_dot(
// CHECK-LABEL: llvm.func @bf16_rank3_dot(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_rank3_dot(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<4x32x16xbf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<4x32x16xbf16, #dst>) -> !ttg.memdesc<4x32x16xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<4x32x16xbf16, #shared, #smem, mutable> -> tensor<4x32x16xbf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<4x32x16x!tt.ptr<bf16>, #dst>
    tt.store %ptrs, %got : tensor<4x32x16x!tt.ptr<bf16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f16_rank3_dot(
// CHECK-LABEL: llvm.func @f16_rank3_dot(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f16_rank3_dot(%out: !tt.ptr<f16>) {
    %values = arith.constant dense<1.0> : tensor<4x32x16xf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<4x32x16xf16, #dst>) -> !ttg.memdesc<4x32x16xf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<4x32x16xf16, #shared, #smem, mutable> -> tensor<4x32x16xf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<4x32x16x!tt.ptr<f16>, #dst>
    tt.store %ptrs, %got : tensor<4x32x16x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_rank3_linear(
// CHECK-LABEL: llvm.func @bf16_rank3_linear(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [4, 0, 0], [8, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_rank3_linear(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<32x4x16xbf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<32x4x16xbf16, #dst>) -> !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable> -> tensor<32x4x16xbf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.store %ptrs, %got : tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f16_rank3_linear(
// CHECK-LABEL: llvm.func @f16_rank3_linear(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [4, 0, 0], [8, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f16_rank3_linear(%out: !tt.ptr<f16>) {
    %values = arith.constant dense<1.0> : tensor<32x4x16xf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<32x4x16xf16, #dst>) -> !ttg.memdesc<32x4x16xf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<32x4x16xf16, #shared, #smem, mutable> -> tensor<32x4x16xf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<f16> -> tensor<32x4x16x!tt.ptr<f16>, #dst>
    tt.store %ptrs, %got : tensor<32x4x16x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_rank2_linear(
// CHECK-LABEL: llvm.func @bf16_rank2_linear(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK-NOT: llvm.load {{.*}}!llvm.ptr<3>
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0], [2, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_rank2_linear(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<128x16xbf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<128x16xbf16, #dst>) -> !ttg.memdesc<128x16xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<128x16xbf16, #shared, #smem, mutable> -> tensor<128x16xbf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<128x16x!tt.ptr<bf16>, #dst>
    tt.store %ptrs, %got : tensor<128x16x!tt.ptr<bf16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_rank3_lane_swap(
// CHECK-LABEL: llvm.func @bf16_rank3_lane_swap(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.load {{.*}} : !llvm.ptr<3>
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 2], [0, 0, 1], [0, 0, 4], [0, 0, 8], [4, 0, 0], [8, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_rank3_lane_swap(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<32x4x16xbf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<32x4x16xbf16, #dst>) -> !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable> -> tensor<32x4x16xbf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.store %ptrs, %got : tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @bf16_rank3_alignment8(
// CHECK-LABEL: llvm.func @bf16_rank3_alignment8(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.load {{.*}} : !llvm.ptr<3>
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [4, 0, 0], [8, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @bf16_rank3_alignment8(%out: !tt.ptr<bf16>) {
    %values = arith.constant dense<1.0> : tensor<32x4x16xbf16, #dst>
    %dst = ttg.local_alloc %values {alignment = 8 : i32} : (tensor<32x4x16xbf16, #dst>) -> !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<32x4x16xbf16, #shared, #smem, mutable> -> tensor<32x4x16xbf16, #dst>
    %ptrs = tt.splat %out : !tt.ptr<bf16> -> tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.store %ptrs, %got : tensor<32x4x16x!tt.ptr<bf16>, #dst>
    tt.return
  }
}

// -----

// PARSE-LABEL: @f32_rank3_linear(
// CHECK-LABEL: llvm.func @f32_rank3_linear(
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.load {{.*}} : !llvm.ptr<3>
// CHECK-NOT: llvm.call @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(
// CHECK: llvm.return

#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#dst = #ttg.linear<{register = [[1, 0, 0], [2, 0, 0], [16, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [4, 0, 0], [8, 0, 0]], warp = [[0, 1, 0], [0, 2, 0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @f32_rank3_linear(%out: !tt.ptr<f32>) {
    %values = arith.constant dense<1.0> : tensor<32x4x16xf32, #dst>
    %dst = ttg.local_alloc %values {alignment = 16 : i32} : (tensor<32x4x16xf32, #dst>) -> !ttg.memdesc<32x4x16xf32, #shared, #smem, mutable>
    %got = ttg.local_load %dst : !ttg.memdesc<32x4x16xf32, #shared, #smem, mutable> -> tensor<32x4x16xf32, #dst>
    %ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<32x4x16x!tt.ptr<f32>, #dst>
    tt.store %ptrs, %got : tensor<32x4x16x!tt.ptr<f32>, #dst>
    tt.return
  }
}
