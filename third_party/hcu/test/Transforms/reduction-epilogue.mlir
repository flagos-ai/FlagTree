// RUN: triton-opt %s --split-input-file --tritonhcugpu-optimize-epilogue --verify-each | FileCheck %s

// Reduced batch-axis MMAC values may retain their slice layout at writeback.
// Other targets, axes, output sizes/types and warp distributions keep the
// existing conversion. The FP32 reduction remains intact in every case.

// CHECK-LABEL: @bf16_S4W4(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#ttg.slice
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @bf16_S4W4(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @fp16_S4W4(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#ttg.slice
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @fp16_S4W4(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<f16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<f16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @bf16_S8W4(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#ttg.slice
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @bf16_S8W4(%acc: tensor<8x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<8x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @bf16_S8W8(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#ttg.slice
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [8, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @bf16_S8W8(%acc: tensor<8x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<8x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @other_target(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx928", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @other_target(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @other_shape(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @other_shape(%acc: tensor<4x32x16xf32, #input>, %ptr: tensor<32x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<32x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x32x16xf32, #input>) -> tensor<32x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<32x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<32x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<32x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<32x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<32x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @other_axis(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @other_axis(%acc: tensor<16x4x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<16x4x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 1, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 1, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 1, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 1, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @other_warps(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @other_warps(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @other_batch(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @other_batch(%acc: tensor<2x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<2x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @fp32_output(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @fp32_output(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<f32>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.negf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xf32, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<f32>, #blocked>
  tt.return
 }
}

// -----

// CHECK-LABEL: @non_mma(
// CHECK: "tt.reduce"
// CHECK: tt.store {{.*}}#blocked
#input = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {ttg.target = "hip:gfx936", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
 tt.func @non_mma(%acc: tensor<4x16x16xf32, #input>, %ptr: tensor<16x16x!tt.ptr<bf16>, #blocked>, %mask: tensor<16x16xi1, #blocked>) {
  %r = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
  }) : (tensor<4x16x16xf32, #input>) -> tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>>
  %out = arith.truncf %r : tensor<16x16xf32, #ttg.slice<{dim = 0, parent = #input}>> to tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>>
  %cvt = ttg.convert_layout %out : tensor<16x16xbf16, #ttg.slice<{dim = 0, parent = #input}>> -> tensor<16x16xbf16, #blocked>
  tt.store %ptr, %cvt, %mask : tensor<16x16x!tt.ptr<bf16>, #blocked>
  tt.return
 }
}
