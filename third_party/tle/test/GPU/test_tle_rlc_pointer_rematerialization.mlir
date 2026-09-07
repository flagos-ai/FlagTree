// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions -triton-tle-select-encodings -tritongpu-remove-layout-conversions -tritongpu-reduce-data-duplication | FileCheck %s

// Dot-operand conversion hoisting handles matrix data, not pointer arithmetic.
// A pure address DAG should be rematerialized in the required load layout by
// RLC, instead of leaving a pointer conversion for shared data staging.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @rematerialize_pointer
  tt.func @rematerialize_pointer(%base: !tt.ptr<bf16>, %other: tensor<128x64xbf16, #rhs>, %output: tensor<2x64x!tt.ptr<f32>, #mma>, %limit: i32) {
    %rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %limits = tt.splat %limit : i32 -> tensor<2xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %active = arith.cmpi slt, %rows, %limits : tensor<2xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %active_rows = tt.expand_dims %active {axis = 1 : i32} : tensor<2xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<2x1xi1, #mma>
    // CHECK: %[[MASK:.*]] = tt.broadcast {{.*}} -> tensor<2x128xi1, #ttg.dot_op<{{.*}}>>
    %mask = tt.broadcast %active_rows : tensor<2x1xi1, #mma> -> tensor<2x128xi1, #mma>
    %converted_mask = ttg.convert_layout %mask : tensor<2x128xi1, #mma> -> tensor<2x128xi1, #lhs>
    %padding = arith.constant dense<0.0> : tensor<2x128xbf16, #lhs>
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %offset = tt.broadcast %row : tensor<1x128xi32, #blocked> -> tensor<2x128xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<bf16> -> tensor<2x128x!tt.ptr<bf16>, #blocked>
    // CHECK: %[[PTR:.*]] = tt.addptr {{.*}} : tensor<2x128x!tt.ptr<bf16>, #ttg.dot_op<{{.*}}>>,
    %ptr = tt.addptr %bases, %offset : tensor<2x128x!tt.ptr<bf16>, #blocked>, tensor<2x128xi32, #blocked>
    %converted = ttg.convert_layout %ptr : tensor<2x128x!tt.ptr<bf16>, #blocked> -> tensor<2x128x!tt.ptr<bf16>, #lhs>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[VALUE:.*]] = tt.load %[[PTR]], %[[MASK]], {{.*}} : tensor<2x128x!tt.ptr<bf16>, #ttg.dot_op<{{.*}}>>
    %value = tt.load %converted, %converted_mask, %padding {isVolatile = true, tle.explicit_memory_encoding = #lhs} : tensor<2x128x!tt.ptr<bf16>, #lhs>
    %zero = arith.constant dense<0.0> : tensor<2x64xf32, #mma>
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    // CHECK-NOT: tt.load
    // CHECK: tt.dot %[[VALUE]],
    %dot = tt.dot %value, %other, %zero : tensor<2x128xbf16, #lhs> * tensor<128x64xbf16, #rhs> -> tensor<2x64xf32, #mma>
    tt.store %output, %dot : tensor<2x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
