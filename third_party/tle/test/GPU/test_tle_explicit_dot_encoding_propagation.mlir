// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s --check-prefixes=CHECK,ANCHOR
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -tritongpu-remove-layout-conversions -canonicalize | FileCheck %s --check-prefix=ANCHOR

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func public @chained_dot_accumulator
  tt.func public @chained_dot_accumulator() attributes {noinline = false} {
    %lhs_value = arith.constant dense<0.0> : tensor<32x32xbf16>
    %lhs_encoded = tle.gpu.set_layout %lhs_value {target_encoding = #lhs} : tensor<32x32xbf16> -> tensor<32x32xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<32x8xbf16>
    %rhs_encoded = tle.gpu.set_layout %rhs_value {target_encoding = #rhs} : tensor<32x8xbf16> -> tensor<32x8xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<32x8xf32>
    %acc_encoded = tle.gpu.set_layout %acc_value {target_encoding = #mma} : tensor<32x8xf32> -> tensor<32x8xf32>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[FIRST:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {{.*}} -> tensor<32x8xf32, #mma>
    %first = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NEXT: %[[SECOND:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[FIRST]] {{.*}} -> tensor<32x8xf32, #mma>
    %second = tt.dot %lhs_encoded, %rhs_encoded, %first : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NOT: ttg.convert_layout
    tt.return
  }

  // An extract cannot absorb a set_layout by reinterpreting its registers.
  // ANCHOR-LABEL: tt.func @extract_layout_anchor
  // ANCHOR: %[[TILE:.*]] = tle.extract_tile {{.*}} : tensor<32x32xf32, #blocked>, i32 -> tensor<16x16xf32, #blocked>
  // ANCHOR: %[[CONVERTED:.*]] = ttg.convert_layout %[[TILE]] {tle.explicit_encoding.0 = #blocked1} : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
  // ANCHOR: tt.store %{{.*}}, %[[CONVERTED]] {{.*}} : tensor<16x16x!tt.ptr<f32>, #blocked1>
  tt.func @extract_layout_anchor(%src: tensor<32x32xf32, #blocked>, %dst: tensor<16x16x!tt.ptr<f32>, #blocked1>) {
    %idx = arith.constant 3 : i32
    %source = tle.gpu.set_layout %src {target_encoding = #blocked} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32>
    %tile = tle.extract_tile %source[%idx] {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32>, i32 -> tensor<16x16xf32>
    %converted = tle.gpu.set_layout %tile {target_encoding = #blocked1} : tensor<16x16xf32> -> tensor<16x16xf32>
    %ptr = tle.gpu.set_layout %dst {target_encoding = #blocked1} : tensor<16x16x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>>
    tt.store %ptr, %converted : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }

  // An already encoded source is a fixed layout boundary. The conversion
  // must remain explicit instead of retagging the function argument. Store the
  // result so canonicalization cannot remove the layout boundary as dead code.
  // ANCHOR-LABEL: tt.func @existing_source_layout
  // ANCHOR-SAME: (%[[SOURCE:.*]]: tensor<16x16xf32, #blocked>,
  // ANCHOR: %[[SOURCE_CONVERTED:.*]] = ttg.convert_layout %[[SOURCE]] {tle.explicit_encoding.0 = #blocked1} : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
  // ANCHOR: tt.store %{{.*}}, %[[SOURCE_CONVERTED]] {{.*}} : tensor<16x16x!tt.ptr<f32>, #blocked1>
  tt.func @existing_source_layout(%source: tensor<16x16xf32, #blocked>, %dst: tensor<16x16x!tt.ptr<f32>, #blocked1>) {
    %converted = tle.gpu.set_layout %source {target_encoding = #blocked1} : tensor<16x16xf32, #blocked> -> tensor<16x16xf32>
    %ptr = tle.gpu.set_layout %dst {target_encoding = #blocked1} : tensor<16x16x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>>
    tt.store %ptr, %converted : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
