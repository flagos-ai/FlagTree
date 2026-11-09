// RUN: triton-opt --ub-usage-opt %s | FileCheck %s

// Test UB usage optimization: dividing of V1 and V2 should change to one small ub dependency.
// math.exp<64x64> -> || ->linalg.fill<64> -> linalg.reduce<64> -> arith.subf<64> -> xxx
// math.exp<64x64> -> linalg.fill<64> -> linalg.reduce<64> -> arith.subf<64> -> || -> xxx

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // CHECK-LABEL: func.func @_attn_fwd
  func.func @_attn_fwd(%arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %out_offset_i64: i64) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %c64 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} 64 : index
    %cst = arith.constant {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %cst_f16 = arith.constant {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f16
    %c0_i32_4 = arith.constant {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %0 = tensor.empty() {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %1 = linalg.fill {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %f16_tensor_empty = tensor.empty() {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
    %f16_tensor = linalg.fill {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f16 : f16) outs(%f16_tensor_empty : tensor<64x64xf16>) -> tensor<64x64xf16>
    %4 = tensor.empty() {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %5 = linalg.fill {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>

    %40:3 = scf.for %arg17 = %c0_i32_4 to %c0_i32_4 step %c0_i32_4 iter_args(%arg18 = %5, %arg19 = %1, %arg20 = %5) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>) : i32 {
      %69 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%f16_tensor, %f16_tensor : tensor<64x64xf16>, tensor<64x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>

      %71 = arith.mulf %69, %1 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %reduced = linalg.reduce ins(%71 : tensor<64x64xf32>) outs(%5 : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
        (%in: f32, %init: f32) {
          %93 = arith.maximumf %in, %init : f32
          linalg.yield %93 : f32
        }
      %72 = arith.maximumf %arg20, %reduced {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %broadcasted_18 = linalg.broadcast ins(%72 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
      %73 = arith.subf %71, %broadcasted_18 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %74 = math.exp %73 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %75 = arith.truncf %74 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
      %83 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%75, %f16_tensor : tensor<64x64xf16>, tensor<64x64xf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>

      // CHECK: %{{.*}} = linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64xf32>) -> tensor<64xf32>
      %84 = linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%4 : tensor<64xf32>) -> tensor<64xf32>
      // CHECK: %{{.*}} = linalg.reduce ins(%{{.*}} : tensor<64x64xf32>) outs(%{{.*}} : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"}
      %reduced_22 = linalg.reduce ins(%74 : tensor<64x64xf32>) outs(%84 : tensor<64xf32>) dimensions = [1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
        (%in: f32, %init: f32) {
          %93 = arith.addf %in, %init : f32
          linalg.yield %93 : f32
        }
      // CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %85 = arith.subf %arg20, %72 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %86 = math.exp %85 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %87 = arith.mulf %arg18, %86 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %88 = arith.addf %87, %reduced_22 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      %broadcasted_23 = linalg.broadcast ins(%86 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"}
      %89 = arith.mulf %arg19, %broadcasted_23 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      %90 = arith.addf %83, %89 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>

      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR"} %88, %90, %72 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR, VECTOR, VECTOR, VECTOR", tt.divisibility_arg1 = dense<64> : tensor<1xi32>}

    %out_offset = arith.index_cast %out_offset_i64 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : i64 to index
    %reinterpret_cast_13 = memref.reinterpret_cast %arg7 to offset: [%out_offset], sizes: [64, 64], strides: [64, 1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %broadcasted = linalg.broadcast ins(%40#0 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
    %55 = arith.divf %40#1, %broadcasted {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %60 = arith.truncf %55 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    bufferization.materialize_in_destination %60 in writable %reinterpret_cast_13 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()

    return
  }

  // Test 2: Index type dependency - block ID should NOT be changed
  // CHECK-LABEL: func.func @test_index_dependency
  func.func @test_index_dependency(%arg0: memref<?xf32> {tt.divisibility = 16 : i32}, %offset: i64) {
    %cst = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c0 = arith.constant {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32

    %empty_1d = tensor.empty() {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %init_1d = linalg.fill {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%empty_1d : tensor<64xf32>) -> tensor<64xf32>

    %result = scf.for %iv = %c0 to %c0 step %c0 iter_args(%arg1 = %init_1d) -> (tensor<64xf32>) : i32 {
      %c1 = arith.constant {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
      %c2 = arith.constant {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
      %idx = arith.index_cast %c1 {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index

      // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %idx2 = arith.addi %c1, %c2 {ssbuffer.block_id = 22 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %t1 = arith.mulf %arg1, %arg1 {ssbuffer.block_id = 24 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>

      scf.yield %t1 : tensor<64xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR"}

    %out_offset = arith.index_cast %offset {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : i64 to index
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%out_offset], sizes: [64], strides: [1] {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %result in writable %reinterpret {ssbuffer.block_id = 25 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64xf32>, memref<64xf32, strided<[1], offset: ?>>) -> ()

    return
  }
}
