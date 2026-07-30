// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Make sure scatter created when index has mod.
// CHECK: scf.for %arg33 = %c0 to %c512 step %c1 {
// CHECK: %extracted = tensor.extract %collapsed[%arg33] : tensor<512xi32>
// CHECK: %[[V26:.*]] = arith.index_cast %extracted : i32 to index
// CHECK: %extracted_slice = tensor.extract_slice %{{.*}}[%arg33, 0] [1, 128] [1, 1] : tensor<512x128xf32> to tensor<1x128xf32>
// CHECK: %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%[[V26]]], sizes: [1, 128], strides: [1, 1] : memref<*xf32> to memref<1x128xf32, strided<[1, 1], offset: ?>>
// CHECK: bufferization.materialize_in_destination %extracted_slice in writable %reinterpret_cast : (tensor<1x128xf32>, memref<1x128xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK-NEXT: }

module attributes {maia.triton_kernel} {
  tt.func public @scatter_with_mod(%arg0: !tt.ptr<bf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {maia.rank = 4 : i32, tt.divisibility = 16 : i32}, %arg2: f32, %arg3: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {maia.rank = 5 : i32, tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<512x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %c20_i32 = arith.constant 20 : i32
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<512x128xf32>

    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.addptr %arg6, %2 : !tt.ptr<i8>, i32
    %14 = tt.addptr %3, %c20_i32 : !tt.ptr<i8>, i32
    %15 = tt.bitcast %14 : !tt.ptr<i8> -> !tt.ptr<i32>
    %16 = tt.load %15 : !tt.ptr<i32>
    %94 = tt.get_program_id z : i32

    %97 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %99 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>


    %101 = tt.expand_dims %97 {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %102 = arith.remsi %101, %cst : tensor<512x1xi32>
    %103 = arith.divsi %101, %cst : tensor<512x1xi32>
    %104 = tt.splat %16 : i32 -> tensor<512x1xi32>
    %105 = arith.addi %103, %104 : tensor<512x1xi32>

    %115 = tt.expand_dims %99 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %117 = tt.broadcast %115 : tensor<1x128xi32> -> tensor<512x128xi32>

    %131 = arith.muli %0, %arg19 : i32
    %132 = arith.muli %94, %arg21 : i32
    %133 = arith.addi %131, %132 : i32
    %134 = tt.addptr %arg5, %133 : !tt.ptr<f32>, i32
    %135 = tt.splat %arg18 : i32 -> tensor<512x1xi32>
    %136 = arith.muli %105, %135 : tensor<512x1xi32>
    %137 = tt.splat %134 : !tt.ptr<f32> -> tensor<512x1x!tt.ptr<f32>>
    %138 = tt.addptr %137, %136 : tensor<512x1x!tt.ptr<f32>>, tensor<512x1xi32>
    %139 = tt.splat %arg20 : i32 -> tensor<512x1xi32>
    %140 = arith.muli %102, %139 : tensor<512x1xi32>
    %141 = tt.addptr %138, %140 : tensor<512x1x!tt.ptr<f32>>, tensor<512x1xi32>
    %142 = tt.broadcast %141 : tensor<512x1x!tt.ptr<f32>> -> tensor<512x128x!tt.ptr<f32>>
    %143 = tt.addptr %142, %117 : tensor<512x128x!tt.ptr<f32>>, tensor<512x128xi32>
    tt.store %143, %cst_6 : tensor<512x128x!tt.ptr<f32>>
    tt.return
  }
}


