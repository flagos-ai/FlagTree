// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// Test CloneOps handling of an scf.while main loop with a cross-block update
// chain (block 9 owns the addition; block 7 builds on it via muli/addi and
// also adds extra vector-only index ops) and the vector-only index chain in
// block 7 that feeds the cube block (block 8). The pass clones the cross-block
// source into block 7 (`clone = 9`) and clones the vector index ops into block
// 8 (`clone = 7`).

// CHECK: func.func @pcb12_tc01_while_matmul_fill
// CHECK: scf.while
// The cross-block update chain source in block 9 stays in place.
// CHECK: arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
// It is cloned into block 7 carrying ssbuffer.clone = 9, so block 7 can
// build its own muli/addi chain without crossing blocks.
// CHECK: arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32} : i32
// Block 7's vector-only index ops land in block 8 with clone = 7.
// CHECK: arith.index_cast %arg7 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
// CHECK: arith.maxsi {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
// CHECK: arith.minsi {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
// CHECK: arith.cmpi slt, {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
// CHECK: arith.index_cast %arg6 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
// The terminator yields the chain's top result for the loop's single iter_arg.
// CHECK: scf.yield %{{.*}} : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pcb12_tc01_while_matmul_fill(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 11 : i32} 0.000000e+00 : f16
    %c1_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 1 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : i32
    %c0 = arith.constant {ssbuffer.block_id = 10 : i32} 0 : index
    %c32 = arith.constant {ssbuffer.block_id = 10 : i32} 32 : index
    %c64 = arith.constant {ssbuffer.block_id = 10 : i32} 64 : index
    %cst_0 = arith.constant {ssbuffer.block_id = 8 : i32} dense<[4, 1, 16, 16]> : tensor<4xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 8 : i32} dense<[16, 4, 16]> : tensor<3xi64>
    %cst_2 = arith.constant {ssbuffer.block_id = 8 : i32} dense<[1, 2, 16, 16]> : tensor<4xi64>
    %cst_3 = arith.constant {ssbuffer.block_id = 8 : i32} dense<[32, 1, 16]> : tensor<3xi64>
    scope.scope : () -> () {
      %alloc_16 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_17 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>

      %0 = scf.while (%arg17 = %c0_i32) : (i32) -> i32 {
        %1 = arith.cmpi slt, %arg17, %arg7 {Undefined, ssbuffer.block_id = 4 : i32} : i32
        scf.condition(%1) %arg17 : i32
      } do {
      ^bb0(%arg17: i32):
        %1 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
        %2 = arith.index_cast %arg7 {ssbuffer.block_id = 7 : i32} : i32 to index
        %3 = arith.maxsi %2, %c0 {ssbuffer.block_id = 7 : i32} : index
        %4 = arith.minsi %3, %c32 {ssbuffer.block_id = 7 : i32} : index
        %5 = arith.cmpi slt, %4, %c32 {ssbuffer.block_id = 7 : i32} : index
        %6 = arith.index_cast %arg6 {ssbuffer.block_id = 7 : i32} : i32 to index
        %7 = arith.maxsi %6, %c0 {ssbuffer.block_id = 7 : i32} : index
        %8 = arith.minsi %7, %c64 {ssbuffer.block_id = 7 : i32} : index
        %9 = arith.cmpi slt, %8, %c64 {ssbuffer.block_id = 7 : i32} : index

        %mj_1 = arith.muli %1, %c1_i32 {ssbuffer.block_id = 7 : i32} : i32
        %mj_2 = arith.addi %mj_1, %arg17 {ssbuffer.block_id = 7 : i32} : i32

        %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf16>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf16>
        scf.if %9 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_5 : memref<64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        scf.if %5 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc : memref<32xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        %10 = arith.muli %arg17, %arg8 {ssbuffer.block_id = 8 : i32} : i32
        %11 = arith.index_cast %10 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0] [%4] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_6 = memref.subview %alloc[0] [%4] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview, %subview_6 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %12 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 8 : i32} : memref<32xf16> to tensor<32xf16>
        %13 = arith.muli %arg17, %arg9 {ssbuffer.block_id = 8 : i32} : i32
        %14 = arith.index_cast %13 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%14], sizes: [64], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
        %subview_8 = memref.subview %reinterpret_cast_7[0] [%8] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_9 = memref.subview %alloc_5[0] [%8] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview_8, %subview_9 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %15 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 8 : i32} : memref<64xf16> to tensor<64xf16>
        %expanded = tensor.expand_shape %12 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 8 : i32} : tensor<32xf16> into tensor<32x1xf16>
        %16 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x16xf16>
        %17 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%16 : tensor<32x16xf16>) -> tensor<32x16xf16>
        %inserted_slice = tensor.insert_slice %expanded into %17[0, 0] [32, 1] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<32x1xf16> into tensor<32x16xf16>
        %expanded_10 = tensor.expand_shape %15 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 8 : i32} : tensor<64xf16> into tensor<1x64xf16>
        %18 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<16x64xf16>
        %19 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%18 : tensor<16x64xf16>) -> tensor<16x64xf16>
        %inserted_slice_11 = tensor.insert_slice %expanded_10 into %19[0, 0] [1, 64] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<1x64xf16> into tensor<16x64xf16>
        %reshape = tensor.reshape %inserted_slice(%cst_3) {ssbuffer.block_id = 8 : i32} : (tensor<32x16xf16>, tensor<3xi64>) -> tensor<32x1x16xf16>
        %20 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<1x32x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<32x1x16xf16>) outs(%20 : tensor<1x32x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_12 = tensor.reshape %transposed(%cst_2) {ssbuffer.block_id = 8 : i32} : (tensor<1x32x16xf16>, tensor<4xi64>) -> tensor<1x2x16x16xf16>
        %reshape_13 = tensor.reshape %inserted_slice_11(%cst_1) {ssbuffer.block_id = 8 : i32} : (tensor<16x64xf16>, tensor<3xi64>) -> tensor<16x4x16xf16>
        %21 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<4x16x16xf16>
        %transposed_14 = linalg.transpose ins(%reshape_13 : tensor<16x4x16xf16>) outs(%21 : tensor<4x16x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_15 = tensor.reshape %transposed_14(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<4x16x16xf16>, tensor<4xi64>) -> tensor<4x1x16x16xf16>
        hivm.hir.copy ins(%reshape_12 : tensor<1x2x16x16xf16>) outs(%alloc_16 : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.copy ins(%reshape_15 : tensor<4x1x16x16xf16>) outs(%alloc_17 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        scf.yield %mj_2 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
