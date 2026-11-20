// RUN: triton-opt --update-loop-ops %s --allow-unregistered-dialect | FileCheck %s

// Test UpdateLoopOps on an scf.while main loop after CreateIfOps. The loop's do
// region already holds the per-block scf.if wrappers with their while_arg
// update chains. The pass emits a prologue sync_block_set, grows the loop from
// 5 to 8 iter_args (adding 3 pipeline sync args), and brackets the body with a
// sync_block_wait / sync_block_set pair while preserving the guarded ifs.

// CHECK: func.func @pcb12_tc01_while_matmul_fill
// A prologue sync_block_set is emitted before the loop.
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// The while op grows to 8 iter_args (5 from ProcessArgs + 3 sync args).
// CHECK: %{{.*}}:8 = scf.while (%arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32, %arg22 = %c0_i32_5, %arg23 = %c0_i32_6, %arg24 = %c0_i32_7) : (i32, i32, i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32, i32, i32)
// CHECK: scf.condition(%{{.*}}) %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24
// CHECK: ^bb0(%arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32, %arg23: i32, %arg24: i32):
// The body opens with a sync_block_wait matching the prologue set.
// CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// Per-block while_arg update chains survive inside the guarded ifs.
// CHECK: arith.addi %{{.*}}, %arg19 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: arith.addi %{{.*}}, %arg20 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: arith.addi %{{.*}}, %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
// A trailing sync_block_set closes the body before the terminator.
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
// CHECK: scf.yield %{{.*}}#0, %{{.*}}#0, %{{.*}}, %{{.*}}#1, %{{.*}}#1, %arg22, %arg23, %arg24 : i32, i32, i32, i32, i32, i32, i32, i32

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
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32} : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>
      %0:5 = scf.while (%arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32) {
        %1 = arith.cmpi slt, %arg17, %arg7 {Undefined, ssbuffer.block_id = 4 : i32} : i32
        scf.condition(%1) %arg17, %arg18, %arg19, %arg20, %arg21 : i32, i32, i32, i32, i32
      } do {
      ^bb0(%arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32):
        %true = arith.constant true
        %1 = scf.if %true -> (i32) {
          %4 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
          %5 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %6 = arith.muli %5, %c1_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %7 = arith.addi %6, %arg19 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
          scf.yield %7 : i32
        } else {
          scf.yield %arg19 : i32
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 9 : i32}
        %true_5 = arith.constant true
        %2:2 = scf.if %true_5 -> (i32, i32) {
          %4 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32} : i32
          %5 = arith.index_cast %arg7 {ssbuffer.block_id = 7 : i32} : i32 to index
          %6 = arith.maxsi %5, %c0 {ssbuffer.block_id = 7 : i32} : index
          %7 = arith.minsi %6, %c32 {ssbuffer.block_id = 7 : i32} : index
          %8 = arith.cmpi slt, %7, %c32 {ssbuffer.block_id = 7 : i32} : index
          %9 = arith.index_cast %arg6 {ssbuffer.block_id = 7 : i32} : i32 to index
          %10 = arith.maxsi %9, %c0 {ssbuffer.block_id = 7 : i32} : index
          %11 = arith.minsi %10, %c64 {ssbuffer.block_id = 7 : i32} : index
          %12 = arith.cmpi slt, %11, %c64 {ssbuffer.block_id = 7 : i32} : index
          %13 = arith.muli %4, %c1_i32 {ssbuffer.block_id = 7 : i32} : i32
          %14 = arith.addi %13, %arg17 {ssbuffer.block_id = 7 : i32} : i32
          %15 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %16 = arith.muli %15, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %17 = arith.addi %16, %arg20 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
          scf.yield %14, %17 : i32, i32
        } else {
          scf.yield %arg17, %arg20 : i32, i32
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 7 : i32}
        %true_6 = arith.constant true
        %3:2 = scf.if %true_6 -> (i32, i32) {
          %4 = arith.index_cast %arg7 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
          %5 = arith.maxsi %4, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %6 = arith.minsi %5, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %7 = arith.cmpi slt, %6, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %8 = arith.index_cast %arg6 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
          %9 = arith.maxsi %8, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %10 = arith.minsi %9, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %11 = arith.cmpi slt, %10, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
          %alloc_7 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf16>
          %alloc_8 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf16>
          scf.if %11 {
            linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_8 : memref<64xf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
          scf.if %7 {
            linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_7 : memref<32xf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
          %12 = arith.muli %arg18, %arg8 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
          %13 = arith.index_cast %12 {ssbuffer.block_id = 8 : i32} : i32 to index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [32], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
          %subview = memref.subview %reinterpret_cast[0] [%6] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
          %subview_9 = memref.subview %alloc_7[0] [%6] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16> to memref<?xf16, strided<[1]>>
          memref.copy %subview, %subview_9 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
          %14 = bufferization.to_tensor %alloc_7 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32xf16> to tensor<32xf16>
          %15 = arith.muli %arg18, %arg9 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
          %16 = arith.index_cast %15 {ssbuffer.block_id = 8 : i32} : i32 to index
          %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [64], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
          %subview_11 = memref.subview %reinterpret_cast_10[0] [%10] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
          %subview_12 = memref.subview %alloc_8[0] [%10] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16> to memref<?xf16, strided<[1]>>
          memref.copy %subview_11, %subview_12 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
          %17 = bufferization.to_tensor %alloc_8 restrict writable {ssbuffer.block_id = 8 : i32} : memref<64xf16> to tensor<64xf16>
          %expanded = tensor.expand_shape %14 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 8 : i32} : tensor<32xf16> into tensor<32x1xf16>
          %18 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x16xf16>
          %19 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%18 : tensor<32x16xf16>) -> tensor<32x16xf16>
          %inserted_slice = tensor.insert_slice %expanded into %19[0, 0] [32, 1] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<32x1xf16> into tensor<32x16xf16>
          %expanded_13 = tensor.expand_shape %17 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 8 : i32} : tensor<64xf16> into tensor<1x64xf16>
          %20 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<16x64xf16>
          %21 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%20 : tensor<16x64xf16>) -> tensor<16x64xf16>
          %inserted_slice_14 = tensor.insert_slice %expanded_13 into %21[0, 0] [1, 64] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<1x64xf16> into tensor<16x64xf16>
          %reshape = tensor.reshape %inserted_slice(%cst_3) {ssbuffer.block_id = 8 : i32} : (tensor<32x16xf16>, tensor<3xi64>) -> tensor<32x1x16xf16>
          %22 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<1x32x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<32x1x16xf16>) outs(%22 : tensor<1x32x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
          %reshape_15 = tensor.reshape %transposed(%cst_2) {ssbuffer.block_id = 8 : i32} : (tensor<1x32x16xf16>, tensor<4xi64>) -> tensor<1x2x16x16xf16>
          %reshape_16 = tensor.reshape %inserted_slice_14(%cst_1) {ssbuffer.block_id = 8 : i32} : (tensor<16x64xf16>, tensor<3xi64>) -> tensor<16x4x16xf16>
          %23 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<4x16x16xf16>
          %transposed_17 = linalg.transpose ins(%reshape_16 : tensor<16x4x16xf16>) outs(%23 : tensor<4x16x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
          %reshape_18 = tensor.reshape %transposed_17(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<4x16x16xf16>, tensor<4xi64>) -> tensor<4x1x16x16xf16>
          hivm.hir.copy ins(%reshape_15 : tensor<1x2x16x16xf16>) outs(%alloc : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.copy ins(%reshape_18 : tensor<4x1x16x16xf16>) outs(%alloc_4 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}
          %24 = arith.addi %c1_i32, %c1_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32, ssbuffer.clone = 9 : i32} : i32
          %25 = arith.muli %24, %c1_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
          %26 = arith.addi %25, %arg18 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
          %27 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %28 = arith.muli %27, %c1_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
          %29 = arith.addi %28, %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
          hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          scf.yield %26, %29 : i32, i32
        } else {
          scf.yield %arg18, %arg21 : i32, i32
        } {hivm.matmul_limited_in_cube, ssbuffer.if = 8 : i32}
        scf.yield %2#0, %3#0, %1, %2#1, %3#1 : i32, i32, i32, i32, i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
