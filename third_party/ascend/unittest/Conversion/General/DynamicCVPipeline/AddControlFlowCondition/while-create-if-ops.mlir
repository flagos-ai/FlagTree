// RUN: triton-opt --create-if-ops %s --allow-unregistered-dialect | FileCheck %s

// Test CreateIfOps on an scf.while main loop after ProcessArgs. The loop
// already carries 5 iter_args. Each block in the do region is wrapped in an
// scf.if guarded by a boolean, and the per-block while_arg update chains (an
// addi/muli/addi triple tagged `ssbuffer.while_arg = 0`) are threaded through
// the if results so they still reach scf.yield.

// CHECK: func.func @pcb12_tc01_while_matmul_fill
// CHECK: scf.while
// CHECK: ^bb0(%arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32):
// Block 9 wrapped into an scf.if returning its single while_arg update.
// CHECK: scf.if %{{.*}} -> (i32) {
// CHECK: arith.muli %{{.*}}, %c1_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: arith.addi %{{.*}}, %arg19 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: } {hivm.matmul_limited_in_cube, ssbuffer.if = 9 : i32}
// Block 7 wrapped into an scf.if returning its condition update and while_arg.
// CHECK: scf.if %{{.*}} -> (i32, i32) {
// CHECK: arith.addi %{{.*}}, %arg20 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: } {hivm.matmul_limited_in_cube, ssbuffer.if = 7 : i32}
// Block 8 wrapped into an scf.if returning its arg update and while_arg.
// CHECK: scf.if %{{.*}} -> (i32, i32) {
// CHECK: arith.addi %{{.*}}, %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
// CHECK: } {hivm.matmul_limited_in_cube, ssbuffer.if = 8 : i32}
// The terminator wires the if results back into the 5 yielded values.
// CHECK: scf.yield %{{.*}}#0, %{{.*}}#0, %{{.*}}, %{{.*}}#1, %{{.*}}#1 : i32, i32, i32, i32, i32

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
        %1 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
        %2 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %3 = arith.muli %2, %c1_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %4 = arith.addi %3, %arg19 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %5 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32} : i32
        %6 = arith.index_cast %arg7 {ssbuffer.block_id = 7 : i32} : i32 to index
        %7 = arith.maxsi %6, %c0 {ssbuffer.block_id = 7 : i32} : index
        %8 = arith.minsi %7, %c32 {ssbuffer.block_id = 7 : i32} : index
        %9 = arith.cmpi slt, %8, %c32 {ssbuffer.block_id = 7 : i32} : index
        %10 = arith.index_cast %arg6 {ssbuffer.block_id = 7 : i32} : i32 to index
        %11 = arith.maxsi %10, %c0 {ssbuffer.block_id = 7 : i32} : index
        %12 = arith.minsi %11, %c64 {ssbuffer.block_id = 7 : i32} : index
        %13 = arith.cmpi slt, %12, %c64 {ssbuffer.block_id = 7 : i32} : index
        %14 = arith.muli %5, %c1_i32 {ssbuffer.block_id = 7 : i32} : i32
        %15 = arith.addi %14, %arg17 {ssbuffer.block_id = 7 : i32} : i32
        %16 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %17 = arith.muli %16, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %18 = arith.addi %17, %arg20 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %19 = arith.index_cast %arg7 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
        %20 = arith.maxsi %19, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %21 = arith.minsi %20, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %22 = arith.cmpi slt, %21, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %23 = arith.index_cast %arg6 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
        %24 = arith.maxsi %23, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %25 = arith.minsi %24, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %26 = arith.cmpi slt, %25, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf16>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf16>
        scf.if %26 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_6 : memref<64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        scf.if %22 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_5 : memref<32xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        %27 = arith.muli %arg18, %arg8 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
        %28 = arith.index_cast %27 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%28], sizes: [32], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0] [%21] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0] [%21] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview, %subview_7 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %29 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32xf16> to tensor<32xf16>
        %30 = arith.muli %arg18, %arg9 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
        %31 = arith.index_cast %30 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [64], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%25] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %alloc_6[0] [%25] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview_9, %subview_10 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %32 = bufferization.to_tensor %alloc_6 restrict writable {ssbuffer.block_id = 8 : i32} : memref<64xf16> to tensor<64xf16>
        %expanded = tensor.expand_shape %29 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 8 : i32} : tensor<32xf16> into tensor<32x1xf16>
        %33 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x16xf16>
        %34 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%33 : tensor<32x16xf16>) -> tensor<32x16xf16>
        %inserted_slice = tensor.insert_slice %expanded into %34[0, 0] [32, 1] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<32x1xf16> into tensor<32x16xf16>
        %expanded_11 = tensor.expand_shape %32 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 8 : i32} : tensor<64xf16> into tensor<1x64xf16>
        %35 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<16x64xf16>
        %36 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%35 : tensor<16x64xf16>) -> tensor<16x64xf16>
        %inserted_slice_12 = tensor.insert_slice %expanded_11 into %36[0, 0] [1, 64] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<1x64xf16> into tensor<16x64xf16>
        %reshape = tensor.reshape %inserted_slice(%cst_3) {ssbuffer.block_id = 8 : i32} : (tensor<32x16xf16>, tensor<3xi64>) -> tensor<32x1x16xf16>
        %37 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<1x32x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<32x1x16xf16>) outs(%37 : tensor<1x32x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_13 = tensor.reshape %transposed(%cst_2) {ssbuffer.block_id = 8 : i32} : (tensor<1x32x16xf16>, tensor<4xi64>) -> tensor<1x2x16x16xf16>
        %reshape_14 = tensor.reshape %inserted_slice_12(%cst_1) {ssbuffer.block_id = 8 : i32} : (tensor<16x64xf16>, tensor<3xi64>) -> tensor<16x4x16xf16>
        %38 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<4x16x16xf16>
        %transposed_15 = linalg.transpose ins(%reshape_14 : tensor<16x4x16xf16>) outs(%38 : tensor<4x16x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_16 = tensor.reshape %transposed_15(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<4x16x16xf16>, tensor<4xi64>) -> tensor<4x1x16x16xf16>
        hivm.hir.copy ins(%reshape_13 : tensor<1x2x16x16xf16>) outs(%alloc : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.copy ins(%reshape_16 : tensor<4x1x16x16xf16>) outs(%alloc_4 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}
        %39 = arith.addi %c1_i32, %c1_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32, ssbuffer.clone = 9 : i32} : i32
        %40 = arith.muli %39, %c1_i32 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
        %41 = arith.addi %40, %arg18 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
        %42 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %43 = arith.muli %42, %c1_i32 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
        %44 = arith.addi %43, %arg21 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        scf.yield %15, %41, %4, %18, %44 : i32, i32, i32, i32, i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
