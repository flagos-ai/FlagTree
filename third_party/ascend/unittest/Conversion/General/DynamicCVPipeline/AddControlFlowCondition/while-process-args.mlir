// RUN: triton-opt --process-args %s --allow-unregistered-dialect | FileCheck %s

// Test ProcessArgs adaptation of an scf.while main loop. The iter_arg used by
// scf.condition (index 0) is referenced by three separate blocks, so the pass
// clones its update chain (an addi/muli/addi triple) once per block and grows
// the loop from 1 to 5 iter_args. Each per-block clone is tagged with
// `ssbuffer.while_arg = 0`, and the buffer-offset ops that consume the induction
// arg are rewritten onto the new pipeline arg and tagged with `ssbuffer.arg = 0`.

// CHECK: func.func @pcb12_tc01_while_matmul_fill
// The while op now carries 5 iter_args and 5 results.
// CHECK: %{{.*}}:5 = scf.while (%arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32) : (i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32)
// scf.condition forwards all 5 iter_args.
// CHECK: scf.condition(%{{.*}}) %arg17, %arg18, %arg19, %arg20, %arg21 : i32, i32, i32, i32, i32
// CHECK: ^bb0(%arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32):
// The tail of block 9's per-block clone chain feeds iter_arg %arg18.
// CHECK: arith.addi %{{.*}}, %arg18 {ssbuffer.block_id = 9 : i32, ssbuffer.while_arg = 0 : i32} : i32
// Block 7's per-block clone chain feeds iter_arg %arg19.
// CHECK: arith.addi %{{.*}}, %arg19 {ssbuffer.block_id = 7 : i32, ssbuffer.while_arg = 0 : i32} : i32
// The buffer-offset op is rewritten onto pipeline arg %arg21 and tagged ssbuffer.arg.
// CHECK: arith.muli %arg21, %arg8 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 8 : i32} : i32
// Block 8's per-block clone chain feeds iter_arg %arg20.
// CHECK: arith.addi %{{.*}}, %arg20 {ssbuffer.block_id = 8 : i32, ssbuffer.while_arg = 0 : i32} : i32
// The terminator yields the original update plus the three per-block updates.
// CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32, i32

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
      %0 = scf.while (%arg17 = %c0_i32) : (i32) -> i32 {
        %1 = arith.cmpi slt, %arg17, %arg7 {Undefined, ssbuffer.block_id = 4 : i32} : i32
        scf.condition(%1) %arg17 : i32
      } do {
      ^bb0(%arg17: i32):
        %1 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 9 : i32} : i32
        %2 = arith.addi %c1_i32, %c1_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.clone = 9 : i32} : i32
        %3 = arith.index_cast %arg7 {ssbuffer.block_id = 7 : i32} : i32 to index
        %4 = arith.maxsi %3, %c0 {ssbuffer.block_id = 7 : i32} : index
        %5 = arith.minsi %4, %c32 {ssbuffer.block_id = 7 : i32} : index
        %6 = arith.cmpi slt, %5, %c32 {ssbuffer.block_id = 7 : i32} : index
        %7 = arith.index_cast %arg6 {ssbuffer.block_id = 7 : i32} : i32 to index
        %8 = arith.maxsi %7, %c0 {ssbuffer.block_id = 7 : i32} : index
        %9 = arith.minsi %8, %c64 {ssbuffer.block_id = 7 : i32} : index
        %10 = arith.cmpi slt, %9, %c64 {ssbuffer.block_id = 7 : i32} : index
        %11 = arith.muli %2, %c1_i32 {ssbuffer.block_id = 7 : i32} : i32
        %12 = arith.addi %11, %arg17 {ssbuffer.block_id = 7 : i32} : i32
        %13 = arith.index_cast %arg7 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
        %14 = arith.maxsi %13, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %15 = arith.minsi %14, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %16 = arith.cmpi slt, %15, %c32 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %17 = arith.index_cast %arg6 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : i32 to index
        %18 = arith.maxsi %17, %c0 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %19 = arith.minsi %18, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %20 = arith.cmpi slt, %19, %c64 {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32} : index
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf16>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<64xf16>
        scf.if %20 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_6 : memref<64xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        scf.if %16 {
          linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%alloc_5 : memref<32xf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        %21 = arith.muli %arg17, %arg8 {ssbuffer.block_id = 8 : i32} : i32
        %22 = arith.index_cast %21 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [32], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0] [%15] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0] [%15] [1] {ssbuffer.block_id = 8 : i32} : memref<32xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview, %subview_7 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %23 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32xf16> to tensor<32xf16>
        %24 = arith.muli %arg17, %arg9 {ssbuffer.block_id = 8 : i32} : i32
        %25 = arith.index_cast %24 {ssbuffer.block_id = 8 : i32} : i32 to index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [64], strides: [1] {ssbuffer.block_id = 8 : i32} : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
        %subview_9 = memref.subview %reinterpret_cast_8[0] [%19] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1], offset: ?>>
        %subview_10 = memref.subview %alloc_6[0] [%19] [1] {ssbuffer.block_id = 8 : i32} : memref<64xf16> to memref<?xf16, strided<[1]>>
        memref.copy %subview_9, %subview_10 {ssbuffer.block_id = 8 : i32} : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16, strided<[1]>>
        %26 = bufferization.to_tensor %alloc_6 restrict writable {ssbuffer.block_id = 8 : i32} : memref<64xf16> to tensor<64xf16>
        %expanded = tensor.expand_shape %23 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 8 : i32} : tensor<32xf16> into tensor<32x1xf16>
        %27 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x16xf16>
        %28 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%27 : tensor<32x16xf16>) -> tensor<32x16xf16>
        %inserted_slice = tensor.insert_slice %expanded into %28[0, 0] [32, 1] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<32x1xf16> into tensor<32x16xf16>
        %expanded_11 = tensor.expand_shape %26 [[0, 1]] output_shape [1, 64] {ssbuffer.block_id = 8 : i32} : tensor<64xf16> into tensor<1x64xf16>
        %29 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<16x64xf16>
        %30 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst : f16) outs(%29 : tensor<16x64xf16>) -> tensor<16x64xf16>
        %inserted_slice_12 = tensor.insert_slice %expanded_11 into %30[0, 0] [1, 64] [1, 1] {ssbuffer.block_id = 8 : i32} : tensor<1x64xf16> into tensor<16x64xf16>
        %reshape = tensor.reshape %inserted_slice(%cst_3) {ssbuffer.block_id = 8 : i32} : (tensor<32x16xf16>, tensor<3xi64>) -> tensor<32x1x16xf16>
        %31 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<1x32x16xf16>
        %transposed = linalg.transpose ins(%reshape : tensor<32x1x16xf16>) outs(%31 : tensor<1x32x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_13 = tensor.reshape %transposed(%cst_2) {ssbuffer.block_id = 8 : i32} : (tensor<1x32x16xf16>, tensor<4xi64>) -> tensor<1x2x16x16xf16>
        %reshape_14 = tensor.reshape %inserted_slice_12(%cst_1) {ssbuffer.block_id = 8 : i32} : (tensor<16x64xf16>, tensor<3xi64>) -> tensor<16x4x16xf16>
        %32 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<4x16x16xf16>
        %transposed_15 = linalg.transpose ins(%reshape_14 : tensor<16x4x16xf16>) outs(%32 : tensor<4x16x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 8 : i32}
        %reshape_16 = tensor.reshape %transposed_15(%cst_0) {ssbuffer.block_id = 8 : i32} : (tensor<4x16x16xf16>, tensor<4xi64>) -> tensor<4x1x16x16xf16>
        hivm.hir.copy ins(%reshape_13 : tensor<1x2x16x16xf16>) outs(%alloc : memref<1x2x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.copy ins(%reshape_16 : tensor<4x1x16x16xf16>) outs(%alloc_4 : memref<4x1x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        scf.yield %12 : i32
      } attributes {Undefined, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
