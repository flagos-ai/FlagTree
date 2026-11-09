// RUN: triton-opt --process-args %s --allow-unregistered-dialect | FileCheck %s

// Test that ProcessArgs correctly handles shared iter_args by cloning computation
// chains and updating operands to new values.

// CHECK: func.func @test_iter_args_update
module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  func.func @test_iter_args_update(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32, %arg22: i32, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i32, %arg35: i32, %arg36: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst_1 = arith.constant {ssbuffer.block_id = 34 : i32} 0.000000e+00 : f32
    %c32_i32 = arith.constant {MixUse, ssbuffer.block_id = 34 : i32} 32 : i32
    %c0_i32 = arith.constant {MixUse, ssbuffer.block_id = 34 : i32} 0 : i32
    %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 34 : i32} 128 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 34 : i32} 8 : i32
    %c32 = arith.constant {DataUse, ssbuffer.block_id = 34 : i32} 32 : index
    %c0 = arith.constant {ssbuffer.block_id = 34 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 34 : i32} 1 : index
    %0 = tensor.empty() {ssbuffer.block_id = 19 : i32} : tensor<32x32xf32>
    %1 = arith.divsi %arg35, %c8_i32 {ssbuffer.block_id = 19 : i32} : i32
    %2 = arith.remsi %arg35, %c8_i32 {ssbuffer.block_id = 19 : i32} : i32
    %3 = arith.index_cast %1 {ssbuffer.block_id = 19 : i32} : i32 to index
    %4 = arith.muli %2, %arg14 {ssbuffer.block_id = 17 : i32} : i32
    %5 = arith.index_cast %4 {ssbuffer.block_id = 17 : i32} : i32 to index
    %6 = arith.index_cast %arg16 {ssbuffer.block_id = 17 : i32} : i32 to index
    %7 = arith.muli %5, %6 {ssbuffer.block_id = 17 : i32} : index
    %8 = arith.muli %arg14, %c32_i32 {ssbuffer.block_id = 17 : i32} : i32
    %9 = arith.muli %arg30, %c32_i32 {ssbuffer.block_id = 17 : i32} : i32
    %10 = arith.index_cast %arg14 {ssbuffer.block_id = 17 : i32} : i32 to index
    %11 = arith.index_cast %arg30 {ssbuffer.block_id = 17 : i32} : i32 to index
    %12 = arith.index_cast %8 {ssbuffer.block_id = 17 : i32} : i32 to index
    %13 = arith.index_cast %9 {ssbuffer.block_id = 17 : i32} : i32 to index
    %14 = arith.minsi %8, %c128_i32 {ssbuffer.block_id = 17 : i32} : i32
    %15 = tensor.empty() {ssbuffer.block_id = 34 : i32} : tensor<32x32xf32>
    %16 = linalg.fill {ssbuffer.block_id = 34 : i32} ins(%cst_1 : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %alloc = memref.alloc() {ssbuffer.block_id = 18 : i32} : memref<32x32xf32>
    %17:2 = scf.for %arg37 = %c0_i32 to %14 step %c32_i32 iter_args(%arg38 = %c0, %arg39 = %c0) -> (index, index)  : i32 {
      %18 = arith.index_cast %arg37 {ssbuffer.block_id = 4 : i32} : i32 to index
      %19 = arith.addi %18, %c32 {ssbuffer.block_id = 4 : i32} : index
      %20 = arith.maxsi %18, %12 {ssbuffer.block_id = 4 : i32} : index
      %21 = arith.minsi %19, %20 {ssbuffer.block_id = 4 : i32} : index
      %22 = arith.subi %21, %18 {ssbuffer.block_id = 4 : i32} : index
      %23 = arith.minsi %22, %c32 {ssbuffer.block_id = 4 : i32} : index
      %24 = arith.cmpi slt, %23, %c32 {ssbuffer.block_id = 4 : i32} : index
      %alloc_1 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<32x32xf32>
      scf.if %24 {
        linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_1 : f32) outs(%alloc_1 : memref<32x32xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
      %25 = arith.addi %3, %arg38 {ssbuffer.block_id = 4 : i32} : index
      %26 = bufferization.to_tensor %alloc_1 restrict writable {ssbuffer.block_id = 4 : i32} : memref<32x32xf32> to tensor<32x32xf32>
      %27 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32} ins(%26, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %28 = arith.index_cast %8 {ssbuffer.block_id = 4 : i32} : i32 to index
      %29 = arith.addi %arg38, %28 {ssbuffer.block_id = 4 : i32} : index
      hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32} ins(%27 : tensor<32x32xf32>) outs(%alloc : memref<32x32xf32>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
      %30 = arith.index_cast %arg37 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : i32 to index
      %31 = arith.addi %30, %c32 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %32 = arith.maxsi %30, %12 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %33 = arith.minsi %31, %32 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %34 = arith.subi %33, %30 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %35 = arith.minsi %34, %c32 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %36 = arith.cmpi slt, %35, %c32 {ssbuffer.block_id = 5 : i32, ssbuffer.clone = 4 : i32} : index
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x32xf32>
      scf.if %36 {
        linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : f32) outs(%alloc_2 : memref<32x32xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
      %37 = arith.addi %7, %arg39 {ssbuffer.block_id = 5 : i32} : index
      %transposed_2 = linalg.transpose ins(%15 : tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
      %38 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<32x32xf32>
      %39 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : f32) outs(%38 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %40 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32} ins(%transposed_2, %15 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%39 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %41 = arith.index_cast %9 {ssbuffer.block_id = 5 : i32} : i32 to index
      %42 = arith.addi %arg39, %41 {ssbuffer.block_id = 5 : i32} : index
      %43 = arith.addi %3, %arg38 {ssbuffer.block_id = 10 : i32} : index
      %44 = arith.addi %7, %arg39 {ssbuffer.block_id = 11 : i32} : index
      // CHECK: %[[RES0:.*]] = arith.addi %3, %arg40 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 10 : i32} : index
      // CHECK: %[[RES1:.*]] = arith.index_cast %8 {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 10 : i32} : i32 to index
      // CHECK: %[[RES2:.*]] = arith.addi %arg40, %[[RES1]] {ssbuffer.arg = 0 : i32, ssbuffer.block_id = 10 : i32} : index
      // CHECK: %[[RES3:.*]] = arith.addi %7, %arg41 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 11 : i32} : index
      // CHECK: %[[RES4:.*]] = arith.index_cast %9 {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 11 : i32} : i32 to index
      // CHECK: %[[RES5:.*]] = arith.addi %arg41, %[[RES4]] {ssbuffer.arg = 1 : i32, ssbuffer.block_id = 11 : i32} : index
      scf.yield %29, %42 : index, index
    } {DataUse, ssbuffer.block_id = 36 : i32, ssbuffer.main_loop = 0 : i32}
    return
  }
}
