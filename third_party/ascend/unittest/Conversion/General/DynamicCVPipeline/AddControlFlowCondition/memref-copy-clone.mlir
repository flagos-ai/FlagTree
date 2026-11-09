// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// CHECK: func.func @test_memref_copy_clone
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_memref_copy_clone(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} dense<[2, 2, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 13 : i32} dense<[32, 2, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 12 : i32} 0.000000e+00 : f32
    %c64_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 64 : i32
    %c2_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 2 : i32
    %c4_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 4 : i32
    %c32_i32 = arith.constant {MixUse, ssbuffer.block_id = 12 : i32} 32 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 8 : i32
    %c31_i32 = arith.constant {ssbuffer.block_id = 12 : i32} 31 : i32
    %c4096_i64 = arith.constant {ssbuffer.block_id = 12 : i32} 4096 : i64
    %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 0 : i32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32} 1 : i32
    %c256 = arith.constant {ssbuffer.block_id = 12 : i32} 256 : index
    %c64 = arith.constant {ssbuffer.block_id = 12 : i32} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : index
    %c32 = arith.constant {ssbuffer.block_id = 12 : i32} 32 : index
    %c8 = arith.constant {ssbuffer.block_id = 12 : i32} 8 : index
    %c512 = arith.constant {ssbuffer.block_id = 12 : i32} 512 : index
    scope.scope : () -> () {
      %0 = arith.divsi %arg15, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
      %1 = arith.remsi %arg15, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
      %2 = arith.addi %arg9, %c31_i32 {ssbuffer.block_id = 8 : i32} : i32
      %3 = arith.divsi %2, %c32_i32 {ssbuffer.block_id = 8 : i32} : i32
      %4 = arith.muli %0, %3 {ssbuffer.block_id = 8 : i32} : i32
      %5 = arith.addi %4, %arg14 {ssbuffer.block_id = 8 : i32} : i32
      %6 = arith.muli %0, %arg9 {ssbuffer.block_id = 8 : i32} : i32
      %7 = arith.muli %6, %c4_i32 {ssbuffer.block_id = 8 : i32} : i32
      %8 = arith.divsi %1, %c2_i32 {ssbuffer.block_id = 8 : i32} : i32
      %9 = arith.addi %7, %8 {ssbuffer.block_id = 8 : i32} : i32
      %10 = arith.muli %9, %c64_i32 {ssbuffer.block_id = 8 : i32} : i32
      %11 = arith.index_cast %10 {ssbuffer.block_id = 8 : i32} : i32 to index
      %12 = arith.muli %5, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
      %13 = arith.addi %12, %1 {ssbuffer.block_id = 8 : i32} : i32
      %14 = arith.extsi %13 {ssbuffer.block_id = 8 : i32} : i32 to i64
      %15 = arith.muli %14, %c4096_i64 {ssbuffer.block_id = 8 : i32} : i64
      %16 = arith.index_cast %15 {ssbuffer.block_id = 8 : i32} : i64 to index
      %17 = arith.muli %arg14, %c32_i32 {MixUse, ssbuffer.block_id = 8 : i32} : i32
      %18 = arith.muli %arg13, %c32_i32 {ssbuffer.block_id = 8 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x32xf16>
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x32xf16>
        %alloc_10 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x32xf16>
        %alloc_11 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x32xf16>
        %56 = arith.muli %arg16, %c32_i32 {ssbuffer.block_id = 3 : i32} : i32
        %57 = arith.maxsi %17, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %58 = arith.index_cast %57 {ssbuffer.block_id = 3 : i32} : i32 to index
        %59 = arith.maxsi %56, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %60 = arith.index_cast %59 {ssbuffer.block_id = 3 : i32} : i32 to index
        %61 = arith.muli %58, %c256 {ssbuffer.block_id = 3 : i32} : index
        %62 = arith.addi %61, %11 {ssbuffer.block_id = 3 : i32} : index
        %63 = arith.index_cast %arg9 {ssbuffer.block_id = 3 : i32} : i32 to index
        %64 = arith.addi %62, %60 {ssbuffer.block_id = 3 : i32} : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg2 to offset: [%64], sizes: [32, 32], strides: [256, 1] {ssbuffer.block_id = 3 : i32} : memref<?xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
        %65 = arith.maxsi %18, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %66 = arith.index_cast %65 {ssbuffer.block_id = 3 : i32} : i32 to index
        %67 = arith.muli %60, %c64 {ssbuffer.block_id = 3 : i32} : index
        %68 = arith.addi %67, %16 {ssbuffer.block_id = 3 : i32} : index
        %69 = arith.addi %68, %66 {ssbuffer.block_id = 3 : i32} : index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg5 to offset: [%69], sizes: [32, 32], strides: [64, 1] {ssbuffer.block_id = 3 : i32} : memref<?xf16> to memref<32x32xf16, strided<[64, 1], offset: ?>>
        %70 = arith.subi %64, %11 {ssbuffer.block_id = 3 : i32} : index
        %71 = arith.divsi %70, %c256 {ssbuffer.block_id = 3 : i32} : index
        %72 = arith.subi %63, %71 {ssbuffer.block_id = 3 : i32} : index
        %73 = arith.maxsi %72, %c0 {ssbuffer.block_id = 3 : i32} : index
        %74 = arith.minsi %73, %c32 {ssbuffer.block_id = 3 : i32} : index
        %75 = arith.remsi %70, %c256 {ssbuffer.block_id = 3 : i32} : index
        %76 = arith.subi %c64, %75 {ssbuffer.block_id = 3 : i32} : index
        %77 = arith.maxsi %76, %c0 {ssbuffer.block_id = 3 : i32} : index
        %78 = arith.minsi %77, %c32 {ssbuffer.block_id = 3 : i32} : index
        %79 = arith.subi %c0_i32, %17 {ssbuffer.block_id = 3 : i32} : i32
        %80 = arith.maxsi %79, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %81 = arith.index_cast %80 {ssbuffer.block_id = 3 : i32} : i32 to index
        %82 = arith.minsi %81, %74 {ssbuffer.block_id = 3 : i32} : index
        %83 = arith.subi %74, %82 {ssbuffer.block_id = 3 : i32} : index
        %84 = arith.subi %c0_i32, %56 {ssbuffer.block_id = 3 : i32} : i32
        %85 = arith.maxsi %84, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %86 = arith.index_cast %85 {ssbuffer.block_id = 3 : i32} : i32 to index
        %87 = arith.minsi %86, %78 {ssbuffer.block_id = 3 : i32} : index
        %88 = arith.subi %78, %87 {ssbuffer.block_id = 3 : i32} : index
        %subview_14 = memref.subview %reinterpret_cast_12[0, 0] [%83, %88] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x32xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %subview_15 = memref.subview %alloc_9[%82, %87] [%83, %88] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        memref.copy %subview_14, %subview_15 {ssbuffer.block_id = 3 : i32} : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        %89 = bufferization.to_tensor %alloc_9 restrict writable {gm_load_bufferable, ssbuffer.block_id = 3 : i32} : memref<32x32xf16> to tensor<32x32xf16>
        %90 = arith.subi %69, %16 {ssbuffer.block_id = 3 : i32} : index
        %91 = arith.divsi %90, %c64 {ssbuffer.block_id = 3 : i32} : index
        %92 = arith.subi %c64, %91 {ssbuffer.block_id = 3 : i32} : index
        %93 = arith.maxsi %92, %c0 {ssbuffer.block_id = 3 : i32} : index
        %94 = arith.minsi %93, %c32 {ssbuffer.block_id = 3 : i32} : index
        %95 = arith.remsi %90, %c64 {ssbuffer.block_id = 3 : i32} : index
        %96 = arith.subi %c64, %95 {ssbuffer.block_id = 3 : i32} : index
        %97 = arith.maxsi %96, %c0 {ssbuffer.block_id = 3 : i32} : index
        %98 = arith.minsi %97, %c32 {ssbuffer.block_id = 3 : i32} : index
        %99 = arith.minsi %86, %94 {ssbuffer.block_id = 3 : i32} : index
        %100 = arith.subi %94, %99 {ssbuffer.block_id = 3 : i32} : index
        %101 = arith.subi %c0_i32, %18 {ssbuffer.block_id = 3 : i32} : i32
        %102 = arith.maxsi %101, %c0_i32 {ssbuffer.block_id = 3 : i32} : i32
        %103 = arith.index_cast %102 {ssbuffer.block_id = 3 : i32} : i32 to index
        %104 = arith.minsi %103, %98 {ssbuffer.block_id = 3 : i32} : index
        %105 = arith.subi %98, %104 {ssbuffer.block_id = 3 : i32} : index
        %subview_16 = memref.subview %reinterpret_cast_13[0, 0] [%100, %105] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x32xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        %subview_17 = memref.subview %alloc_11[%99, %104] [%100, %105] [1, 1] {ssbuffer.block_id = 3 : i32} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        memref.copy %subview_16, %subview_17 {ssbuffer.block_id = 3 : i32} : memref<?x?xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        %106 = bufferization.to_tensor %alloc_11 restrict writable {gm_load_bufferable, ssbuffer.block_id = 3 : i32} : memref<32x32xf16> to tensor<32x32xf16>
        %107 = tensor.empty() {ssbuffer.block_id = 3 : i32} : tensor<32x32xf32>
        %108 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%cst_1 : f32) outs(%107 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %109 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32} ins(%89, %106 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%108 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %110 = arith.divsi %arg16, %c1_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} : i32
        %c2_i32_18 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} 2 : i32
        %111 = arith.remsi %110, %c2_i32_18 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} : i32
        %c0_i32_19 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} 0 : i32
        %112 = arith.cmpi eq, %111, %c0_i32_19 {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} : i32
        scf.if %112 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
        } else {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %112 {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} ins(%109 : tensor<32x32xf32>) outs(%alloc_4 : memref<32x32xf32, #hivm.address_space<ub>>)
        } else {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32} ins(%109 : tensor<32x32xf32>) outs(%alloc_5 : memref<32x32xf32, #hivm.address_space<ub>>)
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 3 : i32}
        scf.if %112 {
          hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 4
        } else {
          hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 8
        } {ssbuffer.block_id = 3 : i32, ssbuffer.cross_buffer = 1 : i32}
        // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 4 : i32,
        // CHECK-SAME: ssbuffer.clone = 5 : i32} : memref<32x32xf16>
        // CHECK: %[[SUBVIEW0:.*]] = memref.subview %{{.*}}[0, 0] [{{.*}}, {{.*}}] [1, 1]
        // CHECK-SAME: {ssbuffer.block_id = 4 : i32, ssbuffer.clone = 3 : i32}
        // CHECK-SAME: : memref<32x32xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        // CHECK: %[[SUBVIEW1:.*]] = memref.subview %[[ALLOC]]{{.*}} [{{.*}}, {{.*}}] [1, 1]
        // CHECK-SAME: {ssbuffer.block_id = 4 : i32, ssbuffer.clone = 3 : i32}
        // CHECK-SAME: : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        // CHECK: memref.copy %[[SUBVIEW0]], %[[SUBVIEW1]] {ssbuffer.block_id = 4 : i32,
        // CHECK-SAME: ssbuffer.clone = 3 : i32}
        // CHECK-SAME: : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        // CHECK: %[[TO_TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
        // CHECK-SAME: {gm_load_bufferable, ssbuffer.block_id = 4 : i32, ssbuffer.clone = 3 : i32}
        // CHECK-SAME: : memref<32x32xf16>
        %reinterpret_cast_20 = memref.reinterpret_cast %arg3 to offset: [%64], sizes: [32, 32], strides: [256, 1] {ssbuffer.block_id = 4 : i32} : memref<?xf16> to memref<32x32xf16, strided<[256, 1], offset: ?>>
        %subview_21 = memref.subview %reinterpret_cast_20[0, 0] [%83, %88] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<32x32xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %subview_22 = memref.subview %alloc_10[%82, %87] [%83, %88] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        memref.copy %subview_21, %subview_22 {ssbuffer.block_id = 4 : i32} : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[32, 1], offset: ?>>
        %113 = bufferization.to_tensor %alloc_10 restrict writable {gm_load_bufferable, ssbuffer.block_id = 4 : i32} : memref<32x32xf16> to tensor<32x32xf16>
        %114 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<32x32xf16>
        %transposed = linalg.transpose ins(%113 : tensor<32x32xf16>) outs(%114 : tensor<32x32xf16>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32}
        %115 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<32x32xf32>
        %116 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_1 : f32) outs(%115 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %117 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32} ins(%89, %transposed : tensor<32x32xf16>, tensor<32x32xf16>) outs(%116 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %118 = arith.divsi %arg16, %c1_i32 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : i32
        %c2_i32_23 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} 2 : i32
        %119 = arith.remsi %118, %c2_i32_23 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : i32
        %c0_i32_24 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} 0 : i32
        %120 = arith.cmpi eq, %119, %c0_i32_24 {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} : i32
        scf.if %120 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        } else {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
        } {ssbuffer.block_id = 4 : i32, ssbuffer.cross_buffer = 1 : i32}
        scf.if %120 {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} ins(%117 : tensor<32x32xf32>) outs(%alloc_2 : memref<32x32xf32, #hivm.address_space<ub>>)
        } else {
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32} ins(%117 : tensor<32x32xf32>) outs(%alloc_3 : memref<32x32xf32, #hivm.address_space<ub>>)
        } {ssbuffer.block_id = 4 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 2 : i32}
        scf.if %120 {
          hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
        } else {
          hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
        } {ssbuffer.block_id = 4 : i32, ssbuffer.cross_buffer = 1 : i32}
      } {DataUse, ssbuffer.block_id = 19 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
