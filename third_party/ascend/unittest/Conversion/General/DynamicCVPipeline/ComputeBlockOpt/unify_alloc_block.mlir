// RUN: triton-opt --unify-alloc-block %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // ============================================
  // Test Case: @_sgemm_lora_a_kernel
  // ============================================
  // This test verifies the core functionality of unify-alloc-block pass:
  // - When memref.alloc, linalg.fill (inside scf.if), and memref.copy exist
  //   with different block_ids, the pass should unify them to the same block_id
  // - Target block_id is determined by the memref.copy's block_id
  // - Only coreOps (alloc, fill, scf.if, directUsers) are unified;
  //   condition dependencies are NOT updated
  // - Two scenarios are tested:
  //   1. VECTOR core: alloc(block_id=7) + fill(block_id=6) + copy(block_id=8)
  //      → coreOps unified to block_id=8, conditionOps unchanged
  //   2. CUBE core: alloc(block_id=3) + fill(block_id=1) + copy(block_id=2)
  //      → coreOps unified to block_id=2, conditionOps unchanged
  // ============================================
  func.func @_sgemm_lora_a_kernel(%arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32) {
    %cst = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16
    %c256 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 256 : index
    %c16 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 16 : index
    %c1 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    %c0 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c16_i32 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 16 : i32
    %c256_i32 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 256 : i32
    %14 = arith.minsi %arg5, %arg7 {MixUse, ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %25 = arith.index_cast %arg5 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %28 = arith.index_cast %arg6 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
    %41 = arith.index_cast %arg5 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
    %24 = arith.divsi %arg6, %c256_i32 {Undefined, ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %0 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<16x16xf32>
    %1 = linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_0 : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %44:3 = scf.for %arg22 = %c0_i32 to %24 step %c1_i32 iter_args(%arg23 = %1, %arg24 = %25, %arg25 = %c0) -> (tensor<16x16xf32>, index, index)  : i32 {
      %64 = arith.muli %arg22, %c256_i32 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %65 = arith.subi %arg6, %64 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %66 = arith.index_cast %65 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %70 = arith.cmpi slt, %66, %c256 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      // CHECK: %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16>
      %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16>
      %71 = arith.addi %28, %c16 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %72 = arith.index_cast %14 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %76 = arith.minsi %71, %c16 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %77 = arith.cmpi slt, %76, %c16 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %78 = arith.ori %70, %77 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : i1
      // CHECK: %{{.*}} = arith.addi %c0, %arg{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %79 = arith.addi %c0, %arg25 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      // CHECK: %{{.*}} = arith.addi %arg{{.*}}, %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %80 = arith.addi %arg25, %41 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast_16 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [256, 16], strides: [%c1, %41] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<256x16xf16, strided<[?, ?], offset: ?>>
      %subview_17 = memref.subview %reinterpret_cast_16[0, 0] [%66, %76] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_18 = memref.subview %alloc[0, 0] [%66, %76] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16> to memref<?x?xf16, strided<[16, 1]>>
      // CHECK: %{{.*}} = arith.addi %arg{{.*}}, %c256 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      %81 = arith.addi %arg25, %c256 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : index
      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%{{.*}} : memref<256x16xf16>)
      // CHECK: } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
      scf.if %78 {
        linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%alloc : memref<256x16xf16>)
      } {hivm.unlikely_condition}
      memref.copy %subview_17, %subview_18 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[16, 1]>>
      annotation.mark %alloc {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16>
      %82 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<256x16xf16> to tensor<256x16xf16>
      // CHECK: %{{.*}} = arith.muli %arg{{.*}}, %c256_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
      %83 = arith.muli %arg22, %c256_i32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
      // CHECK: %{{.*}} = arith.subi %arg{{.*}}, %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
      %84 = arith.subi %arg6, %83 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32
      // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<16x256xf16>
      %alloc_19 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<16x256xf16>
      // CHECK: %{{.*}} = arith.addi %{{.*}}, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %85 = arith.addi %41, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.index_cast %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %86 = arith.index_cast %arg5 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      // CHECK: %{{.*}} = arith.maxsi %{{.*}}, %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %87 = arith.maxsi %41, %86 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.minsi %{{.*}}, %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %88 = arith.minsi %85, %87 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.subi %{{.*}}, %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %89 = arith.subi %88, %41 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.index_cast %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %90 = arith.index_cast %84 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      // CHECK: %{{.*}} = arith.maxsi %{{.*}}, %c0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %91 = arith.maxsi %90, %c0 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.minsi %{{.*}}, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %92 = arith.minsi %91, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.minsi %{{.*}}, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %93 = arith.minsi %89, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.minsi %{{.*}}, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %94 = arith.minsi %92, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.cmpi slt, %{{.*}}, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %95 = arith.cmpi slt, %93, %c16 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.cmpi slt, %{{.*}}, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %96 = arith.cmpi slt, %94, %c256 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      // CHECK: %{{.*}} = arith.ori %{{.*}}, %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i1
      %97 = arith.ori %95, %96 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : i1

      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%{{.*}} : memref<16x256xf16>)
      // CHECK: } {hivm.unlikely_condition, ssbuffer.block_id = 2 : i32}
      scf.if %97 {
        linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc_19 : memref<16x256xf16>)
      } {hivm.unlikely_condition}
      %98 = arith.addi %41, %arg24 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast_20 = memref.reinterpret_cast %arg2 to offset: [%98], sizes: [16, 256], strides: [%41, %c1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<16x256xf16, strided<[?, ?], offset: ?>>
      %subview_21 = memref.subview %reinterpret_cast_20[0, 0] [%93, %94] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<16x256xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_22 = memref.subview %alloc_19[0, 0] [%93, %94] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<16x256xf16> to memref<?x?xf16, strided<[256, 1]>>
      memref.copy %subview_21, %subview_22 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[256, 1]>>
      %99 = bufferization.to_tensor %alloc_19 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<16x256xf16> to tensor<16x256xf16>
      %100 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<16x16xf32>
      %cst_23 = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
      %101 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_23 : f32) outs(%100 : tensor<16x16xf32>) -> tensor<16x16xf32>
      %102 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%99, %82 : tensor<16x256xf16>, tensor<256x16xf16>) outs(%101 : tensor<16x16xf32>) -> tensor<16x16xf32>
      annotation.mark %82 {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : tensor<256x16xf16>
      scf.yield {ssbuffer.core_type = "VECTOR, CUBE, VECTOR"} %102, %arg24, %81 : tensor<16x16xf32>, index, index
    } {DataUse, ssbuffer.core_type = "VECTOR, CUBE, VECTOR"}
    return
  }

  //===--------------------------------------------------------------------===//
  // No scf.if wrapper (fill outside scf.if) - SHOULD skip
  // Condition: linalg.fill directly uses alloc, not inside scf.if
  // Expected: No change
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_no_scfif_should_skip
  func.func @test_no_scfif_should_skip(%arg0: memref<?xf16>) {
    %c0_i64 = arith.constant 0 : index
    %v = arith.constant {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f16

    // alloc with block_id=11 should remain unchanged (no scf.if wrap)
    // CHECK: memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    %alloc = memref.alloc() {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>

    // fill without scf.if wrapper should remain unchanged
    // CHECK: linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} ins(%{{.*}} : f16) outs(%{{.*}} : memref<64x64xf16>)
    linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} ins(%v : f16) outs(%alloc : memref<64x64xf16>)

    %k_sub_124 = arith.constant {ssbuffer.block_id = 21 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0_i64], sizes: [64, 64], strides: [%c64, %c1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<64x64xf16, strided<[?, ?], offset: ?>>
    %subview = memref.subview %reinterpret_cast[0, 0] [%k_sub_124, 64] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[?, ?], offset: ?>>

    // Add memref.copy with block_id=15 - pass should not unify because no scf.if
    // CHECK: memref.copy{{.*}}{ssbuffer.block_id = 15 : i32
    %dst_alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    %dst_subview = memref.subview %dst_alloc[0, 0] [%k_sub_124, 64] [1, 1] {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xf16> to memref<?x64xf16, strided<[64, 1]>>
    memref.copy %subview, %dst_subview {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<?x64xf16, strided<[?, ?], offset: ?>> to memref<?x64xf16, strided<[64, 1]>>

    return
  }

  //===----------------------------------------------------------------------===//
  // Test Case: @test_iter_arg_dependency
  //===----------------------------------------------------------------------===//
  // Pattern based on: awq_gemm_kernel
  // Key scenario: scf.for iter_args with scf.if condition dependency
  //===----------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_iter_arg_dependency
  func.func @test_iter_arg_dependency(%arg0: memref<?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c32_idx = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16

    scf.for %arg17 = %c0 to %c32 step %c1 iter_args(%arg19 = %c0) -> (index) {
      // CHECK: arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} 128 : index
      %c128 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} 128 : index
      // CHECK: arith.cmpi {{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %cond = arith.cmpi slt, %arg19, %c32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index

      // CHECK: memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %140 = arith.addi %arg19, %c128 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index

      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %cond {
        linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc : memref<32x32xf16>)
      } {hivm.unlikely_condition}

      %141 = arith.addi %arg19, %c128 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%141], sizes: [32, 32], strides: [%c32_idx, %c1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      %subview_src = memref.subview %reinterpret_cast[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      %subview_dst = memref.subview %alloc[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>

      // CHECK: memref.copy{{.*}}{ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      memref.copy %subview_src, %subview_dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>

      scf.yield %140 : index
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // Test Case: @test_copy_source_subview_traceback
  //===----------------------------------------------------------------------===//
  // Pattern: trace back memref.copy's source subview to memref.reinterpret_cast
  // Key scenario: The copy's source operand (%subview_src) is a subview of
  //   %reinterpret_cast, which has a DIFFERENT block_id (5) from the copy (2).
  //   The pass should trace back from the copy's source subview through the
  //   view-like chain to the reinterpret_cast, and unify them to the copy's
  //   block_id (2).
  // Before:
  //   %reinterpret_cast {block_id=5}                        // source base
  //   %subview_src = subview %reinterpret_cast {block_id=5} // source subview
  //   %subview_dst = subview %alloc {block_id=5}            // dst subview
  //   memref.copy %subview_src, %subview_dst {block_id=2}
  // After (source view chain unified to block_id=2):
  //   %reinterpret_cast {block_id=2}
  //   %subview_src {block_id=2}
  //   %subview_dst {block_id=2}
  //   memref.copy {block_id=2}
  //   (alloc, fill, scf.if, condition deps also unified to block_id=2)
  //===----------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_copy_source_subview_traceback
  func.func @test_copy_source_subview_traceback(%arg0: memref<?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16

    scf.for %arg17 = %c0 to %c32 step %c1 iter_args(%arg19 = %c0) -> (index) {
      %c128 = arith.constant {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} 128 : index
      %cond = arith.cmpi slt, %arg19, %c32 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      %140 = arith.addi %arg19, %c128 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : index
      scf.if %cond {
        linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc : memref<32x32xf16>)
      } {hivm.unlikely_condition}

      // Source view chain: reinterpret_cast and subview_src have block_id=5,
      // should be unified to block_id=2 by tracing copy's source subview.
      // CHECK: %{{.*}} = memref.reinterpret_cast {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%140], sizes: [32, 32], strides: [%c32, %c1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x32xf16, strided<[?, ?], offset: ?>>
      // CHECK: %{{.*}} = memref.subview {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      %subview_src = memref.subview %reinterpret_cast[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>
      // CHECK: %{{.*}} = memref.subview {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      %subview_dst = memref.subview %alloc[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>

      // CHECK: memref.copy{{.*}}{ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      memref.copy %subview_src, %subview_dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>

      scf.yield %140 : index
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // Test Case: @test_block_boundary_source_chain
  //===----------------------------------------------------------------------===//
  // Verify that when the source view chain of a memref.copy spans across MLIR
  // Block boundaries, only ops in the same Block as the copyOp are collected
  // and unified. Ops in different Blocks are skipped.
  //
  // Scenario:
  //   - arg_subview/reinterpret_cast from external arg, defined in parent block
  //   - outer memref.copy (block_id=15) in the same parent block
  //   - inner memref.copy (block_id=2) inside scf.for, also using arg_subview
  // Expected:
  //   - arg_subview/reinterpret_cast unified to block_id=15 (same block as
  //     outer copy)
  //   - inner copy does NOT affect arg_subview (different block -> skipped)
  //===----------------------------------------------------------------------===//
  // CHECK-LABEL: func.func @test_block_boundary_source_chain
  func.func @test_block_boundary_source_chain(%arg0: memref<?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f16

    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [32, 32], strides: [%c32, %c1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf16> to memref<32x32xf16, strided<[?, ?], offset: ?>>
    // CHECK: %{{.*}} = memref.subview {{.*}} {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
    %arg_subview = memref.subview %reinterpret_cast[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[?, ?], offset: ?>>

    %outer_cond = arith.constant 1 : i1
    // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
    %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
    // CHECK: scf.if {{.*}} {
    // CHECK:   linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
    scf.if %outer_cond {
      linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc : memref<32x32xf16>)
    } {hivm.unlikely_condition}
    // CHECK: %{{.*}} = memref.subview {{.*}} {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
    %dst_subview = memref.subview %alloc[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
    // CHECK: memref.copy{{.*}}{ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"}
    memref.copy %arg_subview, %dst_subview {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>

    scf.for %iv = %c0 to %c1 step %c1 {
      %inner_cond = arith.constant 1 : i1
      // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      %inner_alloc = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16>
      // CHECK: scf.if {{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %inner_cond {
        linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%inner_alloc : memref<32x32xf16>)
      } {hivm.unlikely_condition}
      // CHECK: %{{.*}} = memref.subview {{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      %inner_dst = memref.subview %inner_alloc[0, 0] [%c32, %c32] [1, 1] {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE"} : memref<32x32xf16> to memref<?x?xf16, strided<[32, 1]>>
      // CHECK: memref.copy{{.*}}{ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      memref.copy %arg_subview, %inner_dst {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf16, strided<[?, ?], offset: ?>> to memref<?x?xf16, strided<[32, 1]>>
      scf.yield
    }
    return
  }
}
