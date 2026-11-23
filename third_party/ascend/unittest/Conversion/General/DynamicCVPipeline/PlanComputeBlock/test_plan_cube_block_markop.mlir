// RUN: triton-opt --plan-cube-block %s | FileCheck %s

module {
  // ============================================================================
  // 1. fuse_mark_with_alloc
  // ============================================================================
  // CHECK-LABEL: func.func @fuse_mark_with_alloc
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = [[ID_ALLOC:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: annotation.mark %[[ALLOC]] {ssbuffer.block_id = [[ID_ALLOC]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[T0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = [[ID_MAT:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID_MAT]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @fuse_mark_with_alloc(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<4x4xf16>
    annotation.mark %alloc {ssbuffer.core_type = "CUBE"} : memref<4x4xf16>
    %0 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<4x4xf16> to tensor<4x4xf16>
    %1 = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // -----

  // ============================================================================
  // 2. fuse_mark_with_empty
  // ============================================================================
  // CHECK-LABEL: func.func @fuse_mark_with_empty
  // CHECK: %[[EMPTY:.*]] = tensor.empty() {ssbuffer.block_id = [[ID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: annotation.mark %[[EMPTY]] {ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @fuse_mark_with_empty(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<4x4xf16>
    annotation.mark %0 {ssbuffer.core_type = "CUBE"} : tensor<4x4xf16>
    %1 = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // -----

  // ============================================================================
  // 3. mark_on_block_arg
  // ============================================================================
  // CHECK-LABEL: func.func @mark_on_block_arg
  // CHECK: annotation.mark %arg0 {ssbuffer.block_id = [[ID_MARK:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID_MAT:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @mark_on_block_arg(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    annotation.mark %arg0 {ssbuffer.core_type = "CUBE"} : tensor<4x4xf16>
    %0 = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%arg0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // -----

  // ============================================================================
  // 4. multiple_marks_chain
  // ============================================================================
  // CHECK-LABEL: func.func @multiple_marks_chain
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = [[ID_ALLOC:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: annotation.mark %[[ALLOC]] {ssbuffer.block_id = [[ID_ALLOC]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: annotation.mark %[[ALLOC]] {ssbuffer.block_id = [[ID_ALLOC]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[T0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = [[ID_MAT:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK-NEXT: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID_MAT]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @multiple_marks_chain(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<4x4xf16>
    annotation.mark %alloc {ssbuffer.core_type = "CUBE"} : memref<4x4xf16>
    annotation.mark %alloc {ssbuffer.core_type = "CUBE"} : memref<4x4xf16>
    %0 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<4x4xf16> to tensor<4x4xf16>
    %1 = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // -----

  // ============================================================================
  // 5. non_cube_mark_skipped
  // ============================================================================
  // CHECK-LABEL: func.func @non_cube_mark_skipped
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.core_type = "VECTOR"}
  // CHECK-NEXT: annotation.mark %[[ALLOC]] {ssbuffer.core_type = "VECTOR"} : memref<4x4xf16>
  // CHECK-NOT: ssbuffer.block_id
  // CHECK: %[[T0:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "VECTOR"}
  // CHECK-NEXT: %[[MM:.*]] = linalg.matmul {ssbuffer.block_id = [[ID_MAT:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @non_cube_mark_skipped(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %alloc = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<4x4xf16>
    annotation.mark %alloc {ssbuffer.core_type = "VECTOR"} : memref<4x4xf16>
    %0 = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "VECTOR"} : memref<4x4xf16> to tensor<4x4xf16>
    %1 = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%0, %arg0 : tensor<4x4xf16>, tensor<4x4xf16>) outs(%arg1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // -----

  // ============================================================================
  // 6. if_fill_split_cycle_prevention
  // ============================================================================
  // This test case demonstrates the prevention of a dependency cycle that arises
  // from the combination of annotation.mark (which has memory side-effects) and
  // scf.if containing linalg.fill (which splits the block IDs).
  //
  // Dependency Chain:
  //   %alloc (Block ID: X) -> fill -> memref.copy (Block ID: Y) -> annotation.mark (originally Y)
  //
  // If fuseMarkOpToDef blindly fuses annotation.mark with its defining op (%alloc),
  // annotation.mark would be forced into Block X.
  // This creates a cyclic dependency: Block X -> Block Y -> Block X.
  // The cycle detector must identify this and prevent the fusion, assigning
  // annotation.mark to Block Y instead.
  //
  // Note: Although these blocks are partitioned separately here to maintain safety,
  // a subsequent compute block optimization pass should ideally merge the blocks
  // split by the if-fill control flow back together.
  //
  // CHECK-LABEL: func.func @if_fill_split_cycle_prevention
  // CHECK: %[[ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = [[ID_ALLOC:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: scf.if
  // CHECK:   linalg.fill {ssbuffer.block_id = [[ID_FILL:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview
  // CHECK: memref.copy %[[SUBVIEW_SRC]], %[[SUBVIEW_DST]] {ssbuffer.block_id = [[ID_COPY:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: annotation.mark %[[ALLOC]] {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = [[ID_COPY]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @if_fill_split_cycle_prevention(%arg0: memref<32x16xf32>, %arg1: i1) {
    %cst = arith.constant 0.0 : f32
    %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<32x16xf32>
    scf.if %arg1 {
      linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%alloc : memref<32x16xf32>)
    }
    %subview_src = memref.subview %arg0[0, 0] [32, 16] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<32x16xf32> to memref<32x16xf32, strided<[16, 1]>>
    %subview_dst = memref.subview %alloc[0, 0] [32, 16] [1, 1] {ssbuffer.core_type = "CUBE"} : memref<32x16xf32> to memref<32x16xf32, strided<[16, 1]>>
    memref.copy %subview_src, %subview_dst {ssbuffer.core_type = "CUBE"} : memref<32x16xf32, strided<[16, 1]>> to memref<32x16xf32, strided<[16, 1]>>
    annotation.mark %alloc {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<32x16xf32>
    return
  }
}
