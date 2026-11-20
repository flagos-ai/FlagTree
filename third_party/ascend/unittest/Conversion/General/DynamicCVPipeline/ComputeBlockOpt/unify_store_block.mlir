// RUN: triton-opt --unify-store-block %s | FileCheck %s

module {
  // ============================================
  // Test 1: VECTOR producer hit - basic unification
  //
  // Pattern (from real kernel IR):
  //   arith.truncf(VECTOR, block_id=7)
  //     -> tensor.extract_slice(block_id=8)
  //     -> bufferization.materialize_in_destination(block_id=8)
  //   dest chain: memref.reinterpret_cast(block_id=8) ->
  //   memref.subview(block_id=8)
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=7).
  // The store chain (extract_slice, reinterpret_cast, subview, materialize)
  // and the scalar dep (%c0 used by reinterpret_cast) are all at block_id=8.
  // After pass: all unified to producer's block_id=7.
  // ============================================
  // CHECK-LABEL: func @test_vector_producer_hit
  func.func @test_vector_producer_hit(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // CHECK: arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 2: CUBE producer skip - no unification
  //
  // Pattern:
  //   linalg.matmul(CUBE, block_id=1) -> tensor.extract_slice(block_id=2)
  //     -> bufferization.materialize_in_destination(block_id=2)
  //   (no truncf; producer traced directly to matmul)
  //
  // producer = linalg.matmul (CUBE_ONLY) -> pass skips this store.
  // All block_ids remain unchanged.
  // ============================================
  // CHECK-LABEL: func @test_cube_producer_skip
  func.func @test_cube_producer_skip(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %A: tensor<64x64xf16>, %B: tensor<64x64xf16>, %init: tensor<64x64xf16>) {
    // CHECK: arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"}
    %matmul = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %matmul[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 4: hivm.hir.store use case
  //
  // Same shape as Test 1 but using hivm.hir.store instead of
  // bufferization.materialize_in_destination. Store source operand is the
  // tensor %extract (getSrc()), dest is the memref %subview (getDst()).
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=57).
  // store + extract_slice + subview chain unify to 57.
  // ============================================
  // CHECK-LABEL: func @test_hivm_store
  func.func @test_hivm_store(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast's offset (scalar dep), unified to 57.
    // CHECK: arith.constant {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: hivm.hir.store ins({{.*}} : tensor<64x64xf16>) outs({{.*}} : memref<64x64xf16, strided<[64, 1], offset: ?>>) {ssbuffer.block_id = 57 : i32, ssbuffer.core_type = "VECTOR"}
    hivm.hir.store ins(%extract : tensor<64x64xf16>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>>) {ssbuffer.block_id = 58 : i32, ssbuffer.core_type = "VECTOR"}
    return
  }

  // ============================================
  // Test 5: Multiple vector ops in chain - producer is the first vector
  // encountered when tracing back from store (closest to store), not the
  // upstream one.
  //
  // Data chain:
  //   %v1 = truncf (id=67) -> %v2 = addf (id=68) -> extract_slice (id=69) -> store (id=69)
  //
  // traceProducerOp pierces extract_slice (view), hits %v2 (VECTOR_ONLY) and
  // returns immediately. %v1 is NOT reached, so:
  //   producer = %v2 (id=68), stays 68
  //   %v1 stays 67 (not producer, not collected)
  //   store + extract_slice + dest view chain + %c0 unify to 68
  // ============================================
  // CHECK-LABEL: func @test_multiple_vector_producer_first
  func.func @test_multiple_vector_producer_first(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast's offset (scalar dep), unified to 68.
    // CHECK: arith.constant {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // upstream vector op, NOT the producer; stays 67.
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 67 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %v1 = arith.truncf %in {ssbuffer.block_id = 67 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // downstream vector op (closest to store), IS the producer; stays 68.
    // CHECK: arith.addf %{{.*}}, %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
    %v2 = arith.addf %v1, %v1 {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %v2[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 68 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 69 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 6: clone chained scalar ops (incl. constant) shared between store
  // pattern and external block.
  // ============================================
  // CHECK-LABEL: func @test_clone_scalar_cross_block
  func.func @test_clone_scalar_cross_block(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>, %idx: index) {
    %step = arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %c64 = arith.constant {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} 32 : index
    // clone of %step (inserted before original): keeps block 12, feeds clones of %sub/%add.
    // CHECK: arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    // original %step: stays block 12, feeds originals of %sub/%add.
    // CHECK: arith.constant {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    // %c64 (external constant): stays block 13, feeds %ext.
    // CHECK: arith.constant {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} 32 : index
    // producer stays block 12.
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // clone of %sub: keeps block 13 (inserted before original, feeds clone of %add).
    // CHECK: arith.subi %{{.*}}, %{{.*}} {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    // original %sub: unified to producer's block 12 (feeds original %add).
    // CHECK: arith.subi %{{.*}}, %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    %sub = arith.subi %idx, %step {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    // clone of %add: keeps block 13 (inserted before original, feeds external %ext).
    // CHECK: arith.addi %{{.*}}, %{{.*}} {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    // original %add: unified to producer's block 12 (feeds dest chain).
    // CHECK: arith.addi %{{.*}}, %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : index
    %add = arith.addi %sub, %step {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    // dest chain uses original %add, unified to block 12.
    // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}] {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%add], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    // external user stays block 13, now uses the clone of %add.
    // CHECK: arith.muli %{{.*}}, %{{.*}} {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    %ext = arith.muli %add, %c64 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
    return
  }

  // ============================================
  // Test 7: producer and store already share the same block_id, but viewOps
  // in between have a different block_id. The pass should still collect and
  // unify those viewOps to the producer's block_id.
  //
  // Pattern:
  //   arith.truncf(VECTOR, block_id=5)                <- producer
  //     -> tensor.extract_slice(block_id=6)            <- viewOp to unify
  //     -> bufferization.materialize_in_destination(block_id=5)  <- store
  //   dest chain: memref.reinterpret_cast(block_id=6)  <- viewOp to unify
  //             -> memref.subview(block_id=6)           <- viewOp to unify
  //
  // All viewOps (extract_slice, reinterpret_cast, subview) at block_id=6
  // should be unified to block_id=5.
  // ============================================
  // CHECK-LABEL: func @test_producer_store_same_block
  func.func @test_producer_store_same_block(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0: scalar dep at block 6, unified to 5.
    // CHECK: arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // producer stays at block 5.
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    // extract_slice at block 6, unified to 5.
    // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
    // reinterpret_cast at block 6, unified to 5.
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // subview at block 6, unified to 5.
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    // store stays at block 5.
    // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    return
  }

  // ============================================
  // Test 8: Producer outside scf.for, store inside → NO unification
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=5) lives in the function
  // body, while the store and its data/dest chain are inside scf.for
  // (block_id=8). traceProducerOp still finds the producer because
  // %extract_slice directly references %trunc from the outer scope, but
  // matchStorePattern rejects the match because producer->getBlock() !=
  // storeOp->getBlock() (different IR blocks).
  //
  // All block_ids remain unchanged.
  // ============================================
  // CHECK-LABEL: func @test_producer_outside_for
  func.func @test_producer_outside_for(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast's offset (scalar dep), stays at block 8.
    // CHECK: arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // producer outside scf.for, stays at block 5.
    // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %trunc = arith.truncf %in {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
    %lb = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %ub = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %step = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    // store and its chain inside scf.for; all stay at block 8.
    scf.for %i = %lb to %ub step %step {
      // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
      // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
      bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }

  // ============================================
  // Test 9: Dest view ops outside scf.for, store and producer inside
  //         → unification happens
  //
  // producer = arith.truncf (VECTOR_ONLY, block_id=5) inside scf.for.
  // store = bufferization.materialize_in_destination (block_id=8) inside for.
  // data view = tensor.extract_slice (block_id=8) inside for.
  // dest views = memref.reinterpret_cast, memref.subview (block_id=8) outside
  // for.
  //
  // producer and store are in the same IR block (inside scf.for) → pattern
  // matched.  All matched ops unified to producer's block_id=5.  Scalar
  // deps outside the for loop (%c0, for-loop bounds) stay at their original
  // block_ids.
  // ============================================
  // CHECK-LABEL: func @test_dest_view_outside_for
  func.func @test_dest_view_outside_for(%arg0: memref<?xf16> {tt.divisibility = 16 : i32}, %in: tensor<64x64xf32>) {
    // %c0 feeds reinterpret_cast, stays at block 8 (outside for, not collected
    // by scalar deps).
    // CHECK: arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c0 = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    // dest view ops outside scf.for, unified to producer's block 5.
    // CHECK: memref.reinterpret_cast %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [128, 64], strides: [64, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
    // CHECK: memref.subview %{{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %subview = memref.subview %reinterpret[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %lb = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %ub = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %step = arith.constant {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} 1 : index
    // producer and store inside scf.for.
    scf.for %i = %lb to %ub step %step {
      // producer stays at block 5.
      // CHECK: arith.truncf %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
      %trunc = arith.truncf %in {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xf16>
      // data view unified to block 5.
      // CHECK: tensor.extract_slice %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      %extract = tensor.extract_slice %trunc[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf16> to tensor<64x64xf16>
      // store unified to block 5.
      // CHECK: bufferization.materialize_in_destination %{{.*}} {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
      bufferization.materialize_in_destination %extract in writable %subview {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64x64xf16>, memref<64x64xf16, strided<[64, 1], offset: ?>>) -> ()
    }
    return
  }
}
