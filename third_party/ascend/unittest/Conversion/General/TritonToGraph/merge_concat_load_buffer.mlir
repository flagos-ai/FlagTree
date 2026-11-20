// RUN: triton-opt --merge-concat-load-buffer --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive: the canonical tl.cat pattern.
//
// Two masked loads write disjoint column ranges of identically shaped UB
// buffers and are concatenated by extract_slice + insert_slice. The store
// reads [0, %n) x [0, 512), which the union of the two copies covers exactly,
// so the allocs merge into one and both fills die.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_and_drop_fill
// CHECK-NOT:     linalg.fill
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<32x512xf32>
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0, 0]
// CHECK:         memref.copy
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0, 256]
// CHECK:         memref.copy
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
// CHECK:         %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK:         tensor.extract_slice %[[T]][0, 0] [%{{.*}}, 512] [1, 1]
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
func.func @merge_and_drop_fill(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Positive: the shape actually emitted by triton-to-linalg for a cat kernel,
// i.e. inside scf.for with memref.reinterpret_cast sources and a
// materialize_in_destination store.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_inside_scf_for
// CHECK:       scf.for
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<32x512xf32>
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0, 0]
// CHECK:         memref.copy
// CHECK:         memref.subview %[[ALLOC]][0, 256]
// CHECK:         memref.copy
// CHECK-NOT:     tensor.insert_slice
// CHECK:         %[[T:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK:         tensor.extract_slice %[[T]]
// CHECK:         bufferization.materialize_in_destination
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
func.func @merge_inside_scf_for(%in0: memref<?xf32>, %in1: memref<?xf32>, %out: memref<?xf32>, %ub: i32, %bound: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %cn256 = arith.constant -256 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32

  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    %row = arith.muli %i, %c32_i32 : i32
    %row_idx = arith.index_cast %row : i32 to index
    %off0 = arith.muli %row_idx, %c256 : index
    %rc0 = memref.reinterpret_cast %in0 to offset: [%off0], sizes: [32, 512], strides: [256, 1] : memref<?xf32> to memref<32x512xf32, strided<[256, 1], offset: ?>>

    %alloc0 = memref.alloc() : memref<32x512xf32>
    %hi = arith.addi %row_idx, %c32 : index
    %bnd = arith.index_cast %bound : i32 to index
    %lo = arith.maxsi %row_idx, %bnd : index
    %clamped = arith.minsi %hi, %lo : index
    %len = arith.subi %clamped, %row_idx : index
    %len32 = arith.minsi %len, %c32 : index
    %rows = arith.maxsi %len32, %c0 : index
    linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
    %ss0 = memref.subview %rc0[0, 0] [%rows, 256] [1, 1] : memref<32x512xf32, strided<[256, 1], offset: ?>> to memref<?x256xf32, strided<[256, 1], offset: ?>>
    %sd0 = memref.subview %alloc0[0, 0] [%rows, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
    memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[256, 1], offset: ?>> to memref<?x256xf32, strided<[512, 1]>>
    %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

    %off1 = arith.addi %off0, %cn256 : index
    %rc1 = memref.reinterpret_cast %in1 to offset: [%off1], sizes: [32, 512], strides: [256, 1] : memref<?xf32> to memref<32x512xf32, strided<[256, 1], offset: ?>>
    %alloc1 = memref.alloc() : memref<32x512xf32>
    linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
    %ss1 = memref.subview %rc1[0, 256] [%rows, 256] [1, 1] : memref<32x512xf32, strided<[256, 1], offset: ?>> to memref<?x256xf32, strided<[256, 1], offset: ?>>
    %sd1 = memref.subview %alloc1[0, 256] [%rows, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
    memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[256, 1], offset: ?>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
    %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

    %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
    %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>

    %off_out = arith.muli %row_idx, %c512 : index
    %rc_out = memref.reinterpret_cast %out to offset: [%off_out], sizes: [32, 512], strides: [512, 1] : memref<?xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %store_src = tensor.extract_slice %ins[0, 0] [%rows, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
    %store_dst = memref.subview %rc_out[0, 0] [%rows, 512] [1, 1] : memref<32x512xf32, strided<[512, 1], offset: ?>> to memref<?x512xf32, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %store_src in writable %store_dst : (tensor<?x512xf32>, memref<?x512xf32, strided<[512, 1], offset: ?>>) -> ()
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// Positive: LoadConverter inserts a memref.cast between the destination
// subview and the copy. The pass must look through it.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_through_memref_cast
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<32x512xf32>
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0, 0]
// CHECK:         memref.copy
// CHECK:         memref.subview %[[ALLOC]][0, 256]
// CHECK:         memref.copy
// CHECK:         bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
func.func @merge_through_memref_cast(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %cast0 = memref.cast %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[?, ?], offset: ?>>
  memref.copy %ss0, %cast0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[?, ?], offset: ?>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %cast1 = memref.cast %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[?, ?], offset: ?>>
  memref.copy %ss1, %cast1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[?, ?], offset: ?>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Positive: the padding fill is still wrapped in the scf.if that
// fillTensorWithOtherForMaskScenario emits (not yet folded by the
// canonicalizer). A dead fill must take its guarding scf.if with it.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_drop_guarded_fill
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<32x512xf32>
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     scf.if
// CHECK:         memref.subview %[[ALLOC]][0, 0]
// CHECK:         memref.copy
// CHECK:         memref.subview %[[ALLOC]][0, 256]
// CHECK:         memref.copy
// CHECK:         bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     scf.if
// CHECK-NOT:     tensor.insert_slice
func.func @merge_drop_guarded_fill(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %partial = arith.cmpi slt, %n, %c32 : index

  %alloc0 = memref.alloc() : memref<32x512xf32>
  scf.if %partial {
    linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  } {hivm.unlikely_condition}
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  scf.if %partial {
    linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  } {hivm.unlikely_condition}
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Positive: 1-D cat. The concatenated tensor is consumed whole, but the two
// static copies together cover the entire buffer, so the fills are still dead.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_1d_full_cover
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<512xf32>
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0]
// CHECK:         memref.copy
// CHECK:         memref.subview %[[ALLOC]][256]
// CHECK:         memref.copy
// CHECK-NOT:     tensor.insert_slice
// CHECK:         bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
func.func @merge_1d_full_cover(%arg0: memref<512xf32>, %arg1: memref<512xf32>) -> tensor<512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<512xf32>)
  %ss0 = memref.subview %arg0[0] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1]>>
  %sd0 = memref.subview %alloc0[0] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1]>>
  memref.copy %ss0, %sd0 : memref<256xf32, strided<[1]>> to memref<256xf32, strided<[1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<512xf32> to tensor<512xf32>

  %alloc1 = memref.alloc() : memref<512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<512xf32>)
  %ss1 = memref.subview %arg1[256] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1], offset: 256>>
  %sd1 = memref.subview %alloc1[256] [256] [1] : memref<512xf32> to memref<256xf32, strided<[1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<256xf32, strided<[1], offset: 256>> to memref<256xf32, strided<[1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<512xf32> to tensor<512xf32>

  %ex = tensor.extract_slice %t0[0] [256] [1] : tensor<512xf32> to tensor<256xf32>
  %ins = tensor.insert_slice %ex into %t1[0] [256] [1] : tensor<256xf32> into tensor<512xf32>
  return %ins : tensor<512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Fallback: the concatenated tensor is consumed whole while the copies only
// cover [0, %n) rows, so the padding is observable. The allocs still merge but
// exactly one fill survives, placed before both copies.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @merge_keep_single_fill
// CHECK:         %[[ALLOC:.*]] = memref.alloc() : memref<32x512xf32>
// CHECK:         linalg.fill ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<32x512xf32>)
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK:         memref.subview %[[ALLOC]][0, 0]
// CHECK:         memref.copy
// CHECK:         memref.subview %[[ALLOC]][0, 256]
// CHECK:         memref.copy
// CHECK-NOT:     tensor.insert_slice
// CHECK:         bufferization.to_tensor %[[ALLOC]] restrict writable
// CHECK-NOT:     memref.alloc()
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.insert_slice
func.func @merge_keep_single_fill(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<32x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  return %ins : tensor<32x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: both copies target the same column range, so the dest write is not
// disjoint from the insert region. Merging would clobber data.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_overlapping_writes
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_overlapping_writes(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd1 = memref.subview %alloc1[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the extract offset differs from the insert offset, so the source
// data is shifted on its way into the dest. A plain RAUW would be wrong.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_shifted_placement
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_shifted_placement(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd1 = memref.subview %alloc1[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 256] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the two buffers have different shapes, so they cannot share one
// allocation.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_shape_mismatch
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<64x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_shape_mismatch(%arg0: memref<32x512xf32>, %arg1: memref<64x512xf32>, %n: index) -> tensor<64x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<64x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<64x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<64x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<64x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<64x512xf32> to tensor<64x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<64x512xf32>
  return %ins : tensor<64x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the source buffer has a user the pass does not model (a dealloc),
// so its lifetime cannot be folded into the dest buffer.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_unmodeled_alloc_user
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
// CHECK:         memref.dealloc
func.func @no_merge_unmodeled_alloc_user(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  memref.dealloc %alloc0 : memref<32x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the source tensor is read a second time, so the buffer it comes
// from is still live after the concat.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_source_tensor_reused
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_source_tensor_reused(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> (tensor<32x512xf32>, tensor<32x512xf32>) {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  return %ins, %t0 : tensor<32x512xf32>, tensor<32x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the padding is observable and the two fills use different values,
// so neither dropping nor merging the fills is legal.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_conflicting_fill_values
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         linalg.fill
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         linalg.fill
// CHECK:         tensor.insert_slice
func.func @no_merge_conflicting_fill_values(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<32x512xf32> {
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%zero : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%one : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  return %ins : tensor<32x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: a strided insert_slice does not describe a contiguous region, so
// the box arithmetic the pass relies on does not apply.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_strided_insert
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_strided_insert(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<32x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 2] : tensor<32x256xf32> into tensor<32x512xf32>
  return %ins : tensor<32x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: the fill of the source buffer runs *after* the copy it would
// initialise, so the padding is what the concat observes rather than the copied
// data. The read region is fully covered, so without an ordering check the
// rewrite would wrongly delete both fills.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_late_fill
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.copy
// CHECK:         linalg.fill
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_late_fill(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<?x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>
  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %alloc1 = memref.alloc() : memref<32x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  %out = tensor.extract_slice %ins[0, 0] [%n, 512] [1, 1] : tensor<32x512xf32> to tensor<?x512xf32>
  return %out : tensor<?x512xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Negative: each fill precedes the copies into its own buffer, but the fill of
// the surviving (earlier) alloc sits after the copy into the other one. Merging
// would let that fill clobber data the other copy had already written, so the
// keep-one-fill fallback must be refused too.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_merge_fill_after_other_copy
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         memref.alloc() : memref<32x512xf32>
// CHECK:         tensor.insert_slice
func.func @no_merge_fill_after_other_copy(%arg0: memref<32x512xf32>, %arg1: memref<32x512xf32>, %n: index) -> tensor<32x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32

  %alloc0 = memref.alloc() : memref<32x512xf32>
  %alloc1 = memref.alloc() : memref<32x512xf32>

  linalg.fill ins(%cst : f32) outs(%alloc1 : memref<32x512xf32>)
  %ss1 = memref.subview %arg1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  %sd1 = memref.subview %alloc1[0, 256] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1], offset: 256>>
  memref.copy %ss1, %sd1 : memref<?x256xf32, strided<[512, 1], offset: 256>> to memref<?x256xf32, strided<[512, 1], offset: 256>>

  linalg.fill ins(%cst : f32) outs(%alloc0 : memref<32x512xf32>)
  %ss0 = memref.subview %arg0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  %sd0 = memref.subview %alloc0[0, 0] [%n, 256] [1, 1] : memref<32x512xf32> to memref<?x256xf32, strided<[512, 1]>>
  memref.copy %ss0, %sd0 : memref<?x256xf32, strided<[512, 1]>> to memref<?x256xf32, strided<[512, 1]>>

  %t0 = bufferization.to_tensor %alloc0 restrict writable : memref<32x512xf32> to tensor<32x512xf32>
  %t1 = bufferization.to_tensor %alloc1 restrict writable : memref<32x512xf32> to tensor<32x512xf32>

  %ex = tensor.extract_slice %t0[0, 0] [32, 256] [1, 1] : tensor<32x512xf32> to tensor<32x256xf32>
  %ins = tensor.insert_slice %ex into %t1[0, 0] [32, 256] [1, 1] : tensor<32x256xf32> into tensor<32x512xf32>
  return %ins : tensor<32x512xf32>
}
