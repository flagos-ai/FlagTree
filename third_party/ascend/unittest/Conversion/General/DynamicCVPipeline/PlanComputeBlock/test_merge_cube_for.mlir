// RUN: triton-opt --split-input-file --op-classifier %s | FileCheck %s

// Test Case 1: Store to global memory (GM) via bufferization.materialize_in_destination.
// Verifies that:
// 1. MemoryEffects::Write targeting a function BlockArgument (GM) disqualifies the
//    scf.for loop from being treated as a Cube loader loop.
// 2. All operations inside the loop body are colored as VECTOR.
// 3. None of the operations inside the loop are marked as CUBE.
func.func @test_materialize_to_gm_disqualifies_cube_loader(
    %arg0: memref<?xf16>,
    %arg1: memref<?xf16>,
    %start: i32,
    %end: i32,
    %step: i32,
    %mat_a: tensor<128x128xf16>,
    %mat_b: tensor<128x128xf16>,
    %mat_out: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  // CHECK-LABEL: func.func @test_materialize_to_gm_disqualifies_cube_loader(

  // CHECK:         scf.for
  // CHECK-NOT:       {ssbuffer.core_type = "CUBE"}
  // CHECK:           %[[ALLOC:.+]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<1x512xf16>
  // CHECK:           memref.copy {{.*}}, %[[ALLOC]] {ssbuffer.core_type = "VECTOR"}
  // CHECK:           %[[TENSOR:.+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "VECTOR"}
  // CHECK:           bufferization.materialize_in_destination %[[TENSOR]] in writable {{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK:         }
  scf.for %iv = %start to %end step %step : i32 {
    %idx = arith.index_cast %iv : i32 to index
    %reinterpret_in = memref.reinterpret_cast %arg0 to offset: [%idx], sizes: [1, 512], strides: [512, 1] : memref<?xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x512xf16>
    memref.copy %reinterpret_in, %alloc : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
    %tensor = bufferization.to_tensor %alloc restrict writable : memref<1x512xf16> to tensor<1x512xf16>

    // Write to function argument %arg1 (GM) via reinterpret_cast
    %reinterpret_out = memref.reinterpret_cast %arg1 to offset: [%idx], sizes: [1, 512], strides: [512, 1] : memref<?xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %tensor in writable %reinterpret_out : (tensor<1x512xf16>, memref<1x512xf16, strided<[512, 1], offset: ?>>) -> ()
  }

  // Downstream CUBE operation ensures Cube pipeline is active
  // CHECK:         linalg.matmul {ssbuffer.core_type = "CUBE"}
  %res = linalg.matmul ins(%mat_a, %mat_b : tensor<128x128xf16>, tensor<128x128xf16>) outs(%mat_out : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}

// -----

// Test Case 2: Store to global memory (GM) via direct memref.copy.
// Verifies that direct memref.copy whose destination aliases a function BlockArgument
// also disqualifies the loader loop, ensuring all loop operations remain VECTOR.
func.func @test_copy_to_gm_disqualifies_cube_loader(
    %arg0: memref<?xf32>,
    %arg1: memref<?xf32>,
    %start: i32,
    %end: i32,
    %step: i32,
    %mat_a: tensor<64x64xf32>,
    %mat_b: tensor<64x64xf32>,
    %mat_out: tensor<64x64xf32>
) -> tensor<64x64xf32> {
  // CHECK-LABEL: func.func @test_copy_to_gm_disqualifies_cube_loader(

  // CHECK:         scf.for
  // CHECK-NOT:       {ssbuffer.core_type = "CUBE"}
  // CHECK:           %[[ALLOC:.+]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64xf32>
  // CHECK:           memref.copy {{.*}}, %[[ALLOC]] {ssbuffer.core_type = "VECTOR"}
  // CHECK:           memref.copy %[[ALLOC]], {{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK:         }
  scf.for %iv = %start to %end step %step : i32 {
    %idx = arith.index_cast %iv : i32 to index
    %alloc = memref.alloc() : memref<64xf32>
    %reinterpret_in = memref.reinterpret_cast %arg0 to offset: [%idx], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    memref.copy %reinterpret_in, %alloc : memref<64xf32, strided<[1], offset: ?>> to memref<64xf32>

    // Direct write-back to GM argument %arg1
    %reinterpret_out = memref.reinterpret_cast %arg1 to offset: [%idx], sizes: [64], strides: [1] : memref<?xf32> to memref<64xf32, strided<[1], offset: ?>>
    memref.copy %alloc, %reinterpret_out : memref<64xf32> to memref<64xf32, strided<[1], offset: ?>>
  }

  // CHECK:         linalg.matmul {ssbuffer.core_type = "CUBE"}
  %res = linalg.matmul ins(%mat_a, %mat_b : tensor<64x64xf32>, tensor<64x64xf32>) outs(%mat_out : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

// -----

// Test Case 3: Legitimate Cube loader loop without GM stores.
// Verifies that a true Cube loader loop (loading GM data into a local memref.alloc buffer
// whose results are consumed only by linalg.matmul) is NOT disqualified and is correctly
// colored as CUBE.
func.func @test_legitimate_cube_loader_coloring(
    %arg0: memref<?xf16>,
    %start: i32,
    %end: i32,
    %step: i32,
    %mat_b: tensor<128x128xf16>,
    %mat_out: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  // CHECK-LABEL: func.func @test_legitimate_cube_loader_coloring(
  // CHECK:         %[[INIT:.+]] = tensor.empty() {{.*}} : tensor<128x128xf16>
  // CHECK:         %[[LOADED:.+]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<128x128xf16>)
  // CHECK:           arith.index_cast {{.*}} {ssbuffer.core_type = "CUBE"} : i32 to index
  // CHECK:           memref.reinterpret_cast {{.*}} {ssbuffer.core_type = "CUBE"}
  // CHECK:           %[[ALLOC:.+]] = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
  // CHECK:           memref.copy {{.*}}, %[[ALLOC]] {ssbuffer.core_type = "CUBE"}
  // CHECK:           %[[TENSOR:.+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "CUBE"}
  // CHECK:           scf.yield {ssbuffer.core_type = "CUBE"} %[[TENSOR]]
  // CHECK:         } {ssbuffer.core_type = "CUBE"}
  // CHECK:         linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%[[LOADED]],
  %init = tensor.empty() : tensor<128x128xf16>
  %loaded = scf.for %iv = %start to %end step %step iter_args(%iter = %init) -> (tensor<128x128xf16>) : i32 {
    %idx = arith.index_cast %iv : i32 to index
    %reinterpret = memref.reinterpret_cast %arg0 to offset: [%idx], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
    %alloc = memref.alloc() : memref<128x128xf16>
    memref.copy %reinterpret, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
    %tensor = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16> to tensor<128x128xf16>
    scf.yield %tensor : tensor<128x128xf16>
  }

  %res = linalg.matmul ins(%loaded, %mat_b : tensor<128x128xf16>, tensor<128x128xf16>) outs(%mat_out : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}

// -----

// Test Case 4: Multi-level alias chain resolution (subview over reinterpret_cast of GM argument).
// Reproduces the exact faulty scenario from sparse_flash_attention_grad_kernel:
// 1. Loop contains conditional fill under scf.if.
// 2. Writes targeting a subview of reinterpret_cast pointing to %arg_gm_out.
// 3. Verifies that getViewSource resolves the full alias chain to the func BlockArgument.
// 4. Verifies all ops remain VECTOR and no faulty CUBE coloring occurs.
func.func @test_nested_view_gm_store_vector_coloring(
    %arg_in: memref<?xf16>,
    %arg_gm_out: memref<?xf16>,
    %start: i32,
    %end: i32,
    %step: i32,
    %cond: i1,
    %cst: f16,
    %mat_a: tensor<128x128xf16>,
    %mat_b: tensor<128x128xf16>,
    %mat_out: tensor<128x128xf32>
) -> tensor<128x128xf32> {
  // CHECK-LABEL: func.func @test_nested_view_gm_store_vector_coloring(

  // CHECK:         scf.for
  // CHECK-NOT:       {ssbuffer.core_type = "CUBE"}
  // CHECK:           %[[ALLOC:.+]] = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<1x512xf16>
  // CHECK:           scf.if
  // CHECK:             linalg.fill {ssbuffer.core_type = "VECTOR"}
  // CHECK:           }
  // CHECK:           memref.copy {{.*}}, %[[ALLOC]] {ssbuffer.core_type = "VECTOR"}
  // CHECK:           %[[TENSOR:.+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.core_type = "VECTOR"}
  // CHECK:           %[[SUBVIEW:.+]] = memref.subview {{.*}} {ssbuffer.core_type = "VECTOR"}
  // CHECK:           bufferization.materialize_in_destination %[[TENSOR]] in writable %[[SUBVIEW]] {ssbuffer.core_type = "VECTOR"}
  // CHECK:         }
  scf.for %iv = %start to %end step %step : i32 {
    %idx = arith.index_cast %iv : i32 to index
    %reinterpret_in = memref.reinterpret_cast %arg_in to offset: [%idx], sizes: [1, 512], strides: [512, 1] : memref<?xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x512xf16>

    scf.if %cond {
      linalg.fill ins(%cst : f16) outs(%alloc : memref<1x512xf16>)
    }

    memref.copy %reinterpret_in, %alloc : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
    %tensor = bufferization.to_tensor %alloc restrict writable : memref<1x512xf16> to tensor<1x512xf16>

    // Multi-level view chain: subview -> reinterpret_cast -> %arg_gm_out
    %reinterpret_out = memref.reinterpret_cast %arg_gm_out to offset: [%idx], sizes: [1, 512], strides: [512, 1] : memref<?xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
    %subview_out = memref.subview %reinterpret_out[0, 0] [1, 512] [1, 1] : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16, strided<[512, 1], offset: ?>>
    bufferization.materialize_in_destination %tensor in writable %subview_out : (tensor<1x512xf16>, memref<1x512xf16, strided<[512, 1], offset: ?>>) -> ()
  }

  // CHECK:         linalg.matmul {ssbuffer.core_type = "CUBE"}
  %res = linalg.matmul ins(%mat_a, %mat_b : tensor<128x128xf16>, tensor<128x128xf16>) outs(%mat_out : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
