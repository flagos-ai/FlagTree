// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// Scalar V->C dependency via external-input.
//
// A VECTOR block produces a 1D tensor (math.floor is a VECTOR-only op), then
// `tensor.extract`s an i32 scalar from it (memref-era scalar channel only
// carries i32 through the memref<i32, ssbuf> SSBuffer slot). A CUBE block
// consumes that scalar (arith.addi). analyzeExternalInputs sees the scalar as
// a CUBE-block external input defined by a VECTOR op and records a V->C
// dependency.
//
// inter-core-transfer-and-sync then routes the scalar through the SSBuffer
// scalar channel:
//   VECTOR side: hivm.hir.pointer_cast + memref.store %extracted
//                + sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>]
//   CUBE side:   sync_block_wait[<CUBE>, <PIPE_S>, <PIPE_S>] + memref.load
//                + annotation.mark {memref_ext.volatile}
// The scalar PIPE_S sync stays isolated from the tensor flag space.
// CHECK-LABEL: func.func @scalar_v2c
// CHECK: %{{.*}} = tensor.extract
// CHECK: hivm.hir.pointer_cast({{.*}})
// CHECK: memref.store %{{.*}}, %{{.*}}[] {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: hivm.hir.sync_block_set {{.*}}[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 1
// CHECK: hivm.hir.sync_block_wait {{.*}}[<CUBE>, <PIPE_S>, <PIPE_S>] flag = 1
// CHECK: memref.load %{{.*}}[] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE", ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : memref<i32, #hivm.address_space<ssbuf>>
// CHECK: annotation.mark {{.*}} {memref_ext.volatile, {{.*}}} : i32
// The CUBE op must consume the loaded scalar, not the raw extract.
// CHECK: arith.addi
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @scalar_v2c(%arg0: memref<?xi32> {tt.tensor_kind = 0 : i32}) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c0 = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %t = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %filled = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%t : tensor<64xf32>) -> tensor<64xf32>
    %floor = math.floor %filled {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %cast = arith.fptosi %floor {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32> to tensor<64xi32>
    %scalar = tensor.extract %cast[%c0] {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xi32>
    %cst_c = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} 2 : i32
    %used = arith.addi %scalar, %cst_c {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : i32
    %dst = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
    %subview = memref.subview %dst[0] [1] [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : memref<1xi32, strided<[1]>> to memref<1xi32, strided<[1]>>
    %tt = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<1xi32>
    %filled_c = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%used : i32) outs(%tt : tensor<1xi32>) -> tensor<1xi32>
    %extracted_slice = tensor.extract_slice %filled_c[0] [1] [1] {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<1xi32> to tensor<1xi32>
    bufferization.materialize_in_destination %extracted_slice in writable %subview {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
    return
  }
}
