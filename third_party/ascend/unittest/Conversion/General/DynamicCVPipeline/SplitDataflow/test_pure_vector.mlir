// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

// CHECK-LABEL: func.func @test_pure_vector
// CHECK-NOT: hivm.hir.copy
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: return
module {
  func.func @test_pure_vector(%arg0: memref<128x128xf32>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1.0 : f32

    %alloc = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
    %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%t0 : tensor<128x128xf32>) -> tensor<128x128xf32>

    %add = arith.addf %fill, %fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %mul = arith.mulf %add, %fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>

    return
  }
}
