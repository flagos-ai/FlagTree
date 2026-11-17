// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

// CHECK-LABEL: func.func @test_pure_cube
// CHECK-NOT: hivm.hir.copy
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: return
module {
  func.func @test_pure_cube(%arg0: memref<128x128xf16>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} 0.0 : f16

    %alloc = memref.alloc() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16>
    %t0 = bufferization.to_tensor %alloc {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf16> to tensor<128x128xf16>
    %empty = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf16>
    %fill = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%empty : tensor<128x128xf16>) -> tensor<128x128xf16>

    %mm = linalg.matmul {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} ins(%t0, %fill : tensor<128x128xf16>, tensor<128x128xf16>) outs(%empty : tensor<128x128xf16>) -> tensor<128x128xf16>

    return
  }
}
