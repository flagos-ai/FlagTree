// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

// CHECK-LABEL: func.func @test_non_tensor_type
// CHECK-NOT: hivm.hir.copy
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: return
module {
  func.func @test_non_tensor_type(%a: f32, %b: f32) {
    %add = arith.addf %a, %b {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %mul = arith.mulf %add, %b {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %empty = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%mul : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%fill, %fill : tensor<128x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    return
  }
}
