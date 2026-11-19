// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

// CHECK-LABEL: func.func @test_empty_fill_skip
// CHECK-NOT: hivm.hir.copy
// CHECK-NOT: hivm.hir.fixpipe
// CHECK: return
module {
  func.func @test_empty_fill_skip(%arg0: memref<128x128xf16>, %a: tensor<128x128xf16>, %b: tensor<128x128xf16>) {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 0.0 : f16
    %empty = tensor.empty() {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %fill = linalg.fill {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f16) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<128x128xf16>, tensor<128x128xf16>) outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
    return
  }
}
