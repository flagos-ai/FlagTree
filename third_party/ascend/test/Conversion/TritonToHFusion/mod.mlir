// RUN: triton-adapter-opt %s -triton-to-hfusion | FileCheck %s

// CHECK: tensor.empty() : tensor<1xf32>
// CHECK: hfusion.elemwise_binary {fun = #hfusion.binary_fn<mod>} ins(%arg0, %arg1 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) -> tensor<1xf32>

module {
    tt.func @test_mod(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
        %0 = tt.mod %arg0, %arg1 : tensor<1xf32> tensor<1xf32> -> tensor<1xf32>
        tt.return %0 : tensor<1xf32>
    }
}
