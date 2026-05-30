// RUN: triton-opt %s -triton-to-hfusion  | FileCheck %s


// CHECK-LABEL: tt.func @test_histogram
tt.func @test_histogram(%arg0 : tensor<16xi32>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %{{.*}}, 2
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}
