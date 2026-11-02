// RUN: triton-opt %s -triton-to-hfusion  | FileCheck %s


// CHECK-LABEL: tt.func @test_histogram
tt.func @test_histogram(%arg0 : tensor<16xi32>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %{{.*}}, 2
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}

// CHECK-LABEL: tt.func @test_histogram_mask
// CHECK-SAME: %[[SRC:[^: ]+]]: tensor<16xi32>{{.*}}%[[MASK:[^: ]+]]: tensor<16xi1>
tt.func @test_histogram_mask(%arg0 : tensor<16xi32>, %arg1 : tensor<16xi1>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %[[SRC]], 2, %[[MASK]] : tensor<16xi32>, tensor<16xi1> -> tensor<2xi32>
    %res = tt.histogram %arg0, %arg1 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}
