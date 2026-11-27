// RUN: triton-opt %s -triton-to-hfusion  | FileCheck %s
// RUN: triton-opt %s -triton-to-hfusion  | FileCheck %s --check-prefix=NOHFUSION


// CHECK-LABEL: tt.func @test_histogram
tt.func @test_histogram(%arg0 : tensor<16xi32>) -> tensor<2xi32> {
    // CHECK: hfusion.histogram %{{.*}}, 2
    %res = tt.histogram %arg0 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}

// CHECK-LABEL: tt.func @test_histogram_mask
// CHECK-SAME: %[[SRC:[^: ]+]]: tensor<16xi32>{{.*}}%[[MASK:[^: ]+]]: tensor<16xi1>
// NOHFUSION-LABEL: tt.func @test_histogram_mask
// NOHFUSION-NOT: hfusion.histogram
// NOHFUSION: tt.return
tt.func @test_histogram_mask(%arg0 : tensor<16xi32>, %arg1 : tensor<16xi1>) -> tensor<2xi32> {
    // CHECK: %[[WIDE:.*]] = arith.select %[[MASK]], %{{.*}}, %{{.*}} : tensor<16xi1>, tensor<16xi32>
    // CHECK: scf.for
    // CHECK: tensor.extract %[[SRC]][%{{.*}}] : tensor<16xi32>
    // CHECK: %[[PRED:.*]] = tensor.extract %[[WIDE]][%{{.*}}] : tensor<16xi32>
    // CHECK: arith.cmpi ne, %[[PRED]], %{{.*}} : i32
    // CHECK: %[[OLD:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<2xi32>
    // CHECK: %[[NEW:.*]] = arith.addi %[[OLD]], %{{.*}} : i32
    // CHECK: %[[SELECTED:.*]] = arith.select %{{.*}}, %[[NEW]], %[[OLD]] : i32
    // CHECK: %[[HIST:.*]] = tensor.insert %[[SELECTED]] into %{{.*}}[%{{.*}}] : tensor<2xi32>
    // CHECK: scf.yield %[[HIST]] : tensor<2xi32>
    // CHECK-NOT: hfusion.histogram
    %res = tt.histogram %arg0, %arg1 : tensor<16xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}

// CHECK-LABEL: tt.func @test_histogram_dynamic_mask
// CHECK: %[[SIZE:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi1>
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?xi32>
// CHECK: %[[DYNAMIC_WIDE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<?xi1>, tensor<?xi32>
// CHECK: scf.for
// CHECK: tensor.extract %[[DYNAMIC_WIDE]][%{{.*}}] : tensor<?xi32>
// CHECK: tensor.insert
// NOHFUSION-LABEL: tt.func @test_histogram_dynamic_mask
// NOHFUSION-NOT: hfusion.histogram
// NOHFUSION: tt.return
tt.func @test_histogram_dynamic_mask(%arg0 : tensor<?xi32>, %arg1 : tensor<?xi1>) -> tensor<2xi32> {
    %res = tt.histogram %arg0, %arg1 : tensor<?xi32> -> tensor<2xi32>
    tt.return %res : tensor<2xi32>
}

// CHECK-LABEL: tt.func @test_two_masked_histograms
// CHECK-COUNT-2: scf.for
// CHECK: tt.return
// NOHFUSION-LABEL: tt.func @test_two_masked_histograms
// NOHFUSION-NOT: hfusion.histogram
// NOHFUSION: tt.return
tt.func @test_two_masked_histograms(%arg0 : tensor<16xi32>, %arg1 : tensor<16xi1>, %arg2 : tensor<16xi1>) -> (tensor<2xi32>, tensor<4xi32>) {
    %first = tt.histogram %arg0, %arg1 : tensor<16xi32> -> tensor<2xi32>
    %second = tt.histogram %arg0, %arg2 : tensor<16xi32> -> tensor<4xi32>
    tt.return %first, %second : tensor<2xi32>, tensor<4xi32>
}
