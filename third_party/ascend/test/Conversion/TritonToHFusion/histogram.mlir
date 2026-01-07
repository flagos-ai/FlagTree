// RUN: triton-adapter-opt %s -triton-to-hfusion -verify-diagnostics -split-input-file | FileCheck %set

module {
    func.func @test_histogram(%arg0: tensor<16xf32>, %arg1: tensor<16xi32>) {
        // CHECK: hfusion.histogram
        %cst_min = arith.constant 0.0 : f32
        %cst_max = arith.constant 1.0 : f32
        %cst_bins = arith.constant 256 : i32

        %res = tt.histogram %arg0, %cst_min, %cst_max, %cst_bins 
            {num_bins = 256 : i32, axis = 0 : i32} 
            : (tensor<16xf32>, f32, f32, i32) -> tensor<256xi32>

        tt.return %res : tensor<256xi32>
    }
}