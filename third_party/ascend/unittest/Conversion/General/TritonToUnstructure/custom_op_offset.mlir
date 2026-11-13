// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s

// Test hivm.hir.custom with distributed type and ContinuousMemAccess annotation
// The annotation is processed by checkStructureAnnotated and erased during analysis
// CHECK-LABEL: tt.func @test_custom_op_offset
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.is_distributed
// CHECK-NOT: annotation.mark

tt.func @test_custom_op_offset(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) -> tensor<8xf32> {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %empty = tensor.empty() : tensor<8x!tt.ptr<f32>>

    // hivm.hir.custom with distributed type producing tensor<tt.ptr> output
    %3 = hivm.hir.custom
        {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "test_offset_op_impl", SrcPtrIndex = array<i32: 0>}
        "test_offset_op"
        ins(%2 : tensor<8x!tt.ptr<f32>>)
        outs(%empty : tensor<8x!tt.ptr<f32>>) -> tensor<8x!tt.ptr<f32>>

    // Mark as continuous mem access - this annotation is processed and erased during analysis
    annotation.mark %3 {ContinuousMemAccess} : tensor<8x!tt.ptr<f32>>

    %4 = tt.load %3 : tensor<8x!tt.ptr<f32>>
    tt.return %4 : tensor<8xf32>
}
