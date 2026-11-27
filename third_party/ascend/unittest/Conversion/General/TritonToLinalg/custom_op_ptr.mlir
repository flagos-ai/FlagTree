// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// Test hivm.hir.custom with distributed type producing tensor<tt.ptr> results
// The custom op result type is converted to memref, and load uses the remapped result
// CHECK-LABEL: func.func @test_custom_op_ptr_output
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.is_distributed
// CHECK: memref

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_custom_op_ptr_output(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %empty = tensor.empty() : tensor<8x!tt.ptr<f32>>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>

    // Create input tensor<tt.ptr>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    // hivm.hir.custom with distributed type producing tensor<tt.ptr> output
    // The SrcPtrIndex maps result to input operand 0 for structured analysis
    %custom_result = hivm.hir.custom
        {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "test_ptr_op_impl", SrcPtrIndex = array<i32: 0>}
        "test_ptr_op"
        ins(%src_addptr : tensor<8x!tt.ptr<f32>>)
        outs(%empty : tensor<8x!tt.ptr<f32>>) -> tensor<8x!tt.ptr<f32>>

    // Load from custom result pointer - uses remapped value after conversion
    %data = tt.load %custom_result : tensor<8x!tt.ptr<f32>>

    // Store result
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Test hivm.hir.custom with distributed type producing tt.ptr results
// The custom op result type is converted to memref, and load uses the remapped result
// CHECK-LABEL: func.func @test_custom_op_ptr_output
// CHECK: hivm.hir.custom
// CHECK-SAME: hivm.is_distributed
// CHECK: memref

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_custom_op_ptr_output(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>

    // Create input tensor<tt.ptr>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    // hivm.hir.custom with distributed type producing tensor<tt.ptr> output
    // The SrcPtrIndex maps result to input operand 0 for structured analysis
    %custom_result = hivm.hir.custom
        {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "test_ptr_op_impl"}
        "test_ptr_op"
        ins(%src_ptr : !tt.ptr<f32>) -> !tt.ptr<f32>
    %custom_splat = tt.splat %custom_result : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr1 = tt.addptr %custom_splat, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    // Load from custom result pointer - uses remapped value after conversion
    %data = tt.load %src_addptr1 : tensor<8x!tt.ptr<f32>>

    // Store result
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
