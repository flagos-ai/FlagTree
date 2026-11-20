// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// Test hivm.hir.custom_macro survives triton-to-linalg with sync attrs preserved.
// CHECK-LABEL: func.func @test_custom_macro_op
// CHECK: hivm.hir.custom_macro
// CHECK: hivm.pipe_in = #hivm.pipe<PIPE_MTE2>
// CHECK: hivm.pipe_out = #hivm.pipe<PIPE_V>
// CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK: hivm.vf_mode = #hivm.vf_mode<SIMD>
// CHECK: sync_event_slots = [

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_custom_macro_op(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %empty = tensor.empty() : tensor<8xf32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %src_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %custom_result = hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "test_macro_sync_impl",
         sync_event_slots = [
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, wait, <EVENT_ID1>>
         ]}
        "test_macro_sync"
        ins(%src_addptr, %cst : tensor<8x!tt.ptr<f32>>, tensor<8xf32>)
        outs(%empty : tensor<8xf32>) -> tensor<8xf32>
    %dst_ptrs = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %custom_result : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Test hivm.hir.custom_macro with distributed tensor<tt.ptr> output.
// CHECK-LABEL: func.func @test_custom_macro_op_ptr_output
// CHECK: hivm.hir.custom_macro
// CHECK: hivm.is_distributed
// CHECK: hivm.pipe_in = #hivm.pipe<PIPE_MTE2>
// CHECK: hivm.pipe_out = #hivm.pipe<PIPE_V>
// CHECK: hivm.tcore_type = #hivm.tcore_type<VECTOR>
// CHECK: hivm.vf_mode = #hivm.vf_mode<SIMD>
// CHECK: memref

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_custom_macro_op_ptr_output(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %empty = tensor.empty() : tensor<8x!tt.ptr<f32>>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %custom_result = hivm.hir.custom_macro
        {hivm.is_distributed,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.vf_mode = #hivm.vf_mode<SIMD>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "test_macro_ptr_op_impl",
         SrcPtrIndex = array<i32: 0>}
        "test_macro_ptr_op"
        ins(%src_addptr : tensor<8x!tt.ptr<f32>>)
        outs(%empty : tensor<8x!tt.ptr<f32>>) -> tensor<8x!tt.ptr<f32>>
    %data = tt.load %custom_result : tensor<8x!tt.ptr<f32>>
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
