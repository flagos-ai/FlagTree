// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_parse_atomic_rmw
// CHECK: %[[C8_I32:.*]] = arith.constant 8 : i32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[C8_I32]] : i32)
// CHECK: hivm.hir.store ins(%[[FILL]] : tensor<1xi32>) outs(%{{.*}} : memref<1xi32, strided<[1]>>) atomic = <add>
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<1xi32>
// CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[EXTRACTED]] : i32 to index
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[IDX_CAST]]], sizes: [8], strides: [1]


module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_parse_atomic_rmw(%counter_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %src_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %numel: i32) attributes {noinline = false} {
    %zero_i32 = arith.constant dense<0> : tensor<8xi32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %numel_splat = tt.splat %numel : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offs_full, %numel_splat : tensor<8xi32>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %data = tt.load %src_addptr, %mask, %zero_i32 : tensor<8x!tt.ptr<i32>>

    // Tensor-level atomic: tensor<1>
    %ctr_range = tt.make_range {start = 0 : i32, end = 1 : i32} : tensor<1xi32>
    %ctr_ptrs = tt.splat %counter_ptr : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %ctr_addptr = tt.addptr %ctr_ptrs, %ctr_range : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %incr = arith.constant dense<8> : tensor<1xi32>
    %old = tt.atomic_rmw add, acq_rel, gpu, %ctr_addptr, %incr : (tensor<1x!tt.ptr<i32>>, tensor<1xi32>) -> tensor<1xi32>
    %old_broadcast = tt.broadcast %old : tensor<1xi32> -> tensor<8xi32>
    %write_offs = arith.addi %old_broadcast, %offs : tensor<8xi32>

    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %dst_addptr = tt.addptr %dst_ptrs, %write_offs : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %dst_addptr, %data, %mask : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}
