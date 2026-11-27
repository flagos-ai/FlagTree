// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @test_while_loop_index_tensor
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: scf.while (%[[ARG9:.*]] = %[[C0_I32]], %[[ARG10:.*]] = %[[C0]])
// CHECK: arith.cmpi slt, %[[ARG9]], %[[C4_I32]]
// CHECK: scf.condition
// CHECK: ^bb0(%[[ARG9:.*]]: i32, %[[ARG10:.*]]: index)
// CHECK: arith.addi %[[ARG9]], %[[C1_I32]]
// CHECK: arith.addi %[[ARG10]], %[[C8]]

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_while_loop_index_tensor(
      %src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %numel: i32
  ) attributes {noinline = false} {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %init_offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %step = arith.constant dense<8> : tensor<8xi32>
    %result:2 = scf.while (%offs = %init_offs, %iter = %c0_i32) : (tensor<8xi32>, i32) -> (tensor<8xi32>, i32) {
      %cond = arith.cmpi slt, %iter, %c4_i32 : i32
      scf.condition(%cond) %offs, %iter : tensor<8xi32>, i32
    } do {
    ^bb0(%offs_arg: tensor<8xi32>, %iter_arg: i32):
      %ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %addptr = tt.addptr %ptrs, %offs_arg : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      %numel_splat = tt.splat %numel : i32 -> tensor<8xi32>
      %mask = arith.cmpi slt, %offs_arg, %numel_splat : tensor<8xi32>
      %data = tt.load %addptr, %mask, %zero_f32 : tensor<8x!tt.ptr<f32>>
      %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
      %dst_addptr = tt.addptr %dst_ptrs, %offs_arg : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
      tt.store %dst_addptr, %data, %mask : tensor<8x!tt.ptr<f32>>
      %new_offs = arith.addi %offs_arg, %step : tensor<8xi32>
      %new_iter = arith.addi %iter_arg, %c1_i32 : i32
      scf.yield %new_offs, %new_iter : tensor<8xi32>, i32
    }
    tt.return
  }
}
