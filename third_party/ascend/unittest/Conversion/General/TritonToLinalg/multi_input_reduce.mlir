// RUN: triton-opt --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @three_input_reduce
// CHECK-NOT: linalg.reduce
// CHECK: scf.for
// CHECK: scf.yield
// CHECK-NOT: linalg.reduce
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910_9589">} {
  tt.func public @three_input_reduce(%a_ptr: !tt.ptr<f32>, %b_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %a_ptrs = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %b_ptrs = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %c_ptrs = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %a_addrs = tt.addptr %a_ptrs, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %b_addrs = tt.addptr %b_ptrs, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %c_addrs = tt.addptr %c_ptrs, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %a = tt.load %a_addrs : tensor<4x!tt.ptr<f32>>
    %b = tt.load %b_addrs : tensor<4x!tt.ptr<f32>>
    %c = tt.load %c_addrs : tensor<4x!tt.ptr<f32>>
    %0:3 = "tt.reduce"(%a, %b, %c) <{axis = 0 : i32}> ({
    ^bb0(%a0: f32, %b0: f32, %c0: f32, %a1: f32, %b1: f32, %c1: f32):
      %a_sum = arith.addf %a0, %a1 : f32
      %b_sum = arith.addf %b0, %b1 : f32
      %c_sum = arith.addf %c0, %c1 : f32
      tt.reduce.return %a_sum, %b_sum, %c_sum : f32, f32, f32
    }) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> (f32, f32, f32)
    tt.store %out_ptr, %0#2 : !tt.ptr<f32>
    tt.return
  }
}
