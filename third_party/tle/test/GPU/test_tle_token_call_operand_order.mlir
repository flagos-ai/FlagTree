// RUN: triton-opt %s --nvgpu-test-ws-lower-token | FileCheck %s

// Creation/traversal order is not the callee parameter order. This exercises
// two token arguments whose concrete instances are declared in opposite order.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func private @role
  // CHECK-NOT: !nvws.token
  // CHECK-SAME: !ttg.memdesc<2x1xi64
  // CHECK-SAME: !ttg.memdesc<4x1xi64
  tt.func private @role(%a: tensor<2x!nvws.token>, %b: tensor<4x!nvws.token>, %idx: i32) attributes {noinline = true} {
    nvws.producer_commit %a, %idx {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32
    nvws.producer_commit %b, %idx {async_task_id = array<i32: 0>} : tensor<4x!nvws.token>, i32
    tt.return
  }
  // CHECK-LABEL: tt.func @kernel
  tt.func @kernel(%idx: i32) {
    %a1 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %b1 = nvws.create_token {loadType = 3 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %b2 = nvws.create_token {loadType = 3 : i32, numBuffers = 4 : i32} : tensor<4x!nvws.token>
    %a2 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    // CHECK: tt.call @role({{.*}}) : (i32, !ttg.memdesc<2x1xi64, {{.*}}, !ttg.memdesc<4x1xi64, {{.*}}) -> ()
    tt.call @role(%a1, %b1, %idx) : (tensor<2x!nvws.token>, tensor<4x!nvws.token>, i32) -> ()
    // CHECK: tt.call @role({{.*}}) : (i32, !ttg.memdesc<2x1xi64, {{.*}}, !ttg.memdesc<4x1xi64, {{.*}}) -> ()
    tt.call @role(%a2, %b2, %idx) : (tensor<2x!nvws.token>, tensor<4x!nvws.token>, i32) -> ()
    tt.return
  }
}
