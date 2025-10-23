// RUN: triton-adapter-opt --triton-to-linalg="named-ops=True" %s | FileCheck %s

// CHECK-LABEL: func @loadstorecanonicalizer_simple(
tt.func public @loadstorecanonicalizer_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
  %0 = tt.get_program_id x : i32
  %3 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
  %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  // CHECK: %[[CAST0:.*]] = memref.reinterpret_cast %[[ARG0:.*]]
  // CHECK: memref.copy %[[CAST0]], %[[ALLOC0:.*]]
  %5 = tt.load %4 : tensor<1x!tt.ptr<f32>>
  %6 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
  %7 = tt.splat %6 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  // CHECK: %[[CAST1:.*]] = memref.reinterpret_cast %[[ARG1:.*]]
  // CHECK: memref.copy %[[CAST1]], %[[ALLOC1:.*]]
  %8 = tt.load %7 : tensor<1x!tt.ptr<f32>>
  %9 = arith.addf %5, %8 : tensor<1xf32>
  %10 = tt.addptr %arg2, %0 : !tt.ptr<f32>, i32
  %11 = tt.splat %10 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  // CHECK: %[[CAST2:.*]] = memref.reinterpret_cast %[[ARG2:.*]] to offset
  // CHECK: bufferization.materialize_in_destination %[[VAL:.*]] in writable %[[CAST2]]
  tt.store %11, %9 : tensor<1x!tt.ptr<f32>>
  tt.return
}

