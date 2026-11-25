// RUN: triton-opt --split-input-file %s -triton-loop-vectorize | FileCheck %s --implicit-check-not=tt.loop_vectorize_factor

// Independent stores have no loop-carried results. Widen the memory accesses
// in the main loop, then retain scalar accesses for the remaining iterations.
// CHECK-LABEL: @store_values
tt.func @store_values(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: scf.for
  // CHECK: tt.make_range {{.*}} : tensor<4xi32>
  // CHECK: %[[VALUES:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.store {{.*}}, %[[VALUES]] : tensor<4x!tt.ptr<f32>>
  // CHECK: scf.for
  // CHECK: %[[VALUE:.*]] = tt.load {{.*}} : !tt.ptr<f32>
  // CHECK: tt.store {{.*}}, %[[VALUE]] : !tt.ptr<f32>
  scf.for %i = %c0 to %N step %c1 : i32 {
    %src = tt.addptr %X, %i : !tt.ptr<f32>, i32
    %value = tt.load %src : !tt.ptr<f32>
    %dst = tt.addptr %Y, %i : !tt.ptr<f32>, i32
    tt.store %dst, %value : !tt.ptr<f32>
  } {tt.loop_vectorize_factor = 4 : i32}
  tt.return
}

// -----

// Hoist a loop-local invariant condition while keeping the condition scalar.
// The side-effecting branch remains an scf.if in both main and tail loops.
// CHECK-LABEL: @conditional_copy
tt.func @conditional_copy(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %N: i32, %MODE: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[ENABLED:.*]] = arith.cmpi sgt, %arg3, %c0_i32 : i32
  // CHECK: scf.for
  // CHECK-NOT: arith.cmpi sgt
  // CHECK: scf.if %[[ENABLED]] {
  // CHECK: %[[VALUES:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.store {{.*}}, %[[VALUES]] : tensor<4x!tt.ptr<f32>>
  // CHECK: scf.for
  // CHECK: %[[TAIL_ENABLED:.*]] = arith.cmpi sgt, %arg3, %c0_i32 : i32
  // CHECK: scf.if %[[TAIL_ENABLED]] {
  // CHECK: %[[VALUE:.*]] = tt.load {{.*}} : !tt.ptr<f32>
  // CHECK: tt.store {{.*}}, %[[VALUE]] : !tt.ptr<f32>
  scf.for %i = %c0 to %N step %c1 : i32 {
    %enabled = arith.cmpi sgt, %MODE, %c0 : i32
    scf.if %enabled {
      %src = tt.addptr %X, %i : !tt.ptr<f32>, i32
      %value = tt.load %src : !tt.ptr<f32>
      %dst = tt.addptr %Y, %i : !tt.ptr<f32>, i32
      tt.store %dst, %value : !tt.ptr<f32>
    }
  } {tt.loop_vectorize_factor = 4 : i32}
  tt.return
}

// -----

// Extract the final lane once after the vector loop, and pass it to the scalar
// tail. If no full group executes, the tail must receive INITIAL unchanged.
// CHECK-LABEL: @last_value
tt.func @last_value(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %N: i32, %INITIAL: f32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[INITIAL:.*]] = tt.splat %arg3 : f32 -> tensor<4xf32>
  // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[INITIAL]]) -> (tensor<4xf32>)
  // CHECK: %[[VALUES:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<f32>>
  // CHECK: scf.yield %[[VALUES]] : tensor<4xf32>
  // CHECK: %[[MAIN_RESULT:.*]] = scf.if {{.*}} -> (f32)
  // CHECK: arith.constant dense<3> : tensor<1xi32>
  // CHECK: %[[LAST:.*]] = tt.gather {{.*}} {axis = 0 : i32}
  // CHECK: %[[SCALAR:.*]] = tt.unsplat %[[LAST]] : tensor<1xf32>
  // CHECK: scf.yield %[[SCALAR]] : f32
  // CHECK: } else {
  // CHECK: scf.yield %arg3 : f32
  // CHECK: %[[TAIL:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[MAIN_RESULT]]) -> (f32)
  // CHECK: %[[VALUE:.*]] = tt.load {{.*}} : !tt.ptr<f32>
  // CHECK: scf.yield %[[VALUE]] : f32
  // CHECK: tt.store %arg1, %[[TAIL]] : !tt.ptr<f32>
  %result = scf.for %i = %c0 to %N step %c1 iter_args(%last = %INITIAL) -> (f32) : i32 {
    %src = tt.addptr %X, %i : !tt.ptr<f32>, i32
    %value = tt.load %src : !tt.ptr<f32>
    scf.yield %value : f32
  } {tt.loop_vectorize_factor = 4 : i32}
  tt.store %Y, %result : !tt.ptr<f32>
  tt.return
}

// -----

// Include INITIAL in only one lane, combine the partial sums once, and use
// that sum to initialize the scalar tail. Empty loops preserve INITIAL.
// CHECK-LABEL: @sum_values
tt.func @sum_values(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %N: i32, %INITIAL: f32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: arith.constant dense<-0.000000e+00> : tensor<4xf32>
  // CHECK: %[[INITIAL:.*]] = arith.select {{.*}} : tensor<4xi1>, tensor<4xf32>
  // CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INITIAL]]) -> (tensor<4xf32>)
  // CHECK: %[[VALUES:.*]] = tt.load {{.*}} : tensor<4x!tt.ptr<f32>>
  // CHECK: %[[SUM:.*]] = arith.addf %[[ACC]], %[[VALUES]] : tensor<4xf32>
  // CHECK: scf.yield %[[SUM]] : tensor<4xf32>
  // CHECK: %[[MAIN_RESULT:.*]] = scf.if {{.*}} -> (f32)
  // CHECK: %[[REDUCED:.*]] = "tt.reduce"({{.*}}) <{axis = 0 : i32}>
  // CHECK: tt.reduce.return
  // CHECK: scf.yield %[[REDUCED]] : f32
  // CHECK: } else {
  // CHECK: scf.yield %arg3 : f32
  // CHECK: %[[TAIL:.*]] = scf.for {{.*}} iter_args(%[[SCALAR_ACC:.*]] = %[[MAIN_RESULT]]) -> (f32)
  // CHECK: %[[VALUE:.*]] = tt.load {{.*}} : !tt.ptr<f32>
  // CHECK: %[[SCALAR_SUM:.*]] = arith.addf %[[SCALAR_ACC]], %[[VALUE]] : f32
  // CHECK: scf.yield %[[SCALAR_SUM]] : f32
  // CHECK: tt.store %arg1, %[[TAIL]] : !tt.ptr<f32>
  %result = scf.for %i = %c0 to %N step %c1 iter_args(%acc = %INITIAL) -> (f32) : i32 {
    %src = tt.addptr %X, %i : !tt.ptr<f32>, i32
    %value = tt.load %src : !tt.ptr<f32>
    %sum = arith.addf %acc, %value : f32
    scf.yield %sum : f32
  } {tt.loop_vectorize_factor = 4 : i32}
  tt.store %Y, %result : !tt.ptr<f32>
  tt.return
}
