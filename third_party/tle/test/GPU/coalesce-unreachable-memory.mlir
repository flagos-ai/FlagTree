// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s
// Copyright 2026 FlagOS Contributors
// A dead block deliberately has bottom AxisInfo. It must not be a coalescing
// root or contribute a same-order memory dependency. Live paths still vectorize.
// CHECK-DAG: #[[$VEC:[[:alnum:]_]+]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @unreachable_memory
  // CHECK: scf.if
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @unreachable_memory(%base: !tt.ptr<f32>) {
    %false = arith.constant false
    scf.if %false {
      %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %ptr = tt.addptr %splat, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %value = tt.load %ptr : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %ptr, %value : tensor<128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }

  // This dead load shares the live address chain and is found by slice closure.
  // CHECK-LABEL: tt.func @live_with_dead_dependency
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: scf.if
  // CHECK: tt.load
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: tt.return
  tt.func @live_with_dead_dependency(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %false = arith.constant false
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %value = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked>
    scf.if %false {
      %dead_ptr = tt.addptr %splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %dead_value = tt.load %dead_ptr : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.store %dead_ptr, %dead_value : tensor<1024x!tt.ptr<f32>, #blocked>
    }
    tt.store %ptr, %value : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // A dynamic condition leaves both branches executable.
  // CHECK-LABEL: tt.func @dynamic_live_memory
  // CHECK: scf.if
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: else
  // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, #[[$VEC]]>
  // CHECK: tt.return
  tt.func @dynamic_live_memory(%base: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %condition: i1) {
    scf.if %condition {
      %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      %ptr = tt.addptr %splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %value = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.store %ptr, %value : tensor<1024x!tt.ptr<f32>, #blocked>
    } else {
      %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #blocked>
      %splat = tt.splat %base : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
      %ptr = tt.addptr %splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %value = tt.load %ptr : tensor<1024x!tt.ptr<f32>, #blocked>
      tt.store %ptr, %value : tensor<1024x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}
