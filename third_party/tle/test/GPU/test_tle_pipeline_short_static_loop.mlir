// RUN: triton-opt %s -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @short_static_predicate
  // CHECK-NOT: scf.for
  // CHECK-NOT: ttg.predicate_stage
  // CHECK: %[[ABUFFER:.*]] = ttg.local_alloc
  // CHECK: %[[BBUFFER:.*]] = ttg.local_alloc
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: %[[WAIT:.*]] = ttg.async_wait {{.*}} {num = 0 : i32}
  // CHECK: %[[A0:.*]] = ttg.memdesc_index %[[ABUFFER]]
  // CHECK: %[[AV0:.*]] = ttg.local_load %[[A0]] token %[[WAIT]]
  // CHECK: %[[B0:.*]] = ttg.memdesc_index %[[BBUFFER]]
  // CHECK: %[[BV0:.*]] = ttg.local_load %[[B0]] token %[[WAIT]]
  // CHECK: %[[SUM0:.*]] = arith.addf %[[AV0]], %[[BV0]]
  // CHECK: tt.store
  // CHECK: %[[A1:.*]] = ttg.memdesc_index %[[ABUFFER]]
  // CHECK: %[[AV1:.*]] = ttg.local_load %[[A1]] token %[[WAIT]]
  // CHECK: %[[B1:.*]] = ttg.memdesc_index %[[BBUFFER]]
  // CHECK: %[[BV1:.*]] = ttg.local_load %[[B1]] token %[[WAIT]]
  // CHECK: %[[SUM1:.*]] = arith.addf %[[AV1]], %[[BV1]]
  // CHECK: tt.store
  tt.func public @short_static_predicate(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %1 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %iv = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %5 = tt.splat %iv : i32 -> tensor<1024xi32, #blocked>
      %6 = arith.addi %5, %0 : tensor<1024xi32, #blocked>
      %7 = arith.cmpi slt, %6, %1 : tensor<1024xi32, #blocked>
      %8 = tt.addptr %2, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %9 = tt.load %8, %7 : tensor<1024x!tt.ptr<f32>, #blocked>
      %10 = tt.addptr %3, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %11 = tt.load %10, %7 : tensor<1024x!tt.ptr<f32>, #blocked>
      %12 = arith.addf %9, %11 : tensor<1024xf32, #blocked>
      %13 = tt.addptr %4, %6 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %13, %12, %7 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32, __test_keep_predicate_stage}
    tt.return
  }
}
