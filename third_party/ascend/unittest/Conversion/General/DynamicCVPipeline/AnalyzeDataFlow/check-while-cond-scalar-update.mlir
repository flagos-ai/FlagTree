// RUN: triton-opt --analyze-while-condition-args %s | FileCheck %s

// The loop-carried args feeding the scf.condition are updated purely with
// scalars (memref.load of a scalar + arith). This is the supported pattern,
// so the fallback attr must NOT be set.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @while_cond_updated_by_scalar_load(%arg0: memref<?xi64>, %arg1: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %0:2 = scf.while (%arg2 = %c0_i64, %arg3 = %c0_i64) : (i64, i64) -> (i64, i64) {
      %1 = arith.addi %arg2, %arg3 : i64
      %2 = arith.cmpi sge, %arg1, %1 : i64
      scf.condition(%2) %arg2, %arg3 : i64, i64
    } do {
    ^bb0(%arg2: i64, %arg3: i64):
      %1 = arith.addi %arg2, %c1_i64 : i64
      %2 = arith.index_cast %1 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %3 = memref.load %reinterpret_cast[%c0] : memref<1xi64, strided<[1], offset: ?>>
      scf.yield %1, %3 : i64, i64
    } attributes {ssbuffer.main_loop = 1 : i64}
    return
  }
}

// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK: return
