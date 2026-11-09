// RUN: triton-opt --discrete-mask-access-conversion --triton-to-unstructure --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @isin_by_search_kernel
// The one-element loop mask controls the load itself, including in SIMD mode.
// CHECK: %[[FOR_VAR:.*]]:4 = scf.for
// CHECK-SAME: iter_args({{.*}}, %[[MASK:.*]] = %{{.*}}) -> (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<1xi1>) : i32
// CHECK: %[[PREDICATE:.*]] = tensor.extract %[[MASK]][{{.*}}] {{.*}} : tensor<1xi1>
// CHECK: %[[VALUE:.*]] = scf.if %[[PREDICATE]] -> (i16) {
// CHECK-NEXT: %[[LOAD:.*]] = memref.load
// CHECK-NEXT: scf.yield %[[LOAD]] : i16
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield %{{.*}} : i16
module {
  tt.func public @isin_by_search_kernel(
    %arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32},
    %arg3: i32 {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0> : tensor<1xi32>
    %cst_0 = arith.constant dense<2> : tensor<1xi32>
    %cst_1 = arith.constant dense<1> : tensor<1xi32>
    %cst_2 = arith.constant dense<0> : tensor<1xi64>
    %c0_i32 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant dense<0> : tensor<1xi16>
    %cst_4 = arith.constant dense<false> : tensor<1xi1>
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.cmpi slt, %1, %c1_i64 : i64
    %3 = tt.splat %2 : i1 -> tensor<1xi1>
    %4 = tt.addptr %arg0, %1 : !tt.ptr<i64>, i64
    %5 = tt.splat %4 : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %6 = tt.load %5, %3, %cst_2 : tensor<1x!tt.ptr<i64>>
    %7 = tt.splat %arg3 : i32 -> tensor<1xi32>
    %8 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    %9 = tt.splat %8 : i1 -> tensor<1xi1>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x!tt.ptr<i16>>
    %11:4 = scf.for %arg4 = %c0_i32 to %c20_i32 step %c1_i32 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %7, %arg8 = %9) -> (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<1xi1>)  : i32 {
      %16 = arith.subi %arg7, %arg6 : tensor<1xi32>
      %17 = arith.divsi %16, %cst_0 : tensor<1xi32>
      %18 = arith.addi %arg6, %17 : tensor<1xi32>
      %19 = arith.select %arg8, %18, %cst : tensor<1xi1>, tensor<1xi32>
      %20 = tt.addptr %10, %19 : tensor<1x!tt.ptr<i16>>, tensor<1xi32>
      %21 = tt.load %20, %arg8, %cst_3 : tensor<1x!tt.ptr<i16>>
      %22 = arith.extsi %21 : tensor<1xi16> to tensor<1xi64>
      %23 = arith.cmpi eq, %22, %6 : tensor<1xi64>
      %24 = arith.ori %cst_4, %23 : tensor<1xi1>
      %25 = arith.select %arg8, %24, %arg5 : tensor<1xi1>, tensor<1xi1>
      %26 = arith.cmpi slt, %22, %6 : tensor<1xi64>
      %27 = arith.andi %arg8, %26 : tensor<1xi1>
      %28 = arith.addi %19, %cst_1 : tensor<1xi32>
      %29 = arith.select %27, %28, %arg6 : tensor<1xi1>, tensor<1xi32>
      %30 = arith.cmpi sgt, %22, %6 : tensor<1xi64>
      %31 = arith.andi %arg8, %30 : tensor<1xi1>
      %32 = arith.select %31, %19, %arg7 : tensor<1xi1>, tensor<1xi32>
      %33 = arith.cmpi slt, %29, %32 : tensor<1xi32>
      scf.yield %25, %29, %32, %33 : tensor<1xi1>, tensor<1xi32>, tensor<1xi32>, tensor<1xi1>
    }
    %12 = tt.addptr %arg2, %1 : !tt.ptr<i1>, i64
    %13 = tt.bitcast %12 : !tt.ptr<i1> -> !tt.ptr<i8>
    %14 = tt.splat %13 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %15 = arith.extui %11#0 : tensor<1xi1> to tensor<1xi8>
    tt.store %14, %15, %3 : tensor<1x!tt.ptr<i8>>
    tt.return
  }
}
