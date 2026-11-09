// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

module {
tt.func public @copy_all_layer_kv_cache2(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0> : tensor<1x128xi8>
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c127_i64 = arith.constant 127 : i64
  %true = arith.constant true
  %c128_i64 = arith.constant 128 : i64

  %0 = tt.get_program_id x : i32
  %1 = tt.addptr %arg1, %0 : !tt.ptr<i64>, i32
  %2 = tt.load %1 : !tt.ptr<i64>
  %3 = tt.addptr %arg0, %0 : !tt.ptr<i64>, i32
  %4 = tt.load %3 : !tt.ptr<i64>
  %5 = tt.int_to_ptr %4 : i64 -> !tt.ptr<i8>

  %6 = tt.addptr %arg2, %c0_i32 : !tt.ptr<i32>, i32
  %7 = tt.splat %6 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
  %8 = tt.load %7 : tensor<1x!tt.ptr<i32>>

  %9 = tt.addptr %arg3, %c0_i32 : !tt.ptr<i32>, i32
  %10 = tt.splat %9 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
  %11 = tt.load %10 : tensor<1x!tt.ptr<i32>>

  tt.assert %true, "int32 overflow detected for operation sub" : i1

  %12 = arith.addi %2, %c127_i64 : i64
  %13 = arith.divsi %12, %c128_i64 : i64

  %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %15 = arith.extsi %14 : tensor<128xi32> to tensor<128xi64>
  %16 = tt.splat %2 : i64 -> tensor<128xi64>

  %17 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
  %18 = arith.extsi %17 : tensor<1x1xi32> to tensor<1x1xi64>
  %19 = tt.splat %2 : i64 -> tensor<1x1xi64>
  %20 = arith.muli %18, %19 : tensor<1x1xi64>
  %21 = tt.broadcast %20 : tensor<1x1xi64> -> tensor<1x128xi64>
  %22 = tt.splat %5 : !tt.ptr<i8> -> tensor<1x128x!tt.ptr<i8>>

  %23 = tt.expand_dims %8 {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
  %24 = arith.extsi %23 : tensor<1x1xi32> to tensor<1x1xi64>
  %25 = arith.muli %24, %19 : tensor<1x1xi64>
  %26 = tt.broadcast %25 : tensor<1x1xi64> -> tensor<1x128xi64>

  scf.for %arg4 = %c0_i64 to %13 step %c1_i64 : i64 {
    %27 = arith.muli %arg4, %c128_i64 : i64
    %28 = tt.splat %27 : i64 -> tensor<128xi64>
    %29 = arith.addi %15, %28 : tensor<128xi64>
    %30 = arith.cmpi slt, %29, %16 : tensor<128xi64>
    %31 = tt.expand_dims %30 {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
    %32 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
    %33 = arith.addi %21, %32 : tensor<1x128xi64>
    %34 = tt.addptr %22, %33 : tensor<1x128x!tt.ptr<i8>>, tensor<1x128xi64>
    %35 = tt.load %34, %31, %cst : tensor<1x128x!tt.ptr<i8>>
    %36 = arith.addi %26, %32 : tensor<1x128xi64>
    %37 = tt.addptr %22, %36 : tensor<1x128x!tt.ptr<i8>>, tensor<1x128xi64>
    tt.store %37, %35, %31 : tensor<1x128x!tt.ptr<i8>>
  }

  tt.return
}
}

// CHECK-LABEL: func.func @copy_all_layer_kv_cache2
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: scf.for
// CHECK: linalg.fill
// CHECK: memref.subview
// CHECK: bufferization.materialize_in_destination
