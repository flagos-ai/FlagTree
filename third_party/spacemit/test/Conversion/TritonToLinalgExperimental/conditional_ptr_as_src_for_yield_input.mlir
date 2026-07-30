// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module attributes {} {
  tt.func public @gather_kernel(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>, %arg3: i32 , %arg4: i32 , %arg5: i32 , %arg6: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %2 = arith.muli %0, %arg6 : i32
    %3 = tt.addptr %arg1, %2 : !tt.ptr<i64>, i32
    %4 = tt.splat %3 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %5 = tt.addptr %4, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %7 = arith.cmpi slt, %1, %6 : tensor<512xi32>
    %8 = tt.load %5, %7 : tensor<512x!tt.ptr<i64>>
    %9 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %10 = scf.if %9 -> (tensor<512x!tt.ptr<i64>>) {
      %15 = arith.extsi %arg5 : i32 to i64
      %16 = tt.splat %15 : i64 -> tensor<512xi64>
      %17 = arith.muli %8, %16 : tensor<512xi64>
      %18 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %19 = tt.addptr %18, %17 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      %20 = tt.addptr %19, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
      scf.yield %20 : tensor<512x!tt.ptr<i64>>
    } else {
      %15 = arith.muli %0, %arg5 : i32
      %16 = tt.addptr %arg0, %15 : !tt.ptr<i64>, i32
      %17 = tt.splat %16 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %18 = tt.addptr %17, %8 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      scf.yield %18 : tensor<512x!tt.ptr<i64>>
    }
    %11 = tt.load %10, %7 : tensor<512x!tt.ptr<i64>>
    %12 = tt.addptr %arg2, %2 : !tt.ptr<i64>, i32
    %13 = tt.splat %12 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %14 = tt.addptr %13, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    tt.store %14, %11, %7 : tensor<512x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: tt.
