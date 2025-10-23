// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @auto_gen_kernel_01(%arg0: !tt.ptr<f64>, %arg1: !tt.ptr<f64>) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<512> : tensor<1xi64>
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<[4, 4, 32]> : tensor<3xi64>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg1 : !tt.ptr<f64> -> tensor<512x!tt.ptr<f64>>
    %6 = tt.addptr %5, %4 : tensor<512x!tt.ptr<f64>>, tensor<512xi32>
    %7 = tt.load %6 evictionPolicy = evict_last : tensor<512x!tt.ptr<f64>>
// CHECK:           %[[VAL_2:.*]] = tensor.reshape %[[VAL_0:.*]](%[[VAL_1:.*]]) : (tensor<512xf64>, tensor<3xi64>) -> tensor<4x4x32xf64>
    %8 = tt.reshape %7 : tensor<512xf64> -> tensor<4x4x32xf64>
    %9 = arith.addf %8, %8 : tensor<4x4x32xf64>
// CHECK:           %[[VAL_7:.*]] = tensor.reshape %[[VAL_3:.*]](%[[VAL_4]]) : (tensor<4x4x32xf64>, tensor<1xi64>) -> tensor<512xf64>
    %10 = tt.reshape %9 : tensor<4x4x32xf64> -> tensor<512xf64>
    %11 = tt.splat %arg0 : !tt.ptr<f64> -> tensor<512x!tt.ptr<f64>>
    %12 = tt.addptr %11, %4 : tensor<512x!tt.ptr<f64>>, tensor<512xi32>
    tt.store %12, %10 evictionPolicy = evict_last : tensor<512x!tt.ptr<f64>>
    tt.return
  }
}
