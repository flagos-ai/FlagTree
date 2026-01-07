// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// -----
// op: logical_and, dtype: f8E5M2

module {
  tt.func public @triton_logical_and(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>, %arg2: !tt.ptr<i1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf8E5M2>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E5M2>>
    %8 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f8E5M2>>
    %11 = arith.cmpf une, %7, %cst : tensor<4xf8E5M2>
    %12 = arith.cmpf une, %10, %cst : tensor<4xf8E5M2>
    %13 = arith.andi %11, %12 : tensor<4xi1>
    %14 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %15 = tt.addptr %14, %4 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %16 = tt.bitcast %15 : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %17 = arith.extui %13 : tensor<4xi1> to tensor<4xi8>
    tt.store %16, %17 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_logical_and
// CHECK:       %[[C0:.*]] = arith.constant 0.000000e+00 : f8E5M2
// CHECK:       %[[TENSOR_C0:.*]] = linalg.fill ins(%[[C0]] : f8E5M2) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xf8E5M2>) -> tensor<4xf8E5M2>
// CHECK:       %[[XNE0:.*]] = arith.cmpf une, %[[TENSOR_X:.*]], %[[TENSOR_C0]] : tensor<4xf8E5M2>
// CHECK:       %[[YNE0:.*]] = arith.cmpf une, %[[TENSOR_Y:.*]], %[[TENSOR_C0]] : tensor<4xf8E5M2>
// CHECK:       %[[RES:.*]] = arith.andi %[[XNE0]], %[[YNE0]] : tensor<4xi1>

// -----
// op: logical_and, dtype: f8E4M3FN

module {
  tt.func public @triton_logical_and(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<i1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf8E4M3FN>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E4M3FN>>
    %8 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f8E4M3FN>>
    %11 = arith.cmpf une, %7, %cst : tensor<4xf8E4M3FN>
    %12 = arith.cmpf une, %10, %cst : tensor<4xf8E4M3FN>
    %13 = arith.andi %11, %12 : tensor<4xi1>
    %14 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %15 = tt.addptr %14, %4 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %16 = tt.bitcast %15 : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %17 = arith.extui %13 : tensor<4xi1> to tensor<4xi8>
    tt.store %16, %17 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_logical_and
// CHECK:       %[[C0:.*]] = arith.constant 0.000000e+00 : f8E4M3FN
// CHECK:       %[[TENSOR_C0:.*]] = linalg.fill ins(%[[C0]] : f8E4M3FN) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
// CHECK:       %[[XNE0:.*]] = arith.cmpf une, %[[TENSOR_X:.*]], %[[TENSOR_C0]] : tensor<4xf8E4M3FN>
// CHECK:       %[[YNE0:.*]] = arith.cmpf une, %[[TENSOR_Y:.*]], %[[TENSOR_C0]] : tensor<4xf8E4M3FN>
// CHECK:       %[[RES:.*]] = arith.andi %[[XNE0]], %[[YNE0]] : tensor<4xi1>

// -----
// op: logical_or, dtype: f8E5M2

module {
  tt.func public @triton_logical_or(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>, %arg2: !tt.ptr<i1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf8E5M2>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E5M2>>
    %8 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f8E5M2>>
    %11 = arith.cmpf une, %7, %cst : tensor<4xf8E5M2>
    %12 = arith.cmpf une, %10, %cst : tensor<4xf8E5M2>
    %13 = arith.ori %11, %12 : tensor<4xi1>
    %14 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %15 = tt.addptr %14, %4 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %16 = tt.bitcast %15 : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %17 = arith.extui %13 : tensor<4xi1> to tensor<4xi8>
    tt.store %16, %17 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_logical_or
// CHECK:       %[[C0:.*]] = arith.constant 0.000000e+00 : f8E5M2
// CHECK:       %[[TENSOR_C0:.*]] = linalg.fill ins(%[[C0]] : f8E5M2) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xf8E5M2>) -> tensor<4xf8E5M2>
// CHECK:       %[[XNE0:.*]] = arith.cmpf une, %[[TENSOR_X:.*]], %[[TENSOR_C0]] : tensor<4xf8E5M2>
// CHECK:       %[[YNE0:.*]] = arith.cmpf une, %[[TENSOR_Y:.*]], %[[TENSOR_C0]] : tensor<4xf8E5M2>
// CHECK:       %[[RES:.*]] = arith.ori %[[XNE0]], %[[YNE0]] : tensor<4xi1>

// -----
// op: logical_or, dtype: f8E4M3FN

module {
  tt.func public @triton_logical_or(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<i1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf8E4M3FN>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E4M3FN>>
    %8 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<f8E4M3FN>>
    %11 = arith.cmpf une, %7, %cst : tensor<4xf8E4M3FN>
    %12 = arith.cmpf une, %10, %cst : tensor<4xf8E4M3FN>
    %13 = arith.ori %11, %12 : tensor<4xi1>
    %14 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %15 = tt.addptr %14, %4 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %16 = tt.bitcast %15 : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %17 = arith.extui %13 : tensor<4xi1> to tensor<4xi8>
    tt.store %16, %17 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_logical_or
// CHECK:       %[[C0:.*]] = arith.constant 0.000000e+00 : f8E4M3FN
// CHECK:       %[[TENSOR_C0:.*]] = linalg.fill ins(%[[C0]] : f8E4M3FN) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
// CHECK:       %[[XNE0:.*]] = arith.cmpf une, %[[TENSOR_X:.*]], %[[TENSOR_C0]] : tensor<4xf8E4M3FN>
// CHECK:       %[[YNE0:.*]] = arith.cmpf une, %[[TENSOR_Y:.*]], %[[TENSOR_C0]] : tensor<4xf8E4M3FN>
// CHECK:       %[[RES:.*]] = arith.ori %[[XNE0]], %[[YNE0]] : tensor<4xi1>