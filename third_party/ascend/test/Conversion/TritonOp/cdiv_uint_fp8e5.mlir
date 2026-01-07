// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// -----
// op: cdiv, dtype: uint8

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>) {
    %cst = arith.constant dense<1> : tensor<4xi64>
    %cst_0 = arith.constant dense<0> : tensor<4xi64>
    %cst_1 = arith.constant dense<255> : tensor<4xi64>
    %cst_2 = arith.constant dense<1> : tensor<4xi8>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = arith.extui %10 : tensor<4xi8> to tensor<4xi64>
    %12 = arith.subi %11, %cst : tensor<4xi64>
    %13 = arith.cmpi sle, %12, %cst_1 : tensor<4xi64>
    %14 = arith.cmpi sge, %12, %cst_0 : tensor<4xi64>
    %15 = arith.andi %13, %14 : tensor<4xi1>
    tt.assert %15, "int8 overflow detected for operation sub" : tensor<4xi1>
    %16 = arith.subi %10, %cst_2 : tensor<4xi8>
    %17 = arith.extui %7 : tensor<4xi8> to tensor<4xi64>
    %18 = arith.extui %16 : tensor<4xi8> to tensor<4xi64>
    %19 = arith.addi %17, %18 : tensor<4xi64>
    %20 = arith.cmpi sle, %19, %cst_1 : tensor<4xi64>
    %21 = arith.cmpi sge, %19, %cst_0 : tensor<4xi64>
    %22 = arith.andi %20, %21 : tensor<4xi1>
    tt.assert %22, "int8 overflow detected for operation add" : tensor<4xi1>
    %23 = arith.addi %7, %16 : tensor<4xi8>
    %24 = arith.divui %23, %10 : tensor<4xi8>
    %25 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %26 = tt.addptr %25, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    tt.store %26, %24 : tensor<4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[TENSOR_C1:.*]] = linalg.fill ins(%[[CONST_C1:.*]] : i8) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xi8>) -> tensor<4xi8>
// CHECK:       %[[TENSOR_SUB:.*]] = arith.subi %[[TENSOR_DIV:.*]], %[[TENSOR_C1]] : tensor<4xi8>
// CHECK:       %[[TENSOR_ADD:.*]] = arith.addi %[[TENSOR_X:.*]], %[[TENSOR_SUB]] : tensor<4xi8>
// CHECK:       %[[TENSOR_RES:.*]] = arith.divui %[[TENSOR_ADD]], %[[TENSOR_DIV]] : tensor<4xi8>

// -----
// op: cdiv, dtype: uint16

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>, %arg2: !tt.ptr<i16>) {
    %cst = arith.constant dense<1> : tensor<4xi64>
    %cst_0 = arith.constant dense<0> : tensor<4xi64>
    %cst_1 = arith.constant dense<65535> : tensor<4xi64>
    %cst_2 = arith.constant dense<1> : tensor<4xi16>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i16>>
    %11 = arith.extui %10 : tensor<4xi16> to tensor<4xi64>
    %12 = arith.subi %11, %cst : tensor<4xi64>
    %13 = arith.cmpi sle, %12, %cst_1 : tensor<4xi64>
    %14 = arith.cmpi sge, %12, %cst_0 : tensor<4xi64>
    %15 = arith.andi %13, %14 : tensor<4xi1>
    tt.assert %15, "int16 overflow detected for operation sub" : tensor<4xi1>
    %16 = arith.subi %10, %cst_2 : tensor<4xi16>
    %17 = arith.extui %7 : tensor<4xi16> to tensor<4xi64>
    %18 = arith.extui %16 : tensor<4xi16> to tensor<4xi64>
    %19 = arith.addi %17, %18 : tensor<4xi64>
    %20 = arith.cmpi sle, %19, %cst_1 : tensor<4xi64>
    %21 = arith.cmpi sge, %19, %cst_0 : tensor<4xi64>
    %22 = arith.andi %20, %21 : tensor<4xi1>
    tt.assert %22, "int16 overflow detected for operation add" : tensor<4xi1>
    %23 = arith.addi %7, %16 : tensor<4xi16>
    %24 = arith.divui %23, %10 : tensor<4xi16>
    %25 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %26 = tt.addptr %25, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    tt.store %26, %24 : tensor<4x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[TENSOR_C1:.*]] = linalg.fill ins(%[[CONST_C1:.*]] : i16) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xi16>) -> tensor<4xi16>
// CHECK:       %[[TENSOR_SUB:.*]] = arith.subi %[[TENSOR_DIV:.*]], %[[TENSOR_C1]] : tensor<4xi16>
// CHECK:       %[[TENSOR_ADD:.*]] = arith.addi %[[TENSOR_X:.*]], %[[TENSOR_SUB]] : tensor<4xi16>
// CHECK:       %[[TENSOR_RES:.*]] = arith.divui %[[TENSOR_ADD]], %[[TENSOR_DIV]] : tensor<4xi16>

// -----
// op: cdiv, dtype: uint32

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
    %cst = arith.constant dense<1> : tensor<4xi64>
    %cst_0 = arith.constant dense<0> : tensor<4xi64>
    %cst_1 = arith.constant dense<4294967295> : tensor<4xi64>
    %cst_2 = arith.constant dense<1> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i32>>
    %11 = arith.extui %10 : tensor<4xi32> to tensor<4xi64>
    %12 = arith.subi %11, %cst : tensor<4xi64>
    %13 = arith.cmpi sle, %12, %cst_1 : tensor<4xi64>
    %14 = arith.cmpi sge, %12, %cst_0 : tensor<4xi64>
    %15 = arith.andi %13, %14 : tensor<4xi1>
    tt.assert %15, "int32 overflow detected for operation sub" : tensor<4xi1>
    %16 = arith.subi %10, %cst_2 : tensor<4xi32>
    %17 = arith.extui %7 : tensor<4xi32> to tensor<4xi64>
    %18 = arith.extui %16 : tensor<4xi32> to tensor<4xi64>
    %19 = arith.addi %17, %18 : tensor<4xi64>
    %20 = arith.cmpi sle, %19, %cst_1 : tensor<4xi64>
    %21 = arith.cmpi sge, %19, %cst_0 : tensor<4xi64>
    %22 = arith.andi %20, %21 : tensor<4xi1>
    tt.assert %22, "int32 overflow detected for operation add" : tensor<4xi1>
    %23 = arith.addi %7, %16 : tensor<4xi32>
    %24 = arith.divui %23, %10 : tensor<4xi32>
    %25 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %26 = tt.addptr %25, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %26, %24 : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[TENSOR_C1:.*]] = linalg.fill ins(%[[CONST_C1:.*]] : i32) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:       %[[TENSOR_SUB:.*]] = arith.subi %[[TENSOR_DIV:.*]], %[[TENSOR_C1]] : tensor<4xi32>
// CHECK:       %[[TENSOR_ADD:.*]] = arith.addi %[[TENSOR_X:.*]], %[[TENSOR_SUB]] : tensor<4xi32>
// CHECK:       %[[TENSOR_RES:.*]] = arith.divui %[[TENSOR_ADD]], %[[TENSOR_DIV]] : tensor<4xi32>

// -----
// op: cdiv, dtype: uint64

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<i64>) {
    %cst = arith.constant dense<1> : tensor<4xi64>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i64>>
    %11 = arith.subi %10, %cst : tensor<4xi64>
    %12 = arith.addi %7, %11 : tensor<4xi64>
    %13 = arith.divui %12, %10 : tensor<4xi64>
    %14 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %15 = tt.addptr %14, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    tt.store %15, %13 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[TENSOR_C1:.*]] = linalg.fill ins(%[[CONST_C1:.*]] : i64) outs(%[[TENSOR_EMPTY:.*]] : tensor<4xi64>) -> tensor<4xi64>
// CHECK:       %[[TENSOR_SUB:.*]] = arith.subi %[[TENSOR_DIV:.*]], %[[TENSOR_C1]] : tensor<4xi64>
// CHECK:       %[[TENSOR_ADD:.*]] = arith.addi %[[TENSOR_X:.*]], %[[TENSOR_SUB]] : tensor<4xi64>
// CHECK:       %[[TENSOR_RES:.*]] = arith.divui %[[TENSOR_ADD]], %[[TENSOR_DIV]] : tensor<4xi64>

// -----
// op: cdiv, dtype: f8E5M2

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>, %arg2: !tt.ptr<f8E5M2>) {
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
    %11 = arith.divf %7, %10 : tensor<4xf8E5M2>
    %12 = math.ceil %11 : tensor<4xf8E5M2>
    %13 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %14 = tt.addptr %13, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    tt.store %14, %12 : tensor<4x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[FDIV:.*]] = arith.divf %[[TENSOR_X:.*]], %[[TENSOR_DIV:.*]] : tensor<4xf8E5M2>
// CHECK:       %[[RES:.*]] = math.ceil %[[FDIV]] : tensor<4xf8E5M2>

// -----
// op: cdiv, dtype: f8E4M3FN

module {
  tt.func public @triton_cdiv(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>, %arg2: !tt.ptr<f8E4M3FN>) {
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
    %11 = arith.divf %7, %10 : tensor<4xf8E4M3FN>
    %12 = math.ceil %11 : tensor<4xf8E4M3FN>
    %13 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %14 = tt.addptr %13, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    tt.store %14, %12 : tensor<4x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_cdiv
// CHECK:       %[[FDIV:.*]] = arith.divf %[[TENSOR_X:.*]], %[[TENSOR_DIV:.*]] : tensor<4xf8E4M3FN>
// CHECK:       %[[RES:.*]] = math.ceil %[[FDIV]] : tensor<4xf8E4M3FN>