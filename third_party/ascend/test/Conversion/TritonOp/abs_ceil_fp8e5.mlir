// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

// -----
// op: abs, dtype: f8E5M2

module {
  tt.func public @triton_abs(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E5M2>>
    %8 = math.absf %7 : tensor<4xf8E5M2>
    %9 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_abs
// CHECK:       %[[RES:.*]] = math.absf %[[X:.*]] : tensor<4xf8E5M2>

// -----
// op: abs, dtype: f8E4M3FN

module {
  tt.func public @triton_abs(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E4M3FN>>
    %8 = math.absf %7 : tensor<4xf8E4M3FN>
    %9 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_abs
// CHECK:       %[[RES:.*]] = math.absf %[[X:.*]] : tensor<4xf8E4M3FN>

// -----
// op: ceil, dtype: f8E5M2

module {
  tt.func public @triton_ceil(%arg0: !tt.ptr<f8E5M2>, %arg1: !tt.ptr<f8E5M2>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E5M2>>
    %8 = math.ceil %7 : tensor<4xf8E5M2>
    %9 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_ceil
// CHECK:       %[[RES:.*]] = math.ceil %[[X:.*]] : tensor<4xf8E5M2>

// -----
// op: ceil, dtype: f8E4M3FN

module {
  tt.func public @triton_ceil(%arg0: !tt.ptr<f8E4M3FN>, %arg1: !tt.ptr<f8E4M3FN>) {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %7 = tt.load %6 : tensor<4x!tt.ptr<f8E4M3FN>>
    %8 = math.ceil %7 : tensor<4xf8E4M3FN>
    %9 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %10 = tt.addptr %9, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    tt.store %10, %8 : tensor<4x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @triton_ceil
// CHECK:       %[[RES:.*]] = math.ceil %[[X:.*]] : tensor<4xf8E4M3FN>