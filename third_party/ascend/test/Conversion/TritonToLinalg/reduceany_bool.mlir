// RUN: triton-adapter-opt -split-input-file --triton-to-linalg %s | FileCheck %s 

// CHECK-LABEL: func.func @kernel(
module {
  tt.func public @kernel(%input : !tt.ptr<i1>, %output : !tt.ptr<i1>)
  {
    %cst = arith.constant dense<0> : tensor<128xi8>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %input : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %in = tt.load %3 : tensor<128x!tt.ptr<i8>>
    %4 = arith.cmpi ne, %in, %cst : tensor<128xi8>
    // CHECK: linalg.reduce
    // CHECK: arith.ori
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %6 = arith.ori %arg0, %arg1 : i1
      tt.reduce.return %6 : i1
    }) : (tensor<128xi1>) -> i1
    %7 = tt.bitcast %output : !tt.ptr<i1> -> !tt.ptr<i8>
    %8 = tt.splat %7 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %9 = tt.splat %5 : i1 -> tensor<1xi1>
    %10 = arith.extui %9 : tensor<1xi1> to tensor<1xi8>
    tt.store %8, %10 : tensor<1x!tt.ptr<i8>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @kernel_reduceAnd(
module {
  tt.func public @kernel_reduceAnd(%input : !tt.ptr<i1>, %output : !tt.ptr<i1>)
  {
    %cst = arith.constant dense<0> : tensor<128xi8>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %input : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %in = tt.load %3 : tensor<128x!tt.ptr<i8>>
    %4 = arith.cmpi ne, %in, %cst : tensor<128xi8>
    // CHECK: linalg.reduce
    // CHECK: arith.andi
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %6 = arith.andi %arg0, %arg1 : i1
      tt.reduce.return %6 : i1
    }) : (tensor<128xi1>) -> i1
    %7 = tt.bitcast %output : !tt.ptr<i1> -> !tt.ptr<i8>
    %8 = tt.splat %7 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %9 = tt.splat %5 : i1 -> tensor<1xi1>
    %10 = arith.extui %9 : tensor<1xi1> to tensor<1xi8>
    tt.store %8, %10 : tensor<1x!tt.ptr<i8>>
    tt.return
  }
}

// -----
// CHECK-LABEL: func.func @kernel_reduceAnd(
module {
  tt.func public @kernel_reduceAnd(%input : !tt.ptr<i1>, %output : !tt.ptr<i1>)
  {
    %cst = arith.constant dense<0> : tensor<128xi8>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %input : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %in = tt.load %3 : tensor<128x!tt.ptr<i8>>
    %4 = arith.cmpi ne, %in, %cst : tensor<128xi8>
    // CHECK: linalg.reduce
    // CHECK: arith.xori
    %5 = "tt.reduce"(%4) <{axis = 0 : i32}> ({
    ^bb0(%arg0: i1, %arg1: i1):
      %6 = arith.xori %arg0, %arg1 : i1
      tt.reduce.return %6 : i1
    }) : (tensor<128xi1>) -> i1
    %7 = tt.bitcast %output : !tt.ptr<i1> -> !tt.ptr<i8>
    %8 = tt.splat %7 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %9 = tt.splat %5 : i1 -> tensor<1xi1>
    %10 = arith.extui %9 : tensor<1xi1> to tensor<1xi8>
    tt.store %8, %10 : tensor<1x!tt.ptr<i8>>
    tt.return
  }
}
