// RUN:  triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === reduce and sum use the same case ===
// === i8 u8 version ===
module {
  tt.func public @fn_npu_u8(
    %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}
  ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<8x8x4x!tt.ptr<i8>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i8>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i8>>
    %20 = tt.reshape %19 : tensor<8x8x4xi8> -> tensor<256xi8>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %22 = arith.addi %arg2, %arg3 : i8
      tt.reduce.return %22 : i8
    }) : (tensor<256xi8>) -> i8
    tt.store %arg0, %21 : !tt.ptr<i8>
    tt.return
  }
}

// === i16 u16 version ===
module {
  tt.func public @fn_npu_i16(
    %arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}
  ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<8x8x4x!tt.ptr<i16>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i16>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i16>>
    %20 = tt.reshape %19 : tensor<8x8x4xi16> -> tensor<256xi16>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i16, %arg3: i16):
      %22 = arith.addi %arg2, %arg3 : i16
      tt.reduce.return %22 : i16
    }) : (tensor<256xi16>) -> i16
    tt.store %arg0, %21 : !tt.ptr<i16>
    tt.return
  }
}

// === i32 u32 version ===
module {
  tt.func public @fn_npu_i32(
    %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}
  ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i32>>
    %20 = tt.reshape %19 : tensor<8x8x4xi32> -> tensor<256xi32>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %22 = arith.addi %arg2, %arg3 : i32
      tt.reduce.return %22 : i32
    }) : (tensor<256xi32>) -> i32
    tt.store %arg0, %21 : !tt.ptr<i32>
    tt.return
  }
}

// === i64 u64 version ===
module {
  tt.func public @fn_npu_i64(
    %arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}
  ) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x8x4x!tt.ptr<i64>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<i64>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<i64>>
    %20 = tt.reshape %19 : tensor<8x8x4xi64> -> tensor<256xi64>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i64, %arg3: i64):
      %22 = arith.addi %arg2, %arg3 : i64
      tt.reduce.return %22 : i64
    }) : (tensor<256xi64>) -> i64
    tt.store %arg0, %21 : !tt.ptr<i64>
    tt.return
  }
}

// === f8E4M3FN version ===
module {
  tt.func public @fn_npu_f8E4M3FN(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32},
                             %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<8x8x4x!tt.ptr<f8E4M3FN>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<f8E4M3FN>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<f8E4M3FN>>
    %20 = tt.reshape %19 allow_reorder : tensor<8x8x4xf8E4M3FN> -> tensor<256xf8E4M3FN>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f8E4M3FN, %arg3: f8E4M3FN):
      %22 = arith.addf %arg2, %arg3 : f8E4M3FN
      tt.reduce.return %22 : f8E4M3FN
    }) : (tensor<256xf8E4M3FN>) -> f8E4M3FN
    tt.store %arg0, %21 : !tt.ptr<f8E4M3FN>
    tt.return
  }
}

// === f8E5M2 version ===
module {
  tt.func public @fn_npu_f8E5M2(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
                             %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_0 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<8x8x4x!tt.ptr<f8E5M2>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<f8E5M2>>, tensor<8x8x4xi32>
    %19 = tt.load %18 : tensor<8x8x4x!tt.ptr<f8E5M2>>
    %20 = tt.reshape %19 allow_reorder : tensor<8x8x4xf8E5M2> -> tensor<256xf8E5M2>
    %21 = "tt.reduce"(%20) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f8E5M2, %arg3: f8E5M2):
      %22 = arith.addf %arg2, %arg3 : f8E5M2
      tt.reduce.return %22 : f8E5M2
    }) : (tensor<256xf8E5M2>) -> f8E5M2
    tt.store %arg0, %21 : !tt.ptr<f8E5M2>
    tt.return
  }
}


// ===== CHECKS =====
// CHECK-DAG: arith.constant 0 : i8
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xi8>, {{.*}}) -> tensor<256xi8>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xi8>) outs(%{{.*}} : tensor<i8>) dimensions = [0]
// CHECK-DAG: arith.addi %in, %init : i8

// CHECK-DAG: arith.constant 0 : i16
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xi16>, {{.*}}) -> tensor<256xi16>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xi16>) outs(%{{.*}} : tensor<i16>) dimensions = [0]
// CHECK-DAG: arith.addi %in, %init : i16

// CHECK-DAG: arith.constant 0 : i32
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xi32>, {{.*}}) -> tensor<256xi32>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xi32>) outs(%{{.*}} : tensor<i32>) dimensions = [0]
// CHECK-DAG: arith.addi %in, %init : i32

// CHECK-DAG: arith.constant 0 : i64
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xi64>, {{.*}}) -> tensor<256xi64>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xi64>) outs(%{{.*}} : tensor<i64>) dimensions = [0]
// CHECK-DAG: arith.addi %in, %init : i64

// CHECK-DAG: arith.constant 0.0{{.*}} : f8E4M3FN
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xf8E4M3FN>, {{.*}}) -> tensor<256xf8E4M3FN>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xf8E4M3FN>) outs(%{{.*}} : tensor<f8E4M3FN>) dimensions = [0]
// CHECK-DAG: arith.addf %in, %init : f8E4M3FN

// CHECK-DAG: arith.constant 0.0{{.*}} : f8E5M2
// CHECK-DAG: tensor.reshape %{{.*}} : (tensor<8x8x4xf8E5M2>, {{.*}}) -> tensor<256xf8E5M2>
// CHECK-DAG: linalg.reduce ins(%{{.*}} : tensor<256xf8E5M2>) outs(%{{.*}} : tensor<f8E5M2>) dimensions = [0]
// CHECK-DAG: arith.addf %in, %init : f8E5M2