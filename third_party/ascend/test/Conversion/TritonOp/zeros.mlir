// RUN:  triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === zeros ===
// === i8 u8 version ===
module {
  tt.func public @fn_npu_u8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<8x8x4xi8>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %8 = tt.broadcast %3 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %9 = tt.broadcast %5 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %10 = arith.addi %8, %9 : tensor<8x8x1xi32>
    %11 = tt.broadcast %10 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %12 = tt.broadcast %7 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %13 = arith.addi %11, %12 : tensor<8x8x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x8x4x!tt.ptr<i8>>
    %15 = tt.addptr %14, %13 : tensor<8x8x4x!tt.ptr<i8>>, tensor<8x8x4xi32>
    tt.store %15, %cst : tensor<8x8x4x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK: arith.constant 0 : i8
// CHECK: tensor.empty() : tensor<8x8x4xi8>
// CHECK: linalg.fill ins(%{{.*}} : i8) outs(%{{.*}} : tensor<8x8x4xi8>) -> tensor<8x8x4xi8>
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8, 8, 4]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}}


// === i16 u16 version ===
module {
  tt.func public @fn_npu_u16(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<8x8x4xi16>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %8 = tt.broadcast %3 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %9 = tt.broadcast %5 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %10 = arith.addi %8, %9 : tensor<8x8x1xi32>
    %11 = tt.broadcast %10 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %12 = tt.broadcast %7 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %13 = arith.addi %11, %12 : tensor<8x8x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<8x8x4x!tt.ptr<i16>>
    %15 = tt.addptr %14, %13 : tensor<8x8x4x!tt.ptr<i16>>, tensor<8x8x4xi32>
    tt.store %15, %cst : tensor<8x8x4x!tt.ptr<i16>>
    tt.return
  }
}

// ===== CHECKS =====
// CHECK-DAG: arith.constant 0 : i16
// CHECK-DAG: tensor.empty() : tensor<8x8x4xi16>
// CHECK-DAG: linalg.fill ins(%{{.*}} : i16) outs(%{{.*}} : tensor<8x8x4xi16>) -> tensor<8x8x4xi16>


// === i32 u32 version ===
module {
  tt.func public @fn_npu_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<8x8x4xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %8 = tt.broadcast %3 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %9 = tt.broadcast %5 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %10 = arith.addi %8, %9 : tensor<8x8x1xi32>
    %11 = tt.broadcast %10 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %12 = tt.broadcast %7 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %13 = arith.addi %11, %12 : tensor<8x8x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x4x!tt.ptr<i32>>
    %15 = tt.addptr %14, %13 : tensor<8x8x4x!tt.ptr<i32>>, tensor<8x8x4xi32>
    tt.store %15, %cst : tensor<8x8x4x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-DAG: arith.constant 0 : i32
// CHECK-DAG: tensor.empty() : tensor<8x8x4xi32>
// CHECK-DAG: linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : tensor<8x8x4xi32>) -> tensor<8x8x4xi32>


// === i64 u64 version ===
module {
  tt.func public @fn_npu_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<8x8x4xi64>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %5 = tt.expand_dims %4 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %8 = tt.broadcast %3 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %9 = tt.broadcast %5 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %10 = arith.addi %8, %9 : tensor<8x8x1xi32>
    %11 = tt.broadcast %10 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %12 = tt.broadcast %7 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %13 = arith.addi %11, %12 : tensor<8x8x4xi32>
    %14 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x8x4x!tt.ptr<i64>>
    %15 = tt.addptr %14, %13 : tensor<8x8x4x!tt.ptr<i64>>, tensor<8x8x4xi32>
    tt.store %15, %cst : tensor<8x8x4x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-DAG: arith.constant 0 : i64
// CHECK-DAG: tensor.empty() : tensor<8x8x4xi64>
// CHECK-DAG: linalg.fill ins(%{{.*}} : i64) outs(%{{.*}} : tensor<8x8x4xi64>) -> tensor<8x8x4xi64>


// === f8E4M3FN version ===
module {
  tt.func public @fn_npu_f8E4M3FN(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8x4xf8E4M3FN>
    %cst_0 = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_2 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_2 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst_0 : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<8x8x4x!tt.ptr<f8E4M3FN>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<f8E4M3FN>>, tensor<8x8x4xi32>
    tt.store %18, %cst : tensor<8x8x4x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// -----

// CHECK-DAG: arith.constant 0.0 : f8E4M3FN
// CHECK-DAG: tensor.empty() : tensor<8x8x4xf8E4M3FN>
// CHECK-DAG: linalg.fill ins(%{{.*}} : f8E4M3FN) outs(%{{.*}} : tensor<8x8x4xf8E4M3FN>) -> tensor<8x8x4xf8E4M3FN>


// === f8E5M2 version ===
module {
  tt.func public @fn_npu_f8E5M2(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x8x4xf8E5M2>
    %cst_0 = arith.constant dense<4> : tensor<1x8x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<8x1x1xi32>
    %cst_2 = arith.constant dense<8> : tensor<8x1x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %4 = arith.muli %3, %cst_2 : tensor<8x1x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<8x1x1xi32>
    %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst_0 : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<8x8x1xi32>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %14 = tt.broadcast %11 : tensor<8x8x1xi32> -> tensor<8x8x4xi32>
    %15 = tt.broadcast %13 : tensor<1x1x4xi32> -> tensor<8x8x4xi32>
    %16 = arith.addi %14, %15 : tensor<8x8x4xi32>
    %17 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<8x8x4x!tt.ptr<f8E5M2>>
    %18 = tt.addptr %17, %16 : tensor<8x8x4x!tt.ptr<f8E5M2>>, tensor<8x8x4xi32>
    tt.store %18, %cst : tensor<8x8x4x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// -----

// CHECK-DAG: arith.constant 0.0 : f8E5M2
// CHECK-DAG: tensor.empty() : tensor<8x8x4xf8E5M2>
// CHECK-DAG: linalg.fill ins(%{{.*}} : f8E5M2) outs(%{{.*}} : tensor<8x8x4xf8E5M2>) -> tensor<8x8x4xf8E5M2>
