// RUN: triton-adapter-opt "--triton-to-linalg=global-kernel=false named-ops=True" %s | FileCheck %s

module {
  tt.func public @triton_lshift_2d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<32x16xi8>
    %cst_0 = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst_0 : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i8>>
    %15 = arith.shli %14, %cst : tensor<32x16xi8>
    %16 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %17 = tt.addptr %16, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    tt.store %17, %15 : tensor<32x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lshift_2d
// CHECK:           %[[RES:.*]] = arith.shli %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi8>

// -----

module {
  tt.func public @triton_lshift_2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi16>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i16>>
    %9 = arith.shli %8, %cst : tensor<1x16xi16>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lshift_2d
// CHECK:           %[[RES:.*]] = arith.shli %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi16>

// -----

module {
  tt.func public @triton_lshift_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi32>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i32>>
    %9 = arith.shli %8, %cst : tensor<1x16xi32>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lshift_2d
// CHECK:           %[[RES:.*]] = arith.shli %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi32>

// -----

module {
  tt.func public @triton_lshift_2d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi64>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x16x!tt.ptr<i64>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i64>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i64>>
    %9 = arith.shli %8, %cst : tensor<1x16xi64>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x16x!tt.ptr<i64>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i64>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_lshift_2d
// CHECK:           %[[RES:.*]] = arith.shli %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi64>

// -----

module {
  tt.func public @triton_rshift_2d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<32x16xi8>
    %cst_0 = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst_0 : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i8>>
    %15 = arith.shrui %14, %cst : tensor<32x16xi8>
    %16 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %17 = tt.addptr %16, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    tt.store %17, %15 : tensor<32x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_rshift_2d
// CHECK:           %[[RES:.*]] = arith.shrui %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi8>

// -----

module {
  tt.func public @triton_rshift_2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi16>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i16>>
    %9 = arith.shrui %8, %cst : tensor<1x16xi16>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_rshift_2d
// CHECK:           %[[RES:.*]] = arith.shrui %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi16>

// -----

module {
  tt.func public @triton_rshift_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi32>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i32>>
    %9 = arith.shrui %8, %cst : tensor<1x16xi32>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_rshift_2d
// CHECK:           %[[RES:.*]] = arith.shrui %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi32>

// -----

module {
  tt.func public @triton_rshift_2d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<2> : tensor<1x16xi64>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = arith.muli %0, %c16_i32 : i32
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = tt.splat %2 : i32 -> tensor<1x16xi32>
    %5 = arith.addi %4, %3 : tensor<1x16xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x16x!tt.ptr<i64>>
    %7 = tt.addptr %6, %5 : tensor<1x16x!tt.ptr<i64>>, tensor<1x16xi32>
    %8 = tt.load %7 : tensor<1x16x!tt.ptr<i64>>
    %9 = arith.shrui %8, %cst : tensor<1x16xi64>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x16x!tt.ptr<i64>>
    %11 = tt.addptr %10, %5 : tensor<1x16x!tt.ptr<i64>>, tensor<1x16xi32>
    tt.store %11, %9 : tensor<1x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_rshift_2d
// CHECK:           %[[RES:.*]] = arith.shrui %[[X0:.*]], %[[X1:.*]] : tensor<1x16xi64>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<1x1x16xi8>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.splat %3 : i32 -> tensor<16xi32>
    %6 = arith.addi %4, %5 : tensor<16xi32>
    %7 = arith.muli %0, %c32_i32 : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %9 = arith.muli %1, %c16_i32 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %13 = tt.splat %10 : i32 -> tensor<1x1x16xi32>
    %14 = arith.addi %13, %12 : tensor<1x1x16xi32>
    %15 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x1x16x!tt.ptr<i8>>
    %16 = tt.addptr %15, %14 : tensor<1x1x16x!tt.ptr<i8>>, tensor<1x1x16xi32>
    %17 = tt.load %16 : tensor<1x1x16x!tt.ptr<i8>>
    %18 = arith.subi %cst, %17 : tensor<1x1x16xi8>
    %19 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x1x16x!tt.ptr<i8>>
    %20 = tt.addptr %19, %14 : tensor<1x1x16x!tt.ptr<i8>>, tensor<1x1x16xi32>
    tt.store %20, %18 : tensor<1x1x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[ZERO1_I8:.*]] = arith.constant 0 : i8
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[ZERO1_I8]] : i8) outs({{.*}} : tensor<1x1x16xi8>) -> tensor<1x1x16xi8>
// CHECK:           %[[SUB_RESULT:.*]] = arith.subi %[[FILL_TENSOR]], %{{.*}} : tensor<1x1x16xi8>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<1x1x16xi16>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.splat %3 : i32 -> tensor<16xi32>
    %6 = arith.addi %4, %5 : tensor<16xi32>
    %7 = arith.muli %0, %c32_i32 : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %9 = arith.muli %1, %c16_i32 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %13 = tt.splat %10 : i32 -> tensor<1x1x16xi32>
    %14 = arith.addi %13, %12 : tensor<1x1x16xi32>
    %15 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x1x16x!tt.ptr<i16>>
    %16 = tt.addptr %15, %14 : tensor<1x1x16x!tt.ptr<i16>>, tensor<1x1x16xi32>
    %17 = tt.load %16 : tensor<1x1x16x!tt.ptr<i16>>
    %18 = arith.subi %cst, %17 : tensor<1x1x16xi16>
    %19 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x1x16x!tt.ptr<i16>>
    %20 = tt.addptr %19, %14 : tensor<1x1x16x!tt.ptr<i16>>, tensor<1x1x16xi32>
    tt.store %20, %18 : tensor<1x1x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[ZERO1_I16:.*]] = arith.constant 0 : i16
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[ZERO1_I16]] : i16) outs({{.*}} : tensor<1x1x16xi16>) -> tensor<1x1x16xi16>
// CHECK:           %[[SUB_RESULT:.*]] = arith.subi %[[FILL_TENSOR]], %{{.*}} : tensor<1x1x16xi16>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<1x1x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.splat %3 : i32 -> tensor<16xi32>
    %6 = arith.addi %4, %5 : tensor<16xi32>
    %7 = arith.muli %0, %c32_i32 : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %9 = arith.muli %1, %c16_i32 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %13 = tt.splat %10 : i32 -> tensor<1x1x16xi32>
    %14 = arith.addi %13, %12 : tensor<1x1x16xi32>
    %15 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %16 = tt.addptr %15, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    %17 = tt.load %16 : tensor<1x1x16x!tt.ptr<i32>>
    %18 = arith.subi %cst, %17 : tensor<1x1x16xi32>
    %19 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %20 = tt.addptr %19, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    tt.store %20, %18 : tensor<1x1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[ZERO1_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[ZERO1_I32]] : i32) outs({{.*}} : tensor<1x1x16xi32>) -> tensor<1x1x16xi32>
// CHECK:           %[[SUB_RESULT:.*]] = arith.subi %[[FILL_TENSOR]], %{{.*}} : tensor<1x1x16xi32>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0> : tensor<1x1x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.splat %3 : i32 -> tensor<16xi32>
    %6 = arith.addi %4, %5 : tensor<16xi32>
    %7 = arith.muli %0, %c32_i32 : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %9 = arith.muli %1, %c16_i32 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %12 = tt.expand_dims %11 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %13 = tt.splat %10 : i32 -> tensor<1x1x16xi32>
    %14 = arith.addi %13, %12 : tensor<1x1x16xi32>
    %15 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %16 = tt.addptr %15, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    %17 = tt.load %16 : tensor<1x1x16x!tt.ptr<i32>>
    %18 = arith.subi %cst, %17 : tensor<1x1x16xi32>
    %19 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %20 = tt.addptr %19, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    tt.store %20, %18 : tensor<1x1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[ZERO1_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[ZERO1_I32]] : i32) outs({{.*}} : tensor<1x1x16xi32>) -> tensor<1x1x16xi32>
// CHECK:           %[[SUB_RESULT:.*]] = arith.subi %[[FILL_TENSOR]], %{{.*}} : tensor<1x1x16xi32>

// -----