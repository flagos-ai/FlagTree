// RUN: triton-adapter-opt "--triton-to-linalg=global-kernel=false named-ops=True" %s | FileCheck %s

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi8>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x32x16x!tt.ptr<i8>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i8>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i8>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi8>
    %27 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x32x16x!tt.ptr<i8>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i8>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I8:.*]] = arith.constant -1 : i8
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I8]] : i8) outs({{.*}} : tensor<1x32x16xi8>) -> tensor<1x32x16xi8>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi8>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi16>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x32x16x!tt.ptr<i16>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i16>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i16>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi16>
    %27 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x32x16x!tt.ptr<i16>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i16>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I16:.*]] = arith.constant -1 : i16
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I16]] : i16) outs({{.*}} : tensor<1x32x16xi16>) -> tensor<1x32x16xi16>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi16>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi32>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x32x16x!tt.ptr<i32>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i32>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i32>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi32>
    %27 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x32x16x!tt.ptr<i32>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i32>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I32:.*]] = arith.constant -1 : i32
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I32]] : i32) outs({{.*}} : tensor<1x32x16xi32>) -> tensor<1x32x16xi32>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi32>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi64>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x32x16x!tt.ptr<i64>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i64>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i64>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi64>
    %27 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x32x16x!tt.ptr<i64>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i64>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I64:.*]] = arith.constant -1 : i64
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I64]] : i64) outs({{.*}} : tensor<1x32x16xi64>) -> tensor<1x32x16xi64>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi64>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi8>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x32x16x!tt.ptr<i8>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i8>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i8>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi8>
    %27 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x32x16x!tt.ptr<i8>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i8>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I8:.*]] = arith.constant -1 : i8
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I8]] : i8) outs({{.*}} : tensor<1x32x16xi8>) -> tensor<1x32x16xi8>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi8>


// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi16>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x32x16x!tt.ptr<i16>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i16>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i16>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi16>
    %27 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x32x16x!tt.ptr<i16>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i16>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I16:.*]] = arith.constant -1 : i16
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I16]] : i16) outs({{.*}} : tensor<1x32x16xi16>) -> tensor<1x32x16xi16>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi16>


// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi32>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x32x16x!tt.ptr<i32>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i32>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i32>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi32>
    %27 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x32x16x!tt.ptr<i32>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i32>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I32:.*]] = arith.constant -1 : i32
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I32]] : i32) outs({{.*}} : tensor<1x32x16xi32>) -> tensor<1x32x16xi32>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi32>


// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<-1> : tensor<1x32x16xi64>
    %cst_0 = arith.constant dense<16> : tensor<1x32x1xi32>
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %c32_i32 : i32
    %3 = tt.get_program_id z : i32
    %4 = arith.muli %3, %c16_i32 : i32
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %6 = tt.splat %2 : i32 -> tensor<32xi32>
    %7 = arith.addi %5, %6 : tensor<32xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %4 : i32 -> tensor<16xi32>
    %10 = arith.addi %8, %9 : tensor<16xi32>
    %11 = arith.muli %0, %c32_i32 : i32
    %12 = arith.muli %11, %c16_i32 : i32
    %13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x32xi32> -> tensor<1x32x1xi32>
    %15 = arith.muli %14, %cst_0 : tensor<1x32x1xi32>
    %16 = tt.splat %12 : i32 -> tensor<1x32x1xi32>
    %17 = arith.addi %16, %15 : tensor<1x32x1xi32>
    %18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %20 = tt.broadcast %17 : tensor<1x32x1xi32> -> tensor<1x32x16xi32>
    %21 = tt.broadcast %19 : tensor<1x1x16xi32> -> tensor<1x32x16xi32>
    %22 = arith.addi %20, %21 : tensor<1x32x16xi32>
    %23 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x32x16x!tt.ptr<i64>>
    %24 = tt.addptr %23, %22 : tensor<1x32x16x!tt.ptr<i64>>, tensor<1x32x16xi32>
    %25 = tt.load %24 : tensor<1x32x16x!tt.ptr<i64>>
    %26 = arith.xori %25, %cst : tensor<1x32x16xi64>
    %27 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x32x16x!tt.ptr<i64>>
    %28 = tt.addptr %27, %22 : tensor<1x32x16x!tt.ptr<i64>>, tensor<1x32x16xi32>
    tt.store %28, %26 : tensor<1x32x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[NEG1_I64:.*]] = arith.constant -1 : i64
// CHECK:           %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[NEG1_I64]] : i64) outs({{.*}} : tensor<1x32x16xi64>) -> tensor<1x32x16xi64>
// CHECK:           %[[XOR_RESULT:.*]] = arith.xori %{{.*}}, %[[FILL_TENSOR]] : tensor<1x32x16xi64>


// -----