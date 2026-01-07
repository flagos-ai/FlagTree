// RUN: triton-adapter-opt "--triton-to-linalg=global-kernel=false named-ops=True" %s | FileCheck %s

module {
  tt.func public @triton_gt_2d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i8>>
    %15 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %16 = tt.addptr %15, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %17 = tt.load %16 : tensor<32x16x!tt.ptr<i8>>
    %18 = arith.cmpi ugt, %14, %17 : tensor<32x16xi8>
    %19 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<32x16x!tt.ptr<i8>>
    %20 = tt.addptr %19, %11 : tensor<32x16x!tt.ptr<i8>>, tensor<32x16xi32>
    %21 = arith.extui %18 : tensor<32x16xi1> to tensor<32x16xi8>
    tt.store %20, %21 : tensor<32x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt_2d
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi8>

// -----

module {
  tt.func public @triton_gt_2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<32x16x!tt.ptr<i16>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i16>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i16>>
    %15 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<32x16x!tt.ptr<i16>>
    %16 = tt.addptr %15, %11 : tensor<32x16x!tt.ptr<i16>>, tensor<32x16xi32>
    %17 = tt.load %16 : tensor<32x16x!tt.ptr<i16>>
    %18 = arith.cmpi ugt, %14, %17 : tensor<32x16xi16>
    %19 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<32x16x!tt.ptr<i16>>
    %20 = tt.addptr %19, %11 : tensor<32x16x!tt.ptr<i16>>, tensor<32x16xi32>
    %21 = arith.extui %18 : tensor<32x16xi1> to tensor<32x16xi16>
    tt.store %20, %21 : tensor<32x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt_2d
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi16>

// -----

module {
  tt.func public @triton_gt_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i32>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i32>>
    %15 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>>
    %16 = tt.addptr %15, %11 : tensor<32x16x!tt.ptr<i32>>, tensor<32x16xi32>
    %17 = tt.load %16 : tensor<32x16x!tt.ptr<i32>>
    %18 = arith.cmpi ugt, %14, %17 : tensor<32x16xi32>
    %19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x16x!tt.ptr<i32>>
    %20 = tt.addptr %19, %11 : tensor<32x16x!tt.ptr<i32>>, tensor<32x16xi32>
    %21 = arith.extui %18 : tensor<32x16xi1> to tensor<32x16xi32>
    tt.store %20, %21 : tensor<32x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt_2d
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi32>

// -----

module {
  tt.func public @triton_gt_2d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<32x1xi32>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %2, %3 : tensor<32xi32>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = arith.muli %6, %cst : tensor<32x1xi32>
    %8 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %9 = tt.broadcast %7 : tensor<32x1xi32> -> tensor<32x16xi32>
    %10 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<32x16xi32>
    %11 = arith.addi %9, %10 : tensor<32x16xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<32x16x!tt.ptr<i64>>
    %13 = tt.addptr %12, %11 : tensor<32x16x!tt.ptr<i64>>, tensor<32x16xi32>
    %14 = tt.load %13 : tensor<32x16x!tt.ptr<i64>>
    %15 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x16x!tt.ptr<i64>>
    %16 = tt.addptr %15, %11 : tensor<32x16x!tt.ptr<i64>>, tensor<32x16xi32>
    %17 = tt.load %16 : tensor<32x16x!tt.ptr<i64>>
    %18 = arith.cmpi ugt, %14, %17 : tensor<32x16xi64>
    %19 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<32x16x!tt.ptr<i64>>
    %20 = tt.addptr %19, %11 : tensor<32x16x!tt.ptr<i64>>, tensor<32x16xi32>
    %21 = arith.extui %18 : tensor<32x16xi1> to tensor<32x16xi64>
    tt.store %20, %21 : tensor<32x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @triton_gt_2d
// CHECK:           %[[RES:.*]] = arith.cmpi ugt, %[[X0:.*]], %[[X1:.*]] : tensor<32x16xi64>

// -----

module {
  tt.func public @fn_npu_2d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.splat %3 : i32 -> tensor<16xi32>
    %5 = arith.addi %1, %4 : tensor<16xi32>
    %6 = arith.muli %0, %c16_i32 : i32
    %7 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %8 = tt.splat %6 : i32 -> tensor<1x16xi32>
    %9 = arith.addi %8, %7 : tensor<1x16xi32>
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x16x!tt.ptr<i8>>
    %11 = tt.addptr %10, %9 : tensor<1x16x!tt.ptr<i8>>, tensor<1x16xi32>
    %12 = tt.load %11 : tensor<1x16x!tt.ptr<i8>>
    %13 = arith.extui %12 : tensor<1x16xi8> to tensor<1x16xi32>
    %14 = tt.extern_elementwise %13, %cst {libname = "", libpath = "", pure = true, symbol = "__hmf_flipi32"} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %15 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x16x!tt.ptr<i8>>
    %16 = tt.addptr %15, %9 : tensor<1x16x!tt.ptr<i8>>, tensor<1x16xi32>
    %17 = arith.trunci %14 : tensor<1x16xi32> to tensor<1x16xi8>
    tt.store %16, %17 : tensor<1x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_2d
// CHECK:           %[[EXTUI_OUT:.*]] = arith.extui %[[EXTUI_IN:.*]] : tensor<1x16xi8> to tensor<1x16xi32>
// CHECK:           %[[MAPPED_OUT:.*]] = linalg.map { func.call {callee = @__hmf_flipi32} } ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<1x16xi32>, tensor<1x16xi32>) outs(%[[IN1]] : tensor<1x16xi32>)
// CHECK:           %[[TRUNCI_OUT:.*]] = arith.trunci %[[TRUNCI_IN:.*]] : tensor<1x16xi32> to tensor<1x16xi8>

// -----

module {
  tt.func public @fn_npu_2d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.splat %3 : i32 -> tensor<16xi32>
    %5 = arith.addi %1, %4 : tensor<16xi32>
    %6 = arith.muli %0, %c16_i32 : i32
    %7 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %8 = tt.splat %6 : i32 -> tensor<1x16xi32>
    %9 = arith.addi %8, %7 : tensor<1x16xi32>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %11 = tt.addptr %10, %9 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    %12 = tt.load %11 : tensor<1x16x!tt.ptr<i16>>
    %13 = arith.extui %12 : tensor<1x16xi16> to tensor<1x16xi32>
    %14 = tt.extern_elementwise %13, %cst {libname = "", libpath = "", pure = true, symbol = "__hmf_flipi32"} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %15 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x16x!tt.ptr<i16>>
    %16 = tt.addptr %15, %9 : tensor<1x16x!tt.ptr<i16>>, tensor<1x16xi32>
    %17 = arith.trunci %14 : tensor<1x16xi32> to tensor<1x16xi16>
    tt.store %16, %17 : tensor<1x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_2d
// CHECK:           %[[EXTUI_OUT:.*]] = arith.extui %[[EXTUI_IN:.*]] : tensor<1x16xi16> to tensor<1x16xi32>
// CHECK:           %[[MAPPED_OUT:.*]] = linalg.map { func.call {callee = @__hmf_flipi32} } ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<1x16xi32>, tensor<1x16xi32>) outs(%[[IN1]] : tensor<1x16xi32>)
// CHECK:           %[[TRUNCI_OUT:.*]] = arith.trunci %[[TRUNCI_IN:.*]] : tensor<1x16xi32> to tensor<1x16xi16>

// -----

module {
  tt.func public @fn_npu_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %2, %c16_i32 : i32
    %4 = tt.splat %3 : i32 -> tensor<16xi32>
    %5 = arith.addi %1, %4 : tensor<16xi32>
    %6 = arith.muli %0, %c16_i32 : i32
    %7 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %8 = tt.splat %6 : i32 -> tensor<1x16xi32>
    %9 = arith.addi %8, %7 : tensor<1x16xi32>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %11 = tt.addptr %10, %9 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    %12 = tt.load %11 : tensor<1x16x!tt.ptr<i32>>
    %13 = tt.extern_elementwise %12, %cst {libname = "", libpath = "", pure = true, symbol = "__hmf_flipui32"} : (tensor<1x16xi32>, tensor<1x16xi32>) -> tensor<1x16xi32>
    %14 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x16x!tt.ptr<i32>>
    %15 = tt.addptr %14, %9 : tensor<1x16x!tt.ptr<i32>>, tensor<1x16xi32>
    tt.store %15, %13 : tensor<1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_2d
// CHECK:           %[[MAPPED_OUT:.*]] = linalg.map { func.call {callee = @__hmf_flipui32} } ins(%[[IN1:.*]], %[[IN2:.*]] : tensor<1x16xi32>, tensor<1x16xi32>) outs(%[[IN1]] : tensor<1x16xi32>)

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1> : tensor<1x1x16xi8>
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
    %18 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<1x1x16x!tt.ptr<i8>>
    %19 = tt.addptr %18, %14 : tensor<1x1x16x!tt.ptr<i8>>, tensor<1x1x16xi32>
    %20 = tt.load %19 : tensor<1x1x16x!tt.ptr<i8>>
    %21 = arith.cmpi ult, %17, %20 : tensor<1x1x16xi8>
    %22 = arith.select %21, %17, %cst : tensor<1x1x16xi1>, tensor<1x1x16xi8>
    %23 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<1x1x16x!tt.ptr<i8>>
    %24 = tt.addptr %23, %14 : tensor<1x1x16x!tt.ptr<i8>>, tensor<1x1x16xi32>
    tt.store %24, %22 : tensor<1x1x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[SELECT_OUT:.*]] = arith.select %[[COND:.*]], %[[THEN:.*]], %[[ELSE:.*]] : tensor<1x1x16xi1>, tensor<1x1x16xi8>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1> : tensor<1x1x16xi16>
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
    %18 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<1x1x16x!tt.ptr<i16>>
    %19 = tt.addptr %18, %14 : tensor<1x1x16x!tt.ptr<i16>>, tensor<1x1x16xi32>
    %20 = tt.load %19 : tensor<1x1x16x!tt.ptr<i16>>
    %21 = arith.cmpi ult, %17, %20 : tensor<1x1x16xi16>
    %22 = arith.select %21, %17, %cst : tensor<1x1x16xi1>, tensor<1x1x16xi16>
    %23 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1x1x16x!tt.ptr<i16>>
    %24 = tt.addptr %23, %14 : tensor<1x1x16x!tt.ptr<i16>>, tensor<1x1x16xi32>
    tt.store %24, %22 : tensor<1x1x16x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[SELECT_OUT:.*]] = arith.select %[[COND:.*]], %[[THEN:.*]], %[[ELSE:.*]] : tensor<1x1x16xi1>, tensor<1x1x16xi16>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1> : tensor<1x1x16xi32>
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
    %18 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %19 = tt.addptr %18, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    %20 = tt.load %19 : tensor<1x1x16x!tt.ptr<i32>>
    %21 = arith.cmpi ult, %17, %20 : tensor<1x1x16xi32>
    %22 = arith.select %21, %17, %cst : tensor<1x1x16xi1>, tensor<1x1x16xi32>
    %23 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x1x16x!tt.ptr<i32>>
    %24 = tt.addptr %23, %14 : tensor<1x1x16x!tt.ptr<i32>>, tensor<1x1x16xi32>
    tt.store %24, %22 : tensor<1x1x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[SELECT_OUT:.*]] = arith.select %[[COND:.*]], %[[THEN:.*]], %[[ELSE:.*]] : tensor<1x1x16xi1>, tensor<1x1x16xi32>

// -----

module {
  tt.func public @fn_npu_(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1> : tensor<1x1x16xi64>
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
    %15 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<1x1x16x!tt.ptr<i64>>
    %16 = tt.addptr %15, %14 : tensor<1x1x16x!tt.ptr<i64>>, tensor<1x1x16xi32>
    %17 = tt.load %16 : tensor<1x1x16x!tt.ptr<i64>>
    %18 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<1x1x16x!tt.ptr<i64>>
    %19 = tt.addptr %18, %14 : tensor<1x1x16x!tt.ptr<i64>>, tensor<1x1x16xi32>
    %20 = tt.load %19 : tensor<1x1x16x!tt.ptr<i64>>
    %21 = arith.cmpi ult, %17, %20 : tensor<1x1x16xi64>
    %22 = arith.select %21, %17, %cst : tensor<1x1x16xi1>, tensor<1x1x16xi64>
    %23 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<1x1x16x!tt.ptr<i64>>
    %24 = tt.addptr %23, %14 : tensor<1x1x16x!tt.ptr<i64>>, tensor<1x1x16xi32>
    tt.store %24, %22 : tensor<1x1x16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:     func.func @fn_npu_
// CHECK:           %[[SELECT_OUT:.*]] = arith.select %[[COND:.*]], %[[THEN:.*]], %[[ELSE:.*]] : tensor<1x1x16xi1>, tensor<1x1x16xi64>

