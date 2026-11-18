// RUN: triton-opt --pass-pipeline="builtin.module(auto-blockify{auto-blockify-size=1},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-on-910-95=false compile-mode=simd_simt_template},triton-to-annotation,triton-to-unstructure{compile-on-910-95=false compile-mode=simd_simt_template},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-on-910-95=false enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true compile-mode=simd_simt_template})" --split-input-file %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<242x128xi8>
    %cst_0 = arith.constant dense<0> : tensor<128x242xi8>
    %cst_1 = arith.constant dense<0> : tensor<128x128xi32>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : i32 -> tensor<128xi32>
    %5 = arith.addi %4, %3 : tensor<128xi32>
    %6 = arith.muli %1, %c128_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<128xi32>
    %8 = arith.addi %7, %3 : tensor<128xi32>
    %9 = tt.make_range {end = 242 : i32, start = 0 : i32} : tensor<242xi32>
    %10 = tt.expand_dims %5 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %11 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %12 = arith.muli %10, %11 : tensor<128x1xi32>
    %13 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>>
    %14 = tt.addptr %13, %12 : tensor<128x1x!tt.ptr<i8>>, tensor<128x1xi32>
    %15 = tt.expand_dims %9 {axis = 0 : i32} : tensor<242xi32> -> tensor<1x242xi32>
    %16 = tt.broadcast %14 : tensor<128x1x!tt.ptr<i8>> -> tensor<128x242x!tt.ptr<i8>>
    %17 = tt.broadcast %15 : tensor<1x242xi32> -> tensor<128x242xi32>
    %18 = tt.addptr %16, %17 : tensor<128x242x!tt.ptr<i8>>, tensor<128x242xi32>
    %19 = tt.expand_dims %9 {axis = 1 : i32} : tensor<242xi32> -> tensor<242x1xi32>
    %20 = tt.splat %arg7 : i32 -> tensor<242x1xi32>
    %21 = arith.muli %19, %20 : tensor<242x1xi32>
    %22 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<242x1x!tt.ptr<i8>>
    %23 = tt.addptr %22, %21 : tensor<242x1x!tt.ptr<i8>>, tensor<242x1xi32>
    %24 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %25 = tt.broadcast %23 : tensor<242x1x!tt.ptr<i8>> -> tensor<242x128x!tt.ptr<i8>>
    %26 = tt.broadcast %24 : tensor<1x128xi32> -> tensor<242x128xi32>
    %27 = tt.addptr %25, %26 : tensor<242x128x!tt.ptr<i8>>, tensor<242x128xi32>
    %28 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %29 = arith.cmpi slt, %10, %28 : tensor<128x1xi32>
    %30 = tt.splat %arg5 : i32 -> tensor<1x242xi32>
    %31 = arith.cmpi slt, %15, %30 : tensor<1x242xi32>
    %32 = tt.broadcast %29 : tensor<128x1xi1> -> tensor<128x242xi1>
    %33 = tt.broadcast %31 : tensor<1x242xi1> -> tensor<128x242xi1>
    %34 = arith.andi %32, %33 : tensor<128x242xi1>
    %35 = tt.load %18, %34, %cst_0 : tensor<128x242x!tt.ptr<i8>>
    %36 = tt.splat %arg5 : i32 -> tensor<242x1xi32>
    %37 = arith.cmpi slt, %19, %36 : tensor<242x1xi32>
    %38 = tt.splat %arg4 : i32 -> tensor<1x128xi32>
    %39 = arith.cmpi slt, %24, %38 : tensor<1x128xi32>
    %40 = tt.broadcast %37 : tensor<242x1xi1> -> tensor<242x128xi1>
    %41 = tt.broadcast %39 : tensor<1x128xi1> -> tensor<242x128xi1>
    %42 = arith.andi %40, %41 : tensor<242x128xi1>
    %43 = tt.load %27, %42, %cst : tensor<242x128x!tt.ptr<i8>>
    %44 = tt.dot %35, %43, %cst_1 : tensor<128x242xi8> * tensor<242x128xi8> -> tensor<128x128xi32>
    annotation.mark %44 {enable_i4} : tensor<128x128xi32>
    tt.print " acc: " {hex = false, isSigned = array<i32: 1>} : %44 : tensor<128x128xi32>
    %45 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %46 = arith.muli %10, %45 : tensor<128x1xi32>
    %47 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>>
    %48 = tt.addptr %47, %46 : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>
    %49 = tt.broadcast %48 : tensor<128x1x!tt.ptr<i32>> -> tensor<128x128x!tt.ptr<i32>>
    %50 = tt.broadcast %24 : tensor<1x128xi32> -> tensor<128x128xi32>
    %51 = tt.addptr %49, %50 : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
    %52 = tt.broadcast %29 : tensor<128x1xi1> -> tensor<128x128xi1>
    %53 = tt.broadcast %39 : tensor<1x128xi1> -> tensor<128x128xi1>
    %54 = arith.andi %52, %53 : tensor<128x128xi1>
    tt.store %51, %44, %54 : tensor<128x128x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK-LABEL: func.func @matmul_kernel
// CHECK: linalg.fill
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: linalg.matmul
// CHECK: annotation.mark %{{[0-9]+}} {enable_i4} : tensor<128x128xi32>
// CHECK: call @triton_print{{[_0-9]*}}
// CHECK: bufferization.materialize_in_destination
// CHECK: return
