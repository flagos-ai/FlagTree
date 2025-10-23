// RUN: triton-adapter-opt --discrete-mask-access-conversion "--triton-to-linalg=global-kernel=false named-ops=True" %s | FileCheck %s

module {
  tt.func public @linear_compress_bwd_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  // CHECK-LABEL:   func.func @linear_compress_bwd_kernel
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32x4xf16>
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xf32>
    %cst_2 = arith.constant dense<128> : tensor<32xi32>
    %cst_3 = arith.constant dense<32> : tensor<32xi32>
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %cst_4 = arith.constant dense<128> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %cst_5 = arith.constant dense<16> : tensor<16xi32>
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c8_i32 : i32
    %2 = arith.remsi %0, %c8_i32 : i32
    %3 = tt.get_program_id y : i32
    %4 = tt.get_program_id z : i32
    %5 = arith.divsi %4, %c32_i32 : i32
    %6 = arith.remsi %4, %c32_i32 : i32
    %7 = tt.addptr %arg5, %1 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = tt.addptr %7, %c1_i32 : !tt.ptr<i32>, i32
    %10 = tt.load %9 : !tt.ptr<i32>
    %11 = arith.subi %10, %8 : i32
    %12 = tt.addptr %arg6, %1 : !tt.ptr<i32>, i32
    %13 = tt.load %12 : !tt.ptr<i32>
    %14 = tt.addptr %12, %c1_i32 : !tt.ptr<i32>, i32
    %15 = tt.load %14 : !tt.ptr<i32>
    %16 = arith.subi %15, %13 : i32
    %17 = arith.muli %3, %c16_i32 : i32
    %18 = arith.cmpi sge, %17, %16 : i32
    cf.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %19 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %20 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %21 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %22 = arith.muli %2, %arg8 : i32
    %23 = tt.addptr %arg3, %22 : !tt.ptr<f16>, i32
    %24 = arith.muli %8, %arg7 : i32
    %25 = tt.addptr %23, %24 : !tt.ptr<f16>, i32
    %26 = arith.muli %3, %c256_i32 : i32
    %27 = arith.muli %21, %cst_5 : tensor<16xi32>
    %28 = tt.splat %26 : i32 -> tensor<16xi32>
    %29 = arith.addi %28, %27 : tensor<16xi32>
    %30 = tt.expand_dims %29 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %31 = tt.expand_dims %19 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %32 = tt.broadcast %30 : tensor<16x1xi32> -> tensor<16x32xi32>
    %33 = tt.broadcast %31 : tensor<1x32xi32> -> tensor<16x32xi32>
    %34 = arith.addi %32, %33 : tensor<16x32xi32>
    %35 = tt.expand_dims %34 {axis = 2 : i32} : tensor<16x32xi32> -> tensor<16x32x1xi32>
    %36 = tt.splat %arg7 : i32 -> tensor<16x32x1xi32>
    %37 = arith.muli %35, %36 : tensor<16x32x1xi32>
    %38 = tt.splat %25 : !tt.ptr<f16> -> tensor<16x32x1x!tt.ptr<f16>>
    %39 = tt.addptr %38, %37 : tensor<16x32x1x!tt.ptr<f16>>, tensor<16x32x1xi32>
    %40 = arith.muli %6, %c4_i32 : i32
    %41 = tt.splat %40 : i32 -> tensor<4xi32>
    %42 = arith.addi %41, %20 : tensor<4xi32>
    %43 = tt.expand_dims %42 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %45 = tt.broadcast %39 : tensor<16x32x1x!tt.ptr<f16>> -> tensor<16x32x4x!tt.ptr<f16>>
    %46 = tt.broadcast %44 : tensor<1x1x4xi32> -> tensor<16x32x4xi32>
    %47 = tt.addptr %45, %46 : tensor<16x32x4x!tt.ptr<f16>>, tensor<16x32x4xi32>
    %48 = tt.splat %11 : i32 -> tensor<16x32xi32>
    %49 = arith.cmpi slt, %34, %48 : tensor<16x32xi32>
    %50 = tt.expand_dims %49 {axis = 2 : i32} : tensor<16x32xi1> -> tensor<16x32x1xi1>
    %51 = arith.cmpi slt, %42, %cst_4 : tensor<4xi32>
    %52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<4xi1> -> tensor<1x4xi1>
    %53 = tt.expand_dims %52 {axis = 1 : i32} : tensor<1x4xi1> -> tensor<1x1x4xi1>
    %54 = tt.broadcast %50 : tensor<16x32x1xi1> -> tensor<16x32x4xi1>
    %55 = tt.broadcast %53 : tensor<1x1x4xi1> -> tensor<16x32x4xi1>
    %56 = arith.andi %54, %55 : tensor<16x32x4xi1>
    %57 = arith.muli %2, %arg13 : i32
    %58 = tt.addptr %arg0, %57 : !tt.ptr<f32>, i32
    %59 = arith.muli %8, %arg12 : i32
    %60 = tt.addptr %58, %59 : !tt.ptr<f32>, i32
    %61 = tt.splat %arg12 : i32 -> tensor<16x32x1xi32>
    %62 = arith.muli %35, %61 : tensor<16x32x1xi32>
    %63 = tt.splat %60 : !tt.ptr<f32> -> tensor<16x32x1x!tt.ptr<f32>>
    %64 = tt.addptr %63, %62 : tensor<16x32x1x!tt.ptr<f32>>, tensor<16x32x1xi32>
    %65 = tt.broadcast %64 : tensor<16x32x1x!tt.ptr<f32>> -> tensor<16x32x4x!tt.ptr<f32>>
    %66 = tt.addptr %65, %46 : tensor<16x32x4x!tt.ptr<f32>>, tensor<16x32x4xi32>
    %67 = arith.muli %2, %arg9 : i32
    %68 = tt.addptr %arg4, %67 : !tt.ptr<f16>, i32
    %69 = arith.muli %5, %c32_i32 : i32
    %70 = arith.extsi %arg10 : i32 to i64
    %71 = arith.extsi %arg11 : i32 to i64
    %72 = tt.make_tensor_ptr %68, [%c32_i64, %c128_i64, %c128_i64], [%70, %71, %c1_i64], [%c0_i32, %40, %69] {order = array<i32: 2, 1, 0>} : <tensor<32x4x32xf16>>
    %73 = arith.muli %2, %arg14 : i32
    %74 = tt.addptr %arg2, %73 : !tt.ptr<f32>, i32
    %75 = tt.expand_dims %19 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %76 = tt.expand_dims %75 {axis = 2 : i32} : tensor<32x1xi32> -> tensor<32x1x1xi32>
    %77 = tt.splat %arg15 : i32 -> tensor<32x1x1xi32>
    %78 = arith.muli %76, %77 : tensor<32x1x1xi32>
    %79 = tt.splat %74 : !tt.ptr<f32> -> tensor<32x1x1x!tt.ptr<f32>>
    %80 = tt.addptr %79, %78 : tensor<32x1x1x!tt.ptr<f32>>, tensor<32x1x1xi32>
    %81 = tt.expand_dims %43 {axis = 2 : i32} : tensor<1x4xi32> -> tensor<1x4x1xi32>
    %82 = tt.splat %arg16 : i32 -> tensor<1x4x1xi32>
    %83 = arith.muli %81, %82 : tensor<1x4x1xi32>
    %84 = tt.broadcast %80 : tensor<32x1x1x!tt.ptr<f32>> -> tensor<32x4x1x!tt.ptr<f32>>
    %85 = tt.broadcast %83 : tensor<1x4x1xi32> -> tensor<32x4x1xi32>
    %86 = tt.addptr %84, %85 : tensor<32x4x1x!tt.ptr<f32>>, tensor<32x4x1xi32>
    %87 = tt.splat %69 : i32 -> tensor<32xi32>
    %88 = arith.addi %87, %19 : tensor<32xi32>
    %89 = tt.expand_dims %88 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %90 = tt.expand_dims %89 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32>
    %91 = tt.broadcast %86 : tensor<32x4x1x!tt.ptr<f32>> -> tensor<32x4x32x!tt.ptr<f32>>
    %92 = tt.broadcast %90 : tensor<1x1x32xi32> -> tensor<32x4x32xi32>
    %93 = tt.addptr %91, %92 : tensor<32x4x32x!tt.ptr<f32>>, tensor<32x4x32xi32>
    %94 = arith.cmpi slt, %19, %cst_3 : tensor<32xi32>
    %95 = tt.expand_dims %94 {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>
    %96 = tt.expand_dims %95 {axis = 2 : i32} : tensor<32x1xi1> -> tensor<32x1x1xi1>
    %97 = tt.expand_dims %52 {axis = 2 : i32} : tensor<1x4xi1> -> tensor<1x4x1xi1>
    %98 = tt.broadcast %96 : tensor<32x1x1xi1> -> tensor<32x4x1xi1>
    %99 = tt.broadcast %97 : tensor<1x4x1xi1> -> tensor<32x4x1xi1>
    %100 = arith.andi %98, %99 : tensor<32x4x1xi1>
    %101 = arith.cmpi slt, %88, %cst_2 : tensor<32xi32>
    %102 = tt.expand_dims %101 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
    %103 = tt.expand_dims %102 {axis = 1 : i32} : tensor<1x32xi1> -> tensor<1x1x32xi1>
    %104 = tt.broadcast %100 : tensor<32x4x1xi1> -> tensor<32x4x32xi1>
    %105 = tt.broadcast %103 : tensor<1x1x32xi1> -> tensor<32x4x32xi1>
    %106 = arith.andi %104, %105 : tensor<32x4x32xi1>
    %107 = arith.muli %13, %arg17 : i32
    %108 = tt.addptr %arg1, %107 : !tt.ptr<f16>, i32
    %109 = arith.muli %2, %arg18 : i32
    %110 = tt.addptr %108, %109 : !tt.ptr<f16>, i32
    %111 = arith.extsi %16 : i32 to i64
    %112 = arith.extsi %arg17 : i32 to i64
    %113 = tt.make_tensor_ptr %110, [%111, %c128_i64], [%112, %c1_i64], [%17, %69] {order = array<i32: 1, 0>} : <tensor<16x32xf16>>
    %114 = tt.load %113 {boundaryCheck = array<i32: 0, 1>, padding = 1 : i32} : !tt.ptr<tensor<16x32xf16>>
    %115 = tt.load %72 {boundaryCheck = array<i32: 0, 1, 2>, padding = 1 : i32} : !tt.ptr<tensor<32x4x32xf16>>
    %116 = tt.reshape %115 : tensor<32x4x32xf16> -> tensor<128x32xf16>
    tt.annotation %116 {maybeUnCollapsibleReshape} : tensor<128x32xf16>
    %117 = tt.trans %116 {order = array<i32: 1, 0>} : tensor<128x32xf16> -> tensor<32x128xf16>
    %118 = tt.dot %114, %117, %cst_1 : tensor<16x32xf16> * tensor<32x128xf16> -> tensor<16x128xf32>
    %119 = tt.reshape %118 : tensor<16x128xf32> -> tensor<16x32x4xf32>
    %120 = tt.atomic_rmw fadd, acq_rel, gpu, %66, %119, %56 : (tensor<16x32x4x!tt.ptr<f32>>, tensor<16x32x4xf32>, tensor<16x32x4xi1>) -> tensor<16x32x4xf32>
    // CHECK:           [[VAR0:%[a-zA-Z0-9_]+]] = tensor.reshape {{%[a-zA-Z0-9_]+}}({{%[a-zA-Z0-9_]+}}) : (tensor<16x128xf32>, tensor<3xi64>) -> tensor<16x32x4xf32>
    // CHECK-NEXT:      [[VAR1:%[a-zA-Z0-9_]+]] = arith.select {{%[a-zA-Z0-9_]+}}, [[VAR0]], {{%[a-zA-Z0-9_]+}} : tensor<16x32x4xi1>, tensor<16x32x4xf32>
    // CHECK-NEXT:      [[VAR2:%[a-zA-Z0-9_]+]] = bufferization.to_memref [[VAR1]] : memref<16x32x4xf32>
    // CHECK-NEXT:      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins({{%[a-zA-Z0-9_]+}}, [[VAR2]] : 
    %121 = tt.load %47, %56, %cst : tensor<16x32x4x!tt.ptr<f16>>
    %122 = tt.reshape %121 : tensor<16x32x4xf16> -> tensor<16x128xf16>
    tt.annotation %122 {maybeUnCollapsibleReshape} : tensor<16x128xf16>
    %123 = tt.trans %122 {order = array<i32: 1, 0>} : tensor<16x128xf16> -> tensor<128x16xf16>
    %124 = tt.dot %123, %114, %cst_0 : tensor<128x16xf16> * tensor<16x32xf16> -> tensor<128x32xf32>
    %125 = tt.reshape %124 : tensor<128x32xf32> -> tensor<32x4x32xf32>
    %126 = tt.atomic_rmw fadd, acq_rel, gpu, %93, %125, %106 : (tensor<32x4x32x!tt.ptr<f32>>, tensor<32x4x32xf32>, tensor<32x4x32xi1>) -> tensor<32x4x32xf32>
    tt.return
  }
}
