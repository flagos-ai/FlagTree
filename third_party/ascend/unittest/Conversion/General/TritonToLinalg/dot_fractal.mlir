// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

module {
  tt.func public @case1_a_fractal(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<80> : tensor<160x1xi32>
    %cst_0 = arith.constant dense<80> : tensor<320x1xi32>
    %cst_1 = arith.constant dense<16> : tensor<1x1x16x1xi32>
    %cst_2 = arith.constant dense<256> : tensor<1x10x1x1xi32>
    %cst_3 = arith.constant dense<2560> : tensor<20x1x1x1xi32>
    %0 = tt.make_range {end = 20 : i32, start = 0 : i32} : tensor<20xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<20xi32> -> tensor<20x1xi32>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<20x1xi32> -> tensor<20x1x1xi32>
    %3 = tt.expand_dims %2 {axis = 3 : i32} : tensor<20x1x1xi32> -> tensor<20x1x1x1xi32>
    %4 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<10xi32> -> tensor<1x10xi32>
    %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<1x10xi32> -> tensor<1x10x1xi32>
    %7 = tt.expand_dims %6 {axis = 3 : i32} : tensor<1x10x1xi32> -> tensor<1x10x1x1xi32>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %11 = tt.expand_dims %10 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %12 = tt.expand_dims %10 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %13 = arith.muli %3, %cst_3 : tensor<20x1x1x1xi32>
    %14 = arith.muli %7, %cst_2 : tensor<1x10x1x1xi32>
    %15 = tt.broadcast %13 : tensor<20x1x1x1xi32> -> tensor<20x10x1x1xi32>
    %16 = tt.broadcast %14 : tensor<1x10x1x1xi32> -> tensor<20x10x1x1xi32>
    %17 = arith.addi %15, %16 : tensor<20x10x1x1xi32>
    %18 = arith.muli %11, %cst_1 : tensor<1x1x16x1xi32>
    %19 = tt.broadcast %17 : tensor<20x10x1x1xi32> -> tensor<20x10x16x1xi32>
    %20 = tt.broadcast %18 : tensor<1x1x16x1xi32> -> tensor<20x10x16x1xi32>
    %21 = arith.addi %19, %20 : tensor<20x10x16x1xi32>
    %22 = tt.broadcast %21 : tensor<20x10x16x1xi32> -> tensor<20x10x16x16xi32>
    %23 = tt.broadcast %12 : tensor<1x1x1x16xi32> -> tensor<20x10x16x16xi32>
    %24 = arith.addi %22, %23 : tensor<20x10x16x16xi32>
    %25 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<20x10x16x16x!tt.ptr<f16>>
    %26 = tt.addptr %25, %24 : tensor<20x10x16x16x!tt.ptr<f16>>, tensor<20x10x16x16xi32>
    %27 = tt.load %26 : tensor<20x10x16x16x!tt.ptr<f16>>
    %28 = tt.make_range {end = 320 : i32, start = 0 : i32} : tensor<320xi32>
    %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<320xi32> -> tensor<320x1xi32>
    %30 = arith.muli %29, %cst_0 : tensor<320x1xi32>
    %31 = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32>
    %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<80xi32> -> tensor<1x80xi32>
    %33 = tt.broadcast %30 : tensor<320x1xi32> -> tensor<320x80xi32>
    %34 = tt.broadcast %32 : tensor<1x80xi32> -> tensor<320x80xi32>
    %35 = arith.addi %33, %34 : tensor<320x80xi32>
    %36 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<320x80x!tt.ptr<f16>>
    %37 = tt.addptr %36, %35 : tensor<320x80x!tt.ptr<f16>>, tensor<320x80xi32>
    %38 = tt.load %37 : tensor<320x80x!tt.ptr<f16>>
    %39 = ascend.dot %27, %38 {fractal_a = true} : tensor<20x10x16x16xf16>, tensor<320x80xf16> -> tensor<160x80xf32>
    %40 = tt.make_range {end = 160 : i32, start = 0 : i32} : tensor<160xi32>
    %41 = tt.expand_dims %40 {axis = 1 : i32} : tensor<160xi32> -> tensor<160x1xi32>
    %42 = arith.muli %41, %cst : tensor<160x1xi32>
    %43 = tt.broadcast %42 : tensor<160x1xi32> -> tensor<160x80xi32>
    %44 = tt.broadcast %32 : tensor<1x80xi32> -> tensor<160x80xi32>
    %45 = arith.addi %43, %44 : tensor<160x80xi32>
    %46 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<160x80x!tt.ptr<f16>>
    %47 = tt.addptr %46, %45 : tensor<160x80x!tt.ptr<f16>>, tensor<160x80xi32>
    %48 = arith.truncf %39 : tensor<160x80xf32> to tensor<160x80xf16>
    tt.store %47, %48 : tensor<160x80x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @case1_a_fractal
// CHECK-SAME:    mix_mode = "mix"
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<20x10x16x16xf16>) -> tensor<160x320xf16>
// CHECK:         %[[ACC:.*]] = tensor.empty() : tensor<160x80xf32>
// CHECK:         linalg.matmul {input_precision = "ieee"} ins(%{{.*}}, %{{.*}} : tensor<160x320xf16>, tensor<320x80xf16>) outs(%[[ACC]] : tensor<160x80xf32>) -> tensor<160x80xf32>
// ND result: no ND->Fractal convert afterwards.
// CHECK-NOT:     hivm.hir.convert_layout

// -----

module {
  tt.func public @s_C1_pure_cube(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<2560> : tensor<5x1x1x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x20x1x1xi32>
    %cst_1 = arith.constant dense<5120> : tensor<5x1x1x1xi32>
    %cst_2 = arith.constant dense<16> : tensor<1x1x16x1xi32>
    %cst_3 = arith.constant dense<256> : tensor<1x10x1x1xi32>
    %cst_4 = arith.constant dense<2560> : tensor<20x1x1x1xi32>
    %0 = tt.make_range {end = 20 : i32, start = 0 : i32} : tensor<20xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<20xi32> -> tensor<20x1xi32>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<20x1xi32> -> tensor<20x1x1xi32>
    %3 = tt.expand_dims %2 {axis = 3 : i32} : tensor<20x1x1xi32> -> tensor<20x1x1x1xi32>
    %4 = arith.muli %3, %cst_4 : tensor<20x1x1x1xi32>
    %5 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<10xi32> -> tensor<1x10xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x10xi32> -> tensor<1x10x1xi32>
    %8 = tt.expand_dims %7 {axis = 3 : i32} : tensor<1x10x1xi32> -> tensor<1x10x1x1xi32>
    %9 = arith.muli %8, %cst_3 : tensor<1x10x1x1xi32>
    %10 = tt.broadcast %4 : tensor<20x1x1x1xi32> -> tensor<20x10x1x1xi32>
    %11 = tt.broadcast %9 : tensor<1x10x1x1xi32> -> tensor<20x10x1x1xi32>
    %12 = arith.addi %10, %11 : tensor<20x10x1x1xi32>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %17 = arith.muli %16, %cst_2 : tensor<1x1x16x1xi32>
    %18 = tt.broadcast %12 : tensor<20x10x1x1xi32> -> tensor<20x10x16x1xi32>
    %19 = tt.broadcast %17 : tensor<1x1x16x1xi32> -> tensor<20x10x16x1xi32>
    %20 = arith.addi %18, %19 : tensor<20x10x16x1xi32>
    %21 = tt.expand_dims %15 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %22 = tt.broadcast %20 : tensor<20x10x16x1xi32> -> tensor<20x10x16x16xi32>
    %23 = tt.broadcast %21 : tensor<1x1x1x16xi32> -> tensor<20x10x16x16xi32>
    %24 = arith.addi %22, %23 : tensor<20x10x16x16xi32>
    %25 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<20x10x16x16x!tt.ptr<f16>>
    %26 = tt.addptr %25, %24 : tensor<20x10x16x16x!tt.ptr<f16>>, tensor<20x10x16x16xi32>
    %27 = tt.load %26 : tensor<20x10x16x16x!tt.ptr<f16>>
    %28 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
    %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<5xi32> -> tensor<5x1xi32>
    %30 = tt.expand_dims %29 {axis = 2 : i32} : tensor<5x1xi32> -> tensor<5x1x1xi32>
    %31 = tt.expand_dims %30 {axis = 3 : i32} : tensor<5x1x1xi32> -> tensor<5x1x1x1xi32>
    %32 = arith.muli %31, %cst_1 : tensor<5x1x1x1xi32>
    %33 = tt.expand_dims %0 {axis = 0 : i32} : tensor<20xi32> -> tensor<1x20xi32>
    %34 = tt.expand_dims %33 {axis = 2 : i32} : tensor<1x20xi32> -> tensor<1x20x1xi32>
    %35 = tt.expand_dims %34 {axis = 3 : i32} : tensor<1x20x1xi32> -> tensor<1x20x1x1xi32>
    %36 = arith.muli %35, %cst_0 : tensor<1x20x1x1xi32>
    %37 = tt.broadcast %32 : tensor<5x1x1x1xi32> -> tensor<5x20x1x1xi32>
    %38 = tt.broadcast %36 : tensor<1x20x1x1xi32> -> tensor<5x20x1x1xi32>
    %39 = arith.addi %37, %38 : tensor<5x20x1x1xi32>
    %40 = tt.broadcast %39 : tensor<5x20x1x1xi32> -> tensor<5x20x16x1xi32>
    %41 = tt.broadcast %17 : tensor<1x1x16x1xi32> -> tensor<5x20x16x1xi32>
    %42 = arith.addi %40, %41 : tensor<5x20x16x1xi32>
    %43 = tt.broadcast %42 : tensor<5x20x16x1xi32> -> tensor<5x20x16x16xi32>
    %44 = tt.broadcast %21 : tensor<1x1x1x16xi32> -> tensor<5x20x16x16xi32>
    %45 = arith.addi %43, %44 : tensor<5x20x16x16xi32>
    %46 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<5x20x16x16x!tt.ptr<f16>>
    %47 = tt.addptr %46, %45 : tensor<5x20x16x16x!tt.ptr<f16>>, tensor<5x20x16x16xi32>
    %48 = tt.load %47 : tensor<5x20x16x16x!tt.ptr<f16>>
    %49 = ascend.dot %27, %48 {fractal_a = true, fractal_b = true, fractal_c = true} : tensor<20x10x16x16xf16>, tensor<5x20x16x16xf16> -> tensor<5x10x16x16xf32>
    %50 = arith.muli %31, %cst : tensor<5x1x1x1xi32>
    %51 = tt.broadcast %50 : tensor<5x1x1x1xi32> -> tensor<5x10x1x1xi32>
    %52 = tt.broadcast %9 : tensor<1x10x1x1xi32> -> tensor<5x10x1x1xi32>
    %53 = arith.addi %51, %52 : tensor<5x10x1x1xi32>
    %54 = tt.broadcast %53 : tensor<5x10x1x1xi32> -> tensor<5x10x16x1xi32>
    %55 = tt.broadcast %17 : tensor<1x1x16x1xi32> -> tensor<5x10x16x1xi32>
    %56 = arith.addi %54, %55 : tensor<5x10x16x1xi32>
    %57 = tt.broadcast %56 : tensor<5x10x16x1xi32> -> tensor<5x10x16x16xi32>
    %58 = tt.broadcast %21 : tensor<1x1x1x16xi32> -> tensor<5x10x16x16xi32>
    %59 = arith.addi %57, %58 : tensor<5x10x16x16xi32>
    %60 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<5x10x16x16x!tt.ptr<f16>>
    %61 = tt.addptr %60, %59 : tensor<5x10x16x16x!tt.ptr<f16>>, tensor<5x10x16x16xi32>
    %62 = arith.truncf %49 : tensor<5x10x16x16xf32> to tensor<5x10x16x16xf16>
    tt.store %61, %62 : tensor<5x10x16x16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @s_C1_pure_cube
// CHECK-SAME:    mix_mode = "mix"
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<20x10x16x16xf16>) -> tensor<160x320xf16>
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [320, 80] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>} : (tensor<5x20x16x16xf16>) -> tensor<320x80xf16>
// CHECK:         tensor.empty() : tensor<160x80xf32>
// CHECK:         %[[MM:.*]] = linalg.matmul {input_precision = "ieee"} ins(%{{.*}}, %{{.*}} : tensor<160x320xf16>, tensor<320x80xf16>) outs(%{{.*}} : tensor<160x80xf32>) -> tensor<160x80xf32>
// fractal_c: ND [M,N] -> [N/16, M/16, 16, 16], carrying the f32 accumulator dtype.
// CHECK:         hivm.hir.convert_layout %[[MM]] output_shape [5, 10, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x80xf32>) -> tensor<5x10x16x16xf32>

// -----

module {
  tt.func public @s_C_bothND_fracC(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<1x1x16x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x10x1x1xi32>
    %cst_1 = arith.constant dense<2560> : tensor<5x1x1x1xi32>
    %cst_2 = arith.constant dense<80> : tensor<320x1xi32>
    %cst_3 = arith.constant dense<320> : tensor<160x1xi32>
    %0 = tt.make_range {end = 160 : i32, start = 0 : i32} : tensor<160xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<160xi32> -> tensor<160x1xi32>
    %2 = arith.muli %1, %cst_3 : tensor<160x1xi32>
    %3 = tt.make_range {end = 320 : i32, start = 0 : i32} : tensor<320xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<320xi32> -> tensor<1x320xi32>
    %5 = tt.broadcast %2 : tensor<160x1xi32> -> tensor<160x320xi32>
    %6 = tt.broadcast %4 : tensor<1x320xi32> -> tensor<160x320xi32>
    %7 = arith.addi %5, %6 : tensor<160x320xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<160x320x!tt.ptr<f16>>
    %9 = tt.addptr %8, %7 : tensor<160x320x!tt.ptr<f16>>, tensor<160x320xi32>
    %10 = tt.load %9 : tensor<160x320x!tt.ptr<f16>>
    %11 = tt.expand_dims %3 {axis = 1 : i32} : tensor<320xi32> -> tensor<320x1xi32>
    %12 = arith.muli %11, %cst_2 : tensor<320x1xi32>
    %13 = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<80xi32> -> tensor<1x80xi32>
    %15 = tt.broadcast %12 : tensor<320x1xi32> -> tensor<320x80xi32>
    %16 = tt.broadcast %14 : tensor<1x80xi32> -> tensor<320x80xi32>
    %17 = arith.addi %15, %16 : tensor<320x80xi32>
    %18 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<320x80x!tt.ptr<f16>>
    %19 = tt.addptr %18, %17 : tensor<320x80x!tt.ptr<f16>>, tensor<320x80xi32>
    %20 = tt.load %19 : tensor<320x80x!tt.ptr<f16>>
    %21 = ascend.dot %10, %20 {fractal_c = true} : tensor<160x320xf16>, tensor<320x80xf16> -> tensor<5x10x16x16xf32>
    %22 = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<5xi32> -> tensor<5x1xi32>
    %24 = tt.expand_dims %23 {axis = 2 : i32} : tensor<5x1xi32> -> tensor<5x1x1xi32>
    %25 = tt.expand_dims %24 {axis = 3 : i32} : tensor<5x1x1xi32> -> tensor<5x1x1x1xi32>
    %26 = arith.muli %25, %cst_1 : tensor<5x1x1x1xi32>
    %27 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %28 = tt.expand_dims %27 {axis = 0 : i32} : tensor<10xi32> -> tensor<1x10xi32>
    %29 = tt.expand_dims %28 {axis = 2 : i32} : tensor<1x10xi32> -> tensor<1x10x1xi32>
    %30 = tt.expand_dims %29 {axis = 3 : i32} : tensor<1x10x1xi32> -> tensor<1x10x1x1xi32>
    %31 = arith.muli %30, %cst_0 : tensor<1x10x1x1xi32>
    %32 = tt.broadcast %26 : tensor<5x1x1x1xi32> -> tensor<5x10x1x1xi32>
    %33 = tt.broadcast %31 : tensor<1x10x1x1xi32> -> tensor<5x10x1x1xi32>
    %34 = arith.addi %32, %33 : tensor<5x10x1x1xi32>
    %35 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %36 = tt.expand_dims %35 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %38 = tt.expand_dims %37 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %39 = arith.muli %38, %cst : tensor<1x1x16x1xi32>
    %40 = tt.broadcast %34 : tensor<5x10x1x1xi32> -> tensor<5x10x16x1xi32>
    %41 = tt.broadcast %39 : tensor<1x1x16x1xi32> -> tensor<5x10x16x1xi32>
    %42 = arith.addi %40, %41 : tensor<5x10x16x1xi32>
    %43 = tt.expand_dims %37 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %44 = tt.broadcast %42 : tensor<5x10x16x1xi32> -> tensor<5x10x16x16xi32>
    %45 = tt.broadcast %43 : tensor<1x1x1x16xi32> -> tensor<5x10x16x16xi32>
    %46 = arith.addi %44, %45 : tensor<5x10x16x16xi32>
    %47 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<5x10x16x16x!tt.ptr<f16>>
    %48 = tt.addptr %47, %46 : tensor<5x10x16x16x!tt.ptr<f16>>, tensor<5x10x16x16xi32>
    %49 = arith.truncf %21 : tensor<5x10x16x16xf32> to tensor<5x10x16x16xf16>
    tt.store %48, %49 : tensor<5x10x16x16x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @s_C_bothND_fracC
// CHECK-SAME:    mix_mode = "mix"
// ND operands: no Fractal->ND convert is emitted before the matmul.
// CHECK-NOT:     hivm.hir.convert_layout
// CHECK:         tensor.empty() : tensor<160x80xf32>
// CHECK:         %[[MM:.*]] = linalg.matmul {input_precision = "ieee"} ins(%{{.*}}, %{{.*}} : tensor<160x320xf16>, tensor<320x80xf16>) outs(%{{.*}} : tensor<160x80xf32>) -> tensor<160x80xf32>
// CHECK:         hivm.hir.convert_layout %[[MM]] output_shape [5, 10, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x80xf32>) -> tensor<5x10x16x16xf32>

// -----

module {
  tt.func public @s_mix_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<80> : tensor<160x1xi32>
    %cst_0 = arith.constant dense<80> : tensor<320x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<1x1x16x1xi32>
    %cst_2 = arith.constant dense<128> : tensor<1x10x1x1xi32>
    %cst_3 = arith.constant dense<1280> : tensor<40x1x1x1xi32>
    %0 = tt.make_range {end = 40 : i32, start = 0 : i32} : tensor<40xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<40xi32> -> tensor<40x1xi32>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<40x1xi32> -> tensor<40x1x1xi32>
    %3 = tt.expand_dims %2 {axis = 3 : i32} : tensor<40x1x1xi32> -> tensor<40x1x1x1xi32>
    %4 = arith.muli %3, %cst_3 : tensor<40x1x1x1xi32>
    %5 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<10xi32> -> tensor<1x10xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x10xi32> -> tensor<1x10x1xi32>
    %8 = tt.expand_dims %7 {axis = 3 : i32} : tensor<1x10x1xi32> -> tensor<1x10x1x1xi32>
    %9 = arith.muli %8, %cst_2 : tensor<1x10x1x1xi32>
    %10 = tt.broadcast %4 : tensor<40x1x1x1xi32> -> tensor<40x10x1x1xi32>
    %11 = tt.broadcast %9 : tensor<1x10x1x1xi32> -> tensor<40x10x1x1xi32>
    %12 = arith.addi %10, %11 : tensor<40x10x1x1xi32>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %15 = tt.expand_dims %14 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %17 = arith.muli %16, %cst_1 : tensor<1x1x16x1xi32>
    %18 = tt.broadcast %12 : tensor<40x10x1x1xi32> -> tensor<40x10x16x1xi32>
    %19 = tt.broadcast %17 : tensor<1x1x16x1xi32> -> tensor<40x10x16x1xi32>
    %20 = arith.addi %18, %19 : tensor<40x10x16x1xi32>
    %21 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %22 = tt.expand_dims %21 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %23 = tt.expand_dims %22 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %24 = tt.expand_dims %23 {axis = 2 : i32} : tensor<1x1x8xi32> -> tensor<1x1x1x8xi32>
    %25 = tt.broadcast %20 : tensor<40x10x16x1xi32> -> tensor<40x10x16x8xi32>
    %26 = tt.broadcast %24 : tensor<1x1x1x8xi32> -> tensor<40x10x16x8xi32>
    %27 = arith.addi %25, %26 : tensor<40x10x16x8xi32>
    %28 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<40x10x16x8x!tt.ptr<f32>>
    %29 = tt.addptr %28, %27 : tensor<40x10x16x8x!tt.ptr<f32>>, tensor<40x10x16x8xi32>
    %30 = tt.load %29 : tensor<40x10x16x8x!tt.ptr<f32>>
    %31 = tt.make_range {end = 320 : i32, start = 0 : i32} : tensor<320xi32>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<320xi32> -> tensor<320x1xi32>
    %33 = arith.muli %32, %cst_0 : tensor<320x1xi32>
    %34 = tt.make_range {end = 80 : i32, start = 0 : i32} : tensor<80xi32>
    %35 = tt.expand_dims %34 {axis = 0 : i32} : tensor<80xi32> -> tensor<1x80xi32>
    %36 = tt.broadcast %33 : tensor<320x1xi32> -> tensor<320x80xi32>
    %37 = tt.broadcast %35 : tensor<1x80xi32> -> tensor<320x80xi32>
    %38 = arith.addi %36, %37 : tensor<320x80xi32>
    %39 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<320x80x!tt.ptr<f32>>
    %40 = tt.addptr %39, %38 : tensor<320x80x!tt.ptr<f32>>, tensor<320x80xi32>
    %41 = tt.load %40 : tensor<320x80x!tt.ptr<f32>>
    %42 = ascend.dot %30, %41 {fractal_a = true} : tensor<40x10x16x8xf32>, tensor<320x80xf32> -> tensor<160x80xf32>
    %43 = tt.make_range {end = 160 : i32, start = 0 : i32} : tensor<160xi32>
    %44 = tt.expand_dims %43 {axis = 1 : i32} : tensor<160xi32> -> tensor<160x1xi32>
    %45 = arith.muli %44, %cst : tensor<160x1xi32>
    %46 = tt.broadcast %45 : tensor<160x1xi32> -> tensor<160x80xi32>
    %47 = tt.broadcast %35 : tensor<1x80xi32> -> tensor<160x80xi32>
    %48 = arith.addi %46, %47 : tensor<160x80xi32>
    %49 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<160x80x!tt.ptr<f32>>
    %50 = tt.addptr %49, %48 : tensor<160x80x!tt.ptr<f32>>, tensor<160x80xi32>
    tt.store %50, %42 : tensor<160x80x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @s_mix_f32
// CHECK-SAME:    mix_mode = "mix"
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 8]>} : (tensor<40x10x16x8xf32>) -> tensor<160x320xf32>
// CHECK:         tensor.empty() : tensor<160x80xf32>
// CHECK:         linalg.matmul {input_precision = "ieee"} ins(%{{.*}}, %{{.*}} : tensor<160x320xf32>, tensor<320x80xf32>) outs(%{{.*}} : tensor<160x80xf32>) -> tensor<160x80xf32>

// -----

module {
  tt.func public @s_C_int8(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<1x1x16x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x10x1x1xi32>
    %cst_1 = arith.constant dense<2560> : tensor<4x1x1x1xi32>
    %cst_2 = arith.constant dense<32> : tensor<1x1x32x1xi32>
    %cst_3 = arith.constant dense<1024> : tensor<1x10x1x1xi32>
    %cst_4 = arith.constant dense<10240> : tensor<2x1x1x1xi32>
    %cst_5 = arith.constant dense<32> : tensor<1x1x16x1xi32>
    %cst_6 = arith.constant dense<512> : tensor<1x10x1x1xi32>
    %cst_7 = arith.constant dense<5120> : tensor<10x1x1x1xi32>
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<10xi32> -> tensor<10x1xi32>
    %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<10x1xi32> -> tensor<10x1x1xi32>
    %3 = tt.expand_dims %2 {axis = 3 : i32} : tensor<10x1x1xi32> -> tensor<10x1x1x1xi32>
    %4 = arith.muli %3, %cst_7 : tensor<10x1x1x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<10xi32> -> tensor<1x10xi32>
    %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<1x10xi32> -> tensor<1x10x1xi32>
    %7 = tt.expand_dims %6 {axis = 3 : i32} : tensor<1x10x1xi32> -> tensor<1x10x1x1xi32>
    %8 = arith.muli %7, %cst_6 : tensor<1x10x1x1xi32>
    %9 = tt.broadcast %4 : tensor<10x1x1x1xi32> -> tensor<10x10x1x1xi32>
    %10 = tt.broadcast %8 : tensor<1x10x1x1xi32> -> tensor<10x10x1x1xi32>
    %11 = arith.addi %9, %10 : tensor<10x10x1x1xi32>
    %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
    %15 = tt.expand_dims %14 {axis = 3 : i32} : tensor<1x1x16xi32> -> tensor<1x1x16x1xi32>
    %16 = arith.muli %15, %cst_5 : tensor<1x1x16x1xi32>
    %17 = tt.broadcast %11 : tensor<10x10x1x1xi32> -> tensor<10x10x16x1xi32>
    %18 = tt.broadcast %16 : tensor<1x1x16x1xi32> -> tensor<10x10x16x1xi32>
    %19 = arith.addi %17, %18 : tensor<10x10x16x1xi32>
    %20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %21 = tt.expand_dims %20 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<1x32xi32> -> tensor<1x1x32xi32>
    %23 = tt.expand_dims %22 {axis = 2 : i32} : tensor<1x1x32xi32> -> tensor<1x1x1x32xi32>
    %24 = tt.broadcast %19 : tensor<10x10x16x1xi32> -> tensor<10x10x16x32xi32>
    %25 = tt.broadcast %23 : tensor<1x1x1x32xi32> -> tensor<10x10x16x32xi32>
    %26 = arith.addi %24, %25 : tensor<10x10x16x32xi32>
    %27 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<10x10x16x32x!tt.ptr<i8>>
    %28 = tt.addptr %27, %26 : tensor<10x10x16x32x!tt.ptr<i8>>, tensor<10x10x16x32xi32>
    %29 = tt.load %28 : tensor<10x10x16x32x!tt.ptr<i8>>
    %30 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %32 = tt.expand_dims %31 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
    %33 = tt.expand_dims %32 {axis = 3 : i32} : tensor<2x1x1xi32> -> tensor<2x1x1x1xi32>
    %34 = arith.muli %33, %cst_4 : tensor<2x1x1x1xi32>
    %35 = arith.muli %7, %cst_3 : tensor<1x10x1x1xi32>
    %36 = tt.broadcast %34 : tensor<2x1x1x1xi32> -> tensor<2x10x1x1xi32>
    %37 = tt.broadcast %35 : tensor<1x10x1x1xi32> -> tensor<2x10x1x1xi32>
    %38 = arith.addi %36, %37 : tensor<2x10x1x1xi32>
    %39 = tt.expand_dims %22 {axis = 3 : i32} : tensor<1x1x32xi32> -> tensor<1x1x32x1xi32>
    %40 = arith.muli %39, %cst_2 : tensor<1x1x32x1xi32>
    %41 = tt.broadcast %38 : tensor<2x10x1x1xi32> -> tensor<2x10x32x1xi32>
    %42 = tt.broadcast %40 : tensor<1x1x32x1xi32> -> tensor<2x10x32x1xi32>
    %43 = arith.addi %41, %42 : tensor<2x10x32x1xi32>
    %44 = tt.broadcast %43 : tensor<2x10x32x1xi32> -> tensor<2x10x32x32xi32>
    %45 = tt.broadcast %23 : tensor<1x1x1x32xi32> -> tensor<2x10x32x32xi32>
    %46 = arith.addi %44, %45 : tensor<2x10x32x32xi32>
    %47 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<2x10x32x32x!tt.ptr<i8>>
    %48 = tt.addptr %47, %46 : tensor<2x10x32x32x!tt.ptr<i8>>, tensor<2x10x32x32xi32>
    %49 = tt.load %48 : tensor<2x10x32x32x!tt.ptr<i8>>
    %50 = ascend.dot %29, %49 {fractal_a = true, fractal_b = true, fractal_c = true} : tensor<10x10x16x32xi8>, tensor<2x10x32x32xi8> -> tensor<4x10x16x16xi32>
    %51 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %52 = tt.expand_dims %51 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %53 = tt.expand_dims %52 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %54 = tt.expand_dims %53 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %55 = arith.muli %54, %cst_1 : tensor<4x1x1x1xi32>
    %56 = arith.muli %7, %cst_0 : tensor<1x10x1x1xi32>
    %57 = tt.broadcast %55 : tensor<4x1x1x1xi32> -> tensor<4x10x1x1xi32>
    %58 = tt.broadcast %56 : tensor<1x10x1x1xi32> -> tensor<4x10x1x1xi32>
    %59 = arith.addi %57, %58 : tensor<4x10x1x1xi32>
    %60 = arith.muli %15, %cst : tensor<1x1x16x1xi32>
    %61 = tt.broadcast %59 : tensor<4x10x1x1xi32> -> tensor<4x10x16x1xi32>
    %62 = tt.broadcast %60 : tensor<1x1x16x1xi32> -> tensor<4x10x16x1xi32>
    %63 = arith.addi %61, %62 : tensor<4x10x16x1xi32>
    %64 = tt.expand_dims %14 {axis = 2 : i32} : tensor<1x1x16xi32> -> tensor<1x1x1x16xi32>
    %65 = tt.broadcast %63 : tensor<4x10x16x1xi32> -> tensor<4x10x16x16xi32>
    %66 = tt.broadcast %64 : tensor<1x1x1x16xi32> -> tensor<4x10x16x16xi32>
    %67 = arith.addi %65, %66 : tensor<4x10x16x16xi32>
    %68 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x10x16x16x!tt.ptr<i32>>
    %69 = tt.addptr %68, %67 : tensor<4x10x16x16x!tt.ptr<i32>>, tensor<4x10x16x16xi32>
    tt.store %69, %50 : tensor<4x10x16x16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @s_C_int8
// CHECK-SAME:    mix_mode = "mix"
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [160, 320] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 32]>} : (tensor<10x10x16x32xi8>) -> tensor<160x320xi8>
// CHECK:         hivm.hir.convert_layout %{{.*}} output_shape [320, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<Fractal, fractalSizes = [32, 32]>} : (tensor<2x10x32x32xi8>) -> tensor<320x64xi8>
// i32 accumulator for an int8 input.
// CHECK:         tensor.empty() : tensor<160x64xi32>
// CHECK:         %[[MM:.*]] = linalg.matmul {input_precision = "ieee"} ins(%{{.*}}, %{{.*}} : tensor<160x320xi8>, tensor<320x64xi8>) outs(%{{.*}} : tensor<160x64xi32>) -> tensor<160x64xi32>
// fractal_c block stays [16, 16] even though the int8 input block is [16, 32].
// CHECK:         hivm.hir.convert_layout %[[MM]] output_shape [4, 10, 16, 16] {dstLayout = #hivm.data_layout<Fractal, fractalSizes = [16, 16]>, srcLayout = #hivm.data_layout<ND>} : (tensor<160x64xi32>) -> tensor<4x10x16x16xi32>
