// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @atan2_kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %12 = tt.load %11, %6 : tensor<32x!tt.ptr<f32>>
    %13 = tt.extern_elementwise %9, %12 {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %15, %13 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @atan2_kernel_0123
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]], [[VAR_2:%.+]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_3:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_4:%.+]] = math.atan2 [[in_1]], [[in_2]] : f32
// CHECK:          linalg.yield [[VAR_4]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @pow_kernel_0123(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %12 = tt.load %11, %6 : tensor<32x!tt.ptr<f32>>
    %13 = tt.extern_elementwise %9, %12 {libname = "", libpath = "", pure = true, symbol = "__nv_powf"} : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %15, %13 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @pow_kernel_0123
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]], [[VAR_2:%.+]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_3:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in_1:%.+]]: f32, [[in_2:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_4:%.+]] = math.powf [[in_1]], [[in_2]] : f32
// CHECK:          linalg.yield [[VAR_4]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @fabs_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_fabsf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @fabs_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.absf [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @sin_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_sinf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @sin_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.sin [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @cos_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_cosf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @cos_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.cos [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @tan_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_tanf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @tan_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.tan [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @asin_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_asinf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @asin_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.asin [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @acos_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_acosf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @acos_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.acos [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @atan_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_atanf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @atan_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.atan [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @sinh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_sinhf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @sinh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.sinh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @cosh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_coshf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @cosh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.cosh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @tanh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_tanhf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @tanh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.tanh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @asinh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_asinhf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @asinh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.asinh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @acosh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_acoshf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @acosh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.acosh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @atanh_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_atanhf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @atanh_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.atanh [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @log_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_logf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @log_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.log [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @log10_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_log10f"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @log10_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.log10 [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @log1p_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_log1pf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @log1p_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.log1p [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @exp_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_expf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @exp_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.exp [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @exp2_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_exp2f"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @exp2_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.exp2 [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @erf_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_erff"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @erf_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.erf [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @sqrt_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_sqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @sqrt_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.sqrt [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @rsqrt_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_rsqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @rsqrt_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.rsqrt [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @ceil_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_ceilf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @ceil_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.ceil [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @floor_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_floorf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @floor_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.floor [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>

// -----

module {
  tt.func public @trunc_kernel_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<32xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<32xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %9 = tt.load %8, %6 : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__nv_truncf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @trunc_kernel_012
// CHECK:        [[RES:%.+]] = linalg.generic {indexing_maps = [[[MAP]], [[MAP]]], iterator_types = ["parallel"]} ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]] : tensor<32xf32>) {
// CHECK:        ^bb0([[in:%.+]]: f32, [[out:%.+]]: f32):
// CHECK:          [[VAR_3:%.+]] = math.trunc [[in:%.+]] : f32
// CHECK:          linalg.yield [[VAR_3:%.+]] : f32
// CHECK:        } -> tensor<32xf32>
