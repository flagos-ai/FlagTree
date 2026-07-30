// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s

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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @atan2_kernel_0123
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_7_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_3_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_4_:%.+]] = memref.subview [[RES_1_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_3_]], [[VAR_subview_4_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_8_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_8_]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = math.atan2 [[IN_0_]], [[IN_1_]] : f32
// CHECK:             linalg.yield [[VAR_10_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_5_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @pow_kernel_0123
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_7_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_3_:%.+]] = memref.subview [[VAR_reinterpret_cast_1_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_4_:%.+]] = memref.subview [[RES_1_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_3_]], [[VAR_subview_4_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_8_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_8_]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = math.powf [[IN_0_]], [[IN_1_]] : f32
// CHECK:             linalg.yield [[VAR_10_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_5_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @fabs_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.absf [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @sin_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.sin [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @cos_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.cos [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @tan_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.tan [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @asin_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.asin [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @acos_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.acos [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @atan_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.atan [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @sinh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.sinh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @cosh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.cosh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @tanh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.tanh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @asinh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.asinh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @acosh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.acosh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @atanh_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.atanh [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @log_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.log [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @log10_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.log10 [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @log1p_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.log1p [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @exp_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @exp2_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.exp2 [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @erf_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.erf [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @sqrt_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.sqrt [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @rsqrt_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.rsqrt [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @ceil_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.ceil [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @floor_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.floor [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @trunc_kernel_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[CST_32_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_32_1_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_subview_:%.+]] = memref.subview [[VAR_reinterpret_cast_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_subview_0_:%.+]] = memref.subview [[RES_]][0] {{.}}[[VAR_6_]]{{.}} [1] : memref<32xf32> to memref<?xf32, strided<[1]>>
// CHECK:           memref.copy [[VAR_subview_]], [[VAR_subview_0_]] : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
// CHECK:           [[VAR_7_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]] : tensor<32xf32>) outs([[VAR_7_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = math.trunc [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_8_]] in writable [[VAR_reinterpret_cast_1_]] : (tensor<32xf32>, memref<32xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
