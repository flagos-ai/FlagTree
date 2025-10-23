// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

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
    %9 = tt.load %8, %6 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__hmf_fabsf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   func.func private @__hmf_fabsf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @fabs_kernel_012
// CHECK:        [[RES:%.+]] = linalg.map { func.call {callee = @__hmf_fabsf} } ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]]: tensor<32xf32>)

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
    %9 = tt.load %8, %6 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__hmf_sqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   func.func private @__hmf_sqrtf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @sqrt_kernel_012
// CHECK:        [[RES:%.+]] = linalg.map { func.call {callee = @__hmf_sqrtf} } ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]]: tensor<32xf32>)

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
    %9 = tt.load %8, %6 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    %10 = tt.extern_elementwise %9 {libname = "", libpath = "", pure = true, symbol = "__hmf_rsqrtf"} : (tensor<32xf32>) -> tensor<32xf32>
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %12, %10 evictionPolicy = evict_last : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   func.func private @__hmf_rsqrtf(f32) -> f32 attributes {llvm.readnone}
// CHECK-LABEL: func.func @rsqrt_kernel_012
// CHECK:        [[RES:%.+]] = linalg.map { func.call {callee = @__hmf_rsqrtf} } ins([[VAR_1:%.+]] : tensor<32xf32>) outs([[VAR_2:%.+]]: tensor<32xf32>)
