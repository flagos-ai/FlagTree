// RUN: triton-opt %s --triton-to-unstructure | FileCheck %s
// RUN: triton-opt %s '--triton-to-unstructure=force-scalarize-mode=True' | FileCheck %s
// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True' | FileCheck %s
// RUN: triton-opt %s '--triton-to-unstructure=force-simt-template=True' | FileCheck %s


// CHECK-LABEL: tt.func @triton_indirect_load
tt.func @triton_indirect_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>) -> tensor<4096xf32> {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    %6 = tt.load %5 : tensor<4096x!tt.ptr<f32>>
    annotation.mark %6 {mayDiscretememaccess} : tensor<4096xf32>
    tt.return %6 : tensor<4096xf32>
}


// CHECK-LABEL: tt.func @triton_indirect_load_error_mark
tt.func @triton_indirect_load_error_mark(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>) -> tensor<4096xf32> {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    %6 = tt.load %5 : tensor<4096x!tt.ptr<f32>>
    annotation.mark %6 {error_mark} : tensor<4096xf32>
    tt.return %6 : tensor<4096xf32>
}


// CHECK-LABEL: tt.func @triton_indirect_store
tt.func @triton_indirect_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>) {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %6 = tt.load %5 : tensor<4096x!tt.ptr<f32>>
    annotation.mark %6 {mayDiscretememaccess} : tensor<4096xf32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    tt.store %8, %6 : tensor<4096x!tt.ptr<f32>>
    tt.return
}


// CHECK-LABEL: tt.func @triton_indirect_store_error_mark
tt.func @triton_indirect_store_error_mark(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>) {
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %6 = tt.load %5 : tensor<4096x!tt.ptr<f32>>
    annotation.mark %6 {error_mark} : tensor<4096xf32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    tt.store %8, %6 : tensor<4096x!tt.ptr<f32>>
    tt.return
}


// CHECK-LABEL: tt.func @triton_indirect_atomic_add
tt.func @triton_indirect_atomic_add(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}
) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<4096xf32>
    %cst_0 = arith.constant dense<true> : tensor<4096xi1>
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %cst, %cst_0 : (tensor<4096x!tt.ptr<f32>>, tensor<4096xf32>, tensor<4096xi1>) -> tensor<4096xf32>
    tt.return
}


// CHECK-LABEL: tt.func @triton_indirect_atomic_add_with_discret_mask
tt.func @triton_indirect_atomic_add_with_discret_mask(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}
) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<4096xf32>
    %cst_0 = arith.constant dense<4096> : tensor<4096xi64>
    %c1 = arith.constant 1.0 : f32
    %c0 = arith.constant 0.0 : f32
    %cst_true = tt.splat %c1 : f32 -> tensor<4096xf32>
    %cst_false = tt.splat %c0 : f32 -> tensor<4096xf32>
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %mask = arith.cmpi slt, %3, %cst_0 : tensor<4096xi64>
    %selected_val = arith.select %mask, %cst_true, %cst_false : tensor<4096xi1>, tensor<4096xf32>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    %6 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %selected_val, %mask {DiscreteMask} : (tensor<4096x!tt.ptr<f32>>, tensor<4096xf32>, tensor<4096xi1>) -> tensor<4096xf32>
    tt.return
}


// CHECK-LABEL: tt.func @triton_indirect_atomic_cas
tt.func @triton_indirect_atomic_cas(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}
) attributes {noinline = false} {
    %cst = arith.constant dense<2.0> : tensor<4096xf32>
    %cst_0 = arith.constant dense<1.0> : tensor<4096xf32>
    %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
    %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
    %6 = tt.atomic_cas acq_rel, gpu, %5, %cst_0, %cst : (tensor<4096x!tt.ptr<f32>>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>
    tt.return
}
