// RUN: triton-shared-opt --triton-to-linalg-experimental --split-input-file %s | FileCheck %s

// Make sure arith.minsi was created for mask.
// CHECK-LABLE: tt.func public @index_select_row_index_mask_kernel
// CHECK: %[[LOOP_COUNT:.*]] = arith.constant 32 : index
// CHECK: %{{.*}} = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK: %[[MIN:.*]] = arith.minsi %{{.*}}, %[[LOOP_COUNT]] : index
// CHECK:    scf.for %[[IV:.*]] = %{{.*}} to %[[MIN]] step %{{.*}} {

  tt.func public @index_select_row_index_mask_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %2 = arith.muli %0, %1 : tensor<32xi32>
    %3 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %5 = tt.load %4 : tensor<32x!tt.ptr<i32>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = tt.splat %arg4 : i32 -> tensor<32x1xi32>
    %8 = arith.muli %6, %7 : tensor<32x1xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %11 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %12 = tt.splat %arg5 : i32 -> tensor<1x32xi32>
    %13 = arith.muli %11, %12 : tensor<1x32xi32>
    %14 = tt.broadcast %10 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %15 = tt.broadcast %13 : tensor<1x32xi32> -> tensor<32x32xi32>
    %16 = tt.addptr %14, %15 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %17 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %18 = tt.splat %arg8 : i32 -> tensor<32x1xi32>
    %19 = arith.cmpi slt, %17, %18 : tensor<32x1xi32>
    %20 = tt.broadcast %19 : tensor<32x1xi1> -> tensor<32x32xi1>
    %21 = tt.load %16, %20, %cst : tensor<32x32x!tt.ptr<f32>>
    %22 = tt.splat %arg6 : i32 -> tensor<32x1xi32>
    %23 = arith.muli %17, %22 : tensor<32x1xi32>
    %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %26 = tt.splat %arg7 : i32 -> tensor<1x32xi32>
    %27 = arith.muli %11, %26 : tensor<1x32xi32>
    %28 = tt.broadcast %25 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %29 = tt.broadcast %27 : tensor<1x32xi32> -> tensor<32x32xi32>
    %30 = tt.addptr %28, %29 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %30, %21 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }

// Make sure loop count was changed for imm mask.
// CHECK-LABLE: tt.func public @index_select_row_imm_index_mask_kernel
// CHECK: %[[LOOP_COUNT:.*]] = arith.constant 16 : index
// CHECK:    scf.for %[[IV:.*]] = %{{.*}} to %[[LOOP_COUNT]] step %{{.*}} {
// CHECK-NOT:      arith.cmpi slt
// CHECK-NOT:      scf.if

  tt.func public @index_select_row_imm_index_mask_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %limit = arith.constant 16 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %2 = arith.muli %0, %1 : tensor<32xi32>
    %3 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %5 = tt.load %4 : tensor<32x!tt.ptr<i32>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = tt.splat %arg4 : i32 -> tensor<32x1xi32>
    %8 = arith.muli %6, %7 : tensor<32x1xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %11 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %12 = tt.splat %arg5 : i32 -> tensor<1x32xi32>
    %13 = arith.muli %11, %12 : tensor<1x32xi32>
    %14 = tt.broadcast %10 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %15 = tt.broadcast %13 : tensor<1x32xi32> -> tensor<32x32xi32>
    %16 = tt.addptr %14, %15 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %17 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %18 = tt.splat %limit : i32 -> tensor<32x1xi32>
    %19 = arith.cmpi slt, %17, %18 : tensor<32x1xi32>
    %20 = tt.broadcast %19 : tensor<32x1xi1> -> tensor<32x32xi1>
    %21 = tt.load %16, %20, %cst : tensor<32x32x!tt.ptr<f32>>
    %22 = tt.splat %arg6 : i32 -> tensor<32x1xi32>
    %23 = arith.muli %17, %22 : tensor<32x1xi32>
    %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %26 = tt.splat %arg7 : i32 -> tensor<1x32xi32>
    %27 = arith.muli %11, %26 : tensor<1x32xi32>
    %28 = tt.broadcast %25 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %29 = tt.broadcast %27 : tensor<1x32xi32> -> tensor<32x32xi32>
    %30 = tt.addptr %28, %29 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %30, %21 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }

// Make sure loop count was not changed for out of bound imm mask.
// CHECK-LABLE: tt.func public @index_select_row_out_index_mask_kernel
// CHECK: %[[LOOP_COUNT:.*]] = arith.constant 32 : index
// CHECK:    scf.for %[[IV:.*]] = %{{.*}} to %[[LOOP_COUNT]] step %{{.*}} {
// CHECK-NOT:      arith.cmpi slt
// CHECK-NOT:      scf.if

  tt.func public @index_select_row_out_index_mask_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
    %limit = arith.constant 64 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg3 : i32 -> tensor<32xi32>
    %2 = arith.muli %0, %1 : tensor<32xi32>
    %3 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %5 = tt.load %4 : tensor<32x!tt.ptr<i32>>
    %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %7 = tt.splat %arg4 : i32 -> tensor<32x1xi32>
    %8 = arith.muli %6, %7 : tensor<32x1xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %11 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %12 = tt.splat %arg5 : i32 -> tensor<1x32xi32>
    %13 = arith.muli %11, %12 : tensor<1x32xi32>
    %14 = tt.broadcast %10 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %15 = tt.broadcast %13 : tensor<1x32xi32> -> tensor<32x32xi32>
    %16 = tt.addptr %14, %15 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %17 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %18 = tt.splat %limit : i32 -> tensor<32x1xi32>
    %19 = arith.cmpi slt, %17, %18 : tensor<32x1xi32>
    %20 = tt.broadcast %19 : tensor<32x1xi1> -> tensor<32x32xi1>
    %21 = tt.load %16, %20, %cst : tensor<32x32x!tt.ptr<f32>>
    %22 = tt.splat %arg6 : i32 -> tensor<32x1xi32>
    %23 = arith.muli %17, %22 : tensor<32x1xi32>
    %24 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %26 = tt.splat %arg7 : i32 -> tensor<1x32xi32>
    %27 = arith.muli %11, %26 : tensor<1x32xi32>
    %28 = tt.broadcast %25 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %29 = tt.broadcast %27 : tensor<1x32xi32> -> tensor<32x32xi32>
    %30 = tt.addptr %28, %29 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %30, %21 : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }