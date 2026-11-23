// RUN: triton-opt --discrete-mask-access-conversion --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func @atomic_add_i32
// CHECK: %[[default:.*]] = arith.constant dense<0> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw add, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_add_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw add, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_fadd_f32
// CHECK: %[[default:.*]] = arith.constant dense<0.000000e+00> : tensor<1024xf32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw fadd, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_fadd_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
  %9 = tt.atomic_rmw fadd, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, tensor<1024xi1>) -> tensor<1024xf32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_max_i32
// CHECK: %[[default:.*]] = arith.constant dense<-2147483648> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_max_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw max, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_umax_i32
// CHECK: %[[default:.*]] = arith.constant dense<0> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw umax, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_umax_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw umax, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_min_i32
// CHECK: %[[default:.*]] = arith.constant dense<2147483647> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw min, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_min_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw min, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_umin_i32
// CHECK: %[[default:.*]] = arith.constant dense<2147483647> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw umin, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_umin_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw umin, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_and_i32
// CHECK: %[[default:.*]] = arith.constant dense<2147483647> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw and, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_and_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw and, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_or_i32
// CHECK: %[[default:.*]] = arith.constant dense<0> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw or, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_or_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i32>>
  %9 = tt.atomic_rmw or, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>, tensor<1024xi1>) -> tensor<1024xi32>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_max_i16
// CHECK: %[[default:.*]] = arith.constant dense<-32768> : tensor<1024xi16>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_max_i16(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i16>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<1024x!tt.ptr<i16>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i16>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<i16>>
  %9 = tt.atomic_rmw max, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<i16>>, tensor<1024xi16>, tensor<1024xi1>) -> tensor<1024xi16>
  tt.return
}

// CHECK-LABEL: tt.func @atomic_max_f16
// CHECK: %[[default:.*]] = arith.constant dense<0xFC00> : tensor<1024xf16>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]]
tt.func @atomic_max_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
  %cst = arith.constant dense<200> : tensor<1024xi32>
  %cst_0 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_0 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
  %8 = tt.load %7 : tensor<1024x!tt.ptr<f16>>
  %9 = tt.atomic_rmw max, acq_rel, gpu, %5, %8, %3 : (tensor<1024x!tt.ptr<f16>>, tensor<1024xf16>, tensor<1024xi1>) -> tensor<1024xf16>
  tt.return
}

// -----

// CHECK-LABEL: tt.func @atomic_add_cont_disc_i32
// CHECK: %[[DEFAULT:.*]] = arith.constant dense<0> : tensor<8x8xi32>
// CHECK: %[[CONT_MASK:.*]] = tt.broadcast {{.*}} : tensor<8x1xi1> -> tensor<8x8xi1>
// CHECK: %[[DISC_MASK:.*]] = tt.broadcast {{.*}} : tensor<1x8xi1> -> tensor<8x8xi1>
// CHECK: %[[VALUE:.*]] = arith.select %[[DISC_MASK]], %{{.*}}, %[[DEFAULT]] : tensor<8x8xi1>, tensor<8x8xi32>
// CHECK: tt.atomic_rmw add, acq_rel, gpu, %{{.*}}, %[[VALUE]], %[[CONT_MASK]] : (tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32>, tensor<8x8xi1>) -> tensor<8x8xi32>
tt.func @atomic_add_cont_disc_i32(%arg0: !tt.ptr<i32>, %arg1: i32) {
  %cst_0 = arith.constant dense<8> : tensor<8x1xi32>
  %cst_1 = arith.constant dense<2> : tensor<1x8xi32>
  %cst_2 = arith.constant dense<8> : tensor<1x8xi32>
  %cst_3 = arith.constant dense<1> : tensor<8x8xi32>
  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %3 = tt.splat %arg1 : i32 -> tensor<8x1xi32>
  %4 = arith.cmpi slt, %1, %3 : tensor<8x1xi32>
  %5 = tt.broadcast %4 : tensor<8x1xi1> -> tensor<8x8xi1>
  %6 = arith.muli %2, %cst_1 : tensor<1x8xi32>
  %7 = arith.cmpi slt, %6, %cst_2 : tensor<1x8xi32>
  %8 = tt.broadcast %7 : tensor<1x8xi1> -> tensor<8x8xi1>
  %9 = arith.andi %5, %8 : tensor<8x8xi1>
  %10 = arith.muli %1, %cst_0 : tensor<8x1xi32>
  %11 = tt.broadcast %10 : tensor<8x1xi32> -> tensor<8x8xi32>
  %12 = tt.broadcast %2 : tensor<1x8xi32> -> tensor<8x8xi32>
  %13 = arith.addi %11, %12 : tensor<8x8xi32>
  %14 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x8x!tt.ptr<i32>>
  %15 = tt.addptr %14, %13 : tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32>
  %16 = tt.atomic_rmw add, acq_rel, gpu, %15, %cst_3, %9 : (tensor<8x8x!tt.ptr<i32>>, tensor<8x8xi32>, tensor<8x8xi1>) -> tensor<8x8xi32>
  tt.return
}

// -----

// CHECK-LABEL: tt.func @atomic_min_f32_splat_bitcast
// CHECK-NOT: tt.addptr %arg0
// CHECK: %[[VALUE_PTRS:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x1x1x1x!tt.ptr<f32>>
// CHECK: %[[VALUE:.*]] = tt.load %[[VALUE_PTRS]] : tensor<1x1x1x1x!tt.ptr<f32>>
// CHECK-NOT: tt.addptr %arg1
// CHECK: %[[OUTPUT_PTRS:.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x1x1x1x!tt.ptr<f32>>
// CHECK: tt.atomic_rmw min, acq_rel, gpu, %[[OUTPUT_PTRS]], %[[VALUE]], %{{.*}} : (tensor<1x1x1x1x!tt.ptr<f32>>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xf32>
// CHECK-NOT: tt.atomic_rmw umax
tt.func @atomic_min_f32_splat_bitcast(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0> : tensor<1x1x1x1xi32>
  %cst_0 = arith.constant dense<31> : tensor<1x1x1x1xi32>
  %cst_1 = arith.constant dense<true> : tensor<1x1x1x1xi1>
  %0 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f32>, i32
  %1 = tt.splat %0 : !tt.ptr<f32> -> tensor<1x1x1x1x!tt.ptr<f32>>
  %2 = tt.load %1 : tensor<1x1x1x1x!tt.ptr<f32>>
  %3 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f32>, i32
  %4 = tt.bitcast %2 : tensor<1x1x1x1xf32> -> tensor<1x1x1x1xi32>
  %5 = tt.bitcast %3 : !tt.ptr<f32> -> !tt.ptr<i32>
  %6 = tt.splat %5 : !tt.ptr<i32> -> tensor<1x1x1x1x!tt.ptr<i32>>
  %7 = arith.shrui %4, %cst_0 : tensor<1x1x1x1xi32>
  %8 = arith.cmpi ne, %7, %cst : tensor<1x1x1x1xi32>
  %9 = arith.xori %8, %cst_1 : tensor<1x1x1x1xi1>
  %10 = tt.atomic_rmw min, acq_rel, gpu, %6, %4, %9 : (tensor<1x1x1x1x!tt.ptr<i32>>, tensor<1x1x1x1xi32>, tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi32>
  %11 = tt.atomic_rmw umax, acq_rel, gpu, %6, %4, %8 : (tensor<1x1x1x1x!tt.ptr<i32>>, tensor<1x1x1x1xi32>, tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi32>
  tt.return
}

// -----

// CHECK-LABEL: tt.func @atomic_max_f32_scalar
// CHECK: tt.atomic_rmw max, acq_rel, gpu, %arg1, %arg0, %{{.*}} : (!tt.ptr<f32>, f32, i1) -> f32
// CHECK-NOT: tt.atomic_rmw umin
tt.func @atomic_max_f32_scalar(%arg0: f32, %arg1: !tt.ptr<f32>) {
  %c31_i32 = arith.constant 31 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.bitcast %arg0 : f32 -> i32
  %1 = tt.bitcast %arg1 : !tt.ptr<f32> -> !tt.ptr<i32>
  %2 = arith.shrui %0, %c31_i32 : i32
  %3 = arith.cmpi eq, %2, %c0_i32 : i32
  %4 = arith.cmpi ne, %2, %c0_i32 : i32
  %5 = tt.atomic_rmw max, acq_rel, gpu, %1, %0, %3 : (!tt.ptr<i32>, i32, i1) -> i32
  %6 = tt.atomic_rmw umin, acq_rel, gpu, %1, %0, %4 : (!tt.ptr<i32>, i32, i1) -> i32
  tt.return
}
