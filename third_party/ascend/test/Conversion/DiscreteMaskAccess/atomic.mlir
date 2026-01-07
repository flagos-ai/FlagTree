// RUN: triton-adapter-opt %s --discrete-mask-access-conversion --split-input-file %s | FileCheck %s
// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False'

// CHECK-LABEL: tt.func @atomic_add_i32
// CHECK: %[[default:.*]] = arith.constant dense<0> : tensor<1024xi32>
// CHECK: %[[value:.*]] = arith.select %[[mask:.*]], %[[origin:.*]], %[[default]]
// CHECK: %[[result:.*]] = tt.atomic_rmw add, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw fadd, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw umax, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw min, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw umin, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw and, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw or, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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
// CHECK: %[[result:.*]] = tt.atomic_rmw max, acq_rel, gpu, %[[ptr:.*]], %[[value]] {DiscreteMask}
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