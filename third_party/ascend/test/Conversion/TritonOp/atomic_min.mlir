// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @atomic_min_uint8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i8>>
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %12 = tt.atomic_rmw umin, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i8>>, tensor<4xi8>, tensor<4xi1>) -> tensor<4xi8>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_uint8
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i8, %[[IN_VAL:.+]]: i8, %[[OUT_VAL:.+]]: i8):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minui %[[MEM_VAL]], %[[IN_VAL]] : i8
// CHECK-NEXT: linalg.yield %[[RESULT]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_uint16(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i16>>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %12 = tt.atomic_rmw umin, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i16>>, tensor<4xi16>, tensor<4xi1>) -> tensor<4xi16>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_uint16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i16, %[[IN_VAL:.+]]: i16, %[[OUT_VAL:.+]]: i16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minui %[[MEM_VAL]], %[[IN_VAL]] : i16
// CHECK-NEXT: linalg.yield %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_uint32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %12 = tt.atomic_rmw umin, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_uint32
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[IN_VAL:.+]]: i32, %[[OUT_VAL:.+]]: i32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minui %[[MEM_VAL]], %[[IN_VAL]] : i32
// CHECK-NEXT: linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_uint64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i64>>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %12 = tt.atomic_rmw umin, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>, tensor<4xi1>) -> tensor<4xi64>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_uint64
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i64, %[[IN_VAL:.+]]: i64, %[[OUT_VAL:.+]]: i64):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minui %[[MEM_VAL]], %[[IN_VAL]] : i64
// CHECK-NEXT: linalg.yield %[[RESULT]] : i64
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_int8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i8>>
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x!tt.ptr<i8>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i8>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i8>>, tensor<4xi8>, tensor<4xi1>) -> tensor<4xi8>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_int8
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i8, %[[IN_VAL:.+]]: i8, %[[OUT_VAL:.+]]: i8):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minsi %[[MEM_VAL]], %[[IN_VAL]] : i8
// CHECK-NEXT: linalg.yield %[[RESULT]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_int16(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i16>>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<4x!tt.ptr<i16>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i16>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i16>>, tensor<4xi16>, tensor<4xi1>) -> tensor<4xi16>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_int16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i16, %[[IN_VAL:.+]]: i16, %[[OUT_VAL:.+]]: i16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minsi %[[MEM_VAL]], %[[IN_VAL]] : i16
// CHECK-NEXT: linalg.yield %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_int32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_int32
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[IN_VAL:.+]]: i32, %[[OUT_VAL:.+]]: i32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minsi %[[MEM_VAL]], %[[IN_VAL]] : i32
// CHECK-NEXT: linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_int64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<i64>>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<i64>>, tensor<4xi64>, tensor<4xi1>) -> tensor<4xi64>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_int64
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i64, %[[IN_VAL:.+]]: i64, %[[OUT_VAL:.+]]: i64):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minsi %[[MEM_VAL]], %[[IN_VAL]] : i64
// CHECK-NEXT: linalg.yield %[[RESULT]] : i64
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_bool(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<4xi8>
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %9 = tt.bitcast %8 : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %10 = tt.load %9 : tensor<4x!tt.ptr<i8>>
    %11 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %12 = tt.addptr %11, %2 : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %13 = arith.cmpi ne, %10, %cst : tensor<4xi8>
    %14 = tt.atomic_rmw umin, acq_rel, gpu, %12, %13, %6 : (tensor<4x!tt.ptr<i1>>, tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_bool
// Verify the constant tensor is created via linalg.fill
// CHECK: %[[ZERO_TENSOR:.+]] = linalg.fill ins(%c0_i8 : i8) outs(%{{.+}} : tensor<4xi8>) -> tensor<4xi8>
// Verify the comparison is done on i8 tensors
// CHECK: %[[CMP_RESULT:.+]] = arith.cmpi ne, %{{.+}}, %[[ZERO_TENSOR]] : tensor<4xi8>
// Verify the atomic operation is converted to linalg.generic on i1
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "umin", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i1, %[[IN_VAL:.+]]: i1, %[[OUT_VAL:.+]]: i1):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minui %[[MEM_VAL]], %[[IN_VAL]] : i1
// CHECK-NEXT: linalg.yield %[[RESULT]] : i1
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<f16>>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<4x!tt.ptr<f16>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<f16>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<f16>>, tensor<4xf16>, tensor<4xi1>) -> tensor<4xf16>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_f16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f16, %[[IN_VAL:.+]]: f16, %[[OUT_VAL:.+]]: f16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minnumf %[[MEM_VAL]], %[[IN_VAL]] : f16
// CHECK-NEXT: linalg.yield %[[RESULT]] : f16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xi1>) -> tensor<4xf32>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_f32
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f32, %[[IN_VAL:.+]]: f32, %[[OUT_VAL:.+]]: f32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minnumf %[[MEM_VAL]], %[[IN_VAL]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<bf16>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<bf16>>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<bf16>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<bf16>>, tensor<4xbf16>, tensor<4xi1>) -> tensor<4xbf16>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_bf16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: bf16, %[[IN_VAL:.+]]: bf16, %[[OUT_VAL:.+]]: bf16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minnumf %[[MEM_VAL]], %[[IN_VAL]] : bf16
// CHECK-NEXT: linalg.yield %[[RESULT]] : bf16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_f8e5m2(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<f8E5M2>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<4x!tt.ptr<f8E5M2>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<f8E5M2>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<f8E5M2>>, tensor<4xf8E5M2>, tensor<4xi1>) -> tensor<4xf8E5M2>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_f8e5m2
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E5M2, %[[IN_VAL:.+]]: f8E5M2, %[[OUT_VAL:.+]]: f8E5M2):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minnumf %[[MEM_VAL]], %[[IN_VAL]] : f8E5M2
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E5M2
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_min_f8E4M3FN(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.splat %1 : i32 -> tensor<4xi32>
    %4 = arith.addi %3, %2 : tensor<4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<4xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<4xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %4 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %9 = tt.load %8 : tensor<4x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<4x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %2 : tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xi32>
    %12 = tt.atomic_rmw min, acq_rel, gpu, %11, %9, %6 : (tensor<4x!tt.ptr<f8E4M3FN>>, tensor<4xf8E4M3FN>, tensor<4xi1>) -> tensor<4xf8E4M3FN>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_min_f8E4M3FN
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "min", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E4M3FN, %[[IN_VAL:.+]]: f8E4M3FN, %[[OUT_VAL:.+]]: f8E4M3FN):
// CHECK-NEXT: %[[RESULT:.+]] = arith.minnumf %[[MEM_VAL]], %[[IN_VAL]] : f8E4M3FN
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E4M3FN
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw
