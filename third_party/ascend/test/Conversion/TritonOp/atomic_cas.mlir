// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<768xi8>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<i8>>
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<i8>>
    %13 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<i8>>, tensor<768xi8>, tensor<768xi8>) -> tensor<768xi8>
    %16 = tt.splat %arg3 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i8, %[[CMP_VAL:.+]]: i8, %[[NEW_VAL:.+]]: i8, %[[OUT:.+]]: i8):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpi eq, %[[MEM_VAL]], %[[CMP_VAL]] : i8
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : i8
// CHECK-NEXT: linalg.yield %[[RESULT]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<768xi16>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<i16>>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<i16>>
    %13 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<i16>>, tensor<768xi16>, tensor<768xi16>) -> tensor<768xi16>
    %16 = tt.splat %arg3 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i16, %[[CMP_VAL:.+]]: i16, %[[NEW_VAL:.+]]: i16, %[[OUT:.+]]: i16):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpi eq, %[[MEM_VAL]], %[[CMP_VAL]] : i16
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : i16
// CHECK-NEXT: linalg.yield %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas


// -----
module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<768xi32>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<i32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<i32>>
    %13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<i32>>, tensor<768xi32>, tensor<768xi32>) -> tensor<768xi32>
    %16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[CMP_VAL:.+]]: i32, %[[NEW_VAL:.+]]: i32, %[[OUT:.+]]: i32):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpi eq, %[[MEM_VAL]], %[[CMP_VAL]] : i32
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : i32
// CHECK-NEXT: linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<768xi64>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<i64>>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<i64>>
    %13 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<i64>>, tensor<768xi64>, tensor<768xi64>) -> tensor<768xi64>
    %16 = tt.splat %arg3 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i64, %[[CMP_VAL:.+]]: i64, %[[NEW_VAL:.+]]: i64, %[[OUT:.+]]: i64):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpi eq, %[[MEM_VAL]], %[[CMP_VAL]] : i64
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : i64
// CHECK-NEXT: linalg.yield %[[RESULT]] : i64
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<768xf16>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<f16>>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<f16>>
    %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<f16>>, tensor<768xf16>, tensor<768xf16>) -> tensor<768xf16>
    %16 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f16, %[[CMP_VAL:.+]]: f16, %[[NEW_VAL:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpf ueq, %[[MEM_VAL]], %[[CMP_VAL]] : f16
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : f16
// CHECK-NEXT: linalg.yield %[[RESULT]] : f16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<768xf32>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<f32>>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<f32>>, tensor<768xf32>, tensor<768xf32>) -> tensor<768xf32>
    %16 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f32, %[[CMP_VAL:.+]]: f32, %[[NEW_VAL:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpf ueq, %[[MEM_VAL]], %[[CMP_VAL]] : f32
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<768xbf16>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<bf16>>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<bf16>>
    %13 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<bf16>>, tensor<768xbf16>, tensor<768xbf16>) -> tensor<768xbf16>
    %16 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: bf16, %[[CMP_VAL:.+]]: bf16, %[[NEW_VAL:.+]]: bf16, %[[OUT:.+]]: bf16):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpf ueq, %[[MEM_VAL]], %[[CMP_VAL]] : bf16
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : bf16
// CHECK-NEXT: linalg.yield %[[RESULT]] : bf16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<768xf8E5M2>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<f8E5M2>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<f8E5M2>>
    %13 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<f8E5M2>>, tensor<768xf8E5M2>, tensor<768xf8E5M2>) -> tensor<768xf8E5M2>
    %16 = tt.splat %arg3 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E5M2, %[[CMP_VAL:.+]]: f8E5M2, %[[NEW_VAL:.+]]: f8E5M2, %[[OUT:.+]]: f8E5M2):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpf ueq, %[[MEM_VAL]], %[[CMP_VAL]] : f8E5M2
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : f8E5M2
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E5M2
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas

// -----

module {
  tt.func public @atomic_cas(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<768xf8E4M3FN>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg4 : i32 -> tensor<768xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<768xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %4 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    %9 = tt.load %8, %6, %cst : tensor<768x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %4 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    %12 = tt.load %11, %6, %cst : tensor<768x!tt.ptr<f8E4M3FN>>
    %13 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    %15 = tt.atomic_cas acq_rel, gpu, %14, %12, %9 : (tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xf8E4M3FN>, tensor<768xf8E4M3FN>) -> tensor<768xf8E4M3FN>
    %16 = tt.splat %arg3 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %17 = tt.addptr %16, %2 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    tt.store %17, %15, %6 : tensor<768x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_cas
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "cas", MemSemantic = "acq_rel", MemSyncScope = "gpu"{{.*}} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E4M3FN, %[[CMP_VAL:.+]]: f8E4M3FN, %[[NEW_VAL:.+]]: f8E4M3FN, %[[OUT:.+]]: f8E4M3FN):
// CHECK-NEXT: %[[COND:.+]] = arith.cmpf ueq, %[[MEM_VAL]], %[[CMP_VAL]] : f8E4M3FN
// CHECK-NEXT: %[[RESULT:.+]] = arith.select %[[COND]], %[[NEW_VAL]], %[[MEM_VAL]] : f8E4M3FN
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E4M3FN
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_cas
