// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s
module {
  tt.func public @atomic_add_i8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<i8>>
    %8 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    %10 = tt.atomic_rmw add, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<i8>>, tensor<768xi8>, tensor<768xi1>) -> tensor<768xi8>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 1>} : %10 : tensor<768xi8>
    %11 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<768x!tt.ptr<i8>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<i8>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_i8
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: i8, %[[IN2:.+]]: i8, %[[OUT:.+]]: i8):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[IN1]], %[[IN2]] : i8
// CHECK-NEXT: linalg.yield %[[RESULT]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw


// -----


module {
  tt.func public @atomic_add_i16(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    %10 = tt.atomic_rmw add, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<i16>>, tensor<768xi16>, tensor<768xi1>) -> tensor<768xi16>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 1>} : %10 : tensor<768xi16>
    %11 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<768x!tt.ptr<i16>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<i16>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_i16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: i16, %[[IN2:.+]]: i16, %[[OUT:.+]]: i16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[IN1]], %[[IN2]] : i16
// CHECK-NEXT: linalg.yield %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw


// -----


module {
  tt.func public @atomic_add_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    %10 = tt.atomic_rmw add, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<i32>>, tensor<768xi32>, tensor<768xi1>) -> tensor<768xi32>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 1>} : %10 : tensor<768xi32>
    %11 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<768x!tt.ptr<i32>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<i32>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_i32
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: i32, %[[IN2:.+]]: i32, %[[OUT:.+]]: i32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[IN1]], %[[IN2]] : i32
// CHECK-NEXT: linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw


// -----

module {
  tt.func public @atomic_add_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<i64>>
    %8 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    %10 = tt.atomic_rmw add, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<i64>>, tensor<768xi64>, tensor<768xi1>) -> tensor<768xi64>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 1>} : %10 : tensor<768xi64>
    %11 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<768x!tt.ptr<i64>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<i64>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_i64
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: i64, %[[IN2:.+]]: i64, %[[OUT:.+]]: i64):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[IN1]], %[[IN2]] : i64
// CHECK-NEXT: linalg.yield %[[RESULT]] : i64
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw


// -----

module {
  tt.func public @atomic_add_fp16(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<f16>>
    %8 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<f16>>, tensor<768xf16>, tensor<768xi1>) -> tensor<768xf16>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<768xf16>
    %11 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<768x!tt.ptr<f16>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<f16>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_fp16
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: f16, %[[IN2:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addf %[[IN1]], %[[IN2]] : f16
// CHECK-NEXT: linalg.yield %[[RESULT]] : f16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_add_f32(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<f32>>
    %8 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<f32>>, tensor<768xf32>, tensor<768xi1>) -> tensor<768xf32>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<768xf32>
    %11 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<768x!tt.ptr<f32>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<f32>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_f32
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: f32, %[[IN2:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addf %[[IN1]], %[[IN2]] : f32
// CHECK-NEXT: linalg.yield %[[RESULT]] : f32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_add_f8E5M2(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<f8E5M2>>
    %8 = tt.splat %arg3 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<f8E5M2>>, tensor<768xf8E5M2>, tensor<768xi1>) -> tensor<768xf8E5M2>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<768xf8E5M2>
    %11 = tt.splat %arg4 : !tt.ptr<f8E5M2> -> tensor<768x!tt.ptr<f8E5M2>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<f8E5M2>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_f8E5M2
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: f8E5M2, %[[IN2:.+]]: f8E5M2, %[[OUT:.+]]: f8E5M2):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addf %[[IN1]], %[[IN2]] : f8E5M2
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E5M2
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_add(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<bf16>>
    %8 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<bf16>>, tensor<768xbf16>, tensor<768xi1>) -> tensor<768xbf16>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<768xbf16>
    %11 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<768x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<bf16>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: bf16, %[[IN2:.+]]: bf16, %[[OUT:.+]]: bf16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addf %[[IN1]], %[[IN2]] : bf16
// CHECK-NEXT: linalg.yield %[[RESULT]] : bf16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_add(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %cst_0 = arith.constant dense<0> : tensor<768xi8>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<768x!tt.ptr<i1>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<i1>>, tensor<768xi32>
    %7 = tt.bitcast %6 : tensor<768x!tt.ptr<i1>> -> tensor<768x!tt.ptr<i8>>
    %8 = tt.load %7 : tensor<768x!tt.ptr<i8>>
    %9 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<768x!tt.ptr<i1>>
    %10 = tt.addptr %9, %2 : tensor<768x!tt.ptr<i1>>, tensor<768xi32>
    %11 = arith.cmpi ne, %8, %cst_0 : tensor<768xi8>
    %12 = tt.atomic_rmw add, acq_rel, gpu, %10, %11, %cst : (tensor<768x!tt.ptr<i1>>, tensor<768xi1>, tensor<768xi1>) -> tensor<768xi1>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 1>} : %12 : tensor<768xi1>
    %13 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<768x!tt.ptr<i1>>
    %14 = tt.addptr %13, %2 : tensor<768x!tt.ptr<i1>>, tensor<768xi32>
    %15 = tt.bitcast %14 : tensor<768x!tt.ptr<i1>> -> tensor<768x!tt.ptr<i8>>
    %16 = arith.extui %12 : tensor<768xi1> to tensor<768xi8>
    tt.store %15, %16 : tensor<768x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: i1, %[[IN2:.+]]: i1, %[[OUT:.+]]: i1):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addi %[[IN1]], %[[IN2]] : i1
// CHECK-NEXT: linalg.yield %[[RESULT]] : i1
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_add_f8E4M3FN(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<768xi1>
    %c768_i32 = arith.constant 768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c768_i32 : i32
    %2 = tt.make_range {end = 768 : i32, start = 0 : i32} : tensor<768xi32>
    %3 = tt.splat %1 : i32 -> tensor<768xi32>
    %4 = arith.addi %3, %2 : tensor<768xi32>
    %5 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %6 = tt.addptr %5, %4 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    %7 = tt.load %6 : tensor<768x!tt.ptr<f8E4M3FN>>
    %8 = tt.splat %arg3 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %2 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %7, %cst : (tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xf8E4M3FN>, tensor<768xi1>) -> tensor<768xf8E4M3FN>
    tt.print " tmp1: " {hex = false, isSigned = array<i32: 0>} : %10 : tensor<768xf8E4M3FN>
    %11 = tt.splat %arg4 : !tt.ptr<f8E4M3FN> -> tensor<768x!tt.ptr<f8E4M3FN>>
    %12 = tt.addptr %11, %2 : tensor<768x!tt.ptr<f8E4M3FN>>, tensor<768xi32>
    tt.store %12, %10 : tensor<768x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_add_f8E4M3FN
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[IN1:.+]]: f8E4M3FN, %[[IN2:.+]]: f8E4M3FN, %[[OUT:.+]]: f8E4M3FN):
// CHECK-NEXT: %[[RESULT:.+]] = arith.addf %[[IN1]], %[[IN2]] : f8E4M3FN
// CHECK-NEXT: linalg.yield %[[RESULT]] : f8E4M3FN
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw
