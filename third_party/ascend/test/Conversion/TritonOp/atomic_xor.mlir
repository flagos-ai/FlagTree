// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @atomic_xor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<512xi8>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<i8>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<i8>>
    %10 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<i8>>, tensor<512xi32>
    %12 = tt.atomic_rmw xor, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i8>>, tensor<512xi8>, tensor<512xi1>) -> tensor<512xi8>
    %13 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>>
    %14 = tt.addptr %13, %2 : tensor<512x!tt.ptr<i8>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xor
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "xor", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i8, %[[IN_VAL:.+]]: i8, %[[OUT_VAL:.+]]: i8):
// CHECK-NEXT: %[[RESULT:.+]] = arith.xori %[[MEM_VAL]], %[[IN_VAL]] : i8
// CHECK-NEXT: linalg.yield %[[RESULT]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xor(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<512xi16>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<i16>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<i16>>
    %10 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<i16>>, tensor<512xi32>
    %12 = tt.atomic_rmw xor, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i16>>, tensor<512xi16>, tensor<512xi1>) -> tensor<512xi16>
    %13 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>>
    %14 = tt.addptr %13, %2 : tensor<512x!tt.ptr<i16>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xor
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "xor", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i16, %[[IN_VAL:.+]]: i16, %[[OUT_VAL:.+]]: i16):
// CHECK-NEXT: %[[RESULT:.+]] = arith.xori %[[MEM_VAL]], %[[IN_VAL]] : i16
// CHECK-NEXT: linalg.yield %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xor(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<512xi32>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<i32>>
    %10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    %12 = tt.atomic_rmw xor, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i32>>, tensor<512xi32>, tensor<512xi1>) -> tensor<512xi32>
    %13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %14 = tt.addptr %13, %2 : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xor
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "xor", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[IN_VAL:.+]]: i32, %[[OUT_VAL:.+]]: i32):
// CHECK-NEXT: %[[RESULT:.+]] = arith.xori %[[MEM_VAL]], %[[IN_VAL]] : i32
// CHECK-NEXT: linalg.yield %[[RESULT]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xor(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<512xi64>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<i64>>
    %10 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %12 = tt.atomic_rmw xor, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i64>>, tensor<512xi64>, tensor<512xi1>) -> tensor<512xi64>
    %13 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %14 = tt.addptr %13, %2 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xor
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "xor", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i64, %[[IN_VAL:.+]]: i64, %[[OUT_VAL:.+]]: i64):
// CHECK-NEXT: %[[RESULT:.+]] = arith.xori %[[MEM_VAL]], %[[IN_VAL]] : i64
// CHECK-NEXT: linalg.yield %[[RESULT]] : i64
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xor(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<512xi8>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<512x!tt.ptr<i1>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<i1>>, tensor<512xi32>
    %9 = tt.bitcast %8 : tensor<512x!tt.ptr<i1>> -> tensor<512x!tt.ptr<i8>>
    %10 = tt.load %9, %6, %cst : tensor<512x!tt.ptr<i8>>
    %11 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<512x!tt.ptr<i1>>
    %12 = tt.addptr %11, %2 : tensor<512x!tt.ptr<i1>>, tensor<512xi32>
    %13 = arith.cmpi ne, %10, %cst : tensor<512xi8>
    %14 = tt.atomic_rmw xor, acq_rel, gpu, %12, %13, %6 : (tensor<512x!tt.ptr<i1>>, tensor<512xi1>, tensor<512xi1>) -> tensor<512xi1>
    %15 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<512x!tt.ptr<i1>>
    %16 = tt.addptr %15, %2 : tensor<512x!tt.ptr<i1>>, tensor<512xi32>
    %17 = tt.bitcast %16 : tensor<512x!tt.ptr<i1>> -> tensor<512x!tt.ptr<i8>>
    %18 = arith.extui %14 : tensor<512xi1> to tensor<512xi8>
    tt.store %17, %18, %6 : tensor<512x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xor
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "xor", MemSemantic = "acq_rel", MemSyncScope = "gpu", Software} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i1, %[[IN_VAL:.+]]: i1, %[[OUT_VAL:.+]]: i1):
// CHECK-NEXT: %[[RESULT:.+]] = arith.xori %[[MEM_VAL]], %[[IN_VAL]] : i1
// CHECK-NEXT: linalg.yield %[[RESULT]] : i1
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw