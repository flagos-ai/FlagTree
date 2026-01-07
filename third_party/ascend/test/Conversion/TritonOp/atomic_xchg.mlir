// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %14 = tt.atomic_rmw exch, acq_rel, gpu, %12, %13, %6 : (tensor<512x!tt.ptr<i1>>, tensor<512xi1>, tensor<512xi1>) -> tensor<512xi1>
    %15 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<512x!tt.ptr<i1>>
    %16 = tt.addptr %15, %4 : tensor<512x!tt.ptr<i1>>, tensor<512xi32>
    %17 = tt.bitcast %16 : tensor<512x!tt.ptr<i1>> -> tensor<512x!tt.ptr<i8>>
    %18 = arith.extui %14 : tensor<512xi1> to tensor<512xi8>
    tt.store %17, %18, %6 : tensor<512x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i1, %[[IN_VAL:.+]]: i1, %[[OUT_VAL:.+]]: i1):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : i1
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i8>>, tensor<512xi8>, tensor<512xi1>) -> tensor<512xi8>
    %13 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<512x!tt.ptr<i8>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<i8>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i8, %[[IN_VAL:.+]]: i8, %[[OUT_VAL:.+]]: i8):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : i8
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i16>>, tensor<512xi16>, tensor<512xi1>) -> tensor<512xi16>
    %13 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<512x!tt.ptr<i16>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<i16>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i16, %[[IN_VAL:.+]]: i16, %[[OUT_VAL:.+]]: i16):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : i16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i32>>, tensor<512xi32>, tensor<512xi1>) -> tensor<512xi32>
    %13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[IN_VAL:.+]]: i32, %[[OUT_VAL:.+]]: i32):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<i32>>, tensor<512xi32>, tensor<512xi1>) -> tensor<512xi32>
    %13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<512x!tt.ptr<i32>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<i32>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: i32, %[[IN_VAL:.+]]: i32, %[[OUT_VAL:.+]]: i32):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : i32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf16>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f16>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<f16>>
    %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<f16>>, tensor<512xi32>
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<f16>>, tensor<512xf16>, tensor<512xi1>) -> tensor<512xf16>
    %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<f16>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f16, %[[IN_VAL:.+]]: f16, %[[OUT_VAL:.+]]: f16):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : f16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<f32>>, tensor<512xf32>, tensor<512xi1>) -> tensor<512xf32>
    %13 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f32, %[[IN_VAL:.+]]: f32, %[[OUT_VAL:.+]]: f32):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : f32
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xbf16>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<bf16>>
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<bf16>>, tensor<512xbf16>, tensor<512xi1>) -> tensor<512xbf16>
    %13 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: bf16, %[[IN_VAL:.+]]: bf16, %[[OUT_VAL:.+]]: bf16):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : bf16
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf8E5M2>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<512x!tt.ptr<f8E5M2>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f8E5M2>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<f8E5M2>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<512x!tt.ptr<f8E5M2>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<f8E5M2>>, tensor<512xi32>
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<f8E5M2>>, tensor<512xf8E5M2>, tensor<512xi1>) -> tensor<512xf8E5M2>
    %13 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<512x!tt.ptr<f8E5M2>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<f8E5M2>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E5M2, %[[IN_VAL:.+]]: f8E5M2, %[[OUT_VAL:.+]]: f8E5M2):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : f8E5M2
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw

// -----

module {
  tt.func public @atomic_xchg(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf8E4M3FN>
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<512x!tt.ptr<f8E4M3FN>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f8E4M3FN>>, tensor<512xi32>
    %9 = tt.load %8, %6, %cst : tensor<512x!tt.ptr<f8E4M3FN>>
    %10 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<512x!tt.ptr<f8E4M3FN>>
    %11 = tt.addptr %10, %2 : tensor<512x!tt.ptr<f8E4M3FN>>, tensor<512xi32>
    %12 = tt.atomic_rmw exch, acq_rel, gpu, %11, %9, %6 : (tensor<512x!tt.ptr<f8E4M3FN>>, tensor<512xf8E4M3FN>, tensor<512xi1>) -> tensor<512xf8E4M3FN>
    %13 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<512x!tt.ptr<f8E4M3FN>>
    %14 = tt.addptr %13, %4 : tensor<512x!tt.ptr<f8E4M3FN>>, tensor<512xi32>
    tt.store %14, %12, %6 : tensor<512x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// CHECK-LABEL: func.func @atomic_xchg
// CHECK: linalg.generic {{.*}} attrs =  {GenericAtomicRMW = "exch", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK-NEXT: ^bb0(%[[MEM_VAL:.+]]: f8E4M3FN, %[[IN_VAL:.+]]: f8E4M3FN, %[[OUT_VAL:.+]]: f8E4M3FN):
// CHECK-NEXT: linalg.yield %[[IN_VAL]] : f8E4M3FN
// CHECK-NEXT: }
// CHECK-NOT: tt.atomic_rmw