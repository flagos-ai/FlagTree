// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' %s | FileCheck %s

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<10x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f8E4M3FN>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<f8E4M3FN>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<10x!tt.ptr<f8E4M3FN>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<f8E4M3FN>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<10x!tt.ptr<f8E5M2>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f8E5M2>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<f8E5M2>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<10x!tt.ptr<f8E5M2>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<f8E5M2>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<f8E5M2>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<10x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i32>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<i32>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<10x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<i32>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<i32>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<10x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i64>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<i64>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<10x!tt.ptr<i64>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<i64>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<i64>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<10x!tt.ptr<i16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i16>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<i16>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<10x!tt.ptr<i16>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<i16>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<i16>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<10x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i8>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<i8>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<10x!tt.ptr<i8>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<i8>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<i8>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<10x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f32>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<f32>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<10x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<f32>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<10x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f16>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<f16>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<10x!tt.ptr<f16>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<f16>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<f16>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<10x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<bf16>>, tensor<10xi32>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %3 = tt.load %2 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<bf16>>
    // CHECK: gpu.barrier
    gpu.barrier
    %4 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<10x!tt.ptr<bf16>>
    %5 = tt.addptr %4, %0 : tensor<10x!tt.ptr<bf16>>, tensor<10xi32>
    tt.store %5, %3 : tensor<10x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

module {
  tt.func public @compile_hint_kernel(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    // CHECK: "llvm.intr.assume"(%true) : (i1) -> ()
    "llvm.intr.assume"(%true) : (i1) -> ()
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<10x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i1>>, tensor<10xi32>
    %3 = tt.bitcast %2 : tensor<10x!tt.ptr<i1>> -> tensor<10x!tt.ptr<i8>>
    // CHECK-NOT:tt.constancy
    // CHECK-NOT:tt.contiguity
    // CHECK-NOT:tt.divisibility
    %4 = tt.load %3 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<1> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>} : tensor<10x!tt.ptr<i8>>
    // CHECK: gpu.barrier
    gpu.barrier
    %5 = tt.splat %arg1 : !tt.ptr<i1> -> tensor<10x!tt.ptr<i1>>
    %6 = tt.addptr %5, %0 : tensor<10x!tt.ptr<i1>>, tensor<10xi32>
    %7 = tt.bitcast %6 : tensor<10x!tt.ptr<i1>> -> tensor<10x!tt.ptr<i8>>
    tt.store %7, %4 : tensor<10x!tt.ptr<i8>>
    tt.return
  }
}