// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' %s | FileCheck %s

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i32>>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<i32>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (i32, i32) -> i32
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xi32>, tensor<2xi32> -> tensor<2xi32>
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<i32>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i64>>
    %4 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<i64>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (i64, i64) -> i64
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xi64>, tensor<2xi64> -> tensor<2xi64>
    %8 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<i64>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<2x!tt.ptr<i16>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i16>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i16>>
    %4 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<2x!tt.ptr<i16>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<i16>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<i16>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (i16, i16) -> i16
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xi16>, tensor<2xi16> -> tensor<2xi16>
    %8 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<2x!tt.ptr<i16>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<i16>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<i16>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<2x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i8>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i8>>
    %4 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<2x!tt.ptr<i8>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<i8>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<i8>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (i8, i8) -> i8
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xi8>, tensor<2xi8> -> tensor<2xi8>
    %8 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<2x!tt.ptr<i8>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<i8>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<i8>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<2x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<f16>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<f16>>
    %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<2x!tt.ptr<f16>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<f16>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<f16>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (f16, f16) -> f16
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<2x!tt.ptr<f16>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<f16>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<f16>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<f32>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (f32, f32) -> f32
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
    %8 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<bf16>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<bf16>>
    %4 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<2x!tt.ptr<bf16>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<bf16>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<bf16>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (bf16, bf16) -> bf16
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xbf16>, tensor<2xbf16> -> tensor<2xbf16>
    %8 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<2x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<bf16>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<2x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<f8E4M3FN>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<f8E4M3FN>>
    %4 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<2x!tt.ptr<f8E4M3FN>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<f8E4M3FN>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<f8E4M3FN>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (f8E4M3FN, f8E4M3FN) -> f8E4M3FN
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xf8E4M3FN>, tensor<2xf8E4M3FN> -> tensor<2xf8E4M3FN>
    %8 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<2x!tt.ptr<f8E4M3FN>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<f8E4M3FN>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<f8E4M3FN>>
    tt.return
  }
}

// -----

module {
  tt.func public @triton_asm_add(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<2x!tt.ptr<f8E5M2>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<f8E5M2>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<f8E5M2>>
    %4 = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<2x!tt.ptr<f8E5M2>>
    %5 = tt.addptr %4, %0 : tensor<2x!tt.ptr<f8E5M2>>, tensor<2xi32>
    %6 = tt.load %5 : tensor<2x!tt.ptr<f8E5M2>>
    // CHECK: %[[ASM_RESULT1:.*]] = llvm.inline_asm asm_dialect = att "\0A        ADD.s64 $0, $1, $2\0A        ", "=l,l,l" %{{.*}}, %{{.*}} : (f8E5M2, f8E5M2) -> f8E5M2
    %7 = tt.elementwise_inline_asm "\0A        ADD.s64 $0, $1, $2\0A        " {constraints = "=l,l,l", packed_element = 1 : i32, pure = true} %3, %6 : tensor<2xf8E5M2>, tensor<2xf8E5M2> -> tensor<2xf8E5M2>
    %8 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<2x!tt.ptr<f8E5M2>>
    %9 = tt.addptr %8, %0 : tensor<2x!tt.ptr<f8E5M2>>, tensor<2xi32>
    tt.store %9, %7 : tensor<2x!tt.ptr<f8E5M2>>
    tt.return
  }
}