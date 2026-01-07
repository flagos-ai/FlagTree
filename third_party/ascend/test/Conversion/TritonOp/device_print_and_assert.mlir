// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' %s | FileCheck %s

module {
  // CHECK: func.func private @triton_print_0(tensor<10xi8>) attributes {hex = false, prefix = " Type: uint8: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<10x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i8>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<i8>>
    // CHECK: call @triton_print_0
    tt.print " Type: uint8: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xi8>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xi16>) attributes {hex = false, prefix = " Type: uint16: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<10x!tt.ptr<i16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i16>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<i16>>
    // CHECK: call @triton_print_0
    tt.print " Type: uint16: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xi16>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xi32>) attributes {hex = false, prefix = " Type: uint32: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<10x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i32>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<i32>>
    // CHECK: call @triton_print_0
    tt.print " Type: uint32: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xi32>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xi64>) attributes {hex = false, prefix = " Type: uint64: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<10x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i64>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<i64>>
    // CHECK: call @triton_print_0
    tt.print " Type: uint64: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xi64>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xf32>) attributes {hex = false, prefix = " Type: float32: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<10x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f32>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<f32>>
    // CHECK: call @triton_print_0
    tt.print " Type: float32: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xf32>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xf16>) attributes {hex = false, prefix = " Type: float16: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<10x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<f16>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<f16>>
    // CHECK: call @triton_print_0
    tt.print " Type: float16: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xf16>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<10xbf16>) attributes {hex = false, prefix = " Type: bfloat16: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<10x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<bf16>>, tensor<10xi32>
    %3 = tt.load %2 : tensor<10x!tt.ptr<bf16>>
    // CHECK: call @triton_print_0
    tt.print " Type: bfloat16: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<10xbf16>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_assert_0(tensor<10xi1>) attributes {msg = "device_assert fail!"}
  // CHECK: func.func private @triton_print_0(tensor<10xi1>) attributes {hex = false, prefix = " Type: bool (int1): "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<true> : tensor<10xi1>
    %cst_0 = arith.constant dense<0> : tensor<10xi8>
    %0 = tt.make_range {end = 10 : i32, start = 0 : i32} : tensor<10xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<10x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<10x!tt.ptr<i1>>, tensor<10xi32>
    %3 = tt.bitcast %2 : tensor<10x!tt.ptr<i1>> -> tensor<10x!tt.ptr<i8>>
    %4 = tt.load %3 : tensor<10x!tt.ptr<i8>>
    %5 = arith.cmpi ne, %4, %cst_0 : tensor<10xi8>
    // CHECK: call @triton_assert_0
    // CHECK: call @triton_print_0
    tt.assert %cst, "device_assert fail!" : tensor<10xi1>
    tt.print " Type: bool (int1): " {hex = false, isSigned = array<i32: 1>} : %5 : tensor<10xi1>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<16xf8E5M2>)
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E5M2> -> tensor<16x!tt.ptr<f8E5M2>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f8E5M2>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f8E5M2>>
    // CHECK: call @triton_print_0
    tt.print " val: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<16xf8E5M2>
    tt.return
  }
}

// -----

module {
  // CHECK: func.func private @triton_print_0(tensor<16xf8E4M3FN>) attributes {hex = false, prefix = " val: "}
  // CHECK: func.func @device_print_kernel
  tt.func public @device_print_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<16x!tt.ptr<f8E4M3FN>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f8E4M3FN>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f8E4M3FN>>
    // CHECK: call @triton_print_0 
    tt.print " val: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<16xf8E4M3FN>
    tt.return
  }
}