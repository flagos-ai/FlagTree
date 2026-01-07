// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// dtype : uint8
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32} , %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.uitofp %3 : tensor<32xi8> to tensor<32xf32>
%8 = arith.uitofp %6 : tensor<32xi8> to tensor<32xf32>
%9 = arith.divf %7, %8 : tensor<32xf32>
%10 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%11 = tt.addptr %10, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%12 = arith.fptoui %9 : tensor<32xf32> to tensor<32xi8>
tt.store %11, %12 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.uitofp [[ARG_1:%.+]] : tensor<32xi8> to tensor<32xf32>
// CHECK: [[VAL_2:%.+]] = arith.uitofp [[ARG_2:%.+]] : tensor<32xi8> to tensor<32xf32>
// CHECK: [[RES_0:%.+]] = arith.divf [[VAL_1]], [[VAL_2]] : tensor<32xf32>

// -----

// dtype : uint16
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32} , %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.uitofp %3 : tensor<32xi16> to tensor<32xf32>
%8 = arith.uitofp %6 : tensor<32xi16> to tensor<32xf32>
%9 = arith.divf %7, %8 : tensor<32xf32>
%10 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%11 = tt.addptr %10, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%12 = arith.fptoui %9 : tensor<32xf32> to tensor<32xi16>
tt.store %11, %12 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.uitofp [[ARG_1:%.+]] : tensor<32xi16> to tensor<32xf32>
// CHECK: [[VAL_2:%.+]] = arith.uitofp [[ARG_2:%.+]] : tensor<32xi16> to tensor<32xf32>
// CHECK: [[RES_0:%.+]] = arith.divf [[VAL_1]], [[VAL_2]] : tensor<32xf32>

// -----

// dtype : uint32
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32} , %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.uitofp %3 : tensor<32xi32> to tensor<32xf32>
%8 = arith.uitofp %6 : tensor<32xi32> to tensor<32xf32>
%9 = arith.divf %7, %8 : tensor<32xf32>
%10 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%11 = tt.addptr %10, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%12 = arith.fptoui %9 : tensor<32xf32> to tensor<32xi32>
tt.store %11, %12 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.uitofp [[ARG_1:%.+]] : tensor<32xi32> to tensor<32xf32>
// CHECK: [[VAL_2:%.+]] = arith.uitofp [[ARG_2:%.+]] : tensor<32xi32> to tensor<32xf32>
// CHECK: [[RES_0:%.+]] = arith.divf [[VAL_1]], [[VAL_2]] : tensor<32xf32>

// -----

// dtype : uint64
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.uitofp %3 : tensor<32xi64> to tensor<32xf32>
%8 = arith.uitofp %6 : tensor<32xi64> to tensor<32xf32>
%9 = arith.divf %7, %8 : tensor<32xf32>
%10 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%11 = tt.addptr %10, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%12 = arith.fptoui %9 : tensor<32xf32> to tensor<32xi64>
tt.store %11, %12 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.uitofp [[ARG_1:%.+]] : tensor<32xi64> to tensor<32xf32>
// CHECK: [[VAL_2:%.+]] = arith.uitofp [[ARG_2:%.+]] : tensor<32xi64> to tensor<32xf32>
// CHECK: [[RES_0:%.+]] = arith.divf [[VAL_1]], [[VAL_2]] : tensor<32xf32>

// -----


// dtype : float8_e4m3fn
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.divf %3, %6 : tensor<32xf8E4M3FN>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[RES_0:%.+]] = arith.divf [[ARG_1:%.+]], [[ARG_2:%.+]] : tensor<32xf8E4M3FN>


// -----

// dtype : float8_e5m2
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32} , %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.divf %3, %6 : tensor<32xf8E5M2>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[RES_0:%.+]] = arith.divf [[ARG_1:%.+]], [[ARG_2:%.+]] : tensor<32xf8E5M2>