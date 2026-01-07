// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// dtype : uint8
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32} , %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.remui %3, %6 : tensor<32xi8>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[REINTERPRET_CAST:%.+]] = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [0], sizes: [32], strides: [1] : memref<?xi8> to memref<32xi8, strided<[1]>>
// CHECK: [[ALLOC:%.+]] = memref.alloc() : memref<32xi8>
// CHECK: memref.copy [[REINTERPRET_CAST]], [[ALLOC]] : memref<32xi8, strided<[1]>> to memref<32xi8>
// CHECK: [[VAL_1:%.+]] = bufferization.to_tensor [[ALLOC]] restrict writable : memref<32xi8>
// CHECK: [[REINTERPRET_CAST_0:%.+]] = memref.reinterpret_cast [[ARG_2:%.+]] to offset: [0], sizes: [32], strides: [1] : memref<?xi8> to memref<32xi8, strided<[1]>>
// CHECK: [[ALLOC_1:%.+]] = memref.alloc() : memref<32xi8>
// CHECK: memref.copy [[REINTERPRET_CAST_0]], [[ALLOC_1]] : memref<32xi8, strided<[1]>> to memref<32xi8>
// CHECK: [[VAL_2:%.+]] = bufferization.to_tensor [[ALLOC_1]] restrict writable : memref<32xi8>
// CHECK: [[RES_0:%.+]] = arith.remui [[VAL_1]], [[VAL_2]] : tensor<32xi8>

// -----

// dtype : uint16
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.remui %3, %6 : tensor<32xi16>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[RES_0:%.+]] = arith.remui [[VAL_1]], [[VAL_2]] : tensor<32xi16>

// -----

// dtype : uint32
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.remui %3, %6 : tensor<32xi32>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[RES_0:%.+]] = arith.remui [[VAL_1]], [[VAL_2]] : tensor<32xi32>

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
%7 = arith.remui %3, %6 : tensor<32xi64>
%8 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%9 = tt.addptr %8, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %9, %7 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[RES_0:%.+]] = arith.remui [[VAL_1]], [[VAL_2]] : tensor<32xi64>


// -----

// dtype: bool
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.bitcast %2 : tensor<32x!tt.ptr> -> tensor<32x!tt.ptr>
%4 = tt.load %3 : tensor<32x!tt.ptr>
%5 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%6 = tt.addptr %5, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%7 = tt.bitcast %6 : tensor<32x!tt.ptr> -> tensor<32x!tt.ptr>
%8 = tt.load %7 : tensor<32x!tt.ptr>
%9 = arith.remsi %4, %8 : tensor<32xi8>
%10 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%11 = tt.addptr %10, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%12 = tt.bitcast %11 : tensor<32x!tt.ptr> -> tensor<32x!tt.ptr>
tt.store %12, %9 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[REINTERPRET_CAST:%.+]] = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [0], sizes: [32], strides: [1] : memref<?xi8> to memref<32xi8, strided<[1]>>
// CHECK: [[RES_0:%.+]] = arith.remsi [[VAL_1:%.+]], [[VAL_2:%.+]] : tensor<32xi8>


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
%8 = math.floor %7 : tensor<32xf8E4M3FN>
%9 = arith.mulf %8, %6 : tensor<32xf8E4M3FN>
%10 = arith.subf %3, %9 : tensor<32xf8E4M3FN>
%11 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%12 = tt.addptr %11, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %12, %10 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.divf [[ARG_1:%.+]], [[ARG_2:%.+]] : tensor<32xf8E4M3FN>
// CHECK: [[VAL_2:%.+]] = math.floor [[VAL_1]] : tensor<32xf8E4M3FN>
// CHECK: [[VAL_3:%.+]] = arith.mulf [[VAL_2]], [[ARG_2]] : tensor<32xf8E4M3FN>
// CHECK: [[VAL_4:%.+]] = arith.subf [[ARG_1]], [[VAL_3]] : tensor<32xf8E4M3FN>

// -----

// dtype : float8_e5m2
module {
tt.func public @triton_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
%1 = tt.splat %arg1 : !tt.ptr -> tensor<32x!tt.ptr>
%2 = tt.addptr %1, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%3 = tt.load %2 : tensor<32x!tt.ptr>
%4 = tt.splat %arg2 : !tt.ptr -> tensor<32x!tt.ptr>
%5 = tt.addptr %4, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
%6 = tt.load %5 : tensor<32x!tt.ptr>
%7 = arith.divf %3, %6 : tensor<32xf8E5M2>
%8 = math.floor %7 : tensor<32xf8E5M2>
%9 = arith.mulf %8, %6 : tensor<32xf8E5M2>
%10 = arith.subf %3, %9 : tensor<32xf8E5M2>
%11 = tt.splat %arg0 : !tt.ptr -> tensor<32x!tt.ptr>
%12 = tt.addptr %11, %0 : tensor<32x!tt.ptr>, tensor<32xi32>
tt.store %12, %10 : tensor<32x!tt.ptr>
tt.return
}
}

// CHECK: [[VAL_1:%.+]] = arith.divf [[ARG_1:%.+]], [[ARG_2:%.+]] : tensor<32xf8E5M2>
// CHECK: [[VAL_2:%.+]] = math.floor [[VAL_1]] : tensor<32xf8E5M2>
// CHECK: [[VAL_3:%.+]] = arith.mulf [[VAL_2]], [[ARG_2]] : tensor<32xf8E5M2>
// CHECK: [[VAL_4:%.+]] = arith.subf [[ARG_1]], [[VAL_3]] : tensor<32xf8E5M2>