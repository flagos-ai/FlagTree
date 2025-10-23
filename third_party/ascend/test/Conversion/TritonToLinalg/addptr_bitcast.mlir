// RUN: triton-adapter-opt --triton-to-linalg %s --split-input-file | FileCheck %s 
module {
  // CHECK-LABEL: func.func @addptr_bitcast 
  tt.func public @addptr_bitcast(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_2 = arith.constant dense<0> : tensor<64xi8>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %7 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<64x!tt.ptr<i1>>
    %8 = tt.addptr %7, %6 : tensor<64x!tt.ptr<i1>>, tensor<64xi32>
    %81 = tt.bitcast %8 : tensor<64x!tt.ptr<i1>> -> tensor<64x!tt.ptr<i8>>
    // CHECK: %[[SRC:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [64], strides: [1] : memref<?xi8> to memref<64xi8, strided<[1]>>   
    // CHECK: %[[DST:.*]] = memref.alloc() : memref<64xi8>
    // CHECK: memref.copy %[[SRC]], %[[DST]] : memref<64xi8, strided<[1]>> to memref<64xi8>
    %10 = tt.load %81 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<64x!tt.ptr<i8>>
    %11 = arith.cmpi ne, %10, %cst_2 : tensor<64xi8>
    %12 = arith.uitofp %11 : tensor<64xi1> to tensor<64xf32>
    %16 = tt.make_tensor_ptr %arg1, [%c64_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<64xf32>>
    tt.store %16, %12 : !tt.ptr<tensor<64xf32>>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @addptr_bitcast2 
  tt.func public @addptr_bitcast2(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f16>>
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f16>>
    %11 = arith.cmpf olt, %7, %10 : tensor<1024xf16>
    %12 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %13 = tt.addptr %12, %4 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    %14 = tt.bitcast %13 : tensor<1024x!tt.ptr<i1>> -> tensor<1024x!tt.ptr<i8>>
    // CHECK: %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [[[VAR_1:%.+]]], sizes: [1024], strides: [1] : memref<?xi8> to memref<1024xi8, strided<[1], offset: ?>>
    %15 = arith.extui %11 : tensor<1024xi1> to tensor<1024xi8>
    tt.store %14, %15 : tensor<1024x!tt.ptr<i8>>
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @addptr_bitcast3
  // CHECK-ORIG-SAME:    %[[ARG0_I8:.*]]: memref<*xi8>  
  tt.func public @addptr_bitcast3(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1xi8>
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f16>>
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %9 = tt.addptr %8, %4 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f16>>
    %11 = tt.addptr %arg0, %0 : !tt.ptr<i1>, i32
    %12 = tt.bitcast %11 : !tt.ptr<i1> -> !tt.ptr<i8>    
    // CHECK: %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [%[[PID_0:.*]]], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
    %13 = tt.splat %12 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %14 = tt.load %13 : tensor<1x!tt.ptr<i8>>
    %15 = arith.cmpi ne, %14, %cst : tensor<1xi8>
    %16 = tt.broadcast %15 : tensor<1xi1> -> tensor<1024xi1>
    %17 = arith.select %16, %7, %10 : tensor<1024xi1>, tensor<1024xf16>
    tt.store %6, %17 : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func @addptr_bitcast_bool_1_tensor
tt.func public @addptr_bitcast_bool_1_tensor(%arg0 : !tt.ptr<i1>, %b : i32) -> () {
  %16 = tt.splat %b : i32 -> tensor<1xi32>
  %17 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1x!tt.ptr<i1>>
  //CHECK: %[[VAR_0:.*]]  = arith.index_cast [[ARG_B:%.+]] : i32 to index
  // CHECK: %[[RECAST:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [%0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1], offset: ?>>
  %18 = tt.addptr %17, %16 : tensor<1x!tt.ptr<i1>>, tensor<1xi32>
  %19 = tt.bitcast %18 : tensor<1x!tt.ptr<i1>> -> tensor<1x!tt.ptr<i8>>
  // CHECK: memref.copy %[[RECAST]], %[[ALLOC:.*]]
  %20 = tt.load %19 evictionPolicy = evict_last : tensor<1x!tt.ptr<i8>>
  tt.store %19, %20 : tensor<1x!tt.ptr<i8>>
  tt.return
}

// -----

// CHECK-LABEL: func @addptr_bitcast_bool_1_scalar
tt.func public @addptr_bitcast_bool_1_scalar(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i1>, %arg2: i32) {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %[[VAL_c0:.*]] = arith.constant 0 : index
  // CHECK: %[[RECAST0:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  %0 = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
  // CHECK:           %[[ALLOC:.*]] = memref.alloc() : memref<1xi8>
  // CHECK:           memref.copy %[[RECAST0]], %[[ALLOC]] : memref<1xi8, strided<[1]>> to memref<1xi8>
  // CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<1xi8>
  // CHECK:           %[[VAL_0:.*]] = tensor.extract %[[TENSOR]]{{\[}}%[[VAL_c0]]] : tensor<1xi8>
  %1 = tt.load %0 : !tt.ptr<i8>
  %2 = tt.addptr %arg1, %c0_i32 : !tt.ptr<i1>, i32
  %3 = tt.bitcast %2 : !tt.ptr<i1> -> !tt.ptr<i8>
  // CHECK: %[[VAL_1:.*]] = tensor.empty() : tensor<1xi8>
  // CHECK: %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : i8) outs(%[[VAL_1]] : tensor<1xi8>) -> tensor<1xi8>
  // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi8> to memref<1xi8, strided<[1]>>
  // CHECK: bufferization.materialize_in_destination %[[VAL_2]] in writable %reinterpret_cast_0 : (tensor<1xi8>, memref<1xi8, strided<[1]>>) -> ()
  tt.store %3, %1 : !tt.ptr<i8>
  tt.return
}
