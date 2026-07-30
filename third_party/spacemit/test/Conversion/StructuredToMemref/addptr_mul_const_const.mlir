// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : i32
  )
  {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32}:tensor<1024xi32>
    %2 = tt.splat %0 : i32 -> tensor<1024xi32>
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    //%3: splat(%0) + range(0, 1024)
    //%3: offset = %0, size = 1024, stride = 1
    // vector and scalar are both constant
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %c10 = arith.constant 10 : i32
    %5 = tt.splat %c10 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%c10)*range(2048, 4096);
    //%6: offset = %c10*2048, size = 1024, stride = %c10*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %c10*2048 + %0, size = 1024, stride = %c10*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0 offset = %c10*2048 + pid0, size = 1024, stride = %c10*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1, offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_20480_:%.+]] = arith.constant 20480 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = arith.addi [[VAR_0_]], [[CST_20480_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [1024], strides: [11] : memref<*xbf16> to memref<1024xbf16, strided<[11], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1024], strides: [1] : memref<*xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xbf16, strided<[11], offset: ?>> to memref<1024xbf16>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xbf16>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024xbf16>, memref<1024xbf16, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
