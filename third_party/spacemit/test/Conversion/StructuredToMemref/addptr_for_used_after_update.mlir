// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>
  )
  {
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %1 = tt.make_range {end = 1280 : i32, start = 1024 : i32}:tensor<256xi32>
    // source: null, sizes: 256, offsets: 1024, strides: 1
    %2 = tt.addptr %0, %1 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    // source: arg0, sizes: 256, offsets: 1024, strides: 1
    // gep operand is another gep' output, which is passed into the loop as varible, used after update
    %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        // pointer updates
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        // sizes: 256, offsets: 3, strides: 0
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        // source: arg0, sizes: 256, offsets: 1024 + i, strides: 1
        // perform load
        %3 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr_iter, %3 : tensor<256x!tt.ptr<bf16>>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  %subview = memref.subview %arg0, [%4][256][4] : memref<> -> memref<>                    <- generate subview on getelementptr (already done)
    //  ...
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    // TODO: examples below are not supported since scf.for does not support returning a tensor type
    // Example 3, gep operand is a vector of i32, which is passed into the loop as variable, pointer updated using step, used after update
    //%_ptr3 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %1) -> (tensor<256xi32>) {
    //    // offset update
    //    %3 = tt.splat %c3 : i32 -> tensor<256xi32>
    //    %ptr_iter = arith.addi %3, %ptr : tensor<256xi32>
    //    // generate pointer
    //    %gep_ptr = tt.addptr %0, %ptr_iter : tensor<256x!tt.ptr<bf16>>
    //    // perform load
    //    %4 = tt.load %gep_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
    //    tt.store %gep_ptr, %4 : tensor<256x!tt.ptr<bf16>>
    //    scf.yield %ptr_iter : tensor<256xi32>
    //}
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  %subview = memref.subview %arg0, [%offset_dim0_iter][256][4] : memref<> -> memref<>     <- generate subview on load (new)
    //  ...
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    //// Example 4, gep operand is a vector of i32, which is passed into the loop as variable, pointer updated using step, used before update
    //%_ptr4 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %1) -> (tensor<256xi32>) {
    //    // generate pointer
    //    %gep_ptr = tt.addptr %0, %ptr : tensor<256x!tt.ptr<bf16>>
    //
    //    // perform load
    //    %4 = tt.load %gep_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
    //    tt.store %gep_ptr, %4 : tensor<256x!tt.ptr<bf16>>
    //    // offset update
    //    %3 = tt.splat %c3 : i32 -> tensor<256xi32>
    //    %ptr_iter = arith.addi %3, %ptr : tensor<256xi32>
    //    scf.yield %ptr_iter : tensor<256xi32>
    //}
    // Expected output
    // %offset_dim0 = arith.constant 1024                                                       <- insert instructions to initialize init arg(new)
    // for iter_args (%offset_dim0_iter = %offset_dim0) {                                       <- replace varibles passed in as init arg (new)
    //  %subview = memref.subview %arg0, [%offset_dim0_iter][256][4] : memref<> -> memref<>     <- generate subview on load (new)
    //  ...
    //  %4 = %offset_dim0_iter + %c3                                                            <- replace gep of splat with add (already done)
    //  scf.yield %4                                                                            <- replace yielding an gep output with the corresponding dim variable (new)
    // }
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = scf.for [[VAR_arg7_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg8_:%.+]] = [[CST_1024_]]) -> (index) {
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.addi [[VAR_arg8_]], [[CST_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_1_]]{{.}}, sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<256xbf16, strided<[1], offset: ?>> to memref<256xbf16>
// CHECK:             [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<256xbf16>
// CHECK:             bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_]] : (tensor<256xbf16>, memref<256xbf16, strided<[1], offset: ?>>) -> ()
// CHECK:             scf.yield [[VAR_1_]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }
