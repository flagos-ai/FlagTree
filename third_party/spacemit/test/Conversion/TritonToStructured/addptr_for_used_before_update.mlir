// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

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
    // Example 2, gep operand is another gep's output, which is passed into the loop as varible, used before update
    %_ptr2 = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr = %2) -> (tensor<256x!tt.ptr<bf16>>) {
        // perform load
        %3 = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x!tt.ptr<bf16>>
        tt.store %ptr, %3 : tensor<256x!tt.ptr<bf16>>
        // pointer updates
        %4 = tt.splat %i_c3 : i32 -> tensor<256xi32>
        %ptr_iter = tt.addptr %ptr, %4 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
        scf.yield %ptr_iter : tensor<256x!tt.ptr<bf16>>
    }
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

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg2_:%.+]] = [[CST_1024_]]) -> (index) {
// CHECK-DAG:         [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256], strides: {{.}}[[CST_1_]]{{.}}, offsets: {{.}}[[VAR_arg2_]]{{.}}, shape: [0], order: [] : <bf16> to tensor<256x!tt.ptr<bf16>>
// CHECK:             [[VAR_2_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16>>) -> tensor<256xbf16>
// CHECK:             "tts.store"([[VAR_1_]], [[VAR_2_]]) <{static_mask_dims = array<i64>}> : (tensor<256x!tt.ptr<bf16>>, tensor<256xbf16>) -> ()
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_3_]] : index
// CHECK:             scf.yield [[VAR_3_]] : index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
