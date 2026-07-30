// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : tensor<512x1xi32> -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : tensor<1x256xi32> -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<512x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<bf16>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>>
    %19 = tt.addptr %18, %3 : tensor<256x!tt.ptr<bf16>>, tensor<256xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<512x256xbf16>) -> tensor<256xbf16>
    tt.store %19, %5 : tensor<256x!tt.ptr<bf16>>
    tt.return
    }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [512, 256], strides: [256, 1] : memref<*xbf16> to memref<512x256xbf16, strided<[256, 1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [256], strides: [1] : memref<*xbf16> to memref<256xbf16, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<512x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<512x256xbf16, strided<[256, 1]>> to memref<512x256xbf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<512x256xbf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<256xbf16>
// CHECK:           [[VAR_2_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_1_]] : tensor<256xbf16>) -> tensor<256xbf16>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_0_]] : tensor<512x256xbf16>) outs([[VAR_2_]] : tensor<256xbf16>) dimensions = [0]
// CHECK:             ([[in_:%.+]]: bf16, [[init_:%.+]]: bf16) {
// CHECK:               [[VAR_3_:%.+]] = arith.addf [[in_]], [[init_]] : bf16
// CHECK:               linalg.yield [[VAR_3_]] : bf16
// CHECK:             }
// CHECK:           bufferization.materialize_in_destination [[VAR_reduced_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<256xbf16>, memref<256xbf16, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
