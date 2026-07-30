// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>,
        %out_base: !tt.ptr<bf16>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : tensor<32x1xi32> -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : tensor<1x256xi32> -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : tensor<1x16xi32> -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : tensor<256x16xi32> -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : tensor<1x256x16xi32> -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 : tensor<32x256x16x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 1 : i32} : (tensor<32x256x16xbf16>) -> tensor<32x16xbf16>
    // out pointer, intentionally splat the base pointer for brevity
    %out = tt.splat %out_base : !tt.ptr<bf16> -> tensor<32x16x!tt.ptr<bf16>>
    tt.store %out, %5 : tensor<32x16x!tt.ptr<bf16>>
    tt.return
    }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: memref<*xbf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_empty_offsets_:%.+]] = tensor.empty() : tensor<32x16xi32>
// CHECK-DAG:       [[VAR_zero_offsets_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_]] : tensor<32x16xi32>) -> tensor<32x16xi32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [32, 256, 16], strides: [256, 1, 1] : memref<*xbf16> to memref<32x256x16xbf16, strided<[256, 1, 1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<32x256x16xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<32x256x16xbf16, strided<[256, 1, 1]>> to memref<32x256x16xbf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<32x256x16xbf16>
// CHECK:           [[VAL_7:%.+]] = tensor.empty() : tensor<256x32x16xbf16>
// CHECK:           [[VAL_8:%.+]] = linalg.transpose ins([[VAR_0_]] : tensor<32x256x16xbf16>) outs([[VAL_7]] : tensor<256x32x16xbf16>) permutation = [1, 0, 2]
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<32x16xbf16>
// CHECK:           [[VAR_2_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_1_]] : tensor<32x16xbf16>) -> tensor<32x16xbf16>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAL_8]] : tensor<256x32x16xbf16>) outs([[VAR_2_]] : tensor<32x16xbf16>) dimensions = [0]
// CHECK:             ([[in_:.+]]: bf16, [[init_:.+]]: bf16) {
// CHECK:               [[VAR_3_:%.+]] = arith.addf [[in_]], [[init_]] : bf16
// CHECK:               linalg.yield [[VAR_3_]] : bf16
// CHECK:             }
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[PARAM_2_]] : memref<*xbf16> to memref<?xbf16>
// CHECK:           linalg.generic {indexing_maps = [[[MAP_0_]], [[MAP_0_]]], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_reduced_]] : tensor<32x16xi32>, tensor<32x16xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: bf16):
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[IN_0_]] : i32 to index
// CHECK:             memref.store [[IN_1_]], [[VAR_cast_]]{{.}}[[VAR_5_]]{{.}} : memref<?xbf16>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
