// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @triton_splat_as_mask
  tt.func public @triton_splat_as_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %cst_0 = arith.constant dense<8> : tensor<1x8xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %rmask = arith.cmpi slt, %2, %cst_0 : tensor<1x8xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %5 = arith.cmpi slt, %arg2, %c8_i32 : i32
      %6 = tt.addptr %3, %2 : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %xmask = tt.splat %5 : i1 -> tensor<1x8xi1>
      %mask = arith.andi %rmask, %xmask : tensor<1x8xi1>
      // CHECK-DAG: %[[input_slice:.*]] = memref.subview
      // CHECK-DAG: %[[buffer_slice:.*]] = memref.subview
      // CHECK: memref.copy %[[input_slice]], %[[buffer_slice]]
      %7 = tt.load %6, %mask, %cst : tensor<1x8x!tt.ptr<f32>>
      %8 = tt.addptr %4, %2 : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      // CHECK-DAG: %[[buffer_tensor_slice:.*]] = tensor.extract_slice
      // CHECK-DAG: %[[output_slice:.*]] = memref.subview
      // CHECK: bufferization.materialize_in_destination %[[buffer_tensor_slice]] in writable %[[output_slice]]
      tt.store %8, %7, %mask : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @triton_divide_as_mask
  tt.func public @triton_divide_as_mask(%arg0: !tt.ptr<f32>,
                                        %arg1: !tt.ptr<f32>,
                                        %arg2: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x8xf32>
    %cst_0 = arith.constant dense<8> : tensor<1x8xi32>
    %cst_1 = arith.constant dense<1> : tensor<1x8xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %3 = tt.splat %arg2 : i32 -> tensor<1x8xi32>
    %4 = arith.divsi %2, %3 : tensor<1x8xi32>
    %rmask = arith.cmpi slt, %4, %cst_1 : tensor<1x8xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
    scf.for %arg3 = %c0_i32 to %c8_i32 step %c1_i32  : i32 {
      %7 = arith.cmpi slt, %arg3, %c8_i32 : i32
      %8 = tt.addptr %5, %2 : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      %xmask = tt.splat %7 : i1 -> tensor<1x8xi1>
      %mask = arith.andi %rmask, %xmask : tensor<1x8xi1>
      // CHECK-DAG: %[[input_slice:.*]] = memref.subview
      // CHECK-DAG: %[[buffer_slice:.*]] = memref.subview
      // CHECK: memref.copy %[[input_slice]], %[[buffer_slice]]
      %9 = tt.load %8, %mask, %cst : tensor<1x8x!tt.ptr<f32>>
      %10 = tt.addptr %6, %2 : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi32>
      // CHECK-DAG: %[[buffer_tensor_slice:.*]] = tensor.extract_slice
      // CHECK-DAG: %[[output_slice:.*]] = memref.subview
      // CHECK: bufferization.materialize_in_destination %[[buffer_tensor_slice]] in writable %[[output_slice]]
      tt.store %10, %9, %mask : tensor<1x8x!tt.ptr<f32>>
    }
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: func.func @triton_bool_splat_condition_select
  tt.func public @triton_bool_splat_condition_select(%arg0: !tt.ptr<f32>,
                                                     %arg1: !tt.ptr<f32>,
                                                     %arg2: !tt.ptr<f32>,
                                                     %arg3: i32)
                                                     attributes {noinline = false} {
    // CHECK: %[[baseline:.*]] = arith.constant 0
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %6 = tt.load %5 : tensor<32x!tt.ptr<f32>>
    // CHECK：%[[VAL_2:.*]] = arith.cmpi eq, [[ARG_3:%.+]], %[[baseline]] : i32
    // CHECK：%[[VAL_3:.*]] = tensor.empty() : tensor<32xi1>
    // CHECK：%[[VAL_4:.*]] = linalg.fill ins([[VAL_2]] : i1) outs(%[[VAL_3]] : tensor<32xi1>) -> tensor<32xi1>
    // CHECK：%5 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_4]], %0, %1 : tensor<32xi1>, tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) {
    // CHECK：^bb0(%in: i1, %in_3: f32, %in_4: f32, %out: f32):
    // CHECK：%[[VAL_6:.*]] = arith.select %in, %in_3, %in_4 : f32
    // CHECK：linalg.yield %[[VAL_6]] : f32
    // CHECK：} -> tensor<32xf32>
    %7 = arith.cmpi eq, %arg3, %c0_i32 : i32
    %8 = tt.splat %7 : i1 -> tensor<32xi1>
    %9 = arith.select %8, %3, %6 : tensor<32xi1>, tensor<32xf32>
    %10 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %11 = tt.addptr %10, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %11, %9 : tensor<32x!tt.ptr<f32>>
    tt.return
}
}
