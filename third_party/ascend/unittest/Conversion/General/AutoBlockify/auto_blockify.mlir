// RUN: triton-opt --auto-blockify="auto-blockify-size=5" --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL:   tt.func @kernel(
// CHECK-SAME:                    %[[VAL_0:.*]]: !tt.ptr<f32>) attributes {auto_blockify_size = 5 : i32} {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : tensor<5x8xf32>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<8> : tensor<5xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0> : tensor<5xi32>
// CHECK:           %[[VAL_4:.*]] = tt.get_num_programs x : i32
// CHECK:           %[[VAL_5:.*]] = tt.get_num_programs y : i32
// CHECK:           %[[VAL_6:.*]] = tt.get_num_programs z : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_9:.*]] = tt.get_program_id x {logical_block_id} : i32
// CHECK:           %[[VAL_10:.*]] = tt.get_program_id y {logical_block_id} : i32
// CHECK:           %[[VAL_11:.*]] = tt.get_program_id z {logical_block_id} : i32
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_9]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_16:.*]] = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
// CHECK:           %[[VAL_17:.*]] = tt.splat %[[VAL_15]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]] : tensor<5xi32>
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_8]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_19]] : tensor<5xi32>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi sge, %[[VAL_18]], %[[VAL_3]] : tensor<5xi32>
// CHECK:           %[[VAL_22:.*]] = arith.ori %[[VAL_20]], %[[VAL_21]] : tensor<5xi1>
// CHECK:           %[[VAL_23:.*]] = tt.splat %[[VAL_7]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_24:.*]] = arith.divsi %[[VAL_18]], %[[VAL_23]] : tensor<5xi32>
// CHECK:           %[[VAL_25:.*]] = tt.splat %[[VAL_4]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_26:.*]] = arith.remsi %[[VAL_24]], %[[VAL_25]] : tensor<5xi32>
// CHECK:           %[[VAL_27:.*]] = arith.muli %[[VAL_26]], %[[VAL_2]] : tensor<5xi32>
// CHECK:           %[[VAL_28:.*]] = tt.expand_dims %[[VAL_27]] {axis = 1 : i32} : tensor<5xi32> -> tensor<5x1xi32>
// CHECK:           %[[VAL_29:.*]] = tt.broadcast %[[VAL_28]] : tensor<5x1xi32> -> tensor<5x8xi32>
// CHECK:           %[[VAL_30:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_31:.*]] = tt.expand_dims %[[VAL_30]] {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK:           %[[VAL_32:.*]] = tt.broadcast %[[VAL_31]] : tensor<1x8xi32> -> tensor<5x8xi32>
// CHECK:           %[[VAL_33:.*]] = arith.addi %[[VAL_29]], %[[VAL_32]] : tensor<5x8xi32>
// CHECK:           %[[VAL_34:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<5x8x!tt.ptr<f32>>
// CHECK:           %[[VAL_35:.*]] = tt.addptr %[[VAL_34]], %[[VAL_33]] : tensor<5x8x!tt.ptr<f32>>, tensor<5x8xi32>
// CHECK:           %[[VAL_36:.*]] = tt.expand_dims %[[VAL_22]] {axis = 1 : i32} : tensor<5xi1> -> tensor<5x1xi1>
// CHECK:           %[[VAL_37:.*]] = tt.broadcast %[[VAL_36]] : tensor<5x1xi1> -> tensor<5x8xi1>
// CHECK:           tt.store %[[VAL_35]], %[[VAL_1]], %[[VAL_37]] : tensor<5x8x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<8xi32>
  %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %4 = arith.addi %2, %3 : tensor<8xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  tt.store %6, %cst : tensor<8x!tt.ptr<f32>>
  tt.return
}

// -----

// CHECK-LABEL:   tt.func @kernel2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !tt.ptr<f32>) attributes {auto_blockify_size = 5 : i32} {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<8> : tensor<5xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
// CHECK:           %[[VAL_7:.*]] = tt.get_num_programs x : i32
// CHECK:           %[[VAL_8:.*]] = tt.get_num_programs y : i32
// CHECK:           %[[VAL_9:.*]] = tt.get_num_programs z : i32
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_12:.*]] = tt.get_program_id x {logical_block_id} : i32
// CHECK:           %[[VAL_13:.*]] = tt.get_program_id y {logical_block_id} : i32
// CHECK:           %[[VAL_14:.*]] = tt.get_program_id z {logical_block_id} : i32
// CHECK:           %[[VAL_15:.*]] = arith.muli %[[VAL_12]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_13]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_19:.*]] = tt.make_range {end = 5 : i32, start = 0 : i32} : tensor<5xi32>
// CHECK:           %[[VAL_20:.*]] = tt.splat %[[VAL_18]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_19]] : tensor<5xi32>
// CHECK:           %[[VAL_22:.*]] = tt.splat %[[VAL_10]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_23:.*]] = arith.divsi %[[VAL_21]], %[[VAL_22]] : tensor<5xi32>
// CHECK:           %[[VAL_24:.*]] = tt.splat %[[VAL_7]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_25:.*]] = arith.remsi %[[VAL_23]], %[[VAL_24]] : tensor<5xi32>
// CHECK:           %[[VAL_26:.*]] = tt.splat %[[VAL_9]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_27:.*]] = arith.divsi %[[VAL_21]], %[[VAL_26]] : tensor<5xi32>
// CHECK:           %[[VAL_28:.*]] = tt.splat %[[VAL_8]] : i32 -> tensor<5xi32>
// CHECK:           %[[VAL_29:.*]] = arith.remsi %[[VAL_27]], %[[VAL_28]] : tensor<5xi32>
// CHECK:           %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_29]], %[[VAL_1]] : tensor<5xi32>
// CHECK:           %[[VAL_31:.*]] = arith.muli %[[VAL_25]], %[[VAL_1]] : tensor<5xi32>
// CHECK:           %[[VAL_32:.*]] = tt.expand_dims %[[VAL_31]] {axis = 1 : i32} : tensor<5xi32> -> tensor<5x1xi32>
// CHECK:           %[[VAL_33:.*]] = tt.broadcast %[[VAL_32]] : tensor<5x1xi32> -> tensor<5x8xi32>
// CHECK:           %[[VAL_34:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_35:.*]] = tt.expand_dims %[[VAL_34]] {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK:           %[[VAL_36:.*]] = tt.broadcast %[[VAL_35]] : tensor<1x8xi32> -> tensor<5x8xi32>
// CHECK:           %[[VAL_37:.*]] = arith.addi %[[VAL_33]], %[[VAL_36]] : tensor<5x8xi32>
// CHECK:           %[[VAL_38:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<5x8x!tt.ptr<f32>>
// CHECK:           %[[VAL_39:.*]] = tt.addptr %[[VAL_38]], %[[VAL_37]] : tensor<5x8x!tt.ptr<f32>>, tensor<5x8xi32>
// CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_11]], %[[VAL_18]] : i32
// CHECK:           %[[VAL_41:.*]] = arith.maxsi %[[VAL_40]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_42:.*]] = arith.index_cast %[[VAL_41]] : i32 to index
// CHECK:           %[[VAL_43:.*]] = arith.minsi %[[VAL_42]], %[[VAL_3]] : index
// CHECK:           scf.for %[[VAL_44:.*]] = %[[VAL_5]] to %[[VAL_43]] step %[[VAL_4]] {
// CHECK:             %[[VAL_45:.*]] = tensor.extract %[[VAL_30]]{{\[}}%[[VAL_44]]] : tensor<5xi1>
// CHECK:             scf.if %[[VAL_45]] {
// CHECK:               %[[VAL_46:.*]] = tensor.extract_slice %[[VAL_39]]{{\[}}%[[VAL_44]], 0] [1, 8] [1, 1] : tensor<5x8x!tt.ptr<f32>> to tensor<8x!tt.ptr<f32>>
// CHECK:               tt.store %[[VAL_46]], %[[VAL_6]] : tensor<8x!tt.ptr<f32>>
// CHECK:             }
// CHECK:           } {auto_blockify_loop}
// CHECK:           tt.return
// CHECK:         }
tt.func @kernel2(%arg0: !tt.ptr<f32>) {
  %cst = arith.constant dense<0.000000e+00> : tensor<8xf32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %a = tt.get_program_id y : i32
  %b = arith.cmpi slt, %a, %c8_i32 : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.splat %1 : i32 -> tensor<8xi32>
  %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %4 = arith.addi %2, %3 : tensor<8xi32>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  scf.if %b {
    tt.store %6, %cst : tensor<8x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}
