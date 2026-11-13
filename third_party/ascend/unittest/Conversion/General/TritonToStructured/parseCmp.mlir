// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

tt.func public @test_cmp_ult(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %cst_0 = arith.constant dense<512> : tensor<1024xi32>
  %c1024_i32 = arith.constant 1024 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c1024_i32 {tt.divisibility = dense<512> : tensor<1xi32>} : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %3 = tt.splat %1 : i32 -> tensor<1024xi32>
  %4 = arith.addi %3, %2 : tensor<1024xi32>
  %5 = arith.divsi %4, %cst_0 : tensor<1024xi32>
  %6 = arith.cmpi ult, %5, %cst_0 : tensor<1024xi32>
  %7 = arith.muli %5, %cst_0 : tensor<1024xi32>
  %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %10 = tt.load %9, %6, %cst : tensor<1024x!tt.ptr<f32>>
  %11 = arith.muli %0, %c1024_i32 : i32
  %12 = tt.splat %11 : i32 -> tensor<1024xi32>
  %13 = arith.addi %12, %2 : tensor<1024xi32>
  %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  tt.store %15, %10 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:   tt.func public @test_cmp_ult(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1024> : tensor<1xi64>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<512> : tensor<2xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1024 : i32
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_7]], %[[VAL_6]] {tt.divisibility = dense<512> : tensor<1xi32>} : i32
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_9]], %[[VAL_5]] : index
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_5]] : index
// CHECK:           %[[VAL_13:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[VAL_14:.*]] = arith.muli %[[VAL_13]], %[[VAL_4]] : tensor<2xi32>
// CHECK:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_12]] : index to i32
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_15]] : i32 -> tensor<2xi32>
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_16]] : tensor<2xi32>
// CHECK:           %[[VAL_18:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
// CHECK:           %[[VAL_19:.*]] = tt.addptr %[[VAL_18]], %[[VAL_17]] : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
// CHECK:           %[[VAL_20:.*]] = arith.index_cast %[[VAL_11]] : index to i32
// CHECK:           %[[VAL_21:.*]] = tt.splat %[[VAL_20]] : i32 -> tensor<2xi32>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_13]], %[[VAL_21]] : tensor<2xi32>
// CHECK:           %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_4]] : tensor<2xi32>
// CHECK:           %[[VAL_24:.*]] = tt.load %[[VAL_19]], %[[VAL_23]], %[[VAL_3]] : tensor<2x!tt.ptr<f32>>
// CHECK:           %[[VAL_25:.*]] = tensor.empty() : tensor<2x512xf32>
// CHECK:           %[[VAL_26:.*]] = linalg.broadcast ins(%[[VAL_24]] : tensor<2xf32>) outs(%[[VAL_25]] : tensor<2x512xf32>) dimensions = [1]
// CHECK:           %[[VAL_27:.*]] = tensor.reshape %[[VAL_26]](%[[VAL_2]]) : (tensor<2x512xf32>, tensor<1xi64>) -> tensor<1024xf32>
// CHECK:           %[[VAL_28:.*]] = arith.muli %[[VAL_7]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_29:.*]] = tt.splat %[[VAL_28]] : i32 -> tensor<1024xi32>
// CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_10]] : tensor<1024xi32>
// CHECK:           %[[VAL_31:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_32:.*]] = tt.addptr %[[VAL_31]], %[[VAL_30]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:           tt.store %[[VAL_32]], %[[VAL_27]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }


// -----


tt.func public @test_cmp_uge(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32>
  %cst_0 = arith.constant dense<511> : tensor<1024xi32>
  %cst_1 = arith.constant dense<512> : tensor<1024xi32>
  %c1024_i32 = arith.constant 1024 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c1024_i32 {tt.divisibility = dense<512> : tensor<1xi32>} : i32
  %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %3 = tt.splat %1 : i32 -> tensor<1024xi32>
  %4 = arith.addi %3, %2 : tensor<1024xi32>
  %5 = arith.divsi %4, %cst_1 : tensor<1024xi32>
  %6 = arith.cmpi uge, %cst_0, %5 : tensor<1024xi32>
  %7 = arith.muli %5, %cst_1 : tensor<1024xi32>
  %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  %10 = tt.load %9, %6, %cst : tensor<1024x!tt.ptr<f32>>
  %11 = arith.muli %0, %c1024_i32 : i32
  %12 = tt.splat %11 : i32 -> tensor<1024xi32>
  %13 = arith.addi %12, %2 : tensor<1024xi32>
  %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
  %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
  tt.store %15, %10 : tensor<1024x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:   tt.func public @test_cmp_uge(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : tensor<1024xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<511> : tensor<1024xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<512> : tensor<1024xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1024 : i32
// CHECK:           %[[VAL_6:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_5]] {tt.divisibility = dense<512> : tensor<1xi32>} : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
// CHECK:           %[[VAL_9:.*]] = tt.splat %[[VAL_7]] : i32 -> tensor<1024xi32>
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_8]] : tensor<1024xi32>
// CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_4]] : tensor<1024xi32>
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ule, %[[VAL_11]], %[[VAL_3]] : tensor<1024xi32>
// CHECK:           %[[VAL_13:.*]] = arith.muli %[[VAL_11]], %[[VAL_4]] : tensor<1024xi32>
// CHECK:           %[[VAL_14:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_15:.*]] = tt.addptr %[[VAL_14]], %[[VAL_13]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:           %[[VAL_16:.*]] = tt.load %[[VAL_15]], %[[VAL_12]], %[[VAL_2]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_6]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_18:.*]] = tt.splat %[[VAL_17]] : i32 -> tensor<1024xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_8]] : tensor<1024xi32>
// CHECK:           %[[VAL_20:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[VAL_21:.*]] = tt.addptr %[[VAL_20]], %[[VAL_19]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK:           tt.store %[[VAL_21]], %[[VAL_16]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
