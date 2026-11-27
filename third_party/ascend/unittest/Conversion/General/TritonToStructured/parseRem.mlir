// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

tt.func public @kernel_with_rem_safe(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<0.000000e+00> : tensor<256xf32>
  %cst_0 = arith.constant dense<1024> : tensor<256xi32>
  %c256_i32 = arith.constant 256 : i32
  %cst_1 = arith.constant dense<64> : tensor<256xi32>
  %cst_2 = arith.constant dense<128> : tensor<256xi32>
  %0 = tt.get_program_id x : i32
  %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %2 = arith.remsi %1, %cst_2 : tensor<256xi32>
  %3 = arith.cmpi slt, %2, %cst_1 : tensor<256xi32>
  %4 = arith.muli %0, %c256_i32 : i32
  %5 = tt.splat %4 : i32 -> tensor<256xi32>
  %6 = arith.addi %5, %1 : tensor<256xi32>
  %7 = arith.cmpi slt, %6, %cst_0 : tensor<256xi32>
  %8 = arith.andi %3, %7 : tensor<256xi1>
  %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %10 = tt.addptr %9, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %11 = tt.load %10, %8, %cst : tensor<256x!tt.ptr<f32>>
  %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %13 = tt.addptr %12, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  tt.store %13, %11, %8 : tensor<256x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:   tt.func public @kernel_with_rem_safe(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : tensor<256xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<1024> : tensor<256xi32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<64> : tensor<256xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<128> : tensor<256xi32>
// CHECK:           %[[VAL_7:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK:           %[[VAL_9:.*]] = arith.remsi %[[VAL_8]], %[[VAL_6]] : tensor<256xi32>
// CHECK:           %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_5]] : tensor<256xi32>
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_7]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_11]] : i32 -> tensor<256xi32>
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_8]] : tensor<256xi32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_3]] : tensor<256xi32>
// CHECK:           %[[VAL_15:.*]] = arith.andi %[[VAL_10]], %[[VAL_14]] : tensor<256xi1>
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
// CHECK:           %[[VAL_17:.*]] = tt.addptr %[[VAL_16]], %[[VAL_9]] : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
// CHECK:           %[[VAL_18:.*]] = tt.load %[[VAL_17]], %[[VAL_15]], %[[VAL_2]] : tensor<256x!tt.ptr<f32>>
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
// CHECK:           %[[VAL_20:.*]] = tt.addptr %[[VAL_19]], %[[VAL_8]] : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
// CHECK:           tt.store %[[VAL_20]], %[[VAL_18]], %[[VAL_15]] : tensor<256x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }


// -----


tt.func public @test_remsi_with_broadcast(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<2x2xf32> {
  %c0_f32 = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
  %c4 = arith.constant dense<4> : tensor<2x2xi32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %2 = tt.broadcast %1 : tensor<1x2xi32> -> tensor<2x2xi32>
  %3 = arith.remsi %2, %c4 : tensor<2x2xi32>
  %4 = arith.trunci %3 : tensor<2x2xi32> to tensor<2x2xi1>
  %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
  %vals = tt.load %ptrs, %4, %c0_f32 : tensor<2x2x!tt.ptr<f32>>
  tt.return %vals : tensor<2x2xf32>
}

// CHECK-LABEL: tt.func public @test_remsi_with_broadcast(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<2x2xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<4> : tensor<2x2xi32>
// CHECK:           %[[VAL_3:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[VAL_4:.*]] = tt.expand_dims %[[VAL_3]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK:           %[[VAL_5:.*]] = tt.broadcast %[[VAL_4]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK:           %[[VAL_6:.*]] = arith.remsi %[[VAL_5]], %[[VAL_2]] : tensor<2x2xi32>
// CHECK:           %[[VAL_7:.*]] = arith.trunci %[[VAL_6]] : tensor<2x2xi32> to tensor<2x2xi1>
// CHECK:           %[[VAL_8:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
// CHECK:           %[[VAL_9:.*]] = tt.load %[[VAL_8]], %[[VAL_7]], %[[VAL_1]] : tensor<2x2x!tt.ptr<f32>>
// CHECK:           tt.return %[[VAL_9]] : tensor<2x2xf32>
// CHECK:         }
