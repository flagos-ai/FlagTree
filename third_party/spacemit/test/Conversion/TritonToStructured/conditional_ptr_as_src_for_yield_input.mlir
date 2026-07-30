// RUN: triton-shared-opt --triton-to-structured  --canonicalize  %s | FileCheck %s

module attributes {} {
  tt.func public @gather_kernel(%arg0: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> { tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %2 = arith.muli %0, %arg6 : i32
    %3 = tt.addptr %arg1, %2 : !tt.ptr<i64>, i32
    %4 = tt.splat %3 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %5 = tt.addptr %4, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    %6 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %7 = arith.cmpi slt, %1, %6 : tensor<512xi32>
    %8 = tt.load %5, %7 : tensor<512x!tt.ptr<i64>>
    %9 = arith.cmpi eq, %arg4, %c0_i32 : i32
    %10 = scf.if %9 -> (tensor<512x!tt.ptr<i64>>) {
      %15 = arith.extsi %arg5 : i32 to i64
      %16 = tt.splat %15 : i64 -> tensor<512xi64>
      %17 = arith.muli %8, %16 : tensor<512xi64>
      %18 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %19 = tt.addptr %18, %17 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      %20 = tt.addptr %19, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
      scf.yield %20 : tensor<512x!tt.ptr<i64>>
    } else {
      %15 = arith.muli %0, %arg5 : i32
      %16 = tt.addptr %arg0, %15 : !tt.ptr<i64>, i32
      %17 = tt.splat %16 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
      %18 = tt.addptr %17, %8 : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
      scf.yield %18 : tensor<512x!tt.ptr<i64>>
    }
    %11 = tt.load %10, %7 : tensor<512x!tt.ptr<i64>>
    %12 = tt.addptr %arg2, %2 : !tt.ptr<i64>, i32
    %13 = tt.splat %12 : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
    %14 = tt.addptr %13, %1 : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
    tt.store %14, %11, %7 : tensor<512x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @gather_kernel(
// CHECK-SAME:                                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32},
// CHECK-SAME:                                  %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 512 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_10:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_11:.*]] = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
// CHECK:           %[[VAL_12:.*]] = arith.muli %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : i32 to index
// CHECK:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_12]] : i32 to index
// CHECK:           %[[VAL_15:.*]] = tts.make_tptr %[[VAL_1]] to sizes: [512], strides: [1], offsets: {{\[}}%[[VAL_14]]], shape: [0], order: [] : <i64> to tensor<512x!tt.ptr<i64>>
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<512xi32>
// CHECK:           %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_16]] : tensor<512xi32>
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = arith.minsi %[[VAL_18]], %[[VAL_8]] : index
// CHECK:           %[[VAL_20:.*]] = arith.maxsi %[[VAL_19]], %[[VAL_7]] : index
// CHECK:           %[[VAL_21:.*]] = "tts.load"(%[[VAL_15]], %[[VAL_20]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<512x!tt.ptr<i64>>, index) -> tensor<512xi64>
// CHECK:           %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_23:.*]] = scf.if %[[VAL_22]] -> (tensor<512x!tt.ptr<i64>>) {
// CHECK:             %[[VAL_24:.*]] = arith.extsi %[[VAL_5]] : i32 to i64
// CHECK:             %[[VAL_25:.*]] = tt.splat %[[VAL_24]] : i64 -> tensor<512xi64>
// CHECK:             %[[VAL_26:.*]] = arith.muli %[[VAL_21]], %[[VAL_25]] : tensor<512xi64>
// CHECK:             %[[VAL_27:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
// CHECK:             %[[VAL_28:.*]] = tt.addptr %[[VAL_27]], %[[VAL_26]] : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
// CHECK:             %[[VAL_29:.*]] = tt.addptr %[[VAL_28]], %[[VAL_11]] : tensor<512x!tt.ptr<i64>>, tensor<512xi32>
// CHECK:             scf.yield %[[VAL_29]] : tensor<512x!tt.ptr<i64>>
// CHECK:           } else {
// CHECK:             %[[VAL_30:.*]] = arith.muli %[[VAL_10]], %[[VAL_5]] : i32
// CHECK:             %[[VAL_31:.*]] = tt.addptr %[[VAL_0]], %[[VAL_30]] : !tt.ptr<i64>, i32
// CHECK:             %[[VAL_32:.*]] = tt.splat %[[VAL_31]] : !tt.ptr<i64> -> tensor<512x!tt.ptr<i64>>
// CHECK:             %[[VAL_33:.*]] = tt.addptr %[[VAL_32]], %[[VAL_21]] : tensor<512x!tt.ptr<i64>>, tensor<512xi64>
// CHECK:             scf.yield %[[VAL_33]] : tensor<512x!tt.ptr<i64>>
// CHECK:           }
// CHECK:           %[[VAL_34:.*]] = tt.load %[[VAL_23]], %[[VAL_17]] : tensor<512x!tt.ptr<i64>>
// CHECK:           %[[VAL_35:.*]] = tts.make_tptr %[[VAL_2]] to sizes: [512], strides: [1], offsets: {{\[}}%[[VAL_13]]], shape: [0], order: [] : <i64> to tensor<512x!tt.ptr<i64>>
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_3]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = arith.minsi %[[VAL_36]], %[[VAL_8]] : index
// CHECK:           %[[VAL_38:.*]] = arith.maxsi %[[VAL_37]], %[[VAL_7]] : index
// CHECK:           "tts.store"(%[[VAL_35]], %[[VAL_34]], %[[VAL_38]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<512x!tt.ptr<i64>>, tensor<512xi64>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
