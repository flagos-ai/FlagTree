// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @triton_per_fused_0d1d2d345678910111213(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<f32> , %arg4: !tt.ptr<f32> , %arg5: !tt.ptr<f32> , %arg6: !tt.ptr<f32> , %arg7: !tt.ptr<f32> , %arg8: !tt.ptr<f32> , %arg9: !tt.ptr<f32> , %arg10: !tt.ptr<f32> , %arg11: !tt.ptr<f32> , %arg12: i32 , %arg13: i32 ) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<1x4xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x4xf32>
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %cst_1 = arith.constant dense<4> : tensor<1x4xi32>
    %c0_i32 = arith.constant 0 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4096_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %4 = arith.cmpi slt, %3, %cst_1 : tensor<1x4xi32>
    %5 = arith.select %4, %cst_0, %cst : tensor<1x4xi1>, tensor<1x4xf32>
    %6 = tt.broadcast %5 : tensor<1x4xf32> -> tensor<4096x4xf32>
    %7 = tt.broadcast %3 : tensor<1x4xi32> -> tensor<4096x4xi32>
    %8:2 = "tt.reduce"(%6, %7) <{axis = 1 : i32}> ({
    ^bb0(%arg14: f32 , %arg15: i32 , %arg16: f32 , %arg17: i32 ):
      %12 = arith.cmpf ogt, %arg14, %arg16 : f32
      %13 = arith.cmpf oeq, %arg14, %arg16 : f32
      %14 = arith.cmpf une, %arg14, %arg14 : f32
      %15 = arith.cmpf une, %arg16, %arg16 : f32
      %16 = arith.xori %15, %true : i1
      %17 = arith.andi %14, %16 : i1
      %18 = arith.ori %12, %17 : i1
      %19 = arith.andi %14, %15 : i1
      %20 = arith.ori %13, %19 : i1
      %21 = arith.cmpi slt, %arg15, %arg17 : i32
      %22 = arith.andi %20, %21 : i1
      %23 = arith.ori %18, %22 : i1
      %24 = arith.select %23, %arg14, %arg16 : f32
      %25 = arith.select %23, %arg15, %arg17 : i32
      tt.reduce.return %24, %25 : f32, i32
    }) : (tensor<4096x4xf32>, tensor<4096x4xi32>) -> (tensor<4096xf32>, tensor<4096xi32>)
    %9 = tt.expand_dims %8#0 {axis = 1 : i32} : tensor<4096xf32> -> tensor<4096x1xf32>
    %10 = tt.make_tensor_ptr %arg5, [%c4096_i64, %c4_i64], [%c4_i64, %c1_i64], [%1, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4096x4xf32>>
    %11 = tt.broadcast %9 : tensor<4096x1xf32> -> tensor<4096x4xf32>
    tt.store %10, %11 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<4096x4xf32>>
    tt.return
  }
}


// CHECK:   %[[VAL_89:.*]] = linalg.reduce ins(%[[VAL_6:.*]], %[[VAL_8:.*]] : tensor<4096x4xf32>, tensor<4096x4xi32>) outs(%[[VAL_9:.*]], %[[VAL_10:.*]] : tensor<4096xf32>, tensor<4096xi32>) dimensions = [1]
// CHECK:    (%[[VAL_IN:.*]]: f32, %[[VAL_150:.*]]: i32, %[[VAL_INIT:.*]]: f32,  %[[VAL_151:.*]]: i32) {
// CHECK:      %[[VAL_264:.*]] = arith.cmpf ogt, %[[VAL_IN]], %[[VAL_INIT]] : f32
// CHECK:      %[[VAL_265:.*]] = arith.cmpf oeq, %[[VAL_IN]], %[[VAL_INIT]] : f32
// CHECK:      %[[VAL_266:.*]] = arith.cmpf une, %[[VAL_IN]],  %[[VAL_IN]]  : f32
// CHECK:      %[[VAL_267:.*]] = arith.cmpf une, %[[VAL_INIT]], %[[VAL_INIT]]  : f32
// CHECK:      %[[VAL_268:.*]] = arith.xori %[[VAL_267]], %true  : i1
// CHECK:      %[[VAL_269:.*]] = arith.andi %[[VAL_266]], %[[VAL_268]] : i1
// CHECK:      %[[VAL_270:.*]] = arith.ori %[[VAL_264]], %[[VAL_269]]  : i1
// CHECK:      %[[VAL_271:.*]] = arith.andi %[[VAL_266]], %[[VAL_267]] : i1
// CHECK:      %[[VAL_272:.*]] = arith.ori %[[VAL_265]], %[[VAL_271]]  : i1
// CHECK:      %[[VAL_273:.*]] = arith.cmpi slt, %[[VAL_150]], %[[VAL_151]] : i32
// CHECK:      %[[VAL_274:.*]] = arith.andi  %[[VAL_272]], %[[VAL_273]] : i1
// CHECK:      %[[VAL_275:.*]] = arith.ori %[[VAL_270]], %[[VAL_274]]  : i1
// CHECK:      %[[VAL_276:.*]] = arith.select %[[VAL_275]], %[[VAL_IN]], %[[VAL_INIT]] : f32
// CHECK:      %[[VAL_277:.*]] = arith.select %[[VAL_275]], %[[VAL_150]], %[[VAL_151]] : i32
// CHECK:       linalg.yield %[[VAL_276]], %[[VAL_277]] : f32, i32
// CHECK:     }

// -----

module {
  tt.func public @triton_test_fn_min_with_index_inner_scalar_0d1d2d345(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<i32> , %arg4: i32 , %arg5: i32 ) attributes {noinline = false} {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2_i32 : i32
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %3 = tt.splat %1 : i32 -> tensor<2xi32>
    %4 = arith.addi %3, %2 : tensor<2xi32>
    %5 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<2x!tt.ptr<f16>>
    %6 = tt.addptr %5, %4 : tensor<2x!tt.ptr<f16>>, tensor<2xi32>
    %7 = tt.load %6 : tensor<2x!tt.ptr<f16>>
    %8 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<2x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
    %10 = tt.load %9 : tensor<2x!tt.ptr<i32>>
    %11:2 = "tt.reduce"(%7, %10) <{axis = 0 : i32}> ({
    ^bb0(%arg6: f16 , %arg7: i32 , %arg8: f16 , %arg9: i32 ):
      %12 = arith.cmpf olt, %arg6, %arg8 : f16
      %13 = arith.cmpf oeq, %arg6, %arg8 : f16
      %14 = arith.cmpf une, %arg6, %arg6 : f16
      %15 = arith.cmpf une, %arg8, %arg8 : f16
      %16 = arith.xori %15, %true : i1
      %17 = arith.andi %14, %16 : i1
      %18 = arith.ori %12, %17 : i1
      %19 = arith.andi %14, %15 : i1
      %20 = arith.ori %13, %19 : i1
      %21 = arith.cmpi slt, %arg7, %arg9 : i32
      %22 = arith.andi %20, %21 : i1
      %23 = arith.ori %18, %22 : i1
      %24 = arith.select %23, %arg6, %arg8 : f16
      %25 = arith.select %23, %arg7, %arg9 : i32
      tt.reduce.return %24, %25 : f16, i32
    }) : (tensor<2xf16>, tensor<2xi32>) -> (f16, i32)
    tt.store %arg0, %11#0 : !tt.ptr<f16>
    tt.store %arg1, %11#1 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:   %[[VAL_89:.*]] = linalg.reduce ins(%[[VAL_2:.*]], %[[VAL_4:.*]] : tensor<2xf16>, tensor<2xi32>) outs(%[[VAL_5:.*]], %[[VAL_6:.*]] : tensor<f16>, tensor<i32>) dimensions = [0]
// CHECK:    (%[[VAL_IN:.*]]: f16, %[[VAL_150:.*]]: i32, %[[VAL_INIT:.*]]: f16,  %[[VAL_151:.*]]: i32) {
// CHECK:      %[[VAL_264:.*]] = arith.cmpf olt, %[[VAL_IN]], %[[VAL_INIT]] : f16
// CHECK:      %[[VAL_265:.*]] = arith.cmpf oeq, %[[VAL_IN]], %[[VAL_INIT]] : f16
// CHECK:      %[[VAL_266:.*]] = arith.cmpf une, %[[VAL_IN]],  %[[VAL_IN]]  : f16
// CHECK:      %[[VAL_267:.*]] = arith.cmpf une, %[[VAL_INIT]], %[[VAL_INIT]]  : f16
// CHECK:      %[[VAL_268:.*]] = arith.xori %[[VAL_267]], %true  : i1
// CHECK:      %[[VAL_269:.*]] = arith.andi %[[VAL_266]], %[[VAL_268]] : i1
// CHECK:      %[[VAL_270:.*]] = arith.ori %[[VAL_264]], %[[VAL_269]]  : i1
// CHECK:      %[[VAL_271:.*]] = arith.andi %[[VAL_266]], %[[VAL_267]] : i1
// CHECK:      %[[VAL_272:.*]] = arith.ori %[[VAL_265]], %[[VAL_271]]  : i1
// CHECK:      %[[VAL_273:.*]] = arith.cmpi slt, %[[VAL_150]], %[[VAL_151]] : i32
// CHECK:      %[[VAL_274:.*]] = arith.andi  %[[VAL_272]], %[[VAL_273]] : i1
// CHECK:      %[[VAL_275:.*]] = arith.ori %[[VAL_270]], %[[VAL_274]]  : i1
// CHECK:      %[[VAL_276:.*]] = arith.select %[[VAL_275]], %[[VAL_IN]], %[[VAL_INIT]] : f16
// CHECK:      %[[VAL_277:.*]] = arith.select %[[VAL_275]], %[[VAL_150]], %[[VAL_151]] : i32
// CHECK:       linalg.yield %[[VAL_276]], %[[VAL_277]] : f16, i32
// CHECK:     }
