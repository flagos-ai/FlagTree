// RUN: triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-a5=True force_simt_template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-a5=True force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=True' --split-input-file %s | FileCheck %s

module {
		tt.func public @tt_softmax_3d(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_0 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c4_i32 : i32
		%5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%6 = tt.splat %1 : i32 -> tensor<2xi32>
		%7 = arith.addi %5, %6 : tensor<2xi32>
		%8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%9 = tt.splat %4 : i32 -> tensor<4xi32>
		%10 = arith.addi %8, %9 : tensor<4xi32>
		%11 = tt.expand_dims %7 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
		%14 = arith.muli %13, %cst : tensor<2x1x1xi32>
		%15 = arith.muli %2, %c4_i32 : i32
		%16 = tt.splat %15 : i32 -> tensor<2x1x1xi32>
		%17 = arith.addi %14, %16 : tensor<2x1x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%20 = tt.broadcast %17 : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
		%21 = tt.broadcast %19 : tensor<1x1x4xi32> -> tensor<2x1x4xi32>
		%22 = arith.addi %20, %21 : tensor<2x1x4xi32>
		%23 = tt.splat %arg0 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		%25 = tt.load %24 : tensor<2x1x4x!tt.ptr>
		%26 = tt.fp_to_fp %25 : tensor<2x1x4xf8E4M3FN> -> tensor<2x1x4xf32>
		%27 = "tt.reduce"(%26) <{axis = 0 : i32}> ({
		^bb0(%arg2: f32, %arg3: f32):
		%39 = arith.maxnumf %arg2, %arg3 : f32
		tt.reduce.return %39 : f32
		}) : (tensor<2x1x4xf32>) -> tensor<1x4xf32>
		%28 = tt.expand_dims %27 {axis = 0 : i32} : tensor<1x4xf32> -> tensor<1x1x4xf32>
		%29 = tt.broadcast %28 : tensor<1x1x4xf32> -> tensor<2x1x4xf32>
		%30 = arith.subf %26, %29 : tensor<2x1x4xf32>
		%31 = math.exp %30 : tensor<2x1x4xf32>
		%32 = "tt.reduce"(%31) <{axis = 0 : i32}> ({
		^bb0(%arg2: f32, %arg3: f32):
		%39 = arith.addf %arg2, %arg3 : f32
		tt.reduce.return %39 : f32
		}) : (tensor<2x1x4xf32>) -> tensor<1x4xf32>
		%33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<1x4xf32> -> tensor<1x1x4xf32>
		%34 = tt.broadcast %33 : tensor<1x1x4xf32> -> tensor<2x1x4xf32>
		%35 = arith.divf %31, %34 : tensor<2x1x4xf32>
		%36 = tt.fp_to_fp %35, rounding = rtne : tensor<2x1x4xf32> -> tensor<2x1x4xf8E4M3FN>
		%37 = tt.splat %arg1 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%38 = tt.addptr %37, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		tt.store %38, %36 : tensor<2x1x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x1x4xf8E4M3FN>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x1x4xf8E4M3FN>

// CHECK: %[[FP_TO_F32:[A-Za-z0-9_]+]] = tt.fp_to_fp %[[TENSOR_INPUT]] : tensor<2x1x4xf8E4M3FN> -> tensor<2x1x4xf32>

// CHECK: %[[EMPTY_MAX:[A-Za-z0-9_]+]] = tensor.empty() : tensor<1x4xf32>
// CHECK: %[[FILL_NINF:[A-Za-z0-9_]+]] = linalg.fill ins(%cst_0 : f32) outs(%[[EMPTY_MAX]] : tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[MAX_REDUCED:[A-Za-z0-9_]+]] = linalg.reduce ins(%[[FP_TO_F32]] : tensor<2x1x4xf32>) outs(%[[FILL_NINF]] : tensor<1x4xf32>) dimensions = [0]
// CHECK-NEXT: (%in: f32, %init: f32) {
// CHECK-NEXT: %[[MAX_NUM:[A-Za-z0-9_]+]] = arith.maxnumf %in, %init : f32
// CHECK-NEXT: linalg.yield %[[MAX_NUM]] : f32
// CHECK-NEXT: }

// CHECK: %[[EMPTY_BROADCAST:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x1x4xf32>
// CHECK: %[[MAX_BROADCASTED:[A-Za-z0-9_]+]] = linalg.broadcast ins(%[[MAX_REDUCED]] : tensor<1x4xf32>) outs(%[[EMPTY_BROADCAST]] : tensor<2x1x4xf32>) dimensions = [0]

// CHECK: %[[SUB_MAX:[A-Za-z0-9_]+]] = arith.subf %[[FP_TO_F32]], %[[MAX_BROADCASTED]] : tensor<2x1x4xf32>
// CHECK: %[[EXP_RESULT:[A-Za-z0-9_]+]] = math.exp %[[SUB_MAX]] : tensor<2x1x4xf32>

// CHECK: %[[FILL_ZERO:[A-Za-z0-9_]+]] = linalg.fill ins(%cst : f32) outs(%[[EMPTY_MAX]] : tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[SUM_REDUCED:[A-Za-z0-9_]+]] = linalg.reduce ins(%[[EXP_RESULT]] : tensor<2x1x4xf32>) outs(%[[FILL_ZERO]] : tensor<1x4xf32>) dimensions = [0]
// CHECK-NEXT: (%in: f32, %init: f32) {
// CHECK-NEXT: %[[ADD_SUM:[A-Za-z0-9_]+]] = arith.addf %in, %init : f32
// CHECK-NEXT: linalg.yield %[[ADD_SUM]] : f32
// CHECK-NEXT: }

// CHECK: %[[SUM_BROADCASTED:[A-Za-z0-9_]+]] = linalg.broadcast ins(%[[SUM_REDUCED]] : tensor<1x4xf32>) outs(%[[EMPTY_BROADCAST]] : tensor<2x1x4xf32>) dimensions = [0]

// CHECK: %[[SOFTMAX_F32:[A-Za-z0-9_]+]] = arith.divf %[[EXP_RESULT]], %[[SUM_BROADCASTED]] : tensor<2x1x4xf32>

// CHECK: %[[EMPTY_F8:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x1x4xf8E4M3FN>
// CHECK: %[[SOFTMAX_F8:[A-Za-z0-9_]+]] = hfusion.cast {mode = #hfusion.round_mode} ins(%[[SOFTMAX_F32]] : tensor<2x1x4xf32>) outs(%[[EMPTY_F8]] : tensor<2x1x4xf8E4M3FN>) -> tensor<2x1x4xf8E4M3FN>

// -----

module {
		tt.func public @tt_softmax_3d(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_0 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c4_i32 : i32
		%5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%6 = tt.splat %1 : i32 -> tensor<2xi32>
		%7 = arith.addi %5, %6 : tensor<2xi32>
		%8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%9 = tt.splat %4 : i32 -> tensor<4xi32>
		%10 = arith.addi %8, %9 : tensor<4xi32>
		%11 = tt.expand_dims %7 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
		%14 = arith.muli %13, %cst : tensor<2x1x1xi32>
		%15 = arith.muli %2, %c4_i32 : i32
		%16 = tt.splat %15 : i32 -> tensor<2x1x1xi32>
		%17 = arith.addi %14, %16 : tensor<2x1x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%20 = tt.broadcast %17 : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
		%21 = tt.broadcast %19 : tensor<1x1x4xi32> -> tensor<2x1x4xi32>
		%22 = arith.addi %20, %21 : tensor<2x1x4xi32>
		%23 = tt.splat %arg0 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		%25 = tt.load %24 : tensor<2x1x4x!tt.ptr>
		%26 = tt.fp_to_fp %25 : tensor<2x1x4xf8E5M2> -> tensor<2x1x4xf32>
		%27 = "tt.reduce"(%26) <{axis = 0 : i32}> ({
		^bb0(%arg2: f32, %arg3: f32):
		%39 = arith.maxnumf %arg2, %arg3 : f32
		tt.reduce.return %39 : f32
		}) : (tensor<2x1x4xf32>) -> tensor<1x4xf32>
		%28 = tt.expand_dims %27 {axis = 0 : i32} : tensor<1x4xf32> -> tensor<1x1x4xf32>
		%29 = tt.broadcast %28 : tensor<1x1x4xf32> -> tensor<2x1x4xf32>
		%30 = arith.subf %26, %29 : tensor<2x1x4xf32>
		%31 = math.exp %30 : tensor<2x1x4xf32>
		%32 = "tt.reduce"(%31) <{axis = 0 : i32}> ({
		^bb0(%arg2: f32, %arg3: f32):
		%39 = arith.addf %arg2, %arg3 : f32
		tt.reduce.return %39 : f32
		}) : (tensor<2x1x4xf32>) -> tensor<1x4xf32>
		%33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<1x4xf32> -> tensor<1x1x4xf32>
		%34 = tt.broadcast %33 : tensor<1x1x4xf32> -> tensor<2x1x4xf32>
		%35 = arith.divf %31, %34 : tensor<2x1x4xf32>
		%36 = tt.fp_to_fp %35, rounding = rtne : tensor<2x1x4xf32> -> tensor<2x1x4xf8E5M2>
		%37 = tt.splat %arg1 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%38 = tt.addptr %37, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		tt.store %38, %36 : tensor<2x1x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x1x4xf8E5M2>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x1x4xf8E5M2>

// CHECK: %[[FP_TO_F32:[A-Za-z0-9_]+]] = tt.fp_to_fp %[[TENSOR_INPUT]] : tensor<2x1x4xf8E5M2> -> tensor<2x1x4xf32>

// CHECK: %[[EMPTY_MAX:[A-Za-z0-9_]+]] = tensor.empty() : tensor<1x4xf32>
// CHECK: %[[FILL_NINF:[A-Za-z0-9_]+]] = linalg.fill ins(%cst_0 : f32) outs(%[[EMPTY_MAX]] : tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[MAX_REDUCED:[A-Za-z0-9_]+]] = linalg.reduce ins(%[[FP_TO_F32]] : tensor<2x1x4xf32>) outs(%[[FILL_NINF]] : tensor<1x4xf32>) dimensions = [0]
// CHECK-NEXT: (%in: f32, %init: f32) {
// CHECK-NEXT: %[[MAX_NUM:[A-Za-z0-9_]+]] = arith.maxnumf %in, %init : f32
// CHECK-NEXT: linalg.yield %[[MAX_NUM]] : f32
// CHECK-NEXT: }

// CHECK: %[[EMPTY_BROADCAST:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x1x4xf32>
// CHECK: %[[MAX_BROADCASTED:[A-Za-z0-9_]+]] = linalg.broadcast ins(%[[MAX_REDUCED]] : tensor<1x4xf32>) outs(%[[EMPTY_BROADCAST]] : tensor<2x1x4xf32>) dimensions = [0]

// CHECK: %[[SUB_MAX:[A-Za-z0-9_]+]] = arith.subf %[[FP_TO_F32]], %[[MAX_BROADCASTED]] : tensor<2x1x4xf32>
// CHECK: %[[EXP_RESULT:[A-Za-z0-9_]+]] = math.exp %[[SUB_MAX]] : tensor<2x1x4xf32>

// CHECK: %[[FILL_ZERO:[A-Za-z0-9_]+]] = linalg.fill ins(%cst : f32) outs(%[[EMPTY_MAX]] : tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[SUM_REDUCED:[A-Za-z0-9_]+]] = linalg.reduce ins(%[[EXP_RESULT]] : tensor<2x1x4xf32>) outs(%[[FILL_ZERO]] : tensor<1x4xf32>) dimensions = [0]
// CHECK-NEXT: (%in: f32, %init: f32) {
// CHECK-NEXT: %[[ADD_SUM:[A-Za-z0-9_]+]] = arith.addf %in, %init : f32
// CHECK-NEXT: linalg.yield %[[ADD_SUM]] : f32
// CHECK-NEXT: }

// CHECK: %[[SUM_BROADCASTED:[A-Za-z0-9_]+]] = linalg.broadcast ins(%[[SUM_REDUCED]] : tensor<1x4xf32>) outs(%[[EMPTY_BROADCAST]] : tensor<2x1x4xf32>) dimensions = [0]

// CHECK: %[[SOFTMAX_F32:[A-Za-z0-9_]+]] = arith.divf %[[EXP_RESULT]], %[[SUM_BROADCASTED]] : tensor<2x1x4xf32>

// CHECK: %[[EMPTY_F8:[A-Za-z0-9_]+]] = tensor.empty() : tensor<2x1x4xf8E5M2>
// CHECK: %[[SOFTMAX_F8:[A-Za-z0-9_]+]] = hfusion.cast {mode = #hfusion.round_mode} ins(%[[SOFTMAX_F32]] : tensor<2x1x4xf32>) outs(%[[EMPTY_F8]] : tensor<2x1x4xf8E5M2>) -> tensor<2x1x4xf8E5M2>