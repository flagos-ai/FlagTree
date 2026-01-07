// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @triton_min_1d8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":22:0), %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":22:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":22:0)) attributes {noinline = false} {
    %true = arith.constant true loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc4)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc4)
    %4 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>> loc(#loc5)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i8>>, tensor<16xi32> loc(#loc5)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i8>> loc(#loc6)
    tt.assert %true, "Expecting input to be integer type" : i1 loc(#loc14)
    %7 = arith.extui %6 : tensor<16xi8> to tensor<16xi32> loc(#loc15)
    %8 = "tt.reduce"(%7) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i32 loc(callsite(#loc1 at #loc8)), %arg4: i32 loc(callsite(#loc1 at #loc8))):
      %10 = arith.minsi %arg3, %arg4 : i32 loc(#loc19)
      tt.reduce.return %10 : i32 loc(#loc16)
    }) : (tensor<16xi32>) -> i32 loc(#loc16)
    %9 = arith.trunci %8 : i32 to i8 loc(#loc12)
    tt.store %arg1, %9 : !tt.ptr<i8> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)

// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.minsi %[[VAL_0]], %[[VAL_1]] : i32
// -----------


module {
  tt.func public @triton_min_1d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0), %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0)) attributes {noinline = false} {
    %true = arith.constant true loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc4)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc4)
    %4 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>> loc(#loc5)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i16>>, tensor<16xi32> loc(#loc5)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i16>> loc(#loc6)
    tt.assert %true, "Expecting input to be integer type" : i1 loc(#loc14)
    %7 = arith.extui %6 : tensor<16xi16> to tensor<16xi32> loc(#loc15)
    %8 = "tt.reduce"(%7) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i32 loc(callsite(#loc1 at #loc8)), %arg4: i32 loc(callsite(#loc1 at #loc8))):
      %10 = arith.minsi %arg3, %arg4 : i32 loc(#loc19)
      tt.reduce.return %10 : i32 loc(#loc16)
    }) : (tensor<16xi32>) -> i32 loc(#loc16)
    %9 = arith.trunci %8 : i32 to i16 loc(#loc12)
    tt.store %arg1, %9 : !tt.ptr<i16> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)

// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.minsi %[[VAL_0]], %[[VAL_1]] : i32
// -----------

module {
  tt.func public @triton_min_1d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0), %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/min_uint.py":45:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc2)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc3)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc3)
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc4)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc4)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>> loc(#loc5)
    %7 = "tt.reduce"(%6) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i32 loc(callsite(#loc8 at #loc7)), %arg4: i32 loc(callsite(#loc8 at #loc7))):
      %8 = arith.minui %arg3, %arg4 : i32 loc(#loc15)
      tt.reduce.return %8 : i32 loc(#loc12)
    }) : (tensor<16xi32>) -> i32 loc(#loc12)
    tt.store %arg1, %7 : !tt.ptr<i32> loc(#loc10)
    tt.return loc(#loc11)
  } loc(#loc)
} loc(#loc)

// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.minui %[[VAL_0]], %[[VAL_1]] : i32
// -----------