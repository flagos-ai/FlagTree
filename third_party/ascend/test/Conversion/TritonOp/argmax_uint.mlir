// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' --split-input-file %s | FileCheck %s


module {
  tt.func public @triton_argmax_1d(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u8_argmax.py":21:0), %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u8_argmax.py":21:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u8_argmax.py":21:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc2)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc3)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc3)
    %4 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>> loc(#loc4)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i8>>, tensor<16xi32> loc(#loc4)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i8>> loc(#loc5)
    %7:2 = "tt.reduce"(%6, %1) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i8 loc(callsite(#loc21 at #loc8)), %arg4: i32 loc(callsite(#loc21 at #loc8)), %arg5: i8 loc(callsite(#loc21 at #loc8)), %arg6: i32 loc(callsite(#loc21 at #loc8))):
      %8 = arith.cmpi eq, %arg3, %arg5 : i8 loc(#loc45)
      %9 = arith.cmpi slt, %arg4, %arg6 : i32 loc(#loc46)
      %10 = arith.andi %8, %9 : i1 loc(#loc47)
      %11 = arith.cmpi ugt, %arg3, %arg5 : i8 loc(#loc48)
      %12 = arith.ori %11, %10 : i1 loc(#loc49)
      %13 = arith.select %12, %arg3, %arg5 : i8 loc(#loc50)
      %14 = arith.select %12, %arg4, %arg6 : i32 loc(#loc51)
      tt.reduce.return %13, %14 : i8, i32 loc(#loc29)
    }) : (tensor<16xi8>, tensor<16xi32>) -> (i8, i32) loc(#loc29)
    tt.store %arg1, %7#1 : !tt.ptr<i32> loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)


// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i8
// -----


module {
  tt.func public @triton_argmax_1d(%arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u16_argmax.py":20:0), %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u16_argmax.py":20:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u16_argmax.py":20:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc2)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc3)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc3)
    %4 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>> loc(#loc4)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i16>>, tensor<16xi32> loc(#loc4)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i16>> loc(#loc5)
    %7:2 = "tt.reduce"(%6, %1) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i16 loc(callsite(#loc21 at #loc8)), %arg4: i32 loc(callsite(#loc21 at #loc8)), %arg5: i16 loc(callsite(#loc21 at #loc8)), %arg6: i32 loc(callsite(#loc21 at #loc8))):
      %8 = arith.cmpi eq, %arg3, %arg5 : i16 loc(#loc45)
      %9 = arith.cmpi slt, %arg4, %arg6 : i32 loc(#loc46)
      %10 = arith.andi %8, %9 : i1 loc(#loc47)
      %11 = arith.cmpi ugt, %arg3, %arg5 : i16 loc(#loc48)
      %12 = arith.ori %11, %10 : i1 loc(#loc49)
      %13 = arith.select %12, %arg3, %arg5 : i16 loc(#loc50)
      %14 = arith.select %12, %arg4, %arg6 : i32 loc(#loc51)
      tt.reduce.return %13, %14 : i16, i32 loc(#loc29)
    }) : (tensor<16xi16>, tensor<16xi32>) -> (i16, i32) loc(#loc29)
    tt.store %arg1, %7#1 : !tt.ptr<i32> loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)


// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i16
// -----

module {
  tt.func public @triton_argmax_1d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u32_argmax.py":21:0), %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u32_argmax.py":21:0), %arg2: i32 {tt.divisibility = 16 : i32} loc("/home/l30058175/wxue/test_u32_argmax.py":21:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc2)
    %2 = tt.splat %0 : i32 -> tensor<16xi32> loc(#loc3)
    %3 = arith.addi %2, %1 : tensor<16xi32> loc(#loc3)
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>> loc(#loc4)
    %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<i32>>, tensor<16xi32> loc(#loc4)
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>> loc(#loc5)
    %7:2 = "tt.reduce"(%6, %1) <{axis = 0 : i32}> ({
    ^bb0(%arg3: i32 loc(callsite(#loc21 at #loc8)), %arg4: i32 loc(callsite(#loc21 at #loc8)), %arg5: i32 loc(callsite(#loc21 at #loc8)), %arg6: i32 loc(callsite(#loc21 at #loc8))):
      %8 = arith.cmpi eq, %arg3, %arg5 : i32 loc(#loc45)
      %9 = arith.cmpi slt, %arg4, %arg6 : i32 loc(#loc46)
      %10 = arith.andi %8, %9 : i1 loc(#loc47)
      %11 = arith.cmpi ugt, %arg3, %arg5 : i32 loc(#loc48)
      %12 = arith.ori %11, %10 : i1 loc(#loc49)
      %13 = arith.select %12, %arg3, %arg5 : i32 loc(#loc50)
      %14 = arith.select %12, %arg4, %arg6 : i32 loc(#loc51)
      tt.reduce.return %13, %14 : i32, i32 loc(#loc29)
    }) : (tensor<16xi32>, tensor<16xi32>) -> (i32, i32) loc(#loc29)
    tt.store %arg1, %7#1 : !tt.ptr<i32> loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)

// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.cmpi ugt, %[[VAL_0]], %[[VAL_1]] : i32
// -----