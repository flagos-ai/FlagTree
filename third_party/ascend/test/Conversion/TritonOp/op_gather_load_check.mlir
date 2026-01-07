// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s
// -----
// bfloat16

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<i32> , %arg2: !tt.ptr<bf16>) {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c240 = arith.constant 240 : index 
    %c375144 = arith.constant 375144 : index 
    %c133_i32 = arith.constant 133 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<240> : tensor<133x1xi32> 
    %cst_0 = arith.constant dense<240> : tensor<240xi32> 
    %cst_1 = arith.constant dense<0> : tensor<133xi32> 
    %cst_2 = arith.constant dense<375144> : tensor<133xi32> 
    %c7816_i32 = arith.constant 7816 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c7816_i32 : i32 
    %2 = tt.make_range {end = 133 : i32, start = 0 : i32} : tensor<133xi32> 
    %3 = tt.splat %1 : i32 -> tensor<133xi32> 
    %4 = arith.addi %2, %3 : tensor<133xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<133x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 240 : i32, start = 0 : i32} : tensor<240xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<240xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<133x1x!tt.ptr<bf16>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<240xi32> -> tensor<1x240xi32> 
    %10 = tt.broadcast %9 : tensor<1x240xi32> -> tensor<133x240xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<240xi1> -> tensor<1x240xi1> 
    %12 = tt.broadcast %11 : tensor<1x240xi1> -> tensor<133x240xi1> 
    scf.for %arg3 = %c0_i32 to %c7816_i32 step %c133_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<133xi32> 
      %14 = arith.addi %4, %13 : tensor<133xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<133xi32> 
      %16 = tt.addptr %5, %14 : tensor<133x!tt.ptr<i32>>, tensor<133xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<133x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c375144, %c240], [%c-1, %c0], [-1, 240] : !tt.ptr<bf16>, tensor<133xi32> -> tensor<133x240xbf16> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<133xi32> -> tensor<133x1xi32> 
      %20 = arith.muli %19, %cst : tensor<133x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<133x1x!tt.ptr<bf16>>, tensor<133x1xi32> 
      %22 = tt.broadcast %21 : tensor<133x1x!tt.ptr<bf16>> -> tensor<133x240x!tt.ptr<bf16>> 
      %23 = tt.addptr %22, %10 : tensor<133x240x!tt.ptr<bf16>>, tensor<133x240xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<133xi1> -> tensor<133x1xi1> 
      %25 = tt.broadcast %24 : tensor<133x1xi1> -> tensor<133x240xi1> 
      %26 = arith.andi %25, %12 : tensor<133x240xi1> 
      tt.store %23, %18, %26 : tensor<133x240x!tt.ptr<bf16>> 
    } 
    tt.return 
  } 
}  

// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = bufferization.to_tensor %[[X:.*]] restrict writable {gather_load} : memref<133x240xbf16>


// -----
// uint8

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<i8> , %arg1: !tt.ptr<i32> , %arg2: !tt.ptr<i8> )  {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c37 = arith.constant 37 : index 
    %c324344 = arith.constant 324344 : index 
    %c1729_i32 = arith.constant 1729 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<37> : tensor<1729x1xi32> 
    %cst_0 = arith.constant dense<37> : tensor<37xi32> 
    %cst_1 = arith.constant dense<0> : tensor<1729xi32> 
    %cst_2 = arith.constant dense<324344> : tensor<1729xi32> 
    %c6758_i32 = arith.constant 6758 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c6758_i32 : i32 
    %2 = tt.make_range {end = 1729 : i32, start = 0 : i32} : tensor<1729xi32> 
    %3 = tt.splat %1 : i32 -> tensor<1729xi32> 
    %4 = arith.addi %2, %3 : tensor<1729xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1729x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 37 : i32, start = 0 : i32} : tensor<37xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<37xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<i8> -> tensor<1729x1x!tt.ptr<i8>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<37xi32> -> tensor<1x37xi32> 
    %10 = tt.broadcast %9 : tensor<1x37xi32> -> tensor<1729x37xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<37xi1> -> tensor<1x37xi1> 
    %12 = tt.broadcast %11 : tensor<1x37xi1> -> tensor<1729x37xi1> 
    scf.for %arg3 = %c0_i32 to %c6758_i32 step %c1729_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<1729xi32> 
      %14 = arith.addi %4, %13 : tensor<1729xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<1729xi32> 
      %16 = tt.addptr %5, %14 : tensor<1729x!tt.ptr<i32>>, tensor<1729xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<1729x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c324344, %c37], [%c-1, %c0], [-1, 37] : !tt.ptr<i8>, tensor<1729xi32> -> tensor<1729x37xi8> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<1729xi32> -> tensor<1729x1xi32> 
      %20 = arith.muli %19, %cst : tensor<1729x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<1729x1x!tt.ptr<i8>>, tensor<1729x1xi32> 
      %22 = tt.broadcast %21 : tensor<1729x1x!tt.ptr<i8>> -> tensor<1729x37x!tt.ptr<i8>> 
      %23 = tt.addptr %22, %10 : tensor<1729x37x!tt.ptr<i8>>, tensor<1729x37xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<1729xi1> -> tensor<1729x1xi1> 
      %25 = tt.broadcast %24 : tensor<1729x1xi1> -> tensor<1729x37xi1> 
      %26 = arith.andi %25, %12 : tensor<1729x37xi1> 
      tt.store %23, %18, %26 : tensor<1729x37x!tt.ptr<i8>> 
    } 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = bufferization.to_tensor %[[X:.*]] restrict writable {gather_load} : memref<1729x37xi8>

// -----
// uint16

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<i16> , %arg1: !tt.ptr<i32> , %arg2: !tt.ptr<i16> )  {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c37 = arith.constant 37 : index 
    %c324344 = arith.constant 324344 : index 
    %c864_i32 = arith.constant 864 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<37> : tensor<864x1xi32> 
    %cst_0 = arith.constant dense<37> : tensor<37xi32> 
    %cst_1 = arith.constant dense<0> : tensor<864xi32> 
    %cst_2 = arith.constant dense<324344> : tensor<864xi32> 
    %c6758_i32 = arith.constant 6758 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c6758_i32 : i32 
    %2 = tt.make_range {end = 864 : i32, start = 0 : i32} : tensor<864xi32> 
    %3 = tt.splat %1 : i32 -> tensor<864xi32> 
    %4 = arith.addi %2, %3 : tensor<864xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<864x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 37 : i32, start = 0 : i32} : tensor<37xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<37xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<i16> -> tensor<864x1x!tt.ptr<i16>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<37xi32> -> tensor<1x37xi32> 
    %10 = tt.broadcast %9 : tensor<1x37xi32> -> tensor<864x37xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<37xi1> -> tensor<1x37xi1> 
    %12 = tt.broadcast %11 : tensor<1x37xi1> -> tensor<864x37xi1> 
    scf.for %arg3 = %c0_i32 to %c6758_i32 step %c864_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<864xi32> 
      %14 = arith.addi %4, %13 : tensor<864xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<864xi32> 
      %16 = tt.addptr %5, %14 : tensor<864x!tt.ptr<i32>>, tensor<864xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<864x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c324344, %c37], [%c-1, %c0], [-1, 37] : !tt.ptr<i16>, tensor<864xi32> -> tensor<864x37xi16> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<864xi32> -> tensor<864x1xi32> 
      %20 = arith.muli %19, %cst : tensor<864x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<864x1x!tt.ptr<i16>>, tensor<864x1xi32> 
      %22 = tt.broadcast %21 : tensor<864x1x!tt.ptr<i16>> -> tensor<864x37x!tt.ptr<i16>> 
      %23 = tt.addptr %22, %10 : tensor<864x37x!tt.ptr<i16>>, tensor<864x37xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<864xi1> -> tensor<864x1xi1> 
      %25 = tt.broadcast %24 : tensor<864x1xi1> -> tensor<864x37xi1> 
      %26 = arith.andi %25, %12 : tensor<864x37xi1> 
      tt.store %23, %18, %26 : tensor<864x37x!tt.ptr<i16>> 
    } 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = bufferization.to_tensor %[[X:.*]] restrict writable {gather_load} : memref<864x37xi16>

// -----
// uint32

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<i32> , %arg1: !tt.ptr<i32> , %arg2: !tt.ptr<i32> )  {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c37 = arith.constant 37 : index 
    %c324344 = arith.constant 324344 : index 
    %c432_i32 = arith.constant 432 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<37> : tensor<432x1xi32> 
    %cst_0 = arith.constant dense<37> : tensor<37xi32> 
    %cst_1 = arith.constant dense<0> : tensor<432xi32> 
    %cst_2 = arith.constant dense<324344> : tensor<432xi32> 
    %c6758_i32 = arith.constant 6758 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c6758_i32 : i32 
    %2 = tt.make_range {end = 432 : i32, start = 0 : i32} : tensor<432xi32> 
    %3 = tt.splat %1 : i32 -> tensor<432xi32> 
    %4 = arith.addi %2, %3 : tensor<432xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<432x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 37 : i32, start = 0 : i32} : tensor<37xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<37xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<432x1x!tt.ptr<i32>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<37xi32> -> tensor<1x37xi32> 
    %10 = tt.broadcast %9 : tensor<1x37xi32> -> tensor<432x37xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<37xi1> -> tensor<1x37xi1> 
    %12 = tt.broadcast %11 : tensor<1x37xi1> -> tensor<432x37xi1> 
    scf.for %arg3 = %c0_i32 to %c6758_i32 step %c432_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<432xi32> 
      %14 = arith.addi %4, %13 : tensor<432xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<432xi32> 
      %16 = tt.addptr %5, %14 : tensor<432x!tt.ptr<i32>>, tensor<432xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<432x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c324344, %c37], [%c-1, %c0], [-1, 37] : !tt.ptr<i32>, tensor<432xi32> -> tensor<432x37xi32> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<432xi32> -> tensor<432x1xi32> 
      %20 = arith.muli %19, %cst : tensor<432x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<432x1x!tt.ptr<i32>>, tensor<432x1xi32> 
      %22 = tt.broadcast %21 : tensor<432x1x!tt.ptr<i32>> -> tensor<432x37x!tt.ptr<i32>> 
      %23 = tt.addptr %22, %10 : tensor<432x37x!tt.ptr<i32>>, tensor<432x37xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<432xi1> -> tensor<432x1xi1> 
      %25 = tt.broadcast %24 : tensor<432x1xi1> -> tensor<432x37xi1> 
      %26 = arith.andi %25, %12 : tensor<432x37xi1> 
      tt.store %23, %18, %26 : tensor<432x37x!tt.ptr<i32>> 
    } 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = bufferization.to_tensor %[[X:.*]] restrict writable {gather_load} : memref<432x37xi32>

// -----
// uint64

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c37 = arith.constant 37 : index 
    %c324344 = arith.constant 324344 : index 
    %c216_i32 = arith.constant 216 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<37> : tensor<216x1xi32> 
    %cst_0 = arith.constant dense<37> : tensor<37xi32> 
    %cst_1 = arith.constant dense<0> : tensor<216xi32> 
    %cst_2 = arith.constant dense<324344> : tensor<216xi32> 
    %c6758_i32 = arith.constant 6758 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c6758_i32 : i32 
    %2 = tt.make_range {end = 216 : i32, start = 0 : i32} : tensor<216xi32> 
    %3 = tt.splat %1 : i32 -> tensor<216xi32> 
    %4 = arith.addi %2, %3 : tensor<216xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<216x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 37 : i32, start = 0 : i32} : tensor<37xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<37xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<216x1x!tt.ptr<i64>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<37xi32> -> tensor<1x37xi32> 
    %10 = tt.broadcast %9 : tensor<1x37xi32> -> tensor<216x37xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<37xi1> -> tensor<1x37xi1> 
    %12 = tt.broadcast %11 : tensor<1x37xi1> -> tensor<216x37xi1> 
    scf.for %arg3 = %c0_i32 to %c6758_i32 step %c216_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<216xi32> 
      %14 = arith.addi %4, %13 : tensor<216xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<216xi32> 
      %16 = tt.addptr %5, %14 : tensor<216x!tt.ptr<i32>>, tensor<216xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<216x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c324344, %c37], [%c-1, %c0], [-1, 37] : !tt.ptr<i64>, tensor<216xi32> -> tensor<216x37xi64> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<216xi32> -> tensor<216x1xi32> 
      %20 = arith.muli %19, %cst : tensor<216x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<216x1x!tt.ptr<i64>>, tensor<216x1xi32> 
      %22 = tt.broadcast %21 : tensor<216x1x!tt.ptr<i64>> -> tensor<216x37x!tt.ptr<i64>> 
      %23 = tt.addptr %22, %10 : tensor<216x37x!tt.ptr<i64>>, tensor<216x37xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<216xi1> -> tensor<216x1xi1> 
      %25 = tt.broadcast %24 : tensor<216x1xi1> -> tensor<216x37xi1> 
      %26 = arith.andi %25, %12 : tensor<216x37xi1> 
      tt.store %23, %18, %26 : tensor<216x37x!tt.ptr<i64>> 
    } 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = bufferization.to_tensor %[[X:.*]] restrict writable : memref<216xi32>

// -----
// f8E4M3FN

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c240 = arith.constant 240 : index 
    %c375144 = arith.constant 375144 : index 
    %c133_i32 = arith.constant 133 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<240> : tensor<133x1xi32> 
    %cst_0 = arith.constant dense<240> : tensor<240xi32> 
    %cst_1 = arith.constant dense<0> : tensor<133xi32> 
    %cst_2 = arith.constant dense<375144> : tensor<133xi32> 
    %c7816_i32 = arith.constant 7816 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c7816_i32 : i32 
    %2 = tt.make_range {end = 133 : i32, start = 0 : i32} : tensor<133xi32> 
    %3 = tt.splat %1 : i32 -> tensor<133xi32> 
    %4 = arith.addi %2, %3 : tensor<133xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<133x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 240 : i32, start = 0 : i32} : tensor<240xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<240xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<f8E4M3FN> -> tensor<133x1x!tt.ptr<f8E4M3FN>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<240xi32> -> tensor<1x240xi32> 
    %10 = tt.broadcast %9 : tensor<1x240xi32> -> tensor<133x240xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<240xi1> -> tensor<1x240xi1> 
    %12 = tt.broadcast %11 : tensor<1x240xi1> -> tensor<133x240xi1> 
    scf.for %arg3 = %c0_i32 to %c7816_i32 step %c133_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<133xi32> 
      %14 = arith.addi %4, %13 : tensor<133xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<133xi32> 
      %16 = tt.addptr %5, %14 : tensor<133x!tt.ptr<i32>>, tensor<133xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<133x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c375144, %c240], [%c-1, %c0], [-1, 240] : !tt.ptr<f16>, tensor<133xi32> -> tensor<133x240xf16> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<133xi32> -> tensor<133x1xi32> 
      %20 = arith.muli %19, %cst : tensor<133x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<133x1x!tt.ptr<f8E4M3FN>>, tensor<133x1xi32> 
      %22 = tt.broadcast %21 : tensor<133x1x!tt.ptr<f8E4M3FN>> -> tensor<133x240x!tt.ptr<f8E4M3FN>> 
      %23 = tt.addptr %22, %10 : tensor<133x240x!tt.ptr<f8E4M3FN>>, tensor<133x240xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<133xi1> -> tensor<133x1xi1> 
      %25 = tt.broadcast %24 : tensor<133x1xi1> -> tensor<133x240xi1> 
      %26 = arith.andi %25, %12 : tensor<133x240xi1> 
      %27 = tt.fp_to_fp %18, rounding = rtne : tensor<133x240xf16> -> tensor<133x240xf8E4M3FN> 
      tt.store %23, %27, %26 : tensor<133x240x!tt.ptr<f8E4M3FN>> 
    } 
    tt.return 
  } 
} 


// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [%[[Y:.*]]], sizes: [133, 240], strides: [240, 1] : memref<?xf8E4M3FN> to memref<133x240xf8E4M3FN, strided<[240, 1], offset: ?>>


// -----
// f8E5M2

module {
  tt.func public @basic_gather_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c0 = arith.constant 0 : index 
    %c-1 = arith.constant -1 : index 
    %c37 = arith.constant 37 : index 
    %c324344 = arith.constant 324344 : index 
    %c864_i32 = arith.constant 864 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<37> : tensor<864x1xi32> 
    %cst_0 = arith.constant dense<37> : tensor<37xi32> 
    %cst_1 = arith.constant dense<0> : tensor<864xi32> 
    %cst_2 = arith.constant dense<324344> : tensor<864xi32> 
    %c6758_i32 = arith.constant 6758 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c6758_i32 : i32 
    %2 = tt.make_range {end = 864 : i32, start = 0 : i32} : tensor<864xi32> 
    %3 = tt.splat %1 : i32 -> tensor<864xi32> 
    %4 = arith.addi %2, %3 : tensor<864xi32> 
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<864x!tt.ptr<i32>> 
    %6 = tt.make_range {end = 37 : i32, start = 0 : i32} : tensor<37xi32> 
    %7 = arith.cmpi slt, %6, %cst_0 : tensor<37xi32> 
    %8 = tt.splat %arg2 : !tt.ptr<f8E5M2> -> tensor<864x1x!tt.ptr<f8E5M2>> 
    %9 = tt.expand_dims %6 {axis = 0 : i32} : tensor<37xi32> -> tensor<1x37xi32> 
    %10 = tt.broadcast %9 : tensor<1x37xi32> -> tensor<864x37xi32> 
    %11 = tt.expand_dims %7 {axis = 0 : i32} : tensor<37xi1> -> tensor<1x37xi1> 
    %12 = tt.broadcast %11 : tensor<1x37xi1> -> tensor<864x37xi1> 
    scf.for %arg3 = %c0_i32 to %c6758_i32 step %c864_i32  : i32 {
      %13 = tt.splat %arg3 : i32 -> tensor<864xi32> 
      %14 = arith.addi %4, %13 : tensor<864xi32> 
      %15 = arith.cmpi slt, %14, %cst_2 : tensor<864xi32> 
      %16 = tt.addptr %5, %14 : tensor<864x!tt.ptr<i32>>, tensor<864xi32> 
      %17 = tt.load %16, %15, %cst_1 : tensor<864x!tt.ptr<i32>> 
      %18 = tt.gather_load %arg0, %17, 0, [%c324344, %c37], [%c-1, %c0], [-1, 37] : !tt.ptr<f16>, tensor<864xi32> -> tensor<864x37xf16> 
      %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<864xi32> -> tensor<864x1xi32> 
      %20 = arith.muli %19, %cst : tensor<864x1xi32> 
      %21 = tt.addptr %8, %20 : tensor<864x1x!tt.ptr<f8E5M2>>, tensor<864x1xi32> 
      %22 = tt.broadcast %21 : tensor<864x1x!tt.ptr<f8E5M2>> -> tensor<864x37x!tt.ptr<f8E5M2>> 
      %23 = tt.addptr %22, %10 : tensor<864x37x!tt.ptr<f8E5M2>>, tensor<864x37xi32> 
      %24 = tt.expand_dims %15 {axis = 1 : i32} : tensor<864xi1> -> tensor<864x1xi1> 
      %25 = tt.broadcast %24 : tensor<864x1xi1> -> tensor<864x37xi1> 
      %26 = arith.andi %25, %12 : tensor<864x37xi1> 
      %27 = tt.fp_to_fp %18, rounding = rtne : tensor<864x37xf16> -> tensor<864x37xf8E5M2> 
      tt.store %23, %27, %26 : tensor<864x37x!tt.ptr<f8E5M2>> 
    } 
    tt.return 
  } 
} 

// CHECK-LABEL:   func.func @basic_gather_load
// CHECK: %[[REV:.*]] = memref.reinterpret_cast %[[X:.*]] to offset: [%[[Y:.*]]], sizes: [864, 37], strides: [37, 1] : memref<?xf8E5M2> to memref<864x37xf8E5M2, strided<[37, 1], offset: ?>>