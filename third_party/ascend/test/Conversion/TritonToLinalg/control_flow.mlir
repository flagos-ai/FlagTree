// RUN: triton-adapter-opt --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --bubble-up-operation --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @grouped_gemm_triton_kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32} , %arg3: i32 {tt.divisibility = 16 : i32} , %arg4: i32 {tt.divisibility = 16 : i32} , %arg5: i32 {tt.divisibility = 16 : i32} , %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg7: !tt.ptr<i64> {tt.divisibility = 16 : i32} , %arg8: !tt.ptr<i64> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32 
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32> 
    %c64_i64 = arith.constant 64 : i64 
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16> 
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16> 
    %c1_i32 = arith.constant 1 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %c0_i64 = arith.constant 0 : i64 
    %cst_2 = arith.constant dense<128> : tensor<32x128xi32> 
    %cst_3 = arith.constant dense<128> : tensor<64x128xi32> 
    %c128_i32 = arith.constant 128 : i32 
    %cst_4 = arith.constant dense<2048> : tensor<32x1xi32> 
    %c3145728_i64 = arith.constant 3145728 : i64 
    %cst_5 = arith.constant dense<2048> : tensor<64x1xi64> 
    %cst_6 = arith.constant dense<0> : tensor<32xi32> 
    %cst_7 = arith.constant dense<0> : tensor<64xi32> 
    %c32_i32 = arith.constant 32 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = tt.get_program_id y : i32 
    %2 = tt.addptr %arg8, %arg3 : !tt.ptr<i64>, i32 
    %3 = tt.load %2 : !tt.ptr<i64> 
    %4 = arith.extsi %0 : i32 to i64 
    %5 = arith.cmpi sge, %4, %3 : i64 
    cf.cond_br %5, ^bb1, ^bb2 
  ^bb1:  // 2 preds: ^bb0, ^bb2
    tt.return 
  ^bb2:  // pred: ^bb0
    %6 = scf.for %arg9 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg10 = %c0_i32) -> (i32)  : i32 {
      %86 = tt.addptr %arg8, %arg9 : !tt.ptr<i64>, i32 
      %87 = tt.load %86 : !tt.ptr<i64> 
      %88 = arith.cmpi sge, %4, %87 : i64 
      %89 = arith.select %88, %arg9, %arg10 : i32 
      scf.yield %89 : i32 
    } 
    %7 = tt.addptr %arg8, %6 : !tt.ptr<i64>, i32 
    %8 = tt.load %7 : !tt.ptr<i64> 
    %21 = arith.cmpi eq, %8, %c0_i64 : i64 
    cf.cond_br %21, ^bb1, ^bb3 
  ^bb3:  // pred: ^bb2
    %100 = tt.get_program_id x : i32
    %101 = arith.muli %100, %arg3 : i32
    %102 = tt.addptr %arg1, %101 : !tt.ptr<bf16>, i32
    %103 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %104 = tt.splat %102 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %105 = tt.addptr %104, %103 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %108 = tt.load %105 : tensor<1024x!tt.ptr<bf16>>
    %1017 = math.exp %108 : tensor<1024xbf16>
    %1018 = arith.muli %100, %arg3 : i32
    %1019 = tt.addptr %arg0, %1018 : !tt.ptr<bf16>, i32
    %1020 = tt.splat %1019 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %1021 = tt.addptr %1020, %103 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %1021, %1017 : tensor<1024x!tt.ptr<bf16>>
    tt.return 
  } 
}

//CHECK-LABEL: @grouped_gemm_triton_kernel
//CHECK-NOT: cf.cond_br
//CHECK: %[[COND0:.*]] = arith.cmpi sge, %[[VAL0:.*]], %[[VAL1:.*]] : i64
//CHECK: scf.if %[[COND0]] {
//CHECK: } else {
//CHECK:   %[[COND1:.*]] = arith.cmpi eq, %[[VAL2:.*]], %[[VAL3:.*]] : i64
//CHECK:   scf.if %[[COND1]] {
//CHECK:   } else {
//CHECK:   }
//CHECK: }
