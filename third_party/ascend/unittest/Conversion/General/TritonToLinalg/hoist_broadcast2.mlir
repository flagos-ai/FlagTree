// RUN: triton-opt --pass-pipeline="builtin.module(auto-blockify{auto-blockify-size=1},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-on-910-95=true force-simt-template=false},triton-to-annotation,triton-to-unstructure{compile-on-910-95=true force-scalarize-mode=false force-simt-template=false},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation{enable-aggressive-mode=true},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-on-910-95=true enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true})" --split-input-file %s | FileCheck %s

module {
  tt.func public @_attn_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                            %arg8: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                            %arg9: f32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %cst_1 = arith.constant dense<true> : tensor<128x64xi1>
    %cst_2 = arith.constant dense<64> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32
    %c524288_i32 = arith.constant 524288 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %4 = tt.broadcast %3 : tensor<1x64xi32> -> tensor<128x64xi32>
    %5 = tt.splat %arg9 : f32 -> tensor<128x128xf32>
    scf.for %arg10 = %0 to %c8192_i32 step %c28_i32  : i32 {
      %6 = arith.divsi %arg10, %c8_i32 : i32
      %7 = arith.muli %6, %c8_i32 : i32
      %8 = arith.subi %arg10, %7 : i32
      %9 = arith.muli %6, %c1024_i32 : i32
      %10 = arith.extsi %9 : i32 to i64
      %11 = arith.remsi %6, %c8_i32 : i32
      %12 = arith.muli %11, %c65536_i32 : i32
      %13 = arith.divsi %6, %c8_i32 : i32
      %14 = arith.muli %13, %c524288_i32 : i32
      %15 = arith.addi %12, %14 : i32
      %16 = arith.extsi %15 : i32 to i64
      %17 = tt.addptr %arg0, %16 : !tt.ptr<f16>, i64
      %18 = tt.addptr %arg1, %16 : !tt.ptr<f16>, i64
      %19 = tt.addptr %arg2, %16 : !tt.ptr<f16>, i64
      %20 = tt.addptr %arg3, %16 : !tt.ptr<f16>, i64
      %21 = tt.addptr %arg4, %16 : !tt.ptr<f16>, i64
      %22 = tt.addptr %arg5, %16 : !tt.ptr<f16>, i64
      %23 = tt.addptr %arg6, %16 : !tt.ptr<f16>, i64
      %24 = tt.addptr %arg7, %10 : !tt.ptr<f32>, i64
      %25 = tt.addptr %arg8, %10 : !tt.ptr<f32>, i64
      %26 = arith.muli %8, %c128_i32 : i32
      %27 = tt.splat %26 : i32 -> tensor<128xi32>
      %28 = arith.addi %27, %2 : tensor<128xi32>
      %29 = tt.expand_dims %28 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %30 = arith.muli %29, %cst_2 : tensor<128x1xi32>
      %31 = tt.splat %18 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %32 = tt.addptr %31, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %33 = tt.broadcast %32 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %34 = tt.addptr %33, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %35 = tt.load %34 : tensor<128x64x!tt.ptr<f16>>
      %36 = tt.splat %19 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %37 = tt.addptr %36, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %38 = tt.broadcast %37 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %39 = tt.addptr %38, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %40 = tt.load %39 : tensor<128x64x!tt.ptr<f16>>
      %41 = tt.splat %17 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %42 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %43 = tt.splat %24 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %44 = tt.splat %25 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
      %45 = tt.trans %35 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %46 = tt.trans %40 {order = array<i32: 1, 0>} : tensor<128x64xf16> -> tensor<64x128xf16>
      %47 = tt.splat %21 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %48:2 = scf.for %arg11 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg12 = %cst_0, %arg13 = %cst_0) -> (tensor<128x64xf32>, tensor<128x64xf32>)  : i32 {
        %59 = arith.muli %arg11, %c128_i32 : i32
        %60 = tt.splat %59 : i32 -> tensor<128xi32>
        %61 = arith.addi %60, %2 : tensor<128xi32>
        %62 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %63 = arith.muli %62, %cst_2 : tensor<128x1xi32>
        %64 = tt.addptr %41, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %65 = tt.broadcast %64 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %66 = tt.addptr %65, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %67 = tt.load %66 : tensor<128x64x!tt.ptr<f16>>
        %68 = tt.addptr %42, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %69 = tt.broadcast %68 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %70 = tt.addptr %69, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %71 = tt.load %70 : tensor<128x64x!tt.ptr<f16>>
        %72 = tt.addptr %43, %61 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %73 = tt.load %72 : tensor<128x!tt.ptr<f32>>
        %74 = tt.addptr %44, %61 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
        %75 = tt.load %74 : tensor<128x!tt.ptr<f32>>
        %76 = tt.dot %67, %45, %cst : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %77 = arith.mulf %76, %5 : tensor<128x128xf32>
        %78 = tt.expand_dims %73 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %79 = tt.broadcast %78 : tensor<128x1xf32> -> tensor<128x128xf32>
        %80 = arith.subf %77, %79 : tensor<128x128xf32>
        %81 = math.exp %80 : tensor<128x128xf32>
        %82 = arith.truncf %81 : tensor<128x128xf32> to tensor<128x128xf16>
        %83 = tt.trans %82 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %84 = tt.dot %83, %71, %arg13 : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %85 = tt.dot %71, %46, %cst : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
        %86 = tt.expand_dims %75 {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>
        %87 = tt.broadcast %86 : tensor<128x1xf32> -> tensor<128x128xf32>
        %88 = arith.subf %85, %87 : tensor<128x128xf32>
        %89 = arith.extf %82 : tensor<128x128xf16> to tensor<128x128xf32>
        %90 = arith.mulf %89, %88 : tensor<128x128xf32>
        %91 = arith.mulf %90, %5 : tensor<128x128xf32>
        %92 = arith.truncf %91 : tensor<128x128xf32> to tensor<128x128xf16>
        %93 = tt.trans %92 {order = array<i32: 1, 0>} : tensor<128x128xf16> -> tensor<128x128xf16>
        %94 = tt.dot %93, %67, %arg12 : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %95 = tt.dot %92, %35, %cst_0 : tensor<128x128xf16> * tensor<128x64xf16> -> tensor<128x64xf32>
        %96 = tt.addptr %47, %63 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %97 = tt.broadcast %96 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
        %98 = tt.addptr %97, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %99 = arith.truncf %95 : tensor<128x64xf32> to tensor<128x64xf16>
        %100 = tt.atomic_rmw fadd, acq_rel, gpu, %98, %99, %cst_1 : (tensor<128x64x!tt.ptr<f16>>, tensor<128x64xf16>, tensor<128x64xi1>) -> tensor<128x64xf16>
        scf.yield %94, %84 : tensor<128x64xf32>, tensor<128x64xf32>
      }
      %49 = tt.splat %22 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %50 = tt.addptr %49, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %51 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %52 = tt.addptr %51, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %53 = arith.truncf %48#0 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %52, %53 : tensor<128x64x!tt.ptr<f16>>
      %54 = tt.splat %23 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
      %55 = tt.addptr %54, %30 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
      %56 = tt.broadcast %55 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x64x!tt.ptr<f16>>
      %57 = tt.addptr %56, %4 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
      %58 = arith.truncf %48#1 : tensor<128x64xf32> to tensor<128x64xf16>
      tt.store %57, %58 : tensor<128x64x!tt.ptr<f16>>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @_attn_bwd
// CHECK: linalg.fill
// CHECK: memref.reinterpret_cast
// CHECK: memref.alloc
// CHECK: memref.copy
// CHECK: scf.for
// CHECK: hivm.hir.store
// CHECK: bufferization.materialize_in_destination
