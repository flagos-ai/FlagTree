// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file %s | FileCheck %s

module {
  tt.func public @kernel_randint(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c5_i32 = arith.constant 5 : i32
    %c-766435501_i32 = arith.constant -766435501 : i32
    %c-845247145_i32 = arith.constant -845247145 : i32
    %c-1640531522_i32 = arith.constant -1640531522 : i32
    %c-1150833019_i32 = arith.constant -1150833019 : i32
    %c1013904247_i32 = arith.constant 1013904247 : i32
    %c1993301258_i32 = arith.constant 1993301258 : i32
    %c-626627280_i32 = arith.constant -626627280 : i32
    %c842468239_i32 = arith.constant 842468239 : i32
    %c2027808489_i32 = arith.constant 2027808489 : i32
    %c-308364780_i32 = arith.constant -308364780 : i32
    %c387276962_i32 = arith.constant 387276962 : i32
    %c-1459197799_i32 = arith.constant -1459197799 : i32
    %c-1253254565_i32 = arith.constant -1253254565 : i32
    %c1684936478_i32 = arith.constant 1684936478 : i32
    %c1401181204_i32 = arith.constant 1401181204 : i32
    %c534103459_i32 = arith.constant 534103459 : i32
    %c-616729560_i32 = arith.constant -616729560 : i32
    %c-1879881850_i32 = arith.constant -1879881850 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c6_i32 : i32
    %2 = arith.addi %1, %c6_i32 : i32
    %3 = arith.cmpi sle, %2, %c6_i32 : i32
    %4 = scf.if %3 -> (i32) {
      scf.yield %c6_i32 : i32
    } else {
      %12 = arith.subi %c6_i32, %1 : i32
      scf.yield %12 : i32
    }
    %5 = tt.bitcast %c0_i32 : i32 -> i32
    %6 = tt.mulhiui %c-845247145_i32, %5 : i32
    %7 = arith.xori %6, %5 : i32
    %8 = arith.xori %7, %c5_i32 : i32
    %9 = arith.muli %5, %c-845247145_i32 : i32
    %10 = tt.mulhiui %c-766435501_i32, %8 : i32
    %11 = arith.muli %8, %c-766435501_i32 : i32
    scf.for %arg1 = %c0_i32 to %4 step %c1_i32  : i32 {
      %12 = arith.addi %1, %arg1 : i32
      %13 = arith.addi %12, %c10_i32 : i32
      %14 = tt.bitcast %13 : i32 -> i32
      %15 = tt.mulhiui %c-766435501_i32, %14 : i32
      %16 = arith.xori %15, %5 : i32
      %17 = arith.muli %14, %c-766435501_i32 : i32
      %18 = tt.mulhiui %c-845247145_i32, %16 : i32
      %19 = arith.xori %18, %9 : i32
      %20 = arith.xori %19, %c-1640531522_i32 : i32
      %21 = arith.xori %10, %17 : i32
      %22 = arith.xori %21, %c-1150833019_i32 : i32
      %23 = arith.muli %16, %c-845247145_i32 : i32
      %24 = tt.mulhiui %c-845247145_i32, %22 : i32
      %25 = arith.xori %24, %23 : i32
      %26 = arith.xori %25, %c1013904247_i32 : i32
      %27 = tt.mulhiui %c-766435501_i32, %20 : i32
      %28 = arith.xori %27, %11 : i32
      %29 = arith.xori %28, %c1993301258_i32 : i32
      %30 = arith.muli %22, %c-845247145_i32 : i32
      %31 = arith.muli %20, %c-766435501_i32 : i32
      %32 = tt.mulhiui %c-845247145_i32, %29 : i32
      %33 = arith.xori %32, %30 : i32
      %34 = arith.xori %33, %c-626627280_i32 : i32
      %35 = tt.mulhiui %c-766435501_i32, %26 : i32
      %36 = arith.xori %35, %31 : i32
      %37 = arith.xori %36, %c842468239_i32 : i32
      %38 = arith.muli %29, %c-845247145_i32 : i32
      %39 = arith.muli %26, %c-766435501_i32 : i32
      %40 = tt.mulhiui %c-845247145_i32, %37 : i32
      %41 = arith.xori %40, %38 : i32
      %42 = arith.xori %41, %c2027808489_i32 : i32
      %43 = tt.mulhiui %c-766435501_i32, %34 : i32
      %44 = arith.xori %43, %39 : i32
      %45 = arith.xori %44, %c-308364780_i32 : i32
      %46 = arith.muli %37, %c-845247145_i32 : i32
      %47 = arith.muli %34, %c-766435501_i32 : i32
      %48 = tt.mulhiui %c-845247145_i32, %45 : i32
      %49 = arith.xori %48, %46 : i32
      %50 = arith.xori %49, %c387276962_i32 : i32
      %51 = tt.mulhiui %c-766435501_i32, %42 : i32
      %52 = arith.xori %51, %47 : i32
      %53 = arith.xori %52, %c-1459197799_i32 : i32
      %54 = arith.muli %45, %c-845247145_i32 : i32
      %55 = arith.muli %42, %c-766435501_i32 : i32
      %56 = tt.mulhiui %c-845247145_i32, %53 : i32
      %57 = arith.xori %56, %54 : i32
      %58 = arith.xori %57, %c-1253254565_i32 : i32
      %59 = tt.mulhiui %c-766435501_i32, %50 : i32
      %60 = arith.xori %59, %55 : i32
      %61 = arith.xori %60, %c1684936478_i32 : i32
      %62 = arith.muli %53, %c-845247145_i32 : i32
      %63 = arith.muli %50, %c-766435501_i32 : i32
      %64 = tt.mulhiui %c-845247145_i32, %61 : i32
      %65 = arith.xori %64, %62 : i32
      %66 = arith.xori %65, %c1401181204_i32 : i32
      %67 = tt.mulhiui %c-766435501_i32, %58 : i32
      %68 = arith.xori %67, %63 : i32
      %69 = arith.xori %68, %c534103459_i32 : i32
      %70 = arith.muli %58, %c-766435501_i32 : i32
      %71 = tt.mulhiui %c-766435501_i32, %66 : i32
      %72 = arith.xori %71, %70 : i32
      %73 = arith.xori %72, %c-616729560_i32 : i32
      %74 = arith.muli %69, %c-845247145_i32 : i32
      %75 = tt.mulhiui %c-845247145_i32, %73 : i32
      %76 = arith.xori %75, %74 : i32
      %77 = arith.xori %76, %c-1879881850_i32 : i32
      %78 = tt.addptr %arg0, %12 : !tt.ptr<i8>, i32
      %79 = arith.trunci %77 : i32 to i8
      tt.store %78, %79 : !tt.ptr<i8>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel_randint
// CHECK: scf.for {{.*}} {
// CHECK: arith.mulsi_extended {{.*}}, %c-766435501_i32 : i32
// CHECK: arith.mulsi_extended {{.*}}, %c-845247145_i32 : i32
// CHECK: arith.xori {{.*}}, {{.*}} : i32
// CHECK: arith.trunci {{.*}} : i32 to i8
// CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
// CHECK-NOT: tt.store

// -----

module {
  tt.func public @kernel_randint(%arg0: !tt.ptr<i1> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c5_i32 = arith.constant 5 : i32
    %c-766435501_i32 = arith.constant -766435501 : i32
    %c-845247145_i32 = arith.constant -845247145 : i32
    %c-1640531522_i32 = arith.constant -1640531522 : i32
    %c-1150833019_i32 = arith.constant -1150833019 : i32
    %c1013904247_i32 = arith.constant 1013904247 : i32
    %c1993301258_i32 = arith.constant 1993301258 : i32
    %c-626627280_i32 = arith.constant -626627280 : i32
    %c842468239_i32 = arith.constant 842468239 : i32
    %c2027808489_i32 = arith.constant 2027808489 : i32
    %c-308364780_i32 = arith.constant -308364780 : i32
    %c387276962_i32 = arith.constant 387276962 : i32
    %c-1459197799_i32 = arith.constant -1459197799 : i32
    %c-1253254565_i32 = arith.constant -1253254565 : i32
    %c1684936478_i32 = arith.constant 1684936478 : i32
    %c1401181204_i32 = arith.constant 1401181204 : i32
    %c534103459_i32 = arith.constant 534103459 : i32
    %c-616729560_i32 = arith.constant -616729560 : i32
    %c-1879881850_i32 = arith.constant -1879881850 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c6_i32 : i32
    %2 = arith.addi %1, %c6_i32 : i32
    %3 = arith.cmpi sle, %2, %c6_i32 : i32
    %4 = scf.if %3 -> (i32) {
      scf.yield %c6_i32 : i32
    } else {
      %12 = arith.subi %c6_i32, %1 : i32
      scf.yield %12 : i32
    }
    %5 = tt.bitcast %c0_i32 : i32 -> i32
    %6 = tt.mulhiui %c-845247145_i32, %5 : i32
    %7 = arith.xori %6, %5 : i32
    %8 = arith.xori %7, %c5_i32 : i32
    %9 = arith.muli %5, %c-845247145_i32 : i32
    %10 = tt.mulhiui %c-766435501_i32, %8 : i32
    %11 = arith.muli %8, %c-766435501_i32 : i32
    scf.for %arg1 = %c0_i32 to %4 step %c1_i32  : i32 {
      %12 = arith.addi %1, %arg1 : i32
      %13 = arith.addi %12, %c10_i32 : i32
      %14 = tt.bitcast %13 : i32 -> i32
      %15 = tt.mulhiui %c-766435501_i32, %14 : i32
      %16 = arith.xori %15, %5 : i32
      %17 = arith.muli %14, %c-766435501_i32 : i32
      %18 = tt.mulhiui %c-845247145_i32, %16 : i32
      %19 = arith.xori %18, %9 : i32
      %20 = arith.xori %19, %c-1640531522_i32 : i32
      %21 = arith.xori %10, %17 : i32
      %22 = arith.xori %21, %c-1150833019_i32 : i32
      %23 = arith.muli %16, %c-845247145_i32 : i32
      %24 = tt.mulhiui %c-845247145_i32, %22 : i32
      %25 = arith.xori %24, %23 : i32
      %26 = arith.xori %25, %c1013904247_i32 : i32
      %27 = tt.mulhiui %c-766435501_i32, %20 : i32
      %28 = arith.xori %27, %11 : i32
      %29 = arith.xori %28, %c1993301258_i32 : i32
      %30 = arith.muli %22, %c-845247145_i32 : i32
      %31 = arith.muli %20, %c-766435501_i32 : i32
      %32 = tt.mulhiui %c-845247145_i32, %29 : i32
      %33 = arith.xori %32, %30 : i32
      %34 = arith.xori %33, %c-626627280_i32 : i32
      %35 = tt.mulhiui %c-766435501_i32, %26 : i32
      %36 = arith.xori %35, %31 : i32
      %37 = arith.xori %36, %c842468239_i32 : i32
      %38 = arith.muli %29, %c-845247145_i32 : i32
      %39 = arith.muli %26, %c-766435501_i32 : i32
      %40 = tt.mulhiui %c-845247145_i32, %37 : i32
      %41 = arith.xori %40, %38 : i32
      %42 = arith.xori %41, %c2027808489_i32 : i32
      %43 = tt.mulhiui %c-766435501_i32, %34 : i32
      %44 = arith.xori %43, %39 : i32
      %45 = arith.xori %44, %c-308364780_i32 : i32
      %46 = arith.muli %37, %c-845247145_i32 : i32
      %47 = arith.muli %34, %c-766435501_i32 : i32
      %48 = tt.mulhiui %c-845247145_i32, %45 : i32
      %49 = arith.xori %48, %46 : i32
      %50 = arith.xori %49, %c387276962_i32 : i32
      %51 = tt.mulhiui %c-766435501_i32, %42 : i32
      %52 = arith.xori %51, %47 : i32
      %53 = arith.xori %52, %c-1459197799_i32 : i32
      %54 = arith.muli %45, %c-845247145_i32 : i32
      %55 = arith.muli %42, %c-766435501_i32 : i32
      %56 = tt.mulhiui %c-845247145_i32, %53 : i32
      %57 = arith.xori %56, %54 : i32
      %58 = arith.xori %57, %c-1253254565_i32 : i32
      %59 = tt.mulhiui %c-766435501_i32, %50 : i32
      %60 = arith.xori %59, %55 : i32
      %61 = arith.xori %60, %c1684936478_i32 : i32
      %62 = arith.muli %53, %c-845247145_i32 : i32
      %63 = arith.muli %50, %c-766435501_i32 : i32
      %64 = tt.mulhiui %c-845247145_i32, %61 : i32
      %65 = arith.xori %64, %62 : i32
      %66 = arith.xori %65, %c1401181204_i32 : i32
      %67 = tt.mulhiui %c-766435501_i32, %58 : i32
      %68 = arith.xori %67, %63 : i32
      %69 = arith.xori %68, %c534103459_i32 : i32
      %70 = arith.muli %58, %c-766435501_i32 : i32
      %71 = tt.mulhiui %c-766435501_i32, %66 : i32
      %72 = arith.xori %71, %70 : i32
      %73 = arith.xori %72, %c-616729560_i32 : i32
      %74 = arith.muli %69, %c-845247145_i32 : i32
      %75 = tt.mulhiui %c-845247145_i32, %73 : i32
      %76 = arith.xori %75, %74 : i32
      %77 = arith.xori %76, %c-1879881850_i32 : i32
      %78 = tt.addptr %arg0, %12 : !tt.ptr<i1>, i32
      %79 = tt.bitcast %78 : !tt.ptr<i1> -> !tt.ptr<i8>
      %80 = arith.trunci %77 : i32 to i8
      tt.store %79, %80 : !tt.ptr<i8>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel_randint
// CHECK: scf.for {{.*}} {
// CHECK: arith.mulsi_extended {{.*}}, %c-766435501_i32 : i32
// CHECK: arith.mulsi_extended {{.*}}, %c-845247145_i32 : i32
// CHECK: arith.xori {{.*}}, {{.*}} : i32
// CHECK: arith.trunci {{.*}} : i32 to i8
// CHECK: bufferization.materialize_in_destination {{.*}} : (tensor<1xi8>, memref<1xi8, strided<[1], offset: ?>>) -> ()
// CHECK-NOT: tt.store
