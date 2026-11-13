// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s
// CHECK-LABEL: func.func @matmul_atomic_add
// CHECK-NOT: GenericAtomicRMW
// CHECK: tensor.extract_slice
// CHECK: hivm.hir.store ins(%{{.*}} : tensor<?x?xf32>) outs(%{{.*}} : memref<?x?xf32{{.*}}>) atomic = <add>

 tt.func public @matmul_atomic_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %c16_i32 = arith.constant 16 : i32
    %c16_i32_0 = arith.constant 16 : i32
    %3 = arith.muli %0, %c16_i32_0 : i32
    %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %5 = tt.splat %3 : i32 -> tensor<16xi32>
    %6 = arith.addi %5, %4 : tensor<16xi32>
    %c16_i32_1 = arith.constant 16 : i32
    %c16_i32_2 = arith.constant 16 : i32
    %7 = arith.muli %1, %c16_i32_2 : i32
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %9 = tt.splat %7 : i32 -> tensor<16xi32>
    %10 = arith.addi %9, %8 : tensor<16xi32>
    %11 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %12 = tt.splat %arg10 : i32 -> tensor<16x1xi32>
    %13 = arith.muli %11, %12 : tensor<16x1xi32>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %16 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %17 = tt.splat %arg11 : i32 -> tensor<1x16xi32>
    %18 = arith.muli %16, %17 : tensor<1x16xi32>
    %19 = tt.broadcast %15 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %20 = tt.broadcast %18 : tensor<1x16xi32> -> tensor<16x16xi32>
    %21 = tt.addptr %19, %20 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %22 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %23 = tt.splat %arg3 : i32 -> tensor<16x1xi32>
    %24 = arith.cmpi slt, %22, %23 : tensor<16x1xi32>
    %25 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %26 = tt.splat %arg4 : i32 -> tensor<1x16xi32>
    %27 = arith.cmpi slt, %25, %26 : tensor<1x16xi32>
    %28 = tt.broadcast %24 : tensor<16x1xi1> -> tensor<16x16xi1>
    %29 = tt.broadcast %27 : tensor<1x16xi1> -> tensor<16x16xi1>
    %30 = arith.andi %28, %29 : tensor<16x16xi1>
    %c16_i32_3 = arith.constant 16 : i32
    %c16_i32_4 = arith.constant 16 : i32
    %31 = arith.muli %2, %c16_i32_4 : i32
    %c32_i32 = arith.constant 32 : i32
    %32 = arith.bitcast %31 : i32 to i32
    %33 = arith.bitcast %arg5 : i32 to i32
    %34 = arith.bitcast %c32_i32 : i32 to i32
    %35 = ub.poison : i32
    scf.for %arg12 = %32 to %33 step %34  : i32 {
      %36 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %37 = tt.splat %arg12 : i32 -> tensor<16xi32>
      %38 = arith.addi %37, %36 : tensor<16xi32>
      %39 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %40 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
      %41 = arith.muli %39, %40 : tensor<16x1xi32>
      %42 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %43 = tt.addptr %42, %41 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %44 = tt.expand_dims %38 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %45 = tt.splat %arg7 : i32 -> tensor<1x16xi32>
      %46 = arith.muli %44, %45 : tensor<1x16xi32>
      %47 = tt.broadcast %43 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
      %48 = tt.broadcast %46 : tensor<1x16xi32> -> tensor<16x16xi32>
      %49 = tt.addptr %47, %48 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      %50 = tt.expand_dims %38 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %51 = tt.splat %arg8 : i32 -> tensor<16x1xi32>
      %52 = arith.muli %50, %51 : tensor<16x1xi32>
      %53 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %54 = tt.addptr %53, %52 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %55 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %56 = tt.splat %arg9 : i32 -> tensor<1x16xi32>
      %57 = arith.muli %55, %56 : tensor<1x16xi32>
      %58 = tt.broadcast %54 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
      %59 = tt.broadcast %57 : tensor<1x16xi32> -> tensor<16x16xi32>
      %60 = tt.addptr %58, %59 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      %61 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %62 = tt.splat %arg3 : i32 -> tensor<16x1xi32>
      %63 = arith.cmpi slt, %61, %62 : tensor<16x1xi32>
      %64 = tt.expand_dims %38 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %65 = tt.splat %arg5 : i32 -> tensor<1x16xi32>
      %66 = arith.cmpi slt, %64, %65 : tensor<1x16xi32>
      %67 = tt.broadcast %63 : tensor<16x1xi1> -> tensor<16x16xi1>
      %68 = tt.broadcast %66 : tensor<1x16xi1> -> tensor<16x16xi1>
      %69 = arith.andi %67, %68 : tensor<16x16xi1>
      %70 = tt.expand_dims %38 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %71 = tt.splat %arg5 : i32 -> tensor<16x1xi32>
      %72 = arith.cmpi slt, %70, %71 : tensor<16x1xi32>
      %73 = tt.expand_dims %10 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %74 = tt.splat %arg4 : i32 -> tensor<1x16xi32>
      %75 = arith.cmpi slt, %73, %74 : tensor<1x16xi32>
      %76 = tt.broadcast %72 : tensor<16x1xi1> -> tensor<16x16xi1>
      %77 = tt.broadcast %75 : tensor<1x16xi1> -> tensor<16x16xi1>
      %78 = arith.andi %76, %77 : tensor<16x16xi1>
      %c0_i32 = arith.constant 0 : i32
      %cst = arith.constant dense<0> : tensor<16x16xi32>
      %79 = arith.sitofp %cst : tensor<16x16xi32> to tensor<16x16xf32>
      %80 = tt.load %49, %69, %79 : tensor<16x16x!tt.ptr<f32>>
      %c0_i32_5 = arith.constant 0 : i32
      %cst_6 = arith.constant dense<0> : tensor<16x16xi32>
      %81 = arith.sitofp %cst_6 : tensor<16x16xi32> to tensor<16x16xf32>
      %82 = tt.load %60, %78, %81 : tensor<16x16x!tt.ptr<f32>>
      %cst_7 = arith.constant 0.000000e+00 : f32
      %cst_8 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
      %83 = tt.dot %80, %82, %cst_8 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
      %84 = tt.atomic_rmw fadd, acq_rel, gpu, %21, %83, %30 : (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xf32>, tensor<16x16xi1>) -> tensor<16x16xf32>
    }
    tt.return
  }
