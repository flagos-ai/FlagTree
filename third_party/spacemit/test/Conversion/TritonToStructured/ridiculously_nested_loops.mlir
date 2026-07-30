// RUN: triton-shared-opt --triton-to-structured --canonicalize --remove-dead-values %s | FileCheck %s

module {
  tt.func public @nested_who_knows_how_many_levels(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22 = arith.muli %arg3, %c2_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x2xi32>
    %24:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %25 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %26:3 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6, %arg10 = %25) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
        %29 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.load %29 : tensor<2x2x!tt.ptr<f32>>
        %31:4 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %29, %arg13 = %arg9, %arg14 = %arg10, %arg15 = %30) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
          %33 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %34 = tt.load %33 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg13, %arg14 : tensor<2x2x!tt.ptr<f32>>
          %35 = tt.addptr %arg13, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %35, %arg15 : tensor<2x2x!tt.ptr<f32>>
          %36 = tt.addptr %35, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %36, %34 : tensor<2x2x!tt.ptr<f32>>
          %37 = tt.addptr %36, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %38:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %arg14, %arg18 = %33, %arg19 = %arg15, %arg20 = %34, %arg21 = %37) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:5 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21, %arg27 = %40) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:5 = scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg29 = %42, %arg30 = %arg25, %arg31 = %arg26, %arg32 = %arg27, %arg33 = %43) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                %45 = tt.addptr %arg29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg31, %arg32 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg31, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %arg33 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %50:5 = scf.for %arg34 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg35 = %arg32, %arg36 = %45, %arg37 = %arg33, %arg38 = %46, %arg39 = %49) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                  %51 = tt.load %arg36 : tensor<2x2x!tt.ptr<f32>>
                  %52:5 = scf.for %arg40 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg41 = %arg36, %arg42 = %arg37, %arg43 = %arg38, %arg44 = %arg39, %arg45 = %51) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
                    %53 = tt.addptr %arg41, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                    %54 = tt.load %53 : tensor<2x2x!tt.ptr<f32>>
                    %55:5 = scf.for %arg46 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg47 = %53, %arg48 = %arg43, %arg49 = %arg44, %arg50 = %arg45, %arg51 = %54) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                      %56 = tt.addptr %arg47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %57 = tt.load %56 : tensor<2x2x!tt.ptr<f32>>
                      tt.store %arg49, %arg50 : tensor<2x2x!tt.ptr<f32>>
                      %58 = tt.addptr %arg49, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %58, %arg51 : tensor<2x2x!tt.ptr<f32>>
                      %59 = tt.addptr %58, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %59, %57 : tensor<2x2x!tt.ptr<f32>>
                      %60 = tt.addptr %59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %61:5 = scf.for %arg52 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg53 = %arg50, %arg54 = %56, %arg55 = %arg51, %arg56 = %57, %arg57 = %60) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                        %62 = tt.load %arg54 : tensor<2x2x!tt.ptr<f32>>
                        %63:4 = scf.for %arg58 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg59 = %arg54, %arg60 = %arg55, %arg61 = %arg56, %arg62 = %arg57) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                          %64 = tt.addptr %arg59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                          %65 = tt.load %64 : tensor<2x2x!tt.ptr<f32>>
                          %66:3 = scf.for %arg63 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg64 = %64, %arg65 = %arg61, %arg66 = %arg62) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                            %67 = tt.addptr %arg64, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            %68 = tt.load %67 : tensor<2x2x!tt.ptr<f32>>
                            tt.store %arg66, %62 : tensor<2x2x!tt.ptr<f32>>
                            %69 = tt.addptr %arg66, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %69, %65 : tensor<2x2x!tt.ptr<f32>>
                            %70 = tt.addptr %69, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %70, %68 : tensor<2x2x!tt.ptr<f32>>
                            %71 = tt.addptr %70, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            scf.yield %67, %68, %71 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                          }
                          scf.yield %66#0, %65, %66#1, %66#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                        }
                        scf.yield %62, %63#0, %63#1, %63#2, %63#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                      }
                      scf.yield %61#1, %61#3, %61#4, %61#0, %61#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
                    }
                    scf.yield %55#0, %55#4, %55#1, %55#2, %55#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
                  }
                  scf.yield %52#4, %52#0, %52#1, %52#2, %52#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                }
                scf.yield %50#1, %50#3, %50#4, %50#0, %50#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
              }
              scf.yield %44#0, %44#4, %44#1, %44#2, %44#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
            }
            scf.yield %41#4, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          %39:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %38#0, %arg18 = %38#1, %arg19 = %38#2, %arg20 = %38#3, %arg21 = %38#4) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:4 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:3 = scf.for %arg27 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg28 = %42, %arg29 = %arg25, %arg30 = %arg26) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                %45 = tt.addptr %arg28, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg30, %40 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %43 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                scf.yield %45, %46, %49 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
              }
              scf.yield %44#0, %43, %44#1, %44#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
            }
            scf.yield %40, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %39#1, %39#4, %39#0, %39#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
        }
        %32 = tt.addptr %31#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31#1, %31#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
      }
      %27:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %26#0, %arg9 = %26#1) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %29 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %30:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %32 = tt.addptr %arg11, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %33 = tt.load %32 : tensor<2x2x!tt.ptr<f32>>
          %34:2 = scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg14 = %32, %arg15 = %arg12) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %35 = tt.addptr %arg14, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            %36 = tt.load %35 : tensor<2x2x!tt.ptr<f32>>
            tt.store %arg15, %29 : tensor<2x2x!tt.ptr<f32>>
            %37 = tt.addptr %arg15, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %37, %33 : tensor<2x2x!tt.ptr<f32>>
            %38 = tt.addptr %37, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %38, %36 : tensor<2x2x!tt.ptr<f32>>
            %39 = tt.addptr %38, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            scf.yield %35, %39 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %34#0, %34#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        %31 = tt.addptr %30#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %31, %30#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %28 = tt.addptr %27#0, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %28, %27#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:         tt.func public @nested_who_knows_how_many_levels([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[arg3_]], [[CST_2_]] : i32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.muli [[arg3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.muli [[arg3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = arith.index_cast [[VAR_37_]] : i32 to index
// CHECK-DAG:       [[VAR_39_:%.+]] = arith.muli [[arg3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_40_:%.+]] = arith.index_cast [[VAR_39_]] : i32 to index
// CHECK-DAG:       [[VAR_41_:%.+]]:2 = scf.for [[VAR_arg4_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg5_:%.+]] = [[CST_0_]], [[VAR_arg6_:%.+]] = [[CST_0_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_42_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg5_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:             [[VAR_43_:%.+]] = "tts.load"([[VAR_42_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:         [[VAR_44_:%.+]]:3 = scf.for [[VAR_arg7_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg8_:%.+]] = [[VAR_arg5_]], [[VAR_arg9_:%.+]] = [[VAR_arg6_]], [[VAR_arg10_:%.+]] = [[VAR_43_]]) -> (index, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.addi [[VAR_arg8_]], [[VAR_30_]] : index
// CHECK:               [[VAR_48_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_47_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               [[VAR_49_:%.+]] = "tts.load"([[VAR_48_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:           [[VAR_50_:%.+]]:4 = scf.for [[VAR_arg11_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg12_:%.+]] = [[VAR_47_]], [[VAR_arg13_:%.+]] = [[VAR_arg9_]], [[VAR_arg14_:%.+]] = [[VAR_arg10_]], [[VAR_arg15_:%.+]] = [[VAR_49_]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:             [[VAR_52_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg13_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[VAR_53_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_29_]] : index
// CHECK:                 [[VAR_54_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_53_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 [[VAR_55_:%.+]] = "tts.load"([[VAR_54_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                 "tts.store"([[VAR_52_]], [[VAR_arg14_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_56_:%.+]] = arith.addi [[VAR_arg13_]], [[VAR_28_]] : index
// CHECK:                 [[VAR_57_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_56_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"([[VAR_57_]], [[VAR_arg15_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_58_:%.+]] = arith.addi [[VAR_56_]], [[VAR_27_]] : index
// CHECK:                 [[VAR_59_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_58_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"([[VAR_59_]], [[VAR_55_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_60_:%.+]] = arith.addi [[VAR_58_]], [[VAR_26_]] : index
// CHECK-DAG:             [[VAR_61_:%.+]]:4 = scf.for [[VAR_arg16_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg17_:%.+]] = [[VAR_arg14_]], [[VAR_arg18_:%.+]] = [[VAR_53_]], [[VAR_arg19_:%.+]] = [[VAR_arg15_]], [[VAR_arg20_:%.+]] = [[VAR_60_]]) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:               [[VAR_63_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg18_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   [[VAR_64_:%.+]] = "tts.load"([[VAR_63_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:               [[VAR_65_:%.+]]:4 = scf.for [[VAR_arg21_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg22_:%.+]] = [[VAR_arg18_]], [[VAR_arg23_:%.+]] = [[VAR_arg19_]], [[VAR_arg24_:%.+]] = [[VAR_arg20_]], [[VAR_arg25_:%.+]] = [[VAR_64_]]) -> (index, tensor<2x2xf32>, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                 [[VAR_66_:%.+]] = arith.addi [[VAR_arg22_]], [[VAR_25_]] : index
// CHECK:                     [[VAR_67_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_66_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                     [[VAR_68_:%.+]] = "tts.load"([[VAR_67_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                 [[VAR_69_:%.+]]:4 = scf.for [[VAR_arg26_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg27_:%.+]] = [[VAR_66_]], [[VAR_arg28_:%.+]] = [[VAR_arg24_]], [[VAR_arg29_:%.+]] = [[VAR_arg25_]], [[VAR_arg30_:%.+]] = [[VAR_68_]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                   [[VAR_70_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg28_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                   [[VAR_71_:%.+]] = arith.addi [[VAR_arg27_]], [[VAR_24_]] : index
// CHECK:                       [[VAR_72_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_71_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       [[VAR_73_:%.+]] = "tts.load"([[VAR_72_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                       "tts.store"([[VAR_70_]], [[VAR_arg29_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_74_:%.+]] = arith.addi [[VAR_arg28_]], [[VAR_23_]] : index
// CHECK:                       [[VAR_75_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_74_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"([[VAR_75_]], [[VAR_arg30_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_76_:%.+]] = arith.addi [[VAR_74_]], [[VAR_22_]] : index
// CHECK:                       [[VAR_77_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_76_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"([[VAR_77_]], [[VAR_73_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_78_:%.+]] = arith.addi [[VAR_76_]], [[VAR_21_]] : index
// CHECK-DAG:                   [[VAR_79_:%.+]]:4 = scf.for [[VAR_arg31_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg32_:%.+]] = [[VAR_arg29_]], [[VAR_arg33_:%.+]] = [[VAR_71_]], [[VAR_arg34_:%.+]] = [[VAR_arg30_]], [[VAR_arg35_:%.+]] = [[VAR_78_]]) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:                     [[VAR_80_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg33_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                         [[VAR_81_:%.+]] = "tts.load"([[VAR_80_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                     [[VAR_82_:%.+]]:4 = scf.for [[VAR_arg36_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg37_:%.+]] = [[VAR_arg33_]], [[VAR_arg38_:%.+]] = [[VAR_arg34_]], [[VAR_arg39_:%.+]] = [[VAR_arg35_]], [[VAR_arg40_:%.+]] = [[VAR_81_]]) -> (index, tensor<2x2xf32>, index, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                       [[VAR_83_:%.+]] = arith.addi [[VAR_arg37_]], [[VAR_20_]] : index
// CHECK:                           [[VAR_84_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_83_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                           [[VAR_85_:%.+]] = "tts.load"([[VAR_84_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                       [[VAR_86_:%.+]]:4 = scf.for [[VAR_arg41_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg42_:%.+]] = [[VAR_83_]], [[VAR_arg43_:%.+]] = [[VAR_arg39_]], [[VAR_arg44_:%.+]] = [[VAR_arg40_]], [[VAR_arg45_:%.+]] = [[VAR_85_]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK-DAG:                         [[VAR_87_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg43_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                         [[VAR_88_:%.+]] = arith.addi [[VAR_arg42_]], [[VAR_19_]] : index
// CHECK:                             [[VAR_89_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_88_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             [[VAR_90_:%.+]] = "tts.load"([[VAR_89_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                             "tts.store"([[VAR_87_]], [[VAR_arg44_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             [[VAR_91_:%.+]] = arith.addi [[VAR_arg43_]], [[VAR_18_]] : index
// CHECK:                             [[VAR_92_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_91_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             "tts.store"([[VAR_92_]], [[VAR_arg45_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             [[VAR_93_:%.+]] = arith.addi [[VAR_91_]], [[VAR_17_]] : index
// CHECK:                             [[VAR_94_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_93_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             "tts.store"([[VAR_94_]], [[VAR_90_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             [[VAR_95_:%.+]] = arith.addi [[VAR_93_]], [[VAR_16_]] : index
// CHECK-DAG:                         [[VAR_96_:%.+]]:4 = scf.for [[VAR_arg46_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg47_:%.+]] = [[VAR_arg44_]], [[VAR_arg48_:%.+]] = [[VAR_88_]], [[VAR_arg49_:%.+]] = [[VAR_arg45_]], [[VAR_arg50_:%.+]] = [[VAR_95_]]) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:                           [[VAR_97_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg48_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                           [[VAR_98_:%.+]] = "tts.load"([[VAR_97_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                           [[VAR_99_:%.+]]:3 = scf.for [[VAR_arg51_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg52_:%.+]] = [[VAR_arg48_]], [[VAR_arg53_:%.+]] = [[VAR_arg49_]], [[VAR_arg54_:%.+]] = [[VAR_arg50_]]) -> (index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:                             [[VAR_100_:%.+]] = arith.addi [[VAR_arg52_]], [[VAR_15_]] : index
// CHECK:                                 [[VAR_101_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_100_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                             [[VAR_102_:%.+]] = "tts.load"([[VAR_101_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                             [[VAR_103_:%.+]]:2 = scf.for [[VAR_arg55_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg56_:%.+]] = [[VAR_100_]], [[VAR_arg57_:%.+]] = [[VAR_arg54_]]) -> (index, index)  : i32 {
// CHECK-DAG:                               [[VAR_104_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg57_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                               [[VAR_105_:%.+]] = arith.addi [[VAR_arg56_]], [[VAR_14_]] : index
// CHECK:                                   [[VAR_106_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_105_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   [[VAR_107_:%.+]] = "tts.load"([[VAR_106_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                                   "tts.store"([[VAR_104_]], [[VAR_98_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   [[VAR_108_:%.+]] = arith.addi [[VAR_arg57_]], [[VAR_13_]] : index
// CHECK:                                   [[VAR_109_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_108_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   "tts.store"([[VAR_109_]], [[VAR_102_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   [[VAR_110_:%.+]] = arith.addi [[VAR_108_]], [[VAR_12_]] : index
// CHECK:                                   [[VAR_111_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_110_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   "tts.store"([[VAR_111_]], [[VAR_107_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   [[VAR_112_:%.+]] = arith.addi [[VAR_110_]], [[VAR_11_]] : index
// CHECK:                                   scf.yield [[VAR_105_]], [[VAR_112_]] : index, index
// CHECK:                                 }
// CHECK:                                 scf.yield [[VAR_103_]]#0, [[VAR_102_]], [[VAR_103_]]#1 : index, tensor<2x2xf32>, index
// CHECK:                               }
// CHECK:                               scf.yield [[VAR_98_]], [[VAR_99_]]#0, [[VAR_99_]]#1, [[VAR_99_]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                             }
// CHECK:                             scf.yield [[VAR_96_]]#1, [[VAR_96_]]#3, [[VAR_96_]]#0, [[VAR_96_]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                           }
// CHECK:                           scf.yield [[VAR_86_]]#0, [[VAR_86_]]#3, [[VAR_86_]]#1, [[VAR_86_]]#2 : index, tensor<2x2xf32>, index, tensor<2x2xf32>
// CHECK:                         }
// CHECK:                         scf.yield [[VAR_82_]]#3, [[VAR_82_]]#0, [[VAR_82_]]#1, [[VAR_82_]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                       }
// CHECK:                       scf.yield [[VAR_79_]]#1, [[VAR_79_]]#3, [[VAR_79_]]#0, [[VAR_79_]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                     }
// CHECK:                     scf.yield [[VAR_69_]]#0, [[VAR_69_]]#3, [[VAR_69_]]#1, [[VAR_69_]]#2 : index, tensor<2x2xf32>, index, tensor<2x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield [[VAR_65_]]#3, [[VAR_65_]]#0, [[VAR_65_]]#1, [[VAR_65_]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                 }
// CHECK-DAG:             [[VAR_62_:%.+]]:4 = scf.for [[VAR_arg16_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg17_1_:%.+]] = [[VAR_61_]]#0, [[VAR_arg18_1_:%.+]] = [[VAR_61_]]#1, [[VAR_arg19_1_:%.+]] = [[VAR_61_]]#2, [[VAR_arg20_1_:%.+]] = [[VAR_61_]]#3) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:               [[VAR_63_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg18_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:               [[VAR_64_1_:%.+]] = "tts.load"([[VAR_63_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:               [[VAR_65_1_:%.+]]:3 = scf.for [[VAR_arg21_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg22_1_:%.+]] = [[VAR_arg18_1_]], [[VAR_arg23_1_:%.+]] = [[VAR_arg19_1_]], [[VAR_arg24_1_:%.+]] = [[VAR_arg20_1_]]) -> (index, tensor<2x2xf32>, index)  : i32 {
// CHECK-DAG:                 [[VAR_66_1_:%.+]] = arith.addi [[VAR_arg22_1_]], [[VAR_10_]] : index
// CHECK:                     [[VAR_67_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_66_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                 [[VAR_68_1_:%.+]] = "tts.load"([[VAR_67_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:                 [[VAR_69_1_:%.+]]:2 = scf.for [[VAR_arg25_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg26_1_:%.+]] = [[VAR_66_1_]], [[VAR_arg27_1_:%.+]] = [[VAR_arg24_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:                   [[VAR_70_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg27_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:                   [[VAR_71_1_:%.+]] = arith.addi [[VAR_arg26_1_]], [[VAR_9_]] : index
// CHECK:                       [[VAR_72_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_71_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       [[VAR_73_1_:%.+]] = "tts.load"([[VAR_72_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                       "tts.store"([[VAR_70_1_]], [[VAR_64_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_74_1_:%.+]] = arith.addi [[VAR_arg27_1_]], [[VAR_8_]] : index
// CHECK:                       [[VAR_75_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_74_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"([[VAR_75_1_]], [[VAR_68_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_76_1_:%.+]] = arith.addi [[VAR_74_1_]], [[VAR_7_]] : index
// CHECK:                       [[VAR_77_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_76_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"([[VAR_77_1_]], [[VAR_73_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       [[VAR_78_1_:%.+]] = arith.addi [[VAR_76_1_]], [[VAR_6_]] : index
// CHECK:                       scf.yield [[VAR_71_1_]], [[VAR_78_1_]] : index, index
// CHECK:                     }
// CHECK:                     scf.yield [[VAR_69_1_]]#0, [[VAR_68_1_]], [[VAR_69_1_]]#1 : index, tensor<2x2xf32>, index
// CHECK:                   }
// CHECK:                   scf.yield [[VAR_64_1_]], [[VAR_65_1_]]#0, [[VAR_65_1_]]#1, [[VAR_65_1_]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                 }
// CHECK:                 scf.yield [[VAR_62_]]#1, [[VAR_62_]]#3, [[VAR_62_]]#0, [[VAR_62_]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:               }
// CHECK:               [[VAR_51_:%.+]] = arith.addi [[VAR_50_]]#0, [[VAR_5_]] : index
// CHECK:               scf.yield [[VAR_51_]], [[VAR_50_]]#1, [[VAR_50_]]#2 : index, index, tensor<2x2xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_45_:%.+]]:2 = scf.for [[VAR_arg7_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg8_1_:%.+]] = [[VAR_44_]]#0, [[VAR_arg9_1_:%.+]] = [[VAR_44_]]#1) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_47_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg8_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_1_:%.+]] = "tts.load"([[VAR_47_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:           [[VAR_49_1_:%.+]]:2 = scf.for [[VAR_arg10_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg11_1_:%.+]] = [[VAR_arg8_1_]], [[VAR_arg12_1_:%.+]] = [[VAR_arg9_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:             [[VAR_51_1_:%.+]] = arith.addi [[VAR_arg11_1_]], [[VAR_36_]] : index
// CHECK:                 [[VAR_52_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_51_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[VAR_53_1_:%.+]] = "tts.load"([[VAR_52_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:             [[VAR_54_1_:%.+]]:2 = scf.for [[VAR_arg13_1_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg14_1_:%.+]] = [[VAR_51_1_]], [[VAR_arg15_1_:%.+]] = [[VAR_arg12_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:               [[VAR_55_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg15_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:               [[VAR_56_1_:%.+]] = arith.addi [[VAR_arg14_1_]], [[VAR_35_]] : index
// CHECK:                   [[VAR_57_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_56_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   [[VAR_58_1_:%.+]] = "tts.load"([[VAR_57_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                   "tts.store"([[VAR_55_1_]], [[VAR_48_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   [[VAR_59_1_:%.+]] = arith.addi [[VAR_arg15_1_]], [[VAR_34_]] : index
// CHECK:                   [[VAR_60_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_59_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   "tts.store"([[VAR_60_1_]], [[VAR_53_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   [[VAR_61_1_:%.+]] = arith.addi [[VAR_59_1_]], [[VAR_33_]] : index
// CHECK:                   [[VAR_62_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_61_1_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   "tts.store"([[VAR_62_1_]], [[VAR_58_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   [[VAR_63_2_:%.+]] = arith.addi [[VAR_61_1_]], [[VAR_32_]] : index
// CHECK:                   scf.yield [[VAR_56_1_]], [[VAR_63_2_]] : index, index
// CHECK:                 }
// CHECK:                 scf.yield [[VAR_54_1_]]#0, [[VAR_54_1_]]#1 : index, index
// CHECK:               }
// CHECK:               [[VAR_50_1_:%.+]] = arith.addi [[VAR_49_1_]]#0, [[VAR_38_]] : index
// CHECK:               scf.yield [[VAR_50_1_]], [[VAR_49_1_]]#1 : index, index
// CHECK:             }
// CHECK:             [[VAR_46_:%.+]] = arith.addi [[VAR_45_]]#0, [[VAR_40_]] : index
// CHECK:             scf.yield [[VAR_46_]], [[VAR_45_]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
