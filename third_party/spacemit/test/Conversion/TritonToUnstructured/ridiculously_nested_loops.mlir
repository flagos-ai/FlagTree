// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

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

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
