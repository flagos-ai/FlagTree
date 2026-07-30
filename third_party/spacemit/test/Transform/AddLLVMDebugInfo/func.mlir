// RUN: triton-shared-opt --add-llvm-debug-info --mlir-print-debuginfo %s | FileCheck %s

#loc = loc("/path/to/program.py":9:0)
module {
  tt.func public @kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/path/to/program.py":9:0), %arg1: i32 loc("/path/to/program.py":9:0)) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<2xf32> loc(#loc1)
    %c2_i32 = arith.constant 2 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c2_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<2xi32> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<2xi32> loc(#loc5)
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>> loc(#loc6)
    %6 = tt.addptr %5, %4 : tensor<2x!tt.ptr<f32>>, tensor<2xi32> loc(#loc6)
    tt.store %6, %cst : tensor<2x!tt.ptr<f32>> loc(#loc7)
    tt.return loc(#loc8)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/path/to/program.py":10:24)
#loc3 = loc("/path/to/program.py":12:24)
#loc4 = loc("/path/to/program.py":13:41)
#loc5 = loc("/path/to/program.py":13:28)
#loc6 = loc("/path/to/program.py":25:26)
#loc7 = loc("/path/to/program.py":25:35)
#loc8 = loc("/path/to/program.py":25:4)

// CHECK: loc(#loc9)
// CHECK: #di_file = #llvm.di_file<"program.py" in "/path/to">
// CHECK: #di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>
// CHECK: #di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_Python, file = #di_file, producer = "MLIR", isOptimized = false, emissionKind = LineTablesOnly>
// CHECK: #di_subprogram = #llvm.di_subprogram<id = distinct[1]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "kernel", linkageName = "kernel", file = #di_file, line = 9, subprogramFlags = Definition, type = #di_subroutine_type>
// CHECK: #loc9 = loc(fused<#di_subprogram>[#loc])
