// RUN: triton-opt --pass-pipeline="builtin.module(auto-blockify{auto-blockify-size=1},triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},discrete-mask-access-conversion{compile-on-910-95=false compile-mode=simd_simt_template},triton-to-annotation,triton-to-unstructure{compile-on-910-95=false compile-mode=simd_simt_template},triton-to-hivm,triton-to-hfusion,triton-to-llvm,bubble-up-operation,triton-to-structured{enable-mask-fallback-conversion=false optimize-dynamic-offset=false},triton-to-linalg{compile-on-910-95=false enable-nd2nz-on-vector=false enable-select-analysis=true global-kernel=false named-ops=true compile-mode=simd_simt_template})" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @kernel_B8_dynamic_offset
// CHECK: %[[PIDOFF:.*]] = arith.muli
// CHECK: scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} {
// CHECK:   %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:   arith.addi %[[PIDOFF]], %[[IV32]] : i32
// CHECK:   func.call @triton_print{{[_0-9]*}}({{.*}}) : (i32) -> ()
// CHECK: }

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B8_dynamic_offset(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c8_i32 : i32
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.splat %1 : i32 -> tensor<8xi32>
    %4 = arith.addi %3, %2 : tensor<8xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %7 = tt.load %6 : tensor<8x!tt.ptr<i32>>
    tt.print " idx_B8: " {hex = false, isSigned = array<i32: 1>} : %4 : tensor<8xi32>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %9 = tt.addptr %8, %4 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %9, %7 : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL:  func.func private @triton_print{{[_0-9]*}}(i32)
// CHECK-LABEL:  func.func @triton_max_6d_dimm1
// CHECK-DAG:    %[[C31744:.*]] = arith.constant 31744 : index
// CHECK-DAG:    %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG:    %[[C256:.*]] = arith.constant 256 : index
// CHECK:        scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:          scf.for %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:            scf.for %[[IV2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:              scf.for %[[IV3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:                %{{.*}} = arith.muli %[[IV0]], %[[C31744]] : index
// CHECK:                %{{.*}} = arith.muli %[[IV1]], %[[C1024]] : index
// CHECK:                %{{.*}} = arith.addi
// CHECK:                %{{.*}} = arith.muli %[[IV2]], %[[C256]] : index
// CHECK:                %{{.*}} = arith.addi
// CHECK:                %{{.*}} = arith.addi %{{.*}}, %[[IV3]] : index
// CHECK:                %{{.*}} = arith.index_cast %{{.*}} : index to i32
// CHECK:                func.call @triton_print{{[_0-9]*}}(%{{.*}}) : (i32) -> ()
// CHECK:              }
// CHECK:            }
// CHECK:          }
// CHECK:        }

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @triton_max_6d_dimm1(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<256> : tensor<1x1x4x1x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x31x1x1x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<1x31x1x1x1xi32>
    %cst_2 = arith.constant dense<256> : tensor<4x1x1x1x1xi32>
    %cst_3 = arith.constant dense<4> : tensor<4x1x1x1x1xi32>
    %cst_4 = arith.constant dense<31> : tensor<4x1x1x1x1xi32>
    %cst_5 = arith.constant dense<256> : tensor<1x1x4x1x1x1xi32>
    %cst_6 = arith.constant dense<256> : tensor<1x31x1x1x1x1xi32>
    %cst_7 = arith.constant dense<4> : tensor<1x31x1x1x1x1xi32>
    %cst_8 = arith.constant dense<256> : tensor<4x1x1x1x1x1xi32>
    %cst_9 = arith.constant dense<4> : tensor<4x1x1x1x1x1xi32>
    %cst_10 = arith.constant dense<31> : tensor<4x1x1x1x1x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 31 : i32, start = 0 : i32} : tensor<31xi32>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %5 = tt.expand_dims %4 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %6 = tt.expand_dims %5 {axis = 4 : i32} : tensor<4x1x1x1xi32> -> tensor<4x1x1x1x1xi32>
    %7 = tt.expand_dims %6 {axis = 5 : i32} : tensor<4x1x1x1x1xi32> -> tensor<4x1x1x1x1x1xi32>
    %8 = arith.muli %7, %cst_10 : tensor<4x1x1x1x1x1xi32>
    %9 = arith.muli %8, %cst_9 : tensor<4x1x1x1x1x1xi32>
    %10 = arith.muli %9, %cst_8 : tensor<4x1x1x1x1x1xi32>
    %11 = tt.expand_dims %1 {axis = 0 : i32} : tensor<31xi32> -> tensor<1x31xi32>
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<1x31xi32> -> tensor<1x31x1xi32>
    %13 = tt.expand_dims %12 {axis = 3 : i32} : tensor<1x31x1xi32> -> tensor<1x31x1x1xi32>
    %14 = tt.expand_dims %13 {axis = 4 : i32} : tensor<1x31x1x1xi32> -> tensor<1x31x1x1x1xi32>
    %15 = tt.expand_dims %14 {axis = 5 : i32} : tensor<1x31x1x1x1xi32> -> tensor<1x31x1x1x1x1xi32>
    %16 = arith.muli %15, %cst_7 : tensor<1x31x1x1x1x1xi32>
    %17 = arith.muli %16, %cst_6 : tensor<1x31x1x1x1x1xi32>
    %18 = tt.broadcast %10 : tensor<4x1x1x1x1x1xi32> -> tensor<4x31x1x1x1x1xi32>
    %19 = tt.broadcast %17 : tensor<1x31x1x1x1x1xi32> -> tensor<4x31x1x1x1x1xi32>
    %20 = arith.addi %18, %19 : tensor<4x31x1x1x1x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %23 = tt.expand_dims %22 {axis = 3 : i32} : tensor<1x1x4xi32> -> tensor<1x1x4x1xi32>
    %24 = tt.expand_dims %23 {axis = 4 : i32} : tensor<1x1x4x1xi32> -> tensor<1x1x4x1x1xi32>
    %25 = tt.expand_dims %24 {axis = 5 : i32} : tensor<1x1x4x1x1xi32> -> tensor<1x1x4x1x1x1xi32>
    %26 = arith.muli %25, %cst_5 : tensor<1x1x4x1x1x1xi32>
    %27 = tt.broadcast %20 : tensor<4x31x1x1x1x1xi32> -> tensor<4x31x4x1x1x1xi32>
    %28 = tt.broadcast %26 : tensor<1x1x4x1x1x1xi32> -> tensor<4x31x4x1x1x1xi32>
    %29 = arith.addi %27, %28 : tensor<4x31x4x1x1x1xi32>
    %30 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<1x256xi32> -> tensor<1x1x256xi32>
    %32 = tt.expand_dims %31 {axis = 2 : i32} : tensor<1x1x256xi32> -> tensor<1x1x1x256xi32>
    %33 = tt.expand_dims %32 {axis = 4 : i32} : tensor<1x1x1x256xi32> -> tensor<1x1x1x256x1xi32>
    %34 = tt.expand_dims %33 {axis = 5 : i32} : tensor<1x1x1x256x1xi32> -> tensor<1x1x1x256x1x1xi32>
    %35 = tt.broadcast %29 : tensor<4x31x4x1x1x1xi32> -> tensor<4x31x4x256x1x1xi32>
    %36 = tt.broadcast %34 : tensor<1x1x1x256x1x1xi32> -> tensor<4x31x4x256x1x1xi32>
    %37 = arith.addi %35, %36 : tensor<4x31x4x256x1x1xi32>
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %37 : tensor<4x31x4x256x1x1xi32>
    %38 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x31x4x256x1x1x!tt.ptr<i8>>
    %39 = tt.addptr %38, %37 : tensor<4x31x4x256x1x1x!tt.ptr<i8>>, tensor<4x31x4x256x1x1xi32>
    %40 = tt.load %39 : tensor<4x31x4x256x1x1x!tt.ptr<i8>>
    tt.assert %true, "Expecting input to be integer type" : i1
    %41 = "tt.reduce"(%40) <{axis = 5 : i32}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %59 = arith.maxsi %arg2, %arg3 : i8
      tt.reduce.return %59 : i8
    }) : (tensor<4x31x4x256x1x1xi8>) -> tensor<4x31x4x256x1xi8>
    %42 = arith.muli %6, %cst_4 : tensor<4x1x1x1x1xi32>
    %43 = arith.muli %42, %cst_3 : tensor<4x1x1x1x1xi32>
    %44 = arith.muli %43, %cst_2 : tensor<4x1x1x1x1xi32>
    %45 = arith.muli %14, %cst_1 : tensor<1x31x1x1x1xi32>
    %46 = arith.muli %45, %cst_0 : tensor<1x31x1x1x1xi32>
    %47 = tt.broadcast %44 : tensor<4x1x1x1x1xi32> -> tensor<4x31x1x1x1xi32>
    %48 = tt.broadcast %46 : tensor<1x31x1x1x1xi32> -> tensor<4x31x1x1x1xi32>
    %49 = arith.addi %47, %48 : tensor<4x31x1x1x1xi32>
    %50 = arith.muli %24, %cst : tensor<1x1x4x1x1xi32>
    %51 = tt.broadcast %49 : tensor<4x31x1x1x1xi32> -> tensor<4x31x4x1x1xi32>
    %52 = tt.broadcast %50 : tensor<1x1x4x1x1xi32> -> tensor<4x31x4x1x1xi32>
    %53 = arith.addi %51, %52 : tensor<4x31x4x1x1xi32>
    %54 = tt.broadcast %53 : tensor<4x31x4x1x1xi32> -> tensor<4x31x4x256x1xi32>
    %55 = tt.broadcast %33 : tensor<1x1x1x256x1xi32> -> tensor<4x31x4x256x1xi32>
    %56 = arith.addi %54, %55 : tensor<4x31x4x256x1xi32>
    %57 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x31x4x256x1x!tt.ptr<i8>>
    %58 = tt.addptr %57, %56 : tensor<4x31x4x256x1x!tt.ptr<i8>>, tensor<4x31x4x256x1xi32>
    tt.store %58, %41 : tensor<4x31x4x256x1x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @triton_max_6d_dimm1
// CHECK-DAG:   %[[C95232:.*]] = arith.constant 95232 : index
// CHECK-DAG:   %[[C3072:.*]] = arith.constant 3072 : index
// CHECK-DAG:   %[[C768:.*]] = arith.constant 768 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         scf.for %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           scf.for %[[IV2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             scf.for %[[IV3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %{{.*}} = arith.muli %[[IV0]], %[[C95232]] : index
// CHECK:               %{{.*}} = arith.muli %[[IV1]], %[[C3072]] : index
// CHECK:               %{{.*}} = arith.addi
// CHECK:               %{{.*}} = arith.muli %[[IV2]], %[[C768]] : index
// CHECK:               %{{.*}} = arith.addi
// CHECK:               %{{.*}} = arith.muli %[[IV3]], %[[C3]] : index
// CHECK:               %{{.*}} = arith.addi
// CHECK:               %{{.*}} = arith.index_cast {{.*}} : index to i32
// CHECK:               func.call @triton_print{{[_0-9]*}}({{.*}}) : (i32) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @triton_max_6d_dimm1(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<256> : tensor<1x1x4x1x1xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x31x1x1x1xi32>
    %cst_1 = arith.constant dense<4> : tensor<1x31x1x1x1xi32>
    %cst_2 = arith.constant dense<256> : tensor<4x1x1x1x1xi32>
    %cst_3 = arith.constant dense<4> : tensor<4x1x1x1x1xi32>
    %cst_4 = arith.constant dense<31> : tensor<4x1x1x1x1xi32>
    %cst_5 = arith.constant dense<3> : tensor<4x31x4x256x1x1xi32>
    %cst_6 = arith.constant dense<256> : tensor<1x1x4x1x1x1xi32>
    %cst_7 = arith.constant dense<256> : tensor<1x31x1x1x1x1xi32>
    %cst_8 = arith.constant dense<4> : tensor<1x31x1x1x1x1xi32>
    %cst_9 = arith.constant dense<256> : tensor<4x1x1x1x1x1xi32>
    %cst_10 = arith.constant dense<4> : tensor<4x1x1x1x1x1xi32>
    %cst_11 = arith.constant dense<31> : tensor<4x1x1x1x1x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 31 : i32, start = 0 : i32} : tensor<31xi32>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %5 = tt.expand_dims %4 {axis = 3 : i32} : tensor<4x1x1xi32> -> tensor<4x1x1x1xi32>
    %6 = tt.expand_dims %5 {axis = 4 : i32} : tensor<4x1x1x1xi32> -> tensor<4x1x1x1x1xi32>
    %7 = tt.expand_dims %6 {axis = 5 : i32} : tensor<4x1x1x1x1xi32> -> tensor<4x1x1x1x1x1xi32>
    %8 = arith.muli %7, %cst_11 : tensor<4x1x1x1x1x1xi32>
    %9 = arith.muli %8, %cst_10 : tensor<4x1x1x1x1x1xi32>
    %10 = arith.muli %9, %cst_9 : tensor<4x1x1x1x1x1xi32>
    %11 = tt.expand_dims %1 {axis = 0 : i32} : tensor<31xi32> -> tensor<1x31xi32>
    %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<1x31xi32> -> tensor<1x31x1xi32>
    %13 = tt.expand_dims %12 {axis = 3 : i32} : tensor<1x31x1xi32> -> tensor<1x31x1x1xi32>
    %14 = tt.expand_dims %13 {axis = 4 : i32} : tensor<1x31x1x1xi32> -> tensor<1x31x1x1x1xi32>
    %15 = tt.expand_dims %14 {axis = 5 : i32} : tensor<1x31x1x1x1xi32> -> tensor<1x31x1x1x1x1xi32>
    %16 = arith.muli %15, %cst_8 : tensor<1x31x1x1x1x1xi32>
    %17 = arith.muli %16, %cst_7 : tensor<1x31x1x1x1x1xi32>
    %18 = tt.broadcast %10 : tensor<4x1x1x1x1x1xi32> -> tensor<4x31x1x1x1x1xi32>
    %19 = tt.broadcast %17 : tensor<1x31x1x1x1x1xi32> -> tensor<4x31x1x1x1x1xi32>
    %20 = arith.addi %18, %19 : tensor<4x31x1x1x1x1xi32>
    %21 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
    %23 = tt.expand_dims %22 {axis = 3 : i32} : tensor<1x1x4xi32> -> tensor<1x1x4x1xi32>
    %24 = tt.expand_dims %23 {axis = 4 : i32} : tensor<1x1x4x1xi32> -> tensor<1x1x4x1x1xi32>
    %25 = tt.expand_dims %24 {axis = 5 : i32} : tensor<1x1x4x1x1xi32> -> tensor<1x1x4x1x1x1xi32>
    %26 = arith.muli %25, %cst_6 : tensor<1x1x4x1x1x1xi32>
    %27 = tt.broadcast %20 : tensor<4x31x1x1x1x1xi32> -> tensor<4x31x4x1x1x1xi32>
    %28 = tt.broadcast %26 : tensor<1x1x4x1x1x1xi32> -> tensor<4x31x4x1x1x1xi32>
    %29 = arith.addi %27, %28 : tensor<4x31x4x1x1x1xi32>
    %30 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %31 = tt.expand_dims %30 {axis = 1 : i32} : tensor<1x256xi32> -> tensor<1x1x256xi32>
    %32 = tt.expand_dims %31 {axis = 2 : i32} : tensor<1x1x256xi32> -> tensor<1x1x1x256xi32>
    %33 = tt.expand_dims %32 {axis = 4 : i32} : tensor<1x1x1x256xi32> -> tensor<1x1x1x256x1xi32>
    %34 = tt.expand_dims %33 {axis = 5 : i32} : tensor<1x1x1x256x1xi32> -> tensor<1x1x1x256x1x1xi32>
    %35 = tt.broadcast %29 : tensor<4x31x4x1x1x1xi32> -> tensor<4x31x4x256x1x1xi32>
    %36 = tt.broadcast %34 : tensor<1x1x1x256x1x1xi32> -> tensor<4x31x4x256x1x1xi32>
    %37 = arith.addi %35, %36 : tensor<4x31x4x256x1x1xi32>
    %38 = arith.muli %37, %cst_5 : tensor<4x31x4x256x1x1xi32>
    tt.print ": " {hex = false, isSigned = array<i32: 1>} : %38 : tensor<4x31x4x256x1x1xi32>
    %39 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<4x31x4x256x1x1x!tt.ptr<i8>>
    %40 = tt.addptr %39, %38 : tensor<4x31x4x256x1x1x!tt.ptr<i8>>, tensor<4x31x4x256x1x1xi32>
    %41 = tt.load %40 : tensor<4x31x4x256x1x1x!tt.ptr<i8>>
    tt.assert %true, "Expecting input to be integer type" : i1
    %42 = "tt.reduce"(%41) <{axis = 5 : i32}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %60 = arith.maxsi %arg2, %arg3 : i8
      tt.reduce.return %60 : i8
    }) : (tensor<4x31x4x256x1x1xi8>) -> tensor<4x31x4x256x1xi8>
    %43 = arith.muli %6, %cst_4 : tensor<4x1x1x1x1xi32>
    %44 = arith.muli %43, %cst_3 : tensor<4x1x1x1x1xi32>
    %45 = arith.muli %44, %cst_2 : tensor<4x1x1x1x1xi32>
    %46 = arith.muli %14, %cst_1 : tensor<1x31x1x1x1xi32>
    %47 = arith.muli %46, %cst_0 : tensor<1x31x1x1x1xi32>
    %48 = tt.broadcast %45 : tensor<4x1x1x1x1xi32> -> tensor<4x31x1x1x1xi32>
    %49 = tt.broadcast %47 : tensor<1x31x1x1x1xi32> -> tensor<4x31x1x1x1xi32>
    %50 = arith.addi %48, %49 : tensor<4x31x1x1x1xi32>
    %51 = arith.muli %24, %cst : tensor<1x1x4x1x1xi32>
    %52 = tt.broadcast %50 : tensor<4x31x1x1x1xi32> -> tensor<4x31x4x1x1xi32>
    %53 = tt.broadcast %51 : tensor<1x1x4x1x1xi32> -> tensor<4x31x4x1x1xi32>
    %54 = arith.addi %52, %53 : tensor<4x31x4x1x1xi32>
    %55 = tt.broadcast %54 : tensor<4x31x4x1x1xi32> -> tensor<4x31x4x256x1xi32>
    %56 = tt.broadcast %33 : tensor<1x1x1x256x1xi32> -> tensor<4x31x4x256x1xi32>
    %57 = arith.addi %55, %56 : tensor<4x31x4x256x1xi32>
    %58 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<4x31x4x256x1x!tt.ptr<i8>>
    %59 = tt.addptr %58, %57 : tensor<4x31x4x256x1x!tt.ptr<i8>>, tensor<4x31x4x256x1xi32>
    tt.store %59, %42 : tensor<4x31x4x256x1x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_B1_modulo_offset
// CHECK:         %[[C2:.*]] = arith.constant 2 : i32
// CHECK:         %[[C4:.*]] = arith.constant 4 : i32
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:         %[[DIV:.*]] = arith.divsi %[[IV32]], %[[C4]] : i32
// CHECK:         %[[MUL:.*]] = arith.muli %[[DIV]], %[[C2]] : i32
// CHECK:         %[[REM:.*]] = arith.remsi %[[IV32]], %{{.*}} : i32
// CHECK:         %{{.*}} = arith.addi %[[MUL]], %[[REM]] : i32
// CHECK:         func.call @triton_print{{[_0-9]*}}({{.*}}) : (i32) -> ()
// CHECK:       }
// CHECK-NOT:   arith.remsi {{.*}} : tensor<16xi32>
// CHECK-NOT:   arith.divsi {{.*}} : tensor<16xi32>

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B1_modulo_offset(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %cst_0 = arith.constant dense<4> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.remsi %0, %cst_0 : tensor<16xi32>
    %2 = arith.divsi %0, %cst_0 : tensor<16xi32>
    %3 = arith.muli %2, %cst : tensor<16xi32>
    %4 = arith.addi %3, %1 : tensor<16xi32>
    tt.print " idx_B1: " {hex = false, isSigned = array<i32: 1>} : %4 : tensor<16xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.addptr %5, %4 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %7 = tt.load %6 : tensor<16x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %9, %7 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_B9_neg_stride
// CHECK:         %[[C15:.*]] = arith.constant 15 : i32
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:         %{{.*}} = arith.subi %[[C15]], %[[IV32]] : i32
// CHECK:         func.call @triton_print{{[_0-9]*}}({{.*}}) : (i32) -> ()
// CHECK:       }
// CHECK-NOT:   arith.subi {{.*}} : tensor<16xi32>

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B9_neg_stride(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<15> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.subi %cst, %0 : tensor<16xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %3 = tt.addptr %2, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %4 = tt.load %3 : tensor<16x!tt.ptr<i32>>
    tt.print " idx_B9: " {hex = false, isSigned = array<i32: 1>} : %1 : tensor<16xi32>
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %6, %4 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_B1_modulo_offset
// CHECK:         %[[C4:.*]] = arith.constant 4 : i32
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:         %{{.*}} = arith.remsi %[[IV32]], %[[C4]] : i32
// CHECK:         func.call @triton_print{{[_0-9]*}}({{.*}}) : (i32) -> ()
// CHECK:       }
// CHECK-NOT:   arith.remsi {{.*}} : tensor<16xi32>

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B1_modulo_offset(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.remsi %0, %cst : tensor<16xi32>
    tt.print " idx_B1: " {hex = false, isSigned = array<i32: 1>} : %1 : tensor<16xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %3 = tt.addptr %2, %1 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %4 = tt.load %3 : tensor<16x!tt.ptr<i32>>
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %6, %4 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @triton_max_1d_dimm1
// CHECK:       scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         %[[IV32:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:         func.call @triton_print{{[_0-9]*}}(%[[IV32]]) : (i32) -> ()
// CHECK:       }
// CHECK-NOT:   linalg.generic {{.*}} attrs {{.*}} tt.from_make_range {{.*}} tensor<2xi32>

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @triton_max_1d_dimm1(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    tt.print " idx: " {hex = false, isSigned = array<i32: 1>} : %0 : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<2x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i8>>, tensor<2xi32>
    %3 = tt.load %2 : tensor<2x!tt.ptr<i8>>
    tt.print " x: " {hex = false, isSigned = array<i32: 0>} : %3 : tensor<2xi8>
    tt.assert %true, "Expecting input to be integer type" : i1
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %8 = arith.maxui %arg2, %arg3 : i8
      tt.reduce.return %8 : i8
    }) : (tensor<2xi8>) -> i8
    tt.print " ret: " {hex = false, isSigned = array<i32: 0>} : %4 : i8
    // Wrap scalar store in tensor form so PtrAnalysis can lower it.
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1x!tt.ptr<i8>>
    %6 = tt.splat %4 : i8 -> tensor<1xi8>
    %7 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %8 = tt.addptr %5, %7 : tensor<1x!tt.ptr<i8>>, tensor<1xi32>
    tt.store %8, %6 : tensor<1x!tt.ptr<i8>>
    tt.return
  }
}

// -----

// CHECK:       func.func private @triton_print{{[_0-9]*}}(i32){{.*}}prefix = " idx_B10: "
// CHECK-LABEL: func.func @kernel_B10_2d_dynamic_offset
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           func.call @triton_print{{[_0-9]*}}(%{{.*}}) : (i32) -> ()
// CHECK:         }
// CHECK:       }

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B10_2d_dynamic_offset(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},%arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<4x1xi32>
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c4_i32 : i32
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.splat %2 : i32 -> tensor<4xi32>
    %5 = arith.addi %4, %3 : tensor<4xi32>
    %6 = arith.muli %1, %c8_i32 : i32
    %7 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %8 = tt.splat %6 : i32 -> tensor<8xi32>
    %9 = arith.addi %8, %7 : tensor<8xi32>
    %10 = tt.expand_dims %5 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %11 = arith.muli %10, %cst : tensor<4x1xi32>
    %12 = tt.expand_dims %9 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %13 = tt.broadcast %11 : tensor<4x1xi32> -> tensor<4x8xi32>
    %14 = tt.broadcast %12 : tensor<1x8xi32> -> tensor<4x8xi32>
    %15 = arith.addi %13, %14 : tensor<4x8xi32>
    %16 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
    %17 = tt.addptr %16, %15 : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>
    %18 = tt.load %17 : tensor<4x8x!tt.ptr<i32>>
    tt.print " idx_B10: " {hex = false, isSigned = array<i32: 1>} : %15 : tensor<4x8xi32>
    %19 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
    %20 = tt.addptr %19, %15 : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>
    tt.store %20, %18 : tensor<4x8x!tt.ptr<i32>>
    tt.return
  }
}

// -----


// CHECK-LABEL: func.func @kernel_B11_2d_divmod_linearization
// CHECK-NOT:   func.call @triton_print{{[_0-9]*}}({{.*}}) : (tensor<{{.*}}xi32>)
// CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-DAG:       arith.divsi
// CHECK-DAG:       arith.remsi
// CHECK:           func.call @triton_print{{[_0-9]*}}(%{{.*}}) : (i32) -> ()
// CHECK:         }
// CHECK:       }

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @kernel_B11_2d_divmod_linearization(
      %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst   = arith.constant dense<4> : tensor<4x8xi32>
    %cst_0 = arith.constant dense<8> : tensor<4x8xi32>
    %cst_1 = arith.constant dense<8> : tensor<4x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %4 = arith.muli %1, %cst_1 : tensor<4x1xi32>
    %5 = tt.broadcast %4 : tensor<4x1xi32> -> tensor<4x8xi32>
    %6 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<4x8xi32>
    %7 = arith.addi %5, %6 : tensor<4x8xi32>
    %8 = arith.divsi %7, %cst_0 : tensor<4x8xi32>
    %9 = arith.remsi %7, %cst_0 : tensor<4x8xi32>
    %10 = arith.muli %8, %cst : tensor<4x8xi32>
    %11 = arith.addi %10, %9 : tensor<4x8xi32>
    %12 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
    %13 = tt.addptr %12, %11 : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>
    %14 = tt.load %13 : tensor<4x8x!tt.ptr<i32>>
    tt.print " addr_B11: " {hex = false, isSigned = array<i32: 1>} : %11 : tensor<4x8xi32>
    %15 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
    %16 = tt.addptr %15, %11 : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>
    tt.store %16, %14 : tensor<4x8x!tt.ptr<i32>>
    tt.return
  }
}
