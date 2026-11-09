// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=CFO
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure %s -verify-each | FileCheck %s --check-prefix=T2U --implicit-check-not='tensor.extract {{.*}} : tensor<4x!tt.ptr<f32>>'
// RUN: triton-opt --triton-control-flow-opt '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' %s -verify-each | FileCheck %s --check-prefix=T2U --implicit-check-not='tensor.extract {{.*}} : tensor<4x!tt.ptr<f32>>'
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --triton-to-linalg %s -verify-each | FileCheck %s --check-prefix=LINALG --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast --implicit-check-not=PointerDescriptorBoundary --implicit-check-not=PointerDescriptorRebuild --implicit-check-not=PointerDescriptorOffsetForm

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @if_tensor_ptr_different_base(
      %base0: !tt.ptr<f32>, %base1: !tt.ptr<f32>,
      %output: !tt.ptr<f32>, %cond: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %off1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %off2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %selected = scf.if %cond -> (tensor<4x!tt.ptr<f32>>) {
      %splat0 = tt.splat %base0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %then_ptr = tt.addptr %splat0, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %then_ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %splat1 = tt.splat %base1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %else_ptr = tt.addptr %splat1, %off2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %else_ptr : tensor<4x!tt.ptr<f32>>
    }
    // Ownership must survive an ordinary addptr after the descriptor rebuild.
    %post_addptr = tt.addptr %selected, %off1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %post_addptr : tensor<4x!tt.ptr<f32>>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %output_splat = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_splat, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptr, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CFO-LABEL: tt.func public @if_tensor_ptr_different_base
// CFO-DAG:   %[[BASE0:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<f32> -> i64
// CFO-DAG:   %[[BASE1:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<f32> -> i64
// CFO:       %[[SELECTED:.*]]:2 = scf.if %{{.*}} -> (i64, i32) {
// CFO:         scf.yield %[[BASE0]], %{{.*}} : i64, i32
// CFO:       } else {
// CFO:         scf.yield %[[BASE1]], %{{.*}} : i64, i32
// CFO:       }
// CFO:       PointerDescriptorOffsetForm = "strided_1d"
// CFO:       PointerDescriptorRebuild
// CFO:       tt.addptr
// CFO:       tt.load
// CFO:       tt.store

// T2U-LABEL: tt.func public @if_tensor_ptr_different_base
// T2U-DAG:   %[[BASE0:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<f32> -> i64
// T2U-DAG:   %[[BASE1:.*]] = tt.ptr_to_int %{{.*}} : !tt.ptr<f32> -> i64
// T2U:       %[[SELECTED_BASE:.*]] = arith.select %{{.*}}, %[[BASE0]], %[[BASE1]] : i64
// T2U:       %[[SELECTED_OFFSET:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// T2U:       %[[BASE_PTR:.*]] = tt.int_to_ptr %[[SELECTED_BASE]] : i64 -> !tt.ptr<f32>
// T2U:       tt.addptr
// T2U:       tt.load

// LINALG-LABEL: func.func @if_tensor_ptr_different_base
// LINALG:       arith.select
// LINALG:       hivm.hir.pointer_cast
// LINALG:       memref.reinterpret_cast
// LINALG:       memref.copy
// LINALG:       bufferization.to_tensor
// LINALG:       return
