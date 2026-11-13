// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=CFO
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure %s -verify-each | FileCheck %s --check-prefix=T2U
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --bubble-up-operation --triton-to-linalg %s -verify-each | FileCheck %s --check-prefix=LINALG

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_pointer_descriptor_loop(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<1> : tensor<4xi32>
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %initial = tt.addptr %base_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %final = scf.for %iv = %c0 to %upper step %c1 iter_args(%ptr = %initial) -> (tensor<4x!tt.ptr<f32>>) {
      %next = tt.addptr %ptr, %delta : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %next : tensor<4x!tt.ptr<f32>>
    }
    %value = tt.load %final : tensor<4x!tt.ptr<f32>>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptr, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }

  // A CFO descriptor loop may also carry an ordinary integer offset tensor.
  // Recover its affine row/column layout from the carrier provenance instead
  // of lowering the corresponding load as an indirect gather.
  tt.func public @marked_affine_index_carrier(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %row_stride: i32,
      %upper: index) {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %rows = tt.expand_dims %range {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %stride = tt.splat %row_stride : i32 -> tensor<4x1xi32>
    %scaled_rows = arith.muli %rows, %stride : tensor<4x1xi32>
    %row_offsets = tt.broadcast %scaled_rows : tensor<4x1xi32> -> tensor<4x4xi32>
    %columns = tt.expand_dims %range {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %column_offsets = tt.broadcast %columns : tensor<1x4xi32> -> tensor<4x4xi32>
    %initial_offsets = arith.addi %row_offsets, %column_offsets : tensor<4x4xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %initial_output = tt.addptr %output_base, %initial_offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %advance = arith.constant dense<16> : tensor<4x4xi32>
    %result:2 = scf.for %iv = %c2 to %upper step %c3
        iter_args(%offsets = %initial_offsets, %output_ptrs = %initial_output)
        -> (tensor<4x4xi32>, tensor<4x4x!tt.ptr<f32>>) {
      %input_ptrs = tt.addptr %input_base, %offsets : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      %value = tt.load %input_ptrs : tensor<4x4x!tt.ptr<f32>>
      tt.store %output_ptrs, %value : tensor<4x4x!tt.ptr<f32>>
      %next_offsets = arith.addi %offsets, %advance : tensor<4x4xi32>
      %next_output = tt.addptr %output_ptrs, %advance : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
      scf.yield %next_offsets, %next_output : tensor<4x4xi32>, tensor<4x4x!tt.ptr<f32>>
    }
    tt.return
  }
  // A pointer broadcast after opaque i64 row indexing must retain contiguous
  // i32 columns. Keep the invariant rows outside the loop and load each row
  // with a contiguous copy after advancing only a scalar uniform offset.
  tt.func public @mixed_pointer_broadcast_load(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %row_source: !tt.ptr<i64>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x32xi32>
    %row_lanes = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %row_base = tt.splat %row_source : !tt.ptr<i64> -> tensor<2x!tt.ptr<i64>>
    %row_ptr = tt.addptr %row_base, %row_lanes : tensor<2x!tt.ptr<i64>>, tensor<2xi32>
    %rows = tt.load %row_ptr : tensor<2x!tt.ptr<i64>>
    %row_offsets = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi64> -> tensor<2x1xi64>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %row_ptrs = tt.addptr %base_tensor, %row_offsets : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi64>
    %broadcast = tt.broadcast %row_ptrs : tensor<2x1x!tt.ptr<f32>> -> tensor<2x32x!tt.ptr<f32>>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %columns = tt.expand_dims %range {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %column_offsets = tt.broadcast %columns : tensor<1x32xi32> -> tensor<2x32xi32>
    %initial = tt.addptr %broadcast, %column_offsets : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %initial) -> (tensor<2x32x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
      scf.yield %next : tensor<2x32x!tt.ptr<f32>>
    }
    %loaded = tt.load %final : tensor<2x32x!tt.ptr<f32>>
    %out_row_range = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %out_rows = tt.expand_dims %out_row_range {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %out_stride = arith.constant dense<32> : tensor<2x1xi32>
    %out_scaled_rows = arith.muli %out_rows, %out_stride : tensor<2x1xi32>
    %out_row_offsets = tt.broadcast %out_scaled_rows : tensor<2x1xi32> -> tensor<2x32xi32>
    %out_offsets = arith.addi %out_row_offsets, %column_offsets : tensor<2x32xi32>
    %out_base = tt.splat %output : !tt.ptr<f32> -> tensor<2x32x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %out_offsets : tensor<2x32x!tt.ptr<f32>>, tensor<2x32xi32>
    tt.store %out_ptrs, %loaded : tensor<2x32x!tt.ptr<f32>>
    tt.return
  }
}

// CFO-LABEL: tt.func public @tensor_pointer_descriptor_loop
// CFO:       scf.for
// CFO-SAME:  i32
// CFO:       tt.addptr {{.*}}PointerDescriptorOffsetForm = "strided_1d"
// CFO-SAME:  PointerDescriptorStructuredAxes = array<i32: 1>

// LINALG-LABEL: func.func @tensor_pointer_descriptor_loop
// LINALG:       scf.for
// LINALG-SAME:  i32
// LINALG-NOT:   tensor<4x!tt.ptr
// LINALG-NOT:   unrealized_conversion_cast

// CFO-LABEL: tt.func public @marked_affine_index_carrier
// CFO:       scf.for
// CFO:       PointerDescriptorBoundary

// T2U-LABEL: tt.func public @marked_affine_index_carrier
// T2U:       scf.for %[[IV:[a-zA-Z0-9_]+]] =
// T2U:       arith.subi %[[IV]],
// T2U:       arith.divui
// T2U:       tt.addptr {{.*}}PointerDescriptorRebuild
// T2U-SAME:  PointerDescriptorStructuredAxes = array<i32: 1, 1>
// T2U:       tt.load

// LINALG-LABEL: func.func @marked_affine_index_carrier
// LINALG:       scf.for
// LINALG-NOT:   triton_indirect_load
// LINALG:       return

// CFO-LABEL: tt.func public @mixed_pointer_broadcast_load
// CFO:       tt.broadcast {{.*}} : tensor<2x1xi64> -> tensor<2x32xi64>
// CFO-NOT:   arith.subi
// CFO:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i64)
// CFO:       scf.yield {{.*}} : i64
// CFO:       PointerDescriptorStructuredAxes = array<i32: 0, 1>

// T2U-LABEL: tt.func public @mixed_pointer_broadcast_load
// T2U:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i64)
// T2U:       tt.load {{.*}} : tensor<1x32x!tt.ptr<f32>>

// LINALG-LABEL: func.func @mixed_pointer_broadcast_load
// LINALG:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (i64)
// LINALG-NOT:   triton_indirect_load
// LINALG-NOT:   memref.load
// LINALG:       memref.copy
// LINALG-NOT:   triton_indirect_load
// LINALG-NOT:   memref.load
// LINALG:       return
