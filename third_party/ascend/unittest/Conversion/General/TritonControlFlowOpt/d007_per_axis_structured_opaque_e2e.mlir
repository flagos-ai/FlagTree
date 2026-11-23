// RUN: triton-opt --triton-control-flow-opt %s -verify-each | FileCheck %s --check-prefix=CFO
// RUN: triton-opt --triton-control-flow-opt --triton-to-unstructure --triton-to-linalg %s -verify-each | FileCheck %s --check-prefix=E2E --implicit-check-not='!tt.ptr' --implicit-check-not=unrealized_conversion_cast --implicit-check-not=PointerDescriptorBoundary --implicit-check-not=PointerDescriptorRebuild --implicit-check-not=PointerDescriptorOffsetForm --implicit-check-not=PointerDescriptorStructuredAxes

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // A rank-1 affine range plus a uniform scalar offset remains one
  // Structured axis and keeps the strided_1d handoff.
  tt.func public @d007_rank1_structured(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %condition: i1) {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c5 = arith.constant 5 : i32
    %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %stride = tt.splat %c2 : i32 -> tensor<4xi32>
    %scaled = arith.muli %range, %stride : tensor<4xi32>
    %uniform = tt.splat %c5 : i32 -> tensor<4xi32>
    %offset = arith.addi %scaled, %uniform : tensor<4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %selected = scf.if %condition -> (tensor<4x!tt.ptr<f32>>) {
      %ptr = tt.addptr %base_tensor, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %uniform_alt = tt.splat %c0 : i32 -> tensor<4xi32>
      %offset_alt = arith.addi %scaled, %uniform_alt : tensor<4xi32>
      %ptr = tt.addptr %base_tensor, %offset_alt : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %ptr : tensor<4x!tt.ptr<f32>>
    }
    %loaded = tt.load %selected : tensor<4x!tt.ptr<f32>>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %range : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptr, %loaded : tensor<4x!tt.ptr<f32>>
    tt.return
  }

  // An arbitrary rank-1 offset tensor is Opaque as a whole. The scalar stride
  // and uniform offset fields must be normalized to zero.
  tt.func public @d007_rank1_opaque(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %condition: i1,
      %then_offsets: tensor<4xi32>, %else_offsets: tensor<4xi32>) {
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %selected = scf.if %condition -> (tensor<4x!tt.ptr<f32>>) {
      %ptr = tt.addptr %base_tensor, %then_offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %ptr : tensor<4x!tt.ptr<f32>>
    } else {
      %ptr = tt.addptr %base_tensor, %else_offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      scf.yield %ptr : tensor<4x!tt.ptr<f32>>
    }
    %loaded = tt.load %selected : tensor<4x!tt.ptr<f32>>
    %output_offsets = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %output_offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %output_ptr, %loaded : tensor<4x!tt.ptr<f32>>
    tt.return
  }

  // Axis 0 is a make_range contribution while axis 1 comes from an opaque
  // lane-wise input. The joined descriptor must retain [Structured, Opaque].
  tt.func public @d007_rank2_mixed(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>, %condition: i1,
      %then_axis1: tensor<4xi32>, %else_axis1: tensor<4xi32>) {
    %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %axis0_expanded = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %axis0_broadcast = tt.broadcast %axis0_expanded : tensor<2x1xi32> -> tensor<2x4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %selected = scf.if %condition -> (tensor<2x4x!tt.ptr<f32>>) {
      %axis1_expanded = tt.expand_dims %then_axis1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %axis1_broadcast = tt.broadcast %axis1_expanded : tensor<1x4xi32> -> tensor<2x4xi32>
      %offset = arith.addi %axis0_broadcast, %axis1_broadcast : tensor<2x4xi32>
      %ptr = tt.addptr %base_tensor, %offset : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %ptr : tensor<2x4x!tt.ptr<f32>>
    } else {
      %axis1_expanded = tt.expand_dims %else_axis1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
      %axis1_broadcast = tt.broadcast %axis1_expanded : tensor<1x4xi32> -> tensor<2x4xi32>
      %offset = arith.addi %axis0_broadcast, %axis1_broadcast : tensor<2x4xi32>
      %ptr = tt.addptr %base_tensor, %offset : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %ptr : tensor<2x4x!tt.ptr<f32>>
    }
    %loaded = tt.load %selected : tensor<2x4x!tt.ptr<f32>>
    %output_offsets = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi32>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %output_offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %output_ptr, %loaded : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }

  // A uniform backedge update must remain the scalar uniform-offset field.
  // The invariant opaque-axis contribution stays outside the loop instead of
  // becoming a full tensor loop carrier on every iteration.
  tt.func public @d007_rank2_mixed_uniform_loop(
      %base: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %axis1: tensor<4xi32>, %upper: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delta = arith.constant dense<32> : tensor<2x4xi32>
    %axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %axis0_expanded = tt.expand_dims %axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %axis0_broadcast = tt.broadcast %axis0_expanded : tensor<2x1xi32> -> tensor<2x4xi32>
    %axis1_expanded = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %axis1_broadcast = tt.broadcast %axis1_expanded : tensor<1x4xi32> -> tensor<2x4xi32>
    %offset = arith.addi %axis0_broadcast, %axis1_broadcast : tensor<2x4xi32>
    %base_tensor = tt.splat %base : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %initial = tt.addptr %base_tensor, %offset : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    %final = scf.for %iv = %c0 to %upper step %c1
        iter_args(%pointer = %initial) -> (tensor<2x4x!tt.ptr<f32>>) {
      %next = tt.addptr %pointer, %delta : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
      scf.yield %next : tensor<2x4x!tt.ptr<f32>>
    }
    %loaded = tt.load %final : tensor<2x4x!tt.ptr<f32>>
    %output_offsets = arith.constant dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi32>
    %output_tensor = tt.splat %output : !tt.ptr<f32> -> tensor<2x4x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_tensor, %output_offsets : tensor<2x4x!tt.ptr<f32>>, tensor<2x4xi32>
    tt.store %output_ptr, %loaded : tensor<2x4x!tt.ptr<f32>>
    tt.return
  }
}

// CFO-LABEL: tt.func public @d007_rank1_structured
// CFO: PointerDescriptorStructuredAxes = array<i32: 1>
// CFO: PointerDescriptorOffsetForm = "strided_1d"
// CFO-LABEL: tt.func public @d007_rank1_opaque
// CFO: PointerDescriptorStructuredAxes = array<i32: 0>
// CFO-LABEL: tt.func public @d007_rank2_mixed
// CFO: PointerDescriptorStructuredAxes = array<i32: 1, 0>
// CFO-LABEL: tt.func public @d007_rank2_mixed_uniform_loop
// CFO: scf.for
// CFO-SAME: i32
// CFO-NOT: scf.yield {{.*}}tensor<2x4xi32>
// CFO: PointerDescriptorStructuredAxes = array<i32: 1, 0>

// E2E-LABEL: func.func @d007_rank1_structured
// E2E-NOT: memref.load
// E2E: %[[SOURCE_VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [8], strides: [1]
// E2E-SAME: to memref<8xf32, strided<[1], offset: ?>>
// E2E-NOT: memref.load
// E2E: %[[COPY_BUFFER:.*]] = memref.alloc() : memref<8xf32>
// E2E-NOT: memref.load
// E2E: memref.copy %[[SOURCE_VIEW]], %[[COPY_BUFFER]] : memref<8xf32, strided<[1], offset: ?>> to memref<8xf32>
// E2E-NOT: memref.load
// E2E: %[[SOURCE_TENSOR:.*]] = bufferization.to_tensor %[[COPY_BUFFER]]
// E2E-NOT: memref.load
// E2E: %[[SLICE:.*]] = tensor.extract_slice %[[SOURCE_TENSOR]][0] [4] [2] : tensor<8xf32> to tensor<4xf32>
// E2E-NOT: memref.load
// E2E: bufferization.materialize_in_destination %[[SLICE]]
// E2E-NOT: memref.load
// E2E: return
// E2E-NOT: memref.load
// E2E-LABEL: func.func @d007_rank1_opaque
// E2E: memref.load
// E2E: bufferization.materialize_in_destination
// E2E: return
// E2E-LABEL: func.func @d007_rank2_mixed
// E2E: memref.load
// E2E: bufferization.materialize_in_destination
// E2E: return
// E2E-LABEL: func.func @d007_rank2_mixed_uniform_loop
// E2E: scf.for
// E2E-SAME: iter_args(%{{.*}} = %{{.*}}) -> (i32)
// E2E: memref.load
// E2E: bufferization.materialize_in_destination
// E2E: return
