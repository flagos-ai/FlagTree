// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true compile-mode=simd_simt_template' \
// RUN:                --triton-to-linalg='compile-on-910-95=true compile-mode=simd_simt_template' --split-input-file \
// RUN: | FileCheck %s

// -----
// The `pid % S` strided block-pointer shape is owned by StridedAxisCoalescing,
// before Chunk and StridedLoadStoreRewrite.  It folds the S heads into the
// inner tensor dimension and records the launch-axis shrink metadata.
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 0 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @strided_axis_coalesce
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 4]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_coalesce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %twice = arith.addf %value, %value : tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %twice : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// Reading num_programs on the coalesced grid axis changes its visible value
// after launch-grid division.  Keep the original shape and do not claim the
// coalesce metadata.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @strided_axis_reads_num_programs
// CHECK-NOT: sizes: [16, 4]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_reads_num_programs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %num = tt.get_num_programs x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %num_splat = tt.splat %num : i32 -> tensor<16xi32>
    %num_f = arith.sitofp %num_splat : tensor<16xi32> to tensor<16xf32>
    %sum = arith.addf %value, %num_f : tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %sum : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// The folded head lane may be driven by any launch axis.  Keep the axis value
// in the module metadata so the launcher shrinks grid[1], not grid[0].
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 1 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @strided_axis_coalesce_axis1
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 4]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_coalesce_axis1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id y : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %twice = arith.addf %value, %value : tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %twice : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// A scalar per-head load is only lane-liftable when it is data.  If it feeds
// an address (an indirect gather), lane expansion would change the pointer
// relation, so the entire strided-axis rewrite must bail.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-LABEL: func.func @strided_axis_per_head_indirect_gather_bails
// CHECK-NOT: sizes: [16, 4]
// CHECK: memref.copy
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_per_head_indirect_gather_bails(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                               %indices: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %head_index_ptr = tt.addptr %indices, %head : !tt.ptr<i32>, i32
    %head_index = tt.load %head_index_ptr : !tt.ptr<i32>
    %gather_ptr = tt.addptr %arg0, %head_index : !tt.ptr<f32>, i32
    %gather = tt.load %gather_ptr : !tt.ptr<f32>
    %gather_splat = tt.splat %gather : f32 -> tensor<16xf32>
    %sum = arith.addf %value, %gather_splat : tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %sum : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// Multiple block-pointer loads with the same S are collected as one region and
// must be lifted together before their elementwise merge and matching store.
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 0 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @strided_axis_multiple_same_s_seeds
// Two source block-pointer loads and the destination must all become 2-D.
// CHECK-COUNT-3: memref.reinterpret_cast {{.*}}sizes: [16, 4]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_multiple_same_s_seeds(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src0_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src0 = tt.make_tensor_ptr %src0_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %lhs = tt.load %src0 : !tt.ptr<tensor<16xf32>>
    %src1_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %src1 = tt.make_tensor_ptr %src1_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %rhs = tt.load %src1 : !tt.ptr<tensor<16xf32>>
    %sum = arith.addf %lhs, %rhs : tensor<16xf32>
    %dst_base = tt.addptr %arg2, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %sum : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// Scan along the original T axis remains independent across the newly folded
// S lanes.  The original 1-D scan therefore becomes a [BT, S] scan on axis 0.
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 0 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK: func.func private @[[$SCAN:triton_cumsum[_0-9]*]](tensor<16x4xf32>, i32, i1) -> tensor<16x4xf32>
// CHECK-LABEL: func.func @strided_axis_scan_axis0
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 4]
// CHECK: call @[[$SCAN]]({{.*}}) : (tensor<16x4xf32>, i32, i1) -> tensor<16x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_scan_axis0(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %scanned = "tt.scan"(%value) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<16xf32>) -> tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %scanned : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// A T-axis reduction becomes one independent reduction per folded S lane.  A
// following scalar splat must be rebuilt as an S-lane broadcast, not collapsed.
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 0 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @strided_axis_reduce_axis0
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16, 4]
// CHECK: linalg.reduce ins({{.*}} : tensor<16x4xf32>) outs({{.*}} : tensor<4xf32>) dimensions = [0]
// CHECK: linalg.broadcast ins({{.*}} : tensor<4xf32>) outs({{.*}} : tensor<16x4xf32>) dimensions = [0]
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_reduce_axis0(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %reduced = "tt.reduce"(%value) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<16xf32>) -> f32
    %splat = tt.splat %reduced : f32 -> tensor<16xf32>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %splat : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// A candidate is valid only when the stride agrees with the ih split.  Here
// the block pointer has stride 4 but its base uses `pid % 2`, so Axis must not
// claim it or write coalescing metadata.
// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
// CHECK-NOT: hacc.coalesce_factor
// CHECK-NOT: hacc.coalesce_axis
// CHECK-LABEL: func.func @strided_axis_stride_ih_mismatch_bails
// CHECK-NOT: sizes: [16, 4]
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [16], strides: [4]
// CHECK-NOT: sizes: [16, 4]
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_stride_ih_mismatch_bails(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c2_i32 : i32
    %src_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src = tt.make_tensor_ptr %src_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value = tt.load %src : !tt.ptr<tensor<16xf32>>
    %dst_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst = tt.make_tensor_ptr %dst_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst, %value : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// -----
// Preserve the historical first-seed behavior: S and BT are selected from the
// first seed.  A later same-S seed with BT=8 is still lifted using the first
// seed's BT=16 backing tile and is sliced back to its original [8, 4] region.
// CHECK-LABEL: module attributes {
// CHECK: hacc.coalesce_axis = 0 : i32
// CHECK: hacc.coalesce_factor = 4 : i32
// CHECK-LABEL: func.func @strided_axis_same_s_first_seed_bt
// CHECK-COUNT-4: memref.reinterpret_cast {{.*}}sizes: [16, 4]
// CHECK: tensor.extract_slice {{.*}}[8, 4] {{.*}}tensor<16x4xf32> to tensor<8x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_same_s_first_seed_bt(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                     %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                     %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                     %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %c4_i64 = arith.constant 4 : i64
    %pid = tt.get_program_id x : i32
    %head = arith.remsi %pid, %c4_i32 : i32

    %src0_base = tt.addptr %arg0, %head : !tt.ptr<f32>, i32
    %src0 = tt.make_tensor_ptr %src0_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    %value0 = tt.load %src0 : !tt.ptr<tensor<16xf32>>
    %dst0_base = tt.addptr %arg1, %head : !tt.ptr<f32>, i32
    %dst0 = tt.make_tensor_ptr %dst0_base, [%c16_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<16xf32>>
    tt.store %dst0, %value0 : !tt.ptr<tensor<16xf32>>

    %src1_base = tt.addptr %arg2, %head : !tt.ptr<f32>, i32
    %src1 = tt.make_tensor_ptr %src1_base, [%c8_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<8xf32>>
    %value1 = tt.load %src1 : !tt.ptr<tensor<8xf32>>
    %dst1_base = tt.addptr %arg3, %head : !tt.ptr<f32>, i32
    %dst1 = tt.make_tensor_ptr %dst1_base, [%c8_i64], [%c4_i64], [%c0_i32] {order = array<i32: 0>} : <tensor<8xf32>>
    tt.store %dst1, %value1 : !tt.ptr<tensor<8xf32>>
    tt.return
  }
}
