// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=64' -o - | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=0' -o - | FileCheck %s --check-prefix=CAP0
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=256' -o - | FileCheck %s --check-prefix=DYNAMIC
// RUN: triton-opt %s --verify-each -graph-optimize='rule-mask=4 ub-capacity-bytes=64 compile-mode=simt_only' -o - | FileCheck %s --check-prefix=PURE-SIMT

// Address order, rather than program order, determines the packed value
// layout.  The high interval is deliberately stored first.  The replacement
// must still be anchored at the program-order last store (the low interval),
// after both source values dominate that anchor.
// CHECK-LABEL: tt.func @pack_reverse_program_order(
// CHECK-NOT: tt.store
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi32>
// CHECK-NEXT: %[[LOW:.*]] = tensor.insert_slice %{{.*}} into %[[EMPTY]][0] [4] [1] : tensor<4xi32> into tensor<8xi32>
// CHECK-NEXT: %[[PACKED:.*]] = tensor.insert_slice %{{.*}} into %[[LOW]][4] [4] [1] : tensor<4xi32> into tensor<8xi32>
// CHECK-NEXT: %[[RANGE:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK-NEXT: %[[SPLAT:.*]] = tt.splat %{{.*}} : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
// CHECK-NEXT: %[[ADDRESS:.*]] = tt.addptr %[[SPLAT]], %[[RANGE]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
// CHECK-NEXT: tt.store %[[ADDRESS]], %[[PACKED]] : tensor<8x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK-NOT: tensor.insert_slice
tt.func @pack_reverse_program_order(%base: !tt.ptr<i32>) {
  %low_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %high_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %low_addresses = tt.addptr %base_splat, %low_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %high_addresses = tt.addptr %base_splat, %high_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %low_value = arith.constant dense<1> : tensor<4xi32>
  %high_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %high_addresses, %high_value : tensor<4x!tt.ptr<i32>>
  tt.store %low_addresses, %low_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// PURE-SIMT-LABEL: tt.func @pack_reverse_program_order(
// PURE-SIMT-NOT: tensor.empty
// PURE-SIMT: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// PURE-SIMT: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// PURE-SIMT-NOT: tensor.empty
// PURE-SIMT-LABEL: tt.func @pack_rank2_row_major_nonzero_origin(

// N-D values may be flattened only when their logical row-major order is
// exactly address order.  This fixture also uses a nonzero row origin for the
// second tile and stores that high tile first in program order.
// CHECK-LABEL: tt.func @pack_rank2_row_major_nonzero_origin(
// CHECK: %[[LOW_VALUE:.*]] = arith.constant dense<1> : tensor<2x3xi32>
// CHECK-NEXT: %[[HIGH_VALUE:.*]] = arith.constant dense<2> : tensor<2x3xi32>
// CHECK-NOT: tt.store
// CHECK: %[[EMPTY2D:.*]] = tensor.empty() : tensor<12xi32>
// CHECK-NEXT: %[[LOW_FLAT:.*]] = tt.reshape %[[LOW_VALUE]] : tensor<2x3xi32> -> tensor<6xi32>
// CHECK-NEXT: %[[LOW_PACKED:.*]] = tensor.insert_slice %[[LOW_FLAT]] into %[[EMPTY2D]][0] [6] [1] : tensor<6xi32> into tensor<12xi32>
// CHECK-NEXT: %[[HIGH_FLAT:.*]] = tt.reshape %[[HIGH_VALUE]] : tensor<2x3xi32> -> tensor<6xi32>
// CHECK-NEXT: %[[PACKED2D:.*]] = tensor.insert_slice %[[HIGH_FLAT]] into %[[LOW_PACKED]][6] [6] [1] : tensor<6xi32> into tensor<12xi32>
// CHECK-NEXT: %[[RANGE2D:.*]] = tt.make_range {end = 12 : i32, start = 0 : i32} : tensor<12xi32>
// CHECK-NEXT: %[[SPLAT2D:.*]] = tt.splat {{%.*}} : !tt.ptr<i32> -> tensor<12x!tt.ptr<i32>>
// CHECK-NEXT: %[[ADDRESSES2D:.*]] = tt.addptr %[[SPLAT2D]], %[[RANGE2D]] : tensor<12x!tt.ptr<i32>>, tensor<12xi32>
// CHECK-NEXT: tt.store %[[ADDRESSES2D]], %[[PACKED2D]] : tensor<12x!tt.ptr<i32>>
tt.func @pack_rank2_row_major_nonzero_origin(%base: !tt.ptr<i32>) {
  %low_row = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %low_column = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %high_row = tt.make_range {end = 4 : i32, start = 2 : i32} : tensor<2xi32>
  %high_column = tt.make_range {end = 3 : i32, start = 0 : i32} : tensor<3xi32>
  %three = arith.constant 3 : i32
  %one = arith.constant 1 : i32

  %low_row_expanded = tt.expand_dims %low_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %low_column_expanded = tt.expand_dims %low_column {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %low_row_stride = tt.splat %three : i32 -> tensor<2x1xi32>
  %low_column_stride = tt.splat %one : i32 -> tensor<1x3xi32>
  %low_row_term = arith.muli %low_row_expanded, %low_row_stride : tensor<2x1xi32>
  %low_column_term = arith.muli %low_column_expanded, %low_column_stride : tensor<1x3xi32>
  %low_row_broadcast = tt.broadcast %low_row_term : tensor<2x1xi32> -> tensor<2x3xi32>
  %low_column_broadcast = tt.broadcast %low_column_term : tensor<1x3xi32> -> tensor<2x3xi32>
  %low_offsets = arith.addi %low_row_broadcast, %low_column_broadcast : tensor<2x3xi32>
  %low_base = tt.splat %base : !tt.ptr<i32> -> tensor<2x3x!tt.ptr<i32>>
  %low_addresses = tt.addptr %low_base, %low_offsets : tensor<2x3x!tt.ptr<i32>>, tensor<2x3xi32>

  %high_row_expanded = tt.expand_dims %high_row {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %high_column_expanded = tt.expand_dims %high_column {axis = 0 : i32} : tensor<3xi32> -> tensor<1x3xi32>
  %high_row_stride = tt.splat %three : i32 -> tensor<2x1xi32>
  %high_column_stride = tt.splat %one : i32 -> tensor<1x3xi32>
  %high_row_term = arith.muli %high_row_expanded, %high_row_stride : tensor<2x1xi32>
  %high_column_term = arith.muli %high_column_expanded, %high_column_stride : tensor<1x3xi32>
  %high_row_broadcast = tt.broadcast %high_row_term : tensor<2x1xi32> -> tensor<2x3xi32>
  %high_column_broadcast = tt.broadcast %high_column_term : tensor<1x3xi32> -> tensor<2x3xi32>
  %high_offsets = arith.addi %high_row_broadcast, %high_column_broadcast : tensor<2x3xi32>
  %high_base = tt.splat %base : !tt.ptr<i32> -> tensor<2x3x!tt.ptr<i32>>
  %high_addresses = tt.addptr %high_base, %high_offsets : tensor<2x3x!tt.ptr<i32>>, tensor<2x3xi32>

  %low_value = arith.constant dense<1> : tensor<2x3xi32>
  %high_value = arith.constant dense<2> : tensor<2x3xi32>
  tt.store %high_addresses, %high_value : tensor<2x3x!tt.ptr<i32>>
  tt.store %low_addresses, %low_value : tensor<2x3x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @pack_rank3_row_major(
// CHECK-NOT: tt.store
// CHECK: tensor.empty() : tensor<16xi32>
// CHECK: tt.reshape {{%.*}} : tensor<2x2x2xi32> -> tensor<8xi32>
// CHECK: tt.reshape {{%.*}} : tensor<2x2x2xi32> -> tensor<8xi32>
// CHECK: tt.store {{%.*}}, {{%.*}} : tensor<16x!tt.ptr<i32>>
tt.func @pack_rank3_row_major(%base: !tt.ptr<i32>) {
  %low_axis0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %high_axis0 = tt.make_range {end = 4 : i32, start = 2 : i32} : tensor<2xi32>
  %axis1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %axis2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %four = arith.constant 4 : i32
  %two = arith.constant 2 : i32
  %one = arith.constant 1 : i32

  %low_axis0_e1 = tt.expand_dims %low_axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %low_axis0_e2 = tt.expand_dims %low_axis0_e1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %high_axis0_e1 = tt.expand_dims %high_axis0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %high_axis0_e2 = tt.expand_dims %high_axis0_e1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %axis1_e1 = tt.expand_dims %axis1 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %axis1_e2 = tt.expand_dims %axis1_e1 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
  %axis2_e1 = tt.expand_dims %axis2 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %axis2_e2 = tt.expand_dims %axis2_e1 {axis = 0 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
  %stride0 = tt.splat %four : i32 -> tensor<2x1x1xi32>
  %stride1 = tt.splat %two : i32 -> tensor<1x2x1xi32>
  %stride2 = tt.splat %one : i32 -> tensor<1x1x2xi32>
  %low_term0 = arith.muli %low_axis0_e2, %stride0 : tensor<2x1x1xi32>
  %high_term0 = arith.muli %high_axis0_e2, %stride0 : tensor<2x1x1xi32>
  %term1 = arith.muli %axis1_e2, %stride1 : tensor<1x2x1xi32>
  %term2 = arith.muli %axis2_e2, %stride2 : tensor<1x1x2xi32>
  %low_term0_b = tt.broadcast %low_term0 : tensor<2x1x1xi32> -> tensor<2x2x2xi32>
  %high_term0_b = tt.broadcast %high_term0 : tensor<2x1x1xi32> -> tensor<2x2x2xi32>
  %term1_b = tt.broadcast %term1 : tensor<1x2x1xi32> -> tensor<2x2x2xi32>
  %term2_b = tt.broadcast %term2 : tensor<1x1x2xi32> -> tensor<2x2x2xi32>
  %low_partial = arith.addi %low_term0_b, %term1_b : tensor<2x2x2xi32>
  %high_partial = arith.addi %high_term0_b, %term1_b : tensor<2x2x2xi32>
  %low_offsets = arith.addi %low_partial, %term2_b : tensor<2x2x2xi32>
  %high_offsets = arith.addi %high_partial, %term2_b : tensor<2x2x2xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<2x2x2x!tt.ptr<i32>>
  %low_addresses = tt.addptr %base_splat, %low_offsets : tensor<2x2x2x!tt.ptr<i32>>, tensor<2x2x2xi32>
  %high_addresses = tt.addptr %base_splat, %high_offsets : tensor<2x2x2x!tt.ptr<i32>>, tensor<2x2x2xi32>
  %low_value = arith.constant dense<1> : tensor<2x2x2xi32>
  %high_value = arith.constant dense<2> : tensor<2x2x2xi32>
  tt.store %high_addresses, %high_value : tensor<2x2x2x!tt.ptr<i32>>
  tt.store %low_addresses, %low_value : tensor<2x2x2x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_gap(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_gap(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 9 : i32, start = 5 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Any overlap poisons the entire bucket, rather than merely being skipped.
// CHECK-LABEL: tt.func @reject_overlap(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_overlap(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 6 : i32, start = 2 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_mask(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_mask(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %mask = arith.constant dense<true> : tensor<4xi1>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value, %mask : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// i1 has no byte-addressable element width, so capacity accounting must not
// round it up and pack the stores.
// CHECK-LABEL: tt.func @reject_i1_element(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i1>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i1>>
tt.func @reject_i1_element(%base: !tt.ptr<i1>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
  %first_value = arith.constant dense<true> : tensor<4xi1>
  %second_value = arith.constant dense<false> : tensor<4xi1>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i1>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i1>>
  tt.return
}

// Complete store attributes must match; cache policy alone makes these two
// stores distinct buckets.
// CHECK-LABEL: tt.func @reject_different_cache(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = ca : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = cg : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_cache(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value cacheModifier = ca : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value cacheModifier = cg : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_different_evict(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} evictionPolicy = evict_first : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} evictionPolicy = evict_last : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_evict(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value evictionPolicy = evict_first : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value evictionPolicy = evict_last : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Exact SSA base identity, not an alias guess, is required for a bucket.
// CHECK-LABEL: tt.func @reject_different_base(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_different_base(%first_base: !tt.ptr<i32>, %second_base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %first_splat = tt.splat %first_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %second_splat = tt.splat %second_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %first_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %second_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// The middle load touches only the suffix store's interval.  The first store
// is delayed across the load, but their intervals are disjoint, so packing is
// legal.  This rejects an implementation that protects the whole run instead
// of only the delayed program-order prefix.
// CAP0-LABEL: tt.func @pack_suffix_load_between_stores(
// CAP0-NOT: tensor.empty
// CAP0-NOT: tensor.insert_slice
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CAP0: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-LABEL: tt.func @pack_suffix_load_between_stores(
// CHECK-NOT: tt.store
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK: tensor.empty() : tensor<8xi32>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<8x!tt.ptr<i32>>
// CHECK-NOT: tt.store
tt.func @pack_suffix_load_between_stores(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %suffix_load = tt.load %second_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Address order is low/high, while program order is high/low.  The side store
// overlaps only the suffix low store and has a different cache attribute so it
// remains an intervening effect instead of joining the planned-store bucket.
// CHECK-LABEL: tt.func @pack_reverse_program_order_with_suffix_side_store(
// CHECK-NOT: tt.store
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = ca : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK: tensor.empty() : tensor<8xi32>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<8x!tt.ptr<i32>>
// CHECK-NOT: tt.store
tt.func @pack_reverse_program_order_with_suffix_side_store(%base: !tt.ptr<i32>) {
  %low_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %high_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %low_addresses = tt.addptr %base_splat, %low_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %high_addresses = tt.addptr %base_splat, %high_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %low_value = arith.constant dense<1> : tensor<4xi32>
  %high_value = arith.constant dense<2> : tensor<4xi32>
  %side_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %high_addresses, %high_value : tensor<4x!tt.ptr<i32>>
  tt.store %low_addresses, %side_value cacheModifier = ca : tensor<4x!tt.ptr<i32>>
  tt.store %low_addresses, %low_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// The side store overlaps the delayed prefix store, so packing must remain
// rejected even though it uses a distinct cache-attribute bucket.
// CHECK-LABEL: tt.func @reject_prefix_overlapping_side_store(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} cacheModifier = ca : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_prefix_overlapping_side_store(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %side_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %first_addresses, %side_value cacheModifier = ca : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Each gap gets its own delayed prefix: both loads touch only future stores,
// so the three planned stores can be packed together.
// CHECK-LABEL: tt.func @pack_three_stores_with_suffix_loads(
// CHECK-NOT: tt.store
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK: tensor.empty() : tensor<12xi32>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<12x!tt.ptr<i32>>
// CHECK-NOT: tt.store
tt.func @pack_three_stores_with_suffix_loads(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %third_range = tt.make_range {end = 12 : i32, start = 8 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %third_addresses = tt.addptr %base_splat, %third_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %third_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %second_load = tt.load %second_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  %third_load = tt.load %third_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %third_addresses, %third_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// In the second gap, the load overlaps S0 rather than its immediate
// predecessor S1.  The full delayed prefix must be protected, so no packing
// is legal.
// CHECK-LABEL: tt.func @reject_earlier_prefix_overlap_in_three_store_run(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_earlier_prefix_overlap_in_three_store_run(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %third_range = tt.make_range {end = 12 : i32, start = 8 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %third_addresses = tt.addptr %base_splat, %third_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %third_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  %first_load = tt.load %first_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %third_addresses, %third_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// This load overlaps the delayed prefix S1, so it cannot be crossed by the
// packed store.
// CHECK-LABEL: tt.func @reject_load_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_load_between_stores(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %unused = tt.load %first_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// A distinct entry root does not conflict with the delayed output store under
// the StoreCoalescing ABI contract. The side store remains in program order,
// and the two %base stores pack at the final anchor.
// CHECK-LABEL: tt.func @pack_distinct_root_store_between_stores(
// CHECK-NOT: tt.store
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: tt.store
// CHECK: tensor.empty() : tensor<8xi32>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<8x!tt.ptr<i32>>
// CHECK-NOT: tt.store
tt.func @pack_distinct_root_store_between_stores(%base: !tt.ptr<i32>, %other_base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %other_splat = tt.splat %other_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %other_addresses = tt.addptr %other_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %other_value = arith.constant dense<3> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %other_addresses, %other_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Distinct roots answer only the alias question. A masked intervening load
// remains outside the direct, unpredicated access subset and must block the
// rewrite.
// CHECK-LABEL: tt.func @reject_distinct_root_masked_load_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_distinct_root_masked_load_between_stores(
    %base: !tt.ptr<i32>, %other_base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %other_splat = tt.splat %other_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %other_addresses = tt.addptr %other_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  %mask = arith.constant dense<true> : tensor<4xi1>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %masked_load = tt.load %other_addresses, %mask : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// An arbitrary integer-to-pointer conversion has no tracked entry-argument
// provenance.  It must remain Unknown rather than being treated as a distinct
// no-alias root, so the intervening load blocks the rewrite.
// CHECK-LABEL: tt.func @reject_unknown_root_load_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.int_to_ptr {{.*}} : i64 -> !tt.ptr<i32>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_unknown_root_load_between_stores(%base: !tt.ptr<i32>,
                                                 %address: i64) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %unknown_base = tt.int_to_ptr %address : i64 -> !tt.ptr<i32>
  %unknown_splat = tt.splat %unknown_base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %unknown_addresses = tt.addptr %unknown_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %unknown_load = tt.load %unknown_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @reject_barrier_between_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: gpu.barrier
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_barrier_between_stores(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  gpu.barrier
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// A valid direct-body same-block value defined before the anchor naturally
// dominates it.  The matcher still checks that condition defensively during
// discovery and revalidation.  This nested-region fixture exercises the
// accompanying direct tt.func body/same-block gate: it must remain untouched.
// CHECK-LABEL: tt.func @reject_nested_region_stores(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_nested_region_stores(%base: !tt.ptr<i32>, %condition: i1) {
  scf.if %condition {
    %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %first_value = arith.constant dense<1> : tensor<4xi32>
    %second_value = arith.constant dense<2> : tensor<4xi32>
    tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
    tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  }
  tt.return
}

// CAP0-LABEL: tt.func @reject_capacity_zero(
// CAP0-NOT: tensor.empty
// CAP0-NOT: tensor.insert_slice
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CAP0: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_capacity_zero(%base: !tt.ptr<i32>) {
  %first_range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %second_range = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_range : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// The two output stores share one dynamic origin.  Both the scalar scale load
// and the dynamically indexed input load use distinct entry roots, so they
// may remain between the delayed K store and final packed store.
// CAP0-LABEL: tt.func @pack_dynamic_origin_distinct_entry_loads(
// CAP0-NOT: tensor.empty
// CAP0-NOT: tensor.insert_slice
// CAP0: tt.store {{.*}}, {{.*}} : tensor<64x!tt.ptr<bf16>>
// CAP0: tt.load {{.*}} : !tt.ptr<f32>
// CAP0: tt.load {{.*}} : tensor<64x!tt.ptr<i8>>
// CAP0: tt.store {{.*}}, {{.*}} : tensor<64x!tt.ptr<bf16>>
// DYNAMIC: tt.func @pack_dynamic_origin_distinct_entry_loads(%{{.*}}: !tt.ptr<i8>, %{{.*}}: !tt.ptr<f32>, %[[OUT_ARG:.*]]: !tt.ptr<bf16>, %[[ORIGIN_ARG:.*]]: i32) {
// DYNAMIC-NOT: tt.store
// DYNAMIC: tt.load {{.*}} : !tt.ptr<f32>
// DYNAMIC: tt.load {{.*}} : tensor<64x!tt.ptr<i8>>
// DYNAMIC-NOT: tt.store
// DYNAMIC: %[[EMPTY:.*]] = tensor.empty() : tensor<128xbf16>
// DYNAMIC: %[[FIRST:.*]] = tensor.insert_slice %{{.*}} into %[[EMPTY]][0] [64] [1] : tensor<64xbf16> into tensor<128xbf16>
// DYNAMIC: %[[PACKED:.*]] = tensor.insert_slice %{{.*}} into %[[FIRST]][64] [64] [1] : tensor<64xbf16> into tensor<128xbf16>
// DYNAMIC: %[[RANGE:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// DYNAMIC: %[[ORIGIN:.*]] = tt.splat %[[ORIGIN_ARG]] : i32 -> tensor<128xi32>
// DYNAMIC: %[[OFFSET:.*]] = arith.addi %[[ORIGIN]], %[[RANGE]] : tensor<128xi32>
// DYNAMIC: %[[OUT:.*]] = tt.splat %[[OUT_ARG]] : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
// DYNAMIC: %[[ADDRESSES:.*]] = tt.addptr %[[OUT]], %[[OFFSET]] : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
// DYNAMIC: tt.store %[[ADDRESSES]], %[[PACKED]] : tensor<128x!tt.ptr<bf16>>
// DYNAMIC-NOT: tt.store
// DYNAMIC: tt.return
tt.func @pack_dynamic_origin_distinct_entry_loads(
    %input: !tt.ptr<i8>, %scale: !tt.ptr<f32>, %out: !tt.ptr<bf16>,
    %origin: i32) {
  %c64 = arith.constant 64 : i32
  %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %origin_splat = tt.splat %origin : i32 -> tensor<64xi32>
  %k_offsets = arith.addi %origin_splat, %range : tensor<64xi32>
  %v_origin = arith.addi %origin, %c64 : i32
  %v_origin_splat = tt.splat %v_origin : i32 -> tensor<64xi32>
  %v_offsets = arith.addi %v_origin_splat, %range : tensor<64xi32>
  %out_splat = tt.splat %out : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
  %k_addresses = tt.addptr %out_splat, %k_offsets : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
  %v_addresses = tt.addptr %out_splat, %v_offsets : tensor<64x!tt.ptr<bf16>>, tensor<64xi32>
  %k_value = arith.constant dense<1.000000e+00> : tensor<64xbf16>
  %v_value = arith.constant dense<2.000000e+00> : tensor<64xbf16>
  tt.store %k_addresses, %k_value : tensor<64x!tt.ptr<bf16>>
  %scale_value = tt.load %scale : !tt.ptr<f32>
  %input_splat = tt.splat %input : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>>
  %input_offsets = arith.addi %origin_splat, %range : tensor<64xi32>
  %input_addresses = tt.addptr %input_splat, %input_offsets : tensor<64x!tt.ptr<i8>>, tensor<64xi32>
  %input_value = tt.load %input_addresses : tensor<64x!tt.ptr<i8>>
  tt.store %v_addresses, %v_value : tensor<64x!tt.ptr<bf16>>
  tt.return
}

// Equal output bases are insufficient when their dynamic origins differ.
// CHECK-LABEL: tt.func @reject_dynamic_origin_mismatch(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_dynamic_origin_mismatch(
    %base: !tt.ptr<i32>, %first_origin: i32, %second_origin: i32) {
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %first_origin_splat = tt.splat %first_origin : i32 -> tensor<4xi32>
  %second_origin_splat = tt.splat %second_origin : i32 -> tensor<4xi32>
  %first_offsets = arith.addi %first_origin_splat, %range : tensor<4xi32>
  %second_offsets = arith.addi %second_origin_splat, %range : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}

// Same-root effects still require interval proof. This middle load overlaps
// the delayed first dynamic-origin store, so it must block packing.
// CHECK-LABEL: tt.func @reject_dynamic_origin_same_root_load(
// CHECK-NOT: tensor.empty
// CHECK-NOT: tensor.insert_slice
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.load {{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: tt.store {{.*}}, {{.*}} : tensor<4x!tt.ptr<i32>>
tt.func @reject_dynamic_origin_same_root_load(%base: !tt.ptr<i32>, %origin: i32) {
  %c4 = arith.constant 4 : i32
  %range = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %origin_splat = tt.splat %origin : i32 -> tensor<4xi32>
  %first_offsets = arith.addi %origin_splat, %range : tensor<4xi32>
  %second_origin = arith.addi %origin, %c4 : i32
  %second_origin_splat = tt.splat %second_origin : i32 -> tensor<4xi32>
  %second_offsets = arith.addi %second_origin_splat, %range : tensor<4xi32>
  %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %first_addresses = tt.addptr %base_splat, %first_offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %second_addresses = tt.addptr %base_splat, %second_offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %first_value = arith.constant dense<1> : tensor<4xi32>
  %second_value = arith.constant dense<2> : tensor<4xi32>
  tt.store %first_addresses, %first_value : tensor<4x!tt.ptr<i32>>
  %same_root_load = tt.load %first_addresses : tensor<4x!tt.ptr<i32>>
  tt.store %second_addresses, %second_value : tensor<4x!tt.ptr<i32>>
  tt.return
}
