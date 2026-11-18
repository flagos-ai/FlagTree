// RUN: triton-opt %s --graph-optimize='rule-mask=128' --split-input-file \
// RUN: | FileCheck %s
// RUN: triton-opt %s --graph-optimize='rule-mask=0' --split-input-file \
// RUN: | FileCheck %s --check-prefix=DISABLED

// DiagonalMaskRemoval collapses a diagonal-mask shift of a scan result into a
// single subtraction.  rule-mask=128 isolates the rule so nothing below can be
// attributed to another graph rule.

// -----
// Forward cumulative sum selected by the sub-diagonal (row == col + 1) is a
// left shift, which the forward identity scan[i] - x[i] == scan[i - 1] covers.
// CHECK-LABEL: tt.func public @diagonal_forward_sub_diagonal
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
// CHECK-NOT: tt.reduce
tt.func public @diagonal_forward_sub_diagonal(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// Reverse scan selected by the super-diagonal (row == col - 1) is a right
// shift, covered by scan[i] - x[i] == scan[i + 1].
// CHECK-LABEL: tt.func public @diagonal_reverse_super_diagonal
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
tt.func public @diagonal_reverse_super_diagonal(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.subi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// Integer accumulation has an exact inverse, so an addi scan is rewritten to
// arith.subi.
// CHECK-LABEL: tt.func public @diagonal_integer_cumsum
// CHECK-NOT: tt.reduce
// CHECK: arith.subi
tt.func public @diagonal_integer_cumsum(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                                        %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<i32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %s = arith.addi %a, %b : i32
    tt.scan.return %s : i32
  }) : (tensor<16xi32>) -> tensor<16xi32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %bcast = tt.broadcast %expd : tensor<1x16xi32> -> tensor<16x16xi32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0> : tensor<16x16xi32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xi32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %s = arith.addi %a, %b : i32
    tt.reduce.return %s : i32
  }) : (tensor<16x16xi32>) -> tensor<16xi32>

  %dst_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<i32>>
  tt.return
}

// -----
// The diagonal offset may sit on the 1-D range, before expand_dims, and it may
// be carried by a tt.splat instead of a dense constant.
// CHECK-LABEL: tt.func public @diagonal_offset_before_expand_dims
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
tt.func public @diagonal_offset_before_expand_dims(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                   %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %c1 = arith.constant 1 : i32
  %one_splat = tt.splat %c1 : i32 -> tensor<16xi32>
  %col_shifted = arith.addi %col_range, %one_splat : tensor<16xi32>
  %col_expd = tt.expand_dims %col_shifted {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// `row - 1 == col` denotes the same sub-diagonal as `row == col + 1`, so the
// offset is accepted on the kept side too.
// CHECK-LABEL: tt.func public @diagonal_offset_on_kept_side
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
tt.func public @diagonal_offset_on_kept_side(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                             %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_one = arith.constant dense<1> : tensor<16x1xi32>
  %row_shifted = arith.subi %row_expd, %row_one : tensor<16x1xi32>
  %row_bc = tt.broadcast %row_shifted : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_bc = tt.broadcast %col_expd : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A non-zero make_range start shifts the diagonal exactly like an explicit
// offset, so start 2 against offset 3 is still a unit sub-diagonal.
// CHECK-LABEL: tt.func public @diagonal_non_zero_range_start
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
tt.func public @diagonal_non_zero_range_start(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 18 : i32, start = 2 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %three = arith.constant dense<3> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %three : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// The identity holds per batch row, so a leading batch dimension is supported
// as long as the replicated dimension stays adjacent to the scan axis.
// CHECK-LABEL: tt.func public @diagonal_batched_scan
// CHECK-NOT: tt.reduce
// CHECK: arith.subf
tt.func public @diagonal_batched_scan(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
  %batch = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %batch_expd = tt.expand_dims %batch {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %batch_bc = tt.broadcast %batch_expd : tensor<4x1xi32> -> tensor<4x16xi32>
  %stride = arith.constant dense<16> : tensor<4x16xi32>
  %batch_off = arith.muli %batch_bc, %stride : tensor<4x16xi32>
  %lane = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %lane_expd = tt.expand_dims %lane {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %lane_bc = tt.broadcast %lane_expd : tensor<1x16xi32> -> tensor<4x16xi32>
  %offset = arith.addi %batch_off, %lane_bc : tensor<4x16xi32>
  %src_ptr = tt.addptr %src_base, %offset : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
  %x = tt.load %src_ptr : tensor<4x16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<4x16xf32>) -> tensor<4x16xf32>

  %expd = tt.expand_dims %scan {axis = 1 : i32} : tensor<4x16xf32> -> tensor<4x1x16xf32>
  %bcast = tt.broadcast %expd : tensor<4x1x16xf32> -> tensor<4x16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_lane = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_batch = tt.expand_dims %row_lane {axis = 0 : i32} : tensor<16x1xi32> -> tensor<1x16x1xi32>
  %row_bc = tt.broadcast %row_batch : tensor<1x16x1xi32> -> tensor<4x16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_lane = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_batch = tt.expand_dims %col_lane {axis = 0 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
  %one = arith.constant dense<1> : tensor<1x1x16xi32>
  %col_shifted = arith.addi %col_batch, %one : tensor<1x1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x1x16xi32> -> tensor<4x16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<4x16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<4x16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<4x16x16xi1>, tensor<4x16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 2 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<4x16x16xf32>) -> tensor<4x16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<4x16x!tt.ptr<f32>>, tensor<4x16xi32>
  tt.store %dst_ptr, %red : tensor<4x16x!tt.ptr<f32>>
  tt.return
}

// -----
// A shift of two is a genuine data movement with no O(N) identity, so the
// quadratic form must survive.
// CHECK-LABEL: tt.func public @diagonal_skip_shift_of_two
// CHECK: tt.reduce
tt.func public @diagonal_skip_shift_of_two(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                           %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %two = arith.constant dense<2> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %two : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A non-zero range start that pushes the effective offset to two must not be
// mistaken for a unit shift.  The historical matcher ignored the start and
// would have rewritten this incorrectly.
// CHECK-LABEL: tt.func public @diagonal_skip_range_start_breaks_unit_shift
// CHECK: tt.reduce
tt.func public @diagonal_skip_range_start_breaks_unit_shift(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 17 : i32, start = 1 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A sub-diagonal left shift needs the forward identity; a reverse scan does
// not satisfy it.
// CHECK-LABEL: tt.func public @diagonal_skip_direction_mismatch
// CHECK: tt.reduce
tt.func public @diagonal_skip_direction_mismatch(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                 %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// The unselected lanes must contribute the add identity; a non-zero fill makes
// the boundary element observable.
// CHECK-LABEL: tt.func public @diagonal_skip_non_zero_fill
// CHECK: tt.reduce
tt.func public @diagonal_skip_non_zero_fill(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %fill = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %fill : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A maximum reduce extracts the same element but leaves a different boundary
// value, so it is not covered by the subtraction.
// CHECK-LABEL: tt.func public @diagonal_skip_max_reduce
// CHECK: tt.reduce
tt.func public @diagonal_skip_max_reduce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.maximumf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// Reducing the replicated dimension instead of the scan dimension does not
// select a shifted element at all.
// CHECK-LABEL: tt.func public @diagonal_skip_wrong_reduce_axis
// CHECK: tt.reduce
tt.func public @diagonal_skip_wrong_reduce_axis(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 0 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// Here every type still lines up, but the replicated dimension is inserted in
// front of the batch dimension, so the reduced result is the transpose of the
// scan result and the subtraction would silently change the values.
// CHECK-LABEL: tt.func public @diagonal_skip_replicated_dim_not_adjacent
// CHECK: tt.reduce
tt.func public @diagonal_skip_replicated_dim_not_adjacent(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                                          %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %row = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_e = tt.expand_dims %row {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_b = tt.broadcast %row_e : tensor<16x1xi32> -> tensor<16x16xi32>
  %stride = arith.constant dense<16> : tensor<16x16xi32>
  %row_off = arith.muli %row_b, %stride : tensor<16x16xi32>
  %col = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_e = tt.expand_dims %col {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %col_b = tt.broadcast %col_e : tensor<1x16xi32> -> tensor<16x16xi32>
  %offset = arith.addi %row_off, %col_b : tensor<16x16xi32>
  %src_ptr = tt.addptr %src_base, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
  %x = tt.load %src_ptr : tensor<16x16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16x16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16x16xf32> -> tensor<1x16x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16x16xf32> -> tensor<16x16x16xf32>

  %mrow_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %mrow_lane = tt.expand_dims %mrow_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %mrow_batch = tt.expand_dims %mrow_lane {axis = 2 : i32} : tensor<16x1xi32> -> tensor<16x1x1xi32>
  %mrow_bc = tt.broadcast %mrow_batch : tensor<16x1x1xi32> -> tensor<16x16x16xi32>

  %mcol_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %mcol_lane = tt.expand_dims %mcol_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %mcol_batch = tt.expand_dims %mcol_lane {axis = 0 : i32} : tensor<1x16xi32> -> tensor<1x1x16xi32>
  %one = arith.constant dense<1> : tensor<1x1x16xi32>
  %mcol_shifted = arith.addi %mcol_batch, %one : tensor<1x1x16xi32>
  %mcol_bc = tt.broadcast %mcol_shifted : tensor<1x1x16xi32> -> tensor<16x16x16xi32>

  %cmp = arith.cmpi eq, %mrow_bc, %mcol_bc : tensor<16x16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16x16xi1>, tensor<16x16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 2 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16x16xf32>) -> tensor<16x16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %offset : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
  tt.store %dst_ptr, %red : tensor<16x16x!tt.ptr<f32>>
  tt.return
}

// -----
// The reduced result must have exactly the scan result's type; a rectangular
// broadcast cannot be replaced by an elementwise subtraction.
// CHECK-LABEL: tt.func public @diagonal_skip_non_square
// CHECK: tt.reduce
tt.func public @diagonal_skip_non_square(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                         %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<8x16xf32>

  %row_range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<8x1xi32> -> tensor<8x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<8x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<8x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<8x16xi1>, tensor<8x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<8x16xf32>) -> tensor<8xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
  %dst_range = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %dst_ptr = tt.addptr %dst_base, %dst_range : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
  tt.store %dst_ptr, %red : tensor<8x!tt.ptr<f32>>
  tt.return
}

// -----
// An inequality predicate selects a triangle, not a single diagonal.
// CHECK-LABEL: tt.func public @diagonal_skip_slt_predicate
// CHECK: tt.reduce
tt.func public @diagonal_skip_slt_predicate(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                            %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi slt, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A multiplicative scan has no subtraction inverse.
// CHECK-LABEL: tt.func public @diagonal_skip_mul_scan
// CHECK: tt.reduce
tt.func public @diagonal_skip_mul_scan(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.mulf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}

// -----
// A zero rule mask must leave the pattern alone.  Only the DISABLED run
// inspects this section.
// DISABLED-LABEL: tt.func public @diagonal_disabled_by_rule_mask
// DISABLED: tt.reduce
// DISABLED-NOT: arith.subf
tt.func public @diagonal_disabled_by_rule_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                               %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %src_ptr = tt.addptr %src_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  %x = tt.load %src_ptr : tensor<16x!tt.ptr<f32>>

  %scan = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.scan.return %s : f32
  }) : (tensor<16xf32>) -> tensor<16xf32>

  %expd = tt.expand_dims %scan {axis = 0 : i32} : tensor<16xf32> -> tensor<1x16xf32>
  %bcast = tt.broadcast %expd : tensor<1x16xf32> -> tensor<16x16xf32>

  %row_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %row_expd = tt.expand_dims %row_range {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %row_bc = tt.broadcast %row_expd : tensor<16x1xi32> -> tensor<16x16xi32>

  %col_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %col_expd = tt.expand_dims %col_range {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %one = arith.constant dense<1> : tensor<1x16xi32>
  %col_shifted = arith.addi %col_expd, %one : tensor<1x16xi32>
  %col_bc = tt.broadcast %col_shifted : tensor<1x16xi32> -> tensor<16x16xi32>

  %cmp = arith.cmpi eq, %row_bc, %col_bc : tensor<16x16xi32>
  %zeros = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
  %sel = arith.select %cmp, %bcast, %zeros : tensor<16x16xi1>, tensor<16x16xf32>

  %red = "tt.reduce"(%sel) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %s = arith.addf %a, %b : f32
    tt.reduce.return %s : f32
  }) : (tensor<16x16xf32>) -> tensor<16xf32>

  %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %dst_ptr = tt.addptr %dst_base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  tt.store %dst_ptr, %red : tensor<16x!tt.ptr<f32>>
  tt.return
}
