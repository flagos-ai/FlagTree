// RUN: triton-opt --plan-cube-block %s -o %t
// RUN: FileCheck %s < %t

// Regression for the PlanCubeBlock non-progress hang exposed after
// MemoryEffectsTracker started tracking subview memory dependencies.
//
// The important shape mirrors the original failure:
//   1. VECTOR subview/copy ops touch sibling VECTOR allocs.
//   2. CUBE allocs are filled under non-cube scf.if ops.
//   3. CUBE to_tensor/annotation/transpose/matmul consume those CUBE allocs.
//
// This used to recurse through a ready but unassigned CUBE op during bypass,
// so Phase 2 later scheduled the same op and released its outgoing edges a
// second time. That drove some indegrees negative and eventually made the
// scheduler stop making progress.

module {
  // CHECK-LABEL: func.func @subview_memory_deps_hang_0
  // CHECK: annotation.mark %{{.*}} {MayImplicitTransposeWithLastAxis, ssbuffer.block_id
  // CHECK: linalg.matmul
  // CHECK-SAME: ssbuffer.block_id
  func.func @subview_memory_deps_hang_0(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_1(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_2(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_3(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_4(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_5(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_6(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
  func.func @subview_memory_deps_hang_7(
      %lhs_gm: memref<?xbf16>,
      %rhs_gm: memref<?xbf16>,
      %acc_init: tensor<128x64xf32>,
      %m_limit: index,
      %n_limit: index,
      %k_limit_i32: i32,
      %lhs_base: index,
      %rhs_base: index,
      %lhs_stride_m: index,
      %lhs_stride_k: index,
      %rhs_stride_n: index,
      %rhs_stride_k: index) -> tensor<128x64xf32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64 = arith.constant 64 : index
    %c64_i64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %zero_bf16 = arith.constant 0.000000e+00 : bf16
    %zero_f32 = arith.constant 0.000000e+00 : f32

    %trip = arith.addi %k_limit_i32, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %ub = arith.divsi %trip, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
    %lhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %rhs_step_i = arith.index_cast %c256_i64 {ssbuffer.core_type = "VECTOR"} : i64 to index
    %lhs_step = arith.muli %lhs_step_i, %lhs_stride_k {ssbuffer.core_type = "VECTOR"} : index
    %rhs_step = arith.muli %rhs_step_i, %rhs_stride_k {ssbuffer.core_type = "VECTOR"} : index

    %result:3 = scf.for %iv = %c0_i32 to %ub step %c1_i32
        iter_args(%acc = %acc_init, %lhs_off = %lhs_base, %rhs_off = %rhs_base)
        -> (tensor<128x64xf32>, index, index) : i32 {
      %rhs_cast = memref.reinterpret_cast %rhs_gm to offset: [%rhs_off], sizes: [64, 256], strides: [%rhs_stride_n, %rhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<64x256xbf16, strided<[?, ?], offset: ?>>
      %lhs_cast = memref.reinterpret_cast %lhs_gm to offset: [%lhs_off], sizes: [128, 256], strides: [%lhs_stride_m, %lhs_stride_k]
          {ssbuffer.core_type = "VECTOR"} : memref<?xbf16> to memref<128x256xbf16, strided<[?, ?], offset: ?>>

      %kk = arith.muli %iv, %c256_i32 {ssbuffer.core_type = "VECTOR"} : i32
      %remaining_i32 = arith.subi %k_limit_i32, %kk {ssbuffer.core_type = "VECTOR"} : i32
      %lhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16>
      %lhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %remaining = arith.index_cast %remaining_i32 {ssbuffer.core_type = "VECTOR"} : i32 to index
      %nonneg = arith.maxsi %remaining, %c0 {ssbuffer.core_type = "VECTOR"} : index
      %k_size0 = arith.minsi %nonneg, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_size = arith.minsi %m_limit, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_size = arith.minsi %k_size0, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %m_pad = arith.cmpi slt, %m_size, %c128 {ssbuffer.core_type = "VECTOR"} : index
      %k_pad0 = arith.cmpi slt, %k_size, %c256 {ssbuffer.core_type = "VECTOR"} : index
      %lhs_pad = arith.ori %m_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %lhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%lhs_cube : memref<128x256xbf16>)
      } {hivm.unlikely_condition}
      %lhs_src = memref.subview %lhs_cast[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %lhs_dst = memref.subview %lhs_vec[0, 0] [%m_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<128x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %lhs_src, %lhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %lhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<128x256xbf16>
      %lhs = bufferization.to_tensor %lhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<128x256xbf16> to tensor<128x256xbf16>
      annotation.mark %lhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<128x256xbf16>

      %rhs_vec = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16>
      %rhs_cube = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %n_end = arith.addi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_max = arith.maxsi %n_limit, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_min = arith.minsi %n_end, %n_max {ssbuffer.core_type = "VECTOR"} : index
      %n_diff = arith.subi %n_min, %n_limit {ssbuffer.core_type = "VECTOR"} : index
      %n_size = arith.minsi %n_diff, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %n_pad = arith.cmpi slt, %n_size, %c64 {ssbuffer.core_type = "VECTOR"} : index
      %rhs_pad = arith.ori %n_pad, %k_pad0 {ssbuffer.core_type = "VECTOR"} : i1
      scf.if %rhs_pad {
        linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_bf16 : bf16) outs(%rhs_cube : memref<64x256xbf16>)
      } {hivm.unlikely_condition}
      %rhs_src = memref.subview %rhs_cast[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
      %rhs_dst = memref.subview %rhs_vec[0, 0] [%n_size, %k_size] [1, 1]
          {ssbuffer.core_type = "VECTOR"} : memref<64x256xbf16> to memref<?x?xbf16, strided<[256, 1]>>
      memref.copy %rhs_src, %rhs_dst {ssbuffer.core_type = "VECTOR"}
          : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[256, 1]>>
      annotation.mark %rhs_cube {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : memref<64x256xbf16>
      %rhs = bufferization.to_tensor %rhs_cube restrict writable {ssbuffer.core_type = "CUBE"}
          : memref<64x256xbf16> to tensor<64x256xbf16>
      annotation.mark %rhs {MayImplicitTransposeWithLastAxis, ssbuffer.core_type = "CUBE"} : tensor<64x256xbf16>

      %rhs_t_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<256x64xbf16>
      %rhs_t = linalg.transpose ins(%rhs : tensor<64x256xbf16>) outs(%rhs_t_empty : tensor<256x64xbf16>)
          permutation = [1, 0] {ssbuffer.core_type = "CUBE"}
      %out_empty = tensor.empty() {ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
      %out_init = linalg.fill {ssbuffer.core_type = "CUBE"} ins(%zero_f32 : f32) outs(%out_empty : tensor<128x64xf32>) -> tensor<128x64xf32>
      %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
          ins(%lhs, %rhs_t : tensor<128x256xbf16>, tensor<256x64xbf16>)
          outs(%out_init : tensor<128x64xf32>) -> tensor<128x64xf32>
      %next = arith.addf %mm, %acc {ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
      %next_lhs = arith.addi %lhs_off, %lhs_step {ssbuffer.core_type = "VECTOR"} : index
      %next_rhs = arith.addi %rhs_off, %rhs_step {ssbuffer.core_type = "VECTOR"} : index
      scf.yield %next, %next_lhs, %next_rhs : tensor<128x64xf32>, index, index
    }

    return %result#0 : tensor<128x64xf32>
  }
}
