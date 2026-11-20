// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
// RUN: triton-opt --canonicalize --triton-reorder-broadcast --triton-to-linalg --split-input-file %s | FileCheck %s

// Start at the TD -> HIVM boundary: no distributed dialect is needed in TA.
// A custom result used only by a load must not make its pointer input a
// MetaUse clone. The pointer and token must both survive custom conversion.
// CHECK-LABEL: func.func @consume_splat_zero(
// CHECK-SAME: %[[SRC:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[DST:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[TOKEN:arg[0-9]+]]: i32
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [0], sizes: [1], strides: [1]
// CHECK: %[[CONSUMED:.*]] = hivm.hir.custom
// CHECK-SAME: "dist.aclshmem_consume_token_int64_ptr_1d" ins(%[[INPUT]], %[[TOKEN]] : memref<1xi64, strided<[1]>>, i32) -> memref<1xi64>
// CHECK: %[[READ:.*]] = memref.reinterpret_cast %[[CONSUMED]] to offset: [0], sizes: [1], strides: [1]
// CHECK: memref.copy %[[READ]],
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @consume_splat_zero(%src: !tt.ptr<i64>, %dst: !tt.ptr<i64>, %token: i32) {
    %zero = arith.constant 0 : i32
    %base = tt.addptr %src, %zero : !tt.ptr<i64>, i32
    %ptrs = tt.splat %base : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %empty = tensor.empty() : tensor<1x!tt.ptr<i64>>
    %consumed = hivm.hir.custom
        {hivm.is_distributed, SrcPtrIndex = array<i32: 0>, no_side_effect,
         hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         symbol = "aclshmem_consume_token_int64_ptr_1d"}
        "dist.aclshmem_consume_token_int64_ptr_1d"
        ins(%ptrs, %token : tensor<1x!tt.ptr<i64>>, i32)
        outs(%empty : tensor<1x!tt.ptr<i64>>) -> tensor<1x!tt.ptr<i64>>
    %value = tt.load %consumed : tensor<1x!tt.ptr<i64>>
    %out = tt.splat %dst : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    tt.store %out, %value : tensor<1x!tt.ptr<i64>>
    tt.return
  }
}

// -----

// ReorderBroadcast may turn addptr(splat(base), splat(offset)) into
// splat(addptr(base, offset)). Preserve the nonzero input address and token.
// CHECK-LABEL: func.func @consume_splat_offset(
// CHECK-SAME: %[[SRC:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[DST:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[TOKEN:arg[0-9]+]]: i32
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [7], sizes: [1], strides: [1]
// CHECK: %[[CONSUMED:.*]] = hivm.hir.custom
// CHECK-SAME: "dist.aclshmem_consume_token_int64_ptr_1d" ins(%[[INPUT]], %[[TOKEN]] : memref<1xi64, strided<[1], offset: 7>>, i32) -> memref<1xi64>
// CHECK: %[[READ:.*]] = memref.reinterpret_cast %[[CONSUMED]] to offset: [7], sizes: [1], strides: [1]
// CHECK: memref.copy %[[READ]],
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @consume_splat_offset(%src: !tt.ptr<i64>, %dst: !tt.ptr<i64>, %token: i32) {
    %offsets = arith.constant dense<7> : tensor<1xi32>
    %base = tt.splat %src : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %ptrs = tt.addptr %base, %offsets : tensor<1x!tt.ptr<i64>>, tensor<1xi32>
    %empty = tensor.empty() : tensor<1x!tt.ptr<i64>>
    %consumed = hivm.hir.custom
        {hivm.is_distributed, SrcPtrIndex = array<i32: 0>, no_side_effect,
         hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         symbol = "aclshmem_consume_token_int64_ptr_1d"}
        "dist.aclshmem_consume_token_int64_ptr_1d"
        ins(%ptrs, %token : tensor<1x!tt.ptr<i64>>, i32)
        outs(%empty : tensor<1x!tt.ptr<i64>>) -> tensor<1x!tt.ptr<i64>>
    %value = tt.load %consumed : tensor<1x!tt.ptr<i64>>
    %out = tt.splat %dst : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    tt.store %out, %value : tensor<1x!tt.ptr<i64>>
    tt.return
  }
}

// -----

// The same pointer has a metadata user (load) and a custom consumer. Only the
// ordinary load may use the metadata clone; the custom needs a live memref.
// CHECK-LABEL: func.func @consume_shared_splat(
// CHECK-SAME: %[[SRC:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[DST:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[TOKEN:arg[0-9]+]]: i32
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [0], sizes: [1], strides: [1]
// CHECK: %[[CONSUMED:.*]] = hivm.hir.custom
// CHECK-SAME: "dist.aclshmem_consume_token_int64_ptr_1d" ins(%[[INPUT]], %[[TOKEN]] : memref<1xi64, strided<[1]>>, i32) -> memref<1xi64>
// CHECK: %[[READ:.*]] = memref.reinterpret_cast %[[CONSUMED]] to offset: [0], sizes: [1], strides: [1]
// CHECK: memref.copy %[[READ]],
// CHECK: arith.addi
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @consume_shared_splat(%src: !tt.ptr<i64>, %dst: !tt.ptr<i64>, %token: i32) {
    %ptrs = tt.splat %src : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %before = tt.load %ptrs : tensor<1x!tt.ptr<i64>>
    %empty = tensor.empty() : tensor<1x!tt.ptr<i64>>
    %consumed = hivm.hir.custom
        {hivm.is_distributed, SrcPtrIndex = array<i32: 0>, no_side_effect,
         hivm.pipe = #hivm.pipe<PIPE_S>, hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         symbol = "aclshmem_consume_token_int64_ptr_1d"}
        "dist.aclshmem_consume_token_int64_ptr_1d"
        ins(%ptrs, %token : tensor<1x!tt.ptr<i64>>, i32)
        outs(%empty : tensor<1x!tt.ptr<i64>>) -> tensor<1x!tt.ptr<i64>>
    %after = tt.load %consumed : tensor<1x!tt.ptr<i64>>
    %sum = arith.addi %before, %after : tensor<1xi64>
    %out = tt.splat %dst : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    tt.store %out, %sum : tensor<1x!tt.ptr<i64>>
    tt.return
  }
}

// -----

// CustomMacroOp follows the same operand-use contract as CustomOp.
// CHECK-LABEL: func.func @consume_macro_splat(
// CHECK-SAME: %[[SRC:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[DST:arg[0-9]+]]: memref<?xi64>{{[^%]*}}%[[TOKEN:arg[0-9]+]]: i32
// CHECK: %[[INPUT:.*]] = memref.reinterpret_cast %[[SRC]] to offset: [0], sizes: [1], strides: [1]
// CHECK: %[[CONSUMED:.*]] = hivm.hir.custom_macro
// CHECK-SAME: "consume_macro" ins(%[[INPUT]], %[[TOKEN]] : memref<1xi64, strided<[1]>>, i32) -> memref<1xi64>
// CHECK: %[[READ:.*]] = memref.reinterpret_cast %[[CONSUMED]] to offset: [0], sizes: [1], strides: [1]
// CHECK: memref.copy %[[READ]],
// CHECK: return
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @consume_macro_splat(%src: !tt.ptr<i64>, %dst: !tt.ptr<i64>, %token: i32) {
    %ptrs = tt.splat %src : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    %empty = tensor.empty() : tensor<1x!tt.ptr<i64>>
    %consumed = hivm.hir.custom_macro
        {hivm.is_distributed, SrcPtrIndex = array<i32: 0>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>, hivm.pipe_out = #hivm.pipe<PIPE_V>,
         hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "consume_macro_impl"}
        "consume_macro"
        ins(%ptrs, %token : tensor<1x!tt.ptr<i64>>, i32)
        outs(%empty : tensor<1x!tt.ptr<i64>>) -> tensor<1x!tt.ptr<i64>>
    %value = tt.load %consumed : tensor<1x!tt.ptr<i64>>
    %out = tt.splat %dst : !tt.ptr<i64> -> tensor<1x!tt.ptr<i64>>
    tt.store %out, %value : tensor<1x!tt.ptr<i64>>
    tt.return
  }
}
