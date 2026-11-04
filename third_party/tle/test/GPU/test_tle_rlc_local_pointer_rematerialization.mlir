// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions -triton-tle-select-encodings -tritongpu-remove-layout-conversions -tritongpu-reduce-data-duplication | FileCheck %s

// A shared descriptor's storage layout is not a register layout. Only the
// tensor indices and pointer result share the encoding propagated by RLC.
// The descriptor can have a different rank, and must remain the same SSA value.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 5}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @rematerialize_local_pointer(
  // CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]:
  tt.func @rematerialize_local_pointer(%base: !ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>, %left: tensor<2x64xbf16, #lhs>, %output: tensor<2x128x!tt.ptr<f32>, #mma>) {
    %zero_index = arith.constant dense<0> : tensor<64x128xi32, #blocked>
    %rows = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %columns = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %column = tt.expand_dims %columns {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %row_index = tt.broadcast %row : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %column_index = tt.broadcast %column : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    // CHECK: %[[PTR:.*]] = "tle.local_pointers"(%[[BASE]], {{.*}}) : {{.*}} -> tensor<64x128x!tt.ptr<bf16, 3>, #ttg.dot_op<{{.*}}>>
    %ptr = "tle.local_pointers"(%base, %zero_index, %zero_index, %row_index, %zero_index, %column_index) : (!ttg.memdesc<1x1x64x1x128xbf16, #shared, #smem, mutable>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>, tensor<64x128xi32, #blocked>) -> tensor<64x128x!tt.ptr<bf16, 3>, #blocked>
    %converted = ttg.convert_layout %ptr : tensor<64x128x!tt.ptr<bf16, 3>, #blocked> -> tensor<64x128x!tt.ptr<bf16, 3>, #rhs>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %[[VALUE:.*]] = tt.load %[[PTR]] {{.*}} : tensor<64x128x!tt.ptr<bf16, 3>, #ttg.dot_op<{{.*}}>>
    %value = tt.load %converted {isVolatile = true, tle.explicit_memory_encoding = #rhs} : tensor<64x128x!tt.ptr<bf16, 3>, #rhs>
    %zero = arith.constant dense<0.0> : tensor<2x128xf32, #mma>
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    // CHECK-NOT: tt.load
    // CHECK: tt.dot %{{.*}}, %[[VALUE]],
    %dot = tt.dot %left, %value, %zero : tensor<2x64xbf16, #lhs> * tensor<64x128xbf16, #rhs> -> tensor<2x128xf32, #mma>
    tt.store %output, %dot : tensor<2x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
