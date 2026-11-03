// RUN: triton-opt %s -test-tritonhcugpu-membar | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  // CHECK-LABEL: @reuse_after_async_wait
  tt.func @reuse_after_async_wait(%A: !tt.ptr<f32>, %offsets: tensor<16x64xi32, #blocked>, %out: tensor<16x64x!tt.ptr<f32>, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<16x64xf32, #shared, #smem, mutable>
    %copy = hcug.buffer_load_to_local %A[%offsets] into %buf {contiguity = 1 : i32} : <f32>[tensor<16x64xi32, #blocked>] -> <16x64xf32, #shared, #smem, mutable>
    %wait = hcug.async_wait %copy {num_inst = 0 : i32}
    ttg.local_barrier
    %value = ttg.local_load %buf token %wait : !ttg.memdesc<16x64xf32, #shared, #smem, mutable> -> tensor<16x64xf32, #blocked>
    tt.store %out, %value : tensor<16x64x!tt.ptr<f32>, #blocked>
    // The preceding wait protects RAW only. A second copy must wait for all
    // waves to finish reading the reused tile (WAR).
    // CHECK: ttg.local_load
    // CHECK: tt.store
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}hcug.buffer_load_to_local
    %copy2 = hcug.buffer_load_to_local %A[%offsets] into %buf {contiguity = 1 : i32} : <f32>[tensor<16x64xi32, #blocked>] -> <16x64xf32, #shared, #smem, mutable>
    %wait2 = hcug.async_wait %copy2 {num_inst = 0 : i32}
    ttg.local_barrier
    %value2 = ttg.local_load %buf token %wait2 : !ttg.memdesc<16x64xf32, #shared, #smem, mutable> -> tensor<16x64xf32, #blocked>
    tt.store %out, %value2 : tensor<16x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
