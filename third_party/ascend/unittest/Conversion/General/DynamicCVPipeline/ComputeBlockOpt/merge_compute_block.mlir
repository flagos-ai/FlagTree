// RUN: triton-opt --allow-unregistered-dialect --merge-compute-block %s | FileCheck %s

// Test merge-compute-block pass: merges adjacent VECTOR blocks between CUBE blocks,
// cloning CUBE ops that downstream blocks depend on.
//
// Pattern: C1(6,CUBE) → V1(32,VECTOR) → C2(8,CUBE) → V2(33,VECTOR) → C3(10,CUBE) → C4(12,CUBE)
// After pass:
//   - V2(33) merged into V1(32): all block_id 33 → 32
//   - C3(10) gets cloned alloc/copy/to_tensor chain from C2(8)

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.inter_core_buf_count = 2 : i32} {
  func.func @_swa_bwd_dkdv_kernel(
    %arg0: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg1: memref<?xbf16> {tt.tensor_kind = 0 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32}
  ) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst_bf16 = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32
    %cst_scale = arith.constant 0.0883883461 : f32
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    %empty_2d_bf16 = tensor.empty() : tensor<64x64xbf16>

    // VECTOR block constants
    %cst_v = arith.constant {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %empty_v = tensor.empty() {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %scale = linalg.fill {ssbuffer.block_id = 49 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_scale : f32) outs(%empty_v : tensor<64x64xf32>) -> tensor<64x64xf32>

    // CUBE block constants
    %cst_bf16_cube = arith.constant {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : bf16
    %cst_f32_cube = arith.constant {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %empty_cube_f32 = tensor.empty() {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32>
    %cube_init = linalg.fill {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_f32_cube : f32) outs(%empty_cube_f32 : tensor<64x64xf32>) -> tensor<64x64xf32>

    scf.for %iv = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
      // Shared index computations used by C1 and C4
      %iv_idx = arith.index_cast %iv {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %offset64 = arith.muli %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index
      %stride_idx = arith.index_cast %arg2 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %cond_c2 = arith.cmpi slt, %iv_idx, %c64 {ssbuffer.block_id = 26 : i32, ssbuffer.core_type = "CUBE"} : index

      // ==========================================
      // C1: Block 6 (CUBE) — produces %tensor_c1 used by V1(32) and C4(12)
      // ==========================================
      // CHECK: memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"}
      %alloc_c1 = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16>
      %cond_c1 = arith.cmpi slt, %iv_idx, %c64 {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : index
      scf.if %cond_c1 {
        linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_bf16 : bf16) outs(%alloc_c1 : memref<64x64xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 6 : i32}
      %reint_c1 = memref.reinterpret_cast %arg0 to offset: [%offset64], sizes: [64, 64], strides: [%stride_idx, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %subv_c1_src = memref.subview %reint_c1[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %subv_c1_dst = memref.subview %alloc_c1[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
      memref.copy %subv_c1_src, %subv_c1_dst {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
      %tensor_c1 = bufferization.to_tensor %alloc_c1 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16> to tensor<64x64xbf16>
      %matmul_c1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%tensor_c1, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V1: Block 32 (VECTOR) — CUBE pred=6, CUBE succ=8, edge 32→33 via %v1_exp
      // ==========================================
      // CHECK: arith.mulf {{%.*}}, {{%.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
      %v1_mul = arith.mulf %matmul_c1, %scale {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: math.exp {{%.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
      %v1_exp = math.exp %v1_mul {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: arith.truncf {{%.*}} {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 32 : i32}
      %v1_trunc = arith.truncf %v1_exp {ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C2: Block 8 (CUBE) — has copy chain with dedicated arith ops that C3(10) depends on
      // ==========================================
      // CHECK: memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      %alloc_c2 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16>
      scf.if %cond_c2 {
        linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_bf16 : bf16) outs(%alloc_c2 : memref<64x64xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
      // Dedicated arith ops in C2 for the reinterpret_cast offset computation
      %c2_stride = arith.index_cast %arg2 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
      %c2_mul = arith.muli %iv_idx, %c2_stride {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
      %c2_offset = arith.addi %offset64, %c2_mul {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : index
      %reint_c2 = memref.reinterpret_cast %arg1 to offset: [%c2_offset], sizes: [64, 64], strides: [%c2_stride, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %subv_c2_src = memref.subview %reint_c2[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %subv_c2_dst = memref.subview %alloc_c2[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16> to memref<64x64xbf16, strided<[64, 1]>>
      // CHECK: memref.copy {{%.*}}, {{%.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      memref.copy %subv_c2_src, %subv_c2_dst {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16, strided<[?, 1], offset: ?>> to memref<64x64xbf16, strided<[64, 1]>>
      // CHECK: bufferization.to_tensor {{%.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      %tensor_c2 = bufferization.to_tensor %alloc_c2 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : memref<64x64xbf16> to tensor<64x64xbf16>
      %transposed_v1 = linalg.transpose ins(%v1_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"}
      %matmul_c2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v1, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // C3: Block 10 (CUBE) — uses %tensor_c2 from C2, triggers clone from C2
      // After pass: cloned alloc/index_cast/muli/addi/reinterpret_cast/subview×2/copy/to_tensor from C2
      // ==========================================
      // CHECK: memref.alloc() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: scf.if {{%.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: } {hivm.unlikely_condition, ssbuffer.block_id = 10 : i32}
      // CHECK: arith.index_cast {{%.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: arith.muli {{%.*}}, {{%.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: arith.addi {{%.*}}, {{%.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: memref.reinterpret_cast {{%.*}} to offset: [{{%.*}}], sizes: [64, 64], strides: [{{%.*}}, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: memref.subview {{%.*}}[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: memref.subview {{%.*}}[0, 0] [64, 64] [1, 1] {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: memref.copy {{%.*}}, {{%.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: bufferization.to_tensor {{%.*}} {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"
      %matmul_c3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%tensor_c2, %tensor_c2 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // ==========================================
      // V2: Block 33 (VECTOR) — uses %v1_exp (edge 32→33), %matmul_c3 (edge 10→33)
      // After pass: merged into V1(32), all block_id 33 → 32
      // ==========================================
      // CHECK: arith.subf {{%.*}}, {{%.*}} {{{.*}}ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
      %v2_sub = arith.subf %matmul_c3, %v1_exp {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: arith.mulf {{%.*}}, {{%.*}} {{{.*}}ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
      %v2_mul = arith.mulf %v1_exp, %v2_sub {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: arith.mulf {{%.*}}, {{%.*}} {{{.*}}ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
      %v2_mul2 = arith.mulf %v2_mul, %scale {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
      // CHECK: arith.truncf {{%.*}} {{{.*}}ssbuffer.block_id = 32 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.subBlock = 33 : i32}
      %v2_trunc = arith.truncf %v2_mul2 {ssbuffer.block_id = 33 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32> to tensor<64x64xbf16>

      // ==========================================
      // C4: Block 12 (CUBE) — uses %v2_trunc (edge 33→12) and %tensor_c1 (edge 6→12)
      // ==========================================
      %transposed_v2 = linalg.transpose ins(%v2_trunc : tensor<64x64xbf16>) outs(%empty_2d_bf16 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"}
      // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"
      %matmul_c4 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed_v2, %tensor_c1 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%cube_init : tensor<64x64xf32>) -> tensor<64x64xf32>

      // Store output to prevent DCE
      %reint_out = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [64, 64], strides: [%stride_idx, 1] {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : memref<?xbf16> to memref<64x64xbf16, strided<[?, 1], offset: ?>>
      %trunc_c4 = arith.truncf %matmul_c4 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<64x64xf32> to tensor<64x64xbf16>
      bufferization.materialize_in_destination %trunc_c4 in writable %reint_out {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : (tensor<64x64xbf16>, memref<64x64xbf16, strided<[?, 1], offset: ?>>) -> ()
    }
    return
  }
}
