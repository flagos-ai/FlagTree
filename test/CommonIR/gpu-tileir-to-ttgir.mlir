// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --convert-common-ir-to-ttgir | FileCheck %s --implicit-check-not=tile. --implicit-check-not=builtin.unrealized_conversion_cast

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module {
  tt.func public @gpu_tileir_to_ttgir(%src: tensor<16x!tt.ptr<f32>>) {
    %buf = tile.alloc {layout = 0 : i64, space = 7 : i64, tle.gpu_layout = #shared}
      : <[2, 16], f32, shared>
    %desc = builtin.unrealized_conversion_cast %buf
      : !tile.buf<[2, 16], f32, shared> to !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
    tle.pipe.create %desc {capacity = 2 : i32, field_names = ["payload"], pipe_name = "p", scope = "cta"}
      : !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
    %c0 = arith.constant 0 : index
    %slot = tile.subview %buf[%c0, %c0] [[16]] [[1]] {tle.gpu_layout = #shared1}
      : <[2, 16], f32, shared> -> <[16], f32, shared>
    tile.copy %src -> %slot {src_layout = 0 : i64}
      : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    %slot_desc = builtin.unrealized_conversion_cast %slot
      : !tile.buf<[16], f32, shared> to !ttg.memdesc<16xf32, #shared1, #smem, mutable>
    %ptr = "tle.local_pointers"(%slot_desc)
      : (!ttg.memdesc<16xf32, #shared1, #smem, mutable>) -> tensor<16x!tt.ptr<f32, 3>>
    tt.return
  }
}

// CHECK: %[[BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x16xf32, #shared, #smem, mutable>
// CHECK: tle.pipe.create %[[BUF]]
// CHECK: %[[SLOT:.*]] = ttg.memdesc_index %[[BUF]][%{{.*}}]
// CHECK: tt.load %{{.*}} : tensor<16x!tt.ptr<f32>>
// CHECK: ttg.local_store %{{.*}}, %[[SLOT]]
// CHECK: "tle.local_pointers"(%[[SLOT]])
// CHECK-NOT: "tle.local_pointers"

// CHECK-LABEL: tt.func public @copy_to_global
// CHECK: %[[LOCAL:.*]] = ttg.local_alloc
// CHECK: ttg.local_store %{{.*}}, %[[LOCAL]]
// CHECK: %[[VALUE:.*]] = ttg.local_load %[[LOCAL]]
// CHECK: tt.store %{{.*}}, %[[VALUE]]
// CHECK-NOT: "tle.local_pointers"
module {
  tt.func public @copy_to_global(%dst: tensor<16x!tt.ptr<f32>>, %value: tensor<16xf32>) {
    %buf = tile.alloc {layout = 0 : i64, space = 7 : i64, tle.gpu_layout = #shared1}
      : <[16], f32, shared>
    tile.store_tensor %value -> %buf : tensor<16xf32>, !tile.buf<[16], f32, shared>
    tile.copy %buf -> %dst {src_layout = 0 : i64}
      : !tile.buf<[16], f32, shared>, tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @static_subview
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc
// CHECK: %[[VIEW:.*]] = ttg.memdesc_subslice %[[ALLOC]][8]
// CHECK: ttg.local_load %[[VIEW]]
module {
  tt.func public @static_subview() -> tensor<8xf32> {
    %buf = tile.alloc {layout = 0 : i64, space = 7 : i64, tle.gpu_layout = #shared1}
      : <[16], f32, shared>
    %c8 = arith.constant 8 : index
    %view = tile.subview %buf[%c8] [[8]] [[1]] {tle.gpu_layout = #shared1}
      : <[16], f32, shared> -> <[8], f32, shared>
    %value = tile.to_tensor %view : <[8], f32, shared> -> tensor<8xf32>
    tt.return %value : tensor<8xf32>
  }
}

// CHECK-LABEL: tt.func public @shared_allocation_subslice_slot
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf16, #shared3, #smem, mutable>
// CHECK: %[[LEFT:.*]] = ttg.memdesc_subslice %[[ALLOC]][0, 0, 0] : !ttg.memdesc<2x16x16xf16, #shared3, #smem, mutable> -> !ttg.memdesc<2x8x16xf16, #shared3, #smem, mutable, 2x16x16>
// CHECK: %[[RIGHT:.*]] = ttg.memdesc_subslice %[[ALLOC]][0, 8, 0] : !ttg.memdesc<2x16x16xf16, #shared3, #smem, mutable> -> !ttg.memdesc<2x8x16xf16, #shared3, #smem, mutable, 2x16x16>
// CHECK: %[[SLOT:.*]] = ttg.memdesc_index %[[RIGHT]][%{{.*}}] : !ttg.memdesc<2x8x16xf16, #shared3, #smem, mutable, 2x16x16> -> !ttg.memdesc<8x16xf16, #shared, #smem, mutable, 2x16x16>
// CHECK: "tle.local_pointers"(%[[LEFT]])
// CHECK: "tle.local_pointers"(%[[SLOT]])
module {
  tt.func public @shared_allocation_subslice_slot() {
    %buf = tile.alloc {layout = 0 : i64, space = 7 : i64, tle.gpu_layout = #shared3}
      : <[2, 16, 16], f16, shared>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %left = tile.subview %buf[%c0, %c0, %c0] [[2, 8, 16]] [[1, 1, 1]] {tle.gpu_layout = #shared3}
      : <[2, 16, 16], f16, shared> -> <[2, 8, 16], f16, shared>
    %right = tile.subview %buf[%c0, %c8, %c0] [[2, 8, 16]] [[1, 1, 1]] {tle.gpu_layout = #shared3}
      : <[2, 16, 16], f16, shared> -> <[2, 8, 16], f16, shared>
    %slot = tile.subview %right[%c0, %c0, %c0] [[8, 16]] [[1, 1]] {tle.gpu_layout = #shared}
      : <[2, 8, 16], f16, shared> -> <[8, 16], f16, shared>
    %left_desc = builtin.unrealized_conversion_cast %left
      : !tile.buf<[2, 8, 16], f16, shared> to !ttg.memdesc<2x8x16xf16, #shared3, #smem, mutable, 2x16x16>
    %slot_desc = builtin.unrealized_conversion_cast %slot
      : !tile.buf<[8, 16], f16, shared> to !ttg.memdesc<8x16xf16, #shared, #smem, mutable, 2x16x16>
    %left_ptr = "tle.local_pointers"(%left_desc)
      : (!ttg.memdesc<2x8x16xf16, #shared3, #smem, mutable, 2x16x16>) -> tensor<2x8x16x!tt.ptr<f16, 3>>
    %slot_ptr = "tle.local_pointers"(%slot_desc)
      : (!ttg.memdesc<8x16xf16, #shared, #smem, mutable, 2x16x16>) -> tensor<8x16x!tt.ptr<f16, 3>>
    tt.return
  }
}
