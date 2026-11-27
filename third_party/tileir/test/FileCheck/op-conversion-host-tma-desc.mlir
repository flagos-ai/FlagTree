// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(inline,convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts)" | FileCheck %s

module {
  tt.func public @host_tma_load_store(%in_desc: !tt.tensordesc<tensor<2x128xf16>>, %in_desc_0: !tt.ptr<f16>, %in_desc_1: i32, %in_desc_2: i32 {tt.divisibility = 16 : i32}, %in_desc_3: i64, %in_desc_4: i64, %out_desc: !tt.tensordesc<tensor<2x128xf16>>, %out_desc_5: !tt.ptr<f16>, %out_desc_6: i32, %out_desc_7: i32 {tt.divisibility = 16 : i32}, %out_desc_8: i64, %out_desc_9: i64) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x128xf16>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<2x128xf16>> -> tensor<2x128xf16>
    %1 = arith.addf %0, %cst : tensor<2x128xf16>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %1 : !tt.tensordesc<tensor<2x128xf16>>, tensor<2x128xf16>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_load_store
// CHECK-SAME: {{.*}}, {{.*}}, %[[ARG2:.*]]: tile<i32>, %[[ARG3:.*]]: tile<i32>, %[[ARG4:.+]]: tile<i64>, {{.*}}: tile<i64>, {{.*}}, {{.*}}, %[[ARG8:.*]]: tile<i32>, %[[ARG9:.*]]: tile<i32>, {{.*}}
// CHECK: %[[ASSUME2:.*]] = assume div_by<16>, %[[ARG9]] : tile<i32>
// CHECK: %[[ASSUME1:.*]] = assume div_by<16>, %[[ARG3]] : tile<i32>
// CHECK: %[[EXT8:.*]] = exti %[[ARG8]] signed : tile<i32> -> tile<i64>
// Verify shape bounds use TMA hardware limit (2^32 - 1 = 4294967295)
// CHECK: assume bounded<0, 4294967295>, %[[EXT8]] : tile<i64>
// CHECK: %[[EXT9:.*]] = exti %[[ASSUME2]] signed : tile<i32> -> tile<i64>
// CHECK: assume bounded<0, 4294967295>, %[[EXT9]] : tile<i64>
// CHECK: %[[VIEW2:.*]] = make_tensor_view {{.*}}, shape = [{{.*}}, {{.*}}], {{.*}}

// Verify stride bounds use TMA hardware limit (2^40 - 1 = 1099511627775)
// CHECK: %[[EXT2:.*]] = exti %[[ARG2]] signed : tile<i32> -> tile<i64>
// CHECK: assume bounded<0, 4294967295>, %[[EXT2]] : tile<i64>
// CHECK: %[[EXT3:.*]] = exti %[[ASSUME1]] signed : tile<i32> -> tile<i64>
// CHECK: assume bounded<0, 4294967295>, %[[EXT3]] : tile<i64>
// CHECK: %{{.+}} = assume bounded<0, 1099511627775>, %[[ARG4]] : tile<i64>
// CHECK: %[[VIEW1:.*]] = make_tensor_view {{.*}}, shape = [{{.*}}, {{.*}}], {{.*}}

// -----

module {
  tt.func public @host_tma_nan_padding(%in_desc: !tt.tensordesc<tensor<2x128xf16>> {tileir.padding_nan = 1 : i32}, %in_desc_0: !tt.ptr<f16>, %in_desc_1: i32, %in_desc_2: i32 {tt.divisibility = 16 : i32}, %in_desc_3: i64, %in_desc_4: i64, %out_desc: !tt.tensordesc<tensor<2x128xf16>>, %out_desc_5: !tt.ptr<f16>, %out_desc_6: i32, %out_desc_7: i32 {tt.divisibility = 16 : i32}, %out_desc_8: i64, %out_desc_9: i64) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x128xf16>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.descriptor_load %in_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<2x128xf16>> -> tensor<2x128xf16>
    %1 = arith.addf %0, %cst : tensor<2x128xf16>
    tt.descriptor_store %out_desc[%c0_i32, %c0_i32], %1 : !tt.tensordesc<tensor<2x128xf16>>, tensor<2x128xf16>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_nan_padding
// CHECK-NOT: tileir.padding_nan
// CHECK: make_strided_view {{.*}} padding_value = nan
// CHECK: load_view_tko {{.*}} padding_value = nan

// -----

module {
  tt.func public @host_tma_f32_zero_round_0(
      %desc: !tt.tensordesc<tensor<1x16xf32>>,
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %rows = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %loaded = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<1x16xf32>> -> tensor<1x16xf32>
    tt.descriptor_store %desc[%zero, %zero], %loaded : !tt.tensordesc<tensor<1x16xf32>>, tensor<1x16xf32>
    %gathered = tt.descriptor_gather %desc[%rows, %zero] : (!tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32) -> tensor<8x16xf32>
    tt.descriptor_scatter %desc[%rows, %zero], %gathered : !tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32, tensor<8x16xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_f32_zero_round_0
// CHECK-NOT: tileir.round_f32_to_tf32
// CHECK-NOT: ptr<tf32>
// CHECK: make_tensor_view {{.*}}tensor_view<?x?xf32
// CHECK: make_strided_view {{.*}}padding_value = zero
// CHECK-NOT: ftof
// CHECK: load_view_tko {{.*}}strided_view<{{.*}}padding_value = zero
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: make_gather_scatter_view {{.*}}padding_value = zero
// CHECK: load_view_tko {{.*}}gather_scatter_view
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: return

// -----

module {
  tt.func public @host_tma_f32_nan_round_0(
      %desc: !tt.tensordesc<tensor<1x16xf32>> {tileir.padding_nan = 1 : i32},
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %rows = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %loaded = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<1x16xf32>> -> tensor<1x16xf32>
    tt.descriptor_store %desc[%zero, %zero], %loaded : !tt.tensordesc<tensor<1x16xf32>>, tensor<1x16xf32>
    %gathered = tt.descriptor_gather %desc[%rows, %zero] : (!tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32) -> tensor<8x16xf32>
    tt.descriptor_scatter %desc[%rows, %zero], %gathered : !tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32, tensor<8x16xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_f32_nan_round_0
// CHECK-NOT: tileir.round_f32_to_tf32
// CHECK-NOT: ptr<tf32>
// CHECK: make_tensor_view {{.*}}tensor_view<?x?xf32
// CHECK: make_strided_view {{.*}}padding_value = nan
// CHECK-NOT: ftof
// CHECK: load_view_tko {{.*}}strided_view<{{.*}}padding_value = nan
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: make_gather_scatter_view {{.*}}padding_value = nan
// CHECK: load_view_tko {{.*}}gather_scatter_view
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: return

// -----

module {
  tt.func public @host_tma_f32_zero_round_1(
      %desc: !tt.tensordesc<tensor<1x16xf32>> {tileir.round_f32_to_tf32 = 1 : i32},
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %rows = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %loaded = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<1x16xf32>> -> tensor<1x16xf32>
    tt.descriptor_store %desc[%zero, %zero], %loaded : !tt.tensordesc<tensor<1x16xf32>>, tensor<1x16xf32>
    %gathered = tt.descriptor_gather %desc[%rows, %zero] : (!tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32) -> tensor<8x16xf32>
    tt.descriptor_scatter %desc[%rows, %zero], %gathered : !tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32, tensor<8x16xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_f32_zero_round_1
// CHECK-NOT: tileir.round_f32_to_tf32
// CHECK-NOT: ptr<tf32>
// CHECK: make_tensor_view {{.*}}tensor_view<?x?xf32
// CHECK: make_strided_view {{.*}}padding_value = zero
// CHECK-NOT: ftof
// CHECK: load_view_tko {{.*}}strided_view<{{.*}}padding_value = zero
// CHECK: ftof {{.*}} : tile<1x16xf32> -> tile<1x16xtf32>
// CHECK: ftof {{.*}} : tile<1x16xtf32> -> tile<1x16xf32>
// CHECK: andi
// CHECK: cmpi
// CHECK: select {{.*}} : tile<1x16xi1>, tile<1x16xi32>
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: make_gather_scatter_view {{.*}}padding_value = zero
// CHECK: load_view_tko {{.*}}gather_scatter_view
// CHECK: ftof {{.*}} : tile<8x16xf32> -> tile<8x16xtf32>
// CHECK: ftof {{.*}} : tile<8x16xtf32> -> tile<8x16xf32>
// CHECK: andi
// CHECK: cmpi
// CHECK: select {{.*}} : tile<8x16xi1>, tile<8x16xi32>
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: return

// -----

module {
  tt.func public @host_tma_f32_nan_round_1(
      %desc: !tt.tensordesc<tensor<1x16xf32>> {tileir.padding_nan = 1 : i32, tileir.round_f32_to_tf32 = 1 : i32},
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %rows = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %loaded = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<1x16xf32>> -> tensor<1x16xf32>
    tt.descriptor_store %desc[%zero, %zero], %loaded : !tt.tensordesc<tensor<1x16xf32>>, tensor<1x16xf32>
    %gathered = tt.descriptor_gather %desc[%rows, %zero] : (!tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32) -> tensor<8x16xf32>
    tt.descriptor_scatter %desc[%rows, %zero], %gathered : !tt.tensordesc<tensor<1x16xf32>>, tensor<8xi32>, i32, tensor<8x16xf32>
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_f32_nan_round_1
// CHECK-NOT: tileir.round_f32_to_tf32
// CHECK-NOT: ptr<tf32>
// CHECK: make_tensor_view {{.*}}tensor_view<?x?xf32
// CHECK: make_strided_view {{.*}}padding_value = nan
// CHECK-NOT: ftof
// CHECK: load_view_tko {{.*}}strided_view<{{.*}}padding_value = nan
// CHECK: ftof {{.*}} : tile<1x16xf32> -> tile<1x16xtf32>
// CHECK: ftof {{.*}} : tile<1x16xtf32> -> tile<1x16xf32>
// CHECK: andi
// CHECK: cmpi
// CHECK: select {{.*}} : tile<1x16xi1>, tile<1x16xi32>
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: make_gather_scatter_view {{.*}}padding_value = nan
// CHECK: load_view_tko {{.*}}gather_scatter_view
// CHECK: ftof {{.*}} : tile<8x16xf32> -> tile<8x16xtf32>
// CHECK: ftof {{.*}} : tile<8x16xtf32> -> tile<8x16xf32>
// CHECK: andi
// CHECK: cmpi
// CHECK: select {{.*}} : tile<8x16xi1>, tile<8x16xi32>
// CHECK-NOT: ftof
// CHECK: store_view_tko
// CHECK-NOT: ftof
// CHECK: return

// -----

module {
  tt.func private @read_round_descriptor(%desc: !tt.tensordesc<tensor<1x16xf32>>) -> tensor<1x16xf32> attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %value = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<1x16xf32>> -> tensor<1x16xf32>
    tt.return %value : tensor<1x16xf32>
  }
  tt.func public @host_tma_round_helper_loop_capture(
      %desc: !tt.tensordesc<tensor<1x16xf32>> {tileir.round_f32_to_tf32 = 1 : i32},
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    scf.for %i = %zero to %two step %one : i32 {
      %value = tt.call @read_round_descriptor(%desc) : (!tt.tensordesc<tensor<1x16xf32>>) -> tensor<1x16xf32>
      tt.descriptor_store %desc[%i, %zero], %value : !tt.tensordesc<tensor<1x16xf32>>, tensor<1x16xf32>
    }
    tt.return
  }
}

// CHECK-LABEL: entry @host_tma_round_helper_loop_capture
// CHECK: for
// CHECK: load_view_tko
// CHECK: ftof {{.*}} : tile<1x16xf32> -> tile<1x16xtf32>
// CHECK: ftof {{.*}} : tile<1x16xtf32> -> tile<1x16xf32>
// CHECK: select {{.*}} : tile<1x16xi1>, tile<1x16xi32>
// CHECK-NOT: ftof
// CHECK: store_view_tko


// -----

// Atomic writes discard only their own view padding. Host read rounding remains
// on the subsequent load and must not alter the atomic contribution.
module {
  tt.func public @host_descriptor_atomic_add(
      %desc: !tt.tensordesc<tensor<8x16xf32>> {tileir.padding_nan = 1 : i32, tileir.round_f32_to_tf32 = 1 : i32},
      %ptr: !tt.ptr<f32>, %m: i32, %n: i32,
      %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %value = arith.constant dense<1.0> : tensor<8x16xf32>
    tt.descriptor_reduce add, %desc[%zero, %zero], %value : !tt.tensordesc<tensor<8x16xf32>>, tensor<8x16xf32>
    %loaded = tt.descriptor_load %desc[%zero, %zero] : !tt.tensordesc<tensor<8x16xf32>> -> tensor<8x16xf32>
    tt.descriptor_store %desc[%zero, %zero], %loaded : !tt.tensordesc<tensor<8x16xf32>>, tensor<8x16xf32>
    tt.return
  }
}
// CHECK-LABEL: entry @host_descriptor_atomic_add
// CHECK: make_strided_view {{.*}} padding_value = nan
// CHECK: %[[WRITE:.*]] = make_strided_view {{.*}} : strided_view<tile=(8x16), traversal_strides=[1,1], tensor_view<
// CHECK-NOT: ftof
// CHECK: atomic_red_view_tko relaxed device %[[WRITE]][{{.*}}], addf,
// CHECK: load_view_tko {{.*}} padding_value = nan
// CHECK: ftof

// -----

module {
  tt.func public @device_descriptor_atomic_unsigned(
      %ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %m: i32, %n: i32)
      attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i64
    %stride = arith.extsi %n : i32 to i64
    %desc = tt.make_tensor_descriptor %ptr, [%m, %n], [%stride, %one] : <i32>, <tensor<8x16xui32>>
    %value = arith.constant dense<-1> : tensor<8x16xi32>
    tt.descriptor_reduce min, %desc[%zero, %zero], %value : !tt.tensordesc<tensor<8x16xui32>>, tensor<8x16xi32>
    tt.descriptor_reduce max, %desc[%zero, %zero], %value : !tt.tensordesc<tensor<8x16xui32>>, tensor<8x16xi32>
    tt.return
  }
}
// CHECK-LABEL: entry @device_descriptor_atomic_unsigned
// CHECK: atomic_red_view_tko relaxed device {{.*}}, umin,
// CHECK: atomic_red_view_tko relaxed device {{.*}}, umax,

// -----

module {
  tt.func public @host_descriptor_atomic_signed(
      %desc: !tt.tensordesc<tensor<8x16xi32>>, %ptr: !tt.ptr<i32>,
      %m: i32, %n: i32, %sm: i64, %sn: i64) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %value = arith.constant dense<-1> : tensor<8x16xi32>
    tt.descriptor_reduce min, %desc[%zero, %zero], %value : !tt.tensordesc<tensor<8x16xi32>>, tensor<8x16xi32>
    tt.descriptor_reduce max, %desc[%zero, %zero], %value : !tt.tensordesc<tensor<8x16xi32>>, tensor<8x16xi32>
    tt.return
  }
}
// CHECK-LABEL: entry @host_descriptor_atomic_signed
// CHECK: atomic_red_view_tko relaxed device {{.*}}, min,
// CHECK: atomic_red_view_tko relaxed device {{.*}}, max,
