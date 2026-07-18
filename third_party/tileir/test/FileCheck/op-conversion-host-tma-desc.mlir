// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.experimental\$func(fuse-fma)),reconcile-unrealized-casts)" | FileCheck %s

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
