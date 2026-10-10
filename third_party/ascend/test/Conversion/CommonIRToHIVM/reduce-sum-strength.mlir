// Copyright 2026- Xcoresigma Technology Co., Ltd

// RUN: triton-opt %s -split-input-file -reduce-sum-strength | FileCheck %s

module {
  // CHECK-LABEL: tt.func @rewrite_f32_512
  tt.func @rewrite_f32_512(%arg0: tensor<512xf32>) -> f32 {
    // 512 -> 256 -> 128, final reduce on 128.
    // CHECK: %[[LO:.*]] = tensor.extract_slice %arg0[0] [256] [1] : tensor<512xf32> to tensor<256xf32>
    // CHECK: %[[HI:.*]] = tensor.extract_slice %arg0[256] [256] [1] : tensor<512xf32> to tensor<256xf32>
    // CHECK: %[[ADD:.*]] = arith.addf %[[LO]], %[[HI]] : tensor<256xf32>
    // CHECK: tensor.extract_slice %[[ADD]][0] [128] [1] : tensor<256xf32> to tensor<128xf32>
    // CHECK: tensor.extract_slice %[[ADD]][128] [128] [1] : tensor<256xf32> to tensor<128xf32>
    // CHECK: arith.addf {{.*}} : tensor<128xf32>
    // CHECK: "tt.reduce"({{.*}}) <{axis = 0 : i32}> ({
    // CHECK: }) : (tensor<128xf32>) -> f32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<512xf32>) -> f32
    tt.return %0 : f32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @rewrite_f16_8192
  tt.func @rewrite_f16_8192(%arg0: tensor<8192xf16>) -> f16 {
    // 8192 -> 4096 -> 2048 -> 1024 -> 512 -> 256, final reduce on 256.
    // CHECK: tensor.extract_slice %arg0[0] [4096] [1] : tensor<8192xf16> to tensor<4096xf16>
    // CHECK: tensor.extract_slice %arg0[4096] [4096] [1] : tensor<8192xf16> to tensor<4096xf16>
    // CHECK: arith.addf {{.*}} : tensor<4096xf16>
    // CHECK: tensor.extract_slice {{.*}}[0] [2048] [1] : tensor<4096xf16> to tensor<2048xf16>
    // CHECK: tensor.extract_slice {{.*}}[2048] [2048] [1] : tensor<4096xf16> to tensor<2048xf16>
    // CHECK: arith.addf {{.*}} : tensor<2048xf16>
    // CHECK: tensor.extract_slice {{.*}}[0] [1024] [1] : tensor<2048xf16> to tensor<1024xf16>
    // CHECK: tensor.extract_slice {{.*}}[1024] [1024] [1] : tensor<2048xf16> to tensor<1024xf16>
    // CHECK: arith.addf {{.*}} : tensor<1024xf16>
    // CHECK: tensor.extract_slice {{.*}}[0] [512] [1] : tensor<1024xf16> to tensor<512xf16>
    // CHECK: tensor.extract_slice {{.*}}[512] [512] [1] : tensor<1024xf16> to tensor<512xf16>
    // CHECK: arith.addf {{.*}} : tensor<512xf16>
    // CHECK: tensor.extract_slice {{.*}}[0] [256] [1] : tensor<512xf16> to tensor<256xf16>
    // CHECK: tensor.extract_slice {{.*}}[256] [256] [1] : tensor<512xf16> to tensor<256xf16>
    // CHECK: arith.addf {{.*}} : tensor<256xf16>
    // CHECK: "tt.reduce"
    // CHECK: }) : (tensor<256xf16>) -> f16
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f16, %arg2: f16):
      %1 = arith.addf %arg1, %arg2 : f16
      tt.reduce.return %1 : f16
    }) : (tensor<8192xf16>) -> f16
    tt.return %0 : f16
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @rewrite_bf16_512
  tt.func @rewrite_bf16_512(%arg0: tensor<512xbf16>) -> bf16 {
    // 512 -> 256, final reduce on 256.
    // CHECK: tensor.extract_slice %arg0[0] [256] [1] : tensor<512xbf16> to tensor<256xbf16>
    // CHECK: tensor.extract_slice %arg0[256] [256] [1] : tensor<512xbf16> to tensor<256xbf16>
    // CHECK: arith.addf {{.*}} : tensor<256xbf16>
    // CHECK: "tt.reduce"
    // CHECK: }) : (tensor<256xbf16>) -> bf16
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: bf16, %arg2: bf16):
      %1 = arith.addf %arg1, %arg2 : bf16
      tt.reduce.return %1 : bf16
    }) : (tensor<512xbf16>) -> bf16
    tt.return %0 : bf16
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @rewrite_axis1
  tt.func @rewrite_axis1(%arg0: tensor<4x512xf32>) -> tensor<4xf32> {
    // 512 -> 256 -> 128 along axis 1, final reduce on 4x128.
    // CHECK: tensor.extract_slice %arg0[0, 0] [4, 256] [1, 1] : tensor<4x512xf32> to tensor<4x256xf32>
    // CHECK: tensor.extract_slice %arg0[0, 256] [4, 256] [1, 1] : tensor<4x512xf32> to tensor<4x256xf32>
    // CHECK: arith.addf {{.*}} : tensor<4x256xf32>
    // CHECK: tensor.extract_slice {{.*}}[0, 0] [4, 128] [1, 1] : tensor<4x256xf32> to tensor<4x128xf32>
    // CHECK: tensor.extract_slice {{.*}}[0, 128] [4, 128] [1, 1] : tensor<4x256xf32> to tensor<4x128xf32>
    // CHECK: arith.addf {{.*}} : tensor<4x128xf32>
    // CHECK: "tt.reduce"
    // CHECK: }) : (tensor<4x128xf32>) -> tensor<4xf32>
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<4x512xf32>) -> tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @rewrite_f32_1024
  tt.func @rewrite_f32_1024(%arg0: tensor<1024xf32>) -> f32 {
    // 1024 -> 512 -> 256 -> 128, final reduce on 128.
    // CHECK: tensor.extract_slice %arg0[0] [512] [1] : tensor<1024xf32> to tensor<512xf32>
    // CHECK: tensor.extract_slice %arg0[512] [512] [1] : tensor<1024xf32> to tensor<512xf32>
    // CHECK: arith.addf {{.*}} : tensor<512xf32>
    // CHECK: tensor.extract_slice {{.*}}[0] [256] [1] : tensor<512xf32> to tensor<256xf32>
    // CHECK: tensor.extract_slice {{.*}}[256] [256] [1] : tensor<512xf32> to tensor<256xf32>
    // CHECK: arith.addf {{.*}} : tensor<256xf32>
    // CHECK: tensor.extract_slice {{.*}}[0] [128] [1] : tensor<256xf32> to tensor<128xf32>
    // CHECK: tensor.extract_slice {{.*}}[128] [128] [1] : tensor<256xf32> to tensor<128xf32>
    // CHECK: arith.addf {{.*}} : tensor<128xf32>
    // CHECK: "tt.reduce"
    // CHECK: }) : (tensor<128xf32>) -> f32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<1024xf32>) -> f32
    tt.return %0 : f32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @skip_f32_8200
  tt.func @skip_f32_8200(%arg0: tensor<8200xf32>) -> f32 {
    // Even and inside the f32 range, but not a power of two: must not be
    // rewritten, otherwise the greedy splitting degenerates to a
    // non-power-of-two tail (8200 -> 4100 -> 2050 -> 1025).
    // CHECK-NOT: tensor.extract_slice
    // CHECK: "tt.reduce"(%arg0) <{axis = 0 : i32}>
    // CHECK: }) : (tensor<8200xf32>) -> f32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<8200xf32>) -> f32
    tt.return %0 : f32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @skip_i32_512
  tt.func @skip_i32_512(%arg0: tensor<512xi32>) -> i32 {
    // CHECK-NOT: tensor.extract_slice
    // CHECK: "tt.reduce"(%arg0) <{axis = 0 : i32}>
    // CHECK: }) : (tensor<512xi32>) -> i32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: i32, %arg2: i32):
      %1 = arith.addi %arg1, %arg2 : i32
      tt.reduce.return %1 : i32
    }) : (tensor<512xi32>) -> i32
    tt.return %0 : i32
  }
}

// -----

module {
  // CHECK-LABEL: tt.func @skip_non_add_reduce
  tt.func @skip_non_add_reduce(%arg0: tensor<512xf32>) -> f32 {
    // CHECK-NOT: tensor.extract_slice
    // CHECK: arith.maximumf
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maximumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<512xf32>) -> f32
    tt.return %0 : f32
  }
}
