// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// set_layout can lower to a marker on an existing producer without emitting
// a convert. That producer must be an anchor, and must not be rematerialized
// in its users' preferred layout with an obsolete explicit attribute.
// CHECK-DAG: #[[$POINTWISE_LAYOUT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: tt.func @explicit_pointwise
// CHECK: arith.addf {{.*}} {tle.explicit_encoding.0 = #[[$POINTWISE_LAYOUT]]} : tensor<256xf32, #[[$POINTWISE_LAYOUT]]>
// CHECK: tt.store
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_pointwise(%x: tensor<256xf32, #a>, %y: tensor<256xf32, #a>, %ptr: tensor<256x!tt.ptr<f32>, #a>) {
    %xb = ttg.convert_layout %x : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %yb = ttg.convert_layout %y : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %sum = arith.addf %xb, %yb {tle.explicit_encoding.0 = #b} : tensor<256xf32, #b>
    %out = ttg.convert_layout %sum : tensor<256xf32, #b> -> tensor<256xf32, #a>
    tt.store %ptr, %out : tensor<256x!tt.ptr<f32>, #a>
    tt.return
  }
}

// -----

// The same rule applies to slice encodings, including when the competing
// blocked layout has the same logical tensor shape.
// CHECK-DAG: #[[$SLICE_PARENT:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK-LABEL: tt.func @explicit_slice
// CHECK: arith.addf {{.*}} {tle.explicit_encoding.0 = #ttg.slice<{dim = 0, parent = #[[$SLICE_PARENT]]}>} : tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$SLICE_PARENT]]}>>
// CHECK: tt.store
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @explicit_slice(%x: tensor<256xf32, #a>, %y: tensor<256xf32, #a>, %ptr: tensor<256x!tt.ptr<f32>, #a>) {
    %xs = ttg.convert_layout %x : tensor<256xf32, #a> -> tensor<256xf32, #slice>
    %ys = ttg.convert_layout %y : tensor<256xf32, #a> -> tensor<256xf32, #slice>
    %sum = arith.addf %xs, %ys {tle.explicit_encoding.0 = #slice} : tensor<256xf32, #slice>
    %out = ttg.convert_layout %sum : tensor<256xf32, #slice> -> tensor<256xf32, #a>
    tt.store %ptr, %out : tensor<256x!tt.ptr<f32>, #a>
    tt.return
  }
}

// -----

// Ordinary pointwise code still has its redundant layout round trip removed.
// CHECK-LABEL: tt.func @ordinary_pointwise
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[SUM:.*]] = arith.addf %arg0, %arg1
// CHECK-NEXT: tt.store %arg2, %[[SUM]]
// CHECK-NEXT: tt.return
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ordinary_pointwise(%x: tensor<256xf32, #a>, %y: tensor<256xf32, #a>, %ptr: tensor<256x!tt.ptr<f32>, #a>) {
    %xb = ttg.convert_layout %x : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %yb = ttg.convert_layout %y : tensor<256xf32, #a> -> tensor<256xf32, #b>
    %sum = arith.addf %xb, %yb : tensor<256xf32, #b>
    %out = ttg.convert_layout %sum : tensor<256xf32, #b> -> tensor<256xf32, #a>
    tt.store %ptr, %out : tensor<256x!tt.ptr<f32>, #a>
    tt.return
  }
}

// -----

// Different explicit layouts on successive values are legal. A hard layout
// propagated from %sum must not override the explicit target of %out.
// CHECK-DAG: #[[$FIRST:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$SECOND:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: tt.func @successive_explicit_layouts
// CHECK: %[[FIRST_VALUE:.*]] = arith.addf {{.*}} {tle.explicit_encoding.0 = #[[$FIRST]]} : tensor<256xf32, #[[$FIRST]]>
// CHECK: %[[SECOND_VALUE:.*]] = ttg.convert_layout %[[FIRST_VALUE]] {tle.explicit_encoding.0 = #[[$SECOND]]} : tensor<256xf32, #[[$FIRST]]> -> tensor<256xf32, #[[$SECOND]]>
// CHECK: tt.store {{.*}}, %[[FIRST_VALUE]] : tensor<256x!tt.ptr<f32>, #[[$FIRST]]>
// CHECK: tt.store {{.*}}, %[[SECOND_VALUE]] : tensor<256x!tt.ptr<f32>, #[[$SECOND]]>
#a = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @successive_explicit_layouts(%x: tensor<256xf32, #a>, %y: tensor<256xf32, #a>, %pa: tensor<256x!tt.ptr<f32>, #a>, %pb: tensor<256x!tt.ptr<f32>, #b>) {
    %sum = arith.addf %x, %y {tle.explicit_encoding.0 = #a} : tensor<256xf32, #a>
    %out = ttg.convert_layout %sum {tle.explicit_encoding.0 = #b} : tensor<256xf32, #a> -> tensor<256xf32, #b>
    tt.store %pa, %sum : tensor<256x!tt.ptr<f32>, #a>
    tt.store %pb, %out : tensor<256x!tt.ptr<f32>, #b>
    tt.return
  }
}
