// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt --split-input-file %s --verify-diagnostics

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[16, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{threads per warp}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 2],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{warps per CTA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // expected-error @+1 {{CTAs per CGA}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        // Note it's a 3d tensor here, but #blocked is 2D.
        // expected-error @+1 {{rank}}
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<8x1x1x!tt.ptr<i32,1>, #blocked>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{
    sizePerThread=[1, 1],
    threadsPerWarp=[32, 1],
    warpsPerCTA=[4, 1],
    order=[0, 1], CGALayout = [[0, 0]]
}>
module attributes {
    "ttg.num-warps" = 4 : i32,
    "ttg.num-ctas" = 2 : i32,
    "ttg.threads-per-warp" = 32 : i32
} {
    tt.func public @fn(%arg0: tensor<8xf32, #blocked>) {
        // expected-error @+1 {{rank}}
        %t = tt.expand_dims %arg0 {axis = 0 : i32} : tensor<8xf32, #blocked> -> tensor<8x1xf32, #blocked>
        tt.return
    }
}
