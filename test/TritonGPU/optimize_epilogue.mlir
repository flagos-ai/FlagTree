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

// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-epilogue | FileCheck --check-prefixes=GCN %s

#mfma = #ttg.amd_mfma<{warpsPerCTA=[1,1], instrShape=[32,32], isTranspose=false}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mfma_epilogue_simple
  // CHECK-LABEL: mfma_epilogue_simple
  tt.func public @mfma_epilogue_simple(%data: tensor<64x64xf16, #mfma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = ttg.convert_layout {{.*}} : tensor<{{.*}}, #blocked> -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    %converted_data = ttg.convert_layout %data : tensor<64x64xf16, #mfma> -> tensor<64x64xf16, #blocked>
    tt.store %ptr, %converted_data : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{warpsPerCTA=[1,1], instrShape=[32,32], isTranspose=false}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GCN-LABEL: mfma_epilogue_chained_elementwise
  // CHECK-LABEL: mfma_epilogue_chained_elementwise
  tt.func public @mfma_epilogue_chained_elementwise(%data: tensor<64x64xf32, #mfma>, %ptr: tensor<64x64x!tt.ptr<f16>, #blocked>) {
    // GCN: [[PTR:%[a-z0-9]+]] = ttg.convert_layout {{.*}} : tensor<{{.*}}, #blocked> -> tensor<{{.*}}, #mma>
    // GCN: tt.store [[PTR]], {{.*}} : tensor<{{.*}}, #mma>
    %converted_data = ttg.convert_layout %data : tensor<64x64xf32, #mfma> -> tensor<64x64xf32, #blocked>
    %trunked = arith.truncf %converted_data : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    tt.store %ptr, %trunked : tensor<64x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
