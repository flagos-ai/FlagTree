<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

#### Agenda:

##### Items:
1. H100 updates
2. Triton-Shared layer updates
3. Intel update
4. Open discussion

##### Minutes:
Recording link [here](https://youtu.be/KZAzpKx1ebI)

1. H100 updates
   - Enabled WGMMA by default, now any matmul can reuse it.
   - fp8 formats enabled – 1.3 Petaflops on dense matmul on H100 (gemm performance)
   - Enabled Flash Attention using wgmma, resulting in 450 teraflop on fwd pass and 250 on backward pass – still working on perf for flash attention
   - fp8 numbers with flash attention running in fp8 with matmul is tricky, because the fp8 layout is significantly different than what is returned by wgmma, still wip

2. Triton-Shared layer
   - Please refer to slides for more details
   - Created a repo where you can find the middle layer
   - Available as a plugin into triton

3. Intel Update
   - Please refer to slides for more details
