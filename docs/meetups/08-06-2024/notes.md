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
1. Triton-CPU Update
2. Intel GPU backend update

##### Items:
Meeting notes:
1. Triton-CPU Update: Intel and Meta jointly presented the work on Triton-CPU, highlighting good progress on coverage and performance improvements. They also covered some of the optimizations they leveraged to get performance comparable to torch-native and torch-inductor. More details are in their slides.
2. Intel GPU Backend: Intel GPU backend shows good performance close to expert-tuned kernels and the use of block pointers for performance gains. There were questions around the future of block pointers and their importance for performance gains. With block-pointer deprecation there is a need for a more generic interface to support various backends including Intel GPU.
3. The 2024 Triton conference is on September 17th 2024 in Fremont California! Please register [here](README.md).
##### Minutes:
Recording link [here](https://youtu.be/dfL3L4_3ujg)

Presentations repo [here](https://drive.google.com/drive/folders/1fQ3zVrM7DT8W8FGJWKx1wNr2X53tYbeT?usp=sharing)
