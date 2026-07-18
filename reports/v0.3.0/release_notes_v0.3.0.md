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

<div align="right"><a href="./release_notes_v0.3.0_cn.md">中文版</a></div>

## FlagTree 0.3.0 Release

### Highlights

FlagTree inherits capabilities from the previous version, continuously integrates new backends, and strengthens the ecosystem matrix. The project is currently in its early stage, aiming to be compatible with existing adaptation solutions for various chip backends, unify code repositories, create a collaborative code-building platform, and quickly achieve single-repository multi-backend support. Meanwhile, it continues to develop unified programming interface extensions, build intermediate layer representation and conversion extensions (FLIR), and enhance hardware awareness and compilation guidance support capabilities and scope (flagtree_hints).

### New features

* Added multi-backend Support

Currently supported backends include triton_shared cpu, iluvatar, xpu (klx), mthreads, metax, aipu(arm npu), ascend npu & cpu, tsingmicro, cambricon, __hcu__, with __bold__ indicating newly added ones. <br>
Each new backend maintains the capabilities of the previous version: cross-platform compilation and rapid verification, plugin-based high-differentiation modules, CI/CD, and quality management capabilities. <br>

* Continuous integration with upstream ecosystems

Thanks to the technical support from our partners, FlagTree has added compatibility with Paddle framework, OpenAnolis operating system, and Beijing Super Cloud Computing Center.

* Continuous development of FLIR

Ongoing expansion of DSL, TTIR extensions, Linalg intermediate representation and transformation extensions, and MLIR extensions to provide programming flexibility, enrich expression capabilities, and improve transformation capabilities.

* Established compilation guidance specifications, added unified management module for multi-backend compilation

flagtree_hints provides guidance for hardware unit mapping and compilation transformation optimization choices, and manages backend guidance differences through a unified module.

* Joint construction with FlagGems operator library

Collaborating with [FlagGems](https://github.com/FlagOpen/FlagGems) operator library on version compatibility, backend interfaces, registration mechanisms, and test modifications to support related features.

### Looking ahead

Improving GPGPU backend integration, decoupling backend specialization from main code implementation to establish an engineering foundation for FlagTree's general extensions and optimizations. <br>
Aiming to comprehensively cover various implementation styles in the operator library, enhancing FLIR compilation completeness to match multiple backend requirements and enable compilation for more backends. <br>
flagtree_hints will continue to explore operator performance optimization potential on different backends along both TritonGPU and Linalg compile-paths. <br>
