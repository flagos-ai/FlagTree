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

[中文版](./release_notes_v0.1.0_cn.md)

## FlagTree 0.1.0 Release

### Highlights

FlagTree's initial release is built on Triton 3.1, introducing support for diverse AI chip backends. In its early stage, the project aims to maintain compatibility with existing backend adaptation solutions while unifying the codebase to enable rapid single-version multi-backend support.

### New features

* Multi-Backend Support

Currently supported backends include iluvatar, xpu (klx), mthreads, and cambricon.

* Dual Compilation Path Support

In this initial phase, the project provides basic compatibility for both TritonGPU dialect and Linalg dialect compilation paths.

* Pluggable High-Variance Module Architecture

Enables chip-specific backend customization through a plugin architecture. These non-generic modules are maintained by respective chip vendors and maintain structural consistency with the FlagTree main repository through engineering practices.

* Cross-Compilation and Rapid Validation Capabilities

For developer convenience, FlagTree supports compilation on any hardware platform and Python3 import functionality. Cross-compilation is possible when build and runtime environments are compatible (specifically matching or compatible versions of cpython, glibc, glibcxx, and cxxabi), allowing compiled artifacts to run across platforms with corresponding chip deployments.

* CI/CD Integration

The project implements comprehensive CI/CD pipelines for iluvatar, xpu, mthreads, nvidia, and other backends, enabling end-to-end validation from compilation to testing correctness.

* Quality Management Framework

Beyond CI/CD coverage for multiple backend chips, FlagTree implements quality and compliance assurance mechanisms including Contributor License Agreement (CLA) signing and security compliance scanning.

### Known issues

* Current lack of support for triton-opt, proton, and related tools.

### Looking ahead

FlagTree will continue investing in the Triton ecosystem, focusing on tracking Triton version updates, integrating AI chip backends, improving compilation efficiency, and enhancing cross-platform compatibility. Additionally, FlagTree will explore balancing general usability with chip-specific optimization requirements, providing compatible language-level unified abstractions and explicit specifications for hardware storage hierarchies, parallelism levels, and acceleration units.
