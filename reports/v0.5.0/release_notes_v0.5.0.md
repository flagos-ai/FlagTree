[[中文版](./release_notes_v0.5.0_cn.md)|English]

## FlagTree 0.5.0 Release

### Highlights

FlagTree inherits capabilities from the previous version, continuously integrates new backends, and strengthens the ecosystem. Building upon the foundation of creating a collaborative code-building platform and achieving single-repository multi-backend support, the project continues to develop unified backend specialization design, continues to build intermediate layer representation and transformation extensions (FLIR), enhances hardware-aware and compilation guidance support capabilities and scope (HINTS), and Extend the Triton language (TLE).

### New features

* Added multi-backend Support

Currently supported backends include nvidia, amd, triton_shared cpu, iluvatar, xpu, mthreads, metax, aipu, ascend, tsingmicro, cambricon, hcu, enflame-gcu300, __enflame-gcu400__, __sunrise__, with __bold__ indicating newly added ones. <br>
Each new backend maintains the capabilities of the previous version: cross-platform compilation and rapid verification, plugin-based high-differentiation modules, CI/CD, and quality management capabilities. <br>

* Added Triton version support

Added support for Triton 3.6, and established corresponding protected branches.

* Continuous development of FLIR

Ongoing expansion of Linalg intermediate layer representation and transformation extensions, and MLIR extensions to provide programming flexibility, enrich expression capabilities, and improve transformation capabilities. Established paradigm for multi-backend integration with FLIR, added Tsingmicro backend integration and updated the Ascend backend, with support for TLE lowering on DSA.

* Triton Language Extensions: TLE

In response to the challenges facing Triton development, we propose TLE (Triton Language Extensions), which extends Triton at three levels to meet the urgent needs of users at different levels for operator programming languages. For details, see [wiki](https://github.com/flagos-ai/FlagTree/wiki/TLE).

**TLE-Lite** is a lightweight extension to Triton. All features are compatible with various hardware backends, requiring only minor modifications to existing Triton kernels to achieve significant performance improvements. Primarily targeted at algorithm engineers and rapid performance optimization scenarios.

**TLE-Struct** provides classified extensions (such as GPGPU, DSA) through hardware architecture clustering abstraction to meet further performance optimization needs. Requires developers to have some understanding of target hardware characteristics and optimization techniques.

**TLE-Raw** provides the most direct control over hardware, allowing the use of hardware vendors' native programming languages to achieve ultimate performance. Requires developers to have in-depth knowledge of target hardware, primarily targeted at performance optimization experts. For more information, refer to [wiki](https://github.com/flagos-ai/FlagTree/wiki/EDSL).

* Joint construction with FlagGems operator library

Collaborating with [FlagGems](https://github.com/FlagOpen/FlagGems) operator library on version compatibility, backend interfaces, registration mechanisms, and test modifications to support related features. FlagGems adds operators implemented with TLE.

### Looking ahead

Planned to transition triton_v3.6.x to main branch.<br>
TLE-Lite plans to extend Tensor slicing and distributed primitives.<br>
TLE-Struct plans to expose more hardware-related primitives and improve performance to approach hardware native languages.<br>
TLE-Raw plans to verify performance improvement opportunities in operators, optimize integration with Triton, while exploring other viable integration languages.<br>
