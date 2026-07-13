<div align="right"><a href="./release_notes_v0.2.0_cn.md">中文版</a></div>

## FlagTree 0.2.0 Release

### Highlights

FlagTree inherits capabilities from the previous version, continuously integrates new backends, expands support for Triton versions, and provides hardware-aware optimization capabilities. The project is currently in its early stages, aiming to be compatible with existing adaptation solutions for various AI chip backends, unify the code repository, build a code co-construction platform, and quickly implement multi-backend support in a single repository.

### New features

* Added multi-backend Support

Currently supported backends include triton_shared cpu, iluvatar, xpu (klx), mthreads, __metax__, __aipu__(arm npu), __ascend__ npu & cpu, __tsingmicro__, cambricon, with __bold__ indicating newly added ones. <br>
Each new backend maintains the capabilities of the previous version: cross-platform compilation and rapid verification, plugin-based high-differentiation modules, CI/CD, and quality management capabilities. <br>
Jointly developing common extensions for the middleware layer with backend vendors, and open-sourcing standardized PyTorch backend extensions to support Triton / FlagTree practices. <br>

* Dual Compilation Path Support

Supports TritonGPU and Linalg compilation paths. Provides multiple integration paradigms for non-GPGPU backends, adds FLIR repository support for Linalg Dialect extensions and MLIR extensions for backend compilation.

* Added support for Triton versions

Currently supported Triton versions include 3.0.x, 3.1.x, __3.2.x__, __3.3.x__, with __bold__ indicating newly added ones.

* Hardware-aware optimization support

Supports providing guided programming interfaces for backend-common or specific hardware features. Through compatible extensions, adding guidance information at the frontend to provide flexibility in operator writing and performance tuning.

* Joint construction with FlagGems operator library

Collaborating with the [FlagGems](https://github.com/FlagOpen/FlagGems) operator library to support related features in version adaptation, backend interfaces, registration mechanisms, and test modifications.

### Looking ahead

GPGPU backend code will be integrated, decoupling backend differentiation changes from TritonGPU; non-GPGPU backends will be horizontally integrated on the FLIR foundation, with unified design for common passes. <br>
Providing Triton adaptation version upgrade guides for backend vendors: 3.0 -> 3.1 -> 3.2 -> 3.3. <br>
CI/CD will add FlagGems operator library functional testing. <br>
Integrating C++ Runtime functionality to reduce runtime overhead outside of kernels to be on par with CUDA. <br>
