<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
</div>
</div>

1. <a href="#orgc6a2b10">Introduction</a>
2. <a href="#org9a459f1">How it works</a>
3. <a href="#orgb820ad0">Usage</a>

<a id="orgc6a2b10"></a>

# LLVM Sanitizer Feature Introduction

The scripts in this folder contain build, setup, and run instructions for a custom installation of triton-shared that enables the option of using LLVM sanitizers for detecting programmer bugs in Triton programs, within the triton-shared backend.

The default triton-shared build (following instructions from README.md under /triton-shared) will lack the libraries required to use this bug detection feature. See usage on how to setup triton-shared for this feature.

<a id="org9a459f1"></a>

# Scripts Overview

To enable this feature of using LLVM sanitizers for detecting programmer bugs in Triton programs, we need a custom LLVM installation containing projects like openmp, clang, and compiler-rt to have access to LLVM sanitizer-specific libraries during compiling and linking stages of the triton-shared lowering process. Then since this custom LLVM installation will be used during runtime (triton-shared's JIT compilation), we also need to build triton-shared with this specific LLVM installation, or otherwise triton-shared will be built and ran with different LLVMs, causing conflicts. The scripts `build_llvm_for_sanitizers.sh` and `build_triton_shared_for_sanitizers.sh` contain instructions for building and installing the custom LLVM and building triton-shared with this custom LLVM.

Following this build phase, runtime setup consists of setting environment variables pointing to the newly installed LLVM and triton-shared binaries. `setup_runtime_for_sanitizers.sh` is a one-time per shell script that does this and should be invoked with `source`.

Following setup, LLVM sanitizers are now capable of being enabled during runtime using the environment variable `TRITON_SHARED_SANITIZER_TYPE`, along with other sanitizer-specific settings to suppress false positives. `run_triton_with_sanitizers.sh` is a script that applies these settings automatically depending on the user's specified sanitizer type.

<a id="orgb820ad0"></a>

# Usage

For building LLVM and triton_shared for usage with sanitizers, run the scripts in the following order:
1. `setup_triton_shared_with_venv.sh <desired path to venv> <desired path to triton shared>`
2. Activate the virtual environment created by `setup_triton_shared_with_venv.sh` in the current shell: `source ./path/to/venv/bin/activate`
3. `build_llvm_for_sanitizers.sh <desired path to LLVM installation directory> <existing path to triton shared>`
4. `build_triton_shared_for_sanitizers.sh <existing path to llvm build dir> <existing path to llvm install dir> <existing path to triton shared>`

For runtime setup (one-time per shell): `source setup_runtime_for_sanitizers.sh <existing path to venv> <existing path to llvm install dir> <existing path to triton shared>`

For running a python program with sanitizers enabled: `run_triton_with_sanitizers.sh <sanitizer type> python program.py`. Currently, the supported `<sanitizer type>`s are `asan` and `tsan`.
