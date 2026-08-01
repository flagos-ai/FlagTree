```bash
cd /.../FlagTree
git apply third_party/spacemit/patch/flagtree.patch

# Build and install with spine-mlir and spine-runtime dependencies
FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH=/home/share/nfs_share/llvm-pre-build/llvm-f6ded0be897e2878612dd903f7e8bb85448269e5-build-x86-release/ \
SPINE_MLIR_INSTALL_DIR=/path/to/spine-mlir-install \
SPINE_RUNTIME_INSTALL_DIR=/path/to/spine-runtime-install \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS=2 \
pip install . --no-build-isolation -v
```

## Environment Variables

- `LLVM_SYSPATH`: Path to LLVM installation (provides cmake config and libraries)
- `SPINE_MLIR_INSTALL_DIR`: Path to spine-mlir installation
  - Copies `bin/` tools (spine-opt, llc, mlir-translate, opt) to `backend/bin/`
  - Copies `lib/` shared libraries (libspert.so*, libSpeIR*.so*) to `backend/lib/`
- `SPINE_RUNTIME_INSTALL_DIR`: Path to spine-runtime installation
  - Vendors `include/` headers (spert.hpp, spert_engine.hpp, spert_abi.h) to `backend/include/SpineRuntime/`
- `TRITON_BUILD_PROTON=OFF`: Disable Proton profiler (spacemit backend doesn't support it)
- `MAX_JOBS`: Limit parallel build jobs

## Notes

- If `SPINE_MLIR_INSTALL_DIR` and `SPINE_RUNTIME_INSTALL_DIR` are not set, the build will use prebuilt binaries/libraries/headers from `backend/{bin,lib,include}/` (if present)
- The spert ABI version in `driver.py` must match the libspert.so version copied from `SPINE_MLIR_INSTALL_DIR`
