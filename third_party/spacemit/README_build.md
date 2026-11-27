```bash
cd /.../FlagTree

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

## Cross-compile a riscv64 wheel (for real-board pip install)

The QEMU-RPC test flow (spacemit3.6-build-and-test.yml) keeps the host package
on x86 and only runs kernels on RISC-V. To install FlagTree natively on a
spacemit board you need a wheel whose `libtriton.so` / backend binaries are
riscv64 — built by cross compilation on an x86 host via
`third_party/spacemit/scripts/build_wheel_riscv64.sh`.

Assets (URLs + md5 in `third_party/spacemit/spacemit-ci.env`, sections
`riscv64 cross-compile wheel assets`; per-CPython-version assets are keyed
`MLIR_BINDINGS_{URL,MD5}_<312|313|314>` and `PYENV_{URL,MD5}_<312|313|314>` —
the wheel's Python version must match the board's Python):

- RISCV toolchain (`RISCV_TOOLCHAIN_URL`, provides `RISCV_ROOT_PATH`)
- zlib for riscv64 — built into the toolchain sysroot once per toolchain:
  `RISCV_ROOT_PATH=<toolchain> bash third_party/spacemit/scripts/build_rv_zlib.sh`
- spine-mlir riscv64 install (`SPINE_MLIR_URL` needs the riscv64 variant; the
  one in `spacemit-ci.env` is x86 for the host flow)
- spine-runtime install (`SPINE_RUNTIME_URL`)
- cross LLVM (`LLVM_CROSS_URL`, f6ded0be == LLVM22 cross)
- MLIR Python bindings, riscv64-tagged (per Python version)
- riscv64 CPython headers + libpython (`PYENV_*`, per Python version)

Build (single command, ~30-60 min):

```bash
RISCV_ROOT_PATH=/path/to/spacemit-toolchain \
MAX_JOBS=64 \
bash third_party/spacemit/scripts/build_wheel_riscv64.sh \
    /path/to/llvm-cross-install \
    /path/to/spine-mlir-riscv64-install \
    /path/to/spine-runtime-install \
    /path/to/mlir-bindings-dir \
    /path/to/py-riscv64-root
```

Output: `dist-riscv64/flagtree-...-linux_riscv64.whl` (the script retags the
`linux_x86_64` tag pip produced on the x86 host).

Board install notes:

- The wheel is large; a board with a nearly-full rootfs should
  `pip uninstall triton` first and use `TMPDIR=/tmp` for the install.
- The bundled spine-opt must be checked for dialect generation before use —
  run one of the repo's example kernels; if you see
  `Dialect 'xsmt_async' not found`, replace
  `triton/backends/spacemit/bin/spine-opt` with a known-good build.
- Verify every `.so` in the wheel is RISC-V (`unzip -l`, `file`) before
  shipping — an x86 binary mixed in fails on the board with a confusing
  dlopen error.
