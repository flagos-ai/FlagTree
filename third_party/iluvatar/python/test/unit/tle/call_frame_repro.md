# Scalar spill corruption across device calls

`test_call_frame.py` reproduces the Iluvatar private-frame corruption without
a model, weights, SME operations, a grid barrier or CUDA graph capture.
Its smallest launch is one CTA containing one 64-lane warp. Each thread
records 64 scalar timestamps before and after a device-function call.
The callee saves two VGPRs. The pressure variant clobbers all 256 VGPRs
before the call, forcing live scalar values into private memory. Exact
before/after comparison does not depend on the timestamp values themselves.

The test also covers a nested callee and the original 16-CTA x 1024-thread
launch. It deliberately fails on affected compilers; it is not marked xfail.

## Verified on main

On 2026-10-25, an independent checkout of `origin/main` at
`2f1229e51c03b345a1c72207854828eed7282de9` was built using main's pinned
Iluvatar LLVM22 v0.6.1 archive. No tracked main source was changed. The only
added test source matches this checkout's `test_call_frame.py` byte for byte.

The main Python module and native extension were both loaded from
`/root/repos/FlagTree-main-call-frame-20261025/python/triton`.
The native extension SHA-256 is
`7714d5e2cb26934e59d6f32874bdd8ac8b935f2489897988542434012f020766`.
The installed feature-branch compiler was retained unchanged.

| Launch | Values compared | Direct-call pressure mismatches | Nested-call pressure mismatches |
|:--|--:|--:|--:|
| 1 CTA x 64 threads | 4096 | 128 | 128 |
| 16 CTAs x 1024 threads | 1048576 | 32768 | 32768 |

Both controls have zero mismatches at both launch sizes. All prefix/suffix
guards remain intact. The full suite reports **4 failed, 4 passed**.
The separate minimal command reports **2 failed, 2 passed, 4 deselected**.
This establishes that the failure exists on main independently of FlagMega
and the current feature-branch compiler changes.

## Reproduce the single-warp case

With the independently built main checkout still available on this machine:

```bash
cd /root/repos/FlagTree-main-call-frame-20261025
env -u PYTHONPATH PYTHONPATH="$PWD/python" PYTHONNOUSERSITE=1 \
  FLAGTREE_BACKEND=iluvatar CUDA_VISIBLE_DEVICES=1 CUDA_MODULE_LOADING=0 OMP_NUM_THREADS=1 \
  /root/miniconda3/envs/flagtree/bin/python -m pytest \
  third_party/iluvatar/python/test/unit/tle/test_call_frame.py -q -k one-warp
```

Remove `-k one-warp` to include the original launch geometry. Choose a free
GPU when rerunning. Pytest temporary directories retain the generated LLVM
IR, the unmodified compiler-produced cubin and `result.json` for each case.

The independent build used `setup.py editable_wheel` without installing the
wheel, so the existing development environment still selects the feature
checkout normally. TLE and compiler optimization passes remained enabled;
the optional profiler/debugger and Proton were not built. Build tools
`clang`, `opt`, `llvm-dis` and `llvm-as` all came from the same v0.6.1 archive.
FlagCX's JSON include directory was set to the preloaded nlohmann headers.

The main revision, imported module/native paths, build configuration, LLVM
inputs, original cubins, raw results and both pytest runs are retained in
`python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150/figures/data/triton-tle-alignment/main-branch-repro-v108/`.
This is a failing reproducer and main-branch verification, not a compiler fix.
