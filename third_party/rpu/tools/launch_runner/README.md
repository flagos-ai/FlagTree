# launch_kernel_runner

A small CLI front-end over the **rhino-launch-kernel** runtime library
(`librhino_launch.so`). It loads a flagtree-emitted `.ref` ELF, stages one
SPM/DDR buffer per kernel argument, dispatches the kernel on `/dev/rpu`, and
writes the outputs back to file.

It is what the on-board test
[`../../python/test/board/lk_board_smoke.py`](../../python/test/board/lk_board_smoke.py)
runs via `RPU_LK_RUNNER`.

## Build

Requires the vendor `rhino-launch-kernel` runtime installed (provides
`librhino_launch.so` + headers and the system `libHxil.so` / `librpu_api.so`).

```bash
cd third_party/rpu/tools/launch_runner
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install
cmake --build build
```

This produces `build/launch_kernel_runner`. Point the board test at it:

```bash
export RPU_LK_RUNNER=$PWD/build/launch_kernel_runner
```

### As part of the RPU backend build (optional)

The runner can instead be built together with the RPU backend by enabling the
`RPU_BUILD_LAUNCH_RUNNER` switch (OFF by default). It is declared only inside the
RPU backend CMake, so it has no effect on the main build or other backends. Pass
it through the package build:

```bash
# from the repo root, with FLAGTREE_BACKEND=rpu
export TRITON_APPEND_CMAKE_ARGS="-DRPU_BUILD_LAUNCH_RUNNER=ON -DCMAKE_PREFIX_PATH=/path/to/rhino-launch-kernel/install"
pip3 install . --no-build-isolation -v
```

The runner is emitted at the stable in-tree path
`third_party/rpu/tools/launch_runner/launch_kernel_runner`, so the board test
can point at it directly:

```bash
export RPU_LK_RUNNER=$PWD/third_party/rpu/tools/launch_runner/launch_kernel_runner
```

## CLI

```
launch_kernel_runner <elf> <kernel_name> <arg_spec> [<arg_spec> ...]

arg_spec:
  in:<size_bytes>:<input.bin>     stage input into an SPM/DDR buffer
  out:<size_bytes>:<output.bin>   read the output buffer back to file
```

Each argument gets an independent `LocalSPM_t` + `HostDDR_t` pair and is bound
to the kernel's parameter registers (reg slots `2*i` / `2*i+1` carry the
lo16/hi16 of the SPM address). The kernel is dispatched single-warp on core 0.
`/dev/rpu` access typically needs `sudo`.
