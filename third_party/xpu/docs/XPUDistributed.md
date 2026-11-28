# XPU TLE device distribution

The `triton_xpu` dialect contains the XPU device path for explicit FlagCX
communicator handles. It currently lowers `get_device_id`, `get_num_pes`,
`distributed_barrier` in device space, and scalar `remote_pointer` in device
space. Node and cluster remote pointers and implicit communicator launch
parameters are outside this path.

The frontend accepts a runtime `i64` communicator handle for rank, size, and
barrier operations. A `DistributedRtContext` carries the communicator pointer
for rank and barrier operations and the registered memory pointer for remote
stores. The lowering multiplies the element offset by the pointee size before
passing the byte offset to `flagcxGetIntraPointerWithAccessC`. This call requests
a write-only XSHMEM peer address.

The XPU compiler inserts one `device_init` at the kernel entry, before the
grid-stride loop, using the communicator carried by the distributed operations.
Kernels that pass only a raw memory handle can supply the communicator with
`tle.init(comm)`. Existing explicit initializers for the same communicator
are consolidated into the entry call.

## Prepare the SDKs

Use a KunlunXin environment with the XPU driver, XPU-compatible PyTorch, and
FlagTree's build dependencies. Provision compatible XRE and XCCL SDKs once.
The two-device tests were validated with XRE 5.37.1.0 and XCCL 4a9d27ec.

| Default SDK path | Required files |
| --- | --- |
| `~/.flagtree/xpu/sdk/xre` | `include/xpu/runtime.h`, `so/libxpurt.so`, and the XRE CUDA compatibility libraries |
| `~/.flagtree/xpu/sdk/xccl` | `include/bkcl.h`, `include/xshmem/`, `so/libbkcl.so`, and `lib/libxshmem_device.a` |

The SDK paths can be directories or symlinks to installed SDKs. When the
corresponding cache path is absent, setup can also use `/usr/local/xpu` or
`/usr/local/xccl`.
`FLAGTREE_CACHE_DIR` can replace `~/.flagtree`. For SDKs in other locations,
set `FLAGCX_XPU_DEVICE_HOME` or `FLAGCX_XPU_CCL_HOME`.

## Build and install

Select the backend using `FLAGTREE_BACKEND`, as in the Iluvatar build.
`USE_FLAGCX` controls FlagCX for XPU and defaults to ON.

```bash
FLAGTREE_BACKEND=xpu USE_FLAGCX=ON python -m pip install . --no-build-isolation
```

1. The XPU setup hook stages XTDK LLVM 19 for device compilation and linking.
2. Setup checks out the pinned public FlagCX revision under
   `third_party/xpu/tle/third_party/` and initializes FlagCX's JSON headers.
3. The KunlunXin host recipe builds `libflagcx.so` with XSHMEM support. The
   device IR recipe builds and checks the ABI of `libflagcx_device.bc`.
4. Setup caches `libflagcx.so`, `libflagcx_device.bc`, and `include/` directly
   under `~/.flagtree/xpu/lib/` and stages the files needed by the Python
   package. At JIT time LLVM 19 links the FlagCX bitcode into the kernel.
   The final XPU link uses the XSHMEM device archive from the SDK.

The runtime resolves each artifact independently. It checks its explicit path
override, `FLAGCX_MODULE_PATH`, the XPU package, and the flat XPU cache in that
order. It can also fall back to a standalone FlagCX wheel or the legacy cache.

For an XPU build without FlagCX, set `USE_FLAGCX=OFF`.

## Run on two local devices

```bash
bash third_party/xpu/test/run_tle_distributed.sh
```

Use the same Python environment as the install. The runner configures the SDK
library paths and launches the rank, barrier, remote store, signal, and
signal-wait tests. `tle.get_mem_pool()` initializes communication first. Local
rank 0 builds `flagcx_allocator_xpu_ccl.so` in the temporary directory on a
cache miss. After synchronization, each rank loads the library. Add `debug` to stream FlagCX
diagnostics. Each test also writes a log in the directory printed by the runner.
