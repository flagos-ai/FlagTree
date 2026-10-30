# XPU TLE device distribution

The `triton_xpu` dialect contains the XPU device path for explicit FlagCX
communicator handles. It currently lowers `get_device_id`, `get_num_pes`,
`distributed_barrier` in device space, and scalar `remote_pointer` in device
space. Node and cluster remote pointers and implicit communicator launch
parameters are outside this path.

The frontend accepts a runtime `i64` communicator handle for rank, size, and
barrier operations. A `DistributedRtContext` may be passed as a constexpr;
rank and barrier use its communicator pointer, while remote uses its registered
memory pointer. Remote offsets count elements; the lowering multiplies them by
the pointee size before calling `flagcxGetIntraPointerC`.

The XPU setup hook builds `libflagcx_device.bc` from the downloaded FlagCX sources
and installs it under `third_party/xpu/backend/lib`. `FLAGCX_BITCODE_PATH` can
override this path. The backend links it with LLVM 19 `llvm-link` after
downgrading kernel IR, then verifies that the called FlagCX symbols have
definitions. The staged LLVM 19 package does not contain `llvm-link`; set
`FLAGCX_XPU_LLVM_LINK` to a compatible LLVM 19 executable. A host
`libflagcx.so` cannot satisfy device calls.

For the host library, the XPU setup hook downloads FlagCX under
`third_party/xpu/tle/third_party/flagcx` and discovers the XPU SDK and XCCL
from `XPU_HOME`/`XPU_SDK_HOME` and `XCCL_HOME`/`BKCL_HOME`, or from
`/usr/local/xpu` and `/usr/local/xccl`. Set `FLAGCX_XPU_DEVICE_HOME` and
`FLAGCX_XPU_CCL_HOME` to override those locations. The SDK must contain
`include/xpu/runtime.h` and `so/libxpurt.so*`; XCCL must contain
`include/bkcl.h` and `so/libbkcl.so*`. The setup hook invokes FlagCX's
`bindings/ir/kunlunxin` recipe and expects it to write
`build/xpu/lib/libflagcx_device.bc`. Set `FLAGCX_XPU_CLANG` if the staged
LLVM 19 compiler is unavailable. The pinned FlagCX revision does not yet
contain this recipe. It already includes FlagCX PR #555's KLX XSHMEM
CommTraits, adaptor, and test kernels, but those sources do not produce
`libflagcx_device.bc`. FlagCX must add the device IR recipe before the
bitcode can build.

The compile tests in `python/test/unit/language/test_tle_{get_device_id,n_pes,distributed_barrier,remote_pointers}.py`
run when the bitcode is present. Multi-device tests additionally need the
FlagCX host wrapper and an initialized distributed process group. They have
not been validated on XPU hardware in this repository.
