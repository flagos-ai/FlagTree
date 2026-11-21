# P800 SDNN rank-1 modulo loads

The prebuilt `libTritonSharedForXPU.a` in `xpu-sdnn-objects v0.3.6.9.0`
can lower a rank-1 load of `src[x % PERIOD]` to `src[x]`. This loses wrap
boundaries even though the load mask is retained. For example, with
`BLOCK=16`, `LENGTH=19`, and `PERIOD=17`, the second tile incorrectly reads
indices `16, 17, 18` instead of `16, 0, 1`.

The XPU compiler now captures the wrapped-load contract before
`triton-to-linalg-experimental` and repairs its DMA after SDNN fusion, before
LoopGrid. The repair emits a loop of contiguous DMA segments inside the same
kernel. Each segment stops at the next modulo boundary; its destination is the
next valid portion of the original local buffer. Fully masked tiles execute no
DMA. The vendor objects are not modified.

This repair is enabled for the P800 (`arch == 3`) SDNN path. It supports direct
source arguments, rank-1 unit-stride `lane` or `program_id * BLOCK + lane`
indices in i32, a positive constant period, and the prefix mask `x < LENGTH`
with a constant nonnegative length. The pointer must have a single load user,
and that load must be in the function's entry block. A captured load must remain
uniquely associated with one DMA through fusion. Unsupported tensor modulo
loads or ambiguous DMA provenance produce a compilation error instead of an
incorrect result. Scalar modulo offsets and loads without tensor modulo keep
their existing behavior. Other architectures and the Cluster path are unchanged.

Rebuild the native XPU extension together with the Python backend: the two new
pass bindings are required, not optional fallbacks for an older extension.

## Standalone reproducer

The minimal reproducer is included in
`third_party/xpu/python/examples/repro_rank1_wrap.py`. Select one healthy, idle P800
with both visibility variables set to the same physical device, then run:

```bash
export CUDA_VISIBLE_DEVICES=5 XPU_VISIBLE_DEVICES=5
export TRITON_CACHE_DIR="$(mktemp -d)"
python third_party/xpu/python/examples/repro_rank1_wrap.py --period 32 --output result32.json
python third_party/xpu/python/examples/repro_rank1_wrap.py --period 17 --output result17.json
python third_party/xpu/python/examples/repro_rank1_wrap.py --period 7 --output result7.json
```

Replace `5` with the available physical device. Output paths must not exist.
With the affected unpatched lowering, the ordinary inputs have 0, 2, and 12
mismatches for periods 32, 17, and 7 respectively. After repair, every result
must have `gate_passed: true`, with the source and destination guards intact.
The reproducer records correctness in JSON and exits with status 1 on failure.

## Regression tests

The compiler tests require an XPU build but do not initialize a device:

```bash
CUDA_VISIBLE_DEVICES= XPU_VISIBLE_DEVICES= python -m pytest -q \
  third_party/xpu/python/test/unit/backends/test_rank1_wrap_lowering.py
```

These evaluate the addresses in the actual generated SSA against the original
per-lane modulo expression, including an extra fully masked tile. They also
check unsupported patterns, pass ordering, and unchanged contiguous/scalar
modulo loads.

Run the device tests on an idle, healthy P800 with both visibility variables
selecting the same physical device:

```bash
python -m pytest -q -x third_party/xpu/python/test/unit/language/test_rank1_wrap.py
```

Device coverage includes the original periods 7/17/32, multiple wraps in one
tile, short and exact tails, and period 4099 with blocks 512/1024. FP32, FP16,
and BF16 outputs are checked bitwise for ordinary values, signed zeros,
infinities, NaNs, subnormals, and one-hot data. The source and destination guard
regions must remain unchanged. Physical source allocations are padded so that
an old unwrapped read is detected numerically without a device allocation
overrun. This padding is a test diagnostic, not the fix.

## Vendor updates

The repair assumes the prebuilt lowering's validated DMA structure. The
architecture condition `arch == 3` does not detect the vendor package version.
Revalidate changes to the vendor package, including its content digest: a new
version may preserve modulo itself or emit a different DMA structure. Remove or
disable this compensation when the original lowering is fixed. The version
policy must be agreed before promoting this draft to a permanent backend fix.
