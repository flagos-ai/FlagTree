# WGMMA accumulators across loop boundaries

An asynchronous WGMMA result may be forwarded through `scf.for` initial
operands, iteration arguments, yields, and results. It must reach
`tle.wgmma_wait` before an ordinary tensor consumer reads it. Carrying an
accumulator does not itself wait for the hardware operation.

`TleLowerWGMMA.cpp::verifyWGMMAUses` follows both sides of each loop channel so
an unwaited store after the loop remains an error. `convertLoopCarriedAccumulator`
propagates the native Hopper accumulator type through nested forwarding loops;
ordinary initial values are converted before loop entry. Older MMA encodings are
ready inputs and require a Hopper conversion rather than being treated as an
already encoded async accumulator.

An independent product yielded into an already encoded loop channel must use
that channel's register layout. Previously, `getMMAType` selected a fresh layout
from the product's uses: with eight warps, the next independent product selected
`[4, 2]` while the initial product used `[8, 1]`, causing a mismatched `scf.yield`.
The layout is now selected from the destination iteration argument before the
async dot is emitted. This uses the channel as a layout constraint, not as the
new product's numerical accumulator, and avoids reading pending registers for a
backedge layout conversion.

## Regression checks

Run the TLE lit suite with a built `triton-opt` and run:

```sh
pytest python/test/tle/unit/test_tle_mma_encoding.py
pytest python/test/tle/unit/test_tle_async_accumulator_carry.py
pytest python/test/tle/unit/test_tle_async_manual_ring_wgmma.py
pytest python/test/tle/unit/test_tle_tma_ring_wgmma.py
```

`wgmma-async-loop-carry.mlir` and `wgmma-async-forwarding-loop.mlir` fail on the
base lowering because pending yields are rejected. The legacy accumulator test
fails because MMA v2 inputs bypass the required conversion. The eight-warp
independent-product regression and its GPU counterpart fail before the
loop-channel layout constraint is applied. Error tests also cover an ordinary
consumer without a wait after a forwarding loop.

On Hopper, the GPU tests cover four/eight warps, a partial wait with one group
still pending, caller/producer shared-memory ownership, skewed producers, and
32 repeated launches. The independent-product check compares alternating source
matrices exactly and asserts that the generated PTX retains the partial wait.
These checks establish correctness; they do not measure a performance speedup.

## Runtime descriptor dominance

`third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp::WGMMAOpPattern`
previously constructed shared descriptors in block-local PTX registers inside
the consuming MMA asm. That kept construction near the consumer, but ptxas
loop peeling and unrolling could sink invariant descriptor initializers into
the peeled iteration and reuse uninitialized descriptors on other paths. Even
one runtime iteration of the accumulator kernel could then report an
out-of-range shared address. Straight-line execution passed, and the failing
runtime input passed with complete descriptor operands defined before the loop.

Construction remains late in NVGPU-to-LLVM lowering, immediately before the
consumer's fused register fence, but now produces complete LLVM SSA operands.
The ordinary dataflow exposes initializer dominance across the loop's paths;
variant descriptors remain near their MMA use and invariant descriptors can be
hoisted through normal SSA optimization. Register-A operands and the native
non-TLE path retain their existing behavior. This redesign keeps default LLVM
and ptxas optimization enabled.

The minimal GPU reproducer is:

```sh
pytest 'python/test/tle/integration/test_tle_wgmma_accumulator.py::test_accumulator_numerics[1-1-3-64-64-4-fp8e4nv-fp16]'
```

`wgmma-runtime-loop-descriptors.mlir` checks complete SSA operands, and the
existing descriptor/fence lowering tests retain their instruction shape and
stride checks. The integration matrix runs FP8/FP16/BF16 pairs through single
and nested runtime loops, including zero-trip loops, on four/eight warps.
Results are compared bitwise with a fully unrolled sequence of the same WGMMA
instructions. The original Gaussian inputs are retained: the default FP8
policy permits imprecise accumulation, so a float64 product multiplied by the
trip count is not an oracle for repeated hardware rounding. Zero/one-pair cases
also retain the independent float64 reference check.

MUSA retains its existing independent float64 numerical reference.
