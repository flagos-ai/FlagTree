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
