# Ordered block-FP8 cooperation

This optional SM90 implementation parallelizes the independent raw dot products
inside a dynamically quantized FP8 projection. It retains the original ordered
FP32 recurrence, including the rounded product of activation and weight scales.
It does not implement a tree reduction over K.

The typed geometry requires BF16 scalar activations, FP8 N-major weights
(including packed K lanes), FP32 scales, BF16 scalar or packed-N results,
one token and complete K128 blocks. Activations and scales are broadcast;
weights and results share a single contiguous N owner stage. Partial,
exclusive, cyclic, multistage, dynamic and split-K domains are excluded.
Chip coordinates stay fixed even when C occurs between block axes. Only the
N-owning block coordinates change during peer weight access.

Raw tasks cover each chip-local N64 tile and K128 block exactly once. They use
the original reciprocal-based SM90 activation quantization and raw WGMMA.
Explicit invocation-lifetime FP32 scratch holds `[K_blocks, chip_rows]` raw
results and `[K_blocks]` activation scales. The tile-zero writer publishes each
activation scale once. One grid barrier separates two noinline compute phases;
their wrapper remains inline so the caller owns rendezvous scratch. No private
register values cross the boundary.

The merge reconstructs the original rounded `A_scale * B_scale` and executes
one FP32 FMA per real K block, in the original order, then stores BF16. A
power-of-two register extent does not add padded FMAs. Stores keep the original
output owner and scalar lane mapping. Packed weights use actual component
stride, K-group coordinate and lane offset; global scale rows use the semantic
N coordinate, including interleaved C axes.

Candidate formation proves the type/distribution domain. Workspace requirements
exist before selection. The renderer independently verifies chip sharing,
physical component maps, contiguous FP32 scratch and complete participation;
it rejects private storage, unsupported backing and dynamic views. Ordinary
memory effects, buffer allocation and synchronization govern surrounding calls.
Changing this scratch contract requires recompilation before candidate proposal,
not a late source refresh of an old artifact.

The factory is optional; registering it does not change the generic target.
Regression coverage includes ordinary compilation/publication/runtime, readonly
compact weights, packed-N tails, non-power-of-two K-block counts, C0/C3 and
interleaved-C address oracles, rejected ABIs and an independent FP64 witness for
combined-scale FMA cancellation. Local timing does not establish full-model
correctness, zero spill or end-to-end performance.
