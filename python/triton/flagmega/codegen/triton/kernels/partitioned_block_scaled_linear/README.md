# Rounded block-FP8 K partitions

`nn.partitioned_block_scaled_linear` reads BF16 `[T,K]`, scalar E4M3FN
`[N,K]` weights and FP32 global block scales, producing BF16 `[P,T,N]`.
Each equal contiguous K part ends on a quantization block boundary and rounds
independently. P remains observable: consumers can perform a local BF16 add
before widening and reducing partitions. There is no implicit Partial sum.

The SM90 implementation assigns complete `(partition, token, N tile)` tasks
to CTAs. It reads the original activation/weight through the physical ABI,
quantizes K128 blocks in ascending order, and writes the final chip-shared
partition output. It needs no cross-CTA scratch or internal grid barrier.
`dynamic_fp8_sm90` uses the existing reciprocal quantizer and a rounded combined
scale followed by FP32 FMA. Ordinary `dynamic_fp8` keeps separate scale
multiplications and compensated addition, as the existing block-FP8 kernel.
The N16/N32/N64 task size controls ownership and CTA parallelism. Matrix
operations keep at least 64 rows, masking padding before loads and stores;
otherwise smaller tasks select FP16 MMA after conversion and change the FP8
WGMMA dot's arithmetic. This retains the original numerical method while
allowing small tasks to fill more CTAs. CPU FP32 dot results need not be
bitwise identical to this GPU method for arbitrary input codes.

Input lanes address K, output lanes address N. The renderer supports canonical
and compact chip storage through the normal coordinate helpers. N chip splits
must retain complete output lane groups. P chip splits require exactly one
whole semantic K partition per chip owner. Broadcast scales use global P and
N indices, including when weights occupy a compressed chip allocation.
Semantic block splits, K-part fragments, other targets and non-K128 device
quantization blocks are rejected explicitly.

Core coverage includes op/evaluator/serialization, forward and inverse
distribution, shared-user packing on ordinary and large graph paths, actual
device tails/lanes/multiple tokens, and four-card P/N ownership with local BF16
add followed by the FP32 collective. See the tests named
`test_partitioned_block_scaled_linear`, `test_partitioned_fp8_linear_packing`
and `test_partitioned_fp8_linear_tp4`. Run the last with `torchrun` on four idle
SM90 devices. Op-level success does not establish full-model performance or
the complete compute-body five-operation acceptance gate.
