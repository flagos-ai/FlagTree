# BF16 partitioned row projection

`nn.partitioned_linear(x, w, partitions=P)` takes BF16 `[1,K]` and raw
row-major BF16 `[N,K]` weights. It computes each contiguous K partition with an
FP32 accumulator, rounds each partial independently to BF16, sums the rounded
partials in FP32, then rounds the final result to BF16. A full MatMul with just
one final rounding is not equivalent. FP32 sum/dot trees have the same
implementation freedom as the existing reduction/matmul contracts; the two
BF16 boundaries are mandatory. FP8 weights are not supported by this operation.

Input lanes pack only K and output lanes pack only N. N may be chip-sharded.
K may be chip-sharded only when each K owner holds exactly one complete semantic
partition, including complete input lane groups. Its BF16 Sum-partial result
must be materialized through Boxing before exporting a plain tensor. Exporting
the cooperative partial directly through an ABI with private CTA replicas is
rejected at rendering.

The SM90 `mma` implementation masks incomplete K tiles; `wgmma_local` requires
complete K512 tiles within each semantic partition. Both implementations keep
each output tile in one CTA and need no global scratch or internal grid barrier.
The TMA implementation uses two explicitly typed shared buffers and device
tensor-map handles, preserves their full swizzle alignment, and keeps the
barrier phase sequence across partitions and repeated output tiles. The final
transfer must finish before its slot is reused. Canonical external buffers and
compact chip weight images use their actual row strides and owner origins.

A chip partial assembled by cooperating CTAs must reside in one complete chip
component. The buffer planner uses `COMPACT_PER_CHIP` for chip-only partials in
a chip-shared function pool backed by separate device images. Block partials,
private pools and external/formal partial ABIs retain their existing component
representations. Remote Boxing indexes only the components physically present
in each peer image; chip rank selects the peer window, not an extra in-window
component.

Validation includes evaluator and Python resume tests, ordinary compilation,
both packing propagation paths, guarded real SM90 execution, and a four-rank
K-sharded projection followed by the BF16-partial/FP32-sum collective. Run the
four-rank regression on idle devices, for example:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,4 GLOO_SOCKET_IFNAME=lo \
  conda run -n flagtree python -m torch.distributed.run \
  --master-addr=127.0.0.1 --master-port=29641 --nproc-per-node=4 -m pytest \
  python/test/flagmega/runtime/test_partitioned_linear_tp4.py -q
```
