from itertools import product

import torch


# Ref: https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/utils/triton_op.py#L162
def llama_shapes():
    # batch sizes * seq lengths
    BS = [2**i for i in range(0, 17)]
    # attn: wqkv, wo; ffn: w13, w2
    KN = [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
        (8192, 1280),
        (1024, 8192),
        (8192, 7168),
        (3584, 8192),
        (16384, 2304),
        (2048, 16384),
        (16384, 13312),
        (6656, 16384),
    ]
    return [(bs, n, k) for bs, (k, n) in product(BS, KN)]


# FMT: (M, N, K)
LLAMA_SHAPES = llama_shapes()


# Ref: https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/microbenchmarks/bench_mm_fusion.py#L96
def dynamo_shapes_mm():
    shapes = []
    # alexnet
    shapes.append((128, 4096, 9216))
    shapes.append((128, 4096, 4096))
    shapes.append((128, 1000, 4096))
    # BERT
    shapes.append((2048, 768, 768))
    shapes.append((2048, 3072, 768))
    shapes.append((2048, 768, 3072))
    # hf_GPT2
    shapes.append((1024, 768, 768))
    shapes.append((1024, 3072, 768))
    shapes.append((1024, 768, 3072))
    shapes.append((1024, 2304, 768))
    return shapes


# FMT: (M, N, K)
DYNAMO_SHAPES_MM = dynamo_shapes_mm()


# Ref: https://github.com/pytorch/pytorch/blob/main/benchmarks/operator_benchmark/pt/matrix_mult_test.py#L10,L23
#      https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/microbenchmarks/inductor_bmm.py#L49
def dynamo_shapes_bmm():
    shapes = []
    shapes.append((4, 5, 3, 2))
    shapes.append((32, 25, 20, 30))
    shapes.append((128, 100, 120, 110))
    shapes.append((128, 256, 128, 256))
    shapes.append((512, 1024, 1024, 512))

    # BERT (all)
    shapes.append((192, 128, 128, 64))
    shapes.append((192, 128, 128, 64))
    shapes.append((192, 128, 64, 128))
    # hf_GPT2 (all)
    shapes.append((12, 1024, 64, 1024))
    shapes.append((12, 1024, 1024, 64))
    # hf_Albert (all)
    shapes.append((12, 512, 512, 64))
    shapes.append((12, 512, 64, 512))
    return shapes


# FMT: (B, M, N, K)
DYNAMO_SHAPES_BMM = dynamo_shapes_bmm()


# Ref: https://github.com/FlagOpen/FlagGems/blob/master/benchmark/core_shapes.yaml#L56
def blas_shapes():
    shapes = []
    shapes.append((2, 4096, 4096, 4096))
    shapes.append((16, 384, 384, 384))
    shapes.append((16, 1024, 1024, 1024))
    shapes.append((16, 2048, 2048, 2048))
    shapes.append((16, 4096, 4096, 4096))
    return shapes


# FMT: (B, M, N, K)
BLAS_SHAPES = blas_shapes()


# Ref: {iluvatar_bitbucket}/projects/CSYSLIB/repos/ixblas/browse/bench/gemm_perf.cpp#1057
def ixblas_shapes_fp32():
    shapes = []
    shapes.append((3072, 4096, 30176))
    return shapes


# FMT: (M, N, K)
IXBLAS_SHAPES_FP32 = ixblas_shapes_fp32()


# Ref: {iluvatar_bitbucket}/projects/CSYSLIB/repos/ixblas/browse/bench/gemm_perf.cpp#1094
def ixblas_shapes_fp16():
    shapes = []
    if torch.cuda.get_device_capability()[0] >= 8:
        shapes.append((3072, 4096, 30176))
    else:
        shapes.append((2048, 2048, 8192))
    return shapes


# FMT: (M, N, K)
IXBLAS_SHAPES_FP16 = ixblas_shapes_fp16()


def deal_perf_mm_shapes():
    PERF_MM_SHAPES = LLAMA_SHAPES + DYNAMO_SHAPES_MM
    for _, m, n, k in BLAS_SHAPES:
        PERF_MM_SHAPES.append((m, n, k))

    return list(set(PERF_MM_SHAPES))


PERF_MM_SHAPES = deal_perf_mm_shapes()


def deal_perf_bmm_shapes():
    PERF_BMM_SHAPES = DYNAMO_SHAPES_BMM + BLAS_SHAPES
    return list(set(PERF_BMM_SHAPES))


PERF_BMM_SHAPES = deal_perf_bmm_shapes()
