import sys
import itertools
import logging

import numpy as np
import torch
from torch import nn

import triton
import triton.language as tl

logger = logging.getLogger(__name__)
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"x_block": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    # qzeros_ptr,
    out_ptr,
    numels,
    # maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    out_type: tl.constexpr,
    x_block: tl.constexpr,
):
    # Block indexing
    xoffset = tl.program_id(0) * x_block
    x_index = xoffset + tl.arange(0, x_block)
    xmask = x_index < numels

    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures

    elements_per_feature: tl.constexpr = 32 // bits
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx // elements_per_feature + (outfeatures // elements_per_feature * row_idx)),
        None,
    )

    wf_weights = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (groups * outfeatures + col_idx), None).to(out_type)
    # Unpack weights
    weights = qweights >> wf_weights  # bit shift qweight

    weights = weights & 0xFF
    weights = weights.to(tl.int8).to(out_type)
    weights = scales * weights

    tl.store(out_ptr + (x_index), weights, mask=xmask)


def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None, dtype=torch.float16):
    """
    Launcher for triton dequant kernel.  Only valid for bits = 2, 4, 8
    # compress_ratio = 32 / quant_bit
    # float_weight [IC, OC]
    # qweight [IC, OC / compress_ratio]
    # scales [IC / group_size, OC]
    # qzeros [IC / group_size, OC / compress_ratio]
    # g_idx [IC]
    """
    _ = qzeros
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]
    out = torch.empty((infeatures, outfeatures), device="cpu", dtype=dtype)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    if dtype == torch.float16:
        out_type = tl.float16
    elif dtype == torch.bfloat16:
        out_type = tl.bfloat16
    else:
        raise NotImplementedError(f"dtype: {dtype}")

    # grid = lambda meta: (triton.cdiv(numels, meta["x_block"]),)  # noqa: E731
    def grid(meta):
        return (triton.cdiv(numels, meta["x_block"]),)

    g_idx = g_idx.to(DEVICE)
    scales = scales.to(DEVICE)
    qweight = qweight.to(DEVICE)
    out = out.to(DEVICE)
    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        # qzeros,
        out,
        numels,
        # maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
        out_type=out_type,
    )
    out = out.to('cpu')
    return out


# Copied from https://github.com/IST-DASLab/marlin/pull/1
def _unpack_weight(qweight, qzeros, weight_width=4, row=False):
    # Unpack 4-bit values and interpret them as signed integers
    assert weight_width in [4, 8], "weight only support 4 or 8."
    compress_ratio = int(32 / weight_width)
    if row:
        unpacked_shape = (qweight.shape[0] * compress_ratio, qweight.shape[1])
        idx_list = list(range(qweight.shape[0]))
    else:
        unpacked_shape = (qweight.shape[0], qweight.shape[1] * compress_ratio)
        idx_list = list(range(qweight.shape[1]))

    unpacked_weights = torch.zeros(
        unpacked_shape,
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )

    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * compress_ratio),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    if weight_width == 4:
        mask = 0xF
    elif weight_width == 8:
        mask = 0xFF
    else:
        raise ValueError("weight_width must be 4 or 8")

    for i in range(compress_ratio):
        idx = (np.array(idx_list) * compress_ratio + i).tolist()
        if row:
            unpacked_weights[idx, :] = ((qweight >> (weight_width * i)) & mask).to(torch.int8)
        else:
            unpacked_weights[:, idx] = ((qweight >> (weight_width * i)) & mask).to(torch.int8)

    idx_list = list(range(qzeros.shape[1]))
    for i in range(compress_ratio):
        idx = (np.array(idx_list) * compress_ratio + i).tolist()
        unpacked_zeros[:, idx] = ((qzeros >> (weight_width * i)) & mask).to(torch.int8)

    return unpacked_weights, unpacked_zeros, compress_ratio


# Copied from https://github.com/IST-DASLab/marlin/pull/1
def dequantize_weight(qweight, qzeros, scales, weight_width=4, dtype=torch.float16, row=False):
    is_dtensor = False
    device_mesh = None
    placements = None
    if hasattr(qweight, "to_local"):
        is_dtensor = True
        device_mesh = qweight.device_mesh
        placements = qweight.placements
        qweight = qweight.to_local()
        qweight = qweight.view(torch.int32)

    if hasattr(qzeros, "to_local"):
        qzeros = qzeros.to_local()
        qzeros = qzeros.view(torch.int32)
    if hasattr(scales, "to_local"):
        scales = scales.to_local()

    unpacked_qweight, unpacked_qzeros, compress_ratio = _unpack_weight(qweight, qzeros, weight_width, row)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_dqweight = (unpacked_qweight.to(dtype) - unpacked_qzeros.to(dtype)) * scales
    if is_dtensor:
        from torch.distributed.tensor import DTensor

        unpacked_dqweight = DTensor.from_local(unpacked_dqweight, device_mesh=device_mesh, placements=placements)

    return unpacked_dqweight, group_size, compress_ratio


def dequantize_weight_triton(qweight, qzeros, scales, weight_width=4, dtype=torch.float16):
    if len(qweight.shape) == 4:
        group_size = qweight.shape[2] // scales.shape[2]
    else:
        group_size = qweight.shape[0] // scales.shape[0]
    compress_ratio = int(32 / weight_width)
    if len(qweight.shape) == 4:
        infeatures = qweight.shape[2]
    else:
        infeatures = qweight.shape[0]
    g_idx = torch.tensor([i // group_size for i in range(infeatures)], dtype=torch.int32, device=qweight.device)
    unpacked_dqweight_triton = dequant248(qweight, scales, qzeros, g_idx, weight_width, dtype=dtype)
    return unpacked_dqweight_triton, group_size, compress_ratio


class GPTQ(nn.Module):
    def __init__(
        self,
        mod,
        node_name,
        wq_params,
        prefix,
        # weight_dtype=torch.float16,
    ):
        super().__init__()
        self.bits = 4
        self.group_size = 128
        self.outfeatures, self.infeatures = mod.weight.shape
        self.weight = mod.weight
        self.bias = mod.bias
        self.maxq = 2**self.bits - 1

        # if node_name[-7:] == '_MatMul':
        #     prefix = node_name[:-7]
        # else:
        #     raise ValueError(f'can not identify prefix for {node_name}')

        prefix = prefix + node_name
        # self.g_idx = wq_params[prefix+'.g_idx']
        if prefix + ".qweight" in wq_params and prefix + ".qzeros" in wq_params and prefix + ".scales" in wq_params:
            self.qweight = wq_params[prefix + ".qweight"]
            self.qzeros = wq_params[prefix + ".qzeros"]
            self.scales = wq_params[prefix + ".scales"]
        elif prefix.startswith("model_") and prefix[len("model_") :] + ".qweight" in wq_params:
            prefix = prefix[len("model_") :]
            self.qweight = wq_params[prefix + ".qweight"]
            self.qzeros = wq_params[prefix + ".qzeros"]
            self.scales = wq_params[prefix + ".scales"]
        else:
            logger.error(f"GPTQ: {prefix} not found, please check or set this node in noquant_layers!")
            sys.exit(-1)

    def forward(self, x: torch.Tensor):
        raise NotImplementedError()
        # out_shape = x.shape[:-1] + (self.outfeatures,)
        # x = x.reshape(-1, x.shape[-1])

        # out = torch.matmul(x, self.weight)
        # out = out.to(x_dtype)
        # out = out + self.bias if self.bias is not None else out
        # return out


__all__ = ["GPTQ"]

if __name__ == "__main__":
    batch_size, num_channels, height, width = 1, 16, 64, 64
    qweight = torch.randint(0, 255, (batch_size, num_channels, height, width), dtype=torch.uint8)
    qzeros = torch.randint(0, 255, (batch_size, num_channels, height // 4, width // 4), dtype=torch.uint8)  # 假设零值压缩
    scales = torch.randn(batch_size, num_channels, height // 4, width // 4, dtype=torch.float16)

    unpacked_weight, group_size, compress_ratio = dequantize_weight_triton(
        qweight, qzeros, scales, weight_width=4, dtype=torch.float16
    )

    print(f"反量化权重形状: {unpacked_weight.shape}")
    print(f"分组大小: {group_size}")
    print(f"压缩比: {compress_ratio}")