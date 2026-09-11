#!/usr/bin/env python3
"""
使用triton的do_bench来测量性能，支持各种GPU设备

使用方法:
    export PYTHONPATH=/your/workspace/FlagGems/src
     bash third_party/tsingmicro/scripts/run_tsingmicro.sh pytest test_abs_cuda_time.py
"""

import pytest
import torch
import flag_gems

# 导入triton的do_bench，triton会自动处理不同设备的兼容性
try:
    import triton
    _do_bench = triton.testing.do_bench
except ImportError:
    raise ImportError("triton不可用，请安装triton以支持性能测试")

# 测试配置常量
BASE_SHAPES = [(256, 256), (4096, 4096), (16384, 16384)]
# BASE_SHAPES = [(256, 256)]
DTYPE = torch.float16
WARMUP = 1
REPETITION = 3

# 索引和形状限制常量（避免OOM）
MAX_INDEX_SIZE = 64
MAX_BATCH_SIZE = 8
MAX_SEQ_LEN = 256
MAX_KRON_1D_SIZE = 32
MAX_KRON_2D_SIZE = 16
MAX_KRON_ND_SIZE = 8
MAX_ISIN_TEST_ELEMENTS = 100

# 所有算子列表（从 op_list.md 提取，按字母顺序排序）
OP_LIST = [
    'abs',
    'add',
    'all',
    'amax',
    'angle',
    'any',
    'argmax',
    'argmin',
    'arange',
    'bitwise_and',
    'bitwise_not',
    'bitwise_or',
    'cat',
    'concat_and_cache_mla',
    'contiguous',
    'cos',
    'count_nonzero',
    'cross_entropy_loss',
    'cumsum',
    'diag_embed',
    'diagonal',
    'div',
    'dot',
    'dropout',
    'elu',
    'embedding',
    'eq',
    'erf',
    'exp',
    'eye',
    'fill',
    'flash_attention_forward',
    'flash_mla',
    'flip',
    'floor_divide',
    'full',
    'full_like',
    'fused_add_rms_norm',
    'gather',
    'gelu',
    'gelu_and_mul',
    'ge',
    'glu',
    'gt',
    'hstack',
    'index',
    'index_put',
    'index_select',
    'isin',
    'isfinite',
    'isinf',
    'isclose',
    'isnan',
    'layer_norm',
    'le',
    'linspace',
    'log',
    'log_softmax',
    'logical_and',
    'logical_not',
    'logical_or',
    'logical_xor',
    'lt',
    'masked_fill',
    'masked_select',
    'max',
    'maximum',
    'mean',
    'min',
    'minimum',
    'mm',
    'mse_loss',
    'mul',
    'multinomial',
    'mv',
    'matmul',
    'nan_to_num',
    'ne',
    'neg',
    'nll_loss',
    'nonzero',
    'normal',
    'ones',
    'ones_like',
    'outer',
    'pad',
    'pow',
    'prod',
    'rand',
    'rand_like',
    'randn',
    'randn_like',
    'reciprocal',
    'relu',
    'remainder',
    'repeat_interleave',
    'reshape_and_cache',
    'reshape_and_cache_flash',
    'resolve_conj',
    'resolve_neg',
    'rms_norm',
    'rsqrt',
    'rsub',
    'scaled_dot_product_attention',
    'scatter',
    'select',
    'sigmoid',
    'silu',
    'silu_and_mul',
    'slice_scatter',
    'softmax',
    'sort',
    'stack',
    'sub',
    'sum',
    'tanh',
    'threshold',
    'tile',
    'to',
    'topk',
    'triu',
    'unique',
    'vector_norm',
    'vdot',
    'vstack',
    'where',
    'zeros',
    'zeros_like',
    #'batch_norm','bitwise_xor', 'conv1d', 'conv2d', 'conv_depthwise2d','cummax', 'cummin', 'diag', 'group_norm',
    #, 'index_add','kron', 'lerp', 'log_sigmoid', 'polar', 'quantile', 'randperm', 'var_mean',
]

# 全局标志，确保只启用一次
_flag_gems_enabled = False


def _get_do_bench():
    """获取do_bench函数，triton会自动处理不同设备的兼容性"""
    return _do_bench


def _format_shape_str(op_name, inputs, config, shape):
    """格式化shape字符串用于显示和报告"""
    if config['type'] == 'matrix':
        if op_name == 'addmm':
            return f"bias{inputs[0].shape},mat1{inputs[1].shape},mat2{inputs[2].shape}"
        elif op_name == 'bmm':
            return f"mat1{inputs[0].shape},mat2{inputs[1].shape}"
        elif op_name == 'mv':
            return f"mat{inputs[0].shape},vec{inputs[1].shape}"
        else:
            return f"mat1{inputs[0].shape},mat2{inputs[1].shape}"
    elif config['type'] == 'ternary':
        if op_name == 'lerp':
            weight_str = str(inputs[2]) if isinstance(inputs[2], (int, float)) else str(inputs[2].shape)
            return f"inp{inputs[0].shape},end{inputs[1].shape},weight{weight_str}"
        else:
            return f"inp1{inputs[0].shape},inp2{inputs[1].shape},inp3{inputs[2].shape}"
    elif config['type'] == 'binary':
        return f"{inputs[0].shape}"
    elif config['type'] == 'multi_input':
        return f"inputs{len(inputs[0])}x{inputs[0][0].shape if inputs[0] else '()'}"
    elif config['type'] == 'special_constructor':
        if op_name == 'arange':
            end = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                shape if isinstance(shape, int) else 256)
            return f"arange(0, {end})"
        elif op_name == 'linspace':
            steps = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                shape if isinstance(shape, int) else 256)
            return f"linspace(0, 1, {steps})"
        else:
            return str(shape)
    else:
        return str(inputs.shape if hasattr(inputs, 'shape') else shape)


def _store_result(op_name, shape_str, config, avg_time_ms, avg_time_us, elapsed_time_ms, error=None):
    """存储测试结果到pytest._op_perf_results"""
    if not hasattr(pytest, '_op_perf_results'):
        pytest._op_perf_results = []

    result = {
        'op': op_name,
        'shape': shape_str,
        'config': str(config.get('extra_args', {})) if config.get('extra_args') else '',
        'dtype': str(DTYPE).split('.')[-1],
        'avg_time_ms': avg_time_ms,
        'avg_time_us': avg_time_us,
        'elapsed_time_ms': elapsed_time_ms,
        'type': config['type'],
    }

    if error:
        result['error'] = str(error)

    pytest._op_perf_results.append(result)


def _ensure_flag_gems_enabled():
    """确保FlagGems已启用，避免重复注册"""
    global _flag_gems_enabled
    if not _flag_gems_enabled:
        try:
            flag_gems.enable()
            _flag_gems_enabled = True
        except RuntimeError as e:
            # 如果已经启用，忽略重复注册的错误
            if "already a kernel registered" not in str(e):
                raise
            _flag_gems_enabled = True


def parse_op_list():
    """解析并返回所有算子列表"""
    return OP_LIST


def get_op_config(op_name):
    """
    为每个算子返回测试配置（shape适配和调用方式）

    Args:
        op_name: 算子名称

    Returns:
        dict: 包含type、shapes、extra_args、dtype等配置的字典
    """
    op_name_lower = op_name.lower()

    # 真正需要跳过的算子（不适合性能测试）
    skip_ops = {
        'fused_add_rms_norm',  # 复合算子
        'concat_and_cache_mla',  # 特殊缓存算子
        'reshape_and_cache',  # 特殊缓存算子
        'reshape_and_cache_flash',  # 特殊缓存算子
        'flash_attention_forward',  # 需要很多参数
        'flash_mla',  # 需要很多参数
        'scaled_dot_product_attention',  # 需要很多参数
        'nonzero',  # 返回indices，不适合性能测试
        'unique',  # 返回indices，不适合性能测试
        'to',  # dtype转换，不适合性能测试
        'contiguous',  # 内存操作，不适合性能测试
        'resolve_neg',  # 特殊算子
        'resolve_conj',  # 特殊算子
        'gelu_and_mul',  # 复合算子
        'silu_and_mul',  # 复合算子
    }

    # 需要固定配置的特殊算子（参考FlagGems/tests中的测试用例）
    special_fixed_config_ops = {
        'batch_norm',  # 需要 weight, bias, running_mean, running_var 等
        'layer_norm',  # 需要 normalized_shape, weight, bias 等
        'group_norm',  # 需要 num_groups, weight, bias 等
        'rms_norm',  # 可能需要特殊参数
        'cross_entropy_loss',  # 需要 target
        'nll_loss',  # 需要 target
        'mse_loss',  # 需要 target
        'embedding',  # 需要 indices
        'gather',  # 需要 indices
        'index',  # 需要 indices
        'index_add',  # 需要 indices
        'index_put',  # 需要 indices
        'index_select',  # 需要 index
        'scatter',  # 需要 index
        'slice_scatter',  # 需要 index
        'multinomial',  # 需要 num_samples
        'topk',  # 需要 k
        'quantile',  # 需要 q
        'where',  # 需要 condition
        'masked_fill',  # 需要 mask
        'masked_select',  # 需要 mask
        'select',  # 需要 dim 和 index
        'diagonal',  # 需要 offset, dim1, dim2
        'diag',  # 需要 offset
        'diag_embed',  # 需要 offset, dim1, dim2
        'pad',  # 需要 pad 参数
        'tile',  # 需要 dims 参数
        'repeat_interleave',  # 需要 repeats 参数
        'kron',  # 需要两个输入
        'outer',  # 需要两个输入
        'vdot',  # 需要两个输入
        'dot',  # 需要两个输入
        'polar',  # 需要两个输入
        'atan2',  # 需要两个输入
        'hypot',  # 需要两个输入
        'fmod',  # 需要两个输入
        'isin',  # 需要test_elements参数
        'conv1d',  # 需要 weight, bias, stride, padding 等
        'conv2d',  # 需要 weight, bias, stride, padding 等
        'conv_depthwise2d',  # 需要 weight, bias, stride, padding 等
        'fill',  # 需要 value 参数
    }

    if op_name_lower in skip_ops:
        return {'type': 'skip', 'reason': '不适合性能测试'}

    # 初始化config字典
    config = {
        'type': 'unary',  # default
        'shapes': BASE_SHAPES,
        'extra_args': {},
        'call_func': None,
    }

    if op_name_lower in special_fixed_config_ops:
        config['type'] = 'special_fixed_config'
        # 为每个特殊算子设置固定配置
        if op_name_lower == 'batch_norm':
            # batch_norm: (N, C, H, W) -> (N, C, H, W)
            config['shapes'] = [(16, 32, 32, 32), (32, 64, 64, 64), (64, 128, 128, 128)]
            config['extra_args'] = {'eps': 1e-5, 'momentum': 0.1, 'training': True}
        elif op_name_lower == 'layer_norm':
            # layer_norm: (N, C, H, W) -> (N, C, H, W)
            config['shapes'] = [(16, 32, 32, 32), (32, 64, 64, 64), (64, 128, 128, 128)]
            config['extra_args'] = {'eps': 1e-5}
        elif op_name_lower == 'group_norm':
            # group_norm: (N, C, H, W) -> (N, C, H, W)
            config['shapes'] = [(16, 32, 32, 32), (32, 64, 64, 64), (64, 128, 128, 128)]
            config['extra_args'] = {'num_groups': 8, 'eps': 1e-5}
        elif op_name_lower == 'rms_norm':
            # rms_norm: (N, C) -> (N, C)
            config['shapes'] = [(256, 512), (4096, 4096), (16384, 16384)]
            config['extra_args'] = {'eps': 1e-5}
        elif op_name_lower == 'conv1d':
            # conv1d: (N, C, L) -> (N, C_out, L_out)
            config['shapes'] = [(16, 32, 128), (32, 64, 256), (64, 128, 512)]
            config['extra_args'] = {'stride': 1, 'padding': 1, 'dilation': 1}
        elif op_name_lower == 'conv2d':
            # conv2d: (N, C, H, W) -> (N, C_out, H_out, W_out)
            config['shapes'] = [(16, 32, 32, 32), (32, 64, 64, 64), (64, 128, 128, 128)]
            config['extra_args'] = {'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1}
        elif op_name_lower == 'conv_depthwise2d':
            # conv_depthwise2d: (N, C, H, W) -> (N, C, H_out, W_out)
            config['shapes'] = [(16, 32, 32, 32), (32, 64, 64, 64), (64, 128, 128, 128)]
            config['extra_args'] = {'stride': 1, 'padding': 1, 'dilation': 1}
        elif op_name_lower == 'embedding':
            # embedding: (num_embeddings, embedding_dim), indices -> (indices_shape, embedding_dim)
            config['shapes'] = [(256, 512), (4096, 4096), (16384, 16384)]
            config['extra_args'] = {'num_embeddings': 10000, 'embedding_dim': 512}
        elif op_name_lower == 'gather':
            # gather: input, dim, index -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dim': 0}
        elif op_name_lower == 'index_select':
            # index_select: input, dim, index -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dim': 0}
        elif op_name_lower == 'scatter':
            # scatter: input, dim, index, src -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dim': 0}
        elif op_name_lower == 'topk':
            # topk: input, k -> (values, indices)
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'k': 10, 'dim': -1}
        elif op_name_lower == 'multinomial':
            # multinomial: input, num_samples -> indices
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'num_samples': 10}
        elif op_name_lower == 'quantile':
            # quantile: input, q -> output
            # quantile不支持float16，需要使用float32或float64
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'q': 0.5, 'dim': -1}
            config['dtype'] = torch.float32
        elif op_name_lower == 'where':
            # where: condition, x, y -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'masked_fill':
            # masked_fill: input, mask, value -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'value': 0.0}
        elif op_name_lower == 'masked_select':
            # masked_select: input, mask -> output (1D)
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'select':
            # select: input, dim, index -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dim': 0, 'index': 0}
        elif op_name_lower == 'diagonal':
            # diagonal: input, offset, dim1, dim2 -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'offset': 0, 'dim1': 0, 'dim2': 1}
        elif op_name_lower == 'diag':
            # diag: input, diagonal -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'diagonal': 0}
        elif op_name_lower == 'diag_embed':
            # diag_embed: input, offset, dim1, dim2 -> output
            config['shapes'] = [(256, ), (4096, ), (16384, )]
            config['extra_args'] = {'offset': 0, 'dim1': -2, 'dim2': -1}
        elif op_name_lower == 'pad':
            # pad: input, pad -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'pad': (1, 1, 1, 1), 'mode': 'constant', 'value': 0.0}
        elif op_name_lower == 'tile':
            # tile: input, dims -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dims': (2, 2)}
        elif op_name_lower == 'repeat_interleave':
            # repeat_interleave: input, repeats -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'repeats': 2, 'dim': -1}
        elif op_name_lower == 'kron':
            # kron: input1, input2 -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'outer':
            # outer: input1, input2 -> output
            config['shapes'] = [(256, ), (4096, ), (16384, )]
            config['extra_args'] = {}
        elif op_name_lower == 'vdot':
            # vdot: input1, input2 -> scalar
            config['shapes'] = [(256, ), (4096, ), (16384, )]
            config['extra_args'] = {}
        elif op_name_lower == 'dot':
            # dot: input1, input2 -> scalar or 1D
            config['shapes'] = [(256, ), (4096, ), (16384, )]
            config['extra_args'] = {}
        elif op_name_lower == 'polar':
            # polar: abs, angle -> complex
            # polar不支持float16，需要使用float32
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
            config['dtype'] = torch.float32
        elif op_name_lower == 'atan2':
            # atan2: input1, input2 -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'hypot':
            # hypot: input1, input2 -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'fmod':
            # fmod: input1, input2 -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        elif op_name_lower == 'slice_scatter':
            # slice_scatter: input, dim, src, start, end, step -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'dim': 0, 'start': 0, 'end': 128, 'step': 1}
        elif op_name_lower == 'isin':
            # isin: elements, test_elements -> output
            config['shapes'] = [(256, ), (4096, ), (16384, )]
            config['extra_args'] = {}
        elif op_name_lower == 'fill':
            # fill: input, value -> output
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {'value': 1.0}
        elif op_name_lower == 'cross_entropy_loss':
            # cross_entropy_loss: input, target -> loss
            config['shapes'] = [(256, 10), (4096, 100), (16384, 1000)]
            config['extra_args'] = {}
        elif op_name_lower == 'nll_loss':
            # nll_loss: input, target -> loss
            config['shapes'] = [(256, 10), (4096, 100), (16384, 1000)]
            config['extra_args'] = {}
        elif op_name_lower == 'mse_loss':
            # mse_loss: input, target -> loss
            config['shapes'] = BASE_SHAPES
            config['extra_args'] = {}
        return config

    # 规约算子：保持原shape，默认不带dim（测试全局规约性能）
    reduction_ops = {
        'sum', 'mean', 'max', 'min', 'prod', 'all', 'any', 'amax', 'amin', 'argmax', 'argmin', 'std', 'var', 'var_mean',
        'count_nonzero', 'norm', 'vector_norm'
    }

    # 累积算子：需要dim参数
    cumulative_ops = {'cummax', 'cummin', 'cumsum'}

    # 位运算算子：需要整数类型
    bitwise_ops = {'bitwise_and', 'bitwise_or', 'bitwise_not', 'bitwise_xor'}

    # 需要特殊数据类型的二元算子
    binary_ops_special_dtype = {
        'floor_divide': torch.float32,  # floor_divide不支持float16，需要使用float32
        'polar': torch.float32,  # polar不支持float16，需要使用float32或float64
    }

    # 需要特殊数据类型的构造函数算子
    constructor_ops_special_dtype = {
        'randperm': torch.int64,  # randperm只支持整数类型（int16/int32/int64），使用int64
    }

    # 二元算子：需要两个相同shape的输入
    binary_ops = {
        'add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'remainder',
        'logical_and', 'logical_or', 'logical_xor', 'fmod', 'atan2', 'hypot', 'rsub',  # rsub是二元算子（reverse subtract）
        'isclose',  # isclose是二元算子，需要两个输入和rtol/atol参数
    }

    # 三元算子：需要三个输入
    ternary_ops = {
        'lerp',  # lerp(input, end, weight) 需要三个输入，weight可以是标量或张量
    }

    # 一元逻辑算子
    unary_logical_ops = {
        'logical_not',  # logical_not是一元算子
    }

    # 矩阵运算：需要特定shape
    # mm: (M, K) x (K, N) -> (M, N)
    # bmm: (B, M, K) x (B, K, N) -> (B, M, N)
    # addmm: bias + (M, K) x (K, N) -> (M, N)
    # mv: (M, N) x (N,) -> (M,)
    matrix_ops = {
        'mm': lambda s: (s[0], s[0]),  # 返回 (M, N)，内部会创建 (M, K) 和 (K, N)
        'bmm': lambda s: (8, s[0], s[0]),  # 返回 batch size
        'addmm': lambda s: (s[0], s[0]),  # 返回 (M, N)
        'mv': lambda s: (s[0], s[0]),  # 返回 (M, N)
        'matmul': lambda s: (s[0], s[0]),  # 返回 (M, N)
    }

    # 需要特殊参数的算子
    special_ops = {
        'softmax': {'dim': -1}, 'log_softmax': {'dim': -1}, 'dropout': {'p': 0.5, 'train':
                                                                        True},  # dropout使用train参数，不是training
        'elu': {'alpha': 1.0},  # elu需要alpha参数
        'flip': {'dims': (0, )},  # flip需要dims参数，默认在dim=0上翻转
        'gelu': {}, 'silu': {}, 'relu': {}, 'sigmoid': {}, 'tanh': {}, 'exp': {}, 'log': {}, 'sqrt': {}, 'rsqrt': {},
        'abs': {}, 'neg': {}, 'reciprocal': {}, 'cos': {}, 'sin': {}, 'erf': {}, 'angle': {},  # 一元算子
        'glu': {},  # 一元算子
        'log_sigmoid': {},  # 一元算子
        'isfinite': {},  # 一元算子
        'isinf': {},  # 一元算子
        'isnan': {},  # 一元算子
        'nan_to_num': {},  # 一元算子
        'logical_not': {},  # 一元逻辑算子
        'threshold': {'threshold': 0.5, 'value': 0.0},  # threshold需要threshold和value参数
        'triu': {'diagonal': 0},  # triu需要diagonal参数
        'sort': {'dim': -1},  # sort需要dim参数
        'isclose': {'rtol': 1e-5, 'atol': 1e-8},  # isclose需要rtol和atol参数
    }

    # 需要多个输入的算子
    multi_input_ops = {
        'cat': lambda s: ([torch.randn(s, dtype=DTYPE, device=flag_gems.device) for _ in range(3)], {'dim': 0}),
        'stack': lambda s: ([torch.randn(s, dtype=DTYPE, device=flag_gems.device) for _ in range(3)], {'dim': 0}),
        'hstack': lambda s: ([torch.randn(s, dtype=DTYPE, device=flag_gems.device) for _ in range(3)], {}),
        'vstack': lambda s: ([torch.randn(s, dtype=DTYPE, device=flag_gems.device) for _ in range(3)], {}),
    }

    # 构造函数算子（不需要输入）
    constructor_ops = {
        'ones', 'zeros', 'eye', 'rand', 'randn', 'full', 'empty', 'ones_like', 'zeros_like', 'rand_like', 'randn_like',
        'full_like', 'empty_like', 'normal',  # normal需要mean和std参数，但可以设置默认值
        'randperm',  # randperm需要n参数
    }

    # 需要特殊参数的构造函数算子
    special_constructor_ops = {'arange', 'linspace'}

    # 根据算子类型设置配置
    if op_name_lower in reduction_ops:
        config['type'] = 'reduction'
        # 默认不带dim参数，测试全局规约性能（与原始测试用例 test_accuracy_sum_without_dim 一致）
        # vector_norm需要ord参数，默认使用2
        if op_name_lower == 'vector_norm':
            config['extra_args'] = {'ord': 2}
        else:
            config['extra_args'] = {}
    elif op_name_lower in cumulative_ops:
        config['type'] = 'cumulative'
        # 累积算子需要dim参数，默认使用最后一个维度
        config['extra_args'] = {'dim': -1}
    elif op_name_lower in bitwise_ops:
        config['type'] = 'bitwise'
        # 位运算需要整数类型，使用int16
        config['dtype'] = torch.int16
    elif op_name_lower in binary_ops_special_dtype:
        config['type'] = 'binary'
        # 需要特殊数据类型的二元算子
        config['dtype'] = binary_ops_special_dtype[op_name_lower]
    elif op_name_lower in ternary_ops:
        config['type'] = 'ternary'
        # 三元算子需要三个输入
    elif op_name_lower in binary_ops:
        config['type'] = 'binary'
        # 如果isclose在special_ops中，需要合并extra_args
        if op_name_lower == 'isclose':
            config['extra_args'] = special_ops.get('isclose', {})
    elif op_name_lower in unary_logical_ops:
        config['type'] = 'unary'
        # 一元逻辑算子使用special_ops中的配置
        if op_name_lower in special_ops:
            config['extra_args'] = special_ops[op_name_lower]
    elif op_name_lower in matrix_ops:
        config['type'] = 'matrix'
        config['shapes'] = [matrix_ops[op_name_lower](s) for s in BASE_SHAPES]
    elif op_name_lower in special_ops:
        config['type'] = 'unary'
        config['extra_args'] = special_ops[op_name_lower]
    elif op_name_lower in multi_input_ops:
        config['type'] = 'multi_input'
        config['call_func'] = multi_input_ops[op_name_lower]
    elif op_name_lower in special_constructor_ops:
        config['type'] = 'special_constructor'
        # arange和linspace需要特殊参数，不使用shape
        config['shapes'] = BASE_SHAPES
    elif op_name_lower in constructor_ops:
        config['type'] = 'constructor'
        # 构造函数使用shape作为输出shape
        config['shapes'] = BASE_SHAPES
        # 如果构造函数需要特殊数据类型，设置它
        if op_name_lower in constructor_ops_special_dtype:
            config['dtype'] = constructor_ops_special_dtype[op_name_lower]

    return config


def create_test_inputs(op_name, shape, config):
    """
    根据算子类型和配置创建测试输入

    Args:
        op_name: 算子名称
        shape: 输入shape
        config: 算子配置字典

    Returns:
        tensor或tuple: 测试输入，根据算子类型可能是单个tensor或tuple
    """
    # device = flag_gems.device
    device = "cpu"
    # 获取数据类型，如果配置中指定了则使用配置的，否则使用默认的DTYPE
    dtype = config.get('dtype', DTYPE)

    if config['type'] == 'bitwise':
        # 位运算需要整数类型，使用randint生成
        if op_name == 'bitwise_not':
            # 一元位运算
            return torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device=device).to(flag_gems.device)
        else:
            # 二元位运算
            return (torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype,
                                  device=device).to(flag_gems.device),
                    torch.randint(low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype,
                                  device=device).to(flag_gems.device))
    elif config['type'] == 'ternary':
        # 三元算子需要三个输入
        if op_name == 'lerp':
            # lerp(input, end, weight) - weight可以是标量或张量，这里使用标量
            return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
                    torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device), 0.5  # weight作为标量
                    )
        else:
            # 其他三元算子
            return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
                    torch.randn(shape, dtype=dtype,
                                device=device).to(flag_gems.device), torch.randn(shape, dtype=dtype,
                                                                                 device=device).to(flag_gems.device))
    elif config['type'] == 'binary':
        return (torch.randn(shape, dtype=dtype,
                            device=device).to(flag_gems.device), torch.randn(shape, dtype=dtype,
                                                                             device=device).to(flag_gems.device))
    elif config['type'] == 'matrix':
        if op_name == 'mm':
            # mm: (M, K) x (K, N) -> (M, N)
            M, N = shape
            K = N  # 使用N作为K
            return (torch.randn((M, K), dtype=DTYPE,
                                device=device).to(flag_gems.device), torch.randn((K, N), dtype=DTYPE,
                                                                                 device=device).to(flag_gems.device))
        elif op_name == 'bmm':
            # bmm: (B, M, K) x (B, K, N) -> (B, M, N)
            B, M, N = shape
            K = N
            return (torch.randn((B, M, K), dtype=DTYPE,
                                device=device).to(flag_gems.device), torch.randn((B, K, N), dtype=DTYPE,
                                                                                 device=device).to(flag_gems.device))
        elif op_name == 'addmm':
            # addmm: bias + (M, K) x (K, N) -> (M, N)
            M, N = shape
            K = N
            return (torch.randn((M, ), dtype=DTYPE, device=device).to(flag_gems.device),  # bias
                    torch.randn((M, K), dtype=DTYPE, device=device).to(flag_gems.device),  # mat1
                    torch.randn((K, N), dtype=DTYPE, device=device).to(flag_gems.device)  # mat2
                    )
        elif op_name == 'mv':
            # mv: (M, N) x (N,) -> (M,)
            M, N = shape
            return (torch.randn((M, N), dtype=DTYPE,
                                device=device).to(flag_gems.device), torch.randn((N, ), dtype=DTYPE,
                                                                                 device=device).to(flag_gems.device))
        elif op_name == 'matmul':
            # matmul: 同mm
            M, N = shape
            K = N
            return (torch.randn((M, K), dtype=DTYPE,
                                device=device).to(flag_gems.device), torch.randn((K, N), dtype=DTYPE,
                                                                                 device=device).to(flag_gems.device))
    elif config['type'] == 'multi_input':
        if config['call_func']:
            inputs, kwargs = config['call_func'](shape)
            return (inputs, kwargs)
    elif config['type'] == 'constructor':
        # 构造函数：shape作为输出shape参数
        return shape
    elif config['type'] == 'special_constructor':
        # arange和linspace需要特殊参数，返回None表示需要特殊处理
        return None
    elif config['type'] == 'special_fixed_config':
        # 特殊固定配置算子，根据算子类型创建相应的输入
        return _create_special_fixed_config_inputs(op_name, shape, config, dtype, device=device)

    # 默认：一元算子或规约算子
    dtype = config.get('dtype', DTYPE)
    return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)


def _create_special_fixed_config_inputs(op_name, shape, config, dtype, device):
    """为特殊固定配置算子创建测试输入"""
    if op_name == 'batch_norm':
        # batch_norm: input (N, C, H, W), weight (C,), bias (C,), running_mean (C,), running_var (C,)
        N, C, H, W = shape
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # bias
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # running_mean
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # running_var
                )
    elif op_name == 'layer_norm':
        # layer_norm: input (N, C, H, W), normalized_shape (C, H, W), weight (C*H*W,), bias (C*H*W,)
        N, C, H, W = shape
        normalized_shape = (C, H, W)
        norm_size = C * H * W
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                normalized_shape, torch.randn((norm_size, ), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((norm_size, ), dtype=dtype, device=device).to(flag_gems.device),  # bias
                )
    elif op_name == 'group_norm':
        # group_norm: input (N, C, H, W), num_groups, weight (C,), bias (C,)
        N, C, H, W = shape
        num_groups = config['extra_args'].get('num_groups', 8)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                num_groups, torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # bias
                )
    elif op_name == 'rms_norm':
        # rms_norm: input (N, C), weight (C,)
        N, C = shape
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # weight
                )
    elif op_name == 'conv1d':
        # conv1d: input (N, C, L), weight (C_out, C, K), bias (C_out,)
        N, C, L = shape
        C_out = C  # 输出通道数等于输入通道数
        K = 3  # 卷积核大小
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn((C_out, C, K), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((C_out, ), dtype=dtype, device=device).to(flag_gems.device),  # bias (可选)
                )
    elif op_name == 'conv2d':
        # conv2d: input (N, C, H, W), weight (C_out, C, K, K), bias (C_out,)
        N, C, H, W = shape
        C_out = C  # 输出通道数等于输入通道数
        K = 3  # 卷积核大小
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn((C_out, C, K, K), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((C_out, ), dtype=dtype, device=device).to(flag_gems.device),  # bias (可选)
                )
    elif op_name == 'conv_depthwise2d':
        # conv_depthwise2d: input (N, C, H, W), weight (C, 1, K, K), bias (C,)
        N, C, H, W = shape
        K = 3  # 卷积核大小
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn((C, 1, K, K), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randn((C, ), dtype=dtype, device=device).to(flag_gems.device),  # bias (可选)
                )
    elif op_name == 'embedding':
        # embedding: weight (num_embeddings, embedding_dim), indices (Batch, M)
        # 根据测试用例，indices应该是较小的2D shape，如(Batch, M)，而不是直接使用shape
        num_embeddings = config['extra_args'].get('num_embeddings', 4096)
        embedding_dim = config['extra_args'].get('embedding_dim', 512)
        # 限制indices的shape，避免内存溢出
        # 使用较小的batch和sequence length，参考测试用例中的(Batch, M)格式
        batch_size = min(shape[0] if len(shape) > 0 else 4, MAX_BATCH_SIZE)
        seq_len = min(shape[-1] if len(shape) > 1 else 128, MAX_SEQ_LEN)
        indices_shape = (batch_size, seq_len)
        return (torch.randn((num_embeddings, embedding_dim), dtype=dtype, device=device).to(flag_gems.device),  # weight
                torch.randint(0, num_embeddings, size=indices_shape, dtype=torch.long,
                              device=device).to(flag_gems.device),  # indices
                )
    elif op_name == 'gather':
        # gather: input, dim, index -> output
        dim = config['extra_args'].get('dim', 0)
        index_shape = list(shape)
        index_shape[dim] = min(shape[dim], MAX_INDEX_SIZE)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                dim, torch.randint(0, shape[dim], size=tuple(index_shape), dtype=torch.long,
                                   device=device).to(flag_gems.device),  # index
                )
    elif op_name == 'index_select':
        # index_select: input, dim, index -> output
        dim = config['extra_args'].get('dim', 0)
        index_size = min(shape[dim], MAX_INDEX_SIZE)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                dim, torch.randint(0, shape[dim], size=(index_size, ), dtype=torch.long,
                                   device=device).to(flag_gems.device),  # index
                )
    elif op_name == 'scatter':
        # scatter: input, dim, index, src -> output
        dim = config['extra_args'].get('dim', 0)
        index_shape = list(shape)
        index_shape[dim] = min(shape[dim], 64)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                dim, torch.randint(0, shape[dim], size=tuple(index_shape), dtype=torch.long,
                                   device=device).to(flag_gems.device),  # index
                torch.randn(tuple(index_shape), dtype=dtype, device=device).to(flag_gems.device),  # src
                )
    elif op_name == 'slice_scatter':
        # slice_scatter: input, dim, src, start, end, step -> output
        # slice_scatter(input, dim=dim, src=src, start=start, end=end, step=step)
        dim = config['extra_args'].get('dim', 0)
        start = config['extra_args'].get('start', 0)
        end = config['extra_args'].get('end', shape[dim])
        step = config['extra_args'].get('step', 1)
        # 计算 src 的形状
        size = shape[dim]
        start = start % size
        end = end % (size + 1)
        if end < start:
            end, start = start, end
        elif end == start:
            end = size
        src_size = (end - start + step - 1) // step
        src_shape = list(shape)
        src_shape[dim] = src_size
        return (
            torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
            dim,
            torch.randn(tuple(src_shape), dtype=dtype, device=device).to(flag_gems.device),  # src
            start,
            end,
            step,
        )
    elif op_name == 'index':
        # index: input, indices -> output
        # indices 是一个列表，包含多个索引张量，每个对应输入的一个维度
        # 为了简化，我们为每个维度创建一个索引张量
        indices = []
        for i, dim_size in enumerate(shape):
            # 为每个维度创建一个较小的索引张量
            index_size = min(dim_size, MAX_INDEX_SIZE)
            indices.append(
                torch.randint(0, dim_size, size=(index_size, ), dtype=torch.long, device=device).to(flag_gems.device))
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                indices,  # indices list
                )
    elif op_name == 'index_add':
        # index_add: input, dim, index, source -> output
        # index_add(input, dim, index, source, alpha=1)
        dim = config['extra_args'].get('dim', 0)
        index_max = shape[dim]
        index_len = min(index_max, MAX_INDEX_SIZE)
        index = torch.randperm(index_len, device=device).to(flag_gems.device)  # 1D索引张量
        src_shape = list(shape)
        src_shape[dim] = index_len
        source = torch.randn(tuple(src_shape), dtype=dtype, device=device).to(flag_gems.device)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                dim, index,  # index tensor
                source,  # source tensor
                )
    elif op_name == 'index_put':
        # index_put: input, indices, values -> output
        # index_put(input, indices, values, accumulate=False)
        # indices 是一个列表，包含多个索引张量，每个对应输入的一个维度
        indices = []
        for i, dim_size in enumerate(shape):
            # 为每个维度创建一个较小的索引张量
            index_size = min(dim_size, MAX_INDEX_SIZE)
            indices.append(
                torch.randint(0, dim_size, size=(index_size, ), dtype=torch.long, device=device).to(flag_gems.device))
        # values 的形状需要与 indices 广播后的形状匹配
        # 简化处理：使用第一个索引张量的形状作为 values 的形状
        if len(indices) > 0:
            values_shape = indices[0].shape
        else:
            values_shape = (MAX_INDEX_SIZE, )
        values = torch.randn(values_shape, dtype=dtype, device=device)
        return (torch.randn(shape, dtype=dtype, device=device),  # input
                indices,  # indices list
                values,  # values tensor
                )
    elif op_name == 'topk':
        # topk: input, k, dim -> (values, indices)
        k = config['extra_args'].get('k', 10)
        dim = config['extra_args'].get('dim', -1)
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'multinomial':
        # multinomial: input (probs), num_samples -> indices
        # input需要是概率分布，每行和为1
        probs = torch.rand(shape, dtype=dtype, device=device).to(flag_gems.device)
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 归一化为概率分布
        return probs
    elif op_name == 'quantile':
        # quantile: input, q, dim -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'where':
        # where: condition, x, y -> output
        condition = torch.rand(shape, dtype=dtype, device=device).to(flag_gems.device) > 0.5
        return (condition, torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # x
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # y
                )
    elif op_name == 'masked_fill':
        # masked_fill: input, mask, value -> output
        value = config['extra_args'].get('value', 0.0)
        return (
            torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
            torch.rand(shape, dtype=dtype, device=device).to(flag_gems.device) > 0.5,  # mask
            value,
        )
    elif op_name == 'masked_select':
        # masked_select: input, mask -> output (1D)
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.rand(shape, dtype=dtype, device=device).to(flag_gems.device) > 0.5,  # mask
                )
    elif op_name == 'select':
        # select: input, dim, index -> output
        dim = config['extra_args'].get('dim', 0)
        index = config['extra_args'].get('index', 0)
        return (
            torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
            dim,
            index,
        )
    elif op_name == 'diagonal':
        # diagonal: input, offset, dim1, dim2 -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'diag':
        # diag: input, diagonal -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'diag_embed':
        # diag_embed: input (1D), offset, dim1, dim2 -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'pad':
        # pad: input, pad -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'tile':
        # tile: input, dims -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name == 'repeat_interleave':
        # repeat_interleave: input, repeats, dim -> output
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)  # input
    elif op_name in ['kron', 'outer', 'vdot', 'dot', 'polar', 'atan2', 'hypot', 'fmod']:
        # 二元算子：需要两个输入
        if op_name == 'kron':
            # kron 的输出大小是输入大小的乘积，需要限制输入大小避免内存溢出
            # 参考测试用例中的 KRON_SHAPES，使用较小的形状
            if len(shape) == 1:
                # 1D: 限制大小避免OOM
                kron_shape1 = (min(shape[0], MAX_KRON_1D_SIZE), )
                kron_shape2 = (min(shape[0], MAX_KRON_1D_SIZE), )
            elif len(shape) == 2:
                # 2D: 限制大小避免OOM
                kron_shape1 = (min(shape[0], MAX_KRON_2D_SIZE), min(shape[1], MAX_KRON_2D_SIZE))
                kron_shape2 = (min(shape[0], MAX_KRON_2D_SIZE), min(shape[1], MAX_KRON_2D_SIZE))
            else:
                # 高维: 限制每个维度大小避免OOM
                kron_shape1 = tuple(min(s, MAX_KRON_ND_SIZE) for s in shape)
                kron_shape2 = tuple(min(s, MAX_KRON_ND_SIZE) for s in shape)
            return (
                torch.randn(kron_shape1, dtype=dtype, device=device).to(flag_gems.device),
                torch.randn(kron_shape2, dtype=dtype, device=device).to(flag_gems.device),
            )
        elif op_name in ['outer', 'vdot', 'dot']:
            # 这些算子需要1D输入
            return (
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
            )
        else:
            return (
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),
            )
    elif op_name == 'isin':
        # isin: elements, test_elements -> output
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # elements
                torch.randn((min(MAX_ISIN_TEST_ELEMENTS, shape[0]), ), dtype=dtype,
                            device=device).to(flag_gems.device),  # test_elements
                )
    elif op_name == 'fill':
        # fill: input, value -> output
        value = config['extra_args'].get('value', 1.0)
        return (
            torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
            value,
        )
    elif op_name == 'cross_entropy_loss':
        # cross_entropy_loss: input (N, C), target (N,)
        N, C = shape
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randint(0, C, size=(N, ), dtype=torch.long, device=device).to(flag_gems.device),  # target
                )
    elif op_name == 'nll_loss':
        # nll_loss: input (N, C), target (N,)
        N, C = shape
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randint(0, C, size=(N, ), dtype=torch.long, device=device).to(flag_gems.device),  # target
                )
    elif op_name == 'mse_loss':
        # mse_loss: input, target -> loss
        return (torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # input
                torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device),  # target
                )
    else:
        # 默认：一元算子
        return torch.randn(shape, dtype=dtype, device=device).to(flag_gems.device)


def call_op(op_name, inputs, config, shape=None):
    """
    调用算子

    注意：由于性能测试需要多次调用，使用全局启用的flag_gems.enable()
    而不是每次调用use_gems()，避免重复注册错误。

    Args:
        op_name: 算子名称
        inputs: 输入tensor或tuple
        config: 算子配置字典
        shape: 可选的shape参数（用于构造函数）

    Returns:
        tensor: 算子输出
    """
    # 某些算子需要使用torch.nn.functional
    nn_functional_ops = {
        'dropout': torch.nn.functional.dropout, 'elu': torch.nn.functional.elu, 'relu': torch.nn.functional.relu,
        'gelu': torch.nn.functional.gelu, 'silu': torch.nn.functional.silu, 'sigmoid': torch.nn.functional.sigmoid,
        'tanh': torch.nn.functional.tanh, 'log_sigmoid':
        torch.nn.functional.logsigmoid,  # log_sigmoid在torch.nn.functional中是logsigmoid
    }

    if op_name in nn_functional_ops:
        op_func = nn_functional_ops[op_name]
    else:
        # 特殊处理：vector_norm在torch.linalg中
        if op_name == 'vector_norm':
            try:
                op_func = torch.linalg.vector_norm
            except AttributeError:
                # 如果torch.linalg不存在，尝试torch.vector_norm
                op_func = getattr(torch, 'vector_norm', None)
        else:
            op_func = getattr(torch, op_name, None)
            if op_func is None:
                # 尝试下划线版本
                op_func = getattr(torch, op_name + '_', None)
            if op_func is None:
                # 尝试torch.nn.functional
                op_func = getattr(torch.nn.functional, op_name, None)
            if op_func is None:
                # 尝试torch.linalg（对于其他linalg算子）
                if hasattr(torch, 'linalg'):
                    op_func = getattr(torch.linalg, op_name, None)

    # 对于special_fixed_config类型的算子，op_func可能为None（在_call_special_fixed_config_op中直接调用）
    if op_func is None and config.get('type') != 'special_fixed_config':
        raise ValueError(f"未找到算子: {op_name}")

    extra_args = config.get('extra_args', {})

    # 直接调用，因为flag_gems已经全局启用
    return _call_op_impl(op_func, op_name, inputs, config, extra_args, shape)


def _call_special_fixed_config_op(op_func, op_name, inputs, config, extra_args, shape=None):
    """调用特殊固定配置算子"""
    if op_name == 'batch_norm':
        # batch_norm(input, weight, bias, running_mean, running_var, ...)
        return torch.nn.functional.batch_norm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], **extra_args)
    elif op_name == 'layer_norm':
        # layer_norm(input, normalized_shape, weight, bias, ...)
        return torch.layer_norm(inputs[0], inputs[1], weight=inputs[2], bias=inputs[3], **extra_args)
    elif op_name == 'group_norm':
        # group_norm(input, num_groups, weight, bias, ...)
        # num_groups 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'num_groups'}
        return torch.nn.functional.group_norm(inputs[0], inputs[1], weight=inputs[2], bias=inputs[3], **filtered_args)
    elif op_name == 'rms_norm':
        # rms_norm(input, weight, ...)
        return torch.nn.functional.layer_norm(inputs[0], (inputs[0].shape[-1], ), weight=inputs[1], **extra_args)
    elif op_name == 'conv1d':
        # conv1d(input, weight, bias=None, ...)
        return torch.nn.functional.conv1d(inputs[0], inputs[1], bias=inputs[2], **extra_args)
    elif op_name == 'conv2d':
        # conv2d(input, weight, bias=None, ...)
        return torch.nn.functional.conv2d(inputs[0], inputs[1], bias=inputs[2], **extra_args)
    elif op_name == 'conv_depthwise2d':
        # conv_depthwise2d(input, weight, bias=None, ...)
        return torch.nn.functional.conv2d(inputs[0], inputs[1], bias=inputs[2], groups=inputs[0].shape[1], **extra_args)
    elif op_name == 'embedding':
        # embedding(indices, weight, ...)
        # num_embeddings 和 embedding_dim 只是用于创建 weight 的参数，不应该传递给 embedding 函数
        filtered_args = {k: v for k, v in extra_args.items() if k not in ['num_embeddings', 'embedding_dim']}
        return torch.nn.functional.embedding(inputs[1], inputs[0], **filtered_args)
    elif op_name == 'gather':
        # gather(input, dim, index, ...)
        # dim 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'dim'}
        return torch.gather(inputs[0], inputs[1], inputs[2], **filtered_args)
    elif op_name == 'index_select':
        # index_select(input, dim, index, ...)
        # dim 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'dim'}
        return torch.index_select(inputs[0], inputs[1], inputs[2], **filtered_args)
    elif op_name == 'scatter':
        # scatter(input, dim, index, src, ...)
        # dim 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'dim'}
        return torch.scatter(inputs[0], inputs[1], inputs[2], inputs[3], **filtered_args)
    elif op_name == 'slice_scatter':
        # slice_scatter(input, dim=dim, src=src, start=start, end=end, step=step)
        # dim, start, end, step 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k not in ['dim', 'start', 'end', 'step']}
        return torch.slice_scatter(inputs[0], dim=inputs[1], src=inputs[2], start=inputs[3], end=inputs[4],
                                   step=inputs[5], **filtered_args)
    elif op_name == 'index':
        # index(input, indices) -> output
        # indices 是一个列表，包含多个索引张量
        return torch.ops.aten.index(inputs[0], inputs[1])
    elif op_name == 'index_add':
        # index_add(input, dim, index, source, alpha=1) -> output
        # dim 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'dim'}
        return torch.index_add(inputs[0], inputs[1], inputs[2], inputs[3], **filtered_args)
    elif op_name == 'index_put':
        # index_put(input, indices, values, accumulate=False) -> output
        # indices 是一个列表，包含多个索引张量
        return torch.index_put(inputs[0], inputs[1], inputs[2], **extra_args)
    elif op_name == 'topk':
        # topk(input, k, dim, ...) -> (values, indices)
        k = extra_args.get('k', 10)
        dim = extra_args.get('dim', -1)
        result = torch.topk(inputs, k, dim=dim)
        return result.values  # 只返回values用于性能测试
    elif op_name == 'multinomial':
        # multinomial(input, num_samples, ...)
        num_samples = extra_args.get('num_samples', 10)
        return torch.multinomial(inputs, num_samples, **{k: v for k, v in extra_args.items() if k != 'num_samples'})
    elif op_name == 'quantile':
        # quantile(input, q, dim, ...)
        q = extra_args.get('q', 0.5)
        dim = extra_args.get('dim', -1)
        return torch.quantile(inputs, q, dim=dim, **{k: v for k, v in extra_args.items() if k not in ['q', 'dim']})
    elif op_name == 'where':
        # where(condition, x, y, ...)
        return torch.where(inputs[0], inputs[1], inputs[2], **extra_args)
    elif op_name == 'masked_fill':
        # masked_fill(input, mask, value, ...)
        # value 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k != 'value'}
        return torch.masked_fill(inputs[0], inputs[1], inputs[2], **filtered_args)
    elif op_name == 'masked_select':
        # masked_select(input, mask, ...)
        return torch.masked_select(inputs[0], inputs[1], **extra_args)
    elif op_name == 'select':
        # select(input, dim, index, ...)
        # dim 和 index 已经通过位置参数传递，不应该再通过 extra_args 传递
        filtered_args = {k: v for k, v in extra_args.items() if k not in ['dim', 'index']}
        return torch.select(inputs[0], inputs[1], inputs[2], **filtered_args)
    elif op_name == 'diagonal':
        # diagonal(input, offset, dim1, dim2, ...)
        offset = extra_args.get('offset', 0)
        dim1 = extra_args.get('dim1', 0)
        dim2 = extra_args.get('dim2', 1)
        return torch.diagonal(inputs, offset=offset, dim1=dim1, dim2=dim2)
    elif op_name == 'diag':
        # diag(input, diagonal, ...)
        diagonal = extra_args.get('diagonal', 0)
        return torch.diag(inputs, diagonal=diagonal)
    elif op_name == 'diag_embed':
        # diag_embed(input, offset, dim1, dim2, ...)
        offset = extra_args.get('offset', 0)
        dim1 = extra_args.get('dim1', -2)
        dim2 = extra_args.get('dim2', -1)
        return torch.diag_embed(inputs, offset=offset, dim1=dim1, dim2=dim2)
    elif op_name == 'pad':
        # pad(input, pad, mode, value, ...)
        pad = extra_args.get('pad', (1, 1, 1, 1))
        mode = extra_args.get('mode', 'constant')
        value = extra_args.get('value', 0.0)
        return torch.nn.functional.pad(inputs, pad, mode=mode, value=value)
    elif op_name == 'tile':
        # tile(input, dims, ...)
        dims = extra_args.get('dims', (2, 2))
        return torch.tile(inputs, dims)
    elif op_name == 'repeat_interleave':
        # repeat_interleave(input, repeats, dim, ...)
        repeats = extra_args.get('repeats', 2)
        dim = extra_args.get('dim', -1)
        return torch.repeat_interleave(inputs, repeats, dim=dim)
    elif op_name == 'kron':
        # kron(input1, input2, ...)
        return torch.kron(inputs[0], inputs[1], **extra_args)
    elif op_name == 'outer':
        # outer(input1, input2, ...)
        return torch.outer(inputs[0], inputs[1], **extra_args)
    elif op_name == 'vdot':
        # vdot(input1, input2, ...)
        return torch.vdot(inputs[0], inputs[1], **extra_args)
    elif op_name == 'dot':
        # dot(input1, input2, ...)
        return torch.dot(inputs[0], inputs[1], **extra_args)
    elif op_name == 'polar':
        # polar(abs, angle, ...)
        return torch.polar(inputs[0], inputs[1], **extra_args)
    elif op_name == 'atan2':
        # atan2(input1, input2, ...)
        return torch.atan2(inputs[0], inputs[1], **extra_args)
    elif op_name == 'hypot':
        # hypot(input1, input2, ...)
        return torch.hypot(inputs[0], inputs[1], **extra_args)
    elif op_name == 'fmod':
        # fmod(input1, input2, ...)
        return torch.fmod(inputs[0], inputs[1], **extra_args)
    elif op_name == 'isin':
        # isin(elements, test_elements, ...)
        return torch.isin(inputs[0], inputs[1], **extra_args)
    elif op_name == 'fill':
        # fill(input, value, ...) - 注意：fill是inplace操作
        result = inputs[0].clone()
        result.fill_(inputs[1])
        return result
    elif op_name == 'cross_entropy_loss':
        # cross_entropy_loss(input, target, ...)
        return torch.nn.functional.cross_entropy(inputs[0], inputs[1], **extra_args)
    elif op_name == 'nll_loss':
        # nll_loss(input, target, ...)
        return torch.nn.functional.nll_loss(inputs[0], inputs[1], **extra_args)
    elif op_name == 'mse_loss':
        # mse_loss(input, target, ...)
        return torch.nn.functional.mse_loss(inputs[0], inputs[1], **extra_args)
    else:
        raise ValueError(f"未知的特殊固定配置算子: {op_name}")


def _call_op_impl(op_func, op_name, inputs, config, extra_args, shape=None):
    """实际的算子调用实现"""
    if config['type'] == 'bitwise':
        # 位运算：一元或二元
        if op_name == 'bitwise_not':
            return op_func(inputs, **extra_args)
        else:
            return op_func(inputs[0], inputs[1], **extra_args)
    elif config['type'] == 'ternary':
        # 三元算子
        if op_name == 'lerp':
            # lerp(input, end, weight) - weight可以是标量或张量
            return op_func(inputs[0], inputs[1], weight=inputs[2], **extra_args)
        else:
            return op_func(inputs[0], inputs[1], inputs[2], **extra_args)
    elif config['type'] == 'binary':
        return op_func(inputs[0], inputs[1], **extra_args)
    elif config['type'] == 'matrix':
        if op_name == 'addmm':
            # addmm(bias, mat1, mat2, ...)
            return op_func(inputs[0], inputs[1], inputs[2], **extra_args)
        elif op_name in ['bmm', 'mm', 'matmul']:
            return op_func(inputs[0], inputs[1], **extra_args)
        elif op_name == 'mv':
            return op_func(inputs[0], inputs[1], **extra_args)
    elif config['type'] == 'multi_input':
        input_list, kwargs = inputs
        merged_kwargs = {**extra_args, **kwargs}
        return op_func(input_list, **merged_kwargs)
    elif config['type'] == 'constructor':
        # 构造函数使用shape作为参数
        if op_name == 'full_like':
            # full_like(input, fill_value) 需要参考tensor和fill_value
            ref_tensor = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
            fill_value = 1.0
            return op_func(ref_tensor, fill_value, dtype=DTYPE, device=flag_gems.device, **extra_args)
        elif 'like' in op_name:
            # ones_like等需要参考tensor（inputs就是shape，需要创建参考tensor）
            ref_tensor = torch.randn(shape, dtype=DTYPE, device=flag_gems.device)
            return op_func(ref_tensor, **extra_args)
        elif op_name == 'eye':
            # eye需要矩阵大小，使用shape的第一个维度
            n = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else shape
            m = shape[1] if isinstance(shape, tuple) and len(shape) > 1 else n
            return op_func(n, m, dtype=DTYPE, device=flag_gems.device, **extra_args)
        elif op_name == 'normal':
            # normal(mean, std, size) 需要mean和std参数
            mean = 0.0
            std = 1.0
            return op_func(mean, std, shape, dtype=DTYPE, device=flag_gems.device, **extra_args)
        elif op_name == 'randperm':
            # randperm(n) 需要n参数，使用shape的第一个维度
            # randperm只支持整数类型（int16/int32/int64），使用配置中的dtype或默认int64
            dtype = config.get('dtype', torch.int64)
            n = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (shape if isinstance(shape, int) else 256)
            return op_func(n, dtype=dtype, device=flag_gems.device, **extra_args)
        elif op_name == 'full':
            # full(size, fill_value) 需要fill_value参数
            fill_value = 1.0
            return op_func(shape, fill_value, dtype=DTYPE, device=flag_gems.device, **extra_args)
        else:
            # ones, zeros, rand, randn, empty等直接使用shape
            return op_func(shape, dtype=DTYPE, device=flag_gems.device, **extra_args)
    elif config['type'] == 'special_constructor':
        # arange和linspace需要特殊参数
        if op_name == 'arange':
            # arange(start, end, step, ...)
            # 使用shape的第一个维度作为end值
            end = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                shape if isinstance(shape, int) else 256)
            return op_func(0, end, dtype=DTYPE, device=flag_gems.device, **extra_args)
        elif op_name == 'linspace':
            # linspace(start, end, steps, ...)
            # 使用shape的第一个维度作为steps值
            steps = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                shape if isinstance(shape, int) else 256)
            return op_func(0.0, 1.0, steps, dtype=DTYPE, device=flag_gems.device, **extra_args)
        else:
            raise ValueError(f"未知的特殊构造函数算子: {op_name}")
    elif config['type'] == 'cumulative':
        # 累积算子：cummax和cummin返回命名元组(values, indices)，cumsum返回tensor
        result = op_func(inputs, **extra_args)
        if op_name in ['cummax', 'cummin']:
            # 返回values部分用于性能测试
            return result.values
        return result
    elif config['type'] == 'special_fixed_config':
        # 特殊固定配置算子
        return _call_special_fixed_config_op(op_func, op_name, inputs, config, extra_args, shape)
    elif op_name == 'dropout':
        # dropout需要位置参数：dropout(input, p, train)
        # 使用torch.nn.functional.dropout
        p = extra_args.get('p', 0.5)
        train = extra_args.get('train', True)
        return torch.nn.functional.dropout(inputs, p, train)
    elif op_name == 'flip':
        # flip需要位置参数：flip(input, dims)
        dims = extra_args.get('dims', (0, ))
        return op_func(inputs, dims)
    elif op_name == 'threshold':
        # threshold需要位置参数：threshold(input, threshold, value)
        threshold = extra_args.get('threshold', 0.5)
        value = extra_args.get('value', 0.0)
        return op_func(inputs, threshold, value)
    elif op_name == 'triu':
        # triu需要位置参数：triu(input, diagonal)
        diagonal = extra_args.get('diagonal', 0)
        return op_func(inputs, diagonal)
    elif op_name == 'sort':
        # sort需要位置参数：sort(input, dim)
        dim = extra_args.get('dim', -1)
        result = op_func(inputs, dim=dim)
        # sort返回(values, indices)，只返回values用于性能测试
        return result.values if hasattr(result, 'values') else result[0]
    elif op_name == 'vector_norm':
        # vector_norm需要ord参数，dim和keepdim可选
        ord = extra_args.get('ord', 2)
        dim = extra_args.get('dim', None)
        keepdim = extra_args.get('keepdim', False)
        if dim is not None:
            return op_func(inputs, ord=ord, dim=dim, keepdim=keepdim)
        else:
            return op_func(inputs, ord=ord, keepdim=keepdim)
    elif op_name == 'isclose':
        # isclose需要两个输入和rtol/atol参数
        if config['type'] == 'binary':
            rtol = extra_args.get('rtol', 1e-5)
            atol = extra_args.get('atol', 1e-8)
            return op_func(inputs[0], inputs[1], rtol=rtol, atol=atol)
        else:
            # 如果只有一个输入，创建第二个输入
            inp2 = torch.randn_like(inputs)
            rtol = extra_args.get('rtol', 1e-5)
            atol = extra_args.get('atol', 1e-8)
            return op_func(inputs, inp2, rtol=rtol, atol=atol)
    else:
        # 一元算子或规约算子
        return op_func(inputs, **extra_args)


def test_op_performance(op_name, shape, config):
    """
    测试单个算子的性能

    使用triton的do_bench来测量性能，支持各种GPU设备。
    注意：使用全局启用的flag_gems.enable()，避免每次调用use_gems()导致的重复注册。

    Args:
        op_name: 算子名称
        shape: 测试shape
        config: 算子配置字典

    Returns:
        float: 平均耗时（毫秒）
    """
    do_bench = _get_do_bench()
    _ensure_flag_gems_enabled()

    try:
        # 跳过需要特殊参数的算子
        if config.get('type') == 'skip':
            pytest.skip(f"算子 {op_name}: {config.get('reason', '需要特殊参数')}")

        # 创建测试输入
        inputs = create_test_inputs(op_name, shape, config)

        # 定义要测试的函数
        if inputs is None and config['type'] == 'special_constructor':

            def test_fn():
                return call_op(op_name, None, config, shape=shape)

            # special_constructor的shape格式化
            if op_name == 'arange':
                end = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                    shape if isinstance(shape, int) else 256)
                shape_str = f"arange(0, {end})"
            elif op_name == 'linspace':
                steps = shape[0] if isinstance(shape, tuple) and len(shape) > 0 else (
                    shape if isinstance(shape, int) else 256)
                shape_str = f"linspace(0, 1, {steps})"
            else:
                shape_str = str(shape)
        else:

            def test_fn():
                return call_op(op_name, inputs, config, shape=shape)

            shape_str = _format_shape_str(op_name, inputs, config, shape)

        # 使用do_bench测量性能（返回中位数，单位：毫秒）
        avg_time_ms = do_bench(test_fn, warmup=WARMUP, rep=REPETITION, return_mode="median")
        avg_time_us = avg_time_ms * 1000
        elapsed_time_ms = avg_time_ms * REPETITION

        # 存储结果
        _store_result(op_name, shape_str, config, avg_time_ms, avg_time_us, elapsed_time_ms)

        # 运行时打印
        print(f"    shape={shape_str:<35} avg_time={avg_time_us:>10.2f} us ({avg_time_ms:>8.4f} ms)", flush=True)

        return avg_time_ms

    except Exception as e:
        # 记录失败的测试
        shape_str = str(shape)
        _store_result(op_name, shape_str, config, None, None, None, error=str(e))
        print(f"    shape={shape_str:<35} ERROR: {str(e)}", flush=True)
        pytest.skip(f"算子 {op_name} 测试失败: {e}")


# 为了pytest兼容性，创建一个通用的测试函数
@pytest.mark.parametrize("op_name", [])
def test_op_performance_pytest(op_name):
    """pytest版本的测试函数（通过parametrize动态生成）"""
    ops = parse_op_list()
    if op_name not in ops:
        pytest.skip(f"未知算子: {op_name}")

    config = get_op_config(op_name)
    # 使用第一个shape作为默认测试
    shape = config['shapes'][0] if config['shapes'] else BASE_SHAPES[0]
    test_op_performance(op_name, shape, config)


def print_summary_report():
    """
    打印性能测试总结报告

    按算子分组显示所有测试结果，包括shape、配置、耗时等信息。
    """
    if not hasattr(pytest, '_op_perf_results') or not pytest._op_perf_results:
        return

    results = pytest._op_perf_results

    # 按算子分组
    op_groups = {}
    for r in results:
        op = r['op']
        if op not in op_groups:
            op_groups[op] = []
        op_groups[op].append(r)

    print("\n" + "=" * 120)
    print(" " * 40 + "算子性能测试报告")
    print("=" * 120)

    # 打印每个算子的结果
    for op_name in sorted(op_groups.keys()):
        op_results = op_groups[op_name]
        print(f"\n算子: {op_name.upper()}")
        print("=" * 120)
        print(f"{'Shape':<35} {'Config':<20} {'Avg Time (us)':<18} {'Avg Time (ms)':<18} {'Status':<15}")
        print("=" * 120)

        for r in op_results:
            shape_str = r['shape'][:34] if len(r['shape']) > 34 else r['shape']
            config_str = r['config'][:19] if len(r['config']) > 19 else r['config']

            if r.get('error'):
                print(f"{shape_str:<35} {config_str:<20} {'N/A':<18} {'N/A':<18} {'ERROR':<15}")
                print(f"  Error: {r['error']}")
            elif r['avg_time_us'] is not None:
                print(
                    f"{shape_str:<35} {config_str:<20} {r['avg_time_us']:>15.2f}  {r['avg_time_ms']:>15.4f}  {'OK':<15}"
                )
            else:
                print(f"{shape_str:<35} {config_str:<20} {'N/A':<18} {'N/A':<18} {'SKIPPED':<15}")

    print("\n" + "=" * 120)
    print(f"测试配置:")
    print(f"  * Warmup: {WARMUP}")
    print(f"  * Repetition: {REPETITION}")
    print(f"  * Dtype: {DTYPE}")
    _ensure_flag_gems_enabled()
    print(f"  * Device: {flag_gems.device}")
    print(f"  * Benchmark Method: triton.do_bench")
    print("=" * 120 + "\n")


def print_csv_report(filename='performance_report.csv'):
    """
    将性能测试报告输出为CSV格式

    Args:
        filename: CSV文件名，默认为'performance_report.csv'

    Note:
        使用&作为分隔符，避免shape字段中的逗号导致格式混乱。
    """
    import csv
    import os

    if not hasattr(pytest, '_op_perf_results') or not pytest._op_perf_results:
        print("没有性能测试结果可输出")
        return

    results = pytest._op_perf_results

    # CSV文件路径
    csv_path = os.path.join(os.path.dirname(__file__), filename)

    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'Operator', 'Shape', 'Config', 'Dtype', 'Type', 'Avg Time (us)', 'Avg Time (ms)', 'Elapsed Time (ms)',
            'Status', 'Error'
        ]
        # 使用&作为分隔符，避免shape字段中的逗号导致格式混乱
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='&', quoting=csv.QUOTE_MINIMAL)

        # 写入表头
        writer.writeheader()

        # 写入数据
        for r in results:
            status = 'OK'
            if r.get('error'):
                status = 'ERROR'
            elif r['avg_time_us'] is None:
                status = 'SKIPPED'

            # 将所有字段转换为字符串，确保CSV格式正确
            # csv模块会自动为包含逗号的字段添加引号
            writer.writerow({
                'Operator':
                str(r['op']), 'Shape':
                str(r['shape']),  # 包含逗号的字段会被自动用引号括起来
                'Config':
                str(r['config']),  # 包含逗号的字段会被自动用引号括起来
                'Dtype':
                str(r['dtype']), 'Type':
                str(r['type']), 'Avg Time (us)':
                f"{r['avg_time_us']:.2f}" if r['avg_time_us'] is not None else 'N/A', 'Avg Time (ms)':
                f"{r['avg_time_ms']:.4f}" if r['avg_time_ms'] is not None else 'N/A', 'Elapsed Time (ms)':
                f"{r['elapsed_time_ms']:.4f}" if r.get('elapsed_time_ms') is not None else 'N/A', 'Status':
                str(status), 'Error':
                str(r.get('error', ''))
            })

    print(f"\n性能测试报告已保存为CSV格式: {csv_path}")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """pytest session结束时打印总结报告并输出CSV"""
    print_summary_report()
    print_csv_report()


if __name__ == "__main__":
    # 启用FlagGems（会自动检测可用的GPU设备）
    _ensure_flag_gems_enabled()

    # 初始化结果列表
    pytest._op_perf_results = []

    # 获取所有算子
    ops = parse_op_list()
    print(f"找到 {len(ops)} 个算子，开始性能测试...\n")

    # 运行所有测试
    for op_name in ops:
        config = get_op_config(op_name)
        print(f"测试算子: {op_name} (类型: {config['type']})...")

        # 跳过需要特殊参数的算子
        if config.get('type') == 'skip':
            print(f"  跳过: {config.get('reason', '需要特殊参数')}")
            continue

        for shape in config.get('shapes', []):
            try:
                test_op_performance(op_name, shape, config)
            except Exception as e:
                print(f"  警告: {op_name} shape={shape} 测试失败: {e}")

    # 打印性能测试报告
    print("\n" + "=" * 100)
    print_summary_report()

    # 输出CSV格式报告
    print_csv_report()
