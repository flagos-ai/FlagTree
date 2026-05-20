# Triton GCU Typed Passes API

## 概述

这个模块提供了类型安全的 GCU MLIR passes Python API，替代了之前基于字符串的实现方式。

## 优势

1. **编译时检查**：函数参数类型明确，IDE 可以提供自动补全和类型检查
2. **避免拼写错误**：不再使用字符串形式的 pass 名称，减少运行时错误
3. **更好的可维护性**：API 变更会在编译/导入时立即发现
4. **文档化**：每个函数都有明确的参数说明和类型提示

## 使用示例

### 之前（基于字符串）

```python
passes = [
    f'-gcu-convert-triton-to-tritongpu=num-warps={num_warps} threads-per-warp={warp_size} num-ctas={num_ctas} target=gcu:{arch}',
    '-tritongpu-remove-layout-conversions',
    '-convert-triton-to-gcu=vector-length=' + str(vector_length),
]
return toolkit.triton_gcu_opt(mod, *passes, arch=arch)
```

**问题**：
- 容易拼错 pass 名称或选项
- 没有类型检查
- 错误只能在运行时发现

### 现在（类型化 API）

```python
from triton import passes

# 创建 pipeline
PipelineClass = toolkit.get_gcu_pipeline_class("gcu300")
pm = PipelineClass()

# 添加 passes（带类型检查）
passes.gcu300.add_gcu_convert_triton_to_tritongpu(
    pm, num_warps=8, threads_per_warp=32, num_ctas=1, target='gcu:gcu300')
passes.gcu300.add_tritongpu_remove_layout_conversions(pm)
passes.gcu300.add_convert_triton_to_gcu(pm, vector_length=128)

# 运行 pipeline
result = pm.run(input_mlir)
```

**优势**：
- IDE 自动补全函数名和参数
- 参数类型错误会立即发现
- 更清晰的代码结构

## API 结构

### GCU300 Passes (`passes.gcu300`)

- `add_gcu64_type_verifier(pm)` - 64位类型验证
- `add_gcu_convert_triton_to_tritongpu(pm, num_warps, threads_per_warp, num_ctas, target)` - Triton 到 TritonGPU 转换
- `add_tritongpu_remove_layout_conversions(pm)` - 移除不必要的布局转换
- `add_triton_gpu_to_triton_gcu(pm)` - TritonGPU 到 TritonGCU 转换
- `add_convert_tensor_pointer(pm)` - Tensor pointer 转换
- `add_triton_gcu_dot_layout_optimize(pm)` - Dot 操作布局优化
- `add_convert_triton_load_store_to_gcu_dma(pm, support_stride0=False)` - Load/Store 到 DMA 转换
- `add_gcu_combine_ops(pm)` - 操作合并
- `add_gcu_triton_fusion(pm)` - Triton 操作融合
- `add_triton_gcu_data_layout_optimize(pm)` - 数据布局优化
- `add_triton_gcu_pingpong(pm, num_stages)` - Ping-pong 缓冲优化
- `add_flatten_triton_func(pm)` - 函数扁平化
- `add_convert_triton_to_gcu(pm, vector_length=128)` - Triton 到 GCU 转换

### GCU400 Passes (`passes.gcu400`)

- `add_gcu64_type_verifier(pm)` - 64位类型验证
- `add_tle_to_triton_gcu(pm)` - TLE 到 TritonGCU 转换
- `add_triton_gpu_to_triton_gcu(pm)` - TritonGPU 到 TritonGCU 转换
- `add_convert_tensor_pointer(pm)` - Tensor pointer 转换
- `add_convert_triton_load_store_to_gcu_dma(pm, support_stride0=False)` - Load/Store 到 DMA 转换
- `add_tritongcu_accelerate_matmul(pm)` - Matmul 加速
- `add_triton_wgdot_to_gcu(pm)` - Workgroup dot 转换
- `add_tritongpu_remove_layout_conversions(pm)` - 移除不必要的布局转换
- `add_triton_gcu_data_layout_optimize(pm)` - 数据布局优化
- `add_gcu_combine_ops(pm)` - 操作合并
- `add_gcu_triton_fusion(pm, arch)` - Triton 操作融合
- `add_flatten_triton_func(pm)` - 函数扁平化
- `add_annotate_dot_acc_reuse(pm)` - Dot accumulator 重用标注
- `add_triton_gcu_local_mem_optimize(pm)` - 本地内存优化
- `add_convert_triton_to_gcu(pm)` - Triton 到 GCU 转换

### MLIR Built-in Passes (`passes.mlir`)

MLIR 框架通用passes，用于调试和性能分析：

- `add_print_ir_after_all(pm)` - 在每个pass后打印IR（调试用）
- `add_disable_threading(pm)` - 禁用多线程pass执行（调试用）
- `add_print_ir_module_scope(pm)` - 打印完整模块IR而非单个操作
- `add_timing(pm)` - 启用pass执行时间统计
- `add_timing_display(pm, display_mode)` - 配置时间统计显示格式（'list' 或 'tree'）

## 实现细节

### 架构

```
triton_gcu300_core.so (C++ 核心库)
    ↓ C ABI (triton_gcu300_core.h)
_triton_gcu300.so (pybind11 Python 绑定)
    ↓ Pipeline 类
passes.gcu300 (Python 类型化 API)
    ↓ add_xxx 函数
compiler.py (使用类型化 API)
```

### 关键文件

- `triton_gcu/triton_gcu300/lib/triton_gcu300_core.{h,cpp}` - C++ 核心实现
- `triton_gcu/triton_gcu300/python/triton_gcu300_module.cpp` - pybind11 绑定
- `python/triton/passes/gcu300.py` - GCU300 类型化 API
- `python/triton/passes/gcu400.py` - GCU400 类型化 API
- `python/triton/compiler.py` - 使用类型化 API 的编译器

## 迁移指南

如果你的代码使用了旧的字符串 API：

1. 导入 passes 模块：
   ```python
   from triton import passes
   ```

2. 创建 Pipeline：
   ```python
   PipelineClass = toolkit.get_gcu_pipeline_class(arch)
   pm = PipelineClass()
   ```

3. 替换字符串 pass 为函数调用：
   ```python
   # 旧: passes.append('-canonicalize')
   # 新: pm.add_pass('canonicalize')

   # 旧: passes.append('-convert-triton-to-gcu=vector-length=128')
   # 新: passes.gcu300.add_convert_triton_to_gcu(pm, vector_length=128)
   ```

4. 运行 Pipeline：
   ```python
   result = pm.run(input_mlir)
   ```

## 兼容性

- 旧的 `toolkit.triton_gcu_opt()` 接口仍然保留，向后兼容
- `run_opt()` legacy 接口继续可用
- 新代码推荐使用类型化 Pipeline API
