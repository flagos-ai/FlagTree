# tle.dsa.ascend.raw 接口文档

# 1. 硬件背景

昇腾 NPU 的 Ascend C 提供了大量 Triton 语言层无法直接表达的硬件能力：Vector 单元的 `Sort32` / `MrgSort` / `GatherMask` / `PairReduceSum` / `Duplicate` 等带 repeat/stride 语义的 SIMD 指令、MTE2 上按离散索引搬运行数据并顺带完成 ND2NZ 转换的能力、以及 SIMT 模板形式的 gather/scatter 访存。这些操作要么在 Triton 里没有对应算子，要么经过通用 lowering 后无法达到手写 Ascend C 的性能。

`tle.dsa.ascend.raw` 是 TLE 提供的 **通用定制算子（custom op）调用入口**：按名字调用一个已注册的定制算子，把参数直接映射为 `hivm.hir.custom` 的操作数，由 bishengir-compile 链接到对应的设备侧实现（内置模板或 bitcode 中的 C++ 函数）。它是 `triton.language.extra.cann.extension.custom`（社区 `al.custom`）的别名：

```python
# python/triton/experimental/tle/language/dsa/ascend/core.py
from triton.language.extra.cann.extension import compile_hint, custom as raw, multibuffer
```

通过 `raw` 可调用的算子分为两类：

| 类别 | 名字特征 | 设备侧实现来源 | 数量 |
|------|----------|----------------|------|
| 编译器内置模板算子 | 以 `__builtin_` 开头 | bishengir-compile 自带的 SIMT 模板库，无需 `symbol` / `bitcode` | 4 |
| TLE 随包 bitcode 算子 | 普通名字 | `custom_ops/custom_ops.bc`，由 `registry.py` 提供 `symbol` | 13 |

# 2. 接口说明

```python
def raw(name: str, *args, out=None, **kwargs):
```

## 2.1 入参

| 参数名 | 类型 | 必需 | 说明 |
|-------|-----|-----|------|
| name  | str / tl.constexpr | 是 | 定制算子名。先在注册表中查找；查不到时必须以 `__builtin_`开头，否则断言失败`Custom Op 'xxx' not registered.`|
| *args / **kwargs | tensor / int / float / bool / tuple | 视算子而定 | 算子入参，按注册类`__init__`的签名顺序绑定后展开为`hivm.hir.custom`的 ins 操作数。tuple/list 会按元素逐个展开（如`src_stride=(4, 1)`展开为两个标量操作数）。 |
| out | tensor 或 tensor 列表 | 视算子而定 | 调用前由使用方分配好的输出 buffer，作为 outs 操作数传入；设备侧直接写入该 buffer，Python 侧按同样的类型返回。多输出算子传 list/tuple。 |


## 2.2 返回值

返回值个数与类型完全由 `out` 决定（`_to_result`）：

| `out` 中 buffer 个数 | 返回值 |
|----------------------|--------|
| 0（不传 `out`） | `None` |
| 1 | 一个 tensor，类型与 `out` 相同 |
| N > 1 | N 元 tuple，顺序与 `out` 列表一致 |

```python
# 单输出
pair = tl.zeros([64], dtype=tl.float32)
pair = tle.dsa.ascend.raw("sort32", value, index, repeat_times, out=pair)

# 双输出
dval = tl.zeros([16], dtype=tl.float32)
didx = tl.zeros([16], dtype=tl.int32)
dval, didx = tle.dsa.ascend.raw("unpack_sort", props, 16, out=[dval, didx])

# 无输出（原地写 GM）
tle.dsa.ascend.raw("__builtin_scatter_store", out_ptr, value, index, 1, 0, (1, ), (2, ), (1, ))
```

## 2.3 name 解析规则

`_get_op_class(name)` 的行为决定了「支持哪些 op」：

1. 名字在 `_custom_op_registry` 中：使用注册类的 `core` / `pipe` / `mode` 和 `__init__` 签名，参数按签名绑定并做全部 assert 校验。
2. 名字不在注册表中但以 `__builtin_` 开头：构造一个**匿名 dummy 类**，默认 `CORE.VECTOR` / `PIPE.PIPE_V` / `MODE.SIMT`，`signature` 为空，**不做任何参数校验**，参数按实际书写顺序（先 `*args` 再 `**kwargs` 的插入顺序）展开为操作数。
3. 其余情况：`AssertionError: Custom Op 'xxx' not registered.`

即 `__builtin_` 前缀是保留前缀，构成一条「透传给编译器内置模板」的通道，因此本仓库注册表里没有的 `__builtin_indirect_load`、`__builtin_histogram` 等名字在前端也能写出来，能否编译取决于 bishengir-compile 版本（见第 5 章）。

## 2.4 生成的 IR 与属性

调用最终由 `builder.create_custom_op` 生成 `hivm.hir.custom`，`operandSegmentSizes` 把操作数切分为 `ins / outs / tmps` 三段。常用属性：

| 属性 | 来源 | 说明 |
|------|------|------|
| `hivm.tcore_type` | `op.core` | `VECTOR` / `CUBE` / `CUBE_OR_VECTOR` / `CUBE_AND_VECTOR` |
| `hivm.pipe` | `op.pipe` | `PIPE_S/V/M/MTE1/MTE2/MTE3/FIX/ALL` |
| `hivm.vf_mode` | `op.mode` | `SIMD` / `SIMT` / `MIX` |
| `symbol` | `op.symbol` | 设备侧实现函数名，非 `__builtin_` 算子必填 |
| `bitcode` | `op.bitcode` | bitcode 绝对路径，非 `__builtin_` 算子必填，且文件必须存在 |
| `extra_buffers_types` / `extra_buffers_sizes` | `op.extra_buffers` | 额外 workspace 声明 |
| `extra_attr` | `op.extra_attr` | 透传给设备侧的附加信息，如 `src_stride_len=2` |
| `indexing_map` / `iterator_types` / `arg_attrs` | 可选 | affine 映射、迭代器类型、按参数的 `align_dim` |

# 3. 支持的 builtin op

## 3.1 总览

当前仓库（`python/triton`）注册表共 17 个算子：

| 算子名 | core | pipe | mode | 实现来源 | 输出个数 |
|--------|------|------|------|----------|----------|
| `__builtin_index_select` | VECTOR | PIPE_V | SIMT | 编译器内置模板 | 1 |
| `__builtin_index_put` | VECTOR | PIPE_V | SIMT | 编译器内置模板 | 0 |
| `__builtin_gather_load` | VECTOR | PIPE_V | SIMT | 编译器内置模板 | 1 |
| `__builtin_scatter_store` | VECTOR | PIPE_V | SIMT | 编译器内置模板 | 0 |
| `duplicate_bitwise_mask` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `gather_gm_to_l1` | CUBE | PIPE_MTE2 | SIMD | `custom_ops.bc` | 1 |
| `gather_gm_to_ub` | VECTOR | PIPE_MTE2 | SIMD | `custom_ops.bc` | 1 |
| `gather_mask_builtin_pattern` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 2 |
| `gather_mask_custom_pattern` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 2 |
| `pair_reduce_sum_continuous_mask` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `sort32` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `sort_1d_pack` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `merge_exhaust_sort4` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 2 |
| `mrgsort` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `unpack_sort` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 2 |
| `compare_scalar` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |
| `cast_int4_to_fp16` | VECTOR | PIPE_V | SIMD | `custom_ops.bc` | 1 |

可在运行时自行确认当前环境实际注册了哪些算子：

```python
import triton.experimental.tle as tle
from triton.language.extra.cann.extension.custom_op import _custom_op_registry
for name, op in sorted(_custom_op_registry.items()):
    print(name, op.core.name, op.pipe.name, op.mode.name)
```

注意：已安装的 triton 包与仓库源码树可能不同步（例如较早的安装包只有 15 个算子，缺 `compare_scalar` 和 `cast_int4_to_fp16`）。调试前端行为时建议显式指定 `PYTHONPATH` 指向仓库 `python` 目录。

## 3.2 `__builtin_*` 编译器内置模板算子

这 4 个算子定义在 `python/triton/language/extra/cann/extension/builtin_custom_ops.py`，全部是 GM 与 UB 之间按索引访存的 SIMT 模板，前端做完整参数校验但不提供 `symbol` / `bitcode`。

### `__builtin_index_select`

沿 `dim` 从 GM 的 2D~5D `src` 中按 UB 上的 `index` 取行（切片），写入 UB 的 `out`。


```python
__builtin_index_select(src, index, dim, bound: tl.int64, end_offset, start_offset, src_stride, other=None, out=None)
```

| 参数 | 说明 |
|------|------|
| `src` | GM 源张量指针 |
| `index` | UB 上的整型索引 tensor，rank ∈ [1, 2] |
| `dim` | gather 维度，需满足 `0 <= dim < src_rank`，其中 `src_rank = len(src_stride)` ∈ [2, 5] |
| `bound` | 索引上界，用于越界检查 |
| `end_offset` | 各维终止偏移，长度须为 `index_rank + len(start_offset) - 1` |
| `start_offset` | `src` 各维起始偏移，长度须与 `src_stride` 相同 |
| `src_stride` | `src` 各维 stride，其长度即 `src_rank` |
| `other` | 越界时的填充值（可选） |
| `out` | 必填，UB 输出，`out.dtype` 须等于 `src.dtype.element_ty` |

`end_offset` / `start_offset` / `src_stride` 的元素类型被自动改写为 `index.dtype`，并额外生成 `extra_attr = "src_stride_len=<N>"`。

### `__builtin_index_put`

把 UB 的 `value` 按 UB 的 `index` 沿 `dim` 写回 GM 的 `dst`，无返回值。

```python
__builtin_index_put(dst, index, value, dim, bound: tl.int64, dst_shape, dst_offset, dst_stride)
```

约束：`value` rank ∈ [2, 5]；`0 <= dim < value_rank - 1`；`dst_shape` / `dst_offset` / `dst_stride` 三者长度相同且元素均为编译期 int（元素类型改写为 `index.dtype`）。

### `__builtin_gather_load`

按 UB 的 `index`（rank ∈ [1, 5]）从 GM 逐元素 gather 到 UB，输出 shape 与 `index` 相同。

```python
__builtin_gather_load(src, index, bound: tl.int64, dim, src_stride: tl.int64, index_shape, offsets, out=None)
```

约束：`0 <= dim < index_rank`；`src_stride` / `index_shape` / `offsets` 长度均等于 `index_rank` 且元素为编译期 int；`out` 必填且 `out.shape == index.shape`。

### `__builtin_scatter_store`

`__builtin_gather_load` 的反向：把 UB 的 `value` 按 UB 的 `index` 写入 GM，无返回值。

```python
__builtin_scatter_store(dst, value, index, bound: tl.int64, dim, dst_stride: tl.int64, index_shape, offsets)
```

约束与 `__builtin_gather_load` 对称：`index` rank ∈ [1, 5]，`0 <= dim < index_rank`，`dst_stride` / `index_shape` / `offsets` 长度均为 `index_rank`。

## 3.3 TLE 随包 bitcode 算子

这 13 个算子在 `python/triton/experimental/tle/language/dsa/ascend/custom_ops/registry.py` 注册，设备侧 C++ 实现统一编译进 `custom_ops.bc`，每个算子的 `symbol` 指向其中的函数。参数语义、逐算子的详细数值约束和更多示例见同目录下的 `CUSTOM_OP_USAGE.md`，此处仅给出签名与关键约束。

| 算子名 | 签名（省略 `out`） | symbol | 关键约束 |
|--------|--------------------|--------|----------|
| `duplicate_bitwise_mask` | `(scalar_value, mask, repeat_times, dst_block_stride, dst_repeat_stride)` | `custom_duplicate_bitwise_mask_{half,bf16,float}` | 对应 Ascend C `Duplicate`，仅支持 mask 逐比特模式 |
| `gather_gm_to_l1` | `(src, index, tile_size, D)` | `custom_gather_gm_to_l1_<dtype>` | fp16/bf16；`src` 行连续 2D GM，`index` 为 `(N, 1)` int32；`out` 为 4D L1/CBUF，含 ND2NZ 转换 |
| `gather_gm_to_ub` | `(src, index, tile_size, D)` | `custom_gather_gm_to_ub_<dtype>` | fp16/bf16；`out` 为 2D UB，首维 stride ≥ `D` |
| `gather_mask_builtin_pattern` | `(src0, src1_pattern, reduce_mode, mask, src0_block_stride, repeat_times, src0_repeat_stride, src1_repeat_stride)` | `custom_gather_mask_builtin_pattern_<dtype>` | 对应 `GatherMask`，内置固定 pattern；`out=[dst, rsvd_cnt]` |
| `gather_mask_custom_pattern` | 同上（`src1_pattern` 为用户 tensor） | `custom_gather_mask_custom_pattern_<dtype>` | 用户自定义 pattern；`out=[dst, rsvd_cnt]` |
| `pair_reduce_sum_continuous_mask` | `(src, repeat_times, mask, dst_rep_stride, src_blk_stride, src_rep_stride)` | `custom_pair_reduce_sum_continuous_mask_{half,float}` | 对应 `PairReduceSum`，仅 mask 连续模式；`out` dtype 须与 `src` 一致 |
| `sort32` | `(src0, src1, repeat_times)` | `custom_sort32` | `src0` fp32 值、`src1` uint32 索引、`out` fp32；每次迭代排 32 个数，输出 `[value, index]` 对 |
| `sort_1d_pack` | `(src, tmp_buf, descending, TOPK, index_offset, sort_impl)` | `custom_sort_1d_pack_float` | 全 fp32；`out` 至少 `2 * TOPK` 个 float slot；`sort_impl` 见下表 |
| `merge_exhaust_sort4` | `(src_proposals, ways, off0..off3, len0..len3)` | `custom_merge_exhaust_sort4_float` | 偏移/长度以 proposal 计；`out=[dst_proposals(fp32), consumed_out(int32≥4)]` |
| `mrgsort` | `(src_proposals, off0..off3, len0..len3, if_exhausted_suspension, valid_bit, repeat_times)` | `custom_mrgsort` | 对应 `MrgSort`，最多 4 路归并；全 fp32 |
| `unpack_sort` | `(src_proposals, topk)` | `custom_unpack_sort_float` | 把紧凑 proposal 拆为值和索引；`out=[dst_value(fp32), dst_index(int32)]` |
| `compare_scalar` | `(src, scalar, cmpMode, count)` | `custom_compare_scalar_{half,float}[_mask32]` | `cmpMode` 为 CANN CMPMODE：0=LT 1=GT 2=EQ 3=LE 4=GE 5=NE；`count` 须等于源长度；`out` 为 `uint16[N/16]` 或（仅 fp32）`uint32[N/32]`；size 为 2 的幂，fp32 ∈ [256, 4096]，fp16 ∈ [256, 32768] |
| `cast_int4_to_fp16` | `(src, roundMode, count)` | `custom_cast_int4_to_fp16` | `src` 为 1D uint8，字节数为 2 的幂 ∈ [32, 8192]；`roundMode` 只支持 0（`CAST_NONE`）；`out` 为 `float16[2*N]`，`count == 2*N`；低 nibble 先输出 |

`sort_1d_pack` 的 `sort_impl` 取值（`tle.dsa.ascend.custom_ops` 导出的 constexpr）：

| 常量 | 值 | 适用场景 |
|------|-----|----------|
| `SORT_IMPL_BASE` | 0 | 通用 fallback；非 4096 segment 或不适合特化路径 |
| `SORT_IMPL_S4096_K129_512` | 1 | 固定 4096 输入、`128 < K <= 512` |
| `SORT_IMPL_S4096_K1_128_K2048` | 2 | 固定 4096 输入；`K <= 128` 可提前结束，`K == 2048` 走固定树归并 |

其中 `duplicate_bitwise_mask`、两个 `gather_mask_*`、`pair_reduce_sum_continuous_mask`、`sort32`、`sort_1d_pack`、`merge_exhaust_sort4`、`mrgsort`、`unpack_sort` 均声明了 `extra_buffers = [(tl.float16, 0)]`，会在 IR 上生成 `extra_buffers_types = [f16]` / `extra_buffers_sizes = [0]`。

# 4. 用例示例

## 4.1 单输出：sort32

```python
@triton.jit
def sort32_kernel(score_ptr, index_ptr, out_ptr):
  offs = tl.arange(0, 32)
  score = tl.load(score_ptr + offs)
  index = tl.load(index_ptr + offs).to(tl.uint32)
  dst = tl.full((64, ), 0.0, tl.float32) # 32 个 [value, index] 对
  dst = tle.dsa.ascend.raw("sort32", score, index, 1, out=dst)
  tl.store(out_ptr + tl.arange(0, 64), dst)
```

生成的 TTIR（已验证）：

```mlir
%dst_6 = hivm.hir.custom {
    arg_attrs = [{}, {}, {}, {}],
    bitcode = ".../custom_ops/custom_ops.bc",
    extra_buffers_sizes = [0], extra_buffers_types = [f16],
    hivm.pipe = #hivm.pipe<PIPE_V>,
    hivm.tcore_type = #hivm.tcore_type<VECTOR>,
    hivm.vf_mode = #hivm.vf_mode<SIMD>,
    symbol = "custom_sort32"
  } "sort32" ins(%score_1, %index_3, %dst_5 : tensor<32xf32>, tensor<32xi32>, i32)
             outs(%dst_4 : tensor<64xf32>) -> tensor<64xf32>
```

可以看到：算子名作为 `hivm.hir.custom` 的 `name`，3 个入参进 `ins`，预分配的 `out` 进 `outs`，返回类型与 `out` 一致。该 kernel 在 `Ascend910_95` 与 `Ascend910B` 两个 target 下均能编译通过。

## 4.2 多输出：unpack_sort

```python
@triton.jit
def unpack_kernel(prop_ptr, val_ptr, idx_ptr, TOPK: tl.constexpr):
  prop = tl.load(prop_ptr + tl.arange(0, 2 * TOPK))
  dval = tl.zeros([TOPK], dtype=tl.float32)
  didx = tl.zeros([TOPK], dtype=tl.int32)
  dval, didx = tle.dsa.ascend.raw("unpack_sort", prop, TOPK, out=[dval, didx])
  tl.store(val_ptr + tl.arange(0, TOPK), dval)
  tl.store(idx_ptr + tl.arange(0, TOPK), didx)
```

生成的 TTIR（已验证，`TOPK=16`）：

```mlir
%0:2 = hivm.hir.custom {
    ..., symbol = "custom_unpack_sort_float"
  } "unpack_sort" ins(%prop_2, %c16_i32 : tensor<32xf32>, i32)
                  outs(%dval_3, %didx_4 : tensor<16xf32>, tensor<16xi32>)
                  -> (tensor<16xf32>, tensor<16xi32>)
```

两个 `out` buffer 同时出现在 `outs` 中，Python 侧按同样顺序解包为 tuple。

## 4.3 `__builtin_*` 算子写法

以 `__builtin_gather_load` 为例（社区用例 `third_party/ascend/unittest/custom_op/test_gather_load.py`）：

```python
@triton.jit
def gather_load_kernel(src_ptr, index_ptr, out_ptr):
  cols = tl.arange(0, 2)[None, :]
  rows = tl.arange(0, 2)[:, None]
  mask = (rows < 2) & (cols < 2)
  index = tl.load(index_ptr + rows * 2 + cols, mask)

  dst = tl.full(index.shape, 0, tl.float32)
  gathered = tle.dsa.ascend.raw("__builtin_gather_load", src_ptr, index,
  bound=4, dim=0, src_stride=(2, 1),
  index_shape=(2, 2), offsets=(0, 0), out=dst)
  tl.store(out_ptr + rows * 2 + cols, gathered, mask)

# src = [[1., 2.], [3., 4.], [5., 6.], [7., 8.]], index = [[0, 1], [2, 3]]
# 期望输出：[[1., 4.], [5., 8.]]
```

`__builtin_index_select` 的用法类似（`test_index_select.py`）：`src` 为 4x4 fp32、`index = [2, 0]`、`end_offset=(2, 2)`、`start_offset=(0, 0)`、`src_stride=(4, 1)`，等价于 `torch.index_select(src, 0, index)[:, :2]`，期望输出 `[[30., 31.], [10., 11.]]`。

`__builtin_index_put` / `__builtin_scatter_store` 无返回值，直接丢弃返回：

```python
tle.dsa.ascend.raw("__builtin_index_put", x_ptr, index, value, dim=0, bound=12,
dst_shape=(1, 2, 3), dst_offset=(4, 5, 6), dst_stride=(8, 4, 1))
tle.dsa.ascend.raw("__builtin_scatter_store", out_ptr, value, index, 1, 0, (1, ), (2, ), (1, ))
```

## 4.4 MTE2 gather 算子需要额外的 launch 选项

`gather_gm_to_l1` / `gather_gm_to_ub` 使用 PIPE_MTE2 + `__cbuf__` / `__ubuf__` 的 C++ ABI，必须在 kernel launch 时传 `enable_legacy_insert_load_store_for_mix_cv=True`：

```python
gather_l1_kernel[grid](src, index, query, output,
                       TILE_SIZE=TILE_SIZE, D=D, DTYPE=tl.float16,
                       enable_legacy_insert_load_store_for_mix_cv=True)
```

完整可运行示例见 `python/tutorials/tle/custom/test_custom_ops.py`、`test_compare_scalar.py`、`test_cast_ops.py`。

# 5. 约束说明

## 5.1 `__builtin_*` 算子当前在本仓库无法通过 HIVM 校验

前端可以正常生成 IR，但 `hivm.hir.custom` 的 verifier 会报错。以 `__builtin_scatter_store` 为例，生成的 IR 形态正确：

```mlir
"hivm.hir.custom"(%out_ptr, %x_1, %idx_4, %0, %1, %2, %3, %4)
  <{name = "__builtin_scatter_store", operandSegmentSizes = array<i32: 8, 0, 0>}>
  {hivm.pipe = #hivm.pipe<PIPE_V>,
   hivm.tcore_type = #hivm.tcore_type<VECTOR>,
   hivm.vf_mode = #hivm.vf_mode<SIMT>}
  : (!tt.ptr<f32>, tensor<32xf32>, tensor<32xi32>, i64, i32, i64, i32, i32) -> ()
```

随后失败于：

```bash
error: 'hivm.hir.custom' op Missing implementation function name
RuntimeError: error encountered during parsing
```

原因在于 `CustomOp::verify()` 先判断 `isBuiltin()`，只有内置算子才跳过 `symbol` 检查；而随仓库 vendored 的 AscendNPU-IR 中该表为空：

```cpp
// third_party/ascend/AscendNPU-IR/bishengir/lib/Dialect/HIVM/IR/HIVMOps.cpp:467
bool CustomOp::isBuiltin() { return kBuiltins.contains(getName()); }
// :821
const DenseMap<StringRef, CustomOp::BuiltinInfo> CustomOp::kBuiltins{};
```

`kBuiltins` 为空 ⇒ `isBuiltin()` 恒为 false ⇒ 走普通路径，最后命中 `if (getSymbol().empty()) return emitOpError() << "Missing implementation function name";`。四个 `__builtin_*` 算子（以及 `__builtin_indirect_load` 等未注册名字）表现一致。

**结论**：目前这 4 个 `__builtin_*` 算子只有前端与 IR 生成路径可用，端到端编译需要一个 `kBuiltins` 非空、且包含对应算子的 bishengir 版本。

## 5.2 内置算子集合随 bishengir-compile 版本变化

不同工具链二进制中实际存在的 `__builtin_` 模板差别很大（通过符号扫描确认）：

| bishengir-compile 版本 | 包含的内置算子 |
|------------------------|----------------|
| 1.1.0（cann-9.0.0，2026-04-30） | `__builtin_gather_load` |
| 1.1.0（cann-9.1.0-beta.3，2026-06-25） | 无 |
| 1.2.0（cann-9.1.0，2026-07-29） | 无 |
| 1.2.0（AscendNPU-IR 上游，2026-09-17） | `__builtin_gather_load`、`__builtin_index_select`、`__builtin_indirect_atomic`、`__builtin_histogram`、`__builtin_proton_get_sys_cnt`、`__builtin_proton_circular_store` |

`__builtin_index_put` 与 `__builtin_scatter_store` 在以上任一二进制中都没有出现，说明它们的 Python 侧注册目前超前于编译器实现。使用 `__builtin_*` 前请先确认当前 CANN / bishengir 版本是否提供对应模板。

## 5.3 非 `__builtin_` 算子必须有可用的 bitcode

`_make_attrs` 对非内置算子强制要求 `symbol` 和 `bitcode` 两个字段，且 `_add_bitcode_attr` 会断言文件真实存在：

```
AssertionError: Provided bitcode (bitcode) not exist
```

`registry.py` 中所有算子共用 `custom_ops/custom_ops.bc`。该文件由构建流程生成（`third_party/tle/dsa/dialect/lib/CMakeLists.txt`，或修改 C++ 实现后手动执行 `custom_ops/build_custom_ops.sh`）：各 `.cpp` 按 `dav-c220-vec` / `dav-c220-cube` 用 ccec 编译，再与 Template bitcode 一起 `llvm-link`。若直接以源码树运行（`PYTHONPATH=.../python`）而未构建，会因缺少 `custom_ops.bc` 而报上述断言。

## 5.4 标量参数类型

- Python 裸 `int` 默认降为 **i32**。需要 i64 时用 `al.int64(x)` 包装，或在注册类的 `__init__` 中用类型标注 `bound: tl.int64`，或通过 `self.arg_type['name'] = <dtype>` 动态指定（`__builtin_index_select` 就用 `index.dtype` 覆盖了 `end_offset` / `start_offset` / `src_stride`）。
- Python 裸 `float` 默认降为 f32。
- tuple / list 参数会被逐元素展开为独立操作数，不存在聚合类型。
- 许多算子要求 `count` / `TOPK` / `cmpMode` 等为编译期常量（assert `isinstance(x, int)`），需以 `tl.constexpr` 传入。

## 5.5 其它

- `out` buffer 必须由调用方在 kernel 内先分配（`tl.zeros` / `tl.full` / `tle.dsa.alloc` + `tle.dsa.to_tensor`），`raw` 不会替你创建输出。
- 未注册且不以 `__builtin_` 开头的名字直接断言失败，不存在「按字符串试探」的调用方式。
- 各算子的 UB 对齐、size 为 2 的幂、输入输出不重叠等数值约束由注册类的 assert 在编译期检查，详见 `registry.py` 与 `CUSTOM_OP_USAGE.md`。
- 算子本身不插入流水同步（`compare_scalar` / `cast_int4_to_fp16` 明确说明），跨算子的读写顺序由调用方负责。

# 6. 扩展：注册新的定制算子

若需新增 `raw` 可调用的算子，用 `al.register_custom_op` 装饰一个类，必填 `core` / `pipe` / `mode`，并在 `__init__` 中完成参数校验和 `symbol` / `bitcode` 赋值：


```python
import triton.language.extra.cann.extension as al

@al.register_custom_op
class my_custom_op:
  core = al.CORE.VECTOR
  pipe = al.PIPE.PIPE_V
  mode = al.MODE.SIMD

  def __init__(self, src, count, out=None):
    assert out is not None
    self.arg_type["count"] = tl.int32
    self.symbol = "custom_my_op_float"
    self.bitcode = CUSTOM_OPS_BITCODE
```

`name` 缺省取类名；`core` / `pipe` / `mode` 必须是 `CORE` / `PIPE` / `MODE` 枚举实例；名字重复会在注册时断言失败。设备侧实现需要放入 `custom_ops/` 下对应子目录并重新生成 `custom_ops.bc`。
