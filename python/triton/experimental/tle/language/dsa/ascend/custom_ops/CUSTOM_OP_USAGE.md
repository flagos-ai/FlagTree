# Ascend Custom Ops 使用说明

## 目录结构

```text
custom_ops/
├── CUSTOM_OP_USAGE.md              # 本文档
├── __init__.py                     # 对外导出注册算子和公共常量
├── common.py                       # 存放结构体变量和公共函数
├── registry.py                     # Python custom op 注册表
├── build_custom_ops.sh             # 手动重新编译统一 bitcode
├── custom_ops.bc                   # 所有注册算子共用的 bitcode，编译后生成
├── mem_ops/
│   ├── duplicate.cpp               # 将一个变量或立即数复制多次并填充到向量中，暂只支持 tensor 高维切分计算中 mask 逐比特模式
│   ├── gather_gm_to_l1.cpp         # GM → L1/CBUF 按索引行 gather
│   ├── gather_gm_to_ub.cpp         # GM → UB 按索引行 gather
│   └── gather_mask.cpp             # 以内置固定模式对应的二进制或用户自定义输入的 Tensor 数值对应的二进制为 gather mask, 从源操作数中选取元素写入目的操作数中
├── reduction_ops/
│   └── pair_reduce_sum.cpp         # 相邻两个（奇偶）元素求和, 暂只支持 mask 连续模式
└── sort_ops/
    ├── sort32.cpp                  # 排序函数，一次迭代可以完成32个数的排序
    ├── sort_1d_pack.cpp            # sort_1d_pack ABI 与路径分发
    ├── sort_common.h                # 共享 vmrgsort4 / proposal inline 工具
    ├── sort_base.h               # 通用排序路径
    ├── sort_s4096_k129_512.h     # 4096 segment、128 < K <= 512 的 4×1024 small-K 排序路径
    ├── sort_s4096_k1_128_k2048.h # 4096 segment、K <= 128 或 K == 2048 的分层排序路径
    ├── merge_pack_sort.cpp         # proposal 归并与解包
    ├── mrgsort.cpp                 # proposal 归并
    └── unpack_sort.cpp             # proposal 拆包为 value/index
```

`registry.py` 中的所有算子均引用同一个 `custom_ops.bc`。每个算子的 `.cpp` 分别用对应 ccec 架构（`dav-c220-cube` 或 `dav-c220-vec`）编译为自己的 `.bc`，再与 Template bitcode 一起 `llvm-link` 成 `custom_ops.bc`。

## 调用约定

在 Triton kernel 中通过 `tle.dsa.ascend.raw` 调用：

```python
result = tle.dsa.ascend.raw(
    "op_name",
    input0,
    input1,
    out=result_buffer,
)
```

多输出写成：

```python
output0, output1 = tle.dsa.ascend.raw(
    "op_name",
    input0,
    out=[output0, output1],
)
```

普通位置参数对应 custom op inputs，`out=` 对应 outputs。纯输出 buffer 只应在 `out=` 中出现一次，不要同时作为普通参数重复传入。

## 已注册算子

| 算子 | Core / Pipe | 功能 | `out=` 含义 | C++ 实现 |
| --- | --- | --- | --- | --- |
| `duplicate_bitwise_mask` | VECTOR / V | 将一个变量或立即数复制多次并填充到向量中 | 需要填充数据的向量，对应 Ascend C `const LocalTensor<T>& dst` | `mem_ops/ duplicate.cpp` |
| `gather_gm_to_l1` | CUBE / MTE2 | 按索引将 GM 连续张量中的 half/bf16 数据行收集到 L1/CBUF，并完成 ND2NZ 搬运 | L1/CBUF half/bf16 目标张量，对应 C++ `dst` | `mem_ops/gather_gm_to_l1.cpp` |
| `gather_gm_to_ub` | VECTOR / MTE2 | 按索引将 GM 连续张量中的 half/bf16 数据行收集到 UB | UB half/bf16 目标张量，对应 C++ `dst` | `mem_ops/gather_gm_to_ub.cpp` |
| `gather_mask_builtin_pattern` | VECTOR / V | 以内置固定模式对应的二进制对应的二进制为 gather mask, 从源操作数中选取元素写入目的操作数中 | 目的操作数，`out[0]` 对应 Ascend C `const LocalTensor<T>& dst`, `out[1]` 对应 Ascend C `uint64_t& rsvdCnt` | `mem_ops/gather_mask.cpp` |
| `gather_mask_custom_pattern` | VECTOR / V | 以用户自定义输入的 Tensor 数值对应的二进制为 gather mask, 从源操作数中选取元素写入目的操作数中 | 目的操作数，`out[0]`对应 Ascend C `const LocalTensor<T>& dst`, `out[1]`对应 Ascend C `uint64_t& rsvdCnt` | `mem_ops/gather_mask.cpp` |
| `pair_reduce_sum_continuous_mask` | VECTOR / V | 以 mask 连续模式进行相邻两个（奇偶）元素的求和 | 规约操作结果，对应 Ascend C `const LocalTensor<T>& dst` | `reduction_ops/pair_reduce_sum.cpp` |
| `sort32` | VECTOR / V | 排序函数，一次迭代可以完成32个数的排序, 输出 proposal | 排序结果，对应 Ascend C `const LocalTensor<T>& dst` | `sort_ops/sort32.cpp` |
| `sort_1d_pack` | VECTOR / V | 对一维 float 数据排序，输出前 `TOPK` 个紧凑 proposal | UB float proposal 输出，对应 C++ `dst_proposals` | `sort_ops/sort_1d_pack.cpp:10-18` |
| `merge_exhaust_sort4` | VECTOR / V | 对最多四路有序 proposal 执行一次 exhaustion merge | `[dst_proposals, consumed_out]` | `sort_ops/merge_pack_sort.cpp:56-63` |
| `mrgsort` | VECTOR / V | 对最多四路有序 proposal 执行归并排序 | 排序结果，对应 Ascend C `const LocalTensor<T>& dst` | `sort_ops/mrgsort.cpp` |
| `unpack_sort` | VECTOR / V | 将 `[value, encoded_index]` proposal 拆分成 value 和 index | `[dst_value, dst_index]` | `sort_ops/unpack_sort.cpp:20-24` |

## 算子使用方法

### `duplicate_bitwise_mask`

```python
dst = tle.dsa.ascend.raw(
    "duplicate_bitwise_mask",
    scalar_value,
    mask,
    repeat_times,
    dst_block_stride,
    dst_repeat_stride,
    out=dst,
)
```

- `scalar_value`：UB 一维 half/bf16/fp32，tensor 的第一个数据为被复制的源操作数，类型与dst中元素的数据类型保持一致
- `mask`：UB 一维 uint64, tensor 的前两个数据用作 mask，按位控制哪些元素参与计算
- `repeat_times`：表示迭代的次数，每次迭代处理8个 datablock (每个 block 32 Bytes, 共256 Bytes)
- `dst_block_stride`：单次迭代内，目的操作数不同 datablock 间地址步长
- `dst_repeat_stride`：单次迭代内，目的操作数相同 datablock 地址步长
- `dst`：UB 一维 half/bf16/fp32 输出

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_duplicate`）。

### `gather_gm_to_l1`

```python
tile_k = tle.dsa.ascend.raw(
    "gather_gm_to_l1",
    src,
    src_index,
    tile_size,
    D,
    out=tile_k,
)
```

- `src`：GM 二维 half/bf16 源张量，行连续；
- `src_index`：GM 二维 int32 索引张量（形状 `(N, 1)`、stride `(1, 1)`），输出第 i 行的数据取自源张量第 `index[i]` 行（0-based 行号）；索引起始偏移通过 block ptr 的 `offsets` 表达；
- `tile_size`：本次收集的行数；
- `D`：每行元素数；
- `out`：四维 L1/CBUF half/bf16 输出。

相邻索引（`index[i + 1] == index[i] + 1`）会合并为一次两行搬运。

> **重要**：调用本算子的 kernel 必须传编译选项 `enable_legacy_insert_load_store_for_mix_cv=True`（kernel launch kwarg，后端转发为 bishengir-compile 的 `--enable-legacy-insert-load-store-for-mix-cv`，把 `InsertLoadStoreForMixCV` 整个 pass 回退到重构前的老版本）。CANN 9.1.0（bishengir 1.2.0 正式版）重构了 `InsertLoadStoreForMixCV`，重构版对 custom op 的 memscope / coreType 推断仍有 bug：PIPE_MTE2 custom op 的 out 会被规划到 GM workspace（并误插 cbuf→cbuf load），与本算子 `__cbuf__` 的 C++ ABI 冲突；回退到老版本 pass 即可绕开。
>
> **TODO**：用 `enable_legacy_insert_load_store_for_mix_cv=True`，目前用该option规避。等 bishengir 修复重构版 `InsertLoadStoreForMixCV` 对 custom op 的推断 bug 后，可去掉该 kwarg。

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_gather_gm_to_l1`）。

### `gather_gm_to_ub`

```python
tile_v = tle.dsa.ascend.raw(
    "gather_gm_to_ub",
    src,
    src_index,
    tile_size,
    D,
    out=tile_v,
)
```

参数含义与 `gather_gm_to_l1` 相同，区别是结果写入二维 UB half/bf16 张量。输出第一维 stride 不得小于 `D`。

> **重要**：与 `gather_gm_to_l1` 相同，调用本算子的 kernel 必须传 `enable_legacy_insert_load_store_for_mix_cv=True`（PIPE_MTE2 + `__ubuf__` ABI，原因同上）。
>
> **TODO**：去掉条件同 `gather_gm_to_l1`——等 `InsertLoadStoreForMixCV` 重构版对 custom op 的推断 bug 修复后，该 kwarg 可去掉。

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_gather_gm_to_ub`）。

### `gather_mask_builtin_pattern`

```python
[dst, rsvd_cnt] = tle.dsa.ascend.raw(
    "gather_mask_builtin_pattern",
    src0,
    src1_pattern,
    reduce_mode,
    mask,
    src0_block_stride,
    repeat_times,
    src0_repeat_stride,
    src1_repeat_stride,
    out=[dst, rsvd_cnt],
)
```

- `src0`：UB 一维 half/bf16/uint16/int16/uint32/int32/fp32，源操作数
- `src1_pattern`：立即数，取值范围为[1,7]。1: 每个 repeat 取偶数索引元素；2. 每个 repeat 取奇数索引元素，其他各个值对应的 gather 模式参见 https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0071.html
- `reduce_mode`：false, Normal 模式，每次 repeat 操作256 Byts 数据；ture, Counter模式，每次repeat操作 mask 个元素
- `mask`：用于控制每次迭代内参与计算的元素，Normal 模式下建议设为0
- `src0_block_stride`：单次迭代内，src0 不同 datablock 间地址步长
- `repeat_times`：迭代的次数
- `src0_repeat_stride`：src0 迭代间的地址步长
- `src1_repeat_stride`：src1 迭代间的地址步长
- `dst`：UB 一维 half/bf16/uint16/int16/uint32/int32/fp32，目的操作数
- `rsvd_cnt`：dst 中有效元素个数

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_gather_mask`）。

### `gather_mask_custom_pattern`

```python
[dst, rsvd_cnt] = tle.dsa.ascend.raw(
    "gather_mask_custom_pattern",
    src0,
    src1_pattern,
    reduce_mode,
    mask,
    src0_block_stride,
    repeat_times,
    src0_repeat_stride,
    src1_repeat_stride,
    out=[dst, rsvd_cnt],
)
```

- `src0`：UB 一维 half/bf16/uint16/int16/uint32/int32/fp32，源操作数
- `src1_pattern`：UB 一维 uint16/uint32，存储用于 gather 的索引，元素类型的数据长度与 src0 的元素类型的数据长度一致，迭代间间隔由 src1RepeatStride 决定， 迭代内 src1Pattern 连续消耗
- `reduce_mode`：false, Normal 模式，每次 repeat 操作256 Byts 数据；ture, Counter模式，每次repeat操作 mask 个元素
- `mask`：用于控制每次迭代内参与计算的元素，Normal 模式下建议设为0
- `src0_block_stride`：单次迭代内，src0 不同 datablock 间地址步长
- `repeat_times`：迭代的次数
- `src0_repeat_stride`：src0 迭代间的地址步长
- `src1_repeat_stride`：src1 迭代间的地址步长
- `dst`：UB 一维 half/bf16/uint16/int16/uint32/int32/fp32，目的操作数
- `rsvd_cnt`：dst 中有效元素个数

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_gather_mask`）。

### `pair_reduce_sum_continuous_mask`

```python
dst = tle.dsa.ascend.raw(
    "pair_reduce_sum_continuous_mask",
    src,
    repeat_times,
    mask,
    dst_rep_stride,
    src_blk_stride,
    src_rep_stride,
    out=dst,
)
```

- `src`：UB 一维 half/fp32，源操作数
- `repeat_times`：迭代次数，取值范围为[0,255]
- `mask`：表示前面连续的多少个元素参与计算
- `dst_rep_stride`：目的操作数相邻迭代间的地址步长
- `src_blk_stride`：单次迭代内 datablock 的地址步长
- `src_rep_stride`：源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的 datablock 数目
- `dst`：UB 一维 half/fp32，目的操作数

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_pair_reduce_sum`）。

### `sort32`

```python
dst = tle.dsa.ascend.raw(
    "sort32",
    src0,
    src1,
    repeat_times,
    out=dst,
)
```

- `src0`：UB 一维 fp32，源操作数，存 score
- `src1`：UB 一维 uint32，源操作数，存 index
- `repeat_times`：重复迭代次数，取值范围为[0,255]
- `dst`：UB 一维 fp32，目的操作数，存 proposal 形式的输出

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_sort32`）。

### `sort_1d_pack`

```python
proposals = tle.dsa.ascend.raw(
    "sort_1d_pack",
    src,
    tmp_buf,
    descending,
    TOPK,
    index_offset,
    sort_impl,
    out=proposals,
)
```

proposal 使用两个 float 槽位紧凑存储：

```text
[value0, encoded_index0, value1, encoded_index1, ...]
```

`out` 至少需要容纳 `2 * TOPK` 个 float。`tmp_buf` 是 UB workspace，大小应与所选排序路径匹配。

#### 排序路径

| 路径 | 值 | 适用情况 | 实现 |
| --- | ---: | --- | --- |
| `SORT_IMPL_BASE` | 0 | 通用 fallback；非 4096 segment，或不适合特化路径的场景 | `sort_base.h` |
| `SORT_IMPL_S4096_K129_512` | 1 | 固定 4096 输入、较小 K；当前示例用于 `128 < K <= 512` | `sort_s4096_k129_512.h` |
| `SORT_IMPL_S4096_K1_128_K2048` | 2 | 固定 4096 输入；很小 K 可 early-stop，K == 2048 使用固定树归并 | `sort_s4096_k1_128_k2048.h` |

三条路径由调用方通过 `sort_impl` 选择，C++ 只执行 switch 分发；未知值回退到 BASE，见 `sort_ops/sort_1d_pack.cpp:19-36`。

推荐的选择策略（`seg_len` 为每段输入长度，`K` 为本段需要保留的 proposal 数，即 `min(TOPK, seg_len)`）为：

```text
seg_len == 4096 且 0 < K <= 128  → S4096_K1_128_K2048
seg_len == 4096 且 128 < K <= 512 → S4096_K129_512
seg_len == 4096 且 K == 2048      → S4096_K1_128_K2048
其他情况                           → BASE
```

单算子正确性测试见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_sort_1d_pack`，覆盖三条路径）。

三条路径简述：

- **BASE**：生成 index，通过 `vbitsort` 形成初始 proposal，再用 `vmrgsort4` 做通用多级归并；
- **S4096_K129_512**：把 4096 个输入拆成四个 1024 元素 chunk，各自排序后进行四路 exhaustion merge；
- **S4096_K1_128_K2048**：按 `32 → 128 → 512 → 2048` proposal 的层级归并，小 K 可在中间层提前停止，K == 2048 走固定树。

### `merge_exhaust_sort4`

```python
out_buf, consumed = tle.dsa.ascend.raw(
    "merge_exhaust_sort4",
    src_proposals,
    ways,
    off0, off1, off2, off3,
    len0, len1, len2, len3,
    out=[out_buf, consumed],
)
```

- `off0..off3`：每一路的起始偏移，单位为 proposal；
- `len0..len3`：每一路的 proposal 数量，`0` 表示该路无效；
- `out[0]`：归并后可安全确定的有序 proposal 前缀；
- `out[1]`：至少四个 int32，记录原始四路本次消耗的 proposal 数量。

该算子只执行一次归并。多轮加载、cursor 推进和完整归并由调用方负责。

示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_merge_exhaust_sort4`）。

### `mrgsort`

```python
dst = tle.dsa.ascend.raw(
    "mrgsort",
    src_proposals,
    off0,
    off1,
    off2,
    off3,
    len0,
    len1,
    len2,
    len3,
    if_exhausted_suspension,
    valid_bit,
    repeat_times,
    out=dst,
)
```

- `src_proposals`：UB 一维 fp32，proposal 形式的源操作数，src_proposals[off0..off3]即各路输入，通常是sort32的输出。
- `off0..off3`：各路输入在 src_proposals 上的偏移
- `len0..len3`：各路输入的前面多少个元素参与归并排序
- `if_exhausted_suspension`：是否在任意一路输入的数据耗尽后提前退出排序
- `valid_bit`：有效队列个数，只能是3、7、15。3：前两路输入有效，7：前三路输入有效，15：四路输入全部有效
- `repeat_times`；迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。参数生效条件参见https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0232.html
- `dst`：UB 一维 fp32，目的操作数，存 proposal 形式的输出

完整示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_mrgsort`）。

### `unpack_sort`

```python
values, indices = tle.dsa.ascend.raw(
    "unpack_sort",
    src_proposals,
    topk,
    out=[values, indices],
)
```

输出顺序固定：

```text
out[0] = UB float dst_value
out[1] = UB int32 dst_index
```

`src_proposals` 的有效 view 应覆盖 `2 * topk` 个 float 槽位。示例见 `python/tutorials/tle/custom/test_custom_ops.py`（`test_unpack_sort`）。


## 编译方法

### 默认自动编译

正常构建工程时，CMake 会直接调用 `ccec` / `llvm-link` 生成或更新 `custom_ops.bc`。规则位于 `third_party/tle/dsa/dialect/lib/CMakeLists.txt`：每个算子的 `.cpp` 按各自 aicore 架构编译为独立 `.bc`（中间产物在构建目录 `ascend_custom_ops/` 下），4 个 Template 源码编译后一起 `llvm-link` 成 `custom_ops.bc`。

```bash
rm -rf build
FLAGTREE_BACKEND=ascend MAX_JOBS=32 \
  python3 -m pip install -e . --no-build-isolation -v
```

当 `custom_ops.bc` 不存在，或某个算子声明的 C++ 源码依赖发生变化时，CMake 只重新编译受影响的算子并重新 link。

### 修改 C++ 实现后手动编译

如果只修改了 custom op 的 C++ 实现，并且 Python 注册和 C++ ABI 没有变化，可以直接运行：

```bash
cd /root/xcs_flagtree/python/triton/experimental/tle/language/dsa/ascend/custom_ops
./build_custom_ops.sh
```
