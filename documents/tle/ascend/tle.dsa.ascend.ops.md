# TLE DSA Ascend OP 总表

本文档汇总 `/documents/tle/ascend` 目录下的 Ascend 后端相关 TLE DSA OP 文档。

## 说明

本目录下文档引用自较高版本 `triton-ascend` 社区文档和用例。原社区文档中通常使用如下命名空间：

```python
import triton.language.extra.cann.extension as al
```

在 TLE DSA 文档中，对应接口位于 `tle.dsa.ascend` 命名空间下。除命名空间不同外，接口使用方式保持一致。

例如：

```python
# triton-ascend 社区写法
al.sync_block_all("all", 0)

# TLE DSA 写法
tle.dsa.ascend.sync_block_all("all", 0)
```

地址空间同理：

```python
# triton-ascend 社区写法
al.ascend_address_space.UB

# TLE DSA 写法
tle.dsa.ascend.UB
```

## OP 列表

| OP | 简短描述 | 详细文档 |
|----|----------|----------|
| `tle.dsa.ascend.UB/L1/L0A/L0B/L0C` | Ascend 后端地址空间枚举，用于指定 DSA buffer 所在的硬件内存区域。 | [tle.dsa.ascend.ascend_address_space.md](tle.dsa.ascend.ascend_address_space.md) |
| `tle.dsa.ascend.sub_vec_id` | 获取当前 Vector 子核 ID，用于 AIC/AIV 混合场景下按 sub vector ID 划分数据。 | [tle.dsa.ascend.sub_vec_id.md](tle.dsa.ascend.sub_vec_id.md) |
| `tle.dsa.ascend.sync_block_all` | 插入全局核间同步，支持 cube、vector、all 和 sub vector 等同步模式。 | [tle.dsa.ascend.sync_block_all.md](tle.dsa.ascend.sync_block_all.md) |
| `tle.dsa.ascend.sync_block_set` | 分离式核间同步中的 set 操作，配合 `sync_block_wait` 使用。 | [tle.dsa.ascend.sync_block_set.md](tle.dsa.ascend.sync_block_set.md) |
| `tle.dsa.ascend.sync_block_wait` | 分离式核间同步中的 wait 操作，等待指定同步事件后继续执行。 | [tle.dsa.ascend.sync_block_wait.md](tle.dsa.ascend.sync_block_wait.md) |
| `tle.dsa.ascend.compile_hint` | 向编译器传递优化提示，指导代码生成和性能调优，支持 `bitwise_mask` 等 Hint 类型。 | [tle.dsa.ascend.compile_hint.md](tle.dsa.ascend.compile_hint.md) | |
