# TLE DSA OP 总表

本文档汇总 `/documents/tle` 目录下的 TLE DSA OP 文档。

| OP | 简短描述 | 详细文档 |
|----|----------|----------|
| `tle.dsa.extract_element` | 从有 rank 的 tensor 中按给定索引提取单个标量元素。 | [tle.dsa.extract_element_cn.md](tle.dsa.extract_element_cn.md) |
| `tle.dsa.extract_slice` | 从输入 tensor 中按指定偏移、大小和步长提取一个子 tensor。 | [tle.dsa.extract_slice_cn.md](tle.dsa.extract_slice_cn.md) |
| `tle.dsa.hint` | 通过 `with` 作用域向 TLE DSA builtin 传递编译期提示，目前主要用于 `tle.dsa.copy` 的 `inter_no_alias`。 | [tle.dsa.hint_cn.md](tle.dsa.hint_cn.md) |
| `tle.dsa.insert_slice` | 将子 tensor 插入到目标 tensor 的指定切片区域，并返回插入后的新 tensor。 | [tle.dsa.insert_slice_cn.md](tle.dsa.insert_slice_cn.md) |
| `tle.dsa.parallel` | 表达循环迭代之间无依赖、可按并行语义处理的 JIT 专用循环迭代器。 | [tle.dsa.parallel_cn.md](tle.dsa.parallel_cn.md) |
