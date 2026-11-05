# FlagTree 函数级自动调优器

`flagtree.tune.autotune` 用于调优 Python 函数的参数。被装饰的函数可以提交一个或多个 GPU kernel，也可以包含分布式通信。调优器将候选配置作为关键字参数传入函数，测量候选的执行耗时，缓存最佳配置，并用该配置执行函数、返回实际结果。

本文对应当前仓库中的 [python/flagtree/tune.py](../python/flagtree/tune.py)。使用方式参考 [Triton-distributed 中文调优文档](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/autotuner-cn.md)，接口与行为以 FlagTree 当前实现为准。

支持的主要功能：

- 用字典列表定义候选参数，支持 `key_fn` 和 `prune_fn`。
- 本地调优与指定通信组内的联合调优。
- 内存缓存、JSON 磁盘缓存，以及与任务、配置、硬件和软件环境关联的缓存标识。
- 分布式候选交集、执行参数一致性检查和统一缓存命中决策。

调优器不会自动识别函数是否包含通信。函数涉及多个 rank 协作时，必须显式传入 `autotune_pg`，并由组内所有成员按相同顺序调用。

## 1. 基本用法

下面是完整的 NVIDIA CUDA 向量加法示例。将代码保存为 `.py` 文件执行；磁盘缓存需要读取被装饰函数的代码信息，在交互式终端中定义函数时可能无法启用持久化。

```python
import torch
import triton
import triton.language as tl

from flagtree.tune import autotune, to_hashable


@triton.jit
def add_kernel(X, Y, OUT, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)
    tl.store(OUT + offsets, x + y, mask=mask)


CONFIG_SPACE = [
    {"block_size": 128, "num_warps": 4},
    {"block_size": 256, "num_warps": 4},
    {"block_size": 512, "num_warps": 8},
]


def add_key(x, y, out, **kwargs):
    return to_hashable(x), to_hashable(y), to_hashable(out)


def prune_add(config, x, y, out, **kwargs):
    # 一个简单的裁剪策略：小输入不测试过大的 block。
    # 调用者显式覆盖参数时，也考虑覆盖后的值。
    block_size = kwargs.get("block_size", config["block_size"])
    return block_size <= max(128, triton.next_power_of_2(x.numel()))


@autotune(
    config_space=CONFIG_SPACE,
    key_fn=add_key,
    prune_fn=prune_add,
    cache_version="vector-add-v1",
)
def vector_add(x, y, out, *, block_size, num_warps):
    assert x.is_cuda and x.device == y.device == out.device
    assert x.ndim == y.ndim == out.ndim == 1
    assert x.shape == y.shape == out.shape and x.numel() > 0
    assert x.dtype == y.dtype == out.dtype
    assert x.is_contiguous() and y.is_contiguous() and out.is_contiguous()
    grid = (triton.cdiv(x.numel(), block_size),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=block_size,
                     num_warps=num_warps)
    return out


if __name__ == "__main__":
    torch.cuda.set_device(0)
    x = torch.randn(1 << 20, device="cuda")
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    result = vector_add(x, y, out)
    torch.testing.assert_close(result, x + y)

    # 相同任务再次调用，复用内存中的最佳配置。
    result = vector_add(x, y, out)
    torch.testing.assert_close(result, x + y)
```

装饰器加在 Python 包装函数 `vector_add` 上；`add_kernel` 仍使用 `@triton.jit`。`block_size` 和 `num_warps` 由调优器注入，普通调用不需要提供。

也可以使用完整名称。此时应显式导入子模块：

```python
import flagtree.tune

# 与 from flagtree.tune import autotune 后使用 @autotune 等价。
@flagtree.tune.autotune(CONFIG_SPACE, add_key, prune_fn=prune_add)
def another_add(x, y, out, *, block_size, num_warps):
    grid = (triton.cdiv(x.numel(), block_size),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=block_size,
                     num_warps=num_warps)
    return out
```

## 2. 装饰器参数

```python
def autotune(
    config_space,
    key_fn,
    prune_fn=None,
    *,
    cache_dir=None,
    cache_version: str | None = None,
):
    ...
```

| 参数 | 说明 |
| --- | --- |
| `config_space` | 非空的候选配置列表；每个候选必须是字典，键对应被装饰函数的关键字参数。 |
| `key_fn` | 根据调用参数生成可哈希的任务特征。传入 `None` 表示不区分任务特征，仅适合任务确实固定的情况。 |
| `prune_fn` | 可选的候选裁剪函数；返回真值保留候选，返回假值丢弃候选。 |
| `cache_dir` | 磁盘缓存目录。默认读取环境变量或使用默认路径；传入 `False` 关闭磁盘缓存。 |
| `cache_version` | 字符串或 `None`。加入缓存标识，用于区分需要重新调优的实现或外部依赖版本。 |

### 配置空间

每次执行候选时，调用形式相当于：

```python
original_function(*args, **{**config, **kwargs})
```

候选字典可以包含多个参数；不同候选的字段也可以不同，但每个实际执行的候选都必须满足函数签名。例如，某个候选省略 `num_warps` 时，函数应为它提供默认值。

`triton.Config` 可以作为字典中的值传入，外层仍然是字典列表：

```python
gemm_config_space = [
    {"gemm_config": triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4)},
    {"gemm_config": triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8)},
]
```

被装饰函数需接收 `gemm_config`，并自行将其中的参数用于 kernel 启动；调优器不会自动展开 `triton.Config`。

### `key_fn` 与 `to_hashable`

调优器调用的是 `key_fn(*args, **kwargs)`：其中不包含已经消费的 `autotune` 等控制参数，也不会额外注入当前候选配置。调用者自己显式传入的参数覆盖仍会出现在 `kwargs` 中。

`to_hashable(tensor)` 返回：

```python
(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
```

它不读取张量数据，也不包含张量地址或设备。任务如果还受其他条件影响，应将这些条件加入 key，例如标量模式、矩阵转置标志或上下文中的拓扑信息。设备信息会另外参与缓存标识。

上下文对象，包括 `dataclass`，按普通 Python 对象传入，不会自动拆解。`key_fn` 应从对象中提取稳定特征；不要把每次调用变化的计数器、通信 epoch 或临时缓冲区地址直接作为任务特征。

### `prune_fn`

裁剪函数的调用形式为：

```python
prune_fn(config, *args, **kwargs)
```

它收到原始候选字典和调用参数。共享内存、tile 大小等约束需要由用户在裁剪函数中表达；当前接口没有通用的共享内存自动裁剪规则。

本地模式在有效内存缓存命中时不会重新执行裁剪。不要依赖未纳入 key 或版本标识的可变状态决定候选是否合法。分布式模式每次调用都会先执行本地裁剪，再对齐全组候选。

## 3. 调用参数与本地模式

```python
result = vector_add(
    x, y, out,
    autotune=True,
    autotune_allow_arg_overwrite=False,
    autotune_pg=None,
)
```

| 调用参数 | 默认值 | 行为 |
| --- | --- | --- |
| `autotune` | `True` | 是否测速并使用调优缓存。设为 `False` 的具体选配置规则见下表。 |
| `autotune_allow_arg_overwrite` | `False` | 本地模式下，是否允许调用者显式覆盖候选参数。分布式模式的规则不同，见下文。 |
| `autotune_pg` | `None` | 非 `None` 时进入指定通信组的联合调优；否则进入本地模式。 |

这些是调优器保留的调用参数，不会转发给原函数。其余参数按原函数签名传递。

本地模式只由当前进程调优：有效内存缓存未命中时，先剪枝，再查询磁盘缓存；仍未命中才依次测量候选。测速使用当前 Triton 后端提供的 benchmarker，请求 `quantiles=[0.5]`。预热和重复策略由该 benchmarker 决定。

本地测速过程中，候选执行或 benchmarker 抛出异常，或 benchmarker 返回负数、无穷大、NaN 等无效耗时，该候选会被记为无穷大并继续测试其他候选。没有任何有效候选时抛出 `RuntimeError`。测速结束后会额外执行最佳配置，返回的是原函数的执行结果，不是测速结果。

### 禁用调优

```python
result = vector_add(x, y, out, autotune=False)
```

| 模式 | `autotune=False` 时的行为 |
| --- | --- |
| 本地 | 不剪枝、不查询调优缓存，直接使用原始 `config_space[0]`；参数覆盖检查仍然生效。 |
| 分布式 | 仍执行各 rank 的剪枝、候选交集和一致性检查，使用规范化排序后的第一个共同候选；不测速、不查询调优缓存。 |

因此，`autotune=False` 不表示“强制使用之前选出的最佳配置”，也不表示“强制重新调优”。

### 显式覆盖候选参数

本地模式默认拒绝与任意候选字段重名的调用参数。要固定某个参数、继续调优其余参数，需要显式允许覆盖：

```python
result = vector_add(
    x, y, out,
    block_size=256,
    autotune_allow_arg_overwrite=True,
)
```

此时调用者的 `block_size=256` 优先，并产生一条 Python warning。该检查使用全部候选字段的并集。

当前分布式模式允许显式覆盖，不依赖 `autotune_allow_arg_overwrite` 开关，但要求所有 rank 对每个共同候选使用相同的最终调优参数。普通输入可以不同，候选参数不能因 rank 而异。

## 4. 分布式联合调优

### 使用条件

- 在调用之前初始化 `torch.distributed` 通信组，并设置各进程使用的设备。
- 组内所有 rank 必须按相同顺序进入被装饰函数，且使用相同的 `autotune` 开关。
- `autotune_pg` 应覆盖实际参与算子通信的进程；它不会自动改变函数内部的通信组。
- 开启分布式调优时需要有效的 `LOCAL_WORLD_SIZE`。通过 `torchrun` 启动会提供该环境变量。
- 当前分布式计时使用 Triton 活跃设备接口的 Event 和同步操作，不能据此将它当成通用 CPU 计时器。

### 完整示例：调优分块 AllReduce

下面用 PyTorch NCCL 通信展示函数级联合调优，不依赖 TLE 专用运行库。所有 rank 使用相同输入长度。每次调用先恢复输出，再执行通信，避免反复测速时不断累加上一轮结果。

保存为 `allreduce_autotune.py`：

```python
import os

import torch
import torch.distributed as dist

from flagtree.tune import autotune, to_hashable


def reduce_key(x, out, group, **kwargs):
    return to_hashable(x), to_hashable(out), dist.get_world_size(group)


@autotune(
    config_space=[{"chunk_elements": 1 << 18}, {"chunk_elements": 1 << 20}],
    key_fn=reduce_key,
    cache_version="chunked-allreduce-v1",
)
def chunked_all_reduce(x, out, group, *, chunk_elements):
    out.copy_(x)
    for start in range(0, out.numel(), chunk_elements):
        dist.all_reduce(out[start:start + chunk_elements], group=group)
    return out


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    x = torch.full((1 << 20,), float(rank + 1), device="cuda")
    out = torch.empty_like(x)
    expected = torch.full_like(x, world_size * (world_size + 1) / 2)

    for _ in range(2):
        result = chunked_all_reduce(x, out, group, autotune_pg=group)
        torch.testing.assert_close(result, expected)

    if rank == 0:
        print("AllReduce result is correct")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

单机双卡启动：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 allreduce_autotune.py
```

这里 `group` 是传给原函数的普通参数，`autotune_pg` 是传给调优器的控制参数。两者分别指定算子的通信范围和调优协调范围。

### 调优流程

1. 各 rank 本地剪枝，对候选进行规范化编码、去重。
2. 对全组候选取交集并统一排序。任一 rank 不支持的候选不会进入测速。
3. 检查调用签名、最终调优参数，以及全组 `autotune` 开关是否一致。
4. 收集各 rank 的任务和环境信息，构造全组缓存标识。
5. 组内 rank 0 查询内存、磁盘缓存，并广播命中结果。
6. 未命中时，对每个共同候选固定预热 5 次、测量 10 次；每个 rank 取这 10 次的平均耗时，再通过组内 `MAX` 归约评分。
7. 组内 rank 0 选择评分最低的候选并广播。各 rank 保存内存结果，指定 rank 尝试落盘，最后共同执行最佳配置。

缓存命中会跳过候选测速，但不会跳过前面的一致性检查、上下文收集和广播。不存在“某个 rank 单独命中就提前执行”的分支。

分布式候选和任务 key 需要支持稳定编码。配置可使用 `None`、布尔值、整数、有限浮点数、字符串、列表、元组、字符串键字典及嵌套组合；还支持不含 `pre_hook`、`ir_override` 的 `triton.Config`。任务 key 还支持 `torch.dtype`、`torch.device`、`torch.Size`，但最终 key 本身仍必须可哈希。任意自定义对象不直接作为分布式配置或任务 key 使用。

## 5. 实际示例：TLE 跨节点 AllGather

仓库中提供了以下配套文件：

- [06-inter-node-allgather-autotune.py](../python/tutorials/tle/06-inter-node-allgather-autotune.py)
- [06-inter-node-allgather-autotune.sh](../python/tutorials/tle/06-inter-node-allgather-autotune.sh)

该示例固定为两节点、每节点四卡，每个 rank 输入 16 MiB 的 float32 数据；调优 `block` 与 `node_block` 的四种组合，保留 `num_warps=32`。

装饰器位于 Python 层的 `push_2d_all_gather` 上。调用形式是：

```python
output = push_2d_all_gather(local_input, ctx, autotune_pg=dist.group.WORLD)
```

其中 `ctx` 是 `AllGatherContext` 对象，直接作为参数传入。该示例针对重复调用做了两项配套处理：每次递增信号 epoch，并在本轮所有 GPU 工作之后提交一个全组 GPU 屏障，确保下一轮复用缓冲区前各 rank 已完成上一轮工作。不能只在原来固定 epoch 的一次性算子外添加装饰器就认为它能安全反复测速。

输入拷贝和该 GPU 屏障均包含在测得的算子时间中。上下文和已注册缓冲区应配套使用，不要重置 epoch，也不要在多个流上并发使用同一上下文。

在两台机器的仓库根目录分别执行，替换主节点 IP，并按实际设备配置 `FLAGCX_IB_HCA` 和 `CUDA_VISIBLE_DEVICES`：

```bash
# 节点 0
NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash python/tutorials/tle/06-inter-node-allgather-autotune.sh

# 节点 1
NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash python/tutorials/tle/06-inter-node-allgather-autotune.sh
```

示例依赖与当前 FlagTree 匹配的 FlagCX 运行库、device bitcode、头文件和 wrapper。安装要求见 [FlagTree NVIDIA 指南](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-nvidia)，FlagCX 提交应以本次 checkout 的 [依赖配置](../python/setup_tools/utils/__init__.py)为准。

脚本用两组不同输入对照 PyTorch 校验结果。第一轮可能调优，也可能命中已有磁盘缓存；第二轮在上下文不变时复用内存缓存。

## 6. 缓存配置与失效

### `cache_dir`

| 设置 | 含义 |
| --- | --- |
| 显式路径，如 `cache_dir="/data/autotune"` | 使用指定目录，优先级最高；支持 `~` 展开。 |
| `cache_dir=None` 或省略 | 使用 `FLAGTREE_AUTOTUNE_CACHE_DIR`；环境变量未设置时使用 `~/.flagtree/autotune`。 |
| `cache_dir=False` | 关闭磁盘读写，保留内存缓存；不表示每次都重新测速。 |

环境变量应在创建调优器之前设置，例如在启动 Python 前：

```bash
export FLAGTREE_AUTOTUNE_CACHE_DIR=/data/liujl/autotune_cache
```

文件布局如下：

```text
autotune_cache/
├── local/
│   └── <cache-key>.json
└── distributed/
    └── <cache-key>.json
```

文件记录 `schema`、`key`、`best_config`、`best_time_ms` 和 `identity`。`best_config` 是规范化配置编码，读取时映射回当前有效候选；不持久化任意 Python 对象或张量数据，也不保存所有候选的测速明细。

这与 Triton kernel 编译缓存、`TMPDIR` 下的 FlagCX allocator 编译缓存分别管理。修改这里的目录不会清理后两者。

### 缓存匹配

本地模式保留以 `key_fn` 结果索引的内存缓存，同时检查配置、覆盖参数等组成的缓存标识。查询磁盘前会剪枝，磁盘标识还包含剪枝后的候选。分布式模式使用独立的全组内存缓存。

持久化标识包含函数模块、名称、源码与代码对象摘要，任务 key、候选与覆盖参数，以及设备和软件环境。本地模式还包含剪枝函数代码信息；分布式模式包含按组内 rank 排列的全组上下文、通信后端、组大小和测速协议。

环境信息包含后端 target、设备标识、Python/PyTorch/Triton 版本和 Triton 构建标识等；设备名称、容量、UUID 等属性仅在后端可提供时加入。源码路径、代码位置变化也可能保守地导致缓存未命中。

读取时会校验格式版本、缓存键和最佳配置是否属于当前有效候选。缓存不存在、损坏或校验不通过时按未命中处理。

本地模式无法编码某些 key/config、无法获得函数或剪枝函数代码标识，或无法探测环境时，可以退回内存缓存。此时不要依赖完整的持久化失效机制，任务差异仍需由正确的 `key_fn` 表达。分布式编码及环境准备失败会统一报错；函数代码标识不可得时可以使用全组内存缓存，但不落盘。

### `cache_version`

`cache_version` 是用户指定的版本标签，默认 `None`，不表示关闭缓存。修改它会改变可构造的缓存标识；新标识没有缓存时才重新调优，不会删除旧文件。如果该版本之前已有匹配缓存，仍可命中。

建议在以下变化后更新版本：

- 被装饰函数调用的其他 kernel、辅助函数或外部运行库发生变化，而包装函数本身没有变化。
- 网络拓扑、通信策略、闭包或全局状态等影响性能的因素发生变化，且没有充分体现在任务 key 中。

当前标识不会递归追踪所有代码依赖，也不会完整检测网络拓扑。内部 `_CACHE_SCHEMA` 管理 JSON 格式兼容性，与用户的 `cache_version` 不是同一概念。

### 分布式读写者与节点顺序

只有通信组 rank 0 决定是否命中缓存，并将结果广播给其他成员。它不会在本地未命中后向其他节点寻找缓存文件，也不会在命中后补写其他节点缺失的文件。

测速结束后，满足 `group_rank % LOCAL_WORLD_SIZE == 0` 的 rank 写磁盘。标准两机八卡 WORLD 组下，通常是全局 rank 0 和 4，各自保存同一份全组最佳结果。这是基于编号的规则，在任意子组中不一定对应每个物理节点的首个进程。

若缓存目录是共享文件系统，多个写入者可能写同一个文件；若是节点本地目录，则各自保存副本。固定读取节点或使用共享目录时，额外副本不是命中的必要条件。

交换节点的 `node_rank` 可能改变上下文在缓存键中的顺序，尤其是包含设备 UUID 或不同硬件信息时，可能产生新的缓存键。因此，即使每节点都留有旧副本，也不能保证交换主节点后仍命中。仅改变 shell 命令启动先后、保持 rank 配置及上下文不变，不会因此改变缓存键。

### 写入完整性与并发

一次成功调优只保存最终最佳结果，时间点在最佳候选选定之后、正式执行该配置之前。写入失败会记录 warning，不中断后续正常执行，内存结果仍可使用。

写入流程是：在目标目录创建唯一临时文件，写入完整 JSON，执行 `flush()`、`fsync()`，关闭文件后用 `os.replace()` 原子替换正式路径，最后尝试清理临时文件。

当前没有文件锁。不同进程各写自己的临时文件，可以避免内容交叉写入，但同一缓存键可能被重复调优、重复发布，最后成功替换者生效。保存的 `best_time_ms` 不用于与已有文件比较择优。这里的文件写入机制也不替代分布式 rank 间的协调协议。

## 7. 日志与常见问题

调优器使用 `logging.getLogger("flagtree.tune")`，不自行创建日志文件。应用可以配置日志输出方式：

```python
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("flagtree.tune").setLevel(logging.DEBUG)
```

此示例通过默认 handler 输出到标准错误。磁盘读写异常记录 warning；本地缓存标识构造或环境探测抛出异常时记录 debug。无法获取函数源码而禁用持久化等情况不一定有日志。当前没有逐候选耗时、每次缓存命中的 verbose 日志。

| 现象或错误 | 检查方向 |
| --- | --- |
| `Arguments are also supplied by config_space` | 本地调用显式传入了候选字段；确需覆盖时设置 `autotune_allow_arg_overwrite=True`。 |
| `prune_fn removed all configurations` | 本地剪枝删除了所有候选；调整配置或剪枝规则。 |
| `No common configuration remains after distributed pruning` | 各 rank 剪枝结果没有交集；至少提供一个所有成员都支持的相同候选。 |
| 全组 `autotune` 设置或最终调优参数不一致 | 检查每个 rank 的调用开关与显式覆盖参数。 |
| `LOCAL_WORLD_SIZE` 缺失或非法 | 使用 `torchrun`，或由启动器正确设置每节点进程数。 |
| `Unsupported distributed configuration value` | 将自定义对象转换成受支持的稳定配置值或任务特征。关闭磁盘缓存不会取消分布式编码要求。 |
| 进程重启后仍然调优 | 检查 cache 路径、日志、函数标识是否可用、全组环境是否变化、最佳配置是否仍有效。 |
| 修改 `cache_dir` 后当前实例仍使用原配置 | 内存缓存与磁盘目录分开管理；仅改目录不会清掉已有内存结果。 |

当前没有 `autotune_verbose`、`contextual_autotune` 或 `TRITON_DIST_AUTOTUNE_ALWAYS_TUNE` 等接口。没有公开的 `do_bench` 装饰器参数，也没有独立的“强制每次重新调优”开关。对正常可编码的任务，可通过更换 `cache_version` 使用新的缓存标识。

## 8. 重复执行与错误处理边界

调优会反复调用原函数，不会自动恢复输入、重置通信状态或校验数值正确性。用户需要保证每次调用有效，并在调优外验证输出；涉及累加、原地写入、信号计数或缓冲区复用时尤其需要明确重置与同步规则。

分布式的剪枝、候选编码、参数校验和缓存准备阶段会收集错误，避免健康 rank 在这些准备阶段独自继续。但 benchmark 或正式执行中的 kernel/通信异常会直接传播，不会捕获后继续下一个候选；其他 rank 仍可能等待通信超时。当前实现不提供运行阶段的故障恢复。

同一进程组上的多次调用也必须遵守相同的 collective 顺序。若外层函数内部还嵌套其他调优器，需自行保证其重复调用不会改变通信序列；本接口不会自动协调嵌套调优。
