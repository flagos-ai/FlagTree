# FlagTree 函数级自动调优器

`flagtree.tune.autotune` 用于调优 Python 函数的参数。被装饰的函数可以提交分布式通信 kernel。

提供的主要功能：

- 用字典列表定义候选参数，支持 `key_fn` 和 `prune_fn`。
- 本地调优与指定通信组内的联合调优。
- 分布式候选参数列表取交集、执行参数一致性检查和统一缓存命中决策。
- 自动缓存调优结果到 FLAGTREE_AUTOTUNE_CACHE_DIR 环境变量指定的目录或者调优器参数 cache_dir 指定的目录，默认为 ~/.flagtree/autotune/local | distributed/

调优器不会自动识别函数是否包含分布式通信。函数涉及多个 rank 协作时，必须显式传入 `autotune_pg` 进程组参数进行分布式调优。


## 1. 装饰器参数

```python
def autotune(
    config_space,
    key_fn,
    prune_fn=None,
    cache_dir=None,
    cache_version: str | None = None,
):
    ...
```

| 参数 | 说明 |
| --- | --- |
| `config_space` | 必选参数：包含可调参数的字典列表，键对应被装饰函数的关键字参数。 |
| `key_fn` | 必选参数：接受与被装饰函数相同参数的函数，返回用于缓存的可哈希 key |
| `prune_fn` | 可选参数：可选的候选裁剪函数；返回真值保留候选，返回假值丢弃候选。 |
| `cache_dir` | 可选参数：磁盘缓存目录。默认读取环境变量或使用默认路径；传入 `False` 关闭磁盘缓存。 |
| `cache_version` | 可选参数：字符串或 `None`。用于区分需要重新调优的实现，传入自定义值字符串值可以丢弃旧缓存。 |

## 2. 基本用法

假设有一个处理数据的 GPU 函数 `process_data`，需要选择两个参数：

- `kernel_config`：一个 `triton.Config`，指定 kernel 的配置。
- `batch_size`：每批处理多少行数据。

下面只演示参数如何传递，实际 GPU 计算代码需要填写在函数体中。

```python
import triton
from flagtree.tune import autotune, to_hashable


@autotune(
    config_space=[
        {"kernel_config": triton.Config({"BLOCK_SIZE": 128}), "batch_size": 16},
        {"kernel_config": triton.Config({"BLOCK_SIZE": 256}), "batch_size": 32},
    ],
    key_fn=lambda x, **kwargs: to_hashable(x),
)
def process_data(x, *, kernel_config: triton.Config, batch_size: int):
    # 按 batch_size 分批处理数据，用 kernel_config 配置并启动 kernel。
    ...
```

调优器把每个候选字典的两个字段，分别传给函数的 `kernel_config` 和 `batch_size`。其中 `triton.Config` 作为一个对象传入，由函数自行使用。


## 3. 调用示例

填好函数体后，假设 `x` 是待处理的 GPU 张量，普通调用只需要传入 `x`：

```python
result = process_data(x)
```

没有缓存时，调优器测试上述两个候选组合，选出更快的一组。

| 调用参数 | 默认值 | 行为 |
| --- | --- | --- |
| `autotune` | `True` | 是否测速并使用调优缓存。设为 `False` 的具体选配置规则见下表。 |
| `autotune_allow_arg_overwrite` | `False` | 是否允许调用者显式覆盖候选参数。|
| `autotune_pg` | `None` | 非 `None` 时进入指定通信组的联合调优；否则进入本地调优模式。 |

这些是调优器保留的调用参数，不会转发给原函数。其余参数按原函数签名传递。


### 禁用调优

```python
result = process_data(x, autotune=False)
```

| 模式 | `autotune=False` 时的行为 |
| --- | --- |
| 本地 | 不使用 prune_fn 剪枝、不查询调优缓存，直接使用原始 `config_space[0]`执行被装饰函数；参数覆盖检查仍然生效。 |
| 分布式 | 仍执行参数覆盖权限检查、各 rank 的剪枝，之后对各 rank 剪枝后的 config_space 取交集候选交集，使用交集规范化排序后的`config_space[0]`；不测速、不查询调优缓存。 |


### 显式覆盖候选参数

如果希望每批固定处理 64 行，只需显式传入第二个调优参数 `batch_size=64`，并允许覆盖：

```python
result = process_data(
    x,
    batch_size=64,
    autotune_allow_arg_overwrite=True,
)
```

这时，两个候选的实际参数如下：

| `kernel_config` 中的配置 | 原来的 `batch_size` | 覆盖后的 `batch_size` |
| --- | --- | --- |
| `BLOCK_SIZE=128` | `16` | `64` |
| `BLOCK_SIZE=256` | `32` | `64` |

**`batch_size` 固定为 64，`kernel_config` 继续参与调优。** 正式执行最佳配置时，也使用 `batch_size=64`。

省略 `autotune_allow_arg_overwrite=True` 时则会报参数冲突；允许覆盖时会产生一条 Python warning。

分布式调用在显式覆盖时也需要设置 `autotune_allow_arg_overwrite=True`。任一 rank 未允许覆盖却传入冲突参数时，错误会在准备阶段收集，并由全组统一报错；

允许覆盖后，仍要求所有 rank 对每个共同候选使用相同的最终调优参数。

## 4. 分布式联合调优

### 使用条件

- 在调用之前初始化 `torch.distributed` 通信组，并设置各进程使用的设备( 目前 tle.get_mem_pool() 已经完成了这项操作 )。
- 组内所有 rank 必须按相同顺序进入被装饰函数，且使用相同的 `autotune` 开关。
- `autotune_pg` 应覆盖实际参与算子通信的进程( 通常可以传：dist.group.WORLD )；。
- 开启分布式调优时需要有效的 `LOCAL_WORLD_SIZE`。通过 `torchrun` 启动会提供该环境变量。
