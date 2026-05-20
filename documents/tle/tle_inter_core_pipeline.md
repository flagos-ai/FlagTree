# TLE CV 核间流水设计文档

## 1. 背景

### 1.1 Ascend AI Core架构

华为昇腾910C AI Core包含两种计算单元：

| 核心类型 | 简称 | 主要功能 |
|---------|------|----------|
| **Cube核心** | CUBE | 矩阵运算（MAC），擅长大块矩阵乘法 |
| **Vector核心** | VECTOR | 标量/向量运算，擅长element-wise操作 |

### 1.2 流水线类型

**CV间流水** (Inter-Core Pipeline): Cube核心与Vector核心之间的数据同步与流水并行

**CV内流水** (Intra-Core Pipeline): 单个核心内部指令级的流水线调度

### 1.3 为什么需要手排流水

- 自动流水调度在某些复杂场景下效率不足
- CV间数据依赖关系需要显式同步
- 隐藏数据搬运延迟，提高计算单元利用率

---

## 2. CV间流水分析

### 2.1 流水线同步机制

tle拓展提供事件同步机制，通过PIPE进行数据传输：

```python
import triton.experimental.tle as tle
pipe = tle.dsa.ascend.PIPE

# 发送端设置事件
tle.dsa.ascend.sync_block_set(
    sender="cube",           # 发送方: "cube" 或 "vector"
    receiver="vector",       # 接收方
    event_id=0,              # 事件ID (0-15)
    sender_pipe=pipe.PIPE_FIX,     # 发送方使用的PIPE
    receiver_pipe=pipe.PIPE_MTE2   # 接收方使用的PIPE
)

# 接收端等待事件
tle.dsa.ascend.sync_block_wait(
    sender="cube",
    receiver="vector",
    event_id=0,
    sender_pipe=pipe.PIPE_FIX,
    receiver_pipe=pipe.PIPE_MTE2
)
```

### 2.2 PIPE类型

```python
class PIPE(enum.Enum):
    PIPE_S      = ...   # Scalar pipe
    PIPE_V      = ...   # Vector pipe
    PIPE_M      = ...   # Memory pipe
    PIPE_MTE1   = ...   # MTE1
    PIPE_MTE2   = ...   # MTE2
    PIPE_MTE3   = ...   # MTE3
    PIPE_ALL    = ...   # 所有PIPE
    PIPE_FIX    = ...   # Fixpipe
```

### 2.3 典型CV间流水模式（双缓冲，预加载）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Lightning Indexer 实际流水时序                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  初始化: set(0), set(1)  ← 提前设置，Cube可立即计算2块                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Cube Core                                    │   │
│  ├──────┬────────┬────────┬────────┬────────┬────────┬────────┤      │   │
│  │QK[0] │QK[1]  │ wait(0)│QK[2]  │ wait(1)│QK[3]  │ wait(0)│...   │   │
│  │      │        │        │        │        │        │        │      │   │
│  └──┬───┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘      │   │
│     │set(0)   │set(1)  │        │set(0)  │        │        │          │   │
│     │         │        │        │        │        │        │          │   │
│     ▼         ▼        ▼        ▼        ▼        ▼        ▼          │   │
│  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │                    Vector Core                                  │   │   │
│  ├────────┬────────┬────────┬────────┬────────┬────────┬────────┤      │   │
│  │wait(0) │ReLU[0] │wait(1) │ReLU[1] │wait(0) │ReLU[2] │wait(1) │...   │   │
│  │        │set(0)  │        │set(1)  │        │set(0)  │        │      │   │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘      │   │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  关键点:                                                                │
│  • 初始化 set(0)+set(1) 使Cube能连续计算QK[0]和QK[1] (预加载2块)         │
│  • Cube: set(0) → set(1) → wait(0) → set(0) → wait(1) → set(1) → ...   │
│  • Vector: wait(0) → set(0) → wait(1) → set(1) → wait(0) → ...          │
│  • 相同flag的set/wait配对: set(0)→wait(0)→set(0)→wait(0)               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.4 双缓冲流水代码模式

```python
# 初始化：提前设置两个缓冲区，实现预加载
tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
db_flag = 0

for i in range(iterations):
    # ===== Cube计算 =====
    # 1. 计算当前块
    result = cube_compute(i)

    # 2. 等待Vector完成上一轮处理，释放缓冲区 db_flag
    #    (前两次迭代立即返回，因为初始化已set)
    tle.dsa.ascend.sync_block_wait("vector", "cube", db_flag % 2,
                                    pipe.PIPE_MTE2, pipe.PIPE_FIX)

    # 3. Cube处理：ReLU等操作，存储到workspace
    result = libdevice.relu(result)
    tl.store(workspace_ptr + ..., result)

    # 4. 通知Vector可以处理缓冲区 db_flag中的数据
    tle.dsa.ascend.sync_block_set("cube", "vector", db_flag % 2,
                                  pipe.PIPE_FIX, pipe.PIPE_MTE2)

    # 5. 等待Cube将结果写回GM（完成reduce和输出）
    tle.dsa.ascend.sync_block_wait("cube", "vector", db_flag % 2,
                                    pipe.PIPE_FIX, pipe.PIPE_MTE2)

    # 6. Vector加载并reduce（此时已完成）
    data = tl.load(workspace_ptr + ...)
    output = tl.sum(data * weight, 0)
    tl.store(out_ptr + ..., output)

    # 7. 释放缓冲区，通知Cube可以继续使用
    tle.dsa.ascend.sync_block_set('vector', 'cube', db_flag % 2,
                                  pipe.PIPE_MTE2, pipe.PIPE_FIX)

    db_flag += 1
```

**关键点**:
- **预加载**: 初始化时set(0)和set(1)，使Cube可连续计算前两块
- **数据流向**: Cube计算 → workspace → Vector处理 → GM输出
- **配对模式**: 同一flag的set/wait配对使用
  - `set(vector, cube, 0)` → `wait(vector, cube, 0)`：缓冲区可用
  - `set(cube, vector, 0)` → `wait(cube, vector, 0)`：数据已就绪/处理完成

---

## 3. 简单CV Mix手排流水示例

### 3.1 设计目标

设计一个简单的示例，展示基本的CV间流水：
- **输入**: 矩阵 A [M, K], 矩阵 B [K, N]
- **操作**: C = A @ B, 然后 D = C + 1 + 100×0.001 = C + 1.1
- **流水**: Cube做矩阵乘法（tl.dot），Vector做加法+循环（模拟更长计算时间）
- **单核心**: grid=(1,)，M方向切分

### 3.2 测试结果

**配置**: BLOCK_SIZE_M=64/128, BLOCK_SIZE_N=128, Vector端100次循环模拟长计算

| 测试用例 | M | N | K | 单缓冲 (us) | 双缓冲 (us) | 加速比 |
|---------|---|---|---|-------------|-------------|--------|
| small | 8192 | 128 | 128 | 1132.55 | 1037.62 | **1.09x** |
| medium | 32768 | 128 | 256 | 4517.37 | 4142.96 | **1.09x** |
| large | 65536 | 128 | 256 | 9040.41 | 8282.57 | **1.09x** |

**结论**: 双缓冲流水稳定提升约 9% 的性能。当Vector计算时间占比增加时，流水并行效果更明显。

### 3.3 完整实现

**文件**: [04-cv-mix-pipeline.py](../../python/tutorials/tle/04-cv-mix-pipeline.py)

```python
@triton.jit
def lightning_indexer_tnd_pa_stage1_kernel(...):
    # 初始化同步
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    db_flag = 0

    for k_i in range(k_blk_cnt):
        # 1. Cube计算: Q @ K^T
        k_block = tl.load(k_block_ptr)
        qk_block = tl.dot(q_block, tl.trans(k_block))

        # 2. 等待Vector完成，获取缓冲区
        tle.dsa.ascend.sync_block_wait("vector", "cube", (db_flag % 2),
                                        pipe.PIPE_MTE2, pipe.PIPE_FIX)

        # 3. Vector处理: ReLU
        qk_block = libdevice.relu(qk_block)
        tl.store(wsp_ptr + ..., qk_block)

        # 4. 通知Vector可以处理
        tle.dsa.ascend.sync_block_set("cube", "vector", (db_flag % 2),
                                      pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # 5. 等待Vector完成reduce
        tle.dsa.ascend.sync_block_wait("cube", "vector", (db_flag % 2),
                                        pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # 6. Vector加载并reduce
        qk_slice_0 = tl.load(wsp_ptr + ...)
        tmp_reduce_res_block = tl.sum(qk_slice_0 * weight_block_0, 0)
        tl.store(out_ptr + ..., tmp_reduce_res_block)

        # 7. 设置下一轮同步
        tle.dsa.ascend.sync_block_set('vector', 'cube', (db_flag % 2),
                                      pipe.PIPE_MTE2, pipe.PIPE_FIX)

        db_flag += 1
```

### 3.3 流水时序图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    单缓冲版本时序（顺序执行）                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Cube:   [A@B] ──set(0)──> wait(0) ──>                              │
│                      │                                                │
│  Vector:             wait(0) ─> [Add] ──>  set(0)                       │
│                                                                         │
│  循环重复 (每个M块):                                                    │
│  ┌─────────┬─────────┬─────────┬─────────┐                            │
│  │ Cube[0] │ sync    │ Cube[1] │ sync    │...                         │
│  │ A@B     │         │ A@B     │         │                            │
│  └────┬────┴────┬────┴────┬────┴────┬────┘                            │
│       │         │         │         │                                 │
│       ▼         ▼         ▼         ▼                                 │
│  ┌─────────┬─────────┬─────────┬─────────┐                            │
│  │ Vector  │ Vector  │ Vector  │ Vector  │...                         │
│  │ Add[0]  │         │ Add[1]  │         │                            │
│  └─────────┴─────────┴─────────┴─────────┘                            │
│                                                                         │
│  特点: Cube必须等待Vector完成，无并行                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    双缓冲版本时序（流水并行）                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  初始化: set(0), set(1)  ← 两个缓冲区都可用                              │
│                                                                         │
│  Cube Core:                                                            │
│  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐                     │
│  │A@B[0]│A@B[1]│wait(0)│A@B[2]│wait(1)│A@B[3]│wait(0)│...              │
│  └──┬───┴──┬───┴───┬────┴──┬───┴───┬────┴──┬───┴────┘                     │
│     │set(0)  │set(1)      │set(0)      │set(1)                      │
│                                                                         │
│  Vector Core:                                                          │
│     │wait(0)  │Add[0] │set(0) │wait(1)│Add[1] │set(1)│...               │
│                                                                         │
│  时序详解:                                                              │
│  • block 0: Cube计算A@B → 存入buf[0] → Vector处理Add                    │
│  • block 1: Cube计算A@B → 存入buf[1] → Vector处理Add                    │
│  • block 2: Cube等待buf[0]释放 → 计算并存入buf[0] → Vector处理Add       │
│  • ...                                                                  │
│                                                                         │
│  理论优势: 当Vector计算时间较长时，Cube可以提前开始下一轮计算              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 参数设计说明

| 参数 | 典型值 | 说明 |
|------|--------|------|
| BLOCK_SIZE_M | 64/128 | M维度的块大小 |
| BLOCK_SIZE_N | 128 | N维度的块大小 |
| K | ≤256 | 一次性加载，需满足硬件限制 |

**简化假设**:
- K维度较小（≤256），可一次性加载到L1/L0缓冲区
- 使用单核心grid=(1,)，M方向循环切分
- 双缓冲通过交替使用两个workspace块实现
- Vector端100次循环模拟更长计算时间，展示流水并行效果

---

## 4. 现有案例：Lightning Indexer

### 4.1 案例概述

**文件**: [05-lightning-indexer-v1.py](../../python/tutorials/tle/05-lightning-indexer-v1.py)

Lightning Indexer是一个复杂的算子，包含完整的CV间流水实现：

- **Stage 1**: 计算query-key注意力得分并累加
- **Stage 2**: TopK选择和索引处理

### 4.2 关键流水代码片段

```python
@triton.jit
def lightning_indexer_tnd_pa_stage1_kernel(...):
    # 初始化同步
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    db_flag = 0

    for k_i in range(k_blk_cnt):
        # 1. Cube计算: Q @ K^T
        k_block = tl.load(k_block_ptr)
        qk_block = tl.dot(q_block, tl.trans(k_block))

        # 2. 等待Vector完成，获取缓冲区
        tle.dsa.ascend.sync_block_wait("vector", "cube", (db_flag % 2),
                                        pipe.PIPE_MTE2, pipe.PIPE_FIX)

        # 3. Vector处理: ReLU
        qk_block = libdevice.relu(qk_block)
        tl.store(wsp_ptr + ..., qk_block)

        # 4. 通知Vector可以处理
        tle.dsa.ascend.sync_block_set("cube", "vector", (db_flag % 2),
                                      pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # 5. 等待Vector完成reduce
        tle.dsa.ascend.sync_block_wait("cube", "vector", (db_flag % 2),
                                        pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # 6. Vector加载并reduce
        qk_slice_0 = tl.load(wsp_ptr + ...)
        tmp_reduce_res_block = tl.sum(qk_slice_0 * weight_block_0, 0)
        tl.store(out_ptr + ..., tmp_reduce_res_block)

        # 7. 设置下一轮同步
        tle.dsa.ascend.sync_block_set('vector', 'cube', (db_flag % 2),
                                      pipe.PIPE_MTE2, pipe.PIPE_FIX)

        db_flag += 1
```

### 4.3 流水时序图

```
═════════════════════════════════════════════════════════════════════
时间线 →
─────────────────────────────────────────────────────────────────────

初始化: set(0), set(1)  ← 预加载两块

Cube Core:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│QK[0] │QK[1] │wait(0)│QK[2] │wait(1)│QK[3] │wait(0)│QK[4] │...
└──┬───┴──┬───┴───┬────┴──┬───┴───┬────┴──┬───┴───┬────┘
   │set(0)  │set(1)  │      │set(0)  │      │set(1)  │
   │        │        │      │        │      │        │
   └────────┼────────┼──────┴────────┼──────┴────────┼──── flag=0
            │        │                │                │
            └────────┼────────────────┼────────────────┼──── flag=1
                     │                │                │

Vector Core:
         │wait(0) │ReLU[0] │set(0) │wait(1) │ReLU[1] │set(1) │...
         ▼        ▼        ▼       ▼        ▼        ▼       ▼
┌────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│wait(0) │ReLU  │store │set(0)│wait(1)│ReLU  │store │set(1)│...
│        │[0]   │[0]   │      │      │[1]   │[1]   │      │
└────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

关键时序:
• QK[0], QK[1] 可以连续执行（预加载）
• wait(0) 在第3次迭代才阻塞，等待ReLU[0]完成
• Cube的set和wait都是针对同一flag（配对使用）
```

### 4.4 性能测试结果：手动同步 vs 自动同步

**测试配置**:
- Shape: (32, 8192, 8192, 2048) - batch=32, s1=8192, s2=8192, topk=2048
- Kernel: `lightning_indexer_tnd_pa_stage1_kernel`
- Core Type: MIX_AIC
- 样本数: 35

| 版本 | 总时间 (us) | 最小时间 (us) | 最大时间 (us) | 平均时间 (us) | 占比 |
|------|-------------|---------------|---------------|---------------|------|
| **手动同步** | 4257684.672 | 120666.249 | 123631.690 | **121648.133** | 46.049% |
| **自动同步** | 6861609.987 | 188187.740 | 204373.937 | **196046.000** | 57.907% |

**性能提升**:

| 指标 | 手动同步 | 自动同步 | 提升 |
|------|----------|----------|------|
| 平均时间 | 121648.133 us | 196046.000 us | **+37.9%** ⬆️ |
| 最小时间 | 120666.249 us | 188187.740 us | **+35.9%** |
| 最大时间 | 123631.690 us | 204373.937 us | **+39.6%** |

**结论**:
- 手动同步比自动同步快约 **38%**
- 手动同步精确控制了 Cube 和 Vector 之间的流水时序
- 双缓冲预加载策略显著减少了等待时间
- 对于复杂的 CV 流水算子，手动同步可以带来显著的性能提升

---

## 5. TLE接口参考

### 6.1 CV间同步接口

```python
# 发送同步事件
tle.dsa.ascend.sync_block_set(
    sender: str,           # "cube" 或 "vector"
    receiver: str,         # "cube" 或 "vector"
    event_id: int,         # 0-15
    sender_pipe: PIPE,     # 发送方PIPE
    receiver_pipe: PIPE    # 接收方PIPE
)

# 等待同步事件
tle.dsa.ascend.sync_block_wait(
    sender: str,
    receiver: str,
    event_id: int,
    sender_pipe: PIPE,
    receiver_pipe: PIPE
)

# 全部同步
tle.dsa.ascend.sync_block_all(
    mode: str,             # "all_cube", "all_vector", "all", "all_sub_vector"
    event_id: int
)
```

### 6.2 地址空间

```python
# 内存层级
tle.dsa.ascend.UB   # Unified Buffer (统一缓冲区)
tle.dsa.ascend.L1   # L1 Cache
tle.dsa.ascend.L0A  # Cube输入缓冲区A
tle.dsa.ascend.L0B  # Cube输入缓冲区B
tle.dsa.ascend.L0C  # Cube输出缓冲区
```

### 6.3 Vector子核操作

```python
# 获取当前Vector子核ID (910C上每个AI Core有2个Vector子核)
vec_id = tle.dsa.ascend.sub_vec_id()

# 获取Vector子核数量
vec_num = tle.dsa.ascend.sub_vec_num()
```

### 6.4 切片操作

```python
# 提取切片
sub_tensor = tle.dsa.extract_slice(
    tensor,           # 源张量
    offsets,          # 起始偏移 (tuple)
    sizes,            # 大小 (tuple)
    strides           # 步长 (tuple)
)

# 插入切片
result = tle.dsa.insert_slice(
    dst_tensor,       # 目标张量
    src_tensor,       # 源张量
    offsets,          # 插入位置
    sizes,            # 大小
    strides           # 步长
)
```

---

## 6. TLE 同步 OP 设计

### 6.1 概述

TLE (Triton Language Extensions) 为昇腾 NPU 提供了手排流水同步原语，用于实现 Cube 核心与 Vector 核心之间的显式同步。

### 6.2 `sync_block_set` - 设置同步事件

```python
tle.dsa.ascend.sync_block_set(
    sender: str,           # 发送方核心: "cube" 或 "vector"
    receiver: str,         # 接收方核心: "cube" 或 "vector"
    event_id: int,         # 事件ID (0-15)
    sender_pipe: PIPE,     # 发送方使用的PIPE
    receiver_pipe: PIPE    # 接收方使用的PIPE
)
```

**功能**：发送方核心设置一个同步事件，通知接收方核心可以继续执行。

**参数说明**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `sender` | str | 发送方核心，`"cube"` 或 `"vector"` |
| `receiver` | str | 接收方核心，`"cube"` 或 `"vector"` |
| `event_id` | int | 事件ID，范围 0-15，用于区分不同的同步事件 |
| `sender_pipe` | PIPE | 发送方使用的 PIPE，如 `PIPE_FIX`、`PIPE_MTE2` |
| `receiver_pipe` | PIPE | 接收方使用的 PIPE |

**使用场景**：
- **通知数据就绪**：Cube 计算完成后，通知 Vector 可以读取数据
- **释放缓冲区**：Vector 处理完成后，通知 Cube 可以复用缓冲区

### 6.3 `sync_block_wait` - 等待同步事件

```python
tle.dsa.ascend.sync_block_wait(
    sender: str,           # 等待的发送方核心
    receiver: str,         # 接收方核心（自己）
    event_id: int,         # 等待的事件ID
    sender_pipe: PIPE,     # 发送方使用的PIPE
    receiver_pipe: PIPE    # 接收方使用的PIPE
)
```

**功能**：阻塞当前核心，等待发送方核心设置对应的同步事件。

**参数说明**：与 `sync_block_set` 相同，需要配对使用相同的 `sender`、`receiver`、`event_id`。

**使用场景**：
- **等待数据就绪**：Vector 等待 Cube 将数据写入 workspace
- **等待缓冲区释放**：Cube 等待 Vector 释放缓冲区后才能写入

### 6.4 配对使用模式

`sync_block_set` 和 `sync_block_wait` 必须配对使用：

```python
# Cube端：设置事件，通知Vector
tle.dsa.ascend.sync_block_set("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_MTE2)

# Vector端：等待事件
tle.dsa.ascend.sync_block_wait("cube", "vector", 0, pipe.PIPE_FIX, pipe.PIPE_MTE2)
```

**配对规则**：
1. 相同的 `sender` 和 `receiver`（但位置相反）
2. 相同的 `event_id`
3. 相同的 `sender_pipe` 和 `receiver_pipe`

### 6.5 双缓冲流水示例

```python
# 初始化：提前设置两个缓冲区都可用
tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

db_flag = 0
for block_idx in range(num_blocks):
    buffer_id = block_idx % 2

    # ===== Cube计算 =====
    result = cube_compute(block_idx)

    # 等待Vector释放缓冲区
    tle.dsa.ascend.sync_block_wait("vector", "cube", buffer_id,
                                    pipe.PIPE_MTE2, pipe.PIPE_FIX)

    # 存储到workspace
    tl.store(workspace_ptr + ..., result)

    # 通知Vector可以处理
    tle.dsa.ascend.sync_block_set("cube", "vector", buffer_id,
                                  pipe.PIPE_FIX, pipe.PIPE_MTE2)

    # 等待Vector完成处理
    tle.dsa.ascend.sync_block_wait("cube", "vector", buffer_id,
                                    pipe.PIPE_FIX, pipe.PIPE_MTE2)

    # ===== Vector处理 =====
    data = tl.load(workspace_ptr + ...)
    output = vector_compute(data)
    tl.store(output_ptr + ..., output)

    # 释放缓冲区
    tle.dsa.ascend.sync_block_set('vector', 'cube', buffer_id,
                                  pipe.PIPE_MTE2, pipe.PIPE_FIX)
```

### 6.6 PIPE 类型说明

| PIPE | 说明 |
|------|------|
| `PIPE_FIX` | Fixpipe，固定功能管道，用于Cube到Vector的数据传输 |
| `PIPE_MTE2` | MTE2，数据传输引擎2，用于数据搬运 |
| `PIPE_MTE3` | MTE3，数据传输引擎3 |

**典型组合**：
- Cube → Vector: `sender_pipe=PIPE_FIX`, `receiver_pipe=PIPE_MTE2`
- Vector → Cube: `sender_pipe=PIPE_MTE2`, `receiver_pipe=PIPE_FIX`

---
