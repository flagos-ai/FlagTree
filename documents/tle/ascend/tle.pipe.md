# tle.pipe 接口文档

## 1. 硬件背景

面向 CV mix（Cube/Vector 混合执行）场景的流水线封装接口。

Cube 核做矩阵计算（结果经 FIX 流水从 L0C 搬出），Vector 核做向量后处理（经 MTE2 流水搬入 UB），两侧通过 GM workspace 环形缓冲交接数据，并手工配对 sync_block_set / sync_block_wait 事件表达"数据就绪"（ready）与"slot 释放"（free）。tle.pipe 把这套手写协议封装为与 GPU 侧一致的 producer/consumer 流水边：payload 由 tle.dsa.workspace 提供，同步边由 SyncSpec 描述，底层展开为 sync_block_set / sync_block_wait。

## 2. 接口说明

```python
def pipe(*, capacity, scope="cta", name=None, readers=None, one_shot=False, **fields):  # 公共前端，签名与 GPU 侧逐字一致

def pipe_scheduler(ready_sync=None, free_sync=None, event_base=None)  # Ascend：调度器构造入口

def run_pipeline(functions_and_args, scheduler=None)  # Ascend：role 编排入口（tle.dsa.ascend.run_pipeline）

class pipe_value:
    def writer() -> pipe_writer
    def reader(name=None, fields=None) -> pipe_reader

class pipe_writer:
    def acquire(iteration) -> pipe_slot   # 等 free 事件，返回 stage slot
    def commit(iteration)                 # 发 ready 事件

class pipe_reader:
    def wait(iteration) -> pipe_wait_result  # 等 ready 事件，返回 slot + is_closed
    def release(iteration)                   # 发 free 事件
```

### 返回值

`pipe(...)` 返回 `pipe_value`；`acquire` 返回 `pipe_slot`（字段按名属性化访问，如 `slot.c`）；`wait` 返回 `pipe_wait_result`（`slot` + `is_closed`，Ascend 上 `is_closed` 恒为编译期常量 `False`）。

## 3. 入参说明

| 参数名 | 类型 | 说明 |
|--------|------|------|
| capacity | int | 流水深度（环形 stage 数），编译期整数，取值范围 [1, 16] |
| scope | str | 仅支持 "cta" |
| name | str / None | pipe 名称，仅用于标识；`run_pipeline(scheduler=...)` 的按名配置依赖它 |
| readers | tuple / list | Ascend 暂不支持，传入即报错（原因见第 5 节） |
| one_shot | bool | 单次 ready/full 边模式：commit/wait 可用，acquire/release 拒绝；Ascend 上要求 capacity=1 |
| scheduler | PipeScheduler | 仅 `tle.dsa.ascend.run_pipeline` 接受；由 tle.dsa.ascend.pipe_scheduler 构造，默认 ready 为 cube→vector（FIX→MTE2）、free 为 vector→cube（MTE2→FIX）的标准 CV-mix 握手，event id 自动分配 |
| fields | Workspace | 一个或多个 tle.dsa.workspace payload（关键字传入，如 c=c_workspace），每个 field 的 capacity 必须等于 pipe 的 capacity |
| iteration | int / tensor | 端点方法入参，流水迭代号：stage = iteration % capacity，event id = event_base + stage |

## 4. 与 sync_block_set / sync_block_wait 的区别和联系

**联系**：tle.pipe 的 ready / free 两条同步边，底层正是 sync_block_set / sync_block_wait 这对原语：

| pipe 端点方法 | 展开结果 |
|---------------|----------|
| writer.acquire(i) | sync_block_wait(free_sync, event_base + i % capacity) |
| writer.commit(i) | sync_block_set(ready_sync, event_base + i % capacity) |
| reader.wait(i) | sync_block_wait(ready_sync, event_base + i % capacity) |
| reader.release(i) | sync_block_set(free_sync, event_base + i % capacity) |

生成的 MLIR 即 `hivm.hir.sync_block_set` / `hivm.hir.sync_block_wait`（见同目录两篇文档）。

**区别**：

- **层次不同**：sync_block_set/wait 是单条同步原语，一次调用只发/等一个事件；tle.pipe 是流水边抽象，把成对的握手协议封装为端点方法，用户按 acquire → 写 → commit / wait → 读 → release 的时序编程。
- **参数托管**：set/wait 需要用户手工指定方向、流水类型并管理 event id；pipe 的方向由 SyncSpec 描述（默认即标准 CV-mix 握手），event id 按 stage 自动分配并做冲突检测，首次编排/使用前还自动沿 free 方向预发 capacity 个 set（发射位置固定在 pipe 创建点，位于所有使用它的循环之前），保证首次 acquire 不阻塞。
- **数据面**：set/wait 是纯同步原语；pipe 同时托管数据面——payload 为 GM workspace 环形缓冲，acquire/wait 直接返回对应 stage 的 slot 指针。

## 5. 与 GPU 侧 tle.pipe 的对齐与差异

**已对齐**：工厂签名（关键字传参、参数名与语义一致，公共 `tle.pipe` 与 GPU 版逐字相同）、端点模型（writer.acquire/commit、reader.wait/release）、返回类型（pipe_slot、pipe_wait_result）、参数校验与错误消息（含 readers= 非法输入的报错路径、one_shot 契约）、stage = iteration % capacity 环形语义、tle.language.pipe 顶层符号 re-export。

**用户侧可见的差异**（API 能力与写法）：

| 差异点 | 原因 |
|--------|------|
| 调度配置经 `tle.dsa.ascend.run_pipeline(scheduler=...)` 这一 Ascend 专属入口提供，公共 `tle.pipe` 签名与 GPU 完全一致 | 硬件要求显式指定同步方向（哪端 set/wait、走哪条流水）；GPU mbarrier 相位机制自动且对称，无需配置入口 |
| readers=（SPMC 多读者）不支持 | 跨核事件为定向单播，一条边固定一个 set 端和一个 wait 端，没有 mbarrier 的到达计数聚合；且 event id 仅 16 个 |
| 无 writer.close，is_closed 恒为编译期 False | 事件只有 0/N 计数语义，没有可被 wait 读出的终止态；EOS 由两侧已知的迭代次数表达 |
| payload 为 tle.dsa.workspace（GM ring buffer），非 SMEM buffered_tensor | UB 核内私有，Cube（L0C 输出）与 Vector（UB 消费）没有双侧共享的 SMEM，GM 是唯一数据汇合点 |
| capacity ∈ [1, 16] | GPU 用相位奇偶复用单个 mbarrier，容量无上限；Ascend 无 phase、每 stage 独占一个 event id，硬件共 16 个（[0,15]） |
| one_shot 要求 capacity=1（GPU 无此限制） | one-shot 边没有 acquire/release 周转，只拥有一个 event；capacity>1 的无反压多 stage 形态在 Ascend 上不支持 |
| 多生产者不支持，仅 SPSC | 事件是计数信号量（set +1 / wait −1），无 mbarrier 到达阈值，多个生产者 commit 同一 ready 事件时 reader 首次 wait 即被放行，等不齐全部贡献；等齐需按生产者扇出事件，id 预算不可行 |

**内部实现差异**（契约语义对齐，底层机制不同）：

| 差异点 | 原因 |
|--------|------|
| 端点为编译期描述符内联展开，pipe 不以 IR op 形式存在（GPU 为 tle.pipe.* 一等 IR op，TTGIR 阶段经 lower-pipe-to-nvws 低降） | 复用现有 hivm 同步原语与 pass 管线，避免为 MVP 新增 pipe 方言与低降 pass |

## 6. 约束说明

- capacity ∈ [1, 16]，一个 kernel 内所有 pipe 的 event id 段互不重叠且总和 ≤ 16。
- ready_sync 与 free_sync 必须方向配对（ready.sender == free.receiver 且 ready.receiver == free.sender）；SyncSpec 中 sender != receiver。

## 7. 用例示例

以 CV mix（D = relu(A @ B) + 100 × 0.01）为例：Cube 做矩阵乘写入 pipe，Vector 读 pipe 做后处理。完整可运行版本见 `python/tutorials/tle/08-cv-mix-pipe.py`。

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle

VEC_ADD_ITERS = tl.constexpr(1000)
VEC_ADD_STEP = tl.constexpr(0.01)


@triton.jit
def _cv_mix_cube_producer(c_writer, a_ptr, b_ptr, stride_am, stride_ak, stride_bk, stride_bn,
                          M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid = pid_m * tl.cdiv(N, BN) + pid_n

    write_slot = c_writer.acquire(pid)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BK)):
        ...  # load A/B 分块并 tl.dot 累加
    tl.store(write_slot.c, acc, mask=mask)
    c_writer.commit(pid)


@triton.jit
def _cv_mix_vector_consumer(c_reader, d_ptr, stride_dm, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid = pid_m * tl.cdiv(N, BN) + pid_n

    result = c_reader.wait(pid)
    c = tl.load(result.slot.c)
    d = tl.maximum(c, 0.0)
    for _ in range(VEC_ADD_ITERS):
        d = d + VEC_ADD_STEP
    tl.store(d_gm, d, mask=mask)
    c_reader.release(pid)


@triton.jit
def cv_mix_pipe_dsa_kernel(a_ptr, b_ptr, d_ptr, workspace_ptr, ...,
                           PIPE_CAPACITY: tl.constexpr):
    workspace_base = workspace_ptr + pid * PIPE_CAPACITY * BM * BN
    c_workspace = tle.dsa.workspace(workspace_base, capacity=PIPE_CAPACITY, shape=[BM, BN], dtype=tl.float32)

    # ready/free 默认标准 cube->vector CV-mix 握手，event id 自动分配
    c_pipe = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="cv_mix_c_pipe", c=c_workspace)
    c_writer = c_pipe.writer()
    c_reader = c_pipe.reader()

    tle.dsa.ascend.run_pipeline([
        (_cv_mix_cube_producer, (c_writer, a_ptr, b_ptr, ...)),
        (_cv_mix_vector_consumer, (c_reader, d_ptr, stride_dm, ...)),
    ])
    # 需要自定义方向/event id 时，把调度器传给编排入口：
    #   tle.dsa.ascend.run_pipeline([...], scheduler=(
    #       ("cv_mix_c_pipe", tle.dsa.ascend.pipe_scheduler(
    #           ready_sync=..., free_sync=..., event_base=0)),
    #   ))


# host 侧 launch
cv_mix_pipe_dsa_kernel[(num_m_blocks, num_n_blocks)](
    a, b, d, workspace, ..., disable_auto_inject_block_sync=True)
```
