# GLM-5.3 Flash FP8：H800 四卡 decode

显式 `--arithmetic-profile vllm-0.31-tp4` 已通过完整四卡官方前缀验收：
同一 checkpoint revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`，
initial token 11、45 层、288 experts/top-8、hidden 4096、cache 4096，
两次独立请求复位、各 8 个 greedy token，均逐项等于官方 vLLM 0.31.0：

```text
3847, 284, 1373, 8282, 4218, 3755, 947, 906
```

验收运行 prefix13 使用当时空闲的 3/4/5/6 卡，四个 rank 均为
255 registers/thread、65536 bytes shared、128 bytes ABI stack、零 spill。
logits/状态有限值、序列位置和缓存未写入尾部检查全部通过。
默认 nncase importer 和 strict CPU 验收保留原有门槛；本次通过的是
`official-prefix` 模式，未声称 CPU 或全部 FP32 中间值逐位一致。

已验收源码见 [generated_kernels.py](generated_kernels.py)，SHA256 为
`a4ed588941877a81412a3298e2c74b6c16f4e6e51a3698a36acc2ac7b8113950`；
运行 IR 为 `6d24d1fdbdaa1b0c059fef943ef4243b909e092a5d348a0cbc7e0302ac11ce89`。
从原始 import 经公开入口 fresh 编译，源码与运行版本完全相同。
实际产物还需要 IR、manifest 和 readonly 权重，单个源码文件不能独立运行。
报告、失败过程及复现证据在 `.local/accepted-prefix13/` 和 `.local/iteration-log-zh.md`。

MLA 使用 Q 吸收、单 KV head 的 512 维 latent cache、BF16 online attention
和 V 展开。稀疏索引先稳定压紧再组成 64 项 tile；Q/KV RMS 显式保持连续
8 元素线程分组和全局 partial 树归约。TLE 共享 scatter 的 full-view 误优化
已修复，compute body 内 Pack=0。KDA 保留 raw A_log，在 GPU 上指数化，
使用显式 K 归约布局及 gated RMS 的 sqrt/div；每次请求独立选择 prefill/decode。
mHC 与 FP8 MoE 的数值合同仍由公开 profile 和正常编译流程表达。
合同和回归范围见 [mla-contract.md](mla-contract.md) 与 [kda-contract.md](kda-contract.md)。

最新 fresh 编译为 284.301 秒：load/profile 37.168 秒、compile 219.320 秒、
源码发射 25.429 秒，不含权重发布、CLI 导入及 GPU JIT，仍未达到端到端 4 分钟。
此次只读文件不变的正式内核刷新为 65.422 秒，64 个 readonly 文件的 inode、
大小和 mtime 全部不变；完整设备 JIT 为 285.107 秒。这些计时边界不能混为完整编译。
第二次请求没有状态观测拷贝，四卡每步最慢 rank 的 CUDA event 时间为
137.339–144.125 ms（中位数 141.837 ms），包含状态传递，不代表与 vLLM 的性能对比。

每卡 readonly 权重 80,457,200,384 bytes，external 411,292,769 bytes，
workspace 6,539,776 bytes，最低预算 80,875,032,929 bytes；另需 CUDA 和运行时空间。
TP4 placement 为 `cyx=4x8x16/cbb`，每卡 128 CTA。
当前公开 importer 仍展开各层，没有宣称已用 For 压缩完整模型。

## 复现

在仓库根目录、FlagTree Python 环境下执行。`GLM_CHECKPOINT` 指向原始完整
safetensors checkpoint。需要四张空闲 H800；已验收原生 profile 每卡 rdata 为 80,457,200,384 字节
（约 74.93 GiB）。另需状态、workspace、CUDA 上下文和运行时保留空间。
CPU 参考固定使用 16 个线程，完整模型参考运行需要约 326 GB 主机内存。

```sh
conda activate flagtree
GLM_TUTORIAL=python/tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800
GLM_CHECKPOINT=/path/to/original-checkpoint
GLM_WORK="$GLM_TUTORIAL/.local/reproduce"
mkdir -p "$GLM_WORK"

python -m triton.flagmega import --model "$GLM_CHECKPOINT" \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --full-model --mode decode-1 --numerical-profile nncase \
  --output "$GLM_WORK/imported.py"

python "$GLM_TUTORIAL/compile_tp4.py" --input "$GLM_WORK/imported.py" \
  --checkpoint "$GLM_CHECKPOINT" --trial "$GLM_WORK/native-compile" \
  --arithmetic-profile vllm-0.31-tp4

CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:0 --local_addr=127.0.0.1 "$GLM_TUTORIAL/run_tp4.py" \
  --artifact "$GLM_WORK/native-compile/artifact" --reference "$GLM_TUTORIAL/official-prefix.json" \
  --output "$GLM_WORK/results" --steps 8 --requests 2 --initial-token 11 \
  --acceptance official-prefix
```

目录参数须使用新的输出位置。GPU 编号按实际空闲设备替换。
已有本地 checkpoint 完整字节校验报告时，可向编译命令传入
`--integrity-report /path/to/checkpoint-digests.json`，复用仍匹配文件身份的摘要。
不提供报告时，正常发布流程读取并计算源文件摘要；这部分首次校验时间需单独计入。

`compile_tp4.py` 自动安装 `agent_optimizations` 中已验证的 mHC、GEMV、Boxing 和
归约布局，通过正常模板注册、候选校验和 `write_artifact` 生成产物。默认 compute
warps 为 8；每次 trial 保存分布约束、优化实现及编译时间记录。无需私有 `.local`
代码。核心正确性修复位于正常 FlagMega/TLE 源码及回归测试树。

`run_tp4.py` 使用正常 runtime/prepare/run_into，检查零 spill、有限 logits、
独立 greedy token、全部 KDA 状态、有效 KV/DSA 状态和未写入尾部；每次请求重置状态。
完整缓存的非零检查使用有界分块扫描，避免验收时额外分配整张缓存大小的临时张量。
默认 strict 的参考是 CPU IR 语义；上述命令则使用独立官方报告验收 token。

本例显式原生 profile 从同一 fresh nncase import 构造。若要运行默认 strict
CPU 检查，不设置 `--arithmetic-profile`，并通过 `reference.py` 生成 CPU 参考：

```sh
python "$GLM_TUTORIAL/reference.py" --input "$GLM_WORK/imported.py" \
  --checkpoint "$GLM_CHECKPOINT" --output "$GLM_WORK/cpu-reference" --threads 16 --steps 8
```

增加 `--compile-only` 可只生成已验证 IR 和源码，供检查与 GPU JIT 预检，不加载或
发布权重；这类输出本身不是可运行产物。profile/分布/调度均来自目录内公开 Python
文件，fresh 进程无需私有 .local 扩展。该配置已通过上述官方前缀验收；修改配置后应重新验证。

官方 token 验收在运行命令中增加 `--acceptance official-prefix`，并将 `--reference`
改为 [official-prefix.json](official-prefix.json) 中独立 vLLM 的报告（包含 `versions.vllm="0.31.0"` 和
`requests[].tokens`）。GPU 使用自己的上一步输出推进 greedy 历史，逐步核对所选
前缀；仍检查完整模型配置、零 spill、有限 logits/状态、序列长度和全部缓存尾部。
该模式不要求 CPU logits 或中间状态逐元素接近。默认 `--acceptance strict`
保留原 CPU 数值检查；两种模式在报告中明确区分。上述原生 profile 已通过该模式。

## 历史计时边界

以下是较早配置的测量，当前原生 profile 的计时见前文。
修复 query 缩放及同步规划后的公开入口完整编译发布为 **228.48 秒**，
其中 IR 加载 17.91 秒、编译 169.45 秒、产物生成 38.72 秒。计时复用已完成的
checkpoint 源文件校验，不包含 GPU JIT、权重加载和推理。该次正常编译产物的
独立 GPU JIT 为 222.49 秒；实际四卡为
237 registers/thread、8192 bytes shared memory、零 spill。
128 bytes call-ABI stack 不计为寄存器 spill。
正常运行时复用未变源文件的已验证版本；复制、替换或修改后重新校验，显式产物
审计仍读取全部字节。完整模型的源文件/物理块校验阶段由 35.48 秒降至 7.48 秒，
不含 H2D、GPU JIT 和推理。通信句柄改为运行时参数，并以独立编译期 mode
区分 launcher ABI，使设备编译结果可跨进程地址复用；上述四卡压力回归的实际
prepare 为 0.84–1.19 秒。该次完整加载仍为 362.11–363.74 秒，尚未证明整体
加载提速；校验阶段和 JIT 复用的收益不能直接当作完整加载收益。
当前原生 profile 的官方 greedy 验收已通过；尚未完成与原生 vLLM 的性能对照，不发布相对性能收益。

权重、图和物理缓冲区 ABI 不变的内核迭代，可调用
`triton.flagmega.artifacts.refresh_artifact_codegen(artifact_path)`。
它通过正式模板重新发射源码，以内容摘要命名并原子替换 manifest，保留旧源码。
若要调整已选内核的物理参数，可传入 `kernel_rewrite` 回调，返回完整的既有
KernelDefinition 序列；只允许修改 microkernel 调度和 semantic_attrs，
不能改参数、结果、effects、workspace、transfer ABI、图或常量配方。
此时会正式验证并发射新的 IR。图、常量或内存布局变化仍需正常重新编译发布。

完整产物仅刷新源码实测 45.17 秒；本轮连同 o_proj 调度 IR 更新为 66.80 秒。
两次都未重新打包或 hash 约 605 GiB readonly，64 个 readonly 文件的 inode、
大小、mtime 不变。旧 payload 摘要继续保留，运行时映射和显式产物审计仍执行
原有字节校验；刷新不会为损坏的数据生成新摘要。以上时间不包含 GPU JIT、
权重加载或推理。
