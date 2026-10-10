# Qwen3-1.7B BF16 / Iluvatar BI-V150

本教程在**一张 BI-V150** 上使用 FlagMega 编译完整 28 层 Qwen3-1.7B decode。
按照本任务确认的比较口径，参考实现是 **vLLM 0.24.0 + FlagOS/FlagGems 插件**。
vLLM 负责原生 prefill、调度、页分配和采样；FlagMega 执行每个生成步的完整模型。
4×4 mesh 表示16个 CTA，不是16张GPU。

## 环境与正确性前提

验证环境为 COREX 4.5.0.20260509、PyTorch 2.10.0+corex.4.5.0.20260509、Python 3.12。
使用本 checkout 的可编辑 FlagTree，不能用上游 NVIDIA Triton/PyTorch 替换。
本次源码版本：

| 项目 | commit |
|---|---|
| vLLM | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| vllm-plugin-FL | `ce3a43b28423bad54f832bf3b8a347a47244bcea` |
| FlagGems | `b550ddb996939ee1a30ba4a4b8f8b2d0556703a3`，加下述补丁 |
| Qwen3-1.7B checkpoint | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

本 checkout 的 Iluvatar 编译器包含两个必要修复：

- grid barrier 在所有生产线程完成 device fence 后才报告 CTA 到达，确保跨 SM 的数据可见；
- MMA reduction 循环优化正确保留 accumulator 初值的逐轮缩放。

编译器修改后运行仓库本地脚本 `./build-iluvator.sh` 重建原生扩展。
对未修复的上述 FlagGems checkout，应用
[patches/flaggems-correctness.patch](patches/flaggems-correctness.patch)：

```bash
git -C /path/to/FlagGems apply --check "$TUTORIAL/patches/flaggems-correctness.patch"
git -C /path/to/FlagGems apply "$TUTORIAL/patches/flaggems-correctness.patch"
```

补丁修复 empty unique 和 inplace RoPE 跨 warp 的 partner 读取竞态。
已经有等价修复的 checkout 不应重复应用。双方比较必须使用同一份修复后的依赖。
本环境还按已有部署保留 `sort,sort_stable` 的 COREX 实现；双方配置相同。

## 从原始模型复现

以下命令在 FlagTree 根目录运行。`vllm-iluvatar` 是本机已准备的 FlagOS 环境；
其他机器需提供同等依赖。先用 `ixsmi` 选择空闲GPU。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm-iluvatar
export TUTORIAL="$PWD/python/tutorials/flagmega/01-qwen3-1.7b-bf16/iluvator-bi-v150"
export CHECKPOINT=/root/models/Qwen3-1.7B
export CUDA_VISIBLE_DEVICES=10
export CUDA_MODULE_LOADING=0
export PYTHONPATH="$TUTORIAL${PYTHONPATH:+:$PYTHONPATH}"

python "$TUTORIAL/reproduce_flagos.py" \
  --checkpoint "$CHECKPOINT" \
  --work-dir "$TUTORIAL/.local/fresh-build" \
  --results "$TUTORIAL/.local/fresh-results" \
  --rounds 3 --repeats 3
```

使用新的 work/results 目录。该命令从 checkpoint 构建严格 FlagOS 数值 profile，验证
artifact，再交替执行 native→FlagMega、FlagMega→native、native→FlagMega 三轮比较。
每轮旋转场景顺序，每场景一次预热、三次测量。所有预热和测量的完整 greedy token
序列都必须与原生参考相同，任何失败会阻止结果发布。GPU原生编译和权重载入不计入
请求时间。

若已有通过验证的产物，可用 `--artifact /path/to/serving/artifact` 替代 `--work-dir`。
产物必须包含 `final.py`、manifest、rdata 和生成源码；单独的
[generated_kernels.py](generated_kernels.py) 仅是已验证源码参考，不包含权重。

只编译模型：

```bash
python "$TUTORIAL/optimize.py" --checkpoint "$CHECKPOINT" \
  --work-dir "$TUTORIAL/.local/model-build" --profile serving \
  --numerical-profile vllm-flagos-eager-bf16
python -m triton.flagmega artifact verify "$TUTORIAL/.local/model-build/serving/artifact"
```

FlagOS profile明确保留独立RMSNorm乘weight前的BF16舍入、融合残差的FP32归一化、
matmul输出以及QKV合并后的BF16边界。修改数值profile时必须从原始checkpoint重编译；
后续IR的普通resume不会重新选择分布/TIR。历史 `vllm-493bd8323-inductor-level3` profile
仍由 `optimize.py` 支持，但不用于本次FlagOS结果。

## 测量边界与优化

固定 BF16、batch/concurrency=1、3072最大上下文、256-token KV page、16 pages；
关闭双方prefix cache，独立请求使用32/128/1024/2048-token prompt，各生成64 tokens。
默认基线为 eager；原生 FlagOS 还可以用 vLLM FULL CUDA Graph 做独立对照。测量包括
scheduler、forward、sampler 和输出处理。decode、TTFT、完整请求和P95分别报告，不能
把设备kernel时间当作请求延迟。

### 原生 FlagOS CUDA Graph 对照

当前 vLLM 0.24.0 + FlagOS plugin 不使用旧版 vendor 镜像中的
`VLLM_ENFORCE_CUDA_GRAPH` 环境变量。benchmark 通过 vLLM 配置显式选择
`mode=0`（不走当前不兼容的 Inductor 路径）、`cudagraph_mode=FULL` 和
`cudagraph_capture_sizes=[1]`；同时将 `enforce_eager` 设为 `False`。该模式只对
原生 FlagOS 变体开放。FlagMega worker extension 仍保持 eager 合同，因为它在 vLLM
初始化后的 RPC 阶段安装，不能把已完成的 native graph capture 自动替换成 FlagMega
decode 图。

同一张 BI-V150、同一 Qwen3-1.7B checkpoint 的 2026-09-29 对照（1 次 warmup、3
次测量、每次生成64 tokens）如下；完整原始 JSON/log 保存在
`.local/perf-iteration/flagos-cuda-graph-20260929/`：

| prompt tokens | eager decode median | CUDA Graph decode median | decode speedup | eager request median | CUDA Graph request median |
|---:|---:|---:|---:|---:|---:|
| 32 | 68.824 ms | 11.822 ms | 5.82× | 4422.8 ms | 751.8 ms |
| 128 | 68.233 ms | 11.841 ms | 5.76× | 4386.4 ms | 763.7 ms |
| 1024 | 69.192 ms | 12.538 ms | 5.52× | 4443.7 ms | 828.4 ms |
| 2048 | 68.931 ms | 13.124 ms | 5.25× | 4517.1 ms | 1038.9 ms |

CUDA Graph 启动日志出现 `Capturing CUDA graphs (mixed prefill-decode, FULL)`，四个
prompt 的完整 greedy token 序列与 eager reference 一致。该结果证明 BI-V150 上原生
FlagOS graph 路径可用；它不等同于 FlagMega artifact 已完成 graph 集成验收。

工作负载选择在 [local_optimizations/](local_optimizations/) 内：

- 为MLP down projection选择能充分使用线程的tile，消除原8×16小块导致的大量串行归约；
- QKV按packed weight的物理顺序读取，在FP32中积累，保留最后的输出舍入；
- QKV GEMV在BI-V150上采用`block_k=64`、32 warps，减少K归约的CTA同步；其他GEMV保留已验收的大tile配置；
- 使用 logits-only serving ABI，由vLLM统一采样。

[flagos_adapter.py](flagos_adapter.py) 通过vLLM worker extension API安装。
原生TRITON_ATTN的KV读写接收runtime stride，因此可以共享
`[page, layer, KV, token, head, dimension]` 存储的逐层视图；decode不复制KV，
不执行参考模型forward，不按结果切换回原生模型。

## 独立 chat CLI

```bash
python -m triton.flagmega.serving.chat_cli \
  --artifact "$TUTORIAL/.local/model-build/serving/artifact" \
  --checkpoint "$CHECKPOINT" --ctx-size 3072 --n-predict 64 \
  --chat-template-kwargs '{"enable_thinking": false}' --prompt '请用一句话介绍你自己。'
```

独立CLI不依赖vLLM forward。它的prefill仍是逐token扫描，与上面的原生batched prefill
有不同的TTFT和完整请求边界；本次vLLM集成性能表不代表该路径的prefill速度。

## 结果与原始证据

最终表格与图由 `render_flagos_results.py` 从通过完整token检查的原始数据生成。
每份结果包含模型文件hash、编译器hash、生成源码hash、依赖commit/diff hash、实际
FlagMega调用次数和全部时间样本。Trial111 的三轮交替验收结果如下：

| prompt tokens | FlagOS decode | FlagMega decode | FlagMega request | request speedup |
|---:|---:|---:|---:|---:|
| 32 | 69.88 ms | 20.65 ms | 1344.7 ms | 3.34× |
| 128 | 69.58 ms | 21.99 ms | 1347.0 ms | 3.32× |
| 1024 | 69.54 ms | 29.54 ms | 1924.5 ms | 2.33× |
| 2048 | 69.65 ms | 38.13 ms | 2599.8 ms | 1.76× |

中文逐轮记录在
[.local/perf-iteration/ITERATION.md](.local/perf-iteration/ITERATION.md)，包括所有失败试验
及被后续证据推翻的结论。`.local/` 为本机实验目录，不是复现脚本的输入依赖。
