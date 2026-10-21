# 长上下文 attention 的并行与加载优化

优化目标：降低 BI-V150 的 Qwen3-1.7B batch=1 BF16 长上下文 decode，
使对原生 vLLM FULL CUDA Graph 的加速接近短上下文，并保留原计算顺序。

原性能根因位置是 `python/triton/flagmega/codegen/cpp/templates/kernels/paged_attention_partial/_function.cpp.jinja` 的 local-head 循环。
4 个本地 query heads 串行执行，每个 16-token 块重新读取 K/V，GQA partners 对同一个 KV head 重复读取；
softmax 概率计算与求和集中在一个线程。28 层、4 个 context shards 下，循环成本随上下文增长。
优化前固定 context=2048 的完整 C++ 模型约 18.43ms，vLLM 整步 19.881ms，比原生 13.657ms 慢。

优化位置是 `renderer.py::_select_attention_staging` 与 `paged_attention_partial/_staged.cpp.jinja`。
四组线程同时处理四个 query heads，GQA partners 合作以 16-byte BF16 transactions 加载同一 KV tile；
每次加载 64 tokens，共享行增加 32 BF16 padding。QK 的 16-lane 分解重建原有 64-lane dot tree，不变 query 每个 head group 只读一次并保留在寄存器。
四次 16-token online softmax 更新、概率顺序求和与八条 value streams 的累加次序保持原计算契约。
value 临时存储用两个交替槽，前一槽的读者在两次子块之后重用该槽前已由下一次 value barrier 完成。
CTA 在新 KV batch 与 head group 之前均同步，GQA partner 的共享 K/V 有明确写后读屏障。
内存需求 116992 bytes，复用现有 128KiB shared arena，不新增超预算存储。

性能优化引入的两类数值回归及因果修复：

- 首个候选把四次 16-token softmax 合并为一次 64-token warp sum。这改变 FP32 reduction 与 online state 更新次序，
  BF16 层间舍入放大差异。四个正式 greedy workload 虽一致，但 logprobs/min_tokens 的第 13 个 token 不一致。
  最终恢复每次 16-token 的状态更新，保留 64-token 的 cooperative KV staging 和四 head 并行。
- value 加载提前后，`float product=p*v; if(valid) weighted+=product` 与分离 p/v 寄存器改变了 compiler FMA contraction。
  attention max/sum 相同而 accumulator 差几 ULP，完整 logits 在 context=17 开始不一致。
  最终对 value 加权、accumulator rescale 和 running_sum rescale 显式使用 `__builtin_fmaf`，
  在源头固定运算语义，继续保留加载提前和交替槽优化。

可复现的 failing input：`test_paged_attention_staging.py` 中 seed=41、context=33、随机 BF16 Q/K/V、
非连续 block table。显式 FMA 前 14 个 bit-equality GPU cases 失败，显式 FMA 后全部通过。
最终还将每个 head 的不变 query 提前读取到寄存器，减少长 context 的全局访问；隔离 1–2048 context 逐位相同。
`rejected-warp-softmax/` 保存生成差异和 contraction 的失败日志；`rejected-warp-softmax/tests-attention-bitexact.log.gz`
保留 fail-before，`tests-attention-explicit-fma.log.gz` 保留 pass-after。

最终验收：126 个 C++/tutorial 回归通过；GPU attention 包括 1/17/31/32/33/127/255/256/511/1024/2048 context、
分页 clamp、非连续物理页、+2-byte 未对齐 KV 地址及空 context shard，FP32 partial outputs 与原模板逐位相同，
另有独立 softmax oracle。完整模型连续 2112 步，每步 151936 个 logits 与优化前 artifact 逐位一致，
max_abs=0；命令及原始日志见 `validate_attention_logits.py` 和 `validate-final-bitexact.log.gz`。

正式 3×5 对照为 144 次请求（含 warmups）/9216 个 output tokens，完整 native tokens 全部一致。
32/128/1024/2048 的 C++ 整步中位数是 7.236/7.303/8.304/9.419ms；
原生是 11.815/11.909/12.837/13.651ms，加速 1.63×/1.63×/1.55×/1.45×。
1024/2048 加速比是短上下文均值的 94.7%/88.8%，差距约 5%/11%，并未宣称完全相同。
优化前正式图表与 source/library identity 保存在 `before-long-attention/`，最终 identity 在 `provenance.json`。

复现完整模型逐位验收：

```sh
CUDA_VISIBLE_DEVICES=8 python figures/data/validate_attention_logits.py /path/to/before-artifact /path/to/final-artifact
```

复现实际请求：

```sh
CUDA_VISIBLE_DEVICES=8 python reproduce_flagos.py --checkpoint /root/models/Qwen3-1.7B --artifact /path/to/final-artifact --cuda-graph --rounds 3 --repeats 5 --results /path/to/fresh-results
```

选择范围由 ABI/geometry 证明限定为 `decode_t16`、128 head dimensions、GQA=2、单 query、连续 BF16 KV，
1024 CTA threads 与 64-lane warp。其他 geometry 继续有各自一般模板；没有关闭优化、跳过层或使用原生 decode fallback。
现有原生 prefill、page allocation、sampler 和输入融合保持，decode KV copy bytes 为 0。
修改只在 FlagMega C++ codegen 与 tutorial；Triton/MLIR lowering、其他 vendor backend 与第三方依赖未改变。
