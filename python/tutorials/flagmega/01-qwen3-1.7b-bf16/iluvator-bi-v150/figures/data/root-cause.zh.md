# 单请求 decode 输入准备融合（长上下文优化前的历史记录）

目标：降低 C++ vLLM 集成中非模型计算的剩余时间，保留最终 C++ artifact、原生调度与采样。

根因位置：`/root/repos/vllm-plugin-FL/vllm_fl/worker/model_runner.py::ModelRunnerFL._prepare_inputs`。单请求、单 query、无 speculation 时仍上传 req_indices、query_pos、num_scheduled_tokens 等小张量，再执行 GPU index、cast、add、copy、slot mapping、logits_indices 等独立算子。原生与 C++ 都付出约 2.51 ms 的提交成本；这也使约 6.53 ms 的 C++ graph 的请求 decode 接近 10 ms。这里不能把残差差值当作独立 CPU 时间，因为异步调度使 CPU/GPU 工作重叠。

修改：`decode_inputs.py::FusedDecodeInputs.prepare` 将 scheduler 的 position、token 来源选择、discard 标志及最新物理 page table 合并到一个 pinned CPU buffer，异步上传一次，`prepare_decode_inputs` 一次 kernel 写回原生各 buffer。异步模式仍从上步 sampler 的 GPU tensor 取 token；每步更新当前 page table，不假定物理页连续，不借用 native decode 输出。slot mapping 未用部分仍为 PAD_SLOT_ID=-1。原生 synchronize_input_prep 的 event 继续保护 pinned buffer 的复用。

选择边界：仅原有 batch=1 text paged-attention ABI 的 decode 使用融合；prefill 使用原生准备，包括 one-token prompt。增加 phase 检查避免以 token 数误判 prefill。其他 KV 几何、hybrid、CP、speculation、prompt embeds 不在已支持 ABI 中，安装时明确拒绝。

性能 fail-before/pass-after：同一 GPU8、相同 /tmp/r33_formal_tail_gig_aligned、32-token prompt、输出64、warmup1、repeat3。`benchmark_graph_boundary.py --variant flagmega --reference after.native.boundary.json --require-outside-no-worse` 在 before adapter 上退出1：graph 外残差3.418ms > 原生扣除 LM head 后的2.385ms。after 退出0：graph 外1.117ms、graph6.524ms、engine.step7.641ms。实际 graph 未减小、未省略模型层。原始 JSON 与日志在本目录。

独立 GPU contract 回归测试覆盖同步/异步 token 来源、31/255/256/511/512/3071 positions、非连续物理 page table、原生 GPU/CPU buffer 与 padding、下一请求 prefill 判别。完整请求检查覆盖32/128/1024/2048 context、连续请求及全部输出 token。相邻采样检查覆盖 one-token prefill、255 token 换页、penalties、logprobs/min_tokens、allowed_token_ids、seeded random；每种重复两次，原生/C++所有token一致。无 compiler、native vLLM、FlagGems 或 plugin 源码改动，因此其他 backend 的 lowering 不受影响。

正式图表：3轮×5次请求×4场景×2实现，共144次请求、9216个输出token；所有native/C++ token一致。32/128输入的C++ decode中位数7.567/8.203ms，native11.824/11.901ms，1.56×/1.45×。1024/2048输入仍慢于native。完整tutorial pytest共32项通过。

后续 attention 优化与最终数据见 `attention-root-cause.zh.md` 和 `summary.json`。
