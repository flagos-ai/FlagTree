# r001 resource-qualified candidate — fresh three-way measurement

**NUM PASS, independently verified through the default compiler load path in clean processes.** Both scenarios: first 10 independently generated tokens equal freshly generated llama.cpp golden; all logits finite; eager/graph/reset trajectories agree. Every measured Triton process repeats NUM before timing and resets recurrent/KV state before every round.

| Scenario | Generation start | Candidate first 10 = fresh golden | NUM |
| --- | ---: | --- | --- |
| synthetic32 | 0 | [5328, 3165, 264, 220, 16, 15, 15, 15, 36093, 8627] | PASS |
| chinese64 | 16 | [271, 248068, 198, 97237, 96094, 96876, 4960, 98832, 105258, 122826] | PASS |

Host ms per decode-1 includes GPU greedy and one CPU token return. Setup/load/JIT/warmup/state reset/prompt injection/full-logit diagnostics are outside timing; `prepare_input()` is before the llama.cpp start clock. Frozen synthetic32 = 32 steps from token 1; chinese64 = 64 steps with the frozen Chinese prompt. Each backend/scenario has 2 independent processes × 3 measured rounds plus one excluded warmup round. Backend order rotates and reverses; all three backends were remeasured in this session. Uncertainty is 95% t(df=1) from the two process means.

| Backend / scenario | p0 ms | p1 ms | Mean ±95% t ms | Spread ms | Process SD ms | Bandwidth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r001 resource-qualified / synthetic32 | 316.515610451 | 316.533889238 | 316.524749845 ± 0.116127004 | 0.018278787 | 0.012925054 | 9.606454% |
| accepted r000 remeasured / synthetic32 | 258.988498831 | 259.147301961 | 259.067900396 ± 1.008892541 | 0.158803130 | 0.112290770 | 11.737002% |
| fresh llama.cpp / synthetic32 | 35.917971646 | 35.903605677 | 35.910788661 ± 0.091268470 | 0.014365969 | 0.010158274 | 84.673172% |
| r001 resource-qualified / chinese64 | 316.944657655 | 317.092500149 | 317.018578902 ± 0.939258499 | 0.147842494 | 0.104540430 | 9.591490% |
| accepted r000 remeasured / chinese64 | 259.119014018 | 258.966132154 | 259.042573086 ± 0.971274129 | 0.152881864 | 0.108103802 | 11.738149% |
| fresh llama.cpp / chinese64 | 35.841941484 | 35.862448135 | 35.852194810 ± 0.130280853 | 0.020506651 | 0.014500392 | 84.811555% |

| Scenario | Latency reduction vs r000 | TPS gain vs r000 | Paired ratio U95 | ≥5% point / U95 gate | Strict llama win | ≥80% bandwidth |
| --- | ---: | ---: | ---: | --- | --- | --- |
| synthetic32 | -22.178297% | -18.152403% | 1.226092839 | False / False | False | False |

synthetic32: paired r001/r000 ratios [1.222122263651396, 1.2214438924988236], mean 1.221783078 ± 0.004309761; r001 − llama 280.613961183 ± 0.207395474 ms. Strict win requires the paired difference upper95 < 0; the H001 falsifier requires paired latency-ratio upper95 ≤0.95.
| chinese64 | -22.380879% | -18.287889% | 1.232023513 | False / False | False | False |

chinese64: paired r001/r000 ratios [1.2231624871539464, 1.2244554819242897], mean 1.223808985 ± 0.008214528; r001 − llama 281.166384092 ± 0.808977646 ms. Strict win requires the paired difference upper95 < 0; the H001 falsifier requires paired latency-ratio upper95 ≤0.95.

Both scenarios: ≥5% whole-model gain beyond uncertainty = **False**; strict fresh llama win beyond uncertainty = **False**; ≥80% bandwidth = **False**.

Bandwidth = 13,622,248,096 B / (latency_s × 448e9); 80% ⇔ ≤38.008504732 ms/token. Gain means latency reduction: 100 × (1 − r001/r000). TPS gain is separately reported.

The single r001 primary point belongs to `full64-h001-cap96-104-20261116T155144Z`; artifact.json SHA256 `ae96264431d4f13e1918fbd9cbba180ec5c918c5070c82292f1da2726b77bd32`. Candidate loading uses compiler defaults without local native registration. The default encoder is pinned in protocol.json and checked again in the raw audit.

Accepted r000 artifact: `/mnt/home-nas/work/repo/flagtree/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/.local/milestones/m000/revisions/r000/attempts/candidate-r4-fresh-20261116T1023/artifact`; SHA256 `1868b4438b7ca8048f63f2a8b7bec56c4409d6c4587c2f5a7d347ae72b963375`. Its frozen native v4 encoder is registered for the r000 comparison; the encoder pin is in protocol.json. Reference source is the unmodified pinned llama.cpp commit `5ce06fc71f5e938301f7dc51697a59ae308bd6b7`; prebuilt libraries are from `20261105T031000Z-cpp-decode/rebuild-r1/bin`. Source/config/toolchain/library identities and actual runner linkage are checked before and after each reference run.

**Superseded retained history:** `r001-h001-20261116T143230Z` measures the different unqualified artifact `full64-h001-20261116T142200Z` (215.731 / 216.190 ms). Those raw samples, failures, reports and archived figures remain unchanged. They are excluded from this session’s comparisons and replaced in the canonical r001 primary point. See supersession.json; the sealed baseline/r000 iteration history remains intact.

Resource admission evidence belongs to the kernel optimizer’s separate pinned owner snapshot: nine projections 80 registers/6 CTA; entry 8 104/4; Q16 96/5; GLU20 104/4. Driver/cuobjdump agree; spill/local/stack/shared and SASS LDS/STS/LDL/STL/BAR all zero. Resource admission and measured NUM/performance do not declare reviewer acceptance.

Own-context raw audit passed: 1728 measured decode samples; all measured rounds sampled clear. All raw samples, excluded warmups, reset trajectories, sidecars, stdout/stderr, prelaunch/postexit observations and telemetry remain in the packet. GPU work held exclusive `/tmp/flagmega-gpu0.lock`; ordinary timing was never under ncu. OR-Tools remains 9.10.4067.

Retained report-only failures: a report f-string syntax error and three incomplete pending-render attempts (missing renderer dependency, required model_status, then required x field). Their source snapshots and logs remain under `.local/agents/graph_profiler/measure_r001_qualified/` and `.local/reports/r001-qualified-running-20261116T1622Z`, `…1623Z`, `…1623Z-03`. The corrected pending view was synced by `…1623Z-04`. These CPU publication failures did not alter or replace any GPU run or raw sample.

Still open: in-model per-PC independent audit; full-model dynamic NCU/placement gates; K9; K6/K10; B007. Independent reviewer acceptance and ledger append remain pending. m000 is open; the milestone figure has no achieved point. Graph/operator instrumented evidence stays pending.

Raw packet: [r001-qualified-20261116T1618Z](/mnt/home-nas/work/repo/flagtree/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/.local/milestones/m000/revisions/r001/measurements/r001-qualified-20261116T1618Z/).

Reproduce from this work directory: `/home/sunnycase/miniconda3/envs/flagtree/bin/python benchmark/measure_candidate.py --output FRESH --candidate /mnt/home-nas/work/repo/flagtree/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/.local/milestones/m000/revisions/r001/attempts/full64-h001-cap96-104-20261116T155144Z/artifact --baseline /mnt/home-nas/work/repo/flagtree/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/.local/milestones/m000/revisions/r000/attempts/candidate-r4-fresh-20261116T1023/artifact --revision r001 --candidate-status resource-qualified-review-pending --candidate-default-path --baseline-native-file /mnt/home-nas/work/repo/flagtree/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/.local/milestones/m000/revisions/r000/attempts/candidate-r4-fresh-20261116T1023/source/python/tutorials/flagmega/03-qwen3.8-27b-iq4/nvidia-rtx5060ti/agent_optimizations/iq4_streaming/native.py`.


- [performance-vs-iteration](figures/performance-vs-iteration.svg)
- [performance-vs-milestone](figures/performance-vs-milestone.svg)
- [backend-comparison](figures/backend-comparison.svg)
- [graph-trace](figures/graph-trace.svg)
- [operator-latency](figures/operator-latency.svg)
