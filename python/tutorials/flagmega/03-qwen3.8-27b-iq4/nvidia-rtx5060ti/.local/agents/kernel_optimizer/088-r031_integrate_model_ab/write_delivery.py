from pathlib import Path
import json,hashlib,subprocess,gzip
from datetime import datetime,timezone
A=Path(__file__).resolve().parent;W=A.parents[3]
def read(p):return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(p.absolute()),nbytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=str)+'\n')
s=read(A/'ab-statistics.json');g=read(A/'model-gates-attempt2/status.json');rows=read(A/'integrated-row-verification-attempt2/status.json')
assert g['status']==rows['status']=='PASS' and s['status']=='PASS' and s['full_paired_processes']==3 and s['model_claim'] is not None
closure=read(A/'source-closure.json');resources=read(A/'model-gates-attempt2/affected-resources.json');unique={r['name']:r for r in resources}
for r in unique.values():assert r['metadata']['ptxas_spill_load_bytes']==r['metadata']['ptxas_spill_store_bytes']==0
source=A/'source-tree/python/triton/flagmega'
up27=A.parent/'085-r027_register_40_no_spill/source-tree/python/triton/flagmega'
up17=A.parent/'077-r006_r17_gdn_prep/source-tree/python/triton/flagmega'
subjects={}
for name,implementation,bodies,up in [('R27','tir.iq4_q8_glu.r006_glu_transport_t3',['codegen/triton/kernels/iq4_q8_glu/r006_glu_transport_t3.py.jinja','codegen/triton/kernels/_iq4_streaming.py.jinja'],up27),('R17','tir.gdn_recurrent.r017_split4lut',['codegen/triton/kernels/gdn_recurrent/r017_split4lut.py.jinja'],up17)]:
 # Template filename is recorded by the native catalog; discover without inferring a body.
 bodies=[rel for rel in bodies if (source/rel).exists()]
 if name=='R27':
  bodies += [str(p.relative_to(source)) for p in (source/'codegen/triton/kernels').glob('*transport*.jinja')]
 bound=[]
 for rel in sorted(set(bodies)):
  assert (source/rel).read_bytes()==(up/rel).read_bytes();bound.append(pin(source/rel))
 subject=dict(schema='flagmega.r031.integrated-subject/v1',name=name,implementation=implementation,body_sources=bound,upstream_qualification=closure['r27_upstream_receipt' if name=='R27' else 'r17_upstream_receipt'],model_resource_evidence=pin(A/'model-gates-attempt2/affected-resources.json'),MODEL_TOKEN10=pin(A/'model-gates-attempt2/status.json'),integrated_row_verification=pin(A/'integrated-row-verification-attempt2/status.json'),fallback='r004 frozen package-attempt1',standalone_qualification_transferred_as_model_qualification=False)
 if name=='R27':subject.update(consumer_barrier_abi='H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2',grid=17408,threads=160,registers=40,usable_shared_bytes=897,ptxas_options='--minnctapersm=9 --opt-level=2',R16_integrated=False)
 else:subject.update(native_model_schema='r031.gdn.native-model/v1',phase_order=['partials','prep','core','y'],arena_bytes=56832)
 path=A/f'r031-{name.lower()}-subject-receipt.json';save(path,subject);subjects[name]=pin(path)
all_raw=[]
for root in sorted(A.glob('ab-p[12345]')):
 for p in sorted(root.glob('*/round-*/*/raw.json')):all_raw.append(dict(file=pin(p),data=read(p)))
failures=[]
for p in sorted(A.glob('ab-p[12345]/status.json')):
 r=read(p)
 if r['status']=='FAILED':failures.append(dict(file=pin(p),status=r))
failures += [dict(stage='integrated-row-verification-attempt1',reason='Row observer passed local dynamic positions to a root BoundProgram; corrected to full external ABI without kernel/validator changes',status=read(A/'integrated-row-verification/status.json'),log=pin(A/'verify-rows.log')),dict(stage='build-attempt1',reason='Missing inherited R27 typed 16-byte weight alignment while combining R17 support; restored without validator changes',log=pin(A/'build-attempt1.log')),dict(stage='model-gates-attempt1',reason='Resource observer queried n_regs before handle initialization; observer corrected, resource validator unchanged',log=pin(A/'model-gates-attempt1.log'))]
receipt=dict(schema='flagmega.r031.model-integration-ab/v1',created_utc=datetime.now(timezone.utc).isoformat(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),subjects=subjects,model_artifact=pin(A/'build-attempt2/artifact/artifact.json'),source_closure=pin(A/'source-closure.json'),integration_patch=pin(A/'integration-source.patch'),cpu_audit=pin(A/'cpu-audit.json'),fallback_machine_audit=pin(A/'fallback-machine-audit.json'),fallback_loaded_verification=pin(A/'fallback-loaded-verification.json'),frozen_body_fallback_unchanged=True,MODEL_TOKEN10=g['MODEL_TOKEN10'],integrated_row_verification=dict(status=rows,per_row_sha256_files=[pin(p) for p in sorted((A/'integrated-row-verification-attempt2').glob('*.gz'))],scope='3 seeded autonomous input tokens, every one of 64 GLU and 48 GDN model call sites; all rows of published GLU, qn, kn, beta, decay_log, core, state, y compared to the frozen r004 implementation'),actual_loaded_resources=unique,measurement_protocol=dict(boundary='ordinary CUDA graph replay + GPU greedy + sync/CPU token return; full BF16->Q8 prep included',arms=['r004','r031','llama'],same_process=True,independent_complete_processes=3,rounds_per_scenario_per_process=3,process_mean_as_independent_unit=True,Student_t_df=2,t975=4.302652729696142,zero_exclusion=True,missing_failed_measurements=None,profiler_off=True,flock='/tmp/flagmega-gpu0.lock',same_process_llama_host_bridge=pin(A/'llama-bridge-build.json')),timing_samples=all_raw,process_list=s['processes'],statistics=s['scenarios'],all_attempts_zero_exclusion_audit=True,failed_processes_and_attempts=failures,excluded_samples=[],excluded_processes=[],model_claim=s['model_claim'],no_new_grid_arrival_counter=True,no_generated_source_ptx_cubin_patch=True,R16_integrated=False,weight_reuse=pin(A/'weight-deduplication.json'),gpu_lease_observation=pin(A/'gpu-lease-observation.json'),monitoring=[pin(A/'ab-nvml-5hz.jsonl'),pin(A/'ab-recovery-nvml-5hz.jsonl')],skills=['ai-compile-dev'])
save(A/'integration-ab-receipt.json',receipt)
# Pin the actual audit data as well as keeping every incomplete process in the
# receipt. Missing rounds remain missing; they are never substituted or dropped.
receipt.update(statistics_file=pin(A/'ab-statistics.json'),
               all_attempts_zero_exclusion_audit={name:t['all_attempts_zero_exclusion_audit'] for name,t in s['scenarios'].items()},
               all_available_process_mean_statistics={name:t['all_available_process_mean_statistics'] for name,t in s['scenarios'].items()},
               protected_evidence_verification=pin(A/'protected-verification.json'),
               protected_before_manifest=pin(A/'protected-pins-before.json'),
               MODEL_TOKEN10_evidence=pin(A/'model-gates-attempt2/status.json'))
assert read(A/'protected-verification.json')['status']=='PASS'
save(A/'integration-ab-receipt.json',receipt)
def interval(r):return f"{r['mean']:+.6f} [{r['low95']:+.6f}, {r['high95']:+.6f}]"
x=s['scenarios']['synthetic32'];y=s['scenarios']['chinese64']
lines=[
'MODEL_TOKEN10：synthetic32 PASS；chinese64 PASS（第 7 个 token = 4960）；三 seed 全行逐位 PASS。',
f"整模型实测：{x['arms']['r031']['mean']:.6f} / {y['arms']['r031']['mean']:.6f} ms/token；PID {','.join(str(r['pid']) for r in x['pid_pairs'])} 同进程配对，95% t(df=2)。",
f"严格快于 llama.cpp：两场景 {'PASS' if all(t['strictly_faster_than_fresh_llama'] and t['strictly_faster_than_fixed_llama'] for t in (x,y)) else '未全部通过'}；新配对差 {interval(x['paired_differences_ms']['llama'])} / {interval(y['paired_differences_ms']['llama'])} ms。",
f"decode 带宽：448 spec {x['percent_of_448']:.6f}% / {y['percent_of_448']:.6f}%；428.165 可达 {x['percent_of_428_165']:.6f}% / {y['percent_of_428_165']:.6f}%。",
f"与同 PID r004 的配对差：{interval(x['paired_differences_ms']['r004'])} / {interval(y['paired_differences_ms']['r004'])} ms。",
'下一步：保留独立 r031/R27+R17 主体与收据；可显式回退冻结 r004_column_v6；本轮不叠加 R16。',
'',
'本轮只接入 R27 GLU megakernel 与 R17 split4lut PREP。R27 与 R16 替换同一 GLU，R16 未被选择；−5.369834 µs 未计作节省。新模型策略为 flagmega.r031-model-policy/v1，GDN 原生 ABI 为 r031.gdn.native-model/v1，包版本 v11-native-model-r031。r004 v6 策略和已接受模板保持逐字节相同。冻结回退的 40 个函数，重新编译后可执行、常量和启动 ELF 节全部相同；34 个完整 cubin SHA 仅因 overlay 的调试源路径节不同而变化，差异原文保存在 fallback-machine-audit.json，没有修改旧 cubin 或补丁产物。算法选择只有两个可复用 GLU 定义及一个 GDN 定义发生变化；新资源通过原生 planner 重新分配工作区地址。没有产物补丁或新的全 grid 到达计数器。',
'',
'实测和统计：',
'',
'| 场景 | r031 ms/token（95% CI） | 同 PID r004 | 同 PID llama | 权重 GB/s | 448 % | 428.165 % |',
'|---|---:|---:|---:|---:|---:|---:|']
for name,t in s['scenarios'].items():
 a=t['arms']['r031'];lines.append(f"| {name} | {a['mean']:.6f} [{a['low95']:.6f}, {a['high95']:.6f}] | {t['arms']['r004']['mean']:.6f} | {t['arms']['llama']['mean']:.6f} | {t['achieved_weight_GB_s']:.6f} | {t['percent_of_448']:.6f} | {t['percent_of_428_165']:.6f} |")
lines += ['', '三轮进程均值作为独立单位，t₀.₉₇₅,₂=4.302652729696142；不把 token 或轮次当独立进程。三个主体均在同一 PID 测量，次序按进程和轮次轮转。llama 使用既定 Session/run 的逐字节宿主代码，追加 C ABI 后链接同一 pinned shared libraries；模型/上下文参数、数值前后门禁、计时边界保持原协议。每轮完整准备、热身、state reset、prompt 注入和 NUM 检查在计时外；计时包括完整模型量化、全部投影、GPU greedy 与 CPU token 返回。全程无 ncu。', '', '| 场景 / PID | r004 | r031 | llama | r031−r004 | r031−llama |', '|---|---:|---:|---:|---:|---:|']
for name,t in s['scenarios'].items():
 for r in t['pid_pairs']:
  m=r['means_ms'];lines.append(f"| {name} / {r['pid']} | {m['r004']:.6f} | {m['r031']:.6f} | {m['llama']:.6f} | {m['r031']-m['r004']:+.6f} | {m['r031']-m['llama']:+.6f} |")
lines += ['', '固定历史接受值的差单独列出，避免把固定参考差冒充新同进程配对：', '', '| 场景 | r031−历史 llama（95% t 区间） | r031−历史 r004（95% t 区间） | 实测−条件外推 |', '|---|---:|---:|---:|']
for name,t in s['scenarios'].items():lines.append(f"| {name} | {interval(t['fixed_historical_reference_differences_ms']['llama'])} | {interval(t['fixed_historical_reference_differences_ms']['r004'])} | {t['measured_minus_conditional_prediction_ms']:+.6f} ms |")
lines += ['', '带宽分子沿用 r004 的 13,622,248,096 mandatory weight bytes/token；GB/s = bytes/(ms×10⁶)。分别除以 448 和 428.165，不把 profiler DRAM counters 或 kernel sum 作为整模型计时。条件外推 34.729385/34.804708 仍不是实测，本轮用实测替代。实测未完全兑现独立算子外推，不能在缺少整模型内分项测量时把偏差强行归因到某个算子；本轮整模型结论以实测配对为准。', '', '正确性及资源：两个场景完整 32/64 步自主 token feedback，eager/graph TOKEN10 和完整 endpoint SHA256/finite/guards 一致。三 seed 20261120/1/37 从 reset 状态各选一个确定性输入 token，在两个模型中逐调用核对全部行：64×17408 GLU BF16 标量行；48 组 qn/kn 16 行、beta/decay_log 48 行、core/y 48 行、state 6144 行。中间状态按 typed caller/pool ABI 地址读取，逐行 SHA256 原文存于 gzip JSONL；不取样、不用容差。R27 两个模型 wrapper 均为 40 registers、160 threads、897 B static shared、grid 17408、ptxas 0/0；R17 四阶段也全部 0/0。CPU 四个负向 ABI 检查通过。', '', '零排除和失败：首次构建缺少继承的 R27 对齐声明、首次资源 observer 初始化顺序错误，以及逐行 observer 使用了错误的调用签名，均保留原失败日志；只修正接入/观察脚本，不放宽任何 kernel/ABI 验证规则。权重复用曾误用跨目录软链接，原包路径校验拒绝；已改硬链接，权重字节及节摘要不变。两个受影响进程的失败记录和已有普通计时全部保留。没有把失败或缺失值写成零，也没有删除计时样本。正式三进程区间来自修复后的三个完整配对进程；不完整进程的全部已采样均值、轮数和缺失配对在 ab-statistics.json 的 all_attempts_zero_exclusion_audit 中单列，不能伪装成完成的模型配对进程。', '', '大体积 llama NUM/state 临时快照在原协议完成有限性/逐位检查后保留 SHA256，计时 raw/result 永久保留；没有重复保留数十 GB 同内容快照。源码闭包、主体身份、逐行 hashes、全部 raw timing/PID、失败与 model_claim 均在 integration-ab-receipt.json；两个新主体分别有独立 subject receipt。']
lines += ['', '包含失败进程已有样本的零剔除敏感性审计：synthetic32 加入 PID 3962272（r004/r031/llama 已完成轮数 3/2/3），按各进程已有轮次均值计算；chinese64 在失败进程尚无样本。此表保留不完整进程的影响，不代替上面的三个完整平衡配对进程主判据；仍使用更保守的 t(df=2)，缺失轮次不填补。', '', '| 场景 | 有数据的 PID 数 | r031 ms/token | r031−r004（区间） | r031−llama（区间） |', '|---|---:|---:|---:|---:|']
for name,t in s['scenarios'].items():
 a=t['all_available_process_mean_statistics'];lines.append(f"| {name} | {len(a['pids'])} | {a['arms']['r031']['mean']:.6f} | {interval(a['paired_process_mean_differences']['r004'])} | {interval(a['paired_process_mean_differences']['llama'])} |")
lines += ['', '交付复核 delivery-verification.json 已从原始 token 计时重算进程均值与 Student-t 区间，并流式重比六份全行 SHA256 文件。protected-verification.json 核对 97,909 个受保护文件，变化数为 0。独立分支为 r031/r27-r17-model-ab；可复现命令和冻结回退入口见 RUN.md。']
(A/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('DELIVERY_WRITTEN',flush=True)
