from pathlib import Path
import json,math,statistics,hashlib
A=Path(__file__).resolve().parent
T2=4.302652729696142
WEIGHTS=13622248096
REF={'synthetic32':dict(r004=36.959303,llama=35.928148,prediction=34.729385),'chinese64':dict(r004=37.034625,llama=35.854791,prediction=34.804708)}
def stats(vals,t=T2):
 mu=statistics.mean(vals);sd=statistics.stdev(vals);h=t*sd/math.sqrt(len(vals));return dict(values=vals,n=len(vals),df=len(vals)-1,mean=mu,sd=sd,low95=mu-h,high95=mu+h,t975=t,unit='independent process mean')
processes=[];completed=[];all_samples=[];failed=[]
for i in (1,2,3,4,5):
 root=A/f'ab-p{i}';p=root/'status.json'
 if not p.exists():continue
 status=json.loads(p.read_text());sample=json.loads((root/'samples.json').read_text()) if (root/'samples.json').exists() else []
 processes.append(dict(path=str(root),pid=status['pid'],status=status['status'],samples=sample,error=status.get('error')))
 all_samples.extend(sample)
 if status['status']=='PASS':completed.append((root,status,sample))
 elif status['status']=='FAILED':failed.append(dict(path=str(root),pid=status['pid'],error=status['error'],retained_samples=len(sample)))
result=dict(status='PENDING',processes=processes,all_retained_samples=all_samples,failed_processes=failed,excluded_samples=[],excluded_processes=[],full_paired_processes=len(completed),model_claim=None)
if len(completed)==3:
 for scenario in REF:
  armvals={arm:[] for arm in ('r004','r031','llama')};pidrows=[]
  for root,status,samples in completed:
   vals={arm:[s['mean_ms'] for s in samples if s['scenario']==scenario and s['arm']==arm] for arm in armvals}
   assert all(len(v)==3 for v in vals.values())
   means={k:statistics.mean(v) for k,v in vals.items()}
   pidrows.append(dict(pid=status['pid'],means_ms=means,round_means_ms=vals))
   for k in armvals:armvals[k].append(means[k])
  mean=statistics.mean(armvals['r031'])
  diffs={arm:stats([a-b for a,b in zip(armvals['r031'],armvals[arm],strict=True)]) for arm in ('r004','llama')}
  fixed={arm:stats([a-REF[scenario][arm] for a in armvals['r031']]) for arm in ('r004','llama')}
  result.setdefault('scenarios',{})[scenario]=dict(pid_pairs=pidrows,arms={k:stats(v) for k,v in armvals.items()},paired_differences_ms=diffs,fixed_historical_reference_differences_ms=fixed,weight_bytes=WEIGHTS,achieved_weight_GB_s=WEIGHTS/mean/1e6,percent_of_448=WEIGHTS/mean/1e6/448*100,percent_of_428_165=WEIGHTS/mean/1e6/428.165*100,strictly_faster_than_fresh_llama=diffs['llama']['high95']<0,strictly_faster_than_fixed_llama=fixed['llama']['high95']<0,strictly_faster_than_r004=diffs['r004']['high95']<0,conditional_prediction_ms=REF[scenario]['prediction'],measured_minus_conditional_prediction_ms=mean-REF[scenario]['prediction'])
  # Every available sample from incomplete attempts is separately recomputed,
  # including unpaired rounds; no imputation or pooled-sample independence.
  supplementary=[]
  for proc in processes:
   rows=[s for s in proc['samples'] if s['scenario']==scenario]
   if not rows:continue
   counts={arm:len([s for s in rows if s['arm']==arm]) for arm in armvals}
   means={arm:statistics.mean(s['mean_ms'] for s in rows if s['arm']==arm) for arm in armvals if counts[arm]}
   supplementary.append(dict(pid=proc['pid'],status=proc['status'],retained_round_counts=counts,all_available_arm_means_ms=means,paired_rounds=sorted(set(s['round'] for s in rows if s['arm']=='r031')&set(s['round'] for s in rows if s['arm']=='r004')&set(s['round'] for s in rows if s['arm']=='llama'))))
  result['scenarios'][scenario]['all_attempts_zero_exclusion_audit']=supplementary
  available_units=[v for v in supplementary if all(k in v['all_available_arm_means_ms'] for k in armvals)]
  # Include the valid partial process in a conservative all-available audit.
  # Its unequal round counts remain explicit; missing measurements are not filled.
  audit_arms={arm:stats([v['all_available_arm_means_ms'][arm] for v in available_units]) for arm in armvals}
  audit_diffs={arm:stats([v['all_available_arm_means_ms']['r031']-v['all_available_arm_means_ms'][arm] for v in available_units]) for arm in ('r004','llama')}
  result['scenarios'][scenario]['all_available_process_mean_statistics']=dict(arms=audit_arms,paired_process_mean_differences=audit_diffs,pids=[v['pid'] for v in available_units],t_reference_df=2,conservative_when_n_gt_3=True,complete_balanced_primary=False,missing_arms_not_imputed=True,all_valid_timing_samples_retained=True)

 result['status']='PASS'
 gates=json.loads((A/'model-gates-attempt2/status.json').read_text());rowfile=A/'integrated-row-verification-attempt2/status.json'
 if gates['status']=='PASS' and rowfile.exists() and json.loads(rowfile.read_text())['status']=='PASS':
  result['model_claim']={name:dict(ms_per_token=s['arms']['r031']['mean'],ci95_ms=[s['arms']['r031']['low95'],s['arms']['r031']['high95']],paired_llama_delta_ms=s['paired_differences_ms']['llama'],paired_r004_delta_ms=s['paired_differences_ms']['r004'],percent_of_448=s['percent_of_448'],percent_of_428_165=s['percent_of_428_165']) for name,s in result['scenarios'].items()}
(A/'ab-statistics.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],len(completed),'complete independent PIDs',flush=True)
if result.get('scenarios'):
 for name,s in result['scenarios'].items():print(name,s['arms']['r031']['mean'],s['paired_differences_ms'],flush=True)
