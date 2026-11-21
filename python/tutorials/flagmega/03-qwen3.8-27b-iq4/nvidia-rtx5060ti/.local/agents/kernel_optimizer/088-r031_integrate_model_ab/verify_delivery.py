from pathlib import Path
import json,math,hashlib,gzip
A=Path(__file__).resolve().parent
r=json.loads((A/'integration-ab-receipt.json').read_text());s=json.loads((A/'ab-statistics.json').read_text())
assert all(v['outcome']=='PASS' for v in r['MODEL_TOKEN10'].values())
assert r['MODEL_TOKEN10']['chinese64']['eligible_token_ids'][6]==4960
assert r['integrated_row_verification']['status']['status']=='PASS'
assert len(r['integrated_row_verification']['per_row_sha256_files'])==6
assert r['model_claim'] is not None and r['measurement_protocol']['same_process']
assert r['excluded_samples']==r['excluded_processes']==[]
assert r['R16_integrated'] is False and r['no_new_grid_arrival_counter']
raws={}
for rec in r['timing_samples']:
 data=rec['data'];key=(data['pid'],data.get('scenario'),data['round'],data.get('arm'))
 # llama's frozen raw omits scenario/arm; its typed filesystem path supplies those.
 path=Path(rec['file']['path']);key=(data['pid'],path.parents[2].name,path.parent.parent.name,path.parent.name)
 assert len(data['latency_ns'])==data['steps'] and data['profiler_attached'] is False
 assert all(x>0 for x in data['latency_ns'])
 assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['file']['sha256']
 raws[key]=sum(data['latency_ns'])/len(data['latency_ns'])/1e6
for name,c in s['scenarios'].items():
 for pid in c['pid_pairs']:
  for arm,vals in pid['round_means_ms'].items():
   direct=[]
   for n in range(3):
    key=(pid['pid'],name,f'round-{n}',arm)
    assert key in raws,key
    direct.append(raws[key])
   assert all(abs(x-y)<1e-10 for x,y in zip(vals,direct))
 for pid in c['pid_pairs']:
  root=next(Path(p['path']) for p in s['processes'] if p['pid']==pid['pid'])
  for n in range(3):
   base=json.loads((root/name/f'round-{n}'/'r004/raw.json').read_text())
   new=json.loads((root/name/f'round-{n}'/'r031/raw.json').read_text())
   assert base['tokens']==new['tokens']
   old_endpoint=json.loads((root/name/f'round-{n}'/'r004/pre.json').read_text())['endpoint']
   new_endpoint=json.loads((root/name/f'round-{n}'/'r031/pre.json').read_text())['endpoint']
   assert old_endpoint==new_endpoint
 for arm in ('r004','llama'):
  v=[p['means_ms']['r031']-p['means_ms'][arm] for p in c['pid_pairs']]
  mu=sum(v)/3;se=math.sqrt(sum((x-mu)**2 for x in v)/2/3);t=4.302652729696142
  d=c['paired_differences_ms'][arm]
  assert abs(d['mean']-mu)<1e-10 and abs(d['high95']-(mu+t*se))<1e-10
 assert c['percent_of_448']>=80
 assert c['paired_differences_ms']['llama']['high95']<0
 assert c['fixed_historical_reference_differences_ms']['llama']['high95']<0
 assert c['paired_differences_ms']['r004']['high95']<0
for name,x in r['subjects'].items():
 p=Path(x['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 subject=json.loads(p.read_text())
 for f in subject['body_sources']:
  p=Path(f['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
# Stream every stored row, including already-correct direct bitwise comparisons.
counts={}
for seed in (20261120,1,37):
 a=A/'integrated-row-verification-attempt2'/f'r004-seed-{seed}.jsonl.gz';b=a.with_name(f'r031-seed-{seed}.jsonl.gz')
 total=0;outputs={}
 with gzip.open(a,'rt') as x,gzip.open(b,'rt') as y:
  for left,right in zip(x,y,strict=True):
   assert left==right
   row=json.loads(left);total+=1
   for name,data in row['outputs'].items():outputs[name]=outputs.get(name,0)+len(data['row_sha256'])
 assert total==208
 assert outputs==dict(glu=64*17408,qn=48*16,kn=48*16,beta=48*48,decay_log=48*48,core=48*48,state=48*6144,y=48*48),outputs
 counts[str(seed)]=outputs
(A/'delivery-verification.json').write_text(json.dumps(dict(status='PASS',raw_timing_recomputed=True,Student_t_df2_recomputed=True,all_claimed_thresholds_pass=True,row_counts_per_seed=counts,all_stored_rows_rechecked=True,full_32_64_token_trajectories_and_external_endpoints_match_frozen_r004=True,exclusions=[]),indent=2)+'\n')
print('DELIVERY_VERIFICATION_PASS',flush=True)
