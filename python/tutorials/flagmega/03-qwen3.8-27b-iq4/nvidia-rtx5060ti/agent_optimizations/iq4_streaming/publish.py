"""Publish genuine r001 compiler outputs with inherited upstream lineage explicit."""
from pathlib import Path
from .build import WORK, REPO, BASELINE
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, subprocess, sys
BASE=BASELINE
ACTOR='codex:/root:graph_optimizer:implement_r001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),nbytes=p.stat().st_size)
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def publish(trial):
    trial=trial.resolve()
    if not trial.is_relative_to(WORK/'.local/milestones/m000/revisions/r001/attempts'):
        raise ValueError('Only r001 attempt outputs may be published')
    build=json.loads((trial/'build.json').read_text())
    if build['plan_hash']!='7b04b547f07fd1c71e4ecb8242720a631470c2a49953ab115487f3a1f5a6c361':
        raise ValueError('Wrong frozen subject')
    sync_id='r001-v21-121647-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    packet=WORK/'.local/views/prepared'/sync_id
    packet.mkdir(parents=True,exist_ok=False)
    rows=[];indexes=[]
    def add(p,relative,**identity):
        if not p.is_file():raise FileNotFoundError(p)
        q=packet/'payload'/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        rows.append(dict(source=str(q.relative_to(WORK)),path=relative,sha256=sha(q),**identity))
    # The unchanged imported/vectorized/full-packing chain is reused by its
    # recorded hashes. Later r000 distributed/selected/native output is excluded.
    original=json.loads((BASE/'outputs.json').read_text())
    for root,inherited in ((BASE,True),(trial,False)):
        index=root/'dumps/dumps.json'
        if not index.exists():continue
        records=json.loads(index.read_text());selected=[]
        for r in records:
            if inherited:
                chosen=r['category']=='compile' and r['stage'] in {'imported','vectorized','unused_functions_removed'} and 'AutoDistributedPass' not in r['relative_path']
            else:
                chosen=r['category']=='compile' or r['stage'] in {'distributed_boundary_layout_propagated','selected_tir','selected_tir_variants','selected_microkernels','packaged_tir','bufferized_tir','memory_placed_tir','synchronized_tir','allocated_tir'}
            if not chosen:continue
            selected.append(r)
            for field in ('relative_path','companion_path'):
                p=root/'dumps'/r[field]
                if inherited:
                    recorded=original[str(Path('dumps')/r[field])]
                    if sha(p)!=(recorded if isinstance(recorded,str) else recorded['sha256']):raise ValueError('Upstream inherited IR changed')
                dest='artifacts/ir/'+('inherited-r000/' if inherited else 'r001/')+r[field]
                add(p,dest,**{k:r[k] for k in ('category','stage','function','dialect','semantic_hash','module_semantic_hash')},producer='verified-unchanged-upstream' if inherited else 'normal-r001-compiler',inherited=inherited)
        indexes.append(dict(origin=str(root),inherited=inherited,index=pin(index),records=selected))
    for name in ('packing','distribution'):
        for suffix in ('.py','.il'):
            p=trial/(name+suffix)
            if p.exists():add(p,'artifacts/ir/r001/checkpoints/'+p.name,producer='normal-r001-compiler',inherited=False)
    artifact=trial/'artifact/artifact.json'
    if artifact.exists():
        manifest=json.loads(artifact.read_text())
        for section in manifest.get('sections',[]):
            p=trial/'artifact'/section['path']
            if str(section.get('kind','')).startswith('codegen.triton-') and p.suffix=='.py':
                add(p,'artifacts/generated/'+section['path'],producer='normal-r001-artifact-writer',section=section['name'])
        for p in sorted((trial/'artifact/ir').rglob('*')):
            if p.is_file() and p.suffix in {'.py','.il','.script'}:
                add(p,'artifacts/ir/r001/final/'+str(p.relative_to(trial/'artifact/ir')),producer='normal-r001-artifact-writer')
    shutil.copy2(trial/'build.json',packet/'build.json')
    shutil.copy2(trial/'sources.json',packet/'sources.json')
    save(packet/'dump-lineage.json',indexes)
    source=dict(path=str((packet/'sources.json').relative_to(WORK)),sha256=sha(packet/'sources.json'))
    inputs=[source,dict(path=str((packet/'build.json').relative_to(WORK)),sha256=sha(packet/'build.json')),dict(path=str((packet/'dump-lineage.json').relative_to(WORK)),sha256=sha(packet/'dump-lineage.json'))]
    pointer=WORK/'.local/views/root-artifacts.json'
    # pending stays pending, including actual JIT/static/resource qualification.
    status=dict(build=build['status'],runnable=build['runnable'] and build['status']=='built-and-verified',native='pending-H001-static-pilot',numerical=build.get('NUM','pending'),measurement='pending',review='pending',scope='full64-text-decoder',backend='triton',performance_accepted=False,error=build.get('error'))
    record=dict(schema='flagmega.working-sync/v1',kind='artifacts',sync_id=sync_id,agent_name='graph optimizer',agent_id=ACTOR,source_id=build['source_identity'],source_manifest=source,base_view_sha256=sha(pointer) if pointer.exists() else None,input_pins=inputs,files=rows,status=status)
    save(packet/'sync.json',record)
    subprocess.run([sys.executable,str(REPO/'skills/flagmega-developer/scripts/workspace.py'),'sync','--work-dir',str(WORK),'--agent','graph_optimizer','--kind','artifacts','--record',str(packet/'sync.json')],check=True)
    receipt=WORK/'.local/working/artifacts'/sync_id/'receipt.json'
    result=dict(schema='flagmega.r001-actual-publication/v1',actor=ACTOR,trial=str(trial),build=pin(trial/'build.json'),sync=pin(packet/'sync.json'),receipt=pin(receipt),root=pin(pointer),lineage=pin(packet/'dump-lineage.json'),files=len(rows),status=status)
    save(trial/'publication.json',result)
    assert {p.name for p in (WORK/'artifacts').iterdir()}=={'generated','ir'}
    print(json.dumps(dict(status=status,files=len(rows),receipt=str(receipt)),indent=2))
    return trial/'publication.json'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);publish(p.parse_args().trial)
