# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Real small writer/cache/CPU-loader tests for complete-output reuse."""
from pathlib import Path
from dataclasses import replace
import copy
import errno
import hashlib
import json
import shutil

import numpy as np
import pytest
import torch

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import write_artifact, load_artifact, pack_rdata, verify_rdata
from triton.flagmega.artifacts import rdata
from triton.flagmega.artifacts.materialized_source import (
    verify_materialized_source, prove_output_reuse, digest, RecipeGraph, physical,
)
from triton.flagmega.artifacts.source_layout import content_digest
from triton.flagmega.artifacts.source_verification import file_identity
from triton.flagmega.errors import ArtifactError
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.passes.tir.bufferize.planner import BufferizationOptions
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy


def pin(p):
    raw=p.read_bytes()
    return dict(path=str(p),nbytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    return pin(p)


class Checkpoint:
    def __init__(self,path):
        self.path=path;path.write_bytes(b'synthetic checkpoint identity')
        self.values={'w': np.arange(16,dtype=np.float32)-8, 'other': np.arange(16,dtype=np.float32)+70}
        self.calls=0
    def cache_identity(self,keys):
        stat=self.path.stat()
        return dict(path=str(self.path),device=stat.st_dev,inode=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    tensors=[dict(key=k,shape=[16],dtype='float32',content_sha256=hashlib.sha256(self.values[k].tobytes()).hexdigest()) for k in sorted(keys)])
    def load_tensor(self,key,*,device='cpu'):
        self.calls+=1
        return torch.from_numpy(self.values[key].copy()).to(device)


def module(*,current=False,weight='w',literal=1.0,zero=0.0):
    b=fm.IRBuilder(dialect='tir',stage='selected_tir')
    prefix='new_' if current else 'old_'
    t=fm.tensor_type('float32',(16,));v4=fm.tensor_type(fm.VectorType(fm.DType.FLOAT32,(4,)),(4,))
    v8=fm.tensor_type(fm.VectorType(fm.DType.FLOAT32,(8,)),(2,))
    w=b.weight('weight',t,source='checkpoint',key=weight,id=prefix+'w')
    one=b.call('builtin.splat_const',(),t,attrs={'value':literal},id=prefix+'one')
    affine=b.call('math.vectorized_binary',(w,one),v4,attrs={'binary_op':'add','lhs_pack':{'axes':(0,),'lanes':(4,)},'rhs_pack':{'axes':(0,),'lanes':(4,)}},id=prefix+'add')
    if current:
        dt=fm.DistributedType(v4,(fm.SBP.split_contiguous((0,)),),fm.Placement((2,),'x','b'))
        affine=b.call('distributed.sharded_view',(affine,),dt,attrs={'new_type':dt},id=prefix+'view')
        bt=fm.DistributedType(v8,(fm.SBP.split_contiguous((0,)),),fm.Placement((2,),'x','b'))
    else:bt=v8
    affine=b.call('tensors.bitcast',(affine,),bt,attrs={'dtype':{'kind':'vector','elem_type':'float32','lanes':(8,)}},id=prefix+'affine')
    z=b.call('builtin.splat_const',(),t,attrs={'value':zero},id=prefix+'zero')
    z1=b.call('tensors.pack',(z,),v8,attrs={'axes':(0,),'lanes':(8,)},id=prefix+'z1')
    outputs=[w,affine,z1]
    if current:
        outputs.append(b.call('tensors.pack',(z,),v8,attrs={'axes':(0,),'lanes':(8,)},id=prefix+'z2'))
    from triton.flagmega.ir.distributed_inference import tensor_of
    owned=[]
    for out in outputs:
        dt=fm.DistributedType(tensor_of(out.type),(fm.SBP.broadcast(),),fm.Placement((2,),'x','b'))
        owned.append(b.call('distributed.sharded_view',(out,),dt,attrs={'new_type':dt},id=out.id+'.owner'))
    b.function('main',(),owned)
    m=freeze_constant_islands(b.build(entry='main'))
    m=NttBufferizationPolicy(BufferizationOptions.generic(alignment=16)).bufferize(m)
    return replace(m,stage='bufferized_tir')


@pytest.fixture
def example(tmp_path):
    cp=Checkpoint(tmp_path/'checkpoint.bin'); old=module();current=module(current=True)
    old_artifact=tmp_path/'old';cache=tmp_path/'cache'
    write_artifact(old,old_artifact,target='nvidia-sm120',checkpoint=cp,rdata_cache_dir=cache)
    key_data=rdata._rdata_cache_key_data(old,cp);key=digest(key_data);entry=cache/key
    verify_rdata(entry,module=old) # Actual complete synthetic payload audit.
    manifest=json.loads((old_artifact/'artifact.json').read_text());index=json.loads((entry/'rdata.index.json').read_text())
    evidence=tmp_path/'evidence'
    q=dict(same_original_content=True,path=str(cp.path),identity=list(file_identity(cp.path)),sha256=hashlib.sha256(cp.path.read_bytes()).hexdigest())
    qp=save(evidence/'qualification.json',q)
    source_root=Path(__file__).resolve().parents[4]
    receipt=dict(cache_key=key,semantic_hash=old.semantic_hash,rdata=manifest['rdata'],checkpoint={'sha256':q['sha256']},source_tree=str(source_root))
    rp=save(evidence/'receipt.json',receipt);kp=save(evidence/'key.json',key_data)
    lineage=dict(schema='flagmega.explicit-rdata-current-identity-lineage/v1',status='pass',prior_materialization=rp,qualification=qp,
        current_key_data=kp,reconstructed_historical_key_data=kp,current_cache_key=key,prior_cache_key=key,
        same_original_GGUF_sha256=q['sha256'],current_checkpoint_stat=q['identity'],
        prior_index=pin(entry/'rdata.index.json'),prior_cache_manifest=pin(entry/'cache.json'),current_cache_manifest=pin(entry/'cache.json'),
        image_sha256=index['sha256'],fresh_image_stat=list(file_identity(entry/'rdata.bin')))
    lp=save(evidence/'lineage.json',lineage);mp=pin(old_artifact/'artifact.json')
    source=verify_materialized_source(old,cp,lineage_pin=lp,prior_manifest_pin=mp)
    return cp,old,current,source,lp,mp


def test_writer_loader_cache_and_move(example,tmp_path):
    from triton.flagmega.runtime.source_rdata import load_source_rdata
    cp,old,current,source,_,_=example
    proof=prove_output_reuse(current,source)
    assert len(proof['bindings'])==4
    assert sum(x['rule']=='full-logical-sharded-view' for x in proof['bindings'])==1
    zeros=[p for p in proof['programs'] if p['buffer'].removesuffix('.owner').endswith(('z1','z2'))]
    assert len(zeros)==2 and zeros[0]['operands']==zeros[1]['operands'] and zeros[0]['ranges']!=zeros[1]['ranges']
    calls=cp.calls
    def denied(*args,**kwargs):raise AssertionError('numeric fallback forbidden')
    cp.load_tensor=denied
    artifact=tmp_path/'new';cache=tmp_path/'reuse-cache'
    write_artifact(current,artifact,target='nvidia-sm120',checkpoint=cp,rdata_cache_dir=cache,readonly_source=source)
    index=verify_rdata(artifact/'assets',module=current)
    assert cp.calls==calls and digest(rdata._rdata_cache_key_data(current,cp))!=digest(source.table['key_data'])
    # Normal cache hit revalidates metadata and reuses the snapshot receipt.
    hit=pack_rdata(current,cp,tmp_path/'hit',cache_dir=cache,readonly_source=source)
    assert hit==index
    moved=tmp_path/'moved';artifact.rename(moved)
    shutil.rmtree(tmp_path/'old');shutil.rmtree(tmp_path/'cache')
    manifest,loaded=load_artifact(moved)
    actual=load_source_rdata(moved/'assets',index,loaded,chip=0,device='cpu').numpy()
    for buffer in fm.verify_buffer_plan(loaded).buffers:
        if buffer.storage!='rdata':continue
        expected=np.zeros(16,dtype=np.float32) if buffer.id.removesuffix('.owner').endswith(('z1','z2')) else cp.values['w']+(1 if buffer.id.removesuffix('.owner').endswith('affine') else 0)
        assert actual[buffer.offset:buffer.offset+buffer.nbytes].tobytes()==expected.tobytes()


@pytest.mark.parametrize('weight,literal,zero',[('other',1.,0.),('w',2.,0.),('w',1.,-0.)])
def test_changed_weight_literal_bits_reject(example,weight,literal,zero):
    *_,source,lp,mp=example
    with pytest.raises(ArtifactError,match='byte-equivalent'):
        prove_output_reuse(module(current=True,weight=weight,literal=literal,zero=zero),source)


def test_changed_file_identity_reject(example):
    cp,old,current,source,lp,mp=example
    source.path.chmod(0o644)
    with pytest.raises(ArtifactError,match='version'):
        prove_output_reuse(current,source)


def test_ficlone_failure_is_terminal(example,tmp_path,monkeypatch):
    import fcntl
    cp,old,current,source,_,_=example
    def fail(*args):raise OSError(errno.EOPNOTSUPP,'test unsupported reflink')
    monkeypatch.setattr(fcntl,'ioctl',fail)
    monkeypatch.setattr(shutil,'copyfile',lambda *args,**kwargs: pytest.fail('copy fallback'))
    with pytest.raises(OSError,match='reflink'):
        pack_rdata(current,cp,tmp_path/'failed',cache_dir=tmp_path/'new-cache',readonly_source=source)
    assert not list((tmp_path/'new-cache').glob('*/cache.json'))


@pytest.mark.parametrize('mutation',['prior_output','source_offset','target_offset','recipe','index','missing','padding'])
def test_metadata_mutation_rejects_even_without_bytes(example,tmp_path,mutation):
    cp,old,current,source,_,_=example
    assets=tmp_path/'assets';index=pack_rdata(current,cp,assets,readonly_source=source)
    altered=copy.deepcopy(index)
    if mutation=='prior_output':altered['materialized_sources'][0]['bindings'][0]['prior_output']='absent'
    elif mutation=='source_offset':altered['programs'][0]['operands'][0]['offset']+=1
    elif mutation=='target_offset':altered['programs'][0]['ranges'][0][0]+=1
    elif mutation=='recipe':altered['materialized_sources'][0]['recipes'][0]['nodes'][0]['attrs']['key']='other'
    elif mutation=='index':altered['materialized_sources'][0]['index']['entries'][0]['offset']+=1
    elif mutation=='missing':altered['programs'].pop()
    else:altered['nbytes']+=16;altered['segment_nbytes']+=16
    altered['sha256']=content_digest(altered);save(assets/'rdata.index.json',altered)
    with pytest.raises(ArtifactError):verify_rdata(assets,module=current,verify_image_bytes=False)


def test_pure_effect_and_unknown_ops_reject():
    m=module();r=m.constant_recipes[0]
    node=r.nodes[0]
    with pytest.raises(ArtifactError):RecipeGraph([replace(r,nodes=(replace(node,metadata={'execute':True}),*r.nodes[1:]))])
    with pytest.raises(ArtifactError):RecipeGraph([replace(r,nodes=(replace(node,op='tensors.permute'),*r.nodes[1:]))])


def test_unknown_snapshot_proof_does_not_hash(tmp_path):
    p=tmp_path/'payload';p.write_bytes(b'tiny')
    with pytest.raises(ArtifactError,match='no existing'):
        rdata._verified_snapshot_identity(p,hashlib.sha256(b'tiny').hexdigest(),require_existing=True)
