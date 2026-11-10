# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace

import pytest

import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('materialized_fixture',Path(__file__).with_name('test_materialized_source_rdata.py'))
_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixture)
example, module, pin, save = _fixture.example, _fixture.module, _fixture.pin, _fixture.save
from triton.flagmega import ir as fm
from triton.flagmega.passes.constants import freeze_constant_islands
from triton.flagmega.artifacts import pack_rdata, verify_rdata, write_artifact
from triton.flagmega.artifacts.materialized_source import RecipeGraph, physical, prove_output_reuse, verify_materialized_source
from triton.flagmega.errors import ArtifactError, IRVerificationError, IRSchemaError


@pytest.mark.parametrize('mutation',['operand_order','lanes','operation','dtype','effect','metadata'])
def test_typed_graph_changes_reject(example,mutation):
    cp,old,current,source,_,_=example
    recipes=[]
    for recipe in current.constant_recipes:
        nodes=[]
        for n in recipe.nodes:
            if n.op=='math.vectorized_binary':
                if mutation=='operand_order':n=replace(n,inputs=tuple(reversed(n.inputs)))
                elif mutation=='operation':n=replace(n,attrs={**n.attrs,'binary_op':'mul'})
                elif mutation=='lanes':n=replace(n,attrs={**n.attrs,'lhs_pack':{'axes':(0,),'lanes':(8,)}})
                elif mutation=='dtype':n=replace(n,type=fm.tensor_type(fm.VectorType(fm.DType.FLOAT16,(4,)),(4,)))
                elif mutation=='metadata':n=replace(n,metadata={**n.metadata,'numeric_rounding':'toward_zero'})
                else:n=replace(n,effect=replace(n.effect,kind='read',resource='mutable'))
            nodes.append(n)
        recipes.append(replace(recipe,nodes=tuple(nodes)))
    changed=replace(current,constant_recipes=tuple(recipes))
    with pytest.raises((ArtifactError,IRVerificationError,IRSchemaError,ValueError)):
        prove_output_reuse(changed,source)


@pytest.mark.parametrize('mutation',['stride','offset','compact','backing','alias'])
def test_noncanonical_physical_bytes_reject(example,mutation):
    source=example[3];b=source.owners[0]
    with pytest.raises((ArtifactError,IRVerificationError,IRSchemaError,ValueError)):
        if mutation=='stride':changed=replace(b,strides=tuple(s*2 for s in b.strides))
        elif mutation=='offset':
            # A one-byte moved owner is no longer aligned, even with equal shape.
            changed=replace(b,mem_span=replace(b.mem_span,start=b.mem_span.start+1))
        elif mutation=='compact':changed=replace(b,distributed_storage_kind=fm.DistributedBufferStorageKind.COMPACT_PER_CHIP)
        elif mutation=='backing':changed=replace(b,distributed_backing_type=b.distributed_type)
        else:changed=replace(b,alias='another-owner')
        physical(changed)



def test_recomputed_outer_pin_cannot_change_historical_recipe(example):
    cp,old,current,source,lp,mp=example
    lineage=json.loads(open(lp['path']).read());kp=lineage['current_key_data']
    key=json.loads(open(kp['path']).read());key['recipes'][0]['fingerprint']='0'*64
    changed=save(__import__('pathlib').Path(kp['path']),key)
    lineage['current_key_data']=changed
    lineage['reconstructed_historical_key_data']=changed
    changed_lineage=save(__import__('pathlib').Path(lp['path']),lineage)
    with pytest.raises(ArtifactError,match='preimage'):
        verify_materialized_source(old,cp,lineage_pin=changed_lineage,prior_manifest_pin=mp)


def test_source_corruption_full_audit_rejects(example,tmp_path):
    cp,old,current,source,_,_=example
    assets=tmp_path/'new';index=pack_rdata(current,cp,assets,readonly_source=source)
    p=assets/index['sources'][0]['path']
    with p.open('r+b') as f:
        first=f.read(1);f.seek(0);f.write(bytes([first[0]^1]))
    # Metadata-only checking remains distinct from the complete integrity audit.
    verify_rdata(assets,module=current,verify_image_bytes=False)
    with pytest.raises(ArtifactError,match='hash mismatch'):
        verify_rdata(assets,module=current)


def test_fresh_process_artifact_after_old_source_deleted(example,tmp_path):
    cp,old,current,source,_,_=example
    artifact=tmp_path/'standalone'
    write_artifact(current,artifact,target='nvidia-sm120',checkpoint=cp,readonly_source=source)
    shutil.rmtree(tmp_path/'old');shutil.rmtree(tmp_path/'cache')
    code='from triton.flagmega.artifacts import load_artifact; import sys; m,ir=load_artifact(sys.argv[1]); print(ir.stage)'
    p=subprocess.run([sys.executable,'-c',code,str(artifact)],capture_output=True,text=True,check=True)
    assert p.stdout.strip()=='bufferized_tir'


@pytest.mark.parametrize('ownership',['partial','exclusive'])
def test_partial_exclusive_view_rejected(example,ownership):
    from triton.flagmega.artifacts.materialized_source import _dense_type
    dt=example[3].owners[0].distributed_type
    changed=replace(dt,**{ownership: fm.SBPPartial((0,)) if ownership=='partial' else fm.SBPExclusive((0,))})
    with pytest.raises(ArtifactError,match='partial or exclusive'):_dense_type(changed)


def test_multiple_chips_rejected(example):
    from triton.flagmega.artifacts.materialized_source import module_owners
    current=example[2]
    changed=replace(current,metadata={**current.metadata,'launch_contract':{'grid_mesh':fm.Placement((2,),'c','c').to_data()}})
    with pytest.raises((ArtifactError,IRVerificationError,IRSchemaError,ValueError)):
        module_owners(changed)


@pytest.mark.parametrize('shape,axes',[((2,2),(1,0)),((2,8),(1,0))])
def test_same_multiset_transpose_or_lane_permutation_rejected(shape,axes):
    b=fm.IRBuilder(dialect='tir',stage='selected_tir')
    w=b.weight('w',fm.tensor_type('float32',shape),source='checkpoint',key='w',id='w')
    p=b.call('tensors.permute',(w,),fm.tensor_type('float32',tuple(shape[i] for i in axes)),attrs={'axes':axes},id='permuted')
    b.function('main',(),(p,))
    m=freeze_constant_islands(b.build(entry='main'))
    with pytest.raises(ArtifactError,match='unsupported op'):RecipeGraph(m.constant_recipes)


def test_factory_source_and_memoized_proof_are_immutable(example):
    cp,old,current,source,_,_=example
    proof=prove_output_reuse(current,source)
    with pytest.raises(ArtifactError):source.table['index']['sha256']='0'*64
    with pytest.raises(ArtifactError):source.path=cp.path
    with pytest.raises(TypeError):source.graph.logical['old_w']='0'*64
    with pytest.raises(ArtifactError):proof['programs'][0]['operands'][0]['offset']+=1


@pytest.mark.parametrize('field',[('anchor_sha256',),('original_cache','image_sha256'),('manifest','semantic_hash'),('receipt','cache_key')])
def test_review_lineage_single_field_mutations_reject_without_module(example,tmp_path,field):
    from triton.flagmega.artifacts.materialized_source import validate_materialized_source_table
    cp,old,current,source,_,_=example
    assets=tmp_path/'review';index=pack_rdata(current,cp,assets,readonly_source=source)
    changed=copy.deepcopy(index);target=changed['materialized_sources'][0]['lineage']
    for key in field[:-1]:target=target[key]
    target[field[-1]]='f'*64
    from triton.flagmega.artifacts.source_layout import content_digest
    changed['sha256']=content_digest(changed)
    with pytest.raises(ArtifactError):validate_materialized_source_table(changed,None)
