"""Checkpoint and explicit unsupported-native contracts for speculative drafts."""
import numpy as np
import pytest
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer import GGUFCheckpoint, DFlashImporter, import_model

gguf = pytest.importorskip('gguf')


def write_dflash(path, *, targets=(0, 3), selector=False):
    writer=gguf.GGUFWriter(path, arch='dflash')
    for key,value in {'embedding_length':32,'feed_forward_length':64,'block_count':1,
                      'attention.head_count':4,'attention.head_count_kv':2,'attention.key_length':8}.items():
        writer.add_uint32('dflash.'+key,value)
    if targets is not None: writer.add_array('dflash.target_layers',list(targets))
    shapes={'fc.weight':(32,64),'enc.output_norm.weight':(32,), 'output_norm.weight':(32,),
            'blk.0.attn_norm.weight':(32,), 'blk.0.attn_q.weight':(32,32),'blk.0.attn_k.weight':(16,32),
            'blk.0.attn_output.weight':(32,32),'blk.0.attn_q_norm.weight':(8,), 'blk.0.attn_k_norm.weight':(8,),
            'blk.0.ffn_norm.weight':(32,),'blk.0.ffn_gate.weight':(64,32),
            'blk.0.ffn_up.weight':(64,32),'blk.0.ffn_down.weight':(32,64)}
    if selector: shapes['selector_hidden.weight']=(4,32)
    for name,shape in shapes.items(): writer.add_tensor(name,np.zeros(shape,dtype=np.float32))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()


def test_dflash_import_preserves_target_binding_and_reports_execution_gap(tmp_path):
    path=tmp_path/'draft.gguf';write_dflash(path)
    checkpoint=GGUFCheckpoint(path)
    plan=DFlashImporter(checkpoint).import_plan()
    assert plan['target_layers']==[0,3] and plan['variant']=='dflash1'
    assert plan['decoder_attention']=='noncausal' and plan['executable'] is False
    assert checkpoint.load_tensor('output_norm.weight').sum()==0
    with pytest.raises(ImporterError,match='target-feature KV injection'):
        import_model(checkpoint)


@pytest.mark.parametrize('targets',[None,(0,0),(-1,3)])
def test_dflash_invalid_target_layers_rejected(tmp_path,targets):
    path=tmp_path/'bad.gguf';write_dflash(path,targets=targets)
    with pytest.raises(ImporterError,match='target_layers'): GGUFCheckpoint(path)


def test_dflash2_missing_selector_metadata_rejected(tmp_path):
    path=tmp_path/'bad-selector.gguf';write_dflash(path,selector=True)
    with pytest.raises(ImporterError,match='conv/selector'): DFlashImporter(path).import_plan()


def test_missing_dflash_weights_rejected(tmp_path):
    with pytest.raises(ImporterError,match='existing .gguf'): DFlashImporter(tmp_path/'missing.gguf')
