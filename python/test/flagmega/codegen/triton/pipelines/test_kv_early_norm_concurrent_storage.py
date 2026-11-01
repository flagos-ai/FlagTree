"""Early KV writes must not reuse a Q-A partial still read by parallel CTAs."""
import importlib.util
from pathlib import Path
import pytest
from triton.flagmega.errors import CodegenError

TUTORIAL=Path(__file__).resolve().parents[5]/"tutorials/flagmega/04-glm5.3-flash-fp8/nvidia-h800"
spec=importlib.util.spec_from_file_location("tutorial_kv_before_qb",TUTORIAL/"local_optimizations/kv_before_qb.py")
kv=importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)

def description():
    actors=[f"program_{i:03d}" for i in range(11)]
    source="(workspace.to(tl.pointer_type(tl.bfloat16)) + 9728)"
    stats="(block_local_data.to(tl.pointer_type(tl.float32)) + 64)"
    owned="((norm_apply_local_offsets // 8 % 4 + shard_coord2 * 4) * 8 + norm_apply_local_offsets % 8)"
    q=dict(symbol="qa",signature="workspace, qweight, device_dptr, vd, wd, qa_scale, rdata, block_local_data, kvweight, kvd, rootwindow",
        qa_norm_fused=dict(writer_active="(shard_coord1 == 0)",local_capacity=96),
        qkv_concurrent=dict(kv_arguments="workspace, kvweight, device_dptr, vd, kvd, rootwindow"))
    k=dict(symbol="self_attn_kv_a_proj_with_mqa",signature="workspace, kvweight, device_dptr, vd, kvd, _qkv_window",
        qkv_partial_pointer="(_qkv_window.to(tl.pointer_type(tl.float32)) + 8192)",rows=512,parts=1,accumulation_splits=2)
    st=dict(symbol="kv_stats",source=source,result=stats,result_offsets=["norm_outer_index"],reduction_capacity=512,tile=512,
        norm_stats_reduction_layout="contiguous_8",norm_stats_reduction_algorithm="tree")
    n=dict(symbol="kv_norm",signature="workspace, block_local_data, kv_scale, rdata",
        source=source,stats=stats,scale="kv_scale",bias="rdata.to(tl.pointer_type(tl.float32)) + 79304640",
        result="block_local_data.to(tl.pointer_type(tl.bfloat16))",active="norm_apply_local_offsets < 32",
        writer_active="True",source_offset=owned,stats_square_sum_offset="0",scale_offset=owned,
        bias_offset=owned,result_offset="norm_apply_local_offsets",epsilon=1e-5,input_type="tl.bfloat16",
        output_type="tl.bfloat16",tile=32,local_capacity=32,normalization_size=512,use_mean=False,
        round_before_scale=False,input_conversion_type=None)
    events=[
        dict(kind="tir.kernel_call",family="partitioned_linear",symbol="qa",arguments=q["signature"].replace("rootwindow","_flagmega_call_pool_workspace")),
        dict(kind="barrier",scope="grid"),
        dict(kind="tir.kernel_call",family="dense_matmul",symbol="qb",arguments="workspace"),
        dict(kind="tir.kernel_call",family="partitioned_linear",symbol=k["symbol"]+"__merge",
             arguments="workspace, kvweight, device_dptr, vd, kvd, _flagmega_call_pool_workspace"),
        dict(kind="barrier",scope="grid"),
        dict(kind="tir.kernel_call",family="norm_stats",symbol="kv_stats",arguments="workspace"),
        dict(kind="barrier",scope="block"),
        dict(kind="tir.kernel_call",family="norm_apply",symbol="kv_norm",arguments=n["signature"]),
        dict(kind="barrier",scope="block")]
    resource=dict(name="qkv_concurrent_normalized",nbytes=1024,offset=36864,alignment=256,programs=actors)
    return dict(device_functions=[dict(function="decode_mla_attention_sparse.prefix",schedule=dict(consumer_events=events))],
        programs=[],render_calls=[q,k,st,n],kernels=[dict(symbol=a,entry_function="decode_mla_attention_sparse.prefix") for a in actors],
        persistent_window_resources=dict(resources=[resource]),implementation_override=dict(inputs={}))

def test_early_KV_result_uses_planned_storage():
    d=description()
    old_source=d["render_calls"][-1]["source"]
    revised=kv.describe(None,lambda _:d)
    k=revised["render_calls"][1]["kv_idle_normalization"]
    assert k["result"] != old_source, "KV writes alias live Q-A partial bytes 19456..20480"
    assert k["source"] == k["result"], "The later copy must consume the protected normalized result"
    assert "36864" in k["result"] and "_qkv_window" in k["result"]
    assert k["result_offset"] == d["render_calls"][-1]["source_offset"]
    assert revised["kv_idle_normalization"]["qa_owners"]==list(range(16))
    assert revised["kv_idle_normalization"]["kv_owners"]==list(range(16,32))
    assert revised["kv_idle_normalization"]["original_qb_completion_grid_retained"]

def test_missing_KV_result_resource_is_rejected():
    d=description();d["persistent_window_resources"]["resources"]=[]
    with pytest.raises(CodegenError,match="normalized"):
        kv.describe(None,lambda _:d)

def test_KV_result_resource_requires_all_ordered_actors():
    d=description();d["persistent_window_resources"]["resources"][0]["programs"]= ["program_000"]
    with pytest.raises(CodegenError,match="actor"):
        kv.describe(None,lambda _:d)
