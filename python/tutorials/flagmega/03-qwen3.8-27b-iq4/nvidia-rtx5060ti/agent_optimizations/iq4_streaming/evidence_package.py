"""Freeze resource/PC evidence and sign the actual optimizer's limited scope.

Run with --attempt <resources directory> --pc <final PC report directory>.
No compiler rebuild, GPU execution, graph writes or performance acceptance.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import ortools
import triton

from .evidence_resources import pin


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--attempt',type=Path,required=True)
    ap.add_argument('--pc',type=Path,required=True)
    args=ap.parse_args()
    work=Path(__file__).resolve().parents[2]
    repo=work.parents[4]
    revision=work/'.local/milestones/m000/revisions/r001'
    assignment=revision/'agents/kernel_optimizer/implement_r001'
    assignment.mkdir(exist_ok=False)
    r=json.loads((args.attempt/'resources.json').read_text())
    pc=json.loads((args.pc/'pc-proof.json').read_text())
    old_result=revision/'results/kernel-implementation.json'
    previous=pin(old_result) if old_result.exists() else None
    if previous:
        shutil.copy2(old_result,args.attempt/'prior-kernel-implementation-134557.json')
    baseline_path=work/'.local/milestones/m000/revisions/r000/results/NUM-r4-20261116T1027/result.json'
    b=json.loads(baseline_path.read_text())
    baselines={e['resources']['name']:e for e in b['native_entries']}
    subject=revision/'agents/graph_expert/plan_r001_v2/retry-completion/plan-body-v2.1.md'
    planhash='7b04b547f07fd1c71e4ecb8242720a631470c2a49953ab115487f3a1f5a6c361'
    assert pin(subject)['sha256']==planhash
    assert ortools.__version__=='9.10.4067'
    assert len(r['resources'])==len(pc['entries'])==12
    assert pc['status']=='self-checks-passed-review-pending'
    pc_entries={e['entry']:e for e in pc['entries']}
    comparison=[]
    for e in r['resources']:
        before=baselines[e['entry']]
        cur=e['resources']
        required=before['queried_ctas_per_sm']
        assert required==e['admission']['required_ctas_per_sm']
        x=dict(entry=e['entry'],calls=e['calls'],family=e['family'],N=e['N'],K=e['K'],
               source=e['source'],signature=e['signature'],grid=e['grid'],
               compiled_hash=cur['compiled_hash'],
               baseline=dict(registers_per_thread=before['resources']['registers_per_thread'],
                             static_shared_bytes=before['static_shared_bytes'],
                             dynamic_shared_bytes=before['dynamic_shared_bytes'],
                             cubin_shared_bytes=1024,local_bytes_per_thread=before['local_bytes_per_thread'],
                             stack_bytes=before['resources']['stack_frame_bytes'],
                             spill_bytes=before['resources']['spill_bytes'],ctas_per_sm=required,
                             cubin=before['files']['cubin']),
               candidate=cur,
               gates=dict(admission=e['admission']['passed'],zero_spill=e['zero_spill'],
                          GLU_register_limit=e['glu_register_gate']['passed'],
                          static_self_checks=e['static']['passed'] and all(pc_entries[e['entry']]['checks'].values()),
                          payload_sector='pending-resource-stop',activation_sector='pending-resource-stop',
                          eligible_warps='pending-resource-stop',simultaneous_placement='pending-no-execution'),
               PC_proof=pc_entries[e['entry']])
        # Verify the exact executable evidence instead of trusting JSON metadata.
        for item in list(cur['files'].values())+[e['source'],x['PC_proof']['ledger'],before['files']['cubin']]:
            assert pin(item['path'])['sha256']==item['sha256'], item['path']
        comparison.append(x)
    assert sum(not x['gates']['admission'] for x in comparison)==9
    assert sum(not x['gates']['GLU_register_limit'] for x in comparison)==1
    assert all(x['gates']['zero_spill'] and x['gates']['static_self_checks'] for x in comparison)
    comparepath=args.attempt/'candidate-vs-baseline.json'
    dump(comparepath,dict(status='H001-preserve-failed',baseline=pin(baseline_path),entries=comparison,
                          capacity_scope='CUDA occupancy API; simultaneous residency/placement not executed',
                          shared_scope='driver static + launch dynamic reported separately from cubin reservation',
                          measured_sector_scope='Payload and activation separately pending; no aggregate substituted'))
    sourcepaths=[repo/p for p in ('lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp',
                                 'lib/Dialect/Triton/IR/Ops.cpp',
                                 'python/triton/experimental/tle/language/gpu/core.py',
                                 'python/triton/experimental/tle/language/gpu/__init__.py',
                                 'python/triton/flagmega/codegen/triton/kernels/_iq4_streaming.py.jinja',
                                 'python/triton/flagmega/codegen/triton/kernels/iq4_streaming_projection/stream.py.jinja',
                                 'python/triton/flagmega/codegen/triton/kernels/iq4_streaming_glu/stream.py.jinja',
                                 'python/triton/flagmega/codegen/triton/kernels/iq4_streaming_interleaved/stream.py.jinja')]
    sourcepaths.append(work/'agent_optimizations/iq4_streaming/native.py')
    sourcepins=[pin(p) for p in sourcepaths]
    sourcehash=hashlib.sha256(json.dumps(sourcepins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    library=repo/'python/triton/_C/libtriton.so'
    libpin=pin(library);libpin['mtime_utc']=datetime.datetime.fromtimestamp(library.stat().st_mtime,datetime.timezone.utc).isoformat()
    existing=work/'.local/agents/kernel_optimizer/implement_r001'
    collective=existing/'collective-02/checks.json'
    assert json.loads(collective.read_text())['status']=='passed'
    executables=[pin(Path(__file__)),pin(Path(__file__).with_name('evidence_resources.py')),
                 pin(Path(__file__).with_name('evidence_pc.py')),pin(Path(__file__).with_name('audit_h001.py')),
                 pin(Path(__file__).with_name('qualify.py')),pin(existing/'collective_checks.py'),
                 pin('/usr/local/cuda/bin/cuobjdump'),pin('/usr/local/cuda/bin/nvdisasm')]
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    pc_total=sum(e['shuffle_PCs'] for e in pc['entries'])
    result=dict(schema='flagmega.kernel-implementation/r001-v2.1',milestone_id='m000',revision_id='r001',
                plan_version='v2.1',plan_hash=planhash,agent_name='kernel_optimizer',assignment_id='implement_r001',
                actual_context='codex:/root:kernel_optimizer:implement_r001:resume-20261116T142313Z',
                status='evidence-complete-current-candidate-rejected-H001-preserve',
                decision='reject-current-candidate-resource-gates',previous_result=previous,
                source_hash=sourcehash,source_pins=sourcepins,native_library=libpin,
                compiler_rebuilt_in_this_resume=False,ortools=ortools.__version__,python=sys.executable,
                executable_pins=executables,artifact=r['artifact'],
                resource_audit=pin(args.attempt/'resources.json'),per_shape_comparison=pin(comparepath),
                resource_rows=comparison,PC_proof=pin(args.pc/'pc-proof.json'),
                static=dict(status='self-checks-passed-independent-review-pending',
                            entries=12,SASS_shuffle_PCs=pc_total,
                            same_row_same_warp_lineage='E32/ERow actual IR + source/PTX/SASS PC and producer ledgers; warp-local dataflow induction',
                            shared_cross_warp_barrier_on_attributed_path=0,
                            whole_entry_shared_local_barrier_PCs=0,
                            full_mask_lane_predicated_shuffle_PCs=0,
                            numeric_layout_escape=0,independent_review='pending'),
                self_checks=dict(real_emitted_integrated_entries=12,matched_six_CTA_entries=9,
                                 source_and_cubin_hashes_rechecked=True,
                                 full_PC_source_TTGIR_PTX_SASS_mapping=True,
                                 concrete_SASS_and_PTX_idx_all_128_threads=True,
                                 duplicate_operands_reaching_definitions_verified=True,
                                 isolated_primitive_inherited=pin(collective),
                                 registration_tests_inherited=pin(existing/'registration-tests-04.log'),
                                 resource_gate_passed=False,zero_spill_passed=True),
                gates=dict(H001_static='self-checks-passed-independent-review-pending',
                           integrated_admission='FAILED-9-of-9-six-CTA-entries-now-five',
                           primary_GLU_register_limit='FAILED-125-greater-than-107',
                           zero_spill='passed-driver-cubin-and-PTX-SASS',
                           payload_sector=dict(status='pending-resource-stop',value=None,threshold_useful_B_per_32B=21.6341),
                           activation_sector=dict(status='pending-resource-stop',value=None),
                           eligible_warps=dict(status='pending-resource-stop',value=None,threshold=1.1648),
                           weight_read_unique='pending-resource-stop',dynamic_local_traffic='pending-resource-stop',
                           H001_traffic='pending-resource-stop',H001_time='pending-resource-stop',
                           H001_full='pending-other-owner',NUM_ABI_state='pending-other-owner'),
                counter_cost=dict(activation_request_amplification=4,
                                  BF16_B_per_CTA_K128=dict(baseline=256,candidate=1024),
                                  F32_B_per_CTA_K128=dict(baseline=512,candidate=2048),
                                  scope='Static source request work, not measured sectors/DRAM or wall time',
                                  FP32_leaf_owners=8,activation_and_tree_shuffle_issue_MIO_cost='not-measured',
                                  k_3_to_6_percent='unmeasured-attribution-budget'),
                GPU_kernel_launches=0,ordinary_timing=False,NCU_run=False,
                runnable_native=True,candidate_accepted=False,performance_accepted=False,milestone_achieved=False,
                full_bitwise_NUM='pending-resource-stop',open_gates=['K9','K6/K10','B007','G012'],
                failed_attempts_retained=[
                    'prior implementation/compiler/primitive/pilot failures retained at original paths',
                    'resources-resume-01: GPU busy; log retained',
                    'resources-resume-02: wrong Q entry index 11; corrected to actual IQ4 entry 16; partial artifacts retained',
                    'PC proofs 01-02: strict bijection/identity assumptions rejected; duplicate operands proved instead',
                    'PC proofs 04-05: incomplete ISA classification; final 06 classifies observed uniform/scalar operations'],
                next_action='Reject this candidate under frozen resource gates; reviewer/kernel expert inspect the pinned PC and resource evidence. Sector/eligible/traffic/timing stay pending; no gate relaxation or performance claim.',
                completed_at_utc=stamp)
    dump(old_result,result)
    rows='\n'.join(f"| {x['entry'].removeprefix('flagmega_main_program_')} | {x['N']}×{x['K']} | {x['baseline']['registers_per_thread']} → {x['candidate']['registers_per_thread']} | {x['baseline']['ctas_per_sm']} → {x['candidate']['queried_ctas_per_sm']} | {'PASS' if x['gates']['admission'] else 'FAIL'} |" for x in comparison)
    response=f'''r001 H001 候选被 H001-preserve 资源门否决；机制 landing 不构成性能收益。

本 context 为 `codex:/root:kernel_optimizer:implement_r001:resume-20261116T142313Z`，签署实施证据与拒绝资源不合格候选的决定。冻结 v2.1 subject `{planhash}`；源码 subject `{sourcehash}`；UTC `{stamp}`。没有代签 reviewer / expert / profiler，也没有性能验收。

| integrated entry | N×K | regs/thread r000→r001 | CUDA capacity CTA/SM r000→r001 | admission |
| --- | --- | --- | --- | --- |
{rows}

12 个 entry 的 baseline static/dynamic shared 为 0/512 B、cubin reservation 1024 B；candidate 三项均为 0。全部 candidate 的 driver local、cubin local/stack、reported spills 及 PTX/SASS local instructions 为 0。九个六 CTA entry 全部掉到五 CTA；primary GDN GLU 虽仍四 CTA，125 regs 超过独立 107 上限。attention GLU 96 regs/五 CTA 不能替代这些失败。capacity 来自 CUDA occupancy API；实际同时驻留/placement 未执行，仍 pending。

逐 PC source↔TTGIR↔PTX↔SASS 自检通过，覆盖 12 个实际 emitted integrated entries。所有 numeric tile 为 `4x32xf32`/`4x32xbf16,E32`；row/address metadata 为 ERow，loop/return/epilogue 不缩成 `[4]` numeric tile。每个保留 SHFL PC 映射到 source 与 TTGIR loc、PTX 行，记录 register definitions/uses（含 64-bit address halves 和 K-loop backedges），并从实际 SASS 整数 producers 求出全部 128 线程的 source index，与 PTX 全向量逐项相等，且都在 0..31、同一 warp。ptxas 合并的 shuffle 逐条用相同 input/index reaching definitions 映射到对应 SASS PC，没有把丢失映射当作零。SASS 共 {pc_total} 个 SHFL PC；无 lane-predicated full-mask shuffle，K-loop backedge uniform，退出只发生在最后一次 shuffle 之后。全部 entry 0 LDS/STS/local/barrier；数据传输 opcode 仅 warp-local SHFL.IDX/R2UR，row lineage 结合 actual E32 pointer mapping、逐 PC producer chain 保持在 owning warp。独立审核仍 pending；这不是动态 counter 或完整 NUM 通过声明。

payload sector **pending / null**；activation sector **pending / null**；eligible warps **pending / null**。已发生 non-waivable resource stop，因此本次没有启动 NCU、ordinary timing 或完整 NUM，也不以静态 coalescing 算术冒充实测。activation source 请求成本静态为 BF16 1024 B/CTA/K128（r000 256 B），F32 2048 B（r000 512 B）；4× request amplification 不是 DRAM 实测。仅 8 lanes 持有 FP32 leaves，header/source/tree shuffle 的 issue/MIO 与 serial dependency 成本未测；3–6% 仍是归因预算。

当前 gates：静态自检、zero-spill PASS；integrated admission、primary GLU register cap FAIL；sector（payload 与 activation 分开）、eligible、dynamic traffic、NUM/ABI/state、timing/full 与独立审核 pending。K9、K6/K10、B007、G012 均 OPEN。

结果：[kernel-implementation.json](../../../results/kernel-implementation.json)；逐形状表与全部 cubin/IR/SASS：[candidate-vs-baseline.json](../{args.attempt.name}/candidate-vs-baseline.json)；逐 PC：[pc-proof.json](../{args.attempt.name}/{args.pc.name}/pc-proof.json)。三份资源尝试和六份 PC 尝试（含错误与失败）均保留；旧 kernel-implementation.json 已字节保留到 evidence attempt。未重建 compiler，未改 generated implementation、graph-owned 文件、dashboard/task、cuda workspace，也未 commit/stash/revert。
'''
    responsepath=assignment/'response.md';responsepath.write_text(response)
    tracking=revision/'tracking.md'
    signoff=dict(agent_name='kernel_optimizer',agent_id=result['actual_context'],assignment_id='implement_r001',
                 phase='implementation',decision='reject-current-candidate-H001-preserve',signed_at_utc=stamp,
                 milestone_id='m000',revision_id='r001',plan_version='v2.1',plan_hash=planhash,source_hash=sourcehash,
                 artifact_manifest_hash=r['artifact']['sha256'],tracking_snapshot_sequence=None,
                 tracking_snapshot_hash=pin(tracking)['sha256'],
                 null_scope_reason='Read-only tracking snapshot; no owner sequence assigned to this resume.',
                 evidence=[pin(old_result),pin(responsepath),pin(comparepath),pin(args.pc/'pc-proof.json')])
    handoff=dict(schema='flagmega.agent-handoff/v1',milestone_id='m000',revision_id='r001',baseline_id='r000',
                 agent_name='kernel_optimizer',agent_id=result['actual_context'],assignment_id='implement_r001',
                 phase='implementation',status='completed-evidence-candidate-rejected',
                 inputs=dict(plan=pin(subject),artifact=r['artifact'],baseline=pin(baseline_path)),
                 response=pin(responsepath),evidence=signoff['evidence'],findings=result['gates'],
                 open_gates=result['open_gates'],source_review_passed=True,device_verified=False,
                 device_scope='Actual cubin CUDA resource queries; no integrated kernel execution in resume. Primitive inherited device check pinned.',
                 clear_measured_gain=False,performance_accepted=False,signoffs=[signoff],
                 next_recipient='orchestrator',next_action=result['next_action'])
    dump(assignment/'handoff.json',handoff)
    dump(args.attempt/'package-self-checks.json',dict(status='passed',plan_hash=planhash,ortools=ortools.__version__,
                                                  verified_entries=12,verified_zero_spill=12,admission_failures=9,
                                                  primary_GLU_register_failures=1,PC_checks_passed=12,
                                                  result=pin(old_result),response=pin(responsepath),handoff=pin(assignment/'handoff.json')))
    print(json.dumps(dict(result=pin(old_result),handoff=pin(assignment/'handoff.json'),status=result['status']),indent=2))


if __name__=='__main__':
    main()
