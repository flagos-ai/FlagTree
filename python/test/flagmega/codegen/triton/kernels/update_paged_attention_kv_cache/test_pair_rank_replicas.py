# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Every chip receives a complete pair in its own CHIP-scoped cache."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from python.test.flagmega.devices import has_nvidia_sm90_devices


def _worker(artifact):
    import torch
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import load_artifact
    from triton.flagmega.evaluator import PagedAttentionStateConfig, create_paged_attention_state
    from triton.flagmega.runtime import load

    from python.test.flagmega.ir.ops.nn.test_update_paged_attention_kv_cache_pair import physical_input, state_arguments

    device = f"cuda:{os.environ['LOCAL_RANK']}"
    torch.cuda.set_device(device)
    _, module = load_artifact(artifact)
    runtime = load(artifact, device=device)
    try:
        config = PagedAttentionStateConfig(3, 7, 32, block_size=16, num_blocks=4)
        state = create_paged_attention_state(config, device=device)
        state.kv_caches.fill_(-9)
        state.block_table.copy_(torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device=device))
        semantic, slots = physical_input(config, 3, (2, 4), ('head', 'dim', 'seq'), device=device)
        bindings = runtime.buffer_plan.function_map[module.entry]
        buffers = {}
        for value, names in bindings.parameters:
            buffers.update(zip(names, state_arguments(state), strict=True) if isinstance(module.node_map[value].type, fm.RefType)
                           else [(names[0], slots)])
        for _, names in bindings.outputs:
            buffers.update(zip(names, state_arguments(state), strict=True))
        args = [buffers[str(arg['buffer'])] for arg in runtime.external_arguments]
        runtime.prepare(*args)
        expected = state.kv_caches.clone()
        for position in (13, 15, 31, 34):
            state.seq_lens.fill_(position)
            runtime.run_into(*args)
            for row in range(3):
                page = (2, 0, 3, 1)[(position+row)//16]
                for kind in range(2):
                    expected[page, 1, kind, (position+row)%16].copy_(semantic[row, :, kind*32:(kind+1)*32].reshape(7, 4, 8))
            torch.testing.assert_close(state.kv_caches, expected, rtol=0, atol=0)
            assert state.seq_lens.item() == position+3
            assert state.slot_mapping.item() == position
            assert state.query_start_loc.tolist() == [0, 3]
    finally:
        runtime.close()


@pytest.mark.parametrize('levels', ['cbb', 'bcb'])
def test_pair_publication_preserves_each_chip_cache_replica(tmp_path, levels):
    pytest.importorskip('torch')
    if not has_nvidia_sm90_devices(2):
        pytest.skip('Two SM90 CUDA devices required')
    from triton.flagmega import ir as fm
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.compiler import Compiler
    from triton.flagmega.evaluator import PagedAttentionStateConfig

    from python.test.flagmega.ir.ops.nn.test_update_paged_attention_kv_cache_pair import PairModule

    source = PairModule(PagedAttentionStateConfig(3, 7, 32, block_size=16, num_blocks=4), lanes=(2, 4),
                        layout=('head', 'dim', 'seq'), placement=fm.Placement((2, 2, 2), 'abc', levels),
                        split_axis='head', block_cyclic=True).build()
    module = Compiler().compile(source).module
    artifact = write_artifact(module, tmp_path/'pair', target='nvidia-sm90', emit_executable=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).parents[7]) + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                             str(Path(__file__).resolve()), str(artifact)], env=env, capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == '__main__':
    _worker(sys.argv[1])
