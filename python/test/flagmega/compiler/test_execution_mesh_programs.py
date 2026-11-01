# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Normal compilation must preserve execution domains through optimization."""
import ast
import importlib.util
from dataclasses import replace

import pytest
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.targets import NvidiaSm90Target


def graph(*, dynamic_kind='persistent', dynamic_warps=None, dynamic_max_ctas=None,
          dynamic_maxnreg=None, second_dynamic=False, work_size=4096, annotate=True):
    class Graph(fm.Module):
        def forward(self):
            value_type = fm.tensor_type('float32', (work_size,))
            x = self.input('x', value_type)
            y = self.input('y', value_type)
            first = {'execution_mesh': fm.ExecutionMesh.fixed((8, 16))} if annotate else {}
            a = fm.F.math.add(x, y, name='a', metadata=first)
            b = fm.F.math.add(a, y, name='b', metadata=first)
            c = fm.F.math.add(b, x, name='c', metadata={
                'execution_mesh': '*', 'kernel_program_kind': dynamic_kind,
                **({'kernel_num_warps': dynamic_warps} if dynamic_warps is not None else {}),
                **({'kernel_max_ctas': dynamic_max_ctas} if dynamic_max_ctas is not None else {}),
                **({'kernel_maxnreg': dynamic_maxnreg} if dynamic_maxnreg is not None else {}),
            } if annotate else {})
            if second_dynamic:
                c = fm.F.math.add(c, x, name='c2', metadata={'execution_mesh': '*',
                                                         'kernel_program_kind': dynamic_kind})
            d = fm.F.math.add(c, y, name='d', metadata={
                'execution_mesh': fm.ExecutionMesh.fixed((4, 16)),
            } if annotate else {})
            self.function('main', (x, y), (d,))

    return Graph(dialect='nn', stage='imported', entry='main').build()


def compile_graph(**options):
    compiler = Compiler()
    compiler.target = NvidiaSm90Target()
    return compiler.compile(graph(**options)).module


@pytest.mark.parametrize('boundary', ['propose-distribution', 'distributed'])
def test_mesh_chain_consumes_operand_pack_inside_vector_arithmetic(boundary):
    module = Compiler().compile(graph(), stop_after=boundary).module
    forbidden = {'tensors.pack', 'tensors.slice', 'tensors.slice_to_shape', 'tensors.cast',
                 'ntt.vectorized_cast', 'tensors.concat', 'tensors.unpack'}
    assert not any(node.op in forbidden for node in module.nodes)
    compute = [n for n in module.nodes if n.op == 'math.vectorized_binary']
    assert len(compute) == 4
    assert all(isinstance(n.type.tensor.dtype if isinstance(n.type, fm.DistributedType) else n.type.dtype,
                          fm.VectorType) for n in compute)


@pytest.mark.parametrize('boundary', ['propose-distribution', 'distributed'])
def test_ragged_mesh_chain_owns_its_load_and_store_tail_contracts(boundary):
    module = Compiler().compile(graph(work_size=4103), stop_after=boundary).module
    forbidden = {'tensors.pack', 'tensors.pad', 'tensors.slice', 'tensors.slice_to_shape',
                 'tensors.cast', 'ntt.vectorized_cast', 'tensors.concat', 'tensors.unpack'}
    assert not any(n.op in forbidden for n in module.nodes)
    compute = [n for n in module.nodes if n.op == 'math.vectorized_binary']
    assert len(compute) == 4
    assert all(n.attrs['result_unpack']['shape'] == (4103,) for n in compute)


def test_agent_edits_a_uniform_search_ir_then_reproposes_and_resumes(tmp_path):
    from triton.flagmega.passes.auto_distributed.auto_distributed import AutoDistributedPass
    compiler = Compiler()
    proposal = compiler.compile(graph(annotate=False), stop_after='propose-distribution').module
    assert not any(fm.node_execution_mesh(n) for n in proposal.nodes)
    replacements = {}
    for node in proposal.nodes:
        root = node.metadata.get('vectorization_root', node.id)
        if node.op.startswith('math.') and root in ('a', 'b', 'c', 'd'):
            mesh = '*' if root == 'c' else (4, 16) if root == 'd' else (8, 16)
            replacements[node.id] = fm.with_execution_mesh(node, mesh)
    assert len(replacements) == 4
    edited = replace(proposal, nodes=tuple(replacements.get(n.id, n) for n in proposal.nodes))
    # Distribution candidates from the uniform search are regenerated, rather
    # than keeping their old physical owner configuration or rewriting source.
    edited = AutoDistributedPass.propose(edited, compiler.target)
    checkpoint = fm.emit_module(edited, tmp_path / 'edited_proposal.py')
    module = Compiler().compile(fm.load_module(checkpoint)).module
    assert [p['grid'] for p in describe_tir_package(module)['kernels']] == [[128, 1, 1], [32, 1, 1], [64, 1, 1]]


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_normal_compile_partitions_fixed_dynamic_fixed(kind):
    module = compile_graph(dynamic_kind=kind)
    programs = describe_tir_package(module)['kernels']
    assert len(programs) == 3
    assert [p['execution_mesh']['kind'] for p in programs] == ['fixed', 'dynamic', 'fixed']
    assert programs[1]['program_kind'] == kind
    assert len(programs[1]['calls']) == 1
    assert [p['grid'] for p in programs] == [[128, 1, 1], [32, 1, 1], [64, 1, 1]]
    assert programs[1]['data_placement']['hierarchy'] == [1, 1]


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_adjacent_dynamic_calls_never_fuse(kind):
    module = compile_graph(dynamic_kind=kind, second_dynamic=True)
    programs = describe_tir_package(module)['kernels']
    assert len(programs) == 4
    assert all(len(programs[i]['calls']) == 1 for i in (1, 2))


def test_render_and_reload_use_each_programs_actual_mesh(tmp_path):
    from triton.flagmega.codegen.triton.renderer import render_triton_package
    from triton.flagmega.runtime.kernel_programs import verified_kernel_programs
    module = compile_graph(dynamic_warps=2)
    package = render_triton_package(module, tmp_path)
    spec = importlib.util.spec_from_file_location('_mesh_program_test', tmp_path / 'generated_kernels.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    kernels = tuple(getattr(source, p['symbol']) for p in package['kernels'])
    assert [p['num_warps'] for p in verified_kernel_programs(package, kernels, module)] == [4, 2, 4]
    tree = ast.parse((tmp_path / 'generated_kernels.py').read_text())
    shapes = []
    for builder in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        topology = next(n.value.args[0] for n in builder.body if isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                        and n.value.func.attr == 'device_mesh')
        shapes.append(tuple(size for name, size in ast.literal_eval(topology)['block']))
    assert shapes == [(8, 16), (1, 1), (4, 16)]
    reloaded = fm.load_module(fm.emit_module(module, tmp_path / 'checkpoint.py'))
    assert reloaded.semantic_hash == module.semantic_hash
    assert describe_tir_package(reloaded)['kernels'] == package['kernels']


def test_dynamic_boundary_storage_is_chip_shared():
    module = compile_graph()
    plan = fm.verify_buffer_plan(module)
    dynamic = next(n for n in module.nodes if n.metadata.get('kernel_program_kind') == 'persistent'
                   and n.metadata.get('execution_mesh', {}).get('kind') == 'dynamic')
    buffers = [b for b in plan.buffers if b.source_node == dynamic.id]
    assert buffers
    for buffer in buffers:
        space = next(s for s in plan.memory_spaces if s.name == buffer.mem_span.buffer.memory_space)
        assert space.sharing_scope.value == 'chip'


def test_dynamic_request_cannot_merge_another_call():
    from triton.flagmega.codegen.triton.kernel_programs import target_kernel_programs
    from triton.flagmega.errors import IRVerificationError
    module = compile_graph(second_dynamic=True)
    programs = [dict(p) for p in module.metadata['kernel_program_partition']]
    programs[1]['calls'] = (*programs[1]['calls'], *programs[2]['calls'])
    invalid = replace(module, metadata={**module.metadata, 'kernel_program_partition': programs})
    with pytest.raises(IRVerificationError, match='Dynamic programs never fuse'):
        target_kernel_programs(invalid, NvidiaSm90Target().capability)


def test_same_block_product_different_mapping_stays_separate():
    module = graph(annotate=False)
    nodes = []
    for node in module.nodes:
        if node.op == 'math.add':
            mesh = (8, 16) if node.id in ('a', 'b') else (16, 8)
            node = fm.with_execution_mesh(node, mesh)
        nodes.append(node)
    compiled = Compiler().compile(replace(module, nodes=tuple(nodes))).module
    programs = describe_tir_package(compiled)['kernels']
    assert len(programs) == 2
    assert [p['grid'] for p in programs] == [[128, 1, 1], [128, 1, 1]]
    assert programs[0]['execution_mesh']['shape'] == [8, 16]
    assert programs[1]['execution_mesh']['shape'] == [16, 8]


def test_cannot_forge_a_different_task_count_with_the_same_public_pointers():
    from triton.flagmega.codegen.triton.package_plan import plain_package_value
    from triton.flagmega.errors import CodegenError
    module = compile_graph()
    plan = plain_package_value(module.metadata['codegen_package_plan'])
    requests = plan['programs']
    requests[1]['launch_recipe']['tile_size'] *= 2
    requests[1]['grid'] = [16, 1, 1]
    changed = replace(module, metadata={**module.metadata, 'kernel_program_partition': requests,
                                       'codegen_package_plan': plan})
    with pytest.raises(CodegenError, match='task/tail coverage'):
        describe_tir_package(changed)


def test_program_register_control_requires_backend_capability():
    from types import SimpleNamespace
    from triton.flagmega.codegen.triton.kernel_programs import target_kernel_programs
    from triton.flagmega.errors import IRVerificationError
    module = compile_graph(dynamic_maxnreg=64)
    unsupported = SimpleNamespace(warp_size=32, max_threads_per_block=1024,
                                  supported_program_compile_options=())
    with pytest.raises(IRVerificationError, match='register cap is unsupported'):
        target_kernel_programs(module, unsupported)


@pytest.mark.parametrize('stop', [0, 1, 3])
def test_uniform_fixed_loop_stays_a_whole_execution_region(stop):
    from python.test.flagmega.ir.ops.test_for import loop_module
    compiler = Compiler()
    module = compiler.compile(fm.with_execution_mesh(loop_module(stop=stop), fm.ExecutionMesh.fixed((8, 16)))).module
    description = describe_tir_package(module)
    assert len(description['kernels']) == 1
    assert any(event['kind'] == 'loop' for event in description['programs'][0]['entry_events'])


@pytest.mark.parametrize('stop', [0, 1, 3])
def test_device_uniform_fixed_loop_preserves_recurrence(tmp_path, stop):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    from python.test.flagmega.ir.ops.test_for import loop_module
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    module = Compiler().compile(fm.with_execution_mesh(loop_module(stop=stop), (8, 16))).module
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90', emit_executable=True), device='cuda:0')
    value = torch.arange(4, dtype=torch.float32, device='cuda')
    increment = torch.tensor([1., 2., 3., 4.], device='cuda')
    output = torch.full_like(value, float('nan'))
    try:
        args = tuple(output if s['role'] == 'result' else {'value': value, 'increment': increment}[s['value']]
                     for s in runtime.external_arguments)
        Runtime.prepare(runtime, *args)
        Runtime.run_into(runtime, *args)
        torch.cuda.synchronize()
        assert torch.equal(output, value + stop * increment)
    finally:
        runtime.close()


@pytest.mark.parametrize('kind,work_size,cap', [('ordinary', 4096, None), ('persistent', 4096, 7),
                                             ('ordinary', 131072, None), ('persistent', 4103, 2)])
def test_device_mesh_cuts_preserve_values_rebinding_and_graph(tmp_path, kind, work_size, cap):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip('requires SM90')
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.runtime import load
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule as Runtime
    module = compile_graph(dynamic_kind=kind, dynamic_warps=2, dynamic_max_ctas=cap,
                           dynamic_maxnreg=64, work_size=work_size)
    runtime = load(write_artifact(module, tmp_path / 'artifact', target='nvidia-sm90', emit_executable=True),
                   device='cuda:0')
    generator = torch.Generator(device='cuda').manual_seed(20261027)
    values = [torch.empty((work_size,), device='cuda') for _ in range(2)]
    guard = torch.full((work_size + 32,), -123.0, device='cuda')
    output = guard[16:-16]

    def arguments():
        inputs = iter(values)
        return tuple(output if spec['role'] == 'result' else next(inputs) for spec in runtime.external_arguments)

    try:
        Runtime.prepare(runtime, *arguments())
        from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
        recipe = TaskLaunchRecipe.from_data(runtime.codegen['kernels'][1]['launch_recipe'])
        expected_grid = recipe.resolve(sm_count=int(torch.cuda.get_device_properties(0).multi_processor_count))
        assert tuple(runtime._prepared[1].grid) == expected_grid
        assert tuple(runtime._prepared[0].grid) == (128, 1, 1)
        assert tuple(runtime._prepared[-1].grid) == (64, 1, 1)
        if work_size % 4 == 0:
            assert len(runtime._prepared) == 3
        # Ragged vectorization currently adds pad/slice compute. The device
        # test covers its values/tail; structural IR acceptance stays pending.
        for mutation in range(6):
            for value in values:
                value.copy_(torch.randint(-16, 17, value.shape, generator=generator, device='cuda').float() / 16)
            output.fill_(float('nan'))
            Runtime.run_into(runtime, *arguments())
            torch.cuda.synchronize()
            gold = ((values[0] + values[1]) + values[1]) + values[0]
            gold = gold + values[1]
            assert torch.equal(output, gold)
            assert bool((guard[:16] == -123).all() & (guard[-16:] == -123).all())
            if mutation == 2:
                values = [value.clone() for value in values]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Compiler global scratch belongs to one prepared stream instance.
        # Prepare and warm the capture instance on its stream before capture.
        with torch.cuda.stream(stream):
            Runtime.prepare(runtime, *arguments())
            Runtime.run_into(runtime, *arguments())
        with torch.cuda.graph(capture := torch.cuda.CUDAGraph(), stream=stream):
            Runtime.run_into(runtime, *arguments())
        capture.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, gold)
        assert all(report['spill_bytes'] == 0 for report in runtime.resource_report)
        assert runtime.resource_report[1]['registers_per_thread'] <= 64
    finally:
        runtime.close()
