# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Form execution programs from edited IR, before allocation and scheduling."""
from collections.abc import Mapping
from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import DistributedType, TupleType, local_tensor_type
from triton.flagmega.ir.distributed_type import placement_from_data
from triton.flagmega.ir.execution_mesh import (
    ExecutionMesh, ExecutionMeshKind, KernelProgramKind, effective_execution_mesh,
    node_execution_mesh, placement_for_execution_mesh,
)
from triton.flagmega.passes.functions.graph import function_nodes
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe


def selected_task_recipe(node):
    mapping = node.attrs.get('parameters', {}).get('task_mapping')
    if mapping is None:
        return None
    # Implementations may own an explicit task domain independently of the
    # public tensor representation (for example scalar rows of packed GEMV).
    if isinstance(mapping, Mapping) and 'work_size' in mapping:
        return TaskLaunchRecipe.from_data(mapping)
    if (not isinstance(mapping, Mapping) or set(mapping) != {'schema', 'program_kind', 'max_ctas', 'ctas_per_sm'}
            or mapping.get('schema') != 'flagmega.independent-linear-tiles/v1'):
        raise IRVerificationError('Selected task mapping has an unsupported contract.', node_id=node.id)
    tensor = local_tensor_type(node.type) if isinstance(node.type, DistributedType) else node.type
    if any(not dim.is_fixed for dim in tensor.shape):
        raise IRVerificationError('Selected linear tiles require statically bounded task coverage.', node_id=node.id)
    from triton.flagmega.codegen.triton.vectorization import elementwise_work_size
    work_size = elementwise_work_size(tensor, node.attrs['parameters']['vector_schedule']['contract'])
    tile = node.attrs['parameters']['vector_schedule']['physical']['elements_per_program']
    return TaskLaunchRecipe(work_size, tile, mapping['program_kind'], mapping['max_ctas'], mapping['ctas_per_sm'])


def _region_members(module, node, kernels):
    if node.id in kernels:
        return (kernels[node.id],)
    callee = module.function_map.get(str(node.attrs.get('callee', '')))
    if callee is None:
        return ()
    return tuple(member for child in function_nodes(module, callee)
                 if child.op in {'tir.call', 'control.for'}
                 for member in _region_members(module, child, kernels))


def _execution_policy(module, node, default):
    mesh = effective_execution_mesh(module, node) or default
    if node_execution_mesh(node) is None and mesh.kind is ExecutionMeshKind.FIXED:
        # Boundary planners can introduce a selected kernel in a different
        # owner domain without a user policy annotation. Its typed result
        # fixes those owners; a module default is only an inherited choice.
        def result_placements(value):
            if isinstance(value, DistributedType):
                return (value.placement,)
            if isinstance(value, TupleType):
                return tuple(p for field in value.fields for p in result_placements(field))
            return ()
        placements = result_placements(node.type)
        if placements and all(p == placements[0] for p in placements):
            shape = tuple(size for size, level in zip(placements[0].hierarchy,
                          placements[0].hierarchy_levels, strict=True) if level == 'b') or (1,)
            if mesh.shape != shape:
                mesh = ExecutionMesh.fixed(shape)
    kind = KernelProgramKind(node.metadata.get('kernel_program_kind', module.metadata.get('kernel_program_kind',
        'ordinary' if mesh.kind is ExecutionMeshKind.DYNAMIC else 'persistent')))
    warps = node.metadata.get('kernel_num_warps')
    if warps is not None and (type(warps) is not int or warps not in (1, 2, 4, 8, 16, 32)):
        raise IRVerificationError('kernel_num_warps must be a supported positive power of two.',
                                  stage=module.stage, node_id=node.id)
    maxnreg = node.metadata.get('kernel_maxnreg')
    if maxnreg is not None and (type(maxnreg) is not int or maxnreg < 8 or maxnreg > 256 or maxnreg % 8):
        raise IRVerificationError('kernel_maxnreg must be a multiple of eight in [8, 256].',
                                  stage=module.stage, node_id=node.id)
    return mesh, kind, warps, maxnreg


def _root_unit(module, node, kernels, default, function_name=None):
    if node.id in kernels:
        return node.id, _execution_policy(module, node, default), (kernels[node.id],)
    if node.op not in {'tir.call', 'control.for'}:
        return None
    members = _region_members(module, node, kernels)
    if not members:
        raise IRVerificationError('Execution region has no selected call closure.', stage=module.stage, node_id=node.id)
    policies = tuple(_execution_policy(module, member, default) for member in members)
    first = policies[0]
    explicit = node_execution_mesh(node)
    if explicit is not None and any(explicit != policy[0] for policy in policies):
        raise IRVerificationError('Execution region policy differs from its call closure; replan distribution.',
                                  stage=module.stage, node_id=node.id)
    if first[0].fusion_key is None or any(policy != first for policy in policies[1:]):
        from triton.flagmega.ir.program_execution import CallInvocation, loop_invocation
        return (loop_invocation(module, node) if node.op == 'control.for'
                else CallInvocation(node.id, str(node.attrs['callee'])))
    # Match the typed For identity used by describe_function_schedule. Preserve
    # the complete region, its backedge and its invocation count; no unrolling.
    identity = f'{function_name or module.entry}.{node.id}_iv' if node.op == 'control.for' else node.id
    return identity, first, members


def plan_execution_programs(module, target, kernels):
    """Keep legacy uniform search, applying explicit execution edits as cuts.

    Requests are attached before buffer planning. They cannot be reconstructed
    by splitting generated source after allocation. Explicit agent partitions
    remain authoritative and are verified by the target package planner.
    """
    if 'kernel_program_partition' in module.metadata:
        return module
    if not any(effective_execution_mesh(module, node) is not None for node in module.nodes):
        return module
    launch = module.metadata.get('launch_contract', {})
    data = launch.get('grid_mesh')
    if not isinstance(data, Mapping):
        raise IRVerificationError('Execution mesh edits require a selected data placement.', stage=module.stage)
    placement = placement_from_data(data)
    default = ExecutionMesh.fixed(tuple(size for size, level in zip(
        placement.hierarchy, placement.hierarchy_levels, strict=True) if level == 'b') or (1,))
    by_id = {node.id: node for node in kernels}
    from triton.flagmega.ir.program_execution import (
        CallInvocation, LoopInvocation, ProgramInvocation, FunctionProgramExecution, ProgramExecution,
    )
    programs = []
    functions, active = {}, set()
    structured = False

    def emit_group(function, group):
        mesh, nodes = group['mesh'], group['nodes']
        domain = placement_for_execution_mesh(placement, mesh)
        warps = group['key'][2] or target.plan_launch(module, nodes)['num_warps']
        recipe = selected_task_recipe(nodes[0]) if mesh.kind is ExecutionMeshKind.DYNAMIC else None
        symbol = f'flagmega_{module.entry}_program_{len(programs)}'
        programs.append({
            'symbol': symbol,
            'function': function,
            'calls': tuple(group['calls']),
            'num_warps': warps,
            'grid': recipe.grid_upper_bound if recipe else (domain.physical_level_size('b'), 1, 1),
            'execution_mesh': mesh.to_data(),
            'program_kind': group['kind'].value,
            'data_placement': domain.to_data(),
            **({'maxnreg': group['key'][3]} if group['key'][3] is not None else {}),
            **({'launch_recipe': recipe.to_data()} if recipe is not None else {}),
        })
        return ProgramInvocation(symbol)

    def plan_function(name):
        nonlocal structured
        if name in active:
            raise IRVerificationError('Structured execution cannot plan recursive functions.', stage=module.stage)
        if name in functions:
            return
        active.add(name)
        functions[name] = None
        steps, group = [], None
        for node in function_nodes(module, module.function_map[name]):
            unit = _root_unit(module, node, by_id, default, name)
            if unit is None:
                continue
            if isinstance(unit, (LoopInvocation, CallInvocation)):
                structured = True
                if group is not None:
                    steps.append(emit_group(name, group))
                    group = None
                steps.append(unit)
                plan_function(unit.callee)
                continue
            identity, (mesh, kind, requested_warps, maxnreg), members = unit
            key = (mesh.fusion_key, kind, requested_warps, maxnreg)
            if group is None or mesh.fusion_key is None or group['key'] != key:
                if group is not None:
                    steps.append(emit_group(name, group))
                group = {'key': key, 'mesh': mesh, 'kind': kind, 'nodes': [], 'calls': []}
            group['nodes'].extend(members)
            group['calls'].append(identity)
        if group is not None:
            steps.append(emit_group(name, group))
        functions[name] = FunctionProgramExecution(name, tuple(steps))
        active.remove(name)

    plan_function(module.entry)
    if not programs:
        raise IRVerificationError('Execution mesh edits produced no executable programs.', stage=module.stage)
    metadata = {**module.metadata, 'kernel_program_partition': programs}
    if structured:
        metadata['kernel_program_execution'] = ProgramExecution(module.entry, tuple(functions.values())).to_data()
    else:
        # Uniform fixed regions retain their existing flat invocation ABI.
        metadata['kernel_program_partition'] = [{k: v for k, v in p.items() if k != 'function'} for p in programs]
    return replace(module, metadata=metadata)


def verify_program_mesh(module, request):
    """A request cannot merge dynamic calls or reinterpret stale owner types."""
    from triton.flagmega.ir.execution_mesh import execution_mesh
    mesh = execution_mesh(request['execution_mesh'])
    domain = placement_from_data(request['data_placement'])
    if mesh.kind is ExecutionMeshKind.DYNAMIC and len(request['calls']) != 1:
        raise IRVerificationError('Dynamic programs never fuse with other kernel calls.', stage=module.stage)
    if 'launch_recipe' in request:
        recipe = TaskLaunchRecipe.from_data(request['launch_recipe'])
        if (mesh.kind is not ExecutionMeshKind.DYNAMIC or recipe.program_kind.value != request['program_kind']
                or tuple(request.get('grid', ())) != recipe.grid_upper_bound
                or domain.physical_level_size('b') != 1):
            raise IRVerificationError('Dynamic launch recipe differs from its program/data contract.', stage=module.stage)
    if mesh.kind is ExecutionMeshKind.FIXED and domain.physical_level_size('b') != mesh.fixed_size:
        raise IRVerificationError('Program data mapping differs from its fixed execution mesh.', stage=module.stage)

    def placements(value):
        if isinstance(value, DistributedType):
            return (value.placement,)
        if isinstance(value, TupleType):
            return tuple(p for field in value.fields for p in placements(field))
        return ()

    function_name = request.get('function', module.entry)
    if function_name not in module.function_map:
        raise IRVerificationError('Execution program references an unknown function scope.', stage=module.stage)
    owned = function_nodes(module, module.function_map[function_name])
    owned_ids = {n.id for n in owned}
    for call in request['calls']:
        node = module.node_map.get(call)
        if node is None:
            node = next((n for n in owned
                         if n.op == 'control.for' and call == f'{function_name}.{n.id}_iv'), None)
        if node is None or node.id not in owned_ids:
            raise IRVerificationError('Execution program references an unknown scoped call.', stage=module.stage)
        explicit = node_execution_mesh(node)
        if explicit is not None and explicit != mesh:
            raise IRVerificationError('Program mesh differs from its call policy; resume before distribution.',
                                      stage=module.stage, node_id=call)
        kind = node.metadata.get('kernel_program_kind')
        if kind is not None and kind != request['program_kind']:
            raise IRVerificationError('Program kind differs from its explicit call policy.', stage=module.stage, node_id=call)
        if any(p != domain for p in placements(node.type)):
            raise IRVerificationError('Program owner ABI is stale; resume before distribution and allocation.',
                                      stage=module.stage, node_id=call)
    return mesh, domain
