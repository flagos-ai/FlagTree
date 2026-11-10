# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A56 explicit Kernel registration layered on the unchanged t23 local recipe.

Only the new semantic operations/implementation IDs use the new validators.
Existing operators retain their original providers/encoders/arena validation.
No buffer, emitted source, library or semantic node is rewritten by registration.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal, TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.implementation import TritonImplementation,TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.candidates.gdn_head_gates import _broadcast,_dense,_ref_derived,_shape,dense_value_geometry
from triton.flagmega.ir import DType,RefType,VectorType,T,tensor_type
from triton.flagmega.ir.execution_mesh import effective_execution_mesh,ExecutionMeshKind
from .u4_q8_abi import require
from .gdn_head_externalq8_qkregs_fullhead48_abi import IMPLEMENTATION,VARIANT,_PARAMETERS,_FACTS,prepare_fields,validate_selection,fullhead_value_geometry

OP='u4.gdn_recurrent_core_q8_c2'
RECIPE='gdn_head_externalq8_qkregs_fullhead48'
_REGISTERED=False
SOURCE_FILES=(
 'u4_common_q8_kernels.py','u4_norm_q8_c1_c2_kernel.py','gdn_head_externalq8_qkregs_fullhead48_abi.py','gdn_head_externalq8_arena.py',
 'templates/kernels/u4_norm_apply_q8_c1_c2.cpp.jinja','tir_metadata/kernels/u4_norm_apply_q8_c1_c2.py.jinja',
 'templates/kernels/gdn_recurrent/'+VARIANT+'.cpp.jinja','tir_metadata/kernels/gdn_recurrent/'+VARIANT+'.py.jinja')


def source_identity():
    root=Path(__file__).parent
    pins={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in SOURCE_FILES}
    return hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()


def implementation():
    return TritonImplementation(IMPLEMENTATION,'gdn_recurrent',VARIANT,
        {**_PARAMETERS,'common_q8_source_identity':source_identity()},
        {'fullhead48_original_order_rms':True,'required_codegen_backend':'cpp'},
        ('cooperative_grid','grid_sync'),_FACTS,
        (T.shared_workspace_descriptor('head_statistics',tensor_type('float32',(2,4)),16),
         T.shared_workspace_descriptor('head_core',tensor_type('float32',(128,)),16)))


def applicable(node,context):
    if (node.op!=OP or len(node.inputs)!=10 or context.codegen_backend!='cpp' or not context.cooperative_grid
        or context.target.codegen_platform!='nvidia' or context.target.codegen_architecture!='sm120'
        or node.metadata.get('kernel_num_warps')!=8):return False
    attrs=node.attrs
    flags={'round_beta':True,'round_core':False,'round_normalized_qk':False,'qk_norm_mode':'clamp',
           'qk_norm_epsilon':1e-12,'gate_math_mode':'accurate','norm_math_mode':'fast'}
    if (attrs.get('b_weight_encoding')!='IQ4_XS' or attrs.get('a_weight_encoding')!='IQ4_XS'
        or any(attrs.get(k,v)!=v for k,v in flags.items())
        or attrs.get('key_head_dim')!=128 or attrs.get('value_head_dim')!=128
        or attrs.get('num_key_heads')!=16 or attrs.get('num_value_heads')!=48
        or attrs.get('value_head_order','grouped')!='tiled'):return False
    out=node.type.fields[0];geometry=fullhead_value_geometry(out,48,128)
    if geometry is None or geometry[0]!=128 or out.placement.hierarchy!=(4,12):return False
    mesh=effective_execution_mesh(context.module,node)
    if mesh is None or mesh.kind!=ExecutionMeshKind.FIXED or mesh.shape!=(4,12) or mesh.mapping!='row_major':return False
    operands=[context.module.node_map[i].type for i in node.inputs]
    if operands[2]!=out:return False
    for index,shape,dtype in ((1,(1,10240),DType.BFLOAT16),(3,(1,5120),DType.BFLOAT16),
        (4,(48,2720),DType.UINT8),(5,(48,2720),DType.UINT8),(6,(48,),DType.FLOAT32),
        (7,(48,),DType.FLOAT32),(8,(128,),DType.FLOAT32),(9,(2,160,36),DType.UINT8)):
        if not _broadcast(operands[index],out.placement,shape,dtype):return False
    if any(_ref_derived(i,context.module) for i in node.inputs[1:]):return False
    state=operands[0]
    if not isinstance(state,RefType):return False
    recurrent=dict(state.fields).get('recurrent')
    return recurrent is not None and recurrent.dtype==VectorType(DType.FLOAT32,(4,)) and _shape(recurrent)==(1,48,128,32) and _dense(recurrent)


class ExternalQ8Provider:
    op_names=frozenset((OP,))
    def propose(self,node,context):
        require(applicable(node,context),'A56 requires exact original GDN input0..8 plus complete input9, fixed4x12/8warps CPP SM120')
        from .u4_gdn_q8_c2 import GDNRecurrentCoreQ8C2
        from .u4_norm_q8_c1_c2_vector import validate_gdn_q8_type
        GDNRecurrentCoreQ8C2.verify(node,context.module)
        validate_gdn_q8_type(context.module.node_map[node.inputs[9]].type)
        choices=[i for i in context.implementations('gdn_recurrent') if i.id==IMPLEMENTATION]
        require(len(choices)==1,'A56 requires its exact distinct implementation')
        candidate=context.configure_implementation(choices[0])
        return TritonCandidateProposal((candidate,),candidate.id)


def _has_new(descriptor):
    contexts=descriptor.get('native_program_contexts') or (descriptor,)
    return any(c.get('implementation')==IMPLEMENTATION for ctx in contexts for c in ctx.get('render_calls',()))


def _mark_recipe(descriptor):
    # Descriptor facts name the actual registered override, retaining original
    # installation lineage separately. No implementation or operand is relabelled.
    recipe=descriptor.get('local_iq4_recipe')
    if recipe is not None:
        if recipe.get('gdn_head_projection')!=RECIPE:
            recipe['base_gdn_head_projection']=recipe.get('gdn_head_projection')
        recipe['gdn_head_projection']=RECIPE
        recipe['common_q8_kernel_source_identity']=source_identity()


def register(target=None):
    global _REGISTERED
    from triton.flagmega.targets import get_target
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders,tir_package
    from triton.flagmega.codegen.triton.physical_access import emit_logical_coordinate
    from triton.flagmega.codegen.cpp import renderer
    from jinja2 import ChoiceLoader,FileSystemLoader
    import local_optimizations as recipe_module
    require(recipe_module._CONFIG is not None and recipe_module._CONFIG.get('gdn_head_projection')=='gdn_head_globalq8_qkregs_fullhead48',
            'A56 registration follows the exact installed t23 fullhead48 base recipe')
    target=target or get_target('nvidia-sm120')
    if _REGISTERED:
        require(target.triton_implementation_model.implementation(IMPLEMENTATION) is not None,'A56 target lost registered implementation')
        return target
    policy=target.tir_selection_policy;model=target.triton_implementation_model
    require(model.implementation(IMPLEMENTATION) is None,'A56 duplicate implementation')
    registry=TritonCandidateProviderRegistry()
    for provider in policy.registry.providers:registry.add(provider)
    registry.add(ExternalQ8Provider())
    target.triton_implementation_model=TritonImplementationModel((*model.implementations,implementation()),model.preferences,model.name+'+A56-external-C2-v1')
    def workspace(node,candidates,module,*,mesh_hierarchy):
        result=policy.workspace_annotator(node,candidates,module,mesh_hierarchy=mesh_hierarchy)
        for c in result:
            if c.id==IMPLEMENTATION:
                require(tuple(w['name'] for w in c.parameters.get('workspaces',()))==('core_scratch',),
                        'A56 requires only the original explicit core scratch; no internal Q8 allocation')
        return result
    target.tir_selection_policy=TritonTirSelectionPolicy(workspace,registry)
    original_encoder=encoders._FAMILY_ENCODERS['gdn_recurrent']
    def encode(raw):
        result=original_encoder(raw)
        if raw.get('implementation')!=IMPLEMENTATION:return result
        require(raw.get('parameters',{}).get('common_q8_source_identity')==source_identity(),'A56 source identity changed')
        result['shared_workspaces']=raw.get('shared_workspaces',())
        result.update(prepare_fields(raw,result))
        out=next(x for x in raw['outputs'] if x['formal']=='result_0')['buffers'][0]
        result['head_statistics_first_value']=emit_logical_coordinate(out['abi'],1,('0','0'))
        return result
    encoders._FAMILY_ENCODERS['gdn_recurrent']=encode
    from .u4_norm_q8_c1_c2_kernel import enable as enable_norm,validate_context
    enable_norm(target)
    root=Path(__file__).parent;cpp_base=renderer.CppTemplateRegistry;tir_base=tir_package.TritonTemplateRegistry
    class CommonQ8Cpp(cpp_base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.environment.loader=ChoiceLoader([FileSystemLoader(str(root/'templates')),self.environment.loader])
        def available(self,spec):return spec.kernel=='gdn_recurrent' and spec.variant==VARIANT or super().available(spec)
        def resolve(self,spec):
            if spec.kernel=='gdn_recurrent' and spec.variant==VARIANT:
                require((spec.platform,spec.architecture)==('nvidia','sm120'),'A56 CPP SM120 only')
                return 'kernels/gdn_recurrent/'+VARIANT+'.cpp.jinja'
            return super().resolve(spec)
    class CommonQ8Metadata(tir_base):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.environment.loader=ChoiceLoader([FileSystemLoader(str(root/'tir_metadata')),self.environment.loader])
        def resolve(self,spec):
            if spec.kernel=='gdn_recurrent' and spec.variant==VARIANT:
                require((spec.platform,spec.architecture)==('nvidia','sm120'),'A56 metadata CPP SM120 only')
                return 'kernels/gdn_recurrent/'+VARIANT+'.py.jinja'
            return super().resolve(spec)
    renderer.CppTemplateRegistry=CommonQ8Cpp;tir_package.TritonTemplateRegistry=CommonQ8Metadata
    # Explicit new-identity dispatch at the existing local recipe extension
    # points. Old inputs retain their unchanged validators, never new aliases.
    from . import gdn_head_globalq8_arena as old_arena
    from . import gdn_head_globalq8_qkregs_fullhead48_abi as old_abi
    from .gdn_head_externalq8_arena import configure_descriptor as new_arena
    base_arena=old_arena.configure_descriptor;base_validate=old_abi.validate_selection
    def arena_dispatch(descriptor):
        if not _has_new(descriptor):return base_arena(descriptor)
        _mark_recipe(descriptor)
        return new_arena(descriptor)
    def validate_dispatch(descriptor):
        if not _has_new(descriptor):return base_validate(descriptor)
        _mark_recipe(descriptor)
        return validate_selection(descriptor)
    old_arena.configure_descriptor=arena_dispatch;old_abi.validate_selection=validate_dispatch
    original_describe=renderer.describe_cpp_package
    def describe(module):
        descriptor=original_describe(module)
        for context in descriptor.get('native_program_contexts') or (descriptor,):
            validate_context(context)
        if _has_new(descriptor):
            _mark_recipe(descriptor)
            descriptor['common_q8_kernel_override']={'schema':'flagmega.A56-kernel-override/v1',
                'source_identity':source_identity(),'norm_implementation':'tir.local_u4.norm_apply_q8_c1_c2.k5120.c2.warp4.v1',
                'gdn_implementation':IMPLEMENTATION,'ordinary_only':True,'microprofile_pending':True}
        return descriptor
    renderer.describe_cpp_package=describe
    _REGISTERED=True
    return target
