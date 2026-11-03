"""Share full tensor mHC mix bodies through a scalar pointer ABI."""
import ast,copy
from triton.flagmega.errors import CodegenError
from .sigmoid_unit_reciprocal import SYMBOL as UNIT_RECIPROCAL

FIELDS=('projection_partials','base','scale','post','comb')
FORMALS=('Partials','Bias','Scales','Post','Comb')
SYMBOL='_flagmega_shared_complete_mhc_mix'

def normalized_body(fn,call):
    first=next((i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='_fm_i'),None)
    if first is None:raise CodegenError('Original mix coordinate/body boundary changed')
    mappings={ast.dump(ast.parse(call[k],mode='eval').body,include_attributes=False):v for k,v in zip(FIELDS,FORMALS,strict=True)}
    if len(mappings)!=len(FIELDS):raise CodegenError('Distinct mHC input/output views required')
    class Views(ast.NodeTransformer):
        def visit(self,node):
            name=mappings.get(ast.dump(node,include_attributes=False))
            if name is not None:return ast.Name(id=name,ctx=ast.Load())
            return super().visit(node)
    body=[Views().visit(copy.deepcopy(n)) for n in fn.body[first:]]
    allowed=set(FORMALS)|{'_fm_cta','_fm_ctas','tl','tle','libdevice','float',UNIT_RECIPROCAL}
    definitions={n.id for node in body for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    used={n.id for node in body for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    if used-definitions-allowed:raise CodegenError('Unbound mHC view after normalization: '+str(used-definitions-allowed))
    return first,body

def share_source(source,calls):
    # Preserve source text outside the two selected helpers. These are normal
    # rendered pointer views from the complete typed call ABI.
    selected={c['symbol']+'__mix':c for c in calls if c['family']=='mhc_pre' and c['variant']=='cooperative_stream_fma'}
    if not selected:return source,[]
    tree=ast.parse(source);changes=[];receipts=[]
    for scope in ast.walk(tree):
        if not isinstance(scope,(ast.Module,ast.FunctionDef)):continue
        functions=[n for n in scope.body if isinstance(n,ast.FunctionDef) and n.name in selected]
        if len(functions)<2:continue
        uses_reciprocal=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                            and n.func.id==UNIT_RECIPROCAL for fn in functions for n in ast.walk(fn))
        if uses_reciprocal and not any(isinstance(n,ast.FunctionDef) and n.name==UNIT_RECIPROCAL
                                       for n in scope.body):
            raise CodegenError('Unit reciprocal helper must be declared in the shared mix scope')
        records=[(fn,selected[fn.name],*normalized_body(fn,selected[fn.name])) for fn in functions]
        canonical=ast.dump(ast.Module(body=records[0][3],type_ignores=[]),include_attributes=False)
        if any(ast.dump(ast.Module(body=r[3],type_ignores=[]),include_attributes=False)!=canonical for r in records[1:]):
            raise CodegenError('Full mHC math/geometry differs after only pointer normalization')
        template=copy.deepcopy(functions[0]);template.name=SYMBOL
        template.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in (*FORMALS,'_fm_cta','_fm_ctas')],kwonlyargs=[],kw_defaults=[],defaults=[])
        template.body=copy.deepcopy(records[0][3])
        # Both bodies keep the full nineteen unfolded iterations and the
        # original warp election. No tensor crosses the noinline boundary.
        helper=ast.unparse(ast.fix_missing_locations(template))
        for index,(fn,call,first,body) in enumerate(records):
            wrapper=copy.deepcopy(fn);wrapper.decorator_list=ast.parse('@triton.jit\ndef placeholder(): pass').body[0].decorator_list
            wrapper.body=wrapper.body[:first]+[ast.Expr(value=ast.Call(func=ast.Name(id=SYMBOL,ctx=ast.Load()),args=[ast.parse(call[k],mode='eval').body for k in FIELDS]+[ast.Name(id='_fm_cta',ctx=ast.Load()),ast.Name(id='_fm_ctas',ctx=ast.Load())],keywords=[]))]
            text=ast.unparse(ast.fix_missing_locations(wrapper))
            if index==0:text=helper+'\n\n'+text
            indent=' '*fn.col_offset
            text='\n'.join(indent+line if line else line for line in text.splitlines())+'\n'
            start=min(n.lineno for n in fn.decorator_list)-1
            changes.append((start,fn.end_lineno,text))
            receipts.append(dict(original_helper=fn.name,canonical=SYMBOL,scalar_pointer_fields=list(FIELDS),original_coordinate_prefix_preserved=True,normalized_full_body_AST_equal=True))
    if not changes:return source,[]
    lines=source.splitlines(keepends=True)
    for start,end,text in sorted(changes,reverse=True):lines[start:end]=[text]
    output=''.join(lines);ast.parse(output)
    return output,receipts


_active=False

def activate():
    global _active
    if _active:return
    from pathlib import Path
    import hashlib
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    previous=tir_package.describe_tir_package
    def describe(module):
        result=previous(module)
        calls=[c for c in result['render_calls'] if c['family']=='mhc_pre' and c['variant']=='cooperative_stream_fma']
        for call in calls:
            if (call['hc']!=4 or call['hidden']!=4096 or call['stream_lanes']!=4
                    or call['stream_active']!='4' or call['token_active']!='1'
                    or call['projection_splits']!=8 or call['coefficients']!=24
                    or call['projection_workspace_width']!=204 or call['sinkhorn_iterations']!=20):
                raise CodegenError('Shared full mix requires the original complete GLM decode geometry')
        result['implementation_override']['inputs']['shared_mhc_mix.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['declared_full_mhc_body_sharing']=dict(scalar_pointer_fields=list(FIELDS),scalar_CTA_parameters=['_fm_cta','_fm_ctas'],full_matrix=[4,4],iterations=20,unroll_factor=19,extra_workspace_bytes=0,exact_body_comparison_required=True)
        return result
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    previous_render=TritonTemplateRegistry.render
    def render(self,name,context):
        source=previous_render(self,name,context)
        if name!='module.py.jinja':return source
        changed,receipts=share_source(source,context.get('render_calls',()))
        if receipts:context['full_mhc_body_sharing_proof']=receipts
        return changed
    TritonTemplateRegistry.render=render
    _active=True
