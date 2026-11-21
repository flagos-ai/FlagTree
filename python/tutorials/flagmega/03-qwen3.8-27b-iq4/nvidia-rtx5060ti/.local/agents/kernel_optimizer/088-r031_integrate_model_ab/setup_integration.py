from pathlib import Path
import shutil, json, hashlib, os
A=Path(__file__).resolve().parent
R27=A.parent/'085-r027_register_40_no_spill'
R17=A.parent/'077-r006_r17_gdn_prep'
F=A/'source-tree/python/triton/flagmega'
# Preserve the R27 source overlay, without copying binary compiler intermediates.
def copyfile(src,dst):
 if Path(src).suffix in ('.py','.jinja','.h','.cpp','.td'): return shutil.copy2(src,dst)
 Path(dst).symlink_to(Path(src).resolve()); return dst
shutil.copytree(R27/'source-tree',A/'source-tree',symlinks=True,copy_function=copyfile)
def put(rel,s):
 p=F/rel
 if p.is_symlink():p.unlink()
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def edit(rel,fn):put(rel,fn((F/rel).read_text()))
# R17 existing bodies, rules and native ABI dispatch only.
for rel in ('targets/nvidia/gdn.py','rules/ntt/r004_gdn_phase_bindings.py','rules/ntt/r017_split4lut_bindings.py','codegen/triton/kernels/gdn_recurrent/r017_split4lut.py.jinja','codegen/triton/kernels/gdn_recurrent/r004_contract_v4.py','passes/tir/materialize_kernel_definitions.py'):
 put(rel,(R17/'source-tree/python/triton/flagmega'/rel).read_text())
# Preserve R27's typed readonly-weight alignment requirement.
def restore_q8_alignment(s):
    ref=(R27/'source-tree/python/triton/flagmega/passes/tir/materialize_kernel_definitions.py').read_text()
    a=s.index('def _q8_formal_alignment('); b=s.index('\ndef ',a+1)
    ra=ref.index('def _q8_formal_alignment('); rb=ref.index('\ndef ',ra+1)
    return s[:a]+ref[ra:rb]+s[b:]
edit('passes/tir/materialize_kernel_definitions.py',restore_q8_alignment)
# Extend memberships in R27 renderer/package while retaining its GLU changes.
for rel in ('codegen/triton/kernel_call_renderers.py','codegen/triton/tir_package.py'):
 def extend(s):
  s=s.replace('"r004_column_v6")','"r004_column_v6", "r017_split4lut")').replace("'r004_T_full_p5_v6')","'r004_T_full_p5_v6', 'r017_T_full_p5_split4lut')")
  if rel.endswith('tir_package.py'):
   marker='    return {str(p.relative_to(root.parent.parent)):hashlib'
   addition='''    if plan is not None and plan['module']=='r017_T_full_p5_split4lut':
        paths += [root/'kernels/gdn_recurrent/r017_split4lut.py.jinja',
                  root/'kernels/gdn_recurrent/r004_norm_v6.py.jinja',
                  root/'kernels/gdn_recurrent/r004_column_v6.py.jinja',
                  root/'kernels/_iq4_streaming.py.jinja',
                  root.parent.parent/'rules/ntt/r017_split4lut_bindings.py']
'''
   s=s.replace(marker,addition+marker)
  return s
 edit(rel,extend)
# Distinct opt-in policy. Old v6 policy and its bodies remain byte-identical.
s=(F/'rules/ntt/r004_model_policy_v6.py').read_text().replace('.r004_column_phase_bindings_v6 import (','.r017_split4lut_bindings import (').replace('flagmega.r004-model-policy/v2','flagmega.r031-model-policy/v1').replace('+r004-column:', '+r031-r017:')
s=s.replace('OP = "nn.gdn_recurrent_core"','PLAN_BODY_SHA256 = column_phase_plan()["plan_body_sha256"]\nOP = "nn.gdn_recurrent_core"')
put('rules/ntt/r031_model_policy.py',s)
def policy_routes(s):
 for fn,sig,call in [('model_request','module','module'),('configure_model_phases','module','module'),('model_distribution_target','target, module','target, module'),('propose_model_tir','module, target','module, target')]:
  needle=f'def {fn}({sig}):'
  assert needle in s
  s=s.replace(needle,needle+f'\n    if isinstance(module.metadata.get(REQUEST_KEY), Mapping) and module.metadata[REQUEST_KEY].get("schema") == "flagmega.r031-model-policy/v1":\n        from .r031_model_policy import {fn} as r031_route\n        return r031_route({call})')
 return s
edit('rules/ntt/r004_model_policy.py',policy_routes)
def native(s):
 s=s.replace('    if version == 6:', '''    if version == 31:
        from .r017_split4lut_bindings import column_phase_plan as r17_plan
        plan = r17_plan()
        return {**native_model_contract(6), "schema":"r031.gdn.native-model/v1",
                "implementation":plan["implementation"], "plan_body_sha256":plan["plan_body_sha256"],
                "arena":{k:v for k,v in plan["pool"].items() if k != "root"},
                "entry_order":[e["phase"] for e in plan["entries"]], "phase_module":plan["module"]}
    if version == 6:''')
 s=s.replace('6 if model_request(module)["schema"] == "flagmega.r004-model-policy/v2" else 4','31 if model_request(module)["schema"] == "flagmega.r031-model-policy/v1" else 6 if model_request(module)["schema"] == "flagmega.r004-model-policy/v2" else 4')
 s=s.replace('    if phase not in ("prep", "core", "y"):', '''    if contract is not None and contract.get("schema") == "r031.gdn.native-model/v1":
        if not _same_contract(contract, native_model_contract(31)) or phase not in contract["entry_order"]:
            raise IRVerificationError("Invalid R031 phase contract.")
        return "r031_model_T_" + phase + "_r017"
    if phase not in ("prep", "core", "y"):''')
 return s
edit('rules/ntt/r004_native_model.py',native)
edit('passes/tir/materialize_kernel_definitions.py',lambda s:s.replace('("r004_column_v6" if model_contract["schema"] == "r004.gdn.native-model/v2" else "r004_column_v4")','model_contract["implementation"].rsplit(".",1)[-1]'))
def codegen(s):
 s=s.replace('PROGRAM_SCHEMA_V6 =','PROGRAM_SCHEMA_R031 = "flagmega.kernel-programs/v11-native-model-r031"\n\nPROGRAM_SCHEMA_V6 =')
 s=s.replace("'r004.gdn.native-model/v2': 6}","'r004.gdn.native-model/v2': 6, 'r031.gdn.native-model/v1':31}")
 s=s.replace('phase_plan_for("r004_T_full_p5_v6" if contract["schema"] == "r004.gdn.native-model/v2" else "r004_T_full_p5_v4")','phase_plan_for(contract.get("phase_module", "r004_T_full_p5_v6" if contract["schema"] == "r004.gdn.native-model/v2" else "r004_T_full_p5_v4"))')
 s=s.replace("'program_grid':[48,1,1]","'program_grid':next(e['grid'] for e in plan['entries'] if e['phase']=='prep')")
 s=s.replace('!= [41472]', '!= [plan["pool"]["nbytes"]]')
 s=s.replace("'core':'Core'}","'core':'Core', 'partials':'Partials'}")
 s=s.replace('phase_plan_for("r004_T_full_p5_v6" if is_v6 else "r004_T_full_p5_v4")','phase_plan_for(contract.get("phase_module", "r004_T_full_p5_v6" if is_v6 else "r004_T_full_p5_v4"))')
 s=s.replace("'kernel_program_schema':PROGRAM_SCHEMA_V6 if is_v6 else PROGRAM_SCHEMA","'kernel_program_schema':PROGRAM_SCHEMA_R031 if contract['schema']=='r031.gdn.native-model/v1' else PROGRAM_SCHEMA_V6 if is_v6 else PROGRAM_SCHEMA")
 # Source wrapper selection changes only the new candidate.
 s=s.replace("for p in descriptor['kernel_templates']]}","for p in descriptor['kernel_templates']]}\n")
 s=s.replace("(('kernels/gdn_recurrent/r004_model_v2.py.jinja' if is_v6", "(('kernels/gdn_recurrent/r031_model_r017.py.jinja' if contract['schema']=='r031.gdn.native-model/v1' else 'kernels/gdn_recurrent/r004_model_v2.py.jinja' if is_v6")
 s=s.replace("if p.endswith('/r004_column_v6.py.jinja' if is_v6 else '/r004_column_v4.py.jinja')", "if p.endswith('/r017_split4lut.py.jinja' if contract['schema']=='r031.gdn.native-model/v1' else '/r004_column_v6.py.jinja' if is_v6 else '/r004_column_v4.py.jinja')")
 return s
edit('codegen/triton/r004_native_model.py',codegen)
edit('runtime/r004_native_model.py',lambda s:s.replace('PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6','PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6, PROGRAM_SCHEMA_R031').replace('(PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6)', '(PROGRAM_SCHEMA, PROGRAM_SCHEMA_V6, PROGRAM_SCHEMA_R031)'))
edit('runtime/module.py',lambda s:s.replace('"flagmega.kernel-programs/v10-native-model-v6"))','"flagmega.kernel-programs/v10-native-model-v6", "flagmega.kernel-programs/v11-native-model-r031"))'))
put('codegen/triton/kernels/gdn_recurrent/r031_model_r017.py.jinja','''# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
{% with render_calls=() %}
{% include "kernels/gdn_recurrent/r017_split4lut.py.jinja" %}
{% endwith %}
{% for call in render_calls if call.family == "gdn_recurrent" and call.variant == "r017_split4lut" %}
@triton.jit
def {{ call.symbol }}({{ call.signature }}):
{% if call.phase == "partials" %}
    _r017_partials({{ call.model_native_arguments | join(", ") }})
{% else %}
    _r017_{{ call.phase }}_split4lut({{ call.model_native_arguments | join(", ") }})
{% endif %}
{% endfor %}
''')
print('INTEGRATION_OVERLAY_READY')
