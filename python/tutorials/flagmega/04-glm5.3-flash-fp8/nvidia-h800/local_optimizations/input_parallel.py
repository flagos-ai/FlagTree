"""Explicit typed scratch and CTA cooperation for existing ordered FP32 splits.

Private trial; install before candidate selection and ordinary source rendering.
No generated-source edits, hidden scratch, numerical relaxation, or backend changes.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from jinja2 import ChoiceLoader,DictLoader
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.cooperative_access import require_chip_workspace
from triton.flagmega.codegen.triton.kernel_call_renderers import _buffer,_pointer
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.errors import CodegenError
from triton.flagmega.targets import NvidiaSm90Target

FLAG = "parallel_accumulation_splits"
KEY = "kernels/partitioned_linear/platforms/nvidia/sm90/wgmma_local.py.jinja"


class ParallelTarget(NvidiaSm90Target):
    def propose_tir(self,module):
        proposal = super().propose_tir(module)
        points = []
        for point in proposal.selection_points:
            node = proposal.node_map.get(point.owner)
            if node is None or node.op != "nn.partitioned_linear":
                points.append(point)
                continue
            kind = fm.logical_type(node.type)
            rows = kind.shape[-1].fixed_value * getattr(kind.dtype,"lane_count",1)
            parts,splits = node.attrs["partitions"],node.attrs.get("accumulation_splits",1)
            candidates = []
            for candidate in point.candidates:
                if candidate.parameters.get(FLAG):
                    if parts != 1 or splits != 2:
                        raise CodegenError("The measured trial requires one semantic BF16 part and two FP32 sub-dots.")
                    workspace = dict(name="projection_partials",type=fm.tensor_type("float32",(parts,splits,rows)),
                                     memory_space="workspace",alignment=256,lifetime="invocation")
                    if candidate.parameters.get("workspaces"):
                        raise CodegenError("Parallel input projection has unexpected preexisting scratch.")
                    candidate = replace(candidate,parameters={**candidate.parameters,"workspaces":(workspace,)})
                candidates.append(candidate)
            points.append(replace(point,candidates=tuple(candidates)))
        return fm.verify_module(replace(proposal,selection_points=tuple(points)))


def target():
    base = NvidiaSm90Target().triton_implementation_model
    values = []
    for value in base.implementations:
        if value.family == "partitioned_linear" and value.variant == "wgmma_local":
            value = replace(value,parameters={**value.parameters,FLAG:True},
                facts={**value.facts,"internal_grid_barriers":1},
                requires=tuple(dict.fromkeys((*value.requires,"grid_sync"))))
        values.append(value)
    return ParallelTarget(triton_implementation_model=replace(base,implementations=tuple(values)))


@contextmanager
def activate():
    original = TritonTemplateRegistry.render
    stock = (Path(__import__("triton.flagmega.codegen.triton.templates",fromlist=["x"]).__file__).parent/KEY).read_text()
    match = '{% if call.family == "partitioned_linear" and call.variant == "wgmma_local" %}'
    assert stock.count(match) == 1
    source = stock.replace(match,match[:-2]+'and not call.parameters.get("'+FLAG+'", false) %}')
    source += Path(__file__).with_name("parallel.py.jinja").read_text()

    def render(self,name,context):
        if name != "module.py.jinja":
            return original(self,name,context)
        for call in context.get("render_calls",()):
            if not call.get("parameters",{}).get(FLAG):
                continue
            if (call["family"] != "partitioned_linear" or call["variant"] != "wgmma_local"
                    or call["participation_active"] != "True" or call["internal_grid_barriers"] != 1
                    or call["parts"] != 1 or call["accumulation_splits"] != 2
                    or call["accumulation_k"] % call["block_k"]):
                raise CodegenError("Parallel split renderer requires complete K sub-dots and all-CTA chip participation.")
            workspace = _buffer(call,"workspaces","projection_partials")
            require_chip_workspace(workspace,(call["parts"],call["accumulation_splits"],call["rows"]),
                                   "projection_partials",owner="Ordered parallel input projection")
            call["partial_pointer"] = _pointer(workspace)
            call["parallel_tasks"] = call["parts"]*call["accumulation_splits"]*call["tasks"]
        previous = self.environment.loader
        self.environment.loader = ChoiceLoader([DictLoader({KEY:source}),previous])
        try:
            return original(self,name,context)
        finally:
            self.environment.loader = previous
    TritonTemplateRegistry.render = render
    try:
        yield
    finally:
        TritonTemplateRegistry.render = original
