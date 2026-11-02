"""A kernel's interned name cannot decide its semantic scheduling role."""
from dataclasses import dataclass, replace
from types import SimpleNamespace

from local_optimizations.compile_model import measured_physical_tir


@dataclass(frozen=True)
class Microkernel:
    implementation: str
    parameters: dict
    shared_workspaces: tuple = ()


@dataclass(frozen=True)
class Dispatch:
    microkernel: Microkernel
    shared_workspace_buffers: tuple = ()


@dataclass(frozen=True)
class Definition:
    name: str
    dispatch: Dispatch


@dataclass(frozen=True)
class Module:
    kernel_definitions: tuple
    nodes: tuple


def test_renamed_dense_down_keeps_its_unmeasured_semantic_role(monkeypatch):
    import triton.flagmega.ir.tir.kernel_definition as kd
    import triton.flagmega.codegen.triton.microkernels.materialization as materialization
    monkeypatch.setattr(kd, "replace_kernel_dispatch", lambda d, v: replace(d, dispatch=v))
    monkeypatch.setattr(materialization, "materialize_shared_workspace_buffers", lambda *_: ())
    names = ["kernel_matmul_glu_cooperative_wgmma_97494b8867efabf6",
             "kernel_dense_down_with_a_new_contract_digest", *[f"measured_{i}" for i in range(6)]]
    definitions = tuple(Definition(name, Dispatch(Microkernel("implementation", {}))) for name in names)
    roles = ["decode_linear_attention_dense_dense_mlp_gate_up_fp8",
             "decode_linear_attention_dense_dense_mlp_down.partitions", *[f"other_{i}" for i in range(6)]]
    origins = ["nn.packed_matmul_glu", "nn.partitioned_block_scaled_linear", *["other"] * 6]
    nodes = tuple(SimpleNamespace(id=role, op="tir.call", attrs={"callee": name},
                                  metadata={"lowered_from": origin})
                  for name, role, origin in zip(names, roles, origins))
    chosen = SimpleNamespace(id="implementation.decode_measured", parameters={}, shared_workspaces=())
    target = SimpleNamespace(triton_implementation_model=SimpleNamespace(implementation=lambda _: chosen),
                             verify=lambda _: None)
    result = measured_physical_tir(Module(definitions, nodes), target)
    assert result.kernel_definitions[:2] == definitions[:2]
    assert all(d.dispatch.microkernel.implementation == chosen.id for d in result.kernel_definitions[2:])
