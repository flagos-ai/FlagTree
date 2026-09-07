# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.implementation import (
    TritonImplementation,
    TritonImplementationModel,
)
from triton.flagmega.targets.selection import CapabilitySelectionPolicy
from triton.flagmega.codegen.triton.microkernels import (
    TritonMicroKernelSelectionPolicy,
    default_triton_microkernel_registry,
)


def semantic_packed_qkv_module(
    *, op: str = "ntt.packed_qkv_parallel_linear_fused_rhs"
) -> fm.IRModule:
    input_type = fm.tensor_type("bfloat16", (1, 2048))
    weight_type = fm.tensor_type(
        fm.VectorType(fm.DType.BFLOAT16, (8, 2, 8)), (128, 32)
    )
    output_types = (
        fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 256)),
        fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 128)),
        fm.tensor_type(fm.VectorType(fm.DType.BFLOAT16, (8,)), (1, 128)),
    )
    outputs = ("q_output", "k_output", "v_output")
    dispatch = fm.T.kernel_dispatch(
        semantic_op=op,
        semantic_candidate="ntt.packed_qkv_parallel_linear",
        arguments=("input", "fused_weight"),
        outputs=outputs,
        semantic_attrs={
            "rhs_layout": "k_major",
            "projection_n_capacities": (2048, 1024, 1024),
        },
        reads=("input", "fused_weight"),
        writes=outputs,
    )
    parameters = (
        fm.T.prim_parameter("input", input_type, fm.T.PrimParameterRole.INPUT),
        fm.T.prim_parameter(
            "fused_weight", weight_type, fm.T.PrimParameterRole.INPUT
        ),
        *(fm.T.prim_parameter(name, value_type, fm.T.PrimParameterRole.OUTPUT)
          for name, value_type in zip(outputs, output_types)),
    )
    function = fm.T.prim_function(
        "packed_qkv",
        "triton",
        parameters,
        fm.T.sequential((dispatch,)),
        fm.T.return_(tuple(
            fm.T.return_binding(fm.T.value_ref(name, value_type), name)
            for name, value_type in zip(outputs, output_types)
        )),
    )
    builder = fm.IRBuilder(dialect="semantic_tir", stage="tir_canonicalized")
    source = builder.var("source", input_type, id="source")
    weight = builder.weight(
        "fused_weight",
        weight_type,
        source="weights.safetensors",
        key="qkv",
        id="fused_weight",
    )
    builder.prim_function(function)
    result = builder.call(
        "tir.call",
        (source, weight),
        fm.TupleType(output_types),
        id="result",
        attrs={"callee": function.name},
    )
    builder.function("main", (source,), (result,))
    return fm.verify_module(builder.build(entry="main"))


def implementation_model() -> TritonImplementationModel:
    implementations = (
        TritonImplementation(
            "test.qkv.scalar",
            "qkv_parallel_linear",
            "scalar",
            {"block_n": 8, "block_k": 64},
            {"input_kind": "fused_rhs", "rhs_layout": "k_major"},
            facts={"portable_reference": True},
        ),
        TritonImplementation(
            "test.qkv.pipeline",
            "qkv_parallel_linear",
            "pipeline",
            {"block_n": 64, "block_k": 128, "stages": 2},
            {"input_kind": "fused_rhs", "rhs_layout": "k_major"},
            requires=("async_matrix",),
        ),
    )
    return TritonImplementationModel(
        implementations,
        {"qkv_parallel_linear": ("test.qkv.pipeline", "test.qkv.scalar")},
        "test-machine/v1",
    )


class StubCapability:
    def __init__(self, supported=("async_matrix",)) -> None:
        self.supported = frozenset(supported)

    def supports(self, requirements) -> bool:
        return set(requirements).issubset(self.supported)


class StubTarget:
    policy_version = "test-target/v1"

    def __init__(self, *, supported=("async_matrix",)) -> None:
        self.capability = StubCapability(supported)
        self.triton_implementation_model = implementation_model()
        self.selection_policy = CapabilitySelectionPolicy()
        self.microkernel_selection_policy = TritonMicroKernelSelectionPolicy(
            default_triton_microkernel_registry()
        )

    def add_default_selections(
        self, module, points, rationale, *, policy_version=None
    ):
        return self.selection_policy.add_defaults(
            module,
            points,
            rationale,
            capability=self.capability,
            target_name="test-target",
            policy_version=policy_version or self.policy_version,
        )

    def propose_microkernels(self, module):
        return self.microkernel_selection_policy.propose(module, self)

    def select_microkernels(self, module):
        return self.microkernel_selection_policy.apply(module, self)

    def verify(self, module) -> None:
        del module
