"""Graph admission/invalidation regressions for the frozen r001 interface."""
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import ast

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_optimizations.iq4_streaming.graph import (
    PLAN_HASH, RESIDENCY_ENCODING, FAMILIES, StreamingConfig,
    configure_target, configure_distribution_target, prepare_distribution,
)
from agent_optimizations.iq4_streaming.build import BASELINE, verify_packing_input, verify_gate
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega import Compiler, CompileOptions
from triton.flagmega.ir import load_module
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
from triton.flagmega.codegen.triton.renderer import render_triton_package


def model(**changes):
    return tuple(TritonImplementation("r001." + f, f, "stream",
        parameters={"tile_n": 4, "num_warps": 4, "plan_hash": PLAN_HASH,
                    "residency_encoding": RESIDENCY_ENCODING, **changes},
        facts={"internal_grid_barriers": 0, "required_codegen_backend": "triton"})
        for f in sorted(set(FAMILIES.values())))


@pytest.mark.parametrize("changes", [dict(tile_n=8), dict(num_warps=8),
    dict(plan_hash="r000"), dict(residency_encoding={"threads_per_warp": [4, 8]})])
def test_stale_or_mismatched_native_rejected(changes):
    c = Compiler(CompileOptions(target="nvidia-sm120", codegen_backend="triton"))
    with pytest.raises(CodegenError, match="pending or incompatible"):
        configure_target(c.target, native_model=model(**changes))


def test_target_preserves_options_and_policies():
    c = Compiler(CompileOptions(target="nvidia-sm120", codegen_backend="triton"))
    base = c.target
    target = configure_target(base, native_model=model())
    for name in ("options", "packing_policy", "vectorization_policy", "selection_policy",
                 "tir_lowering_policy", "microkernel_selection_policy"):
        assert getattr(target, name) is getattr(base, name)
    moved = target.with_placement(target.options.placements[0])
    assert moved.triton_implementation_model is target.triton_implementation_model
    assert moved.tir_selection_policy is target.tir_selection_policy
    assert moved.bufferization_policy is target.bufferization_policy
    assert moved.codegen_backend == "triton"
    assert PLAN_HASH in moved.distribution_policy.identity
    with pytest.raises(ValueError, match="fixes R4"):
        StreamingConfig(8, 4)
    base.codegen_backend = "cpp"
    with pytest.raises(CodegenError, match="SM120 Triton"):
        configure_distribution_target(base)


def test_signed_complete_packing_and_stale_identity(tmp_path):
    assert len(verify_gate()["peers"]) == 5
    source = BASELINE / "dumps/Compile/02_AutoPackingPass/After"
    assert len(verify_packing_input(source)) == 6
    packed = load_module(source)
    current = prepare_distribution(packed, StreamingConfig())
    assert current.node_map["decode_attention_1_q.packed"].metadata["kernel_maxnreg"] == 96
    assert current.node_map["decode_attention_1_mlp_gate_up"].metadata["kernel_maxnreg"] == 104
    assert prepare_distribution(current, StreamingConfig()) is current
    stale = replace(packed, metadata={**packed.metadata, "iq4_streaming": {"plan_hash": "r000"}})
    with pytest.raises(IRVerificationError, match="before distribution"):
        prepare_distribution(stale, StreamingConfig())
    with pytest.raises(IRVerificationError, match="complete AutoPacking"):
        prepare_distribution(replace(packed, stage="packed"), StreamingConfig())
    shutil.copytree(source, tmp_path / "packing")
    (tmp_path / "packing/main.py").unlink()
    with pytest.raises(ValueError, match="complete accepted"):
        verify_packing_input(tmp_path / "packing")


@pytest.mark.parametrize("family", sorted(set(FAMILIES.values())))
def test_actual_registered_native_preserves_scalar_abi(tmp_path, family):
    """Normal compiler/renderer integration; device H001-static remains separate."""
    from agent_optimizations.iq4_streaming import native

    class Graph(fm.Module):
        def forward(self):
            n = 12288 if family.endswith("interleaved") else 16
            x = self.input("x", fm.tensor_type("bfloat16", (1, 256)))
            w = self.input("w", fm.tensor_type("uint8", (n, 136)))
            scale = self.input("scale", fm.tensor_type("float32", ((n + 127) // 128, 2)))
            attrs = dict(weight_block_n=128, weight_block_k=128, weight_encoding="IQ4_XS")
            if family.endswith("interleaved"):
                result = fm.F.nn.interleaved_projection(x, w, scale, feature_shape=(24, 256),
                    output_lanes=((8,), (8,)), **attrs)
                outputs = fm.F.tensors.get_items(result, 0, 1)
            elif family.endswith("glu"):
                attrs.pop("weight_encoding")
                result = fm.F.nn.matmul_glu(x, w, w, scale, scale, activation="silu",
                    gate_weight_encoding="IQ4_XS", up_weight_encoding="IQ4_XS", **attrs)
                outputs = (result,)
            else:
                outputs = (fm.F.math.block_scaled_matmul(x, w, scale, **attrs),)
            self.function("main", (x, w, scale), outputs)

    compiler = Compiler(CompileOptions(target="nvidia-sm120", codegen_backend="triton"))
    compiler.target = configure_target(compiler.target)
    module = compiler.compile(Graph(dialect="high_level", stage="imported", entry="main").build()).module
    call = next(c for c in describe_tir_package(module)["render_calls"] if c["family"] == family)
    assert call["schema"] == native.SCHEMA
    assert call["tile_n"] == 4 and call["internal_grid_barriers"] == 0
    assert call["parameters"]["plan_hash"] == PLAN_HASH
    assert call["parameters"]["residency_encoding"] == RESIDENCY_ENCODING
    render_triton_package(module, tmp_path / "generated")
    ast.parse((tmp_path / "generated/generated_kernels.py").read_text())
