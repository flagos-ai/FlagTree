"""Validated merged-QKV baseline template, using the normal target catalog."""

from dataclasses import replace
from pathlib import Path

from jinja2 import ChoiceLoader, FileSystemLoader
from agent_optimizations import create_target as base_target, install as base_install
from agent_optimizations.sigmoid_product import SigmoidProductRegistry
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package

_core_encoder = kernel_call_renderers._qkv_parallel_linear_call
ROOT = Path(__file__).parent / "templates"


def encode_merged(raw):
    call = _core_encoder(raw)
    if raw.get("variant") == "packed_fused_gemv":
        call["coalesced"] = bool(raw["parameters"].get("coalesced_gemv", False))
        call["packed_n_capacity"] = sum(output["local_n_capacity"] for output in call["outputs"])
        call["tile_n"] = kernel_call_renderers._bounded_vector_tile(
            raw["parameters"]["tile_n"], call["packed_n_capacity"], name="Merged QKV output tile")
    return call


class QKVRegistry(SigmoidProductRegistry):
    def __init__(self):
        super().__init__()
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(ROOT)), self.environment.loader])

    def resolve(self, spec):
        for path in self.candidates(spec):
            if (ROOT / path).is_file():
                return path
        return super().resolve(spec)


def install(*, merged=True):
    base_install()
    kernel_call_renderers._FAMILY_ENCODERS["qkv_parallel_linear"] = encode_merged if merged else _core_encoder
    if merged:
        tir_package.TritonTemplateRegistry = QKVRegistry


def create_target():
    target = base_target()
    model = target.triton_implementation_model
    entries = tuple(replace(value, parameters={**value.parameters, "block_k": 128, "tile_n": 64})
                    if value.id == "tir.qkv_parallel_linear.packed_fused_gemv" else value
                    for value in model.implementations)
    target.triton_implementation_model = replace(model, implementations=entries)
    return target
