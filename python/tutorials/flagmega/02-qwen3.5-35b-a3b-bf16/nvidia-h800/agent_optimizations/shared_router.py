"""Routing over expert logits plus an independent shared-expert gate column."""

from dataclasses import replace
from pathlib import Path
from jinja2 import ChoiceLoader, FileSystemLoader

from .qkv import QKVRegistry, create_target as base_target
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, CodegenError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.core import OpDefinition, OpCost, OpCostFactors, PythonCall, NodeRef, input_parameter, attribute_parameter, op_definition
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.targets import NvidiaSm90Target
from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
from triton.flagmega.codegen.triton import kernel_call_renderers, tir_package
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.codegen.triton.physical_access import emit_local_scalar_offset
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain


@op_definition("local.shared_router", display_name="Local.SharedRouter")
class SharedRouter(OpDefinition):
    value = input_parameter(is_tensor())
    experts = attribute_parameter()
    k = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        if any(type(attrs[key]) is not int or attrs[key] <= 0 for key in ("experts", "k")) or attrs["k"] > attrs["experts"]:
            raise IRSchemaError("SharedRouter needs positive expert/k counts with k <= experts")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        tensor = tensor_of(source)
        if tensor.rank != 2 or tensor.dtype != fm.DType.FLOAT32 or tensor.shape[1] != fm.dim(attrs["experts"] + 1):
            raise IRSchemaError("SharedRouter needs FP32 [tokens, experts+1] logits")
        results = (fm.tensor_type("float32", (tensor.shape[0], attrs["k"] + 1)),
                   fm.tensor_type("int64", (tensor.shape[0], attrs["k"] + 1)))
        if isinstance(source, fm.DistributedType):
            if source.partial is not None or source.axis_policies[1] != fm.SBP.broadcast():
                raise IRSchemaError("SharedRouter requires a materialized complete routing row")
            results = tuple(fm.DistributedType(t, source.axis_policies, source.placement, exclusive=source.exclusive) for t in results)
        return fm.TupleType(results)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        experts, k = node.attrs["experts"], node.attrs["k"]
        probabilities = value[:, :experts].softmax(-1)
        indices = context.torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :k]
        selected = probabilities.gather(1, indices)
        weights = selected / selected.sum(-1, keepdim=True)
        return (context.torch.cat((weights, value[:, experts:].sigmoid()), dim=-1),
                context.torch.cat((indices, context.torch.full((value.shape[0], 1), experts, dtype=context.torch.int64, device=value.device)), dim=-1))

    @classmethod
    def cost(cls, node):
        rows = tensor_of(node.type.fields[0]).shape[0]
        if not rows.is_fixed:
            return OpCost(notes=("dynamic-routing-rows",))
        rows = rows.fixed_value
        return OpCost(flops=router_work(node.attrs, rows), bytes_read=rows * (node.attrs["experts"] + 1) * 4,
                      bytes_written=rows * (node.attrs["k"] + 1) * 12, communication_bytes=0,
                      synchronizations=0, notes=("stable-bitonic-topk-independent-shared-gate",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        value = cls.value.type_of(inputs)
        local = local_tensor_type(value) if isinstance(value, fm.DistributedType) else tensor_of(value)
        if not local.shape[0].is_fixed:
            return None
        rows = local.shape[0].fixed_value
        return OpCostFactors(elementwise_operations=router_work(attrs, rows),
                             block_local_memory_load_bytes=rows * (attrs["experts"] + 1) * 4,
                             block_local_memory_store_bytes=rows * (attrs["k"] + 1) * 12)

    @classmethod
    def python_call(cls, node):
        return PythonCall("__import__('agent_optimizations.shared_router', fromlist=['shared_router']).shared_router",
                          (NodeRef(node.inputs[0]),), {**node.attrs, "name": node.id, "metadata": dict(node.metadata)})


def shared_router(value, *, experts, k, name=None, metadata=None):
    return SharedRouter.construct(value, experts=experts, k=k, name=name, metadata=metadata)


def router_work(attrs, rows):
    # The pruned bitonic network sorts K-sized subsequences, then merges them.
    log_n = (attrs["experts"] - 1).bit_length()
    log_k = (attrs["k"] - 1).bit_length()
    remaining = 1 << log_n
    comparisons = (remaining // 2) * log_k * (log_k + 1) // 2
    while remaining > 1 << log_k:
        remaining //= 2
        comparisons += remaining + (remaining // 2) * log_k
    return rows * (comparisons + 5 * attrs["experts"] + 2 * attrs["k"] + 4)


class Provider:
    op_names = frozenset({SharedRouter.op_name, "local.gather_reduce_shared_router"})

    def propose(self, node, context):
        variant = "local" if node.op == SharedRouter.op_name else "gather_reduce"
        candidates = tuple(context.configure_implementation(i) for i in context.implementations("shared_router")
                           if i.variant == variant and i.parameters["elements_per_program"] >= node.attrs["experts"])
        if not candidates:
            raise CodegenError("No SharedRouter implementation has sufficient expert tile capacity")
        chosen = min(candidates, key=lambda c: c.parameters["elements_per_program"])
        return TritonCandidateProposal(candidates, chosen.id)


class Target(NvidiaSm90Target):
    def pre_post_ops_rules(self):
        from .shared_router_collective import fuse_shared_router_rule
        return (*super().pre_post_ops_rules(), fuse_shared_router_rule())

    def register_auto_distributed_candidate_providers(self, registry):
        super().register_auto_distributed_candidate_providers(registry)
        registry.add(TypeInferenceCandidateProvider(frozenset({SharedRouter.op_name})))


def create_target():
    from . import shared_router_collective  # Register the explicit post-distribution operation.
    base = base_target()
    entries = tuple(TritonImplementation(f"tir.shared_router.{variant}.t{tile}", "shared_router", variant,
                                        {"elements_per_program": tile}, {}, facts={"portable_triton": True})
                    for variant in ("local", "gather_reduce") for tile in (64, 128, 256, 512))
    model = base.triton_implementation_model
    target = Target(triton_implementation_model=replace(model, implementations=(*model.implementations, *entries)))
    target.tir_selection_policy.registry = base.tir_selection_policy.registry
    target.tir_selection_policy.registry.add(Provider())
    return target


def encode(raw):
    if raw["semantic_op"] == "local.gather_reduce_shared_router":
        from .shared_router_collective import encode_collective
        return encode_collective(raw)
    get = kernel_call_renderers._buffer
    source, weights, ids = get(raw, "inputs", "value"), get(raw, "outputs", "result_0"), get(raw, "outputs", "result_1")
    domain = local_reduction_domain(source["abi"], (1,), int(raw["parameters"]["elements_per_program"]))
    def offset(abi, column):
        coords = list(domain["coordinates"])
        coords[1] = column
        return emit_local_scalar_offset(abi, coords)
    pointer = kernel_call_renderers._pointer
    attrs = raw["semantic_attrs"]
    return {**domain, "tile": 1 << (attrs["experts"] - 1).bit_length(), "experts": attrs["experts"], "k": attrs["k"],
            "selected_tile": 1 << (attrs["k"] - 1).bit_length(), "source": pointer(source),
            "source_offset": emit_local_scalar_offset(source["abi"], domain["coordinates"]),
            "gate_offset": offset(source["abi"], str(attrs["experts"])),
            "weights": pointer(weights), "ids": pointer(ids),
            "weights_offset": offset(weights["abi"], "_sr_selected_offsets"),
            "ids_offset": offset(ids["abi"], "_sr_selected_offsets"),
            "gate_weight_offset": offset(weights["abi"], str(attrs["k"])),
            "gate_id_offset": offset(ids["abi"], str(attrs["k"])),
            "weights_active": kernel_call_renderers._canonical_writer_active(weights["abi"]),
            "ids_active": kernel_call_renderers._canonical_writer_active(ids["abi"])}


class Registry(QKVRegistry):
    def __init__(self):
        super().__init__()
        self.extra_root = Path(__file__).parent / "templates"
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(self.extra_root)), self.environment.loader])

    def resolve(self, spec):
        for relative in self.candidates(spec):
            if (self.extra_root / relative).is_file():
                return relative
        return super().resolve(spec)


def install():
    from . import shared_router_collective
    kernel_call_renderers._FAMILY_ENCODERS["shared_router"] = encode
    tir_package.TritonTemplateRegistry = Registry
