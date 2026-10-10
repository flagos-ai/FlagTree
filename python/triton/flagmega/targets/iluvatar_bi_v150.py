"""Iluvatar BI-V150 (COREX) physical target for FlagMega."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping

from triton.flagmega.errors import IRSchemaError, CodegenError
from triton.flagmega.ir import Placement
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
from triton.flagmega.targets.pyntt import PyNttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions


def corex_triton_implementation_model():
    """Return the SM90 catalog restricted to templates available on COREX."""
    from triton.flagmega.codegen.triton.templates import KernelTemplateSpec, TritonTemplateRegistry
    from triton.flagmega.targets.portable_triton_implementations import portable_triton_implementation_model
    model = portable_triton_implementation_model()
    registry = TritonTemplateRegistry()
    from dataclasses import replace
    from triton.flagmega.codegen.triton.implementation import TritonImplementation
    from triton.flagmega.ir import tensor_type
    from triton.flagmega.ir.tir import (TIRSharedWorkspaceDescriptor,
        TIRTransferPipelineChannel, TIRTransferPipelineContract)
    qkv_pipeline_stages = int(os.environ.get("FLAGMEGA_QKV_PIPELINE_STAGES", "2"))
    if qkv_pipeline_stages not in {2, 4}:
        raise ValueError(
            "FLAGMEGA_QKV_PIPELINE_STAGES must be 2 or 4; "
            "COREX shared layouts require a power-of-two stage extent"
        )
    qkv_block_n = int(os.environ.get("FLAGMEGA_QKV_BLOCK_N", "32"))
    if qkv_block_n not in {16, 32, 64}:
        raise ValueError("FLAGMEGA_QKV_BLOCK_N must be 16, 32, or 64")
    pointer_pipeline_id = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    pointer_pipeline = TritonImplementation(
        pointer_pipeline_id, "qkv_parallel_linear", "packed_pointer_smem_pipeline",
        {"block_n": qkv_block_n, "block_k": 64, "num_stages": qkv_pipeline_stages,
         "producer_warps": 4, "producer_registers": 64,
         "consumer_warps": 16, "worker_width": 64, "compute_num_warps": 16,
         "descriptor_kind": "pointer",
         # Two physical Shared slots let layer i+1 overlap layer i; the
         # source scheduler retains the lifetime drain before slot i is reused.
         "inter_stage_buffer_slots": 2},
        # Trial 80: the original contract over-specified one particular
        # site shape (reduction extent 2048, f32 outputs, lanes (4,2,8),
        # full placement ownership) and rejected every other legal
        # fused-RHS site, including the sharded decode tutorial. Keep only
        # structural requirements; _applicable's generic checks plus the
        # renderer ABI handle the rest.
        {"input_kind": "fused_rhs", "rhs_layout": "k_major",
         "required_input_dtype": "bfloat16", "required_weight_dtype": "bfloat16",
         "required_input_lanes": (),
         "required_weight_rank": 3,
         "required_local_rows": 1,
         "requires_matching_output_lanes": True,
         "requires_none_optional_inputs": True},
        ("async_copy", "warp_specialize"), {"transfer_pipeline": True},
        (TIRSharedWorkspaceDescriptor(
            "rhs_stage", tensor_type("bfloat16", (qkv_pipeline_stages, 4, qkv_block_n * 16)),
                                      128, matrix_compatible=False),),
        TIRTransferPipelineContract((TIRTransferPipelineChannel("weight", (1,), (0,), 16),),
                                     capacity=qkv_pipeline_stages))
    implementations = []
    for implementation in model.implementations:
        # Tensor-map descriptors and transfer pipelines use the CUDA TMA ABI;
        # COREX lowers these kernels through ordinary pointer loads instead.
        if implementation.facts.get("host_tensor_descriptor"):
            continue
        try:
            registry.resolve(KernelTemplateSpec(implementation.family, implementation.variant, "corex", "bi_v150"))
        except CodegenError:
            continue
        implementations.append(implementation)
    if pointer_pipeline is not None:
        try:
            registry.resolve(KernelTemplateSpec(pointer_pipeline.family, pointer_pipeline.variant, "corex", "bi_v150"))
            implementations.append(pointer_pipeline)
        except CodegenError:
            pass
    supported = {value.id for value in implementations}
    preferences = {
        family: tuple(candidate for candidate in candidates if candidate in supported)
        for family, candidates in model.preferences.items()
    }
    # Keep the async pointer pipeline available for explicit experiments, but
    # do not select it by default until its full 28-layer resource contract is
    # feasible on ivcore11.  The primitive copy path is validated; the current
    # megakernel still reaches the 256-register/thread runtime boundary and
    # has not completed full 28-layer serving validation.
    # FLAGMEGA_QKV_PIPELINE=1 promotes it to the front of the qkv
    # preference order for producer/consumer experiments (Trial 80).
    if os.environ.get("FLAGMEGA_QKV_PIPELINE") == "1" and pointer_pipeline_id in supported:
        qkv_pref = preferences.get("qkv_parallel_linear", ())
        preferences["qkv_parallel_linear"] = (
            (pointer_pipeline_id,) + tuple(c for c in qkv_pref if c != pointer_pipeline_id))
    preferences = {family: candidates for family, candidates in preferences.items() if candidates}
    return replace(model, implementations=tuple(implementations), preferences=preferences, name="iluvatar-bi-v150/v1")


@dataclass(frozen=True)
class IluvatarBiV150Capability:
    """The capabilities exposed by the BI-V150 COREX compiler/runtime."""

    compute_capability: tuple[int, int] = (7, 1)
    warp_size: int = 64
    max_threads_per_block: int = 4096
    max_threads_per_sm: int = 8192
    max_registers_per_sm: int = 262144
    max_shared_memory_bytes: int = 131072
    supports_fp8_mma: bool = False
    supports_fp8: bool = True
    supports_async_copy: bool = True
    supports_cooperative_grid: bool = True
    supports_tma: bool = True
    supports_warp_specialize: bool = True
    supports_grid_sync: bool = True
    supports_mma_v3: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute_capability", tuple(self.compute_capability))
        if self.compute_capability != (7, 1):
            raise IRSchemaError("IluvatarBiV150Capability requires compute capability (7, 1).")
        for name in ("warp_size", "max_threads_per_block", "max_threads_per_sm",
                     "max_registers_per_sm", "max_shared_memory_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise IRSchemaError(f"IluvatarBiV150Capability {name} must be positive.")
        if self.max_threads_per_block > self.max_threads_per_sm:
            raise IRSchemaError("BI-V150 max_threads_per_block cannot exceed max_threads_per_sm.")
        for name in ("supports_fp8_mma", "supports_fp8", "supports_async_copy",
                     "supports_cooperative_grid", "supports_tma", "supports_warp_specialize",
                     "supports_grid_sync", "supports_mma_v3"):
            if type(getattr(self, name)) is not bool:
                raise IRSchemaError(f"IluvatarBiV150Capability {name} must be bool.")

    @property
    def features(self) -> frozenset[str]:
        values = {"fp8"} if self.supports_fp8 else set()
        for enabled, name in (
            (self.supports_fp8_mma, "fp8_mma"),
            (self.supports_async_copy, "async_copy"),
            (self.supports_cooperative_grid, "cooperative_grid"),
            (self.supports_tma, "tma"),
            (self.supports_warp_specialize, "warp_specialize"),
            (self.supports_grid_sync, "grid_sync"),
            (self.supports_mma_v3, "mma_v3"),
        ):
            if enabled:
                values.add(name)
        return frozenset(values)

    def missing(self, requirements: Iterable[object]) -> tuple[str, ...]:
        return tuple(sorted({str(value) for value in requirements} - self.features))

    def supports(self, requirements: Iterable[object]) -> bool:
        return not self.missing(requirements)

    def to_data(self) -> dict[str, object]:
        return {
            "schema": "flagmega.iluvatar-bi-v150-capability/v1",
            "compute_capability": list(self.compute_capability),
            "warp_size": self.warp_size,
            "max_threads_per_block": self.max_threads_per_block,
            "max_threads_per_sm": self.max_threads_per_sm,
            "max_registers_per_sm": self.max_registers_per_sm,
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "supports_fp8_mma": self.supports_fp8_mma,
            "supports_fp8": self.supports_fp8,
            "supports_async_copy": self.supports_async_copy,
            "supports_cooperative_grid": self.supports_cooperative_grid,
            "supports_tma": self.supports_tma,
            "supports_warp_specialize": self.supports_warp_specialize,
            "supports_grid_sync": self.supports_grid_sync,
            "supports_mma_v3": self.supports_mma_v3,
        }

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> "IluvatarBiV150Capability":
        if data.get("schema") != "flagmega.iluvatar-bi-v150-capability/v1":
            raise IRSchemaError(f"Unsupported BI-V150 capability schema {data.get('schema')!r}.")
        return cls(
            compute_capability=tuple(data["compute_capability"]),
            warp_size=data["warp_size"],
            max_threads_per_block=data["max_threads_per_block"],
            max_threads_per_sm=data["max_threads_per_sm"],
            max_registers_per_sm=data["max_registers_per_sm"],
            max_shared_memory_bytes=data["max_shared_memory_bytes"],
            supports_fp8_mma=data["supports_fp8_mma"],
            supports_fp8=data["supports_fp8"],
            supports_async_copy=data["supports_async_copy"],
            supports_cooperative_grid=data["supports_cooperative_grid"],
            supports_tma=data["supports_tma"],
            supports_warp_specialize=data["supports_warp_specialize"],
            supports_grid_sync=data["supports_grid_sync"],
            supports_mma_v3=data["supports_mma_v3"],
        )


class IluvatarBiV150Machine(NvidiaSm90Machine):
    """Physical services shared with Triton templates, tuned for one BI-V150."""

    name = "iluvatar-bi-v150"
    policy_version = "iluvatar-bi-v150-machine/v1"
    codegen_platform = "corex"
    codegen_architecture = "bi_v150"

    def __init__(self, capability: IluvatarBiV150Capability | None = None, *, implementation_model=None):
        self.capability = capability or IluvatarBiV150Capability()
        if implementation_model is not None:
            from triton.flagmega.targets.nvidia.shared_layout import verify_shared_workspaces
            verify_shared_workspaces(implementation_model)
        self._implementation_model = implementation_model

    def triton_implementation_model(self):
        return self._implementation_model or corex_triton_implementation_model()

    def default_ntt_options(self) -> NttTargetOptions:
        return NttTargetOptions(
            placements=(Placement((4, 4), "yx", "bb"),),
            vector_lane_bytes=16,
            vector_max_axes=1,
            packing_vector_bytes=16,
            packing_k_pack=2,
        )

    def distributed_operation_cost_model(self):
        from triton.flagmega.passes.auto_distributed.operation_cost import DistributedOperationCostModel
        return DistributedOperationCostModel(
            block_local_read_bytes_per_cycle=512,
            block_local_write_bytes_per_cycle=512,
            block_local_latency_cycles=20,
            elementwise_elements_per_cycle=64,
            simt_fma_per_cycle=64,
            chip_global_read_bytes_per_cycle=512,
            chip_global_write_bytes_per_cycle=512,
            chip_global_latency_cycles=300,
            block_synchronization_cycles=25,
            grid_synchronization_cycles=2200,
            identity="iluvatar.bi-v150-target-op-cost/v1",
        )


class IluvatarBiV150Target(PyNttTarget):
    """FlagMega PyNTT target selecting COREX-compatible Triton kernels."""

    name = "iluvatar-bi-v150"
    policy_version = "pyntt-iluvatar-bi-v150/v1"

    def __init__(self, capability: IluvatarBiV150Capability | None = None, *,
                 options: NttTargetOptions | None = None, triton_implementation_model=None, **kwargs):
        machine = IluvatarBiV150Machine(capability, implementation_model=triton_implementation_model)
        super().__init__(machine, target_name=self.name, policy_version=self.policy_version,
                         options=options or machine.default_ntt_options(),
                         triton_implementation_model=triton_implementation_model, **kwargs)


__all__ = ["IluvatarBiV150Capability", "IluvatarBiV150Machine", "IluvatarBiV150Target", "corex_triton_implementation_model"]
