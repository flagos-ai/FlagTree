# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Strict loader and runtime for generated FlagMega artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass
from math import prod

from triton.flagmega.artifacts import verify_rdata
from triton.flagmega.errors import ArtifactError, IRSchemaError, RuntimeContractError
from triton.flagmega.ir import (
    DType,
    RefType,
    TensorType,
    TupleType,
    VectorType,
    logical_type,
    verify_buffer_plan,
)
from triton.flagmega.ir.ops.nn._gdn_state import (
    GatedDeltaNetStateConfig,
    GatedDeltaNetStateKind,
)
from triton.flagmega.ir.types import data_type_from_data
from triton.flagmega.ir.ops.nn._paged_attention_state import (
    PagedAttentionState,
    PagedAttentionStateConfig,
    create_paged_attention_state,
)
from triton.flagmega.runtime.prepared import ResourceContract, prepare_jit_kernel
from triton.flagmega.runtime.nvidia_launch import compilation_options
from triton.flagmega.runtime.lifecycle import RuntimeModule
from triton.flagmega.runtime.tensor_descriptor import TensorDescriptorCache
from triton.flagmega.runtime.workspace_diagnostics import (
    materialize_memory_pool_views,
    memory_pool_view_specs,
)


class GeneratedElementwiseModule(RuntimeModule):
    def __init__(self, artifact: Path, manifest: dict[str, object], ir_module, kernel) -> None:
        super().__init__()
        self.artifact = artifact
        self.manifest = manifest
        self.ir_module = ir_module
        self.kernel = kernel
        self.codegen = manifest["codegen"]
        self._prepared = None
        self.prepare_count = 0
        function = ir_module.function_map[ir_module.entry]
        self._input_types = tuple(
            logical_type(ir_module.node_map[node_id].type)
            for node_id in function.parameters
        )
        self._output_type = logical_type(ir_module.node_map[function.outputs[0]].type)

    @property
    def prepared(self) -> bool:
        return self._prepared is not None

    @property
    def resource_report(self):
        return None if self._prepared is None else self._prepared.resource_report

    def load(self, device: str = "cuda:0") -> GeneratedElementwiseModule:
        torch = _torch()
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeContractError(f"Executable target nvidia-sm90 requires a CUDA device, got {device!r}.")
        capability = torch.cuda.get_device_capability(torch.device(device))
        if capability != (9, 0):
            raise RuntimeContractError(f"Artifact target nvidia-sm90 requires capability (9, 0), got {capability}.")
        self._mark_loaded(device)
        return self

    def prepare(self, *inputs, output=None) -> GeneratedElementwiseModule:
        torch = _torch()
        self._require_loaded()
        self._validate_inputs(inputs)
        if output is None:
            output = self._allocate_output(torch)
        self._validate_tensor("output", output, self._output_type)
        if str(output.device) != self._device:
            raise RuntimeContractError(f"Output must be on prepared device {self._device}, got {output.device}.")
        contract = ResourceContract(compute_num_warps=int(self.codegen["num_warps"]), resident_blocks_per_sm=1)
        self._prepared = prepare_jit_kernel(
            self.kernel,
            (*inputs, output),
            tuple(int(value) for value in self.codegen["dynamic_argument_indices"]),
            grid=tuple(int(value) for value in self.codegen["grid"]),
            contract=contract,
            **compilation_options(contract),
        )
        self.prepare_count += 1
        self._mark_prepared()
        return self

    def run_into(self, output, *inputs, stream=None) -> None:
        self._require_prepared()
        if self._prepared is None:
            raise RuntimeContractError("Artifact must be prepared before run_into().")
        self._validate_inputs(inputs)
        self._validate_tensor("output", output, self._output_type)
        self._prepared.launch(*inputs, output, stream=stream)

    def run(self, *inputs, stream=None):
        torch = _torch()
        self._require_prepared()
        if self._prepared is None:
            raise RuntimeContractError("Artifact must be prepared before run().")
        self._validate_inputs(inputs)
        output = self._allocate_output(torch)
        self.run_into(output, *inputs, stream=stream)
        return output

    def _release_resources(self) -> None:
        self._prepared = None

    def _validate_inputs(self, inputs) -> None:
        if len(inputs) != len(self._input_types):
            raise RuntimeContractError(f"Artifact expects {len(self._input_types)} inputs, got {len(inputs)}.")
        for index, (value, value_type) in enumerate(zip(inputs, self._input_types)):
            self._validate_tensor(f"input {index}", value, value_type)
            if str(value.device) != self._device:
                raise RuntimeContractError(f"Input {index} must be on prepared device {self._device}, got {value.device}.")

    def _allocate_output(self, torch):
        if not isinstance(self._output_type, TensorType):
            raise RuntimeContractError("Elementwise output must have a TensorType ABI.")
        shape = tuple(int(dimension.value) for dimension in self._output_type.shape)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self._output_type.dtype.value]
        return torch.empty(shape, dtype=dtype, device=self._device)

    @staticmethod
    def _validate_tensor(name: str, value, value_type) -> None:
        torch = _torch()
        if not isinstance(value_type, TensorType) or not isinstance(value, torch.Tensor):
            raise RuntimeContractError(f"{name} must be a torch.Tensor matching a TensorType ABI.")
        shape = tuple(int(dimension.value) for dimension in value_type.shape)
        dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[value_type.dtype.value]
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise RuntimeContractError(
                f"{name} must have shape {shape} and dtype {dtype}, got {tuple(value.shape)} and {value.dtype}.")
        if not value.is_contiguous():
            raise RuntimeContractError(f"{name} must be contiguous.")
        if value.data_ptr() % 16:
            raise RuntimeContractError(f"{name} must be at least 16-byte aligned.")


# Artifact v1 compatibility name.  New packages use the operation-neutral
# adapter above; old elementwise_add/v1 manifests still resolve to this alias.
GeneratedAddModule = GeneratedElementwiseModule


@dataclass
class GatedDeltaNetState:
    convolution: object
    recurrent: object


class GeneratedTirCallGraphModule(RuntimeModule):
    """Model-neutral runtime for the flattened bufferized-TIR entry ABI."""

    def __init__(self, artifact: Path, manifest: dict[str, object], ir_module, kernel) -> None:
        super().__init__()
        self.artifact = artifact
        self.manifest = manifest
        self.ir_module = ir_module
        self.kernel = kernel
        self.codegen = manifest["codegen"]
        binding = self.codegen.get("runtime_binding")
        if not isinstance(binding, Mapping):
            raise ArtifactError("TIR call-graph artifact has no runtime binding.")
        self.runtime_binding = binding
        self.buffer_plan = verify_buffer_plan(ir_module)
        self._prepared = None
        self._pool_values: dict[str, object] = {}
        self._tensor_descriptors = TensorDescriptorCache()
        self.prepare_count = 0

        arguments = tuple(binding.get("arguments", ()))
        pools = tuple(binding.get("pools", ()))
        signature = tuple(str(value) for value in binding.get("signature", ()))
        expected = tuple(str(value["name"]) for value in (*arguments, *pools))
        if signature != expected:
            raise ArtifactError(
                "TIR runtime binding signature does not match arguments and pools."
            )
        raw_specs = self.codegen.get("host_tensor_descriptor_specs", ())
        if not isinstance(raw_specs, (tuple, list)):
            raise ArtifactError(
                "TIR host tensor descriptor specs must be a sequence."
            )
        self._descriptor_specs = tuple(raw_specs)
        try:
            descriptor_names = TensorDescriptorCache.validate_specs(
                self._descriptor_specs, source_names=expected
            )
        except RuntimeContractError as error:
            raise ArtifactError(
                f"Invalid TIR host tensor descriptor ABI: {error}"
            ) from error
        expected_codegen_signature = (*expected, *descriptor_names)
        codegen_signature = tuple(
            str(value) for value in self.codegen.get(
                "signature_arguments", expected_codegen_signature
            )
        )
        if codegen_signature != expected_codegen_signature:
            raise ArtifactError(
                "TIR generated signature does not match runtime roots and "
                "host tensor descriptors."
            )
        external_names = {str(value["name"]) for value in arguments}
        descriptor_base = len(expected)
        self._dynamic_descriptor_indices = tuple(
            index
            for index, spec in enumerate(self._descriptor_specs)
            if str(spec["source"]) in external_names
        )
        expected_dynamic = (
            *range(len(arguments)),
            *(
                descriptor_base + index
                for index in self._dynamic_descriptor_indices
            ),
        )
        dynamic = tuple(
            int(value)
            for value in self.codegen.get("dynamic_argument_indices", ())
        )
        if dynamic != expected_dynamic:
            raise ArtifactError(
                "TIR call-graph dynamic arguments must be its external buffers "
                "followed by descriptors backed by those buffers."
            )

    @property
    def prepared(self) -> bool:
        return self._prepared is not None

    @property
    def resource_report(self):
        return None if self._prepared is None else self._prepared.resource_report

    @property
    def external_arguments(self) -> tuple[Mapping[str, object], ...]:
        return tuple(self.runtime_binding["arguments"])

    def load(self, device: str = "cuda:0") -> "GeneratedTirCallGraphModule":
        torch = _torch()
        _validate_sm90_device(torch, device)
        pools: dict[str, object] = {}
        for pool in self.runtime_binding["pools"]:
            name = str(pool["name"])
            storage = str(pool["storage"])
            nbytes = int(pool["nbytes"])
            if storage == "rdata":
                pools[name] = self._load_rdata(torch, device, nbytes)
            elif storage == "workspace" or storage in self.buffer_plan.memory_space_map:
                space = self.buffer_plan.memory_space_map[storage]
                if space.kind == "shared" or space.allocation_scope.value == "external":
                    raise ArtifactError(
                        f"Unsupported TIR runtime pool storage {storage!r}."
                    )
                pools[name] = torch.empty(
                    (nbytes,), dtype=torch.uint8, device=device
                )
            else:
                raise ArtifactError(
                    f"Unsupported TIR runtime pool storage {storage!r}."
                )
        self._pool_values = pools
        self._mark_loaded(device)
        return self

    def prepare(self, *external_arguments) -> "GeneratedTirCallGraphModule":
        self._prepare_external(tuple(external_arguments))
        return self

    def run_into(self, *external_arguments, stream=None) -> None:
        self._launch_external(tuple(external_arguments), stream=stream)

    def diagnostic_workspace(self, *, clone: bool = False) -> dict[str, object]:
        """Return views over the default SAT pool (compatibility API).

        Keys in the entry function are logical buffer ids.  Reusable/nested
        function buffers use ``call/path::buffer`` keys so two invocations do
        not masquerade as one SSA value.  Because SAT allocations and call
        frames are reused, these are current raw bytes, not historical
        snapshots of every value at its definition point.
        """

        return self.diagnostic_memory_pools(clone=clone).get(
            self.buffer_plan.default_workspace, {}
        )

    def diagnostic_memory_pools(
        self, *, clone: bool = False
    ) -> dict[str, dict[str, object]]:
        """Return typed current-value views for every runtime SAT pool.

        A block- or device-scoped runtime allocation has a leading physical
        scope dimension. Nested function frames remain resolved relative to
        each scope, exactly as they are in the generated call ABI.
        """

        self._require_loaded()
        result: dict[str, dict[str, object]] = {}
        for pool in self.runtime_binding["pools"]:
            storage = str(pool["storage"])
            space = self.buffer_plan.memory_space_map.get(storage)
            if (
                space is None
                or space.strategy.value != "sat"
                or space.allocation_scope.value != "function"
                or space.kind == "shared"
            ):
                continue
            name = str(pool["name"])
            try:
                allocation = self._pool_values[name]
            except KeyError as error:
                raise RuntimeContractError(
                    f"Runtime memory pool {name!r} has not been loaded."
                ) from error
            scope_count = int(pool.get("scope_count", 1))
            scope_nbytes = int(
                pool.get(
                    "scope_nbytes",
                    int(pool["nbytes"]) // scope_count,
                )
            )
            specs = memory_pool_view_specs(
                self.buffer_plan,
                entry=self.ir_module.entry,
                memory_space=storage,
            )
            result[storage] = materialize_memory_pool_views(
                allocation,
                specs,
                scope_count=scope_count,
                scope_nbytes=scope_nbytes,
                clone=clone,
            )
        return result

    def _prepare_bound(self, values_by_buffer: Mapping[str, object]) -> None:
        self._prepare_external(self._external_values(values_by_buffer))

    def _launch_bound(
        self,
        values_by_buffer: Mapping[str, object],
        *,
        stream=None,
    ) -> None:
        self._launch_external(
            self._external_values(values_by_buffer), stream=stream
        )

    def _external_values(
        self, values_by_buffer: Mapping[str, object]
    ) -> tuple[object, ...]:
        values: list[object] = []
        for argument in self.external_arguments:
            buffer = str(argument["buffer"])
            try:
                values.append(values_by_buffer[buffer])
            except KeyError as error:
                raise RuntimeContractError(
                    f"Runtime adapter did not bind external buffer {buffer!r}."
                ) from error
        return tuple(values)

    def _prepare_external(self, external_arguments: tuple[object, ...]) -> None:
        self._require_loaded()
        self._validate_external_arguments(external_arguments)
        pool_arguments = tuple(
            self._pool_values[str(pool["name"])]
            for pool in self.runtime_binding["pools"]
        )
        descriptors = self._materialize_tensor_descriptors(
            external_arguments
        )
        arguments = (*external_arguments, *pool_arguments, *descriptors)
        contract = ResourceContract(
            compute_num_warps=int(self.codegen["num_warps"]),
            resident_blocks_per_sm=1,
        )
        self._prepared = prepare_jit_kernel(
            self.kernel,
            arguments,
            tuple(int(value) for value in self.codegen["dynamic_argument_indices"]),
            grid=tuple(int(value) for value in self.codegen["grid"]),
            contract=contract,
            **compilation_options(contract),
        )
        self.prepare_count += 1
        self._mark_prepared()

    def _launch_external(
        self, external_arguments: tuple[object, ...], *, stream=None
    ) -> None:
        self._require_prepared()
        if self._prepared is None:
            raise RuntimeContractError("Artifact must be prepared before launch.")
        self._validate_external_arguments(external_arguments)
        descriptors = self._materialize_tensor_descriptors(
            external_arguments
        )
        dynamic_descriptors = tuple(
            descriptors[index] for index in self._dynamic_descriptor_indices
        )
        self._prepared.launch(
            *external_arguments,
            *dynamic_descriptors,
            stream=stream,
        )

    def _materialize_tensor_descriptors(
        self, external_arguments: tuple[object, ...]
    ) -> tuple[object, ...]:
        if not self._descriptor_specs:
            return ()
        sources = {
            str(descriptor["name"]): value
            for descriptor, value in zip(
                self.external_arguments, external_arguments, strict=True
            )
        }
        sources.update(self._pool_values)
        return self._tensor_descriptors.materialize_many(
            str(self.codegen["symbol"]),
            self._descriptor_specs,
            sources,
        )

    def _validate_external_arguments(
        self, external_arguments: tuple[object, ...]
    ) -> None:
        if len(external_arguments) != len(self.external_arguments):
            raise RuntimeContractError(
                f"TIR entry expects {len(self.external_arguments)} external "
                f"buffers, got {len(external_arguments)}."
            )
        torch = _torch()
        for descriptor, value in zip(
            self.external_arguments, external_arguments, strict=True
        ):
            if not isinstance(value, torch.Tensor):
                raise RuntimeContractError(
                    f"TIR external buffer {descriptor['buffer']!r} must be a torch.Tensor."
                )
            if str(value.device) != self._device or not value.is_contiguous():
                raise RuntimeContractError(
                    f"TIR external buffer {descriptor['buffer']!r} must be contiguous "
                    f"and on {self._device}."
                )

    def _load_rdata(self, torch, device: str, expected_nbytes: int):
        if self.manifest.get("rdata") is None:
            raise ArtifactError(
                "Executable TIR artifact requires a packed rdata image."
            )
        index = verify_rdata(self.artifact / "assets", module=self.ir_module)
        if int(index["nbytes"]) != expected_nbytes:
            raise ArtifactError(
                "TIR rdata pool size does not match its runtime binding."
            )
        host = torch.from_file(
            str(self.artifact / "assets" / str(index["image"])),
            shared=False,
            size=expected_nbytes,
            dtype=torch.uint8,
        )
        return host.to(device=device)

    def _release_resources(self) -> None:
        self._prepared = None
        self._pool_values = {}
        self._tensor_descriptors.clear()


class GeneratedTirSingleTensorModule(GeneratedTirCallGraphModule):
    """Convenience adapter for any tensor-only entry with one tensor result.

    Bufferized TIR keeps result buffers explicit and may own rdata/workspace
    pools even for a one-op graph.  This adapter preserves that physical ABI
    while exposing the ordinary ``prepare(inputs)``/``run(inputs)`` interface.
    Selection is purely structural and has no model or operation allow-list.
    """

    result_kind = "tensor"

    def __init__(self, artifact, manifest, ir_module, kernel) -> None:
        super().__init__(artifact, manifest, ir_module, kernel)
        function = ir_module.function_map[ir_module.entry]
        input_types = tuple(
            logical_type(ir_module.node_map[value].type)
            for value in function.parameters
        )
        if not all(isinstance(value, TensorType) for value in input_types):
            raise ArtifactError(
                "Single-tensor runtime requires tensor-only entry parameters."
            )
        self._input_types = input_types
        self._output_name, self._output_type = _single_tensor_output(ir_module)

    def create_outputs(self):
        self._require_loaded()
        return _allocate_tensor(_torch(), self._output_type, self._device)

    def prepare(self, *inputs, output=None):
        self._validate_inputs(inputs)
        if output is None:
            output = self.create_outputs()
        _validate_ir_tensor("output", output, self._output_type, self._device)
        self._prepare_bound(self._entry_buffer_values(inputs, output))
        return self

    def run_into(self, output, *inputs, stream=None) -> None:
        self._validate_inputs(inputs)
        _validate_ir_tensor("output", output, self._output_type, self._device)
        self._launch_bound(
            self._entry_buffer_values(inputs, output), stream=stream
        )

    def run(self, *inputs, stream=None):
        output = self.create_outputs()
        self.run_into(output, *inputs, stream=stream)
        return output

    def _validate_inputs(self, inputs) -> None:
        self._require_loaded()
        if len(inputs) != len(self._input_types):
            raise RuntimeContractError(
                f"Entry expects {len(self._input_types)} tensor inputs, "
                f"got {len(inputs)}."
            )
        for index, (value, value_type) in enumerate(
            zip(inputs, self._input_types, strict=True)
        ):
            _validate_ir_tensor(
                f"input {index}", value, value_type, self._device
            )

    def _entry_buffer_values(self, inputs, output) -> dict[str, object]:
        function = self.buffer_plan.function_map[self.ir_module.entry]
        values: dict[str, object] = {}
        for (value, buffers), actual in zip(
            function.parameters, inputs, strict=True
        ):
            if len(buffers) != 1:
                raise RuntimeContractError(
                    f"Entry parameter {value!r} has a non-scalar buffer tuple."
                )
            values[buffers[0]] = actual
        output_bindings = dict(function.outputs)[self._output_name]
        if len(output_bindings) != 1:
            raise RuntimeContractError(
                f"Entry output {self._output_name!r} has a non-scalar buffer tuple."
            )
        values[output_bindings[0]] = output
        return values


class GeneratedTirGatedDeltaNetModule(GeneratedTirCallGraphModule):
    """State-friendly runtime adapter; generated code remains model-neutral."""

    def __init__(self, artifact, manifest, ir_module, kernel) -> None:
        super().__init__(artifact, manifest, ir_module, kernel)
        self._state_config = _gdn_state_config_from_entry(ir_module)

    result_kind = "tensor_state"

    def create_outputs(self):
        torch = _torch()
        self._require_loaded()
        _, output_type = _single_tensor_output(self.ir_module)
        return _allocate_tensor(torch, output_type, self._device)

    def create_state(self) -> GatedDeltaNetState:
        torch = _torch()
        self._require_loaded()
        config = self._state_config
        return GatedDeltaNetState(
            torch.zeros(
                config.storage_shape(GatedDeltaNetStateKind.CONVOLUTION),
                dtype=torch.bfloat16,
                device=self._device,
            ),
            torch.zeros(
                config.storage_shape(GatedDeltaNetStateKind.RECURRENT),
                dtype=torch.float32,
                device=self._device,
            ),
        )

    def prepare(self, input_ids, state: GatedDeltaNetState, *, output=None):
        torch = _torch()
        self._require_loaded()
        self._validate_state(input_ids, state)
        _, output_type = _single_tensor_output(self.ir_module)
        if output is None:
            output = _allocate_tensor(torch, output_type, self._device)
        _validate_ir_tensor("output", output, output_type, self._device)
        values = self._entry_buffer_values(
            input_ids, state, output=output
        )
        self._prepare_bound(values)
        return self

    def run_into(self, output, input_ids, state: GatedDeltaNetState, *, stream=None):
        self._validate_state(input_ids, state)
        _, output_type = _single_tensor_output(self.ir_module)
        _validate_ir_tensor("output", output, output_type, self._device)
        self._launch_bound(
            self._entry_buffer_values(input_ids, state, output=output),
            stream=stream,
        )

    def run(self, input_ids, state: GatedDeltaNetState, *, stream=None):
        torch = _torch()
        _, output_type = _single_tensor_output(self.ir_module)
        output = _allocate_tensor(torch, output_type, self._device)
        self.run_into(output, input_ids, state, stream=stream)
        return output, state

    def _entry_buffer_values(self, input_ids, state, *, output):
        function = self.buffer_plan.function_map[self.ir_module.entry]
        values: dict[str, object] = {}
        tensor_parameters = [
            (value, buffers)
            for value, buffers in function.parameters
            if not isinstance(self.ir_module.node_map[value].type, RefType)
        ]
        if len(tensor_parameters) != 1 or len(tensor_parameters[0][1]) != 1:
            raise RuntimeContractError(
                "GDN runtime requires one tensor entry parameter."
            )
        values[tensor_parameters[0][1][0]] = input_ids
        state_buffers = next(
            buffers for value, buffers in function.parameters
            if isinstance(self.ir_module.node_map[value].type, RefType)
        )
        values.update(dict(zip(
            state_buffers,
            (state.convolution, state.recurrent),
            strict=True,
        )))
        for value, buffers in function.outputs:
            value_type = self.ir_module.node_map[value].type
            if isinstance(value_type, RefType):
                values.update(dict(zip(
                    buffers,
                    (state.convolution, state.recurrent),
                    strict=True,
                )))
            else:
                if len(buffers) != 1:
                    raise RuntimeContractError(
                        f"GDN output {value!r} has a non-scalar buffer tuple."
                    )
                values[buffers[0]] = output
        return values

    def _validate_state(self, input_ids, state) -> None:
        torch = _torch()
        if (
            not isinstance(input_ids, torch.Tensor)
            or tuple(input_ids.shape) != (1,)
            or input_ids.dtype != torch.int32
        ):
            raise RuntimeContractError("input_ids must be int32[1].")
        if not isinstance(state, GatedDeltaNetState):
            raise RuntimeContractError("state must be GatedDeltaNetState.")
        config = self._state_config
        expected = (
            config.storage_shape(GatedDeltaNetStateKind.CONVOLUTION),
            config.storage_shape(GatedDeltaNetStateKind.RECURRENT),
        )
        if (
            tuple(state.convolution.shape) != expected[0]
            or state.convolution.dtype != torch.bfloat16
            or tuple(state.recurrent.shape) != expected[1]
            or state.recurrent.dtype != torch.float32
        ):
            raise RuntimeContractError("GDN state does not match the final IR ABI.")


class _GeneratedTirPagedAttentionBase(GeneratedTirCallGraphModule):
    def __init__(self, artifact, manifest, ir_module, kernel) -> None:
        super().__init__(artifact, manifest, ir_module, kernel)
        self._state_config = _paged_state_config_from_entry(ir_module)

    @property
    def state_config(self) -> PagedAttentionStateConfig:
        return self._state_config

    def create_state(self) -> PagedAttentionState:
        self._require_loaded()
        return create_paged_attention_state(
            self._state_config, device=self._device
        )

    def _validate_inputs(
        self,
        input_ids,
        state: PagedAttentionState,
        *,
        dynamic_values: bool = True,
    ) -> None:
        torch = _torch()
        if (
            not isinstance(input_ids, torch.Tensor)
            or tuple(input_ids.shape) != (1,)
            or input_ids.dtype != torch.int32
        ):
            raise RuntimeContractError("input_ids must be int32[1].")
        if not isinstance(state, PagedAttentionState):
            raise RuntimeContractError("state must be PagedAttentionState.")
        if state.config != self._state_config:
            raise RuntimeContractError(
                "Paged-attention state does not match the final IR ABI."
            )
        state.validate(dynamic_values=dynamic_values)

    def _entry_buffer_values(
        self,
        input_ids,
        state: PagedAttentionState,
        outputs_by_value: Mapping[str, object],
    ) -> dict[str, object]:
        function = self.buffer_plan.function_map[self.ir_module.entry]
        values: dict[str, object] = {}
        state_values = {
            "kv_caches": state.kv_caches,
            "query_start_loc": state.query_start_loc,
            "seq_lens": state.seq_lens,
            "slot_mapping": state.slot_mapping,
            "block_table": state.block_table,
        }
        for value, buffers in function.parameters:
            value_type = self.ir_module.node_map[value].type
            if isinstance(value_type, RefType):
                values.update(_bind_ref_buffers(value_type, buffers, state_values))
            else:
                if len(buffers) != 1:
                    raise RuntimeContractError(
                        f"Entry parameter {value!r} has a non-scalar buffer tuple."
                    )
                values[buffers[0]] = input_ids
        for value, buffers in function.outputs:
            value_type = self.ir_module.node_map[value].type
            if isinstance(value_type, RefType):
                values.update(_bind_ref_buffers(value_type, buffers, state_values))
                continue
            try:
                output = outputs_by_value[value]
            except KeyError as error:
                raise RuntimeContractError(
                    f"Runtime adapter did not bind entry output {value!r}."
                ) from error
            if len(buffers) != 1:
                raise RuntimeContractError(
                    f"Entry output {value!r} has a non-scalar buffer tuple."
                )
            values[buffers[0]] = output
        return values


class GeneratedTirPagedAttentionLayerModule(_GeneratedTirPagedAttentionBase):
    """Friendly adapter for an entry returning one tensor and updated state."""

    result_kind = "tensor_state"

    def create_outputs(self):
        torch = _torch()
        self._require_loaded()
        _, output_type = _single_tensor_output(self.ir_module)
        return _allocate_tensor(torch, output_type, self._device)

    def prepare(self, input_ids, state: PagedAttentionState, *, output=None):
        torch = _torch()
        self._require_loaded()
        self._validate_inputs(input_ids, state, dynamic_values=True)
        output_name, output_type = _single_tensor_output(self.ir_module)
        if output is None:
            output = _allocate_tensor(torch, output_type, self._device)
        _validate_ir_tensor("output", output, output_type, self._device)
        self._prepare_bound(self._entry_buffer_values(
            input_ids, state, {output_name: output}
        ))
        return self

    def run_into(self, output, input_ids, state: PagedAttentionState, *, stream=None):
        # prepare() proves content-dependent invariants such as cache capacity
        # and the physical page table.  A prepared launch may update cache
        # contents but not its structural ABI; re-running CUDA reductions to
        # prove the same invariant on every token would put validation kernels
        # on the decode critical path.
        self._validate_inputs(input_ids, state, dynamic_values=False)
        output_name, output_type = _single_tensor_output(self.ir_module)
        _validate_ir_tensor("output", output, output_type, self._device)
        self._launch_bound(
            self._entry_buffer_values(input_ids, state, {output_name: output}),
            stream=stream,
        )

    def run(self, input_ids, state: PagedAttentionState, *, stream=None):
        torch = _torch()
        output_name, output_type = _single_tensor_output(self.ir_module)
        output = _allocate_tensor(torch, output_type, self._device)
        self.run_into(output, input_ids, state, stream=stream)
        return output, state


class GeneratedTirPagedAttentionModelModule(_GeneratedTirPagedAttentionBase):
    """Friendly adapter for logits/token outputs and an updated cache state."""

    result_kind = "logits_token_state"

    def create_outputs(self):
        torch = _torch()
        self._require_loaded()
        outputs = _named_tensor_outputs(self.ir_module)
        _, logits_type = _output_by_dtype(outputs, DType.FLOAT32)
        _, token_type = _output_by_dtype(outputs, DType.INT32)
        return (
            _allocate_tensor(torch, logits_type, self._device),
            _allocate_tensor(torch, token_type, self._device),
        )

    def prepare(
        self,
        input_ids,
        state: PagedAttentionState,
        *,
        logits=None,
        next_token=None,
    ):
        torch = _torch()
        self._require_loaded()
        self._validate_inputs(input_ids, state, dynamic_values=True)
        outputs = _named_tensor_outputs(self.ir_module)
        logits_name, logits_type = _output_by_dtype(outputs, DType.FLOAT32)
        token_name, token_type = _output_by_dtype(outputs, DType.INT32)
        if logits is None:
            logits = _allocate_tensor(torch, logits_type, self._device)
        if next_token is None:
            next_token = _allocate_tensor(torch, token_type, self._device)
        _validate_ir_tensor("logits", logits, logits_type, self._device)
        _validate_ir_tensor("next_token", next_token, token_type, self._device)
        self._prepare_bound(self._entry_buffer_values(
            input_ids,
            state,
            {logits_name: logits, token_name: next_token},
        ))
        return self

    def run_into(
        self,
        logits,
        next_token,
        input_ids,
        state: PagedAttentionState,
        *,
        stream=None,
    ) -> None:
        self._validate_inputs(input_ids, state, dynamic_values=False)
        outputs = _named_tensor_outputs(self.ir_module)
        logits_name, logits_type = _output_by_dtype(outputs, DType.FLOAT32)
        token_name, token_type = _output_by_dtype(outputs, DType.INT32)
        _validate_ir_tensor("logits", logits, logits_type, self._device)
        _validate_ir_tensor("next_token", next_token, token_type, self._device)
        self._launch_bound(
            self._entry_buffer_values(
                input_ids,
                state,
                {logits_name: logits, token_name: next_token},
            ),
            stream=stream,
        )

    def run(self, input_ids, state: PagedAttentionState, *, stream=None):
        torch = _torch()
        outputs = _named_tensor_outputs(self.ir_module)
        _, logits_type = _output_by_dtype(outputs, DType.FLOAT32)
        _, token_type = _output_by_dtype(outputs, DType.INT32)
        logits = _allocate_tensor(torch, logits_type, self._device)
        next_token = _allocate_tensor(torch, token_type, self._device)
        self.run_into(logits, next_token, input_ids, state, stream=stream)
        return logits, next_token, state


def create_tir_runtime(artifact, manifest, ir_module, kernel):
    """Select a user-facing runtime adapter at the artifact ABI boundary.

    The generated package remains a model-independent TIR call graph. Friendly
    state wrappers are selected from the entry's structural RefType ABI, not
    from importer architecture names or codegen-side model reconstruction.
    """

    function = ir_module.function_map[ir_module.entry]
    if any(
        isinstance(ir_module.node_map[value].type, TupleType)
        for value in (*function.parameters, *function.outputs)
    ):
        # Aggregate entries expose flattened buffers through the general ABI.
        # Tensor/state convenience adapters require top-level named tensors;
        # a one-leaf tuple is still an aggregate, not a scalar tensor result.
        return GeneratedTirCallGraphModule(artifact, manifest, ir_module, kernel)
    reference_types = tuple(
        ir_module.node_map[value].type
        for value in function.parameters
        if isinstance(ir_module.node_map[value].type, RefType)
    )
    reference_fields = (
        frozenset(name for name, _ in reference_types[0].fields)
        if len(reference_types) == 1
        else frozenset()
    )
    if reference_fields == {"convolution", "recurrent"}:
        return GeneratedTirGatedDeltaNetModule(
            artifact, manifest, ir_module, kernel
        )
    if reference_fields == {
        "kv_caches",
        "query_start_loc",
        "seq_lens",
        "slot_mapping",
        "block_table",
    }:
        tensor_outputs = _named_tensor_outputs(ir_module)
        runtime_type = (
            GeneratedTirPagedAttentionModelModule
            if len(tensor_outputs) == 2
            else GeneratedTirPagedAttentionLayerModule
        )
        return runtime_type(artifact, manifest, ir_module, kernel)
    if not reference_types and len(_named_tensor_outputs(ir_module)) == 1:
        parameter_types = tuple(
            logical_type(ir_module.node_map[value].type)
            for value in function.parameters
        )
        if all(isinstance(value, TensorType) for value in parameter_types):
            return GeneratedTirSingleTensorModule(
                artifact, manifest, ir_module, kernel
            )
    return GeneratedTirCallGraphModule(
        artifact, manifest, ir_module, kernel
    )


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeContractError("FlagMega generated runtime requires PyTorch tensor/storage APIs.") from error
    return torch


def _gdn_state_config_from_entry(ir_module) -> GatedDeltaNetStateConfig:
    function = ir_module.function_map[ir_module.entry]
    state_types = [
        ir_module.node_map[value].type
        for value in function.parameters
        if isinstance(ir_module.node_map[value].type, RefType)
    ]
    if len(state_types) != 1:
        raise ArtifactError(
            "GDN runtime requires exactly one RefType entry parameter."
        )
    fields = dict(state_types[0].fields)
    try:
        convolution = fields["convolution"]
        recurrent = fields["recurrent"]
    except KeyError as error:
        raise ArtifactError(
            "GDN runtime state must contain convolution and recurrent fields."
        ) from error
    if not isinstance(convolution, TensorType) or not isinstance(recurrent, TensorType):
        raise ArtifactError("GDN runtime state fields must be tensors.")
    if not isinstance(convolution.dtype, VectorType) or not isinstance(recurrent.dtype, VectorType):
        raise ArtifactError("GDN runtime state fields must retain vector lanes.")
    conv_shape = _fixed_shape(convolution, "GDN convolution state")
    recurrent_shape = _fixed_shape(recurrent, "GDN recurrent state")
    if len(conv_shape) != 3 or len(recurrent_shape) != 4:
        raise ArtifactError("GDN runtime state has an unsupported rank.")
    convolution_lanes = tuple(int(value) for value in convolution.dtype.lanes)
    recurrent_lanes = tuple(int(value) for value in recurrent.dtype.lanes)
    num_layers, packed_conv_dim, history = conv_shape
    _, num_value_heads, value_head_dim, packed_key_dim = recurrent_shape
    conv_dim = packed_conv_dim * prod(convolution_lanes)
    key_head_dim = packed_key_dim * prod(recurrent_lanes)
    residual = conv_dim - num_value_heads * value_head_dim
    denominator = 2 * key_head_dim
    if residual <= 0 or residual % denominator:
        raise ArtifactError(
            "GDN state dimensions cannot recover an integral key-head count."
        )
    _, output_type = _single_tensor_output(ir_module)
    output_shape = _fixed_shape(output_type, "GDN entry output")
    if len(output_shape) != 2 or output_shape[0] != 1:
        raise ArtifactError("GDN entry output must have shape [1, hidden_size].")
    hidden_size = output_shape[1]
    return GatedDeltaNetStateConfig(
        num_layers=num_layers,
        num_key_heads=residual // denominator,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_size=history + 1,
        hidden_size=hidden_size,
        convolution_lanes=convolution_lanes,
        recurrent_lanes=recurrent_lanes,
    )


def _fixed_shape(value_type: TensorType, name: str) -> tuple[int, ...]:
    if not all(dimension.is_fixed for dimension in value_type.shape):
        raise ArtifactError(f"{name} requires a static shape.")
    return tuple(dimension.fixed_value for dimension in value_type.shape)


def _paged_state_config_from_entry(ir_module) -> PagedAttentionStateConfig:
    function = ir_module.function_map[ir_module.entry]
    state_types = [
        ir_module.node_map[value].type
        for value in function.parameters
        if isinstance(ir_module.node_map[value].type, RefType)
    ]
    if len(state_types) != 1:
        raise ArtifactError(
            "Paged-attention runtime requires exactly one RefType entry parameter."
        )
    fields = dict(state_types[0].fields)
    cache = fields.get("kv_caches")
    if not isinstance(cache, TensorType) or not isinstance(cache.dtype, VectorType):
        raise ArtifactError(
            "Paged-attention runtime requires a vectorized kv_caches tensor."
        )
    shape = _fixed_shape(cache, "Paged-attention KV cache")
    if len(shape) != 6 or shape[2] != 2:
        raise ArtifactError("Paged-attention KV cache has an unsupported layout.")
    lanes = prod(cache.dtype.lanes)
    return PagedAttentionStateConfig(
        num_layers=shape[1],
        num_kv_heads=shape[4],
        head_dim=shape[5] * lanes,
        block_size=shape[3],
        num_blocks=shape[0],
        lanes=lanes,
    )


def _bind_ref_buffers(
    ref_type: RefType,
    buffers: tuple[str, ...],
    values_by_field: Mapping[str, object],
) -> dict[str, object]:
    fields = tuple(name for name, _ in ref_type.fields)
    if len(fields) != len(buffers):
        raise RuntimeContractError(
            f"RefType {ref_type.name!r} field/buffer arity differs."
        )
    try:
        return {
            buffer: values_by_field[field]
            for field, buffer in zip(fields, buffers, strict=True)
        }
    except KeyError as error:
        raise RuntimeContractError(
            f"Runtime state does not bind RefType field {error.args[0]!r}."
        ) from error


def _named_tensor_outputs(ir_module) -> dict[str, TensorType]:
    function = ir_module.function_map[ir_module.entry]
    result: dict[str, TensorType] = {}
    for value in function.outputs:
        value_type = ir_module.node_map[value].type
        if isinstance(value_type, RefType):
            continue
        logical = logical_type(value_type)
        if not isinstance(logical, TensorType):
            raise ArtifactError(
                f"Entry output {value!r} does not have a tensor runtime ABI."
            )
        result[value] = logical
    return result


def _single_tensor_output(ir_module) -> tuple[str, TensorType]:
    outputs = _named_tensor_outputs(ir_module)
    if len(outputs) != 1:
        raise ArtifactError(
            f"Layer runtime requires one tensor output, found {tuple(outputs)}."
        )
    return next(iter(outputs.items()))


def _output_by_dtype(
    outputs: Mapping[str, TensorType], dtype: DType
) -> tuple[str, TensorType]:
    matched = tuple(
        (name, value_type)
        for name, value_type in outputs.items()
        if value_type.dtype == dtype
    )
    if len(matched) != 1:
        raise ArtifactError(
            f"Entry requires one {dtype.value} output, found "
            f"{tuple(name for name, _ in matched)}."
        )
    return matched[0]


def _allocate_tensor(torch, value_type: TensorType, device: str):
    return torch.empty(
        _runtime_tensor_shape(value_type, "Runtime output"),
        dtype=_torch_dtype(torch, value_type.dtype),
        device=device,
    )


def _validate_ir_tensor(
    name: str,
    value,
    value_type: TensorType,
    device: str,
) -> None:
    torch = _torch()
    shape = _runtime_tensor_shape(value_type, f"Runtime {name}")
    dtype = _torch_dtype(torch, value_type.dtype)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
    ):
        raise RuntimeContractError(
            f"{name} must have shape {shape} and dtype {dtype}."
        )
    if str(value.device) != device or not value.is_contiguous():
        raise RuntimeContractError(
            f"{name} must be contiguous and on {device}."
        )


def _runtime_tensor_shape(value_type: TensorType, name: str) -> tuple[int, ...]:
    shape = _fixed_shape(value_type, name)
    if isinstance(value_type.dtype, VectorType):
        shape += tuple(value_type.dtype.lanes)
    return shape


def _torch_dtype(torch, dtype):
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    try:
        return {
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32,
            DType.INT32: torch.int32,
            DType.INT64: torch.int64,
            DType.FLOAT8_E4M3FN: torch.float8_e4m3fn,
        }[scalar]
    except KeyError as error:
        raise ArtifactError(
            f"Runtime cannot materialize dtype {scalar!r}."
        ) from error


def _validate_sm90_device(torch, device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeContractError(f"Executable target nvidia-sm90 requires a CUDA device, got {device!r}.")
    capability = torch.cuda.get_device_capability(torch.device(device))
    if capability != (9, 0):
        raise RuntimeContractError(f"Artifact target nvidia-sm90 requires capability (9, 0), got {capability}.")


def _typed_byte_view(storage, offset: int, nbytes: int, dtype: object, shape: tuple[int, ...]):
    torch = _torch()
    try:
        data_type = data_type_from_data(dtype)
    except (IRSchemaError, TypeError, ValueError) as error:
        raise ArtifactError(f"Generated runtime cannot decode rdata/workspace dtype {dtype!r}.") from error
    scalar_type = data_type.elem_type if isinstance(data_type, VectorType) else data_type
    dtype_value = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }.get(scalar_type.value)
    if dtype_value is None:
        raise ArtifactError(f"Generated runtime cannot materialize rdata/workspace dtype {dtype!r}.")
    if offset < 0 or nbytes < 0 or offset + nbytes > storage.numel():
        raise ArtifactError(f"Typed byte view [{offset}, {offset + nbytes}) is outside its storage.")
    lanes = data_type.lanes if isinstance(data_type, VectorType) else ()
    view = storage.narrow(0, offset, nbytes).view(dtype_value).reshape((*shape, *lanes))
    if view.data_ptr() % 16:
        raise ArtifactError("Generated static buffer is not 16-byte aligned.")
    return view
