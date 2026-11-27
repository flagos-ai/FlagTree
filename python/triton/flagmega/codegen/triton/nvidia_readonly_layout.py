# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed readonly owner layouts for NVIDIA descriptor consumers.

The canonical BufferPlan remains the artifact ABI.  A separate immutable view
describes the runtime pool after the common reversible byte permutation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from math import prod
from types import MappingProxyType

from triton.flagmega.codegen.readonly_layout import (
    READONLY_DEVICE_LAYOUT_SCHEMA,
    readonly_roots,
    owner_major_weight_records,
)
from triton.flagmega.codegen.readonly_layout_policy import ReadonlyLayoutPolicy
from triton.flagmega.codegen.triton.templates import module_template_target
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import verify_buffer_plan

SCHEMA = "flagmega.triton-readonly-layout/v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _dense_strides(shape, scalar_lanes):
    strides = []
    running = scalar_lanes
    for extent in reversed(shape):
        strides.append(running)
        running *= extent
    return tuple(reversed(strides))


@dataclass(frozen=True)
class OwnerMajorWeightView:
    buffer: str
    artifact_abi_sha256: str
    local_shape: tuple[int, ...]
    scalar_lanes: int
    owners: int
    roots: tuple[str, ...]

    def __post_init__(self):
        if (not self.local_shape or any(type(v) is not int or v <= 0 for v in self.local_shape)
                or type(self.scalar_lanes) is not int or self.scalar_lanes <= 0 or type(self.owners) is not int
                or self.owners <= 1 or not self.roots):
            raise CodegenError("Invalid readonly owner-component geometry.")

    @property
    def component_elements(self):
        return prod(self.local_shape) * self.scalar_lanes

    def resolve(self, artifact_abi):
        """Derive a real runtime component ABI; never mutate the artifact ABI."""
        if (digest(artifact_abi) != self.artifact_abi_sha256
                or tuple(artifact_abi["local_capacity_shape"]) != self.local_shape
                or artifact_abi["scalar_lane_count"] != self.scalar_lanes or artifact_abi["owner_count"] != self.owners
                or artifact_abi["scalar_dtype"] != "bfloat16"):
            raise CodegenError("Readonly device view does not match its canonical binding.")
        return {
            **artifact_abi, "storage_kind": "compact_per_owner", "coordinate_space": "local", "scalar_storage_strides":
            _dense_strides(self.local_shape,
                           self.scalar_lanes), "component_stride_scalar_elements": self.component_elements,
            "component_stride_bytes": self.component_elements * 2, "owner_stride_bytes": self.component_elements * 2
        }

    def to_data(self):
        return {
            "buffer": self.buffer, "artifact_abi_sha256": self.artifact_abi_sha256, "scalar_dtype": "bfloat16",
            "local_shape": list(self.local_shape), "scalar_lanes": self.scalar_lanes, "owners": self.owners,
            "component_nbytes": self.component_elements * 2, "total_nbytes": self.component_elements * self.owners * 2,
            "scalar_strides": list(_dense_strides(self.local_shape, self.scalar_lanes)), "roots": list(self.roots)
        }


@dataclass(frozen=True)
class ReadonlyDevicePlan:
    views: Mapping
    data: Mapping


def plan_readonly_layout(module, calls, policy):
    """Validate the complete root/reader closure before selecting either weight."""
    if not isinstance(policy, ReadonlyLayoutPolicy):
        raise CodegenError("Readonly layout selection requires an immutable policy.")
    if module_template_target(module) != ("nvidia", "sm90"):
        raise CodegenError("Owner-major descriptor layout requires NVIDIA SM90.")
    plan = verify_buffer_plan(module)
    records, views, selected = {}, {}, set()
    calls = tuple(calls)
    for index, call in enumerate(calls):
        if call.get("family") not in policy.families:
            continue
        if (call.get("variant") != "packed_tensor_descriptor_smem_pipeline_gemv"
                or call.get("descriptor_kind") != "table" or not call.get("paired_weight_fields")):
            raise CodegenError("Selected readonly GLU requires paired owner-table descriptors.")
        pair = []
        for role in ("gate_weight", "up_weight"):
            argument = next((v for v in call["inputs"] if v["formal"] == role), None)
            if argument is None or len(argument.get("buffers", ())) != 1:
                raise CodegenError("Readonly GLU requires exactly one physical leaf per weight.")
            binding = argument["buffers"][0]
            abi = binding["abi"]
            found = owner_major_weight_records(plan, binding, int(call["local_n_capacity"]),
                                               int(call["local_k_capacity"]), int(abi["scalar_storage_strides"][0]))
            if not found:
                raise CodegenError("Readonly GLU weight lacks a closed full-K contiguous owner proof.")
            local_shape = tuple(int(v) for v in abi["local_capacity_shape"])
            view = OwnerMajorWeightView(binding["actual"], digest(abi), local_shape, int(abi["scalar_lane_count"]),
                                        int(abi["owner_count"]), tuple(sorted({v["buffer"]
                                                                               for v in found})))
            if any(v["nbytes"] != view.component_elements * view.owners * 2 for v in found):
                raise CodegenError("Readonly owner components do not cover their complete weight.")
            pair.append((role, binding, view, found))
        for role, binding, view, found in pair:
            previous = views.setdefault(binding["actual"], view)
            if previous != view:
                raise CodegenError("One readonly formal has conflicting device layouts.")
            selected.add((index, role, binding["actual"]))
            for record in found:
                previous = records.setdefault(record["offset"], record)
                if previous != record:
                    raise CodegenError("Readonly roots overlap with conflicting layout identities.")
    if not records:
        raise CodegenError("Readonly policy selected no closed GLU weights.")
    ordered = tuple(records[k] for k in sorted(records))
    if any(a["offset"] + a["nbytes"] > b["offset"] for a, b in zip(ordered, ordered[1:])):
        raise CodegenError("Readonly device layout ranges overlap.")

    def intersects(buffer):
        span = buffer.physical_access_span
        return any(span.offset < r["offset"] + r["nbytes"] and r["offset"] < span.offset + span.nbytes for r in ordered)

    for index, call in enumerate(calls):
        for argument in call.get("inputs", ()):
            for binding in argument.get("buffers", ()):
                roots = readonly_roots(plan, plan.buffer_map[binding["actual"]])
                if any(intersects(root)
                       for root in roots) and (index, argument["formal"], binding["actual"]) not in selected:
                    raise CodegenError("A readonly device-layout root has an incompatible reader.")
    # Returning a selected root exposes a canonical reader outside this closed
    # executable.  Nested calls are already included in the reader census.
    for _, output_buffers in plan.entry_outputs:
        for buffer_id in output_buffers:
            if any(intersects(root) for root in readonly_roots(plan, plan.buffer_map[buffer_id])):
                raise CodegenError("A readonly device-layout root escapes as a program output.")
    device_layout = {"schema": READONLY_DEVICE_LAYOUT_SCHEMA, "buffers": list(ordered)}
    data = {
        "schema": SCHEMA, "policy": policy.to_data(), "module_semantic_hash": module.semantic_hash, "device_layout":
        device_layout, "bindings": [views[k].to_data() for k in sorted(views)]
    }
    data["plan_sha256"] = digest(data)
    return ReadonlyDevicePlan(MappingProxyType(views), MappingProxyType(data))


def finish_layout_data(plan, descriptor_specs):
    data = dict(plan.data)
    data.pop("plan_sha256")
    data["descriptor_sha256"] = digest(descriptor_specs)
    data["plan_sha256"] = digest(data)
    return data


def validate_layout_data(codegen, *, module_semantic_hash=None):
    """Check the sealed device-layout/descriptor identity before pool mapping."""
    from triton.flagmega.errors import ArtifactError
    data = codegen.get("triton_readonly_layout")
    if data is None:
        return
    if not isinstance(data, Mapping) or data.get("schema") != SCHEMA:
        raise ArtifactError("Invalid readonly device-layout manifest schema.")
    payload = {k: v for k, v in data.items() if k not in {"plan_sha256", "file", "file_sha256"}}
    if (data.get("plan_sha256") != digest(payload)
            or data.get("descriptor_sha256") != digest(codegen.get("host_tensor_descriptor_specs", ()))
            or data.get("device_layout") != codegen.get("triton_readonly_device_layout")
            or module_semantic_hash is not None and data.get("module_semantic_hash") != module_semantic_hash):
        raise ArtifactError("Readonly layout, canonical module, and tensor descriptors have inconsistent identities.")


def write_layout_manifest(destination, codegen):
    data = codegen.get("triton_readonly_layout")
    if data is None:
        return
    validate_layout_data(codegen)
    name = "readonly-layout.json"
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    (destination / name).write_text(text, encoding="utf-8")
    data["file"] = name
    data["file_sha256"] = hashlib.sha256(text.encode()).hexdigest()


def validate_layout_section(artifact, codegen, sections, module_semantic_hash):
    from triton.flagmega.artifacts.manifest import resolve_artifact_path
    from triton.flagmega.errors import ArtifactError
    data = codegen.get("triton_readonly_layout")
    section = sections.get("codegen.readonly_layout")
    if data is None:
        if section is not None:
            raise ArtifactError("Readonly layout section has no policy identity.")
        return
    validate_layout_data(codegen, module_semantic_hash=module_semantic_hash)
    if (section is None or section.kind != SCHEMA or section.path != data.get("file")
            or section.sha256 != data.get("file_sha256")):
        raise ArtifactError("Readonly device layout does not match its sealed section.")
    saved = json.loads(resolve_artifact_path(artifact, section.path).read_text(encoding="utf-8"))
    if saved != {k: v for k, v in data.items() if k not in {"file", "file_sha256"}}:
        raise ArtifactError("Readonly device layout differs from its sealed section content.")


__all__ = ["OwnerMajorWeightView", "ReadonlyDevicePlan", "plan_readonly_layout"]
