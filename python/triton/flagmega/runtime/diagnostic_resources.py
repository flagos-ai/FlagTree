# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Immutable reviewed diagnostic resource identities, separate from timing policy.

This input records an exact native footprint. It never changes compilation
options, hardware capacities, ordinary spill policy or numerical admission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import re

from triton.flagmega.artifacts.manifest import resolve_artifact_path
from triton.flagmega.artifacts.program_sources import program_source_records
from triton.flagmega.errors import ArtifactError, RuntimeContractError


_SEAL = object()
_SCHEMA = "flagmega.diagnostic-resource-contract/v1"
_RESOURCE_FIELDS = {
    "registers_per_thread", "physical_num_warps", "block_threads",
    "dynamic_shared_memory_bytes", "grid", "stack_frame_bytes", "local_memory_bytes",
    "spill_store_bytes", "spill_load_bytes", "spill_bytes",
    "global_scratch_bytes_per_CTA", "global_scratch_alignment",
    "profile_scratch_bytes_per_CTA", "profile_scratch_alignment",
}
_TRACE_FIELDS = {"schema", "config_sha256", "dictionary_sha256",
                 "config_file_sha256", "dictionary_file_sha256"}


def _error(message):
    raise ArtifactError("Diagnostic resource contract: " + message)


def _object(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        _error(f"invalid {label} fields.")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ArtifactError("Diagnostic resource contract requires finite JSON data.") from error


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _error("requires complete SHA256 identities.")


def _pin(value):
    _object(value, {"path", "sha256", "nbytes"}, "file identity")
    _sha(value["sha256"])
    if not isinstance(value["path"], str) or not Path(value["path"]).is_absolute():
        _error("file identities require absolute paths.")
    if type(value["nbytes"]) is not int or value["nbytes"] < 0:
        _error("file byte counts must be nonnegative integers.")


def _read(value):
    _pin(value)
    try:
        payload = Path(value["path"]).read_bytes()
    except OSError as error:
        raise ArtifactError(f"Cannot read diagnostic identity {value['path']}.") from error
    if len(payload) != value["nbytes"] or hashlib.sha256(payload).hexdigest() != value["sha256"]:
        _error(f"changed file identity {value['path']}.")
    return payload


def _json(value):
    try:
        data = json.loads(_read(value))
    except (ValueError, UnicodeError) as error:
        raise ArtifactError("Diagnostic identity must contain valid JSON.") from error
    if type(data) is not dict:
        _error("diagnostic JSON identity must contain an object.")
    return data


def _validate(data):
    _object(data, {"schema", "execution_purpose", "accepted_timing_eligible", "artifact_identity",
                   "base_semantic_hash", "target", "codegen_backend", "trace_identity", "entries",
                   "evidence_refs", "approval_provenance"}, "top-level")
    if (data["schema"] != _SCHEMA or data["execution_purpose"] != "diagnostic_only"
            or data["accepted_timing_eligible"] is not False):
        _error("unsupported schema or execution purpose; diagnostics are timing-ineligible.")
    if data["target"] != "nvidia-sm90" or data["codegen_backend"] != "triton":
        _error("unsupported diagnostic backend/target.")
    _sha(data["base_semantic_hash"])
    _pin(data["artifact_identity"])
    _object(data["trace_identity"], _TRACE_FIELDS, "trace identity")
    if data["trace_identity"]["schema"] not in {"flagmega.triton-trace/v1", "flagmega.triton-census/v1"}:
        _error("unsupported trace schema.")
    for key in _TRACE_FIELDS - {"schema"}:
        _sha(data["trace_identity"][key])
    entries = data["entries"]
    if type(entries) is not list or not entries:
        _error("requires complete entries.")
    names = set()
    for entry in entries:
        _object(entry, {"symbol", "source_identity", "native_key", "native_cubin_identity",
                        "native_metadata_identity", "exact_resources"}, "entry")
        if not isinstance(entry["symbol"], str) or not entry["symbol"].isidentifier() or entry["symbol"] in names:
            _error("invalid or duplicated entry.")
        names.add(entry["symbol"])
        _sha(entry["native_key"])
        for field in ("source_identity", "native_cubin_identity", "native_metadata_identity"):
            _pin(entry[field])
        resources = entry["exact_resources"]
        _object(resources, _RESOURCE_FIELDS, "resource component")
        for field in _RESOURCE_FIELDS - {"grid"}:
            if type(resources[field]) is not int or resources[field] < 0:
                _error("resource components must be nonnegative integers.")
        for field in ("registers_per_thread", "physical_num_warps", "block_threads",
                      "global_scratch_alignment", "profile_scratch_alignment"):
            if resources[field] == 0:
                _error("resource counts/alignments must be positive.")
        for field in ("global_scratch_alignment", "profile_scratch_alignment"):
            if resources[field] & (resources[field] - 1):
                _error("scratch alignment must be a power of two.")
        grid = resources["grid"]
        if type(grid) is not list or len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid):
            _error("grid must have three positive integer dimensions.")
        if resources["spill_bytes"] != resources["spill_store_bytes"] + resources["spill_load_bytes"]:
            _error("spill sum differs from its separate store/load components.")
    if type(data["evidence_refs"]) is not list or not data["evidence_refs"]:
        _error("requires independent evidence references.")
    for value in data["evidence_refs"]:
        _pin(value)
    provenance = data["approval_provenance"]
    _object(provenance, {"resource_input_identity", "resource_input_key", "root_flow_identity",
                         "plan_identity", "approval_refs"}, "approval provenance")
    for field in ("resource_input_identity", "root_flow_identity", "plan_identity"):
        _pin(provenance[field])
    if not isinstance(provenance["resource_input_key"], str) or not provenance["resource_input_key"]:
        _error("requires a reviewed resource-input key.")
    if type(provenance["approval_refs"]) is not list or not provenance["approval_refs"]:
        _error("requires applicable independent approvals.")
    for value in provenance["approval_refs"]:
        _pin(value)


def _verify_provenance(data):
    provenance = data["approval_provenance"]
    reviewed = _json(provenance["resource_input_identity"])
    technical = {key: value for key, value in data.items() if key != "approval_provenance"}
    if reviewed.get("fixtures", {}).get(provenance["resource_input_key"]) != technical:
        _error("resource data differs from independently reviewed input.")
    plan = _read(provenance["plan_identity"])
    if provenance["resource_input_identity"]["sha256"].encode() not in plan:
        _error("reviewed plan does not bind the resource-input bytes.")
    flow = _json(provenance["root_flow_identity"])
    if (flow.get("schema") != "flagmega.orchestrator-flow-signoff/v1"
            or flow.get("implementation_authorized") is not True):
        _error("requires an authorized root flow.")
    expected_plan = flow.get("technical_plan", {})
    if (expected_plan.get("sha256") != provenance["plan_identity"]["sha256"]
            or expected_plan.get("bytes", expected_plan.get("nbytes")) != provenance["plan_identity"]["nbytes"]
            or Path(expected_plan.get("path", "")).resolve() != Path(provenance["plan_identity"]["path"]).resolve()):
        _error("root flow selects another plan.")
    supplied = provenance["approval_refs"]
    required = flow.get("applicable_own_scope_receipts", [])
    if not required or len(supplied) != len(required) or len({p["path"] for p in supplied}) != len(supplied):
        _error("partial or duplicated approval coverage.")
    for row, value in zip(required, supplied, strict=True):
        record = row.get("receipt", {})
        if (row.get("technical_body_sha256") != provenance["plan_identity"]["sha256"]
                or record.get("sha256") != value["sha256"]
                or record.get("bytes", record.get("nbytes")) != value["nbytes"]
                or Path(record.get("path", "")).resolve() != Path(value["path"]).resolve()):
            _error("approval differs from applicable root-flow coverage.")
        _json(value)
    for value in data["evidence_refs"]:
        _read(value)


@dataclass(frozen=True)
class DiagnosticResourceContract:
    """Strict input constructed from reviewed resource data and root provenance."""

    _payload: str

    @classmethod
    def from_data(cls, data):
        payload = _canonical(data)
        _validate(json.loads(payload))
        return cls(payload)

    def bind(self, artifact, manifest):
        data = json.loads(self._payload)
        _validate(data)
        root = Path(artifact).resolve()
        if Path(data["artifact_identity"]["path"]).resolve() != root / "artifact.json":
            _error("selected artifact differs from reviewed artifact.")
        if _json(data["artifact_identity"]) != manifest:
            _error("selected manifest differs from its exact bytes.")
        if (manifest.get("semantic_hash") != data["base_semantic_hash"]
                or manifest.get("target") != data["target"]
                or manifest.get("codegen_backend", "triton") != data["codegen_backend"]):
            _error("artifact semantic/backend identity differs.")
        codegen = manifest.get("codegen", {})
        if codegen.get("kind") != "tir_call_graph/v1" or codegen.get("program_entry_aliases"):
            _error("unsupported diagnostic adapter or aliased entry.")
        trace = codegen.get("triton_trace", {})
        if (not isinstance(trace, dict) or not isinstance(trace.get("config"), dict)
                or not isinstance(trace.get("dictionary"), dict)
                or trace.get("accepted_timing_eligible") is not False):
            _error("requires existing timing-ineligible typed diagnostics.")
        if {key: trace.get(key) for key in _TRACE_FIELDS} != data["trace_identity"]:
            _error("trace configuration/dictionary identity differs.")
        from triton.flagmega.codegen.triton.trace_config import validate_option
        from triton.flagmega.errors import CodegenError
        try:
            validate_option(trace.get("config"), target=data["target"], codegen_backend=data["codegen_backend"])
        except CodegenError as error:
            raise ArtifactError(str(error)) from error
        pins = [data["artifact_identity"]]
        for field in ("config", "dictionary"):
            value = trace.get(field)
            digest = hashlib.sha256(_canonical(value).encode()).hexdigest()
            if digest != trace[field + "_sha256"] or value.get("base_semantic_hash") != data["base_semantic_hash"]:
                _error("broken trace semantic/config/dictionary links.")
            source = resolve_artifact_path(root, trace[field + "_file"])
            sections = [s for s in manifest.get("sections", ()) if s.get("name") == "diagnostic.trace." + field]
            if len(sections) != 1 or sections[0].get("path") != trace[field + "_file"]:
                _error("missing or ambiguous diagnostic section.")
            section = sections[0]
            identity = dict(path=str(source), sha256=section.get("sha256"), nbytes=section.get("nbytes"))
            if identity["sha256"] != trace[field + "_file_sha256"] or _json(identity) != value:
                _error("diagnostic section differs from descriptor.")
            pins.append(identity)
        if trace["dictionary"].get("config_sha256") != trace["config_sha256"]:
            _error("dictionary points to another configuration.")
        census = trace["schema"] == "flagmega.triton-census/v1"
        dictionary_schema = "flagmega.census-sites/v1" if census else "flagmega.trace-sites/v1"
        if trace["dictionary"].get("schema") != dictionary_schema:
            _error("dictionary schema differs from trace purpose.")
        native_contract = trace.get("native_contract", {})
        if not isinstance(native_contract, dict):
            _error("missing native recorder contract.")
        if census:
            if (not isinstance(trace["dictionary"].get("occurrences"), list)
                    or len(trace["dictionary"]["occurrences"]) != trace["config"]["expected_occurrences"]
                    or native_contract.get("occurrences") != trace["config"]["expected_occurrences"]
                    or native_contract.get("ctas") != trace["config"]["ctas"]):
                _error("incomplete census occurrence/CTA coverage.")
        elif (not isinstance(trace["dictionary"].get("sites"), list)
              or native_contract.get("typed_ctas") != trace["config"]["typed_ctas"]
              or native_contract.get("capacity") != trace["config"]["capacity"]
              or native_contract.get("record_bytes") != 64
              or native_contract.get("runtime_stream_filtering") is not False):
            _error("incomplete ring recorder coverage/ABI.")
        specs = codegen.get("kernels")
        symbols = [s["symbol"] for s in specs] if specs is not None else [codegen.get("symbol")]
        if symbols != [entry["symbol"] for entry in data["entries"]] or len(set(symbols)) != len(symbols):
            _error("incomplete or reordered program entry coverage.")
        records = program_source_records(codegen)
        launch_specs = specs if specs is not None else [codegen]
        for entry, launch_spec in zip(data["entries"], launch_specs, strict=True):
            paths = [r["source"] for r in records if entry["symbol"] in r["symbols"]]
            if records and any(r["source_sha256"] != entry["source_identity"]["sha256"]
                               for r in records if entry["symbol"] in r["symbols"]):
                _error("entry source digest differs from declared program source.")
            if not records:
                paths = [codegen.get("source", "")]
            if len(paths) != 1 or resolve_artifact_path(root, paths[0]) != Path(entry["source_identity"]["path"]).resolve():
                _error("entry source differs from selected package.")
            _read(entry["source_identity"])
            _read(entry["native_cubin_identity"])
            native = _json(entry["native_metadata_identity"])
            if native.get("hash") != entry["native_key"] or native.get("name") != entry["symbol"]:
                _error("reviewed native metadata selects another entry/key.")
            resources = entry["exact_resources"]
            if launch_spec.get("grid") != resources["grid"] or "launch_recipe" in launch_spec:
                _error("diagnostic resource grid differs from the declared fixed launch.")
            fields = {"physical_num_warps": "num_warps", "dynamic_shared_memory_bytes": "shared",
                      "stack_frame_bytes": "ptxas_stack_frame_bytes", "spill_store_bytes": "ptxas_spill_store_bytes",
                      "spill_load_bytes": "ptxas_spill_load_bytes", "global_scratch_bytes_per_CTA": "global_scratch_size",
                      "global_scratch_alignment": "global_scratch_align", "profile_scratch_bytes_per_CTA": "profile_scratch_size",
                      "profile_scratch_alignment": "profile_scratch_align"}
            if any(type(native.get(field)) is not int or native[field] != resources[key]
                   for key, field in fields.items()):
                _error("reviewed native metadata/resource components disagree.")
            target = native.get("target", {})
            if (target != {"backend": "cuda", "arch": 90, "warp_size": 32}
                    or resources["block_threads"] != resources["physical_num_warps"] * target["warp_size"]):
                _error("reviewed native target/thread geometry differs.")
            if type(native.get("num_ctas")) is not int or native["num_ctas"] <= 0:
                _error("native CTA count must be a positive integer.")
            ctas = resources["grid"][0] * resources["grid"][1] * resources["grid"][2] * native["num_ctas"]
            coverage = trace["config"]["ctas"] if census else trace["config"]["typed_ctas"]
            if coverage != (ctas if census else list(range(ctas))):
                _error("recorder must cover the complete physical CTA domain.")
            pins.extend(entry[field] for field in ("source_identity", "native_cubin_identity", "native_metadata_identity"))
        _verify_provenance(data)
        provenance = data["approval_provenance"]
        pins.extend(data["evidence_refs"])
        pins.extend(provenance[field] for field in ("resource_input_identity", "root_flow_identity", "plan_identity"))
        pins.extend(provenance["approval_refs"])
        return BoundDiagnosticResourceContract(self._payload, str(root), tuple(_canonical(p) for p in pins), _SEAL)


@dataclass(frozen=True)
class BoundDiagnosticResourceContract:
    _payload: str
    artifact: str
    _pins: tuple[str, ...]
    _seal: object

    def verify_artifact(self, artifact, manifest, codegen):
        if self._seal is not _SEAL or self.artifact != str(Path(artifact).resolve()):
            _error("unbound or cross-artifact diagnostic context.")
        data = json.loads(self._payload)
        if _json(data["artifact_identity"]) != manifest or manifest.get("codegen") != codegen:
            _error("runtime artifact/configuration changed after binding.")

    def for_entry(self, symbol):
        if self._seal is not _SEAL:
            _error("unbound diagnostic context.")
        entries = json.loads(self._payload)["entries"]
        matches = [entry for entry in entries if entry["symbol"] == symbol]
        if len(matches) != 1:
            _error("entry not covered by bound context.")
        return BoundDiagnosticEntry(_canonical(matches[0]), self._pins, self._payload, _SEAL)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _same_typed_value(actual, expected, *, types_only=False):
    """Compare JSON leaf types without treating bool/int/float as aliases.

    Container normalization happens in _plain: compiled tuple fields and the
    target dataclass retain their JSON representation. The early type pass
    defers missing fields to the existing assembler/complete-identity checks.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if any(type(key) is not str for key in actual):
            return False
        if not types_only and actual.keys() != expected.keys():
            return False
        return all(_same_typed_value(actual[key], expected[key], types_only=types_only)
                   for key in actual.keys() & expected.keys())
    if isinstance(expected, list):
        return ((types_only or len(actual) == len(expected))
                and all(_same_typed_value(value, reference, types_only=types_only)
                        for value, reference in zip(actual, expected)))
    return types_only or actual == expected


def _require_driver_counter_types(compiled, *, required):
    # CompiledKernel installs these plain integer attributes when run loads
    # its handles. Check already loaded values early, and require both later.
    unloaded = object()
    for field in ("n_regs", "n_spills"):
        value = getattr(compiled, field, unloaded)
        if value is unloaded and not required:
            continue
        if type(value) is not int or value < 0:
            raise RuntimeContractError(f"Diagnostic driver counter {field} requires a nonnegative integer type.")


@dataclass(frozen=True)
class BoundDiagnosticEntry:
    _entry: str
    _pins: tuple[str, ...]
    _payload: str
    _seal: object

    def require_bound(self):
        if self._seal is not _SEAL:
            raise RuntimeContractError("Unbound diagnostic resource entry.")

    def verify_identity(self, compiled):
        """Verify byte identities before reading the driver-loading launcher."""
        self.require_bound()
        for identity in self._pins:
            _read(json.loads(identity))
        entry = json.loads(self._entry)
        if compiled.name != entry["symbol"] or getattr(compiled, "hash", None) != entry["native_key"]:
            raise RuntimeContractError("Diagnostic native entry/key identity differs.")
        cubin = getattr(compiled, "asm", {}).get("cubin")
        if not isinstance(cubin, bytes) or hashlib.sha256(cubin).hexdigest() != entry["native_cubin_identity"]["sha256"]:
            raise RuntimeContractError("Diagnostic actual cubin identity differs.")
        group = getattr(compiled, "metadata_group", {})
        metadata_path = group.get(entry["symbol"] + ".json")
        binary_path = group.get(entry["symbol"] + ".cubin")
        if not metadata_path or not binary_path or sum(key.endswith(".json") for key in group) != 1:
            raise RuntimeContractError("Diagnostic native metadata group is incomplete.")
        for path, field in ((metadata_path, "native_metadata_identity"), (binary_path, "native_cubin_identity")):
            _read({**entry[field], "path": str(Path(path).resolve())})
        expected = _json(entry["native_metadata_identity"])
        expected.setdefault("cluster_dims", [1, 1, 1])
        metadata = compiled.metadata
        actual = metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
        if not _same_typed_value(_plain(actual), expected, types_only=True):
            raise RuntimeContractError("Diagnostic actual native metadata type differs.")
        _require_driver_counter_types(compiled, required=False)
        return entry, expected

    def verify(self, compiled, resources, grid):
        entry, expected = self.verify_identity(compiled)
        _require_driver_counter_types(compiled, required=True)
        metadata = compiled.metadata
        actual = metadata._asdict() if hasattr(metadata, "_asdict") else vars(metadata)
        if not _same_typed_value(_plain(actual), expected):
            raise RuntimeContractError("Diagnostic actual native metadata differs.")
        target = expected.get("target", {})
        if target.get("backend") != "cuda" or target.get("arch") != 90 or target.get("warp_size") != 32:
            raise RuntimeContractError("Diagnostic native target differs.")
        actual_resources = dict(resources, registers_per_thread=compiled.n_regs,
                                physical_num_warps=metadata.num_warps,
                                block_threads=metadata.num_warps * target["warp_size"],
                                dynamic_shared_memory_bytes=metadata.shared, grid=list(grid),
                                global_scratch_bytes_per_CTA=getattr(metadata, "global_scratch_size", 0),
                                global_scratch_alignment=getattr(metadata, "global_scratch_align", 1),
                                profile_scratch_bytes_per_CTA=getattr(metadata, "profile_scratch_size", 0),
                                profile_scratch_alignment=getattr(metadata, "profile_scratch_align", 1))
        if not _same_typed_value(actual_resources, entry["exact_resources"]):
            raise RuntimeContractError("Diagnostic native resource component differs from reviewed footprint.")

    @property
    def policy_report(self):
        self.require_bound()
        entry = json.loads(self._entry)
        return dict(execution_purpose="diagnostic_only", accepted_timing_eligible=False,
                    diagnostic_native_key=entry["native_key"],
                    diagnostic_cubin_sha256=entry["native_cubin_identity"]["sha256"])
