# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""CPU public-loader/native-policy tests; device operations are explicit carriers."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton.flagmega.errors import ArtifactError, RuntimeContractError
from triton.flagmega.runtime import loader
from triton.flagmega.runtime import prepared
from triton.flagmega.runtime import module as runtime_module
from triton.flagmega.runtime.module import GeneratedTirCallGraphModule


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value))
    return _pin(path)


def _pin(path):
    payload = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), nbytes=len(payload))


def _fixture(tmp_path, variant="full-ring"):
    root = tmp_path / variant
    root.mkdir()
    semantic = "a" * 64
    config = dict(schema="flagmega.triton-trace/v1", base_semantic_hash=semantic,
                  capacity=4096, coarse_calls=True, detailed_calls=[],
                  typed_ctas=list(range(128)), layer_bindings={"main": {"constant": -1}})
    dictionary = dict(schema="flagmega.trace-sites/v1", base_semantic_hash=semantic,
                      config_sha256=_digest(config), sites=[])
    census = variant == "full-census"
    if census:
        config = dict(schema="flagmega.triton-census/v1", base_semantic_hash=semantic,
                      ctas=128, expected_occurrences=367, service_calls=[],
                      layer_bindings={"main": {"constant": -1}})
        dictionary = dict(schema="flagmega.census-sites/v1", base_semantic_hash=semantic,
                          config_sha256=_digest(config), occurrences=[dict(index=i) for i in range(367)])
    config_pin = _write(root / "trace-config.json", config)
    dictionary_pin = _write(root / "trace-dictionary.json", dictionary)
    source = root / "kernel.py"
    source.write_text("# CPU source identity: " + variant)
    source_pin = _pin(source)
    trace = dict(schema=config["schema"], config=config, dictionary=dictionary,
                 config_sha256=_digest(config), dictionary_sha256=_digest(dictionary),
                 config_file="trace-config.json", dictionary_file="trace-dictionary.json",
                 config_file_sha256=config_pin["sha256"], dictionary_file_sha256=dictionary_pin["sha256"],
                 accepted_timing_eligible=False,
                 native_contract=dict(typed_ctas=list(range(128)), capacity=4096, record_bytes=64,
                                      runtime_stream_filtering=False))
    if census:
        trace["native_contract"] = dict(occurrences=367, ctas=128)
    codegen = dict(kind="tir_call_graph/v1", symbol="entry", source="kernel.py", num_warps=8,
                   grid=[128, 1, 1], dynamic_argument_indices=[], triton_trace=trace,
                   runtime_binding=dict(arguments=[], pools=[], signature=[]))
    sections = [dict(name="diagnostic.trace." + name, path=path.name, kind="diagnostic.trace-" + name + "/v1",
                     sha256=pin["sha256"], nbytes=pin["nbytes"])
                for name, path, pin in [("config", root / "trace-config.json", config_pin),
                                        ("dictionary", root / "trace-dictionary.json", dictionary_pin)]]
    manifest = dict(status="executable", codegen_backend="triton", target="nvidia-sm90",
                    semantic_hash=semantic, codegen=codegen, sections=sections)
    artifact_pin = _write(root / "artifact.json", manifest)
    minimal = variant == "minimal-ring"
    stack, stores, loads = (576, 1228, 684) if minimal else (584, 1240, 696)
    if census:
        stack = stores = loads = 0
    registers, profile = (167, 11776) if census else (168, 3146496)
    key = _digest(variant)
    metadata = dict(name="entry", hash=key, target=dict(backend="cuda", arch=90, warp_size=32),
                    num_warps=12, num_ctas=1, cluster_dims=[1, 1, 1], shared=155900,
                    global_scratch_size=2144, global_scratch_align=4,
                    profile_scratch_size=profile, profile_scratch_align=64,
                    ptxas_stack_frame_bytes=stack, ptxas_spill_store_bytes=stores,
                    ptxas_spill_load_bytes=loads)
    meta_pin = _write(root / "entry.json", metadata)
    cubin = root / "entry.cubin"
    cubin.write_bytes(b"CPU native identity carrier " + variant.encode())
    resources = dict(registers_per_thread=registers, physical_num_warps=12, block_threads=384,
                     dynamic_shared_memory_bytes=155900, grid=[128, 1, 1],
                     stack_frame_bytes=stack, local_memory_bytes=stack,
                     spill_store_bytes=stores, spill_load_bytes=loads, spill_bytes=stores + loads,
                     global_scratch_bytes_per_CTA=2144, global_scratch_alignment=4,
                     profile_scratch_bytes_per_CTA=profile, profile_scratch_alignment=64)
    payload = dict(schema="flagmega.diagnostic-resource-contract/v1", execution_purpose="diagnostic_only",
                   accepted_timing_eligible=False, artifact_identity=artifact_pin,
                   base_semantic_hash=semantic, target="nvidia-sm90", codegen_backend="triton",
                   trace_identity={k: trace[k] for k in ("schema", "config_sha256", "dictionary_sha256",
                                                        "config_file_sha256", "dictionary_file_sha256")},
                   entries=[dict(symbol="entry", source_identity=source_pin, native_key=key,
                                 native_cubin_identity=_pin(cubin), native_metadata_identity=meta_pin,
                                 exact_resources=resources)], evidence_refs=[source_pin])
    _approve(root, payload, variant)
    compiled = SimpleNamespace(name="entry", hash=key, n_regs=registers, n_spills=stack // 4,
                               metadata=SimpleNamespace(**metadata), asm={"cubin": cubin.read_bytes()},
                               metadata_group={"entry.json": str(root / "entry.json"),
                                               "entry.cubin": str(cubin)}, run=lambda *args, **kwargs: None)
    return root, manifest, payload, compiled


def _approve(root, payload, variant):
    technical = {key: value for key, value in payload.items() if key != "approval_provenance"}
    input_pin = _write(root / "reviewed-input.json", {"fixtures": {variant: technical}})
    plan = root / "reviewed-plan.md"
    plan.write_text("Reviewed resource input sha256 " + input_pin["sha256"])
    plan_pin = _pin(plan)
    approval_pin = _write(root / "approval.json", {"body_sha256": plan_pin["sha256"]})
    flow = dict(schema="flagmega.orchestrator-flow-signoff/v1", implementation_authorized=True,
                technical_plan=plan_pin, applicable_own_scope_receipts=[
                    dict(technical_body_sha256=plan_pin["sha256"], receipt=approval_pin)])
    flow_pin = _write(root / "flow.json", flow)
    payload["approval_provenance"] = dict(resource_input_identity=input_pin, resource_input_key=variant,
                                           root_flow_identity=flow_pin, plan_identity=plan_pin,
                                           approval_refs=[approval_pin])


def _public_prepare(monkeypatch, root, manifest, payload, compiled):
    # Only artifact decoding, source import and CUDA allocation are carriers.
    # Public load, context binding, per-entry factory and resource validator are real.
    calls = []
    compilers = compiled if isinstance(compiled, tuple) else (compiled,)
    kernels = tuple(SimpleNamespace(run=lambda *args, _compiled=item, **kwargs: (calls.append(kwargs), _compiled)[1])
                    for item in compilers)
    monkeypatch.setattr(runtime_module, "verify_buffer_plan", lambda ir: SimpleNamespace())
    module = GeneratedTirCallGraphModule(root, manifest, SimpleNamespace(metadata={}),
                                         kernels if len(kernels) > 1 else kernels[0])
    module._test_jit_calls = calls
    module._device = "cpu"
    module._prepared = None
    module.prepare_count = 0
    module._require_loaded = lambda: None
    module._validate_external_arguments = lambda args: None
    module._materialize_tensor_descriptors = lambda args: ()
    module._mark_prepared = lambda: None
    factory = lambda *args: module
    monkeypatch.setattr(loader, "load_artifact", lambda *args, **kwargs: (manifest, None))
    monkeypatch.setattr(loader, "create_tir_runtime", factory)
    monkeypatch.setattr(loader.package_registry, "resolve", lambda *args: SimpleNamespace(factory=factory))
    names = ([spec["symbol"] for spec in manifest["codegen"]["kernels"]]
             if manifest["codegen"].get("kernels") is not None else ["entry"])
    monkeypatch.setattr(loader, "_import_source", lambda *args: SimpleNamespace(**dict(zip(names, kernels, strict=True))))
    monkeypatch.setattr(prepared, "_prepare_global_scratch", lambda *args, **kwargs: None)
    try:
        contract_type = importlib.import_module("triton.flagmega.runtime.diagnostic_resources").DiagnosticResourceContract
    except ModuleNotFoundError as error:
        if error.name != "triton.flagmega.runtime.diagnostic_resources":
            raise
        typed = payload
    else:
        typed = contract_type.from_data(payload)
    result = loader.load(root, diagnostic_resources=typed)
    result._prepare_external(())
    return result


@pytest.mark.parametrize("variant", ["full-ring", "minimal-ring"])
def test_reviewed_diagnostic_uses_public_loader_and_real_validator(tmp_path, monkeypatch, variant):
    root, manifest, payload, compiled = _fixture(tmp_path, variant)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    report = result.resource_report
    assert report["execution_purpose"] == "diagnostic_only"
    assert report["accepted_timing_eligible"] is False
    assert report["spill_bytes"] == payload["entries"][0]["exact_resources"]["spill_bytes"]
    assert result._prepared[0].contract.forbid_spills is True
    assert result._prepared[0].contract.spill_tolerance_bytes == 0
    assert all("diagnostic_resources" not in options for options in result._test_jit_calls)


def _typed(payload):
    from triton.flagmega.runtime.diagnostic_resources import DiagnosticResourceContract
    return DiagnosticResourceContract.from_data(payload)


@pytest.mark.parametrize("field", ["spill_store_bytes", "spill_load_bytes", "stack_frame_bytes", "local_memory_bytes",
                                    "registers_per_thread", "dynamic_shared_memory_bytes",
                                    "global_scratch_bytes_per_CTA", "profile_scratch_bytes_per_CTA"])
def test_each_actual_resource_change_rejects(tmp_path, monkeypatch, field):
    root, manifest, payload, compiled = _fixture(tmp_path)
    if field == "local_memory_bytes":
        compiled.n_spills += 1
    elif field == "registers_per_thread":
        compiled.n_regs += 1
    else:
        key = dict(spill_store_bytes="ptxas_spill_store_bytes", spill_load_bytes="ptxas_spill_load_bytes",
                   stack_frame_bytes="ptxas_stack_frame_bytes", dynamic_shared_memory_bytes="shared",
                   global_scratch_bytes_per_CTA="global_scratch_size",
                   profile_scratch_bytes_per_CTA="profile_scratch_size")[field]
        setattr(compiled.metadata, key, getattr(compiled.metadata, key) + 1)
    with pytest.raises(RuntimeContractError, match="metadata differs|resource component differs"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


def test_compensating_spill_components_do_not_hide_drift(tmp_path, monkeypatch):
    root, manifest, payload, compiled = _fixture(tmp_path)
    compiled.metadata.ptxas_spill_store_bytes += 1
    compiled.metadata.ptxas_spill_load_bytes -= 1
    with pytest.raises(RuntimeContractError, match="metadata differs"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


@pytest.mark.parametrize("change", ["key", "cubin", "metadata_group", "native_target", "entry"])
def test_actual_native_identity_drift_rejects(tmp_path, monkeypatch, change):
    root, manifest, payload, compiled = _fixture(tmp_path)
    if change == "key":
        compiled.hash = "b" * 64
    elif change == "cubin":
        compiled.asm["cubin"] += b"changed"
    elif change == "metadata_group":
        compiled.metadata_group = {}
    elif change == "native_target":
        compiled.metadata.target["arch"] = 89
    else:
        compiled.name = "another_entry"
    with pytest.raises(RuntimeContractError, match="identity differs|metadata group|metadata differs"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


@pytest.mark.parametrize("field", ["artifact_identity", "source_identity", "native_cubin_identity",
                                    "native_metadata_identity", "config", "dictionary", "approval", "evidence"])
def test_changed_file_bytes_reject_before_publication(tmp_path, field):
    root, manifest, payload, compiled = _fixture(tmp_path)
    entry = payload["entries"][0]
    path = (Path(payload[field]["path"]) if field == "artifact_identity" else
            Path(entry[field]["path"]) if field.endswith("identity") else
            root / ("trace-" + field + ".json") if field in ("config", "dictionary") else
            root / "approval.json" if field == "approval" else Path(payload["evidence_refs"][0]["path"]))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ArtifactError, match="changed file identity"):
        _typed(payload).bind(root, manifest)


def test_cross_variant_same_semantics_does_not_bind(tmp_path):
    full_root, full_manifest, full, _ = _fixture(tmp_path, "full-ring")
    minimal_root, minimal_manifest, minimal, _ = _fixture(tmp_path, "minimal-ring")
    assert full["base_semantic_hash"] == minimal["base_semantic_hash"]
    with pytest.raises(ArtifactError, match="selected artifact differs"):
        _typed(full).bind(minimal_root, minimal_manifest)


@pytest.mark.parametrize("change", ["unknown", "schema", "timing", "negative", "nonfinite", "boolean", "missing",
                                    "duplicate", "spill_sum", "missing_provenance", "backend"])
def test_strict_input_rejects_malformed_or_unreviewed_contract(tmp_path, change):
    _, _, payload, _ = _fixture(tmp_path)
    if change == "unknown":
        payload["allow_spills"] = True
    elif change == "schema":
        payload["schema"] += "/unsupported"
    elif change == "timing":
        payload["accepted_timing_eligible"] = True
    elif change in ("negative", "nonfinite", "boolean"):
        payload["entries"][0]["exact_resources"]["local_memory_bytes"] = {
            "negative": -1, "nonfinite": float("nan"), "boolean": True}[change]
    elif change == "missing":
        del payload["entries"][0]["exact_resources"]["stack_frame_bytes"]
    elif change == "duplicate":
        payload["entries"].append(copy.deepcopy(payload["entries"][0]))
    elif change == "spill_sum":
        payload["entries"][0]["exact_resources"]["spill_bytes"] += 1
    elif change == "missing_provenance":
        del payload["approval_provenance"]
    else:
        payload["target"] = "iluvatar-bi-v150"
    with pytest.raises(ArtifactError):
        _typed(payload)


@pytest.mark.parametrize("change", ["partial_entries", "reordered_entries", "ordinary_artifact", "config_hash",
                                    "dictionary_link", "partial_CTA", "unreviewed_tuple", "partial_approvals"])
def test_complete_binding_rejects_partial_or_unlinked_input(tmp_path, change):
    root, manifest, payload, _ = _fixture(tmp_path)
    if change in ("partial_entries", "reordered_entries"):
        manifest["codegen"]["kernels"] = [dict(symbol="entry"), dict(symbol="second")]
        if change == "reordered_entries":
            manifest["codegen"]["kernels"].reverse()
    elif change == "ordinary_artifact":
        del manifest["codegen"]["triton_trace"]
    elif change == "config_hash":
        payload["trace_identity"]["config_sha256"] = "b" * 64
    elif change == "dictionary_link":
        manifest["codegen"]["triton_trace"]["dictionary"]["config_sha256"] = "b" * 64
    elif change == "partial_CTA":
        payload["entries"][0]["exact_resources"]["grid"] = [129, 1, 1]
    elif change == "unreviewed_tuple":
        payload["entries"][0]["exact_resources"]["registers_per_thread"] += 1
    else:
        payload["approval_provenance"]["approval_refs"] = []
    payload["artifact_identity"] = _write(root / "artifact.json", manifest)
    with pytest.raises(ArtifactError):
        _typed(payload).bind(root, manifest)


def test_explicit_input_is_typed_and_cannot_be_a_trace_tag(tmp_path):
    root, manifest, payload, _ = _fixture(tmp_path)
    with pytest.raises(ArtifactError, match="typed DiagnosticResourceContract"):
        loader.load(root, diagnostic_resources=payload)


@pytest.mark.parametrize("capacity", ["shared", "registers", "resident_registers", "warps", "grid"])
def test_bound_context_preserves_physical_checks(tmp_path, monkeypatch, capacity):
    from dataclasses import replace
    root, manifest, payload, compiled = _fixture(tmp_path)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    contract = result._prepared[0].contract
    changes = dict(shared=dict(shared_memory_capacity_bytes=1000), registers=dict(registers_per_thread_limit=32),
                   resident_registers=dict(register_file_capacity_per_sm=1000), warps=dict(compute_num_warps=16),
                   grid=dict(requires_grid_wide_sync=True, cooperative_launch_admission=False, available_sm_count=1))
    with pytest.raises(RuntimeContractError):
        prepared._validate_resources(compiled, replace(contract, **changes[capacity]), (128, 1, 1))


def test_missing_assembler_fields_still_rejects_bound_context(tmp_path, monkeypatch):
    root, manifest, payload, compiled = _fixture(tmp_path)
    del compiled.metadata.ptxas_spill_load_bytes
    with pytest.raises(RuntimeContractError, match="assembler resource metadata"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


def test_ordinary_and_unbound_trace_remain_zero_spill(tmp_path):
    root, manifest, payload, compiled = _fixture(tmp_path)
    contract = runtime_module._resource_contract(manifest["codegen"], "nvidia-sm90")
    with pytest.raises(RuntimeContractError, match="1936|1240 spill-store"):
        prepared._validate_resources(compiled, contract, (128, 1, 1))
    compiled.n_spills = 0
    compiled.metadata.ptxas_spill_store_bytes = compiled.metadata.ptxas_spill_load_bytes = 0
    prepared._validate_resources(compiled, contract, (128, 1, 1))
    compiled.metadata.ptxas_spill_store_bytes = 1
    with pytest.raises(RuntimeContractError, match="forbids spills"):
        prepared._validate_resources(compiled, contract, (128, 1, 1))


def test_input_and_bound_context_are_immutable(tmp_path):
    from dataclasses import FrozenInstanceError
    root, manifest, payload, _ = _fixture(tmp_path)
    typed = _typed(payload)
    payload["entries"].clear()
    bound = typed.bind(root, manifest)
    with pytest.raises(FrozenInstanceError):
        bound.artifact = "other"
    with pytest.raises(FrozenInstanceError):
        typed._payload = "{}"


def test_census_context_binds_zero_and_rejects_new_spill(tmp_path, monkeypatch):
    root, manifest, payload, compiled = _fixture(tmp_path, "full-census")
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    assert result.resource_report["spill_bytes"] == 0
    assert result.resource_report["accepted_timing_eligible"] is False
    compiled.metadata.ptxas_spill_store_bytes = 1
    with pytest.raises(RuntimeContractError):
        prepared._validate_resources(compiled, result._prepared[0].contract, (128, 1, 1))


def test_bound_context_cannot_disable_ordinary_guard_or_enter_non_ptxas_backend(tmp_path, monkeypatch):
    from dataclasses import replace
    root, manifest, payload, compiled = _fixture(tmp_path)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    contract = result._prepared[0].contract
    for change in (dict(forbid_spills=False), dict(spill_tolerance_bytes=1936), dict(reports_ptxas_resource_fields=False)):
        with pytest.raises(RuntimeContractError, match="preserve.*zero-spill"):
            replace(contract, **change)
    with pytest.raises(ArtifactError, match="SM90"):
        runtime_module._resource_contract(manifest["codegen"], "iluvatar-bi-v150",
                                          diagnostic_resources=contract.diagnostic_resources)


def test_unknown_adapter_rejects_before_import_or_device_load(tmp_path, monkeypatch):
    root, manifest, payload, _ = _fixture(tmp_path)
    monkeypatch.setattr(loader, "load_artifact", lambda *args, **kwargs: (manifest, None))
    monkeypatch.setattr(loader.package_registry, "resolve", lambda *args: SimpleNamespace(factory=lambda *args: None))
    monkeypatch.setattr(loader, "_import_source", lambda *args: pytest.fail("must reject before source/device load"))
    with pytest.raises(ArtifactError, match="adapter does not support"):
        loader.load(root, diagnostic_resources=_typed(payload))


def test_multiple_entries_select_their_own_native_contract(tmp_path, monkeypatch):
    root, manifest, payload, first = _fixture(tmp_path)
    second = copy.deepcopy(first)
    second.name = second.metadata.name = "second"
    second.hash = second.metadata.hash = _digest("second")
    second_meta = _write(root / "second.json", vars(second.metadata))
    (root / "second.cubin").write_bytes(b"second CPU cubin carrier")
    second.asm["cubin"] = (root / "second.cubin").read_bytes()
    second.metadata_group = {"second.json": str(root / "second.json"), "second.cubin": str(root / "second.cubin")}
    entry = copy.deepcopy(payload["entries"][0])
    entry.update(symbol="second", native_key=second.hash, native_metadata_identity=second_meta,
                 native_cubin_identity=_pin(root / "second.cubin"))
    payload["entries"].append(entry)
    manifest["codegen"]["kernels"] = [dict(symbol=name, grid=[128, 1, 1], dynamic_argument_indices=[])
                                        for name in ("entry", "second")]
    payload["artifact_identity"] = _write(root / "artifact.json", manifest)
    _approve(root, payload, "two-entries")
    result = _public_prepare(monkeypatch, root, manifest, payload, (first, second))
    assert [item.resource_report["diagnostic_native_key"] for item in result._prepared] == [first.hash, second.hash]
    assert result._prepared[0].contract.diagnostic_resources != result._prepared[1].contract.diagnostic_resources


def test_unbound_or_forged_context_is_not_a_resource_permission(tmp_path):
    from triton.flagmega.runtime.diagnostic_resources import BoundDiagnosticEntry
    with pytest.raises(RuntimeContractError, match="bound typed"):
        prepared.ResourceContract(8, 1, diagnostic_resources={"accepted_timing_eligible": False})
    forged = BoundDiagnosticEntry("{}", (), "{}", object())
    with pytest.raises(RuntimeContractError, match="Unbound"):
        prepared.ResourceContract(8, 1, diagnostic_resources=forged)


@pytest.mark.parametrize("identity", ["plan_identity", "root_flow_identity", "resource_input_identity"])
def test_changed_approval_lineage_bytes_reject(tmp_path, identity):
    root, manifest, payload, _ = _fixture(tmp_path)
    path = Path(payload["approval_provenance"][identity]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ArtifactError, match="changed file identity"):
        _typed(payload).bind(root, manifest)


def test_changed_files_after_binding_are_rechecked_during_prepare(tmp_path, monkeypatch):
    root, manifest, payload, compiled = _fixture(tmp_path)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    path = Path(payload["entries"][0]["source_identity"]["path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ArtifactError, match="changed file identity"):
        prepared._validate_resources(compiled, result._prepared[0].contract, (128, 1, 1))


def test_wrong_actual_native_key_rejects_before_driver_loading(tmp_path):
    root, manifest, payload, compiled = _fixture(tmp_path)
    context = _typed(payload).bind(root, manifest).for_entry("entry")
    contract = runtime_module._resource_contract(manifest["codegen"], "nvidia-sm90", diagnostic_resources=context)

    class Unloaded:
        name, hash = "entry", "b" * 64
        metadata, asm, metadata_group = compiled.metadata, compiled.asm, compiled.metadata_group
        run_reads = 0

        @property
        def run(self):
            self.run_reads += 1
            pytest.fail("wrong native identity must reject before driver loading")

    unloaded = Unloaded()
    with pytest.raises(RuntimeContractError, match="entry/key identity"):
        prepared._validate_resources(unloaded, contract, (128, 1, 1))
    assert unloaded.run_reads == 0


@pytest.mark.parametrize("change", ["manifest", "detached_codegen"])
def test_runtime_configuration_mutation_invalidates_bound_instance(tmp_path, monkeypatch, change):
    root, manifest, payload, compiled = _fixture(tmp_path)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    if change == "detached_codegen":
        result.codegen = copy.deepcopy(result.codegen)
    result.codegen["triton_trace"]["config"]["capacity"] += 1
    with pytest.raises(ArtifactError, match="configuration changed"):
        result._prepare_external(())


@pytest.mark.parametrize("metadata", [[], {"num_warps": 12.0}])
def test_malformed_native_metadata_does_not_coerce_resource_types(tmp_path, metadata):
    root, manifest, payload, compiled = _fixture(tmp_path)
    if isinstance(metadata, dict):
        metadata = {**vars(compiled.metadata), **metadata}
    payload["entries"][0]["native_metadata_identity"] = _write(root / "entry.json", metadata)
    with pytest.raises(ArtifactError, match="JSON identity|resource components"):
        _typed(payload).bind(root, manifest)


@pytest.mark.parametrize("variant", ["full-ring", "minimal-ring"])
@pytest.mark.parametrize("field,value", [("num_warps", 12.0), ("num_ctas", True), ("n_regs", 168.25)])
def test_live_numeric_aliases_reject_through_public_prepare(tmp_path, monkeypatch, variant, field, value):
    root, manifest, payload, compiled = _fixture(tmp_path, variant)
    if field == "n_regs":
        compiled.n_regs = value
    else:
        setattr(compiled.metadata, field, value)
    # Native files, key, cubin and reviewed tuple remain unchanged. Only the
    # live object has drifted; file-metadata validation cannot catch this.
    with pytest.raises(RuntimeContractError, match="type"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


@pytest.mark.parametrize("field,value", [
    ("shared", 155900.0), ("ptxas_spill_store_bytes", 1240.0),
    ("global_scratch_align", 4.0), ("cluster_dims", [True, 1, 1]),
    ("target", {"backend": "cuda", "arch": 90.0, "warp_size": 32}),
    ("n_regs", 168.0), ("n_spills", 146.0), ("n_spills", 146.25),
])
def test_neighboring_live_type_aliases_reject(tmp_path, monkeypatch, field, value):
    root, manifest, payload, compiled = _fixture(tmp_path)
    if field in ("n_regs", "n_spills"):
        setattr(compiled, field, value)
    else:
        setattr(compiled.metadata, field, value)
    with pytest.raises(RuntimeContractError, match="type"):
        _public_prepare(monkeypatch, root, manifest, payload, compiled)


@pytest.mark.parametrize("variant", ["full-ring", "minimal-ring"])
def test_genuine_packed_metadata_and_integer_counters_remain_valid(tmp_path, monkeypatch, variant):
    from collections import namedtuple
    from triton.backends.compiler import GPUTarget
    root, manifest, payload, compiled = _fixture(tmp_path, variant)
    values = vars(compiled.metadata).copy()
    values["target"] = GPUTarget(**values["target"])
    values["cluster_dims"] = tuple(values["cluster_dims"])
    compiled.metadata = namedtuple("NativeMetadata", sorted(values))(**values)
    result = _public_prepare(monkeypatch, root, manifest, payload, compiled)
    assert result.resource_report["diagnostic_native_key"] == compiled.hash
    assert result.resource_report["accepted_timing_eligible"] is False


@pytest.mark.parametrize("bad_field", [None, "n_regs", "n_spills"])
def test_driver_counters_are_checked_when_lazy_load_materializes_them(tmp_path, monkeypatch, bad_field):
    root, manifest, payload, compiled = _fixture(tmp_path)

    class LazyCounterCarrier:
        name, hash = compiled.name, compiled.hash
        metadata, asm, metadata_group = compiled.metadata, compiled.asm, compiled.metadata_group
        run_reads = 0

        @property
        def run(self):
            self.run_reads += 1
            self.n_regs, self.n_spills = compiled.n_regs, compiled.n_spills
            if bad_field is not None:
                setattr(self, bad_field, getattr(self, bad_field) + 0.25)
            return lambda: None

    lazy = LazyCounterCarrier()
    assert not hasattr(lazy, "n_regs") and not hasattr(lazy, "n_spills")
    if bad_field is None:
        result = _public_prepare(monkeypatch, root, manifest, payload, lazy)
        assert result.resource_report["registers_per_thread"] == compiled.n_regs
    else:
        with pytest.raises(RuntimeContractError, match="driver counter.*integer type"):
            _public_prepare(monkeypatch, root, manifest, payload, lazy)
    assert lazy.run_reads == 1


@pytest.mark.parametrize("field", ["spill_bytes", "local_memory_bytes"])
def test_resource_report_numeric_aliases_do_not_compare_equal(tmp_path, field):
    root, manifest, payload, compiled = _fixture(tmp_path)
    context = _typed(payload).bind(root, manifest).for_entry("entry")
    contract = runtime_module._resource_contract(manifest["codegen"], "nvidia-sm90", diagnostic_resources=context)
    resources = prepared._spill_resource_report(compiled, contract)
    resources[field] = float(resources[field])
    with pytest.raises(RuntimeContractError, match="resource component"):
        context.verify(compiled, resources, (128, 1, 1))
