# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Owner-major device weights retain canonical artifacts and closed readers."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
from types import SimpleNamespace

import pytest

from triton.flagmega.codegen.readonly_layout_policy import ReadonlyLayoutPolicy
from triton.flagmega.codegen.triton import describe_tir_package
from triton.flagmega.codegen.triton.renderer import render_triton_package
from triton.flagmega.codegen.triton.nvidia_readonly_layout import (
    plan_readonly_layout,
    validate_layout_data,
)
from triton.flagmega.errors import ArtifactError, CodegenError

from .conftest import _packed_glu_descriptor_pipeline_module


@pytest.fixture(scope="module")
def readonly_glu_module():
    return _packed_glu_descriptor_pipeline_module(
        "tir.dense_matmul_glu.packed_tensor_descriptor_table_paired_smem_pipeline_inline_gemv", output_extent=6144,
        round_activation=False)


def test_owner_layout_emission_preserves_none_and_full_weight_size(readonly_glu_module, tmp_path):
    module = readonly_glu_module
    ordinary = render_triton_package(module, tmp_path / "ordinary")
    explicit_none = render_triton_package(module, tmp_path / "none", readonly_layout_policy=None)
    assert ordinary == explicit_none
    assert (tmp_path / "ordinary/generated_kernels.py").read_bytes() == (tmp_path /
                                                                         "none/generated_kernels.py").read_bytes()
    selected = render_triton_package(module, tmp_path / "selected", readonly_layout_policy=ReadonlyLayoutPolicy())
    validate_layout_data(selected, module_semantic_hash=module.semantic_hash)
    layout = selected["triton_readonly_layout"]
    assert len(layout["device_layout"]["buffers"]) == 2
    assert all(v["nbytes"] == 25165824 for v in layout["device_layout"]["buffers"])
    assert all(v["owners"] == 128 and v["component_nbytes"] == 196608 and v["total_nbytes"] == 25165824
               for v in layout["bindings"])
    descriptors = selected["host_tensor_descriptor_specs"]
    assert descriptors
    for spec in descriptors:
        assert spec["entries"][0]["strides"] == [768, 128, 64, 1]
        entries = spec["entries"]
        assert len(entries) == 128
        assert entries[-1]["offset_bytes"] - entries[0]["offset_bytes"] == 127 * 196608
    assert render_triton_package(module, tmp_path / "again") == ordinary


def test_policy_is_immutable_and_typed():
    policy = ReadonlyLayoutPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.kind = "other"
    for kwargs in ({"kind": "other"}, {"families": ["dense_matmul_glu"]}, {"families": ()}):
        with pytest.raises(CodegenError):
            ReadonlyLayoutPolicy(**kwargs)


def test_unsupported_consumer_and_stale_view_are_rejected(readonly_glu_module):
    module = readonly_glu_module
    calls = describe_tir_package(module)["render_calls"]
    plan = plan_readonly_layout(module, calls, ReadonlyLayoutPolicy())
    call = next(v for v in calls if v["family"] == "dense_matmul_glu")
    binding = next(v for v in call["inputs"] if v["formal"] == "gate_weight")["buffers"][0]
    other = {"family": "elementwise", "inputs": [{"formal": "value", "buffers": [binding]}]}
    with pytest.raises(CodegenError, match="incompatible reader"):
        plan_readonly_layout(module, [*calls, other], ReadonlyLayoutPolicy())
    changed = deepcopy(binding["abi"])
    changed["scalar_storage_strides"] = (changed["scalar_storage_strides"][0] + 128,
                                         *changed["scalar_storage_strides"][1:])
    with pytest.raises(CodegenError, match="canonical binding"):
        plan.views[binding["actual"]].resolve(changed)


@pytest.mark.parametrize("field,value", [("descriptor_kind", "single"), ("paired_weight_fields", False)])
def test_incompatible_glu_channel_contract_rejects_both_weights(readonly_glu_module, field, value):
    calls = deepcopy(describe_tir_package(readonly_glu_module)["render_calls"])
    next(v for v in calls if v["family"] == "dense_matmul_glu")[field] = value
    with pytest.raises(CodegenError, match="paired owner-table"):
        plan_readonly_layout(readonly_glu_module, calls, ReadonlyLayoutPolicy())


def test_manifest_descriptor_and_layout_digest_agree(readonly_glu_module, tmp_path):
    result = render_triton_package(readonly_glu_module, tmp_path, readonly_layout_policy=ReadonlyLayoutPolicy())
    bad = deepcopy(result)
    bad["host_tensor_descriptor_specs"][0]["entries"][0]["offset_bytes"] += 16
    with pytest.raises(ArtifactError, match="inconsistent identities"):
        validate_layout_data(bad)
    bad = deepcopy(result)
    bad["triton_readonly_layout"]["bindings"][0]["total_nbytes"] //= 128
    with pytest.raises(ArtifactError, match="inconsistent identities"):
        validate_layout_data(bad)


def test_incompatible_shared_readonly_pool_fails_before_mapping():
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule
    instance = SimpleNamespace(codegen={"triton_readonly_device_layout": {"schema": "test"}}, manifest={"rdata": {}})
    with pytest.raises(ArtifactError, match="Shared readonly.*device layout"):
        GeneratedTirCallGraphModule._load_rdata(instance, None, "cpu", 0, shared_rdata=object())


@pytest.mark.parametrize("field,value", [("storage_kind", "compact_per_owner"), ("coordinate_space", "local"),
                                         ("scalar_dtype", "float16"), ("owner_count", 1)])
def test_unproved_physical_geometry_is_rejected_without_mutation(readonly_glu_module, field, value):
    calls = deepcopy(describe_tir_package(readonly_glu_module)["render_calls"])
    call = next(v for v in calls if v["family"] == "dense_matmul_glu")
    gate = next(v for v in call["inputs"] if v["formal"] == "gate_weight")["buffers"][0]
    gate["abi"][field] = value
    before = deepcopy(calls)
    with pytest.raises(CodegenError, match="closed full-K contiguous owner"):
        plan_readonly_layout(readonly_glu_module, calls, ReadonlyLayoutPolicy())
    assert calls == before


def test_artifact_roundtrip_and_runtime_fresh_pool_preserve_canonical_bits(readonly_glu_module, tmp_path):
    import torch
    from triton.flagmega.artifacts import load_artifact, write_artifact
    from triton.flagmega.artifacts.module import refresh_artifact_codegen
    from triton.flagmega.runtime.module import GeneratedTirCallGraphModule

    class Checkpoint:

        def load_tensor(self, key, *, device="cpu"):
            assert key in {"gate", "up"}
            # Exercise all BF16 bit patterns, byte ordering, owners and origins.
            bits = torch.arange(6144 * 2048, dtype=torch.int32, device=device)
            return (bits + (127 if key == "up" else 0)).to(torch.int16).view(torch.bfloat16).reshape(6144, 2048)

    module = readonly_glu_module
    roots = {}
    for name, options in (("ordinary", {}), ("none", {"readonly_layout_policy": None}),
                          ("selected", {"readonly_layout_policy": ReadonlyLayoutPolicy()})):
        roots[name] = write_artifact(module, tmp_path / name, target="nvidia-sm90", checkpoint=Checkpoint(),
                                     emit_executable=True, **options)
    assert (roots["ordinary"] / "artifact.json").read_bytes() == (roots["none"] / "artifact.json").read_bytes()
    canonical_image = (roots["ordinary"] / "assets/rdata.bin").read_bytes()
    assert all((root / "assets/rdata.bin").read_bytes() == canonical_image for root in roots.values())
    assert all(
        (root / "assets/rdata.index.json").read_bytes() == (roots["ordinary"] / "assets/rdata.index.json").read_bytes()
        for root in roots.values())
    root = roots["selected"]
    manifest, reloaded = load_artifact(root)
    assert reloaded.semantic_hash == module.semantic_hash
    layout = manifest["codegen"]["triton_readonly_layout"]
    assert any(s["name"] == "codegen.readonly_layout" for s in manifest["sections"])
    canonical = torch.frombuffer(bytearray(canonical_image), dtype=torch.uint8)
    runtime = SimpleNamespace(artifact=root, ir_module=reloaded, manifest=manifest, codegen=manifest["codegen"])
    for _ in range(2):
        actual = GeneratedTirCallGraphModule._load_rdata(runtime, torch, "cpu", canonical.numel())
        expected = canonical.clone()
        for record in layout["device_layout"]["buffers"]:
            before = canonical.narrow(0, record["offset"], record["nbytes"]).view(*record["shape"])
            after = actual.narrow(0, record["offset"], record["nbytes"]).view(record["shape"][1], record["shape"][0],
                                                                              *record["shape"][2:])
            assert torch.equal(after, before.permute(*record["axes"]))
            assert torch.equal(after.permute(1, 0, 2, 3).contiguous(), before)
            expected.narrow(0, record["offset"], record["nbytes"]).copy_(before.permute(*record["axes"]).reshape(-1))
        assert torch.equal(actual, expected)
        assert (root / "assets/rdata.bin").read_bytes() == canonical_image
    with pytest.raises(ArtifactError, match="fresh emission"):
        refresh_artifact_codegen(root)
    # A damaged sealed file is rejected before runtime mapping.
    manifest_file = root / "artifact.json"
    descriptor_file = root / layout["file"]
    saved = descriptor_file.read_bytes()
    descriptor_file.write_bytes(saved + b" ")
    with pytest.raises(ArtifactError):
        load_artifact(root)
    descriptor_file.write_bytes(saved)
    damaged = deepcopy(manifest)
    damaged["codegen"]["host_tensor_descriptor_specs"][0]["entries"][0]["offset_bytes"] += 16
    manifest_file.write_text(json.dumps(damaged))
    with pytest.raises(ArtifactError, match="inconsistent identities"):
        load_artifact(root)
    manifest_file.write_text(json.dumps(manifest))
    assert hashlib.sha256(descriptor_file.read_bytes()).hexdigest() == layout["file_sha256"]
