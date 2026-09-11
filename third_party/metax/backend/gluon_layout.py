"""MetaX Gluon whole-module layout candidate protocol.

This module owns the boundary between the C++ candidate builder, Triton's
compilation cache, and standalone TTGIR compilation.  Keeping the protocol out
of the general MetaX backend makes candidate identity and file ownership
auditable without coupling unrelated compilation stages to layout autotuning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from triton._C.libtriton import ir, metax, passes


_LOGGER = logging.getLogger(__name__)

DOMAIN_ATTR = "ttg.gluon.layout-domain-digest"
VARIANT_ATTR = "ttg.gluon.layout-variant-digest"
GVM_FINALIZED_ATTR = "ttg.gluon.gvm-finalized"
RUNTIME_CONTRACT_ATTR = "ttg.gluon.layout-runtime-contract"
CANDIDATE_MANIFEST_ATTR = "ttg.gluon.layout-candidate-manifest"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"", "0", "false", "off", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def has_variant_identity(metadata: Dict[str, Any]) -> bool:
    """Return whether downstream compilation belongs to a layout candidate."""
    return metadata.get("gluon_layout_variant_digest") is not None


def should_enable_downstream_debug(metadata: Dict[str, Any]) -> bool:
    """Gate expensive MLIR reproducers for independently compiled variants."""
    return (
        not has_variant_identity(metadata)
        or env_flag("TRITON_METAX_GLUON_LAYOUT_REPRODUCER", False)
    )


def needs_generic_tensor_select_combine(src) -> bool:
    """Keep the ordinary combine only for modules outside final Gluon closure."""
    return src.get_int_attr(GVM_FINALIZED_ATTR) != 1


def get_single_kernel_name(src: str) -> str:
    """Return the single kernel proved by a Gluon runtime-effect contract."""
    if not src:
        raise RuntimeError(
            "cannot discover a Gluon candidate kernel from empty LLVM IR"
        )
    names = [
        match.group(1)
        for match in re.finditer(
            r"^\s*define\s+metaxgpu_kernel\s+void\s+@([^\s(]+)\(",
            src,
            flags=re.MULTILINE,
        )
    ]
    if len(names) != 1:
        raise RuntimeError(
            "a Gluon layout candidate must contain exactly one launchable "
            "MetaX kernel"
        )
    return names[0]


def _get_module_str_attr(module, name: str):
    """Read a module attribute through the existing generic operation API."""
    get_operation = getattr(module, "get_operation", None)
    if not callable(get_operation):
        return None
    operation = get_operation()
    get_str_attr = getattr(operation, "get_str_attr", None)
    return get_str_attr(name) if callable(get_str_attr) else None


def publish_export(
    output_dir: str,
    source_manifest: Dict[str, Any],
    variants: Iterable[tuple[str, bytes, str]],
) -> Dict[str, Any]:
    """Atomically publish a closed v5 domain of finalized TTGIR modules."""
    fallback = source_manifest["fallback"]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_variants = []
    temporary_paths = []
    final_paths = []
    try:
        for index, (digest, content, stage) in enumerate(variants):
            if index == 0 and digest != fallback:
                raise RuntimeError(
                    "Gluon layout finalization lost its fallback ordering"
                )
            if stage != "final-ttgir":
                raise RuntimeError(
                    "the C++ layout bundle must contain finalized TTGIR only"
                )
            if not is_sha256(digest):
                raise RuntimeError(
                    "Gluon layout finalization produced an invalid digest"
                )
            filename = f"{digest}.ttgir"
            final_path = directory / filename
            temporary_path = directory / f"{filename}.tmp"
            temporary_path.write_bytes(content)
            temporary_paths.append(temporary_path)
            final_paths.append(final_path)
            manifest_variants.append(
                {
                    "digest": digest,
                    "mlir_file": filename,
                    "mlir_sha256": hashlib.sha256(content).hexdigest(),
                    "stage": stage,
                }
            )
            _LOGGER.debug(
                "published temporary Gluon TTGIR digest=%s bytes=%d",
                digest,
                len(content),
            )
            del content
        if not manifest_variants:
            raise RuntimeError("Gluon layout finalization retained no variants")
        manifest = {
            "version": 5,
            "digest": source_manifest["digest"],
            "fallback": fallback,
            "fallback_only": len(manifest_variants) == 1,
            "variants": manifest_variants,
        }
        manifest_tmp = directory / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, separators=(",", ":"), ensure_ascii=True),
            encoding="utf-8",
        )
        temporary_paths.append(manifest_tmp)
        for temporary_path, final_path in zip(
            temporary_paths[:-1], final_paths
        ):
            os.replace(temporary_path, final_path)
        os.replace(manifest_tmp, directory / "manifest.json")
        _LOGGER.debug(
            "committed Gluon layout export domain=%s variants=%d fallback=%s",
            manifest["digest"],
            len(manifest_variants),
            fallback,
        )
        return manifest
    except Exception:
        for path in temporary_paths + final_paths + [
            directory / "manifest.json"
        ]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def read_export(
    output_dir: str,
    runtime_contract: Dict[str, Any],
    consume_variant: Optional[
        Callable[[Dict[str, Any], bytes], None]
    ] = None,
) -> Dict[str, Any]:
    """Validate a complete v5 export before exposing any candidate bytes."""
    directory = Path(output_dir)
    validated_runtime_contract = parse_runtime_contract(
        json.dumps(runtime_contract, separators=(",", ":"))
    )
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "cannot read a valid Gluon layout export manifest from "
            f"'{manifest_path}'"
        ) from exc

    expected_manifest_keys = {
        "version",
        "digest",
        "fallback",
        "fallback_only",
        "variants",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_keys
    ):
        raise RuntimeError(
            "the Gluon layout export manifest does not use the exact v5 schema"
        )
    domain_digest = manifest["digest"]
    fallback_digest = manifest["fallback"]
    fallback_only = manifest["fallback_only"]
    variants = manifest["variants"]
    if (
        manifest["version"] != 5
        or not is_sha256(domain_digest)
        or not is_sha256(fallback_digest)
        or not isinstance(fallback_only, bool)
        or not isinstance(variants, list)
        or not variants
    ):
        raise RuntimeError(
            "the Gluon layout export manifest has invalid v5 identities"
        )

    expected_variant_keys = {
        "digest",
        "mlir_file",
        "mlir_sha256",
        "stage",
    }
    seen_digests = set()
    seen_files = set()
    validated_variants = []
    for variant in variants:
        if (
            not isinstance(variant, dict)
            or set(variant) != expected_variant_keys
        ):
            raise RuntimeError(
                "a Gluon layout export variant does not use the exact v5 schema"
            )
        digest = variant["digest"]
        filename = variant["mlir_file"]
        content_digest = variant["mlir_sha256"]
        stage = variant["stage"]
        expected_filename = f"{digest}.ttgir"
        if (
            not is_sha256(digest)
            or not is_sha256(content_digest)
            or not isinstance(filename, str)
            or filename != expected_filename
            or Path(filename).name != filename
            or digest in seen_digests
            or filename in seen_files
            or stage != "final-ttgir"
        ):
            raise RuntimeError(
                "a Gluon layout export variant has an invalid identity "
                "or filename"
            )
        variant_path = directory / filename
        try:
            content = variant_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"cannot read exported Gluon layout TTGIR '{filename}'"
            ) from exc
        if hashlib.sha256(content).hexdigest() != content_digest:
            raise RuntimeError(
                f"exported Gluon layout TTGIR '{filename}' does not match "
                "its manifest SHA256"
            )
        seen_digests.add(digest)
        seen_files.add(filename)
        validated_variants.append(dict(variant))
        del content

    if (
        fallback_digest != variants[0]["digest"]
        or fallback_digest not in seen_digests
    ):
        raise RuntimeError(
            "the first Gluon layout export variant must be the unique fallback"
        )
    if fallback_only != (len(variants) == 1):
        raise RuntimeError(
            "the Gluon layout export fallback_only flag is inconsistent"
        )
    try:
        exported_entries = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise RuntimeError(
            f"cannot enumerate Gluon layout export directory '{directory}'"
        ) from exc
    expected_entries = {"manifest.json", *seen_files}
    if exported_entries != expected_entries:
        raise RuntimeError(
            "the Gluon layout export directory contains files outside "
            "its v5 manifest"
        )

    # Consumers run only after the full manifest and every payload have passed
    # validation.  Re-read one file at a time to keep this transaction bounded.
    if consume_variant is not None:
        for variant in validated_variants:
            content = (directory / variant["mlir_file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != variant["mlir_sha256"]:
                raise RuntimeError(
                    "exported Gluon layout TTGIR changed after validation"
                )
            consume_variant(variant, content)
            del content
    _LOGGER.debug(
        "validated Gluon layout export domain=%s variants=%d",
        domain_digest,
        len(validated_variants),
    )
    return {
        "manifest": dict(manifest),
        "runtime_contract": validated_runtime_contract,
        "variants": validated_variants,
    }


def cache_export(
    output_dir: str,
    compilation_hash: str,
    runtime_contract: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist validated variant bytes without exposing the temporary path."""
    from triton.runtime.cache import get_cache_manager

    cache = get_cache_manager(compilation_hash)

    def cache_variant(variant, content):
        cache.put(content, variant["mlir_file"], binary=True)

    exported = read_export(output_dir, runtime_contract, cache_variant)
    _LOGGER.debug(
        "cached Gluon layout export compilation=%s domain=%s",
        compilation_hash,
        exported["manifest"]["digest"],
    )
    return exported


def parse_runtime_contract(value: str) -> Dict[str, Any]:
    try:
        contract = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "the Gluon layout runtime contract is not valid JSON"
        ) from exc
    expected_keys = {
        "version",
        "argument_index_space",
        "replayable",
        "deterministic",
        "tensor_args",
        "read_args",
        "written_args",
        "write_only_args",
        "atomic_args",
        "required_noalias",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise RuntimeError(
            "the Gluon layout runtime contract has an invalid schema"
        )
    if (
        contract["version"] != 2
        or contract["argument_index_space"]
        != "lowered-tt-func-runtime-abi"
        or not isinstance(contract["replayable"], bool)
        or not isinstance(contract["deterministic"], bool)
    ):
        raise RuntimeError(
            "the Gluon layout runtime contract has an invalid identity"
        )

    index_names = (
        "tensor_args",
        "read_args",
        "written_args",
        "write_only_args",
        "atomic_args",
    )
    for name in index_names:
        values = contract[name]
        if (
            not isinstance(values, list)
            or any(type(index) is not int or index < 0 for index in values)
            or values != sorted(set(values))
        ):
            raise RuntimeError(
                f"the Gluon layout runtime contract has invalid {name}"
            )
    tensor_args = set(contract["tensor_args"])
    if any(
        not set(contract[name]).issubset(tensor_args)
        for name in (
            "read_args",
            "written_args",
            "write_only_args",
            "atomic_args",
        )
    ):
        raise RuntimeError(
            "the Gluon layout runtime contract has inconsistent effects"
        )
    read_args = set(contract["read_args"])
    written_args = set(contract["written_args"])
    if (
        set(contract["write_only_args"]) != written_args - read_args
        or not set(contract["atomic_args"]).issubset(read_args & written_args)
        or (
            contract["replayable"]
            and contract["deterministic"] != (not contract["atomic_args"])
        )
    ):
        raise RuntimeError(
            "the Gluon layout runtime contract has inconsistent effects"
        )

    noalias = contract["required_noalias"]
    if not isinstance(noalias, list):
        raise RuntimeError(
            "the Gluon layout runtime contract has invalid no-alias pairs"
        )
    canonical_pairs = []
    for pair in noalias:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(index) is not int or index < 0 for index in pair)
            or pair[0] >= pair[1]
            or any(index not in tensor_args for index in pair)
        ):
            raise RuntimeError(
                "the Gluon layout runtime contract has invalid no-alias pairs"
            )
        canonical_pairs.append(tuple(pair))
    if canonical_pairs != sorted(set(canonical_pairs)):
        raise RuntimeError(
            "the Gluon layout runtime contract has invalid no-alias pairs"
        )
    return contract


def record_variant_identity(src, metadata) -> None:
    """Read and validate the identity embedded in standalone candidate TTGIR."""
    recorded_keys = {
        "gluon_layout_domain_digest",
        "gluon_layout_variant_digest",
        "gluon_layout_runtime_contract",
    }
    present_keys = recorded_keys.intersection(metadata)
    if present_keys:
        if present_keys != recorded_keys:
            raise RuntimeError(
                "Gluon layout metadata must carry the complete "
                "candidate identity"
            )
        if (
            not is_sha256(metadata["gluon_layout_domain_digest"])
            or not is_sha256(metadata["gluon_layout_variant_digest"])
        ):
            raise RuntimeError("Gluon layout metadata has an invalid identity")
        metadata["gluon_layout_runtime_contract"] = parse_runtime_contract(
            json.dumps(
                metadata["gluon_layout_runtime_contract"],
                separators=(",", ":"),
            )
        )
        return

    domain_digest = _get_module_str_attr(src, DOMAIN_ATTR)
    variant_digest = _get_module_str_attr(src, VARIANT_ATTR)
    runtime_contract = _get_module_str_attr(src, RUNTIME_CONTRACT_ATTR)
    if (domain_digest is None) != (variant_digest is None):
        raise RuntimeError(
            "an independent Gluon layout TTGIR must carry both domain "
            "and variant identities"
        )
    if domain_digest is None:
        if runtime_contract is not None:
            metadata["gluon_layout_runtime_contract"] = (
                parse_runtime_contract(runtime_contract)
            )
        return
    if not is_sha256(domain_digest) or not is_sha256(variant_digest):
        raise RuntimeError(
            "an independent Gluon layout TTGIR has an invalid identity"
        )
    if runtime_contract is None:
        raise RuntimeError(
            "an independent Gluon layout TTGIR has no runtime-effect contract"
        )
    metadata["gluon_layout_domain_digest"] = domain_digest
    metadata["gluon_layout_variant_digest"] = variant_digest
    metadata["gluon_layout_runtime_contract"] = parse_runtime_contract(
        runtime_contract
    )


def finalize_standalone_ttgir(mod, metadata):
    """Reject candidate sources that bypassed the C++ finalization pipeline."""
    variant = _get_module_str_attr(mod, VARIANT_ATTR)
    if variant is None:
        variant = metadata.get("gluon_layout_variant_digest")
    if variant is not None and mod.get_int_attr(GVM_FINALIZED_ATTR) != 1:
        raise RuntimeError(
            "Gluon layout candidates must be finalized by the C++ "
            f"CandidateBundle builder: {variant}"
        )
    return mod


def build_candidate_bundle(src, metadata, options, capability):
    """Build, validate, export, and cache one closed whole-module domain."""
    scenarios = {
        item for item in options.scenario.split(";") if item
    }
    legacy = {
        "transformGluonLayoutFinalize",
        "directGluonLayoutFinalize",
        "manualGluonLayouts",
    }
    if legacy & scenarios:
        raise RuntimeError(
            "legacy Gluon layout route selection was removed; C500 now "
            "builds one closed candidate bundle automatically"
        )
    if os.getenv("TRITON_METAX_GLUON_MANUAL_LAYOUTS"):
        raise RuntimeError(
            "TRITON_METAX_GLUON_MANUAL_LAYOUTS is no longer supported; "
            "source-authored layouts use the same C++ CandidateBundle "
            "finalization as inferred layouts"
        )
    if src.get_int_attr("ttg.gluon.manual-layouts"):
        raise RuntimeError(
            "ttg.gluon.manual-layouts is a removed pipeline bypass; "
            "source-authored layouts must use CandidateBundle"
        )

    context = src.context
    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.gluon.add_inliner(pm)
    pm.run(src, "gluon_inline")

    bundle = metax.build_gluon_layout_candidate_bundle(src, capability)
    expected_bundle_keys = {
        "version",
        "digest",
        "fallback",
        "fallback_only",
        "runtime_contract",
        "variants",
    }
    if not isinstance(bundle, dict) or set(bundle) != expected_bundle_keys:
        raise RuntimeError("C++ returned an invalid Gluon layout bundle")
    variants = bundle["variants"]
    if (
        bundle["version"] != 1
        or not is_sha256(bundle["digest"])
        or not is_sha256(bundle["fallback"])
        or not isinstance(bundle["fallback_only"], bool)
        or not isinstance(variants, list)
        or not variants
        or bundle["fallback_only"] != (len(variants) == 1)
    ):
        raise RuntimeError("C++ returned invalid Gluon layout identities")
    for variant in variants:
        if (
            not isinstance(variant, dict)
            or set(variant) != {"digest", "source"}
            or not is_sha256(variant["digest"])
            or not isinstance(variant["source"], str)
        ):
            raise RuntimeError(
                "C++ returned a malformed finalized layout variant"
            )
    if variants[0]["digest"] != bundle["fallback"]:
        raise RuntimeError("C++ layout bundle lost fallback ordering")

    runtime_contract = parse_runtime_contract(bundle["runtime_contract"])
    source_manifest = {
        "version": 1,
        "digest": bundle["digest"],
        "fallback": bundle["fallback"],
        "fallback_only": bundle["fallback_only"],
        "variants": [
            {"digest": variant["digest"]} for variant in variants
        ],
    }
    with tempfile.TemporaryDirectory(
        prefix="triton-gluon-layout-final-"
    ) as output_dir:
        publish_export(
            output_dir,
            source_manifest,
            (
                (
                    variant["digest"],
                    variant["source"].encode("utf-8"),
                    "final-ttgir",
                )
                for variant in variants
            ),
        )
        fallback_path = Path(output_dir) / f"{variants[0]['digest']}.ttgir"
        mod = ir.parse_mlir_module(str(fallback_path), context)
        mod.context = context
        exported = cache_export(
            output_dir, metadata["hash"], runtime_contract
        )
    if exported["runtime_contract"] != runtime_contract:
        raise RuntimeError(
            "finalized layout candidates changed their runtime contract"
        )

    exported_manifest = exported["manifest"]
    manifest_text = json.dumps(
        exported_manifest, separators=(",", ":"), ensure_ascii=True
    )
    mod.set_attr(
        CANDIDATE_MANIFEST_ATTR,
        ir.builder(context).get_string_attr(manifest_text),
    )
    metadata["gluon_layout_manifest"] = exported_manifest
    metadata["gluon_layout_runtime_contract"] = runtime_contract
    metadata["gluon_layout_domain_digest"] = bundle["digest"]
    metadata["gluon_layout_variant_digest"] = bundle["fallback"]
    metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
    _LOGGER.debug(
        "built Gluon layout bundle domain=%s variants=%d fallback=%s",
        bundle["digest"],
        len(variants),
        bundle["fallback"],
    )
    return mod
