"""Build verified prefill/decode artifacts in one shared readonly address space."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from triton.flagmega import ir as fm
from triton.flagmega.artifacts import load_artifact, verify_rdata, write_artifact
from triton.flagmega.artifacts.readonly_layout import relocate_readonly
from triton.flagmega.runtime.shared_readonly import readonly_fingerprints


def build_bundle(phases, destination, *, checkpoint, rdata_cache_dir=None):
    """Keep each parameter bank contiguous and share equal units across phases.

    Matching uses verified numeric bytes or source byte-address programs,
    including complete owner layouts. Each phase keeps distinct allocations
    for its own equal-valued units, so its artifact remains nonoverlapping.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    prepared, available, end = [], defaultdict(list), 0
    for name, path in phases.items():
        path = Path(path)
        manifest, module = load_artifact(path, verify_rdata_payload=False)
        imported = fm.load_module(path.parent / "input.py", expected_stage="imported")
        if imported.metadata.get("numerical_contract") != module.metadata.get("numerical_contract"):
            raise ValueError("Compiled phase and imported reference have different numerical contracts")
        index = verify_rdata(path / "assets", module=module, segment=0, reuse_source_proofs=True)
        if index.get("chip_count") != 1:
            raise ValueError("This TP1 request bundle requires one readonly chip segment")
        plan = fm.verify_buffer_plan(module)
        fingerprints = readonly_fingerprints(path / "assets", index, module)
        groups = defaultdict(list)
        for entry in index["entries"]:
            buffer = plan.buffer_map[entry["buffer"]]
            groups[buffer.rdata_group or buffer.id].append(entry)
        mapping, used, new_units = {}, set(), []
        for group in sorted(groups.values(), key=lambda rows: min(e["offset"] for e in rows)):
            group.sort(key=lambda e: e["offset"])
            begin = group[0]["offset"]
            size = max(e["offset"] + e["nbytes"] for e in group) - begin
            signature = [(e["offset"] - begin, e["nbytes"], fingerprints[e["buffer"]]) for e in group]
            digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            offset = next((v for v in available[digest] if v not in used), None)
            if offset is None:
                alignment = max(plan.buffer_map[e["buffer"]].alignment for e in group)
                offset = (end + alignment - 1) // alignment * alignment
                end = offset + size
                new_units.append((digest, offset))
            used.add(offset)
            mapping.update((e["buffer"], offset + e["offset"] - begin) for e in group)
        for digest, offset in new_units:
            available[digest].append(offset)
        prepared.append((name, manifest["target"], module, mapping, imported))
        print(name, "readonly union bytes", end, flush=True)
    if end >= 76 * 1024**3:
        raise ValueError("The compiled phase union leaves insufficient H800 space for model state and scratch")
    records, references = {}, {}
    for name, target, module, mapping, imported in prepared:
        relocated = relocate_readonly(module, mapping, nbytes=end)
        artifact = write_artifact(relocated, destination / name, target=target, checkpoint=checkpoint,
                                  emit_executable=True, rdata_cache_dir=rdata_cache_dir)
        records[name] = str(artifact.relative_to(destination))
        fm.emit_module(imported, artifact / "imported.py")
        references[name] = imported.semantic_hash
        print(name, "artifact complete", flush=True)
    (destination / "bundle.json").write_text(json.dumps({
        "schema": "flagmega.qwen35-phase-bundle/v1", "rdata_nbytes": end, "phases": records,
        "imported_semantic_hashes": references,
    }, indent=2) + "\n")
    return destination
