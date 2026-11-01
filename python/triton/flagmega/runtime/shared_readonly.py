# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Content-proved readonly sharing between separately compiled model phases."""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from triton.flagmega.artifacts.source_layout import SOURCE_LAYOUT_DIGEST, component_layouts
from triton.flagmega.errors import ArtifactError
from triton.flagmega.ir import verify_buffer_plan


def readonly_fingerprints(root, index, module, *, chip=0):
    """Identity of each entry's bytes, independent of its destination offset.

    Numeric entries are streamed. Virtual entries include the audited source
    bytes and their complete byte-address program and owner layout. Checkpoint
    key, shape, or dtype alone is insufficient to prove shared storage.
    The caller must verify the artifact payload before using these identities.
    """
    root = Path(root)
    buffers = verify_buffer_plan(module).buffer_map
    programs = {p["buffer"]: p for p in index.get("programs", ())}
    result = {}
    with (root / index["image"]).open("rb") as stream:
        for entry in index["entries"]:
            identity = entry["buffer"]
            if identity in programs:
                program = {k: v for k, v in programs[identity].items() if k not in {"buffer", "ranges"}}
                program["operands"] = [
                    {**o, "source": index["sources"][o["source"]]["sha256"]}
                    for o in program["operands"]]
                program["components"] = [
                    (offset - entry["offset"], shape, maps)
                    for offset, shape, maps in component_layouts(buffers[identity], chip)]
                digest = hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
            else:
                hasher = hashlib.sha256()
                layouts = [(offset - entry["offset"], shape)
                           for offset, shape, _ in component_layouts(buffers[identity], chip)]
                hasher.update(json.dumps(layouts, sort_keys=True).encode())
                for relative, shape in layouts:
                    stream.seek(chip * index["segment_nbytes"] + entry["offset"] + relative)
                    remaining = prod(shape)
                    while remaining:
                        block = stream.read(min(remaining, 8 * 1024 * 1024))
                        if not block:
                            raise ArtifactError("Readonly entry extends beyond its verified image.")
                        hasher.update(block)
                        remaining -= len(block)
                digest = hasher.hexdigest()
            result[identity] = (entry["dtype"], entry["shape"], digest)
    return result


class SharedReadonlyPool:
    """One device allocation whose occupied intervals have immutable proofs.

    Phase artifacts must use a common relocated address space. Conflicting or
    partially overlapping claims fail before any new bytes are copied. Binding
    another phase loads only its new entries, preserving existing weights.
    """

    def __init__(self, nbytes, *, device):
        if type(nbytes) is not int or nbytes <= 0:
            raise ArtifactError("A shared readonly pool requires a positive byte extent.")
        import torch
        self.nbytes = nbytes
        self.device = torch.device(device)
        self.tensor = None
        self._regions = {}

    def _bind_verified(self, root, index, module, *, chip, device, expected_nbytes):
        import torch
        from triton.flagmega.runtime.source_rdata import load_source_rdata
        if (index.get("digest_kind") != SOURCE_LAYOUT_DIGEST
                or expected_nbytes != self.nbytes or index["segment_nbytes"] != self.nbytes
                or torch.device(device) != self.device):
            raise ArtifactError("Shared readonly binding requires a matching source-layout extent and device.")
        fingerprints = readonly_fingerprints(root, index, module, chip=chip)
        buffers = verify_buffer_plan(module).buffer_map
        claims, fresh = {}, []
        for entry in index["entries"]:
            span = buffers[entry["buffer"]].physical_access_span
            region = (span.offset, span.nbytes)
            proof = fingerprints[entry["buffer"]]
            for (start, size), prior in (*self._regions.items(), *claims.items()):
                if region[0] < start + size and start < region[0] + region[1]:
                    if region != (start, size) or proof != prior:
                        raise ArtifactError("Shared readonly intervals overlap without identical byte proofs.")
            claims[region] = proof
            if region not in self._regions:
                fresh.append(entry["buffer"])
        if self.tensor is None:
            self.tensor = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.device)
        load_source_rdata(root, index, module, chip=chip, device=device,
                          pool=self.tensor, buffer_ids=tuple(fresh))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._regions.update(claims)
        return self.tensor


__all__ = ["SharedReadonlyPool"]
