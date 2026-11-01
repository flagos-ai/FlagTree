# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Fixed fork/join execution with per-program scratch and resource admission."""

from math import prod

from triton.flagmega.errors import RuntimeContractError
from triton.flagmega.ir.program_order import program_ancestors
from triton.flagmega.runtime.program_resources import maximum_antichain_weight


def verify_joint_resources(specs, prepared, sm_count):
    """Bound each SM resource over the actual dependency/stream frontiers."""
    if program_ancestors(specs) is None:
        raise RuntimeContractError("Joint resource admission requires an explicit dependency DAG.")
    ancestors, tails, footprints = {}, {}, {}
    rounded = lambda value, unit: -(-value // unit) * unit
    for spec, kernel in zip(specs, prepared, strict=True):
        report, contract = kernel.resource_report, kernel.contract
        if contract.warp_size != 32 or report["num_warps"] % 4:
            raise RuntimeContractError("Joint SM90 residency currently requires complete four-warp groups.")
        resident = -(-prod(spec["grid"]) // sm_count) if spec["requires_grid_wide_sync"] else 1
        footprint = (
            rounded(report["registers_per_thread"], 8) * report["num_warps"] * contract.warp_size * resident,
            # CUDA also reserves 1 KiB per resident CTA on Hopper:
            # https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
            (rounded(report["shared_memory_bytes"], 256) + 1024) * resident,
            report["num_warps"] * contract.warp_size * resident,
            resident,
        )
        parents = set(spec["depends_on"])
        previous = tails.get(spec["stream_index"])
        if previous is not None:
            parents.add(previous)
        symbol = spec["symbol"]
        ancestors[symbol] = frozenset(parents).union(*(ancestors[p] for p in parents))
        tails[spec["stream_index"]] = symbol
        footprints[symbol] = footprint
    totals = tuple(maximum_antichain_weight(ancestors, {name: value[i] for name, value in footprints.items()})
                   for i in range(4))
    register_limit = min(p.contract.register_file_capacity_per_sm for p in prepared)
    shared_limit = min(p.contract.shared_memory_capacity_bytes for p in prepared)
    if totals[0] > register_limit or totals[1] > shared_limit or totals[2] > 2048 or totals[3] > 32:
        raise RuntimeContractError(
            "Concurrent kernel programs exceed their joint SM90 residency contract: "
            f"registers={totals[0]}, shared={totals[1]}, threads={totals[2]}."
        )
    return {
        "registers": totals[0],
        "shared_bytes": totals[1],
        "threads": totals[2],
        "blocks": totals[3],
        "streams": len(tails),
    }


class ProgramDAG:
    def __init__(self, specs, prepared, *, device, target):
        import torch

        if target != "nvidia-sm90":
            raise RuntimeContractError("Concurrent kernel programs require the declared SM90 resource model.")
        self.specs, self.prepared, self.device = specs, prepared, device
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) != (9, 0):
            raise RuntimeContractError("Concurrent kernel programs require an SM90 device.")
        self.resource_report = verify_joint_resources(specs, prepared, properties.multi_processor_count)
        from triton.flagmega.runtime.nvidia_function_config import configure_joint_shared_carveout

        with torch.cuda.device(device):
            self.resource_report["shared_carveout_percent"] = configure_joint_shared_carveout(
                prepared,
                self.resource_report["shared_bytes"],
                properties.shared_memory_per_multiprocessor,
            )
        self.streams = tuple(torch.cuda.Stream(device=device) for _ in range(max(s["stream_index"] for s in specs) + 1))
        self.events = {s["symbol"]: torch.cuda.Event(enable_timing=False) for s in specs}
        self.mutable = frozenset(index for s in specs for index in s["external_writes"])
        parents = {parent for s in specs for parent in s["depends_on"]}
        self.sinks = tuple(s["symbol"] for s in specs if s["symbol"] not in parents)
        self.caller_stream = None

    def _verify_external_aliases(self, arguments):
        spans = []
        for index, value in enumerate(arguments):
            if hasattr(value, "data_ptr"):
                begin = value.data_ptr()
                spans.append((index, begin, begin + value.numel() * value.element_size()))
        for position, (left, lo, hi) in enumerate(spans):
            for right, other_lo, other_hi in spans[position + 1 :]:
                if (left in self.mutable or right in self.mutable) and lo < other_hi and other_lo < hi:
                    raise RuntimeContractError(
                        "Concurrent kernel programs cannot bind undeclared writable external aliases."
                    )

    def launch(self, external_arguments, descriptors, *, stream=None):
        import torch

        caller = (
            torch.cuda.current_stream(self.device)
            if stream is None
            else stream
            if isinstance(stream, torch.cuda.Stream)
            else torch.cuda.ExternalStream(stream, device=self.device)
        )
        if self.caller_stream is not None and self.caller_stream != caller.cuda_stream:
            raise RuntimeContractError(
                "A concurrent program instance is bound to its first caller stream; prepare another instance."
            )
        self.caller_stream = caller.cuda_stream
        self._verify_external_aliases(external_arguments)
        for branch in self.streams:
            branch.wait_stream(caller)
        for spec, prepared in zip(self.specs, self.prepared, strict=True):
            branch = self.streams[spec["stream_index"]]
            for parent in spec["depends_on"]:
                branch.wait_event(self.events[parent])
            prepared.launch(*external_arguments, *descriptors, stream=branch.cuda_stream)
            self.events[spec["symbol"]].record(branch)
        for sink in self.sinks:
            caller.wait_event(self.events[sink])
