# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Target-declared expansion of synchronization into known leaf operations."""
from collections.abc import Mapping
from dataclasses import dataclass
from triton.flagmega.errors import IRVerificationError

PROTOCOL_METADATA = "memory_synchronization_protocol"
PROTOCOL_SCHEMA = "flagmega.memory-synchronization-protocol/v1"


@dataclass(frozen=True)
class MemorySynchronizationProtocol:
    identity: str
    grid_cta_barriers: int
    device_grid_barriers: int = 2

    def __post_init__(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise IRVerificationError("Synchronization protocol requires an identity.")
        for value in (self.grid_cta_barriers, self.device_grid_barriers):
            if type(value) is not int or value < 0:
                raise IRVerificationError("Synchronization protocol barrier counts must be nonnegative integers.")

    def to_data(self):
        return dict(schema=PROTOCOL_SCHEMA, identity=self.identity,
                    grid_cta_barriers=self.grid_cta_barriers, device_grid_barriers=self.device_grid_barriers)

    @classmethod
    def from_data(cls, data):
        if not isinstance(data, Mapping) or set(data) != {
            "schema", "identity", "grid_cta_barriers", "device_grid_barriers"
        } or data["schema"] != PROTOCOL_SCHEMA:
            raise IRVerificationError("Malformed synchronization protocol contract.")
        return cls(data["identity"], data["grid_cta_barriers"], data["device_grid_barriers"])


def module_synchronization_protocol(module):
    data = module.metadata.get(PROTOCOL_METADATA)
    return None if data is None else MemorySynchronizationProtocol.from_data(data)
