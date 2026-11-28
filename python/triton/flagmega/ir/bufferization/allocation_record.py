# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Serializable allocator provenance, independent of the solving backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AllocationRecord:
    function: str
    memory_space: str
    allocator: str
    status: str
    # Timing belongs in diagnostics, never in an executable semantic hash.
    objectives: tuple[tuple[str, str, int, float | None], ...] = ()

    def to_data(self):
        return {
            "function":
            self.function, "memory_space":
            self.memory_space, "allocator":
            self.allocator, "status":
            self.status, "objectives": [{"name": name, "status": status, "value": value, "best_bound": bound}
                                        for name, status, value, bound in self.objectives]
        }

    @classmethod
    def from_data(cls, value):
        return cls(
            str(value["function"]), str(value["memory_space"]), str(value["allocator"]), str(value["status"]),
            tuple((str(item["name"]), str(item["status"]), int(item["value"]),
                   None if item["best_bound"] is None else float(item["best_bound"]))
                  for item in value.get("objectives", ())))
