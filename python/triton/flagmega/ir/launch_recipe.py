# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Runtime launch recipes for independent tasks over one chip-local value."""
from collections.abc import Mapping
from dataclasses import dataclass

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir.execution_mesh import KernelProgramKind

TASK_LAUNCH_SCHEMA = 'flagmega.independent-linear-tiles/v1'


@dataclass(frozen=True)
class TaskLaunchRecipe:
    work_size: int
    tile_size: int
    program_kind: KernelProgramKind
    max_ctas: int | None = None
    ctas_per_sm: int = 1

    def __post_init__(self):
        for name in ('work_size', 'tile_size', 'ctas_per_sm'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise IRSchemaError(f'Task launch {name} must be a positive integer.')
        if self.max_ctas is not None and (type(self.max_ctas) is not int or self.max_ctas <= 0):
            raise IRSchemaError('Task launch max_ctas must be a positive integer.')
        try:
            object.__setattr__(self, 'program_kind', KernelProgramKind(self.program_kind))
        except (ValueError, TypeError) as exc:
            raise IRSchemaError('Unknown task launch program kind.') from exc
        if self.program_kind is KernelProgramKind.ORDINARY and self.max_ctas is not None:
            raise IRSchemaError('Ordinary linear tiles launch one CTA per task; a CTA cap requires persistent mapping.')

    @property
    def task_count(self):
        return (self.work_size + self.tile_size - 1) // self.tile_size

    @property
    def grid_upper_bound(self):
        return min(self.task_count, self.max_ctas or self.task_count), 1, 1

    def resolve(self, *, sm_count: int, max_ctas: int | None = None):
        """No device reads or CPU synchronization; SM count is host metadata.

        Independent persistent tasks have no all-grid residency requirement.
        SM count is a scheduling choice, not a progress proof for algorithms
        with cross-CTA waits; those require a different explicit recipe.
        """
        if type(sm_count) is not int or sm_count <= 0:
            raise IRSchemaError('Task launch requires a positive runtime SM count.')
        if max_ctas is not None and (type(max_ctas) is not int or max_ctas <= 0):
            raise IRSchemaError('Runtime CTA cap must be a positive integer.')
        if self.program_kind is KernelProgramKind.ORDINARY:
            if max_ctas is not None:
                raise IRSchemaError('Ordinary mapping cannot change its one-CTA-per-task grid.')
            count = self.task_count
        else:
            count = min(self.task_count, sm_count * self.ctas_per_sm,
                        self.max_ctas or self.task_count, max_ctas or self.task_count)
        return count, 1, 1

    def to_data(self):
        return {'schema': TASK_LAUNCH_SCHEMA, 'work_size': self.work_size, 'tile_size': self.tile_size,
                'program_kind': self.program_kind.value, 'max_ctas': self.max_ctas, 'ctas_per_sm': self.ctas_per_sm}

    @classmethod
    def from_data(cls, data):
        fields = {'schema', 'work_size', 'tile_size', 'program_kind', 'max_ctas', 'ctas_per_sm'}
        if not isinstance(data, Mapping) or set(data) != fields or data.get('schema') != TASK_LAUNCH_SCHEMA:
            raise IRSchemaError('Invalid or unsupported independent-task launch recipe.')
        return cls(data['work_size'], data['tile_size'], data['program_kind'], data['max_ctas'], data['ctas_per_sm'])
