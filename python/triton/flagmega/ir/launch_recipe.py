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
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r050-projection-consumer-prefetch-row/v1':
            return ProjectionConsumerPrefetchLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r049-glu-consumer-prefetch-row/v1':
            return ConsumerPrefetchRowLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r048-glu-larger-tile-row/v1':
            return LargerTileStagingRowLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r047-glu-occupancy-row/v1':
            return OccupancyStagingRowLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r046-glu-staging-row/v1':
            return StagingRowLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r038-global-prefetch-row/v1':
            return StaticGlobalPrefetchLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == 'flagmega.r006-glu-transport-row/v1':
            return TransportRowLaunchRecipe.from_data(data)
        if isinstance(data, Mapping) and data.get('schema') == PERSISTENT_ROW_SCHEMA:
            return PersistentRowLaunchRecipe.from_data(data)
        fields = {'schema', 'work_size', 'tile_size', 'program_kind', 'max_ctas', 'ctas_per_sm'}
        if not isinstance(data, Mapping) or set(data) != fields or data.get('schema') != TASK_LAUNCH_SCHEMA:
            raise IRSchemaError('Invalid or unsupported independent-task launch recipe.')
        return cls(data['work_size'], data['tile_size'], data['program_kind'], data['max_ctas'], data['ctas_per_sm'])


PERSISTENT_ROW_SCHEMA = 'flagmega.r006-glu-persistent-row/v1'

# Explicit shape/mode and launch qualifications. The GLU entry preserves R038.
# Entries are (family, N, K, MODE, max_ctas, ctas_per_sm).
STATIC_PREFETCH_QUALIFICATIONS = {
    'glu': ('iq4_q8_glu', 17408, 5120, 1, 2176, 61),
    'mlp_down': ('iq4_q8_projection', 5120, 17408, 0, 1280, 36),
    'out_proj': ('iq4_q8_projection', 5120, 6144, 0, 1280, 36),
    'in_proj_qkv': ('iq4_q8_projection', 10240, 5120, 0, 1280, 36),
    'in_proj_z': ('iq4_q8_projection', 6144, 5120, 0, 768, 22),
    'interleaved_q': ('iq4_q8_interleaved', 12288, 5120, 2, 1536, 43),
    'logits': ('iq4_q8_projection', 248320, 5120, 0, 15520, 432),
    'packed_kv': ('iq4_q8_projection', 1024, 5120, 0, 128, 4),
}


@dataclass(frozen=True)
class StaticGlobalPrefetchLaunchRecipe(TaskLaunchRecipe):
    token_rows: int = 5
    qualification: str = 'glu'

    def __post_init__(self):
        super().__post_init__()
        qualified = STATIC_PREFETCH_QUALIFICATIONS.get(self.qualification)
        if (qualified is None or type(self.token_rows) is not int or not 2 <= self.token_rows <= 5
                or (self.work_size, self.tile_size, self.max_ctas, self.ctas_per_sm)
                != (qualified[1], 1, qualified[4], qualified[5])
                or self.program_kind is not KernelProgramKind.PERSISTENT):
            raise IRSchemaError('Static prefetch requires 2..5 tokens and a per-family qualified mapping.')

    def to_data(self):
        data = {**super().to_data(), 'schema': 'flagmega.r038-global-prefetch-row/v1',
                'identity': 'r038_glu_global_prefetch_v1', 'token_rows': self.token_rows,
                'row_mapping': 'pid + row_iteration * grid', 'prefetch_issue_window': 2,
                'weight_staging': False, 'k_steps_per_output_column': (STATIC_PREFETCH_QUALIFICATIONS[self.qualification][2] // 256 + 15) // 16,
                'token_mapping': 'all token rows reuse each decoded weight packet',
                'partial_recycle': 'all token partial reads -> count128 barrier -> next column'}
        if self.qualification != 'glu':
            data['qualification'] = self.qualification
        return data

    @classmethod
    def from_data(cls, data):
        qualification = data.get('qualification', 'glu')
        qualified = STATIC_PREFETCH_QUALIFICATIONS.get(qualification)
        if qualified is None:
            raise IRSchemaError('Unknown static prefetch family qualification.')
        expected = cls(qualified[1], 1, 'persistent', qualified[4], qualified[5], data.get('token_rows'), qualification)
        if dict(data) != expected.to_data():
            raise IRSchemaError('Invalid R038 global-prefetch launch recipe.')
        return expected

PERSISTENT_ROW_PROTOCOL = {'row_mapping': 'pid + row_iteration * grid',
    'epoch_mapping': '2 * row_iteration + it', 'k_steps_per_row': 2,
    'slot_capacity': 2, 'close_mapping': '2 * cdiv(work_size - pid, grid)',
    'partial_recycle': 'all consumer partial_read -> bar5,count128 -> next row partial_store'}

@dataclass(frozen=True)
class PersistentRowLaunchRecipe(TaskLaunchRecipe):
    token_rows: int = 1

    def __post_init__(self):
        super().__post_init__()
        if type(self.token_rows) is not int or self.token_rows <= 0:
            raise IRSchemaError('Persistent token_rows must be a positive static integer.')
        if (self.work_size not in (17408,5120) or self.tile_size != 1
            or self.program_kind is not KernelProgramKind.PERSISTENT
            or (self.token_rows == 1 and (self.max_ctas != 36 or self.ctas_per_sm != 1))
            or (self.token_rows > 1 and (self.max_ctas != 72 or self.ctas_per_sm != 2))):
            raise IRSchemaError('r006 persistent fixes one output column per task, with cap36/one CTA per SM for ROWS=1 and cap72/two CTAs per SM for ROWS>1.')
    def to_data(self):
        data = {**super().to_data(), 'schema': PERSISTENT_ROW_SCHEMA,
                'identity': 'r006_glu_persistent_v1', **PERSISTENT_ROW_PROTOCOL}
        if self.token_rows > 1:
            data.update(token_rows=self.token_rows,
                        token_mapping='all token rows reuse each decoded weight packet',
                        epoch_mapping='row_iteration', k_steps_per_row=2,
                        commits_per_output_column=1,
                        close_mapping='cdiv(work_size - pid, grid)',
                        queue_payload='column index after bounded L2 prefetch issue',
                        partial_recycle='all token partial reads -> count128 barrier -> next column partial stores')
        return data
    @classmethod
    def from_data(cls, data):
        rows = data.get('token_rows', 1)
        expected = cls(17408, 1, 'persistent', 36 if rows == 1 else 72, 1 if rows == 1 else 2, rows)
        if dict(data) != expected.to_data():
            raise IRSchemaError('Invalid r006 persistent row/epoch/recycle recipe.')
        return expected


@dataclass(frozen=True)
class TransportRowLaunchRecipe(TaskLaunchRecipe):
    capacity: int = 2
    whole_row: bool = False
    def __post_init__(self):
        super().__post_init__()
        if (self.work_size,self.tile_size) not in ((17408,1),(5120,1)) or self.max_ctas not in (544,272,136,68,36,72,108,144,180,216,252,288,324,432,864,1728,2176,3456,4352,8640,8704,17408) or self.ctas_per_sm != (self.max_ctas+35)//36 or self.program_kind is not KernelProgramKind.PERSISTENT:
            raise IRSchemaError('transport requires N17408 row1 and a declared grid')
        if self.capacity not in (1,2,3,4,8,16,17) or type(self.whole_row) is not bool:
            raise IRSchemaError('transport fix capacity/layout mismatch')
    def to_data(self):
        return {**super().to_data(),'schema':'flagmega.r006-glu-transport-row/v1','capacity':self.capacity,'whole_row':self.whole_row}
    @classmethod
    def from_data(cls,data):
        value=cls(data['work_size'],data['tile_size'],data['program_kind'],data['max_ctas'],data['ctas_per_sm'],data['capacity'],data['whole_row'])
        if dict(data)!=value.to_data():raise IRSchemaError('transport mapping keys')
        return value


@dataclass(frozen=True)
class StagingRowLaunchRecipe(TaskLaunchRecipe):
    capacity: int = 2
    transport_kind: str = 'cp_async'
    register_budget: int = 56
    token_rows: int = 2

    def __post_init__(self):
        super().__post_init__()
        if ((self.work_size, self.tile_size) != (17408, 1)
                or self.max_ctas not in (432, 2176)
                or self.ctas_per_sm != (self.max_ctas + 35) // 36
                or self.program_kind is not KernelProgramKind.PERSISTENT
                or type(self.capacity) is not int or self.capacity not in (2, 3, 4)
                or self.transport_kind not in ('tma', 'cp_async')
                or type(self.register_budget) is not int or self.register_budget not in (56, 64, 80, 104)
                or type(self.token_rows) is not int or self.token_rows != 2):
            raise IRSchemaError('R046 staging requires its exact shape, token, ring and register recipe')

    def to_data(self):
        return {**super().to_data(), 'schema': 'flagmega.r046-glu-staging-row/v1',
                'identity': 'r046_glu_staging_v1', 'capacity': self.capacity,
                'transport_kind': self.transport_kind, 'register_budget': self.register_budget,
                'token_rows': self.token_rows, 'payload_bytes_per_bank': 2720,
                'stage_bytes': 5632, 'weight_staging': True}

    @classmethod
    def from_data(cls, data):
        value = cls(data['work_size'], data['tile_size'], data['program_kind'],
                    data['max_ctas'], data['ctas_per_sm'], data['capacity'],
                    data['transport_kind'], data['register_budget'], data['token_rows'])
        if dict(data) != value.to_data():
            raise IRSchemaError('R046 staging mapping keys or payload mismatch')
        return value


@dataclass(frozen=True)
class OccupancyStagingRowLaunchRecipe(TaskLaunchRecipe):
    capacity: int = 2
    transport_kind: str = 'cp_async'
    register_budget: int = 56
    token_rows: int = 2
    cta_warps: int = 8

    def __post_init__(self):
        super().__post_init__()
        if ((self.work_size, self.tile_size) != (17408, 1)
                or self.max_ctas not in (288, 432, 864, 2176)
                or self.ctas_per_sm != (self.max_ctas + 35) // 36
                or self.program_kind is not KernelProgramKind.PERSISTENT
                or type(self.capacity) is not int or self.capacity not in (2, 3, 4)
                or self.transport_kind != 'cp_async'
                or type(self.register_budget) is not int or self.register_budget not in (32, 40, 48, 56)
                or type(self.cta_warps) is not int or self.cta_warps not in (2, 4, 8)
                or type(self.token_rows) is not int or self.token_rows != 2):
            raise IRSchemaError('R047 requires its exact shape, physical CTA, ring and register recipe')

    def to_data(self):
        return {**super().to_data(), 'schema': 'flagmega.r047-glu-occupancy-row/v1',
                'identity': 'r047_glu_occupancy_v1', 'capacity': self.capacity,
                'transport_kind': self.transport_kind, 'register_budget': self.register_budget,
                'token_rows': self.token_rows, 'cta_warps': self.cta_warps,
                'consumer_warps': 4 if self.cta_warps == 8 else self.cta_warps,
                'preferred_shared_carveout_percent': 100,
                'partial_sync': 'mbarrier-count128' if self.cta_warps == 8 else 'named-physical-CTA-count',
                'payload_bytes_per_bank': 2720,
                'stage_bytes': 5632, 'weight_staging': True}

    @classmethod
    def from_data(cls, data):
        value = cls(data['work_size'], data['tile_size'], data['program_kind'],
                    data['max_ctas'], data['ctas_per_sm'], data['capacity'],
                    data['transport_kind'], data['register_budget'], data['token_rows'], data['cta_warps'])
        if dict(data) != value.to_data():
            raise IRSchemaError('R047 mapping keys or payload mismatch')
        return value


@dataclass(frozen=True)
class ConsumerPrefetchRowLaunchRecipe(TaskLaunchRecipe):
    prefetch_distance: int = 2
    register_budget: int = 56
    token_rows: int = 2
    cta_warps: int = 4

    def __post_init__(self):
        super().__post_init__()
        if ((self.work_size, self.tile_size) != (17408, 1)
                or self.max_ctas not in (432, 1088, 2176)
                or self.ctas_per_sm != (self.max_ctas + 35) // 36
                or self.program_kind is not KernelProgramKind.PERSISTENT
                or type(self.prefetch_distance) is not int or self.prefetch_distance not in (1, 2, 4, 8)
                or type(self.register_budget) is not int or self.register_budget not in (40, 48, 56, 64)
                or type(self.cta_warps) is not int or self.cta_warps not in (4, 8)
                or type(self.token_rows) is not int or self.token_rows != 2):
            raise IRSchemaError('R049 requires its exact consumer, prefetch and register recipe')

    def to_data(self):
        return {**super().to_data(), 'schema': 'flagmega.r049-glu-consumer-prefetch-row/v1',
                'identity': 'r049_glu_consumer_prefetch_v1', 'prefetch_distance': self.prefetch_distance,
                'register_budget': self.register_budget, 'token_rows': self.token_rows,
                'cta_warps': self.cta_warps, 'consumer_warps': self.cta_warps, 'producer_warps': 0,
                'preferred_shared_carveout_percent': 100,
                'partial_sync': 'named-physical-CTA-count', 'weight_staging': False}

    @classmethod
    def from_data(cls, data):
        value = cls(data['work_size'], data['tile_size'], data['program_kind'],
                    data['max_ctas'], data['ctas_per_sm'], data['prefetch_distance'],
                    data['register_budget'], data['token_rows'], data['cta_warps'])
        if dict(data) != value.to_data():
            raise IRSchemaError('R049 mapping keys or payload mismatch')
        return value


@dataclass(frozen=True)
class ProjectionConsumerPrefetchLaunchRecipe(TaskLaunchRecipe):
    prefetch_distance: int = 1
    register_budget: int = 64
    token_rows: int = 2
    cta_warps: int = 4
    qualification: str = 'mlp_down'
    columns: int = 1

    def __post_init__(self):
        super().__post_init__()
        q = STATIC_PREFETCH_QUALIFICATIONS.get(self.qualification)
        if (self.qualification not in ('mlp_down', 'in_proj_qkv', 'in_proj_z', 'out_proj')
                or type(self.columns) is not int or self.columns not in (1, 4 if self.qualification=='mlp_down' else 2)
                or (self.work_size, self.tile_size) != (q[1], self.columns)
                or self.max_ctas not in (432, q[1]//8, q[1]//4, q[1]//2,
                                         q[1]//(8*self.columns),q[1]//(4*self.columns),q[1]//(2*self.columns),
                                         288,576,864,1152,1728,2304,q[1]//self.columns)
                or self.ctas_per_sm != (self.max_ctas+35)//36
                or self.max_ctas > self.work_size//self.columns
                or self.program_kind is not KernelProgramKind.PERSISTENT
                or type(self.prefetch_distance) is not int or self.prefetch_distance not in (1,2,4)
                or type(self.register_budget) is not int or self.register_budget not in (40,48,56,64)
                or type(self.cta_warps) is not int or self.cta_warps not in (4,8)
                or type(self.token_rows) is not int or self.token_rows != 2):
            raise IRSchemaError('R050 requires its exact projection consumer recipe')

    def to_data(self):
        return {**super().to_data(), 'schema':'flagmega.r050-projection-consumer-prefetch-row/v1',
                'identity':'r050_projection_consumer_prefetch_v1','qualification':self.qualification,'columns':self.columns,
                'prefetch_distance':self.prefetch_distance,'register_budget':self.register_budget,
                'token_rows':self.token_rows,'cta_warps':self.cta_warps,
                'consumer_warps':self.cta_warps,'producer_warps':0,
                'partial_sync':'named-physical-CTA-count','weight_staging':False}

    @classmethod
    def from_data(cls,data):
        value=cls(data['work_size'],data['tile_size'],data['program_kind'],
                  data['max_ctas'],data['ctas_per_sm'],data['prefetch_distance'],
                  data['register_budget'],data['token_rows'],data['cta_warps'],data['qualification'],data['columns'])
        if dict(data)!=value.to_data():
            raise IRSchemaError('R050 mapping keys or payload mismatch')
        return value


@dataclass(frozen=True)
class LargerTileStagingRowLaunchRecipe(TaskLaunchRecipe):
    capacity: int = 2
    register_budget: int = 56
    token_rows: int = 2
    cta_warps: int = 4
    columns: int = 2

    def __post_init__(self):
        super().__post_init__()
        if ((self.work_size, self.tile_size) != (17408, 1)
                or self.max_ctas not in (144, 432, 2176)
                or self.ctas_per_sm != (self.max_ctas + 35) // 36
                or self.program_kind is not KernelProgramKind.PERSISTENT
                or type(self.capacity) is not int or self.capacity not in (1, 2, 3, 4)
                or type(self.register_budget) is not int or self.register_budget not in (48, 56, 64)
                or type(self.cta_warps) is not int or self.cta_warps not in (2, 4, 8)
                or type(self.columns) is not int or self.columns not in (2, 4, 8, 16)
                or type(self.token_rows) is not int or self.token_rows != 2):
            raise IRSchemaError('R048 requires its exact tile, CTA, ring and register recipe')

    def to_data(self):
        return {**super().to_data(), 'schema': 'flagmega.r048-glu-larger-tile-row/v1',
                'identity': 'r048_glu_larger_tiles_v1', 'capacity': self.capacity,
                'transport_kind': 'cp_async', 'register_budget': self.register_budget,
                'token_rows': self.token_rows, 'cta_warps': self.cta_warps,
                'consumer_warps': self.cta_warps, 'columns': self.columns,
                'preferred_shared_carveout_percent': 100,
                'partial_sync': 'named-physical-CTA-count',
                'payload_bytes_per_bank': self.columns * 2720,
                'stage_bytes': 2 * self.columns * 2720,
                'physical_stage_bytes': 2 * self.columns * 2816,
                'physical_stages': 4 if self.capacity == 3 else self.capacity,
                'weight_staging': True}

    @classmethod
    def from_data(cls, data):
        value = cls(data['work_size'], data['tile_size'], data['program_kind'],
                    data['max_ctas'], data['ctas_per_sm'], data['capacity'],
                    data['register_budget'], data['token_rows'], data['cta_warps'], data['columns'])
        if dict(data) != value.to_data():
            raise IRSchemaError('R048 mapping keys or payload mismatch')
        return value
