"""RDNA IQ4 kernel replacements for the Qwen3.8-27B decode tutorial.

``install()`` puts this package's templates in front of FlagMega's for the
current process, the same mechanism as the NVIDIA tutorial's
``agent_optimizations``. With ``q8_activations`` the native IQ4_XS GEMVs
quantize each call's activation once per CTA to q8_1 (round-to-nearest-even
int8, one fp16 scale per 32 values; the NVIDIA tutorial's ``q8_1_rne``
contract and llama.cpp's MMVQ activations) in LDS and dot it with
``v_dot4_i32_iu8``. Q4_K and every other projection keep the default path.

The IQ4_XS gate+up GEMVs also take their 8-row tiles from a
dynamic queue over the whole output (one template counter), so blocks that
stream slower DRAM pages take fewer tiles. Every row is still dotted once by
one tile in the same order, so results are bitwise; ``FLAGMEGA_AMD_TILE_QUEUE=0``
disables it.
"""

import os
from math import prod
from pathlib import Path
from types import SimpleNamespace

from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

ROOT = Path(__file__).parent / 'templates'


class RdnaIQ4Registry(TritonTemplateRegistry):

    def __init__(self):
        super().__init__()
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(ROOT)), self.environment.loader])

    def resolve(self, spec):
        for path in self.candidates(spec):
            if (ROOT / path).is_file():
                return path
        return super().resolve(spec)

    def counter_slots(self, platform, architecture):
        if platform != 'amd' or architecture not in ('gfx1201', 'gfx1100'):
            return 0
        return 0 if os.environ.get('FLAGMEGA_AMD_TILE_QUEUE', '1') == '0' else 1

    def counter_families(self, platform, architecture):
        return frozenset({'matmul_glu'})

    def render(self, template_name, context):
        prefetch = os.environ.get('FLAGMEGA_AMD_Q8_PREFETCH_KB')
        if prefetch is not None:
            context = {**context, 'q8_weight_prefetch_kb': int(prefetch)}
        plans = _tile_queue_plans(context)
        if plans:
            context = {**context, 'native_gguf_tile_queue': plans, 'glu_dot_rows': QUEUE_ROWS}
        return super().render(template_name, context)


QUEUE_ROWS = 8
# Claims beyond an even share of tiles, so a block on fast pages can take more.
QUEUE_SPARE_STEPS = 7
_CLAMP = SimpleNamespace(minimum=min, maximum=max)


def _row_progression(call, extent_y, extent_x):
    """Global-row forms of a GLU call's owner-local row expressions, or None.

    Evaluates the call's destination and weight offsets for every active row of
    every block. The queue applies when the destinations are exactly one
    arithmetic progression ``d0 + g * ds`` (``g < N``, each row owned once) and
    the weight offsets are affine in the same ``g``, whatever the owner layout.
    """
    try:
        active = compile(str(call['active_n']), '<active_n>', 'eval')
        weight = compile(str(call['weight_row_offset']), '<weight_row_offset>', 'eval')
        destination = compile(str(call['destination_offset']), '<destination_offset>', 'eval')
        capacity = int(call['local_n_capacity'])
        rows = {}
        for shard_y in range(extent_y):
            for shard_x in range(extent_x):
                names = {'tl': _CLAMP, 'shard_x': shard_x, 'shard_y': shard_y}
                count = eval(active, {'__builtins__': {}}, names)  # noqa: S307 - renderer-built text
                if not isinstance(count, int) or not 0 <= count <= capacity:
                    return None
                for local in range(count):
                    names['local_n_offsets'] = local
                    row = eval(destination, {'__builtins__': {}}, names)  # noqa: S307
                    if row in rows:
                        return None
                    rows[row] = eval(weight, {'__builtins__': {}}, names)  # noqa: S307
    except Exception:  # noqa: BLE001 - any other name or form means another layout
        return None
    if len(rows) < 2:
        return None
    first, total = min(rows), len(rows)
    step, remainder = divmod(max(rows) - first, total - 1)
    if remainder or step <= 0:
        return None
    weight_first = rows[first]
    weight_step = rows.get(first + step, weight_first) - weight_first
    if any(rows.get(first + row * step) != weight_first + row * weight_step for row in range(total)):
        return None
    return total, f'({first} + queue_rows * {step})', f'({weight_first} + queue_rows * {weight_step})'


def _tile_queue_plans(context):
    """Per-call tile-queue plans for the IQ4_XS gate+up GEMVs, all or none."""
    if not context.get('counter_slots'):
        return {}
    calls = [call for call in context.get('render_calls', ())
             if call.get('family') == 'matmul_glu' and call.get('weight_encoding') == 'IQ4_XS']
    hierarchy = tuple(int(value) for value in context.get('mesh_hierarchy', ()))
    programs = prod(int(value) for value in context['grid'])
    # The queue's first tile is shard_index = shard_y * extent_x + shard_x.
    if not calls or len(hierarchy) != 2 or ''.join(context.get('mesh_levels', 'bb')) != 'bb' or prod(hierarchy) != programs:
        return {}
    plans = {}
    for call in calls:
        progression = _row_progression(call, *hierarchy)
        if progression is None:
            return {}
        total_rows, destination, weight = progression
        tiles = -(-total_rows // QUEUE_ROWS)
        plans[str(call['symbol'])] = {
            'counter': 0, 'rows': QUEUE_ROWS, 'programs': programs, 'tiles': tiles,
            'total_rows': total_rows, 'steps': -(-tiles // programs) + QUEUE_SPARE_STEPS,
            'weight_row_offset': weight, 'destination_offset': destination,
        }
    # One counter serves every call; each invocation makes programs * (steps - 1)
    # claims and the last claimer resets it.
    steps = max(plan['steps'] for plan in plans.values())
    for plan in plans.values():
        plan['steps'] = steps
        plan['claims'] = programs * (steps - 1)
    return plans


def install(*, q8_activations=True):
    if q8_activations:
        tir_package.TritonTemplateRegistry = RdnaIQ4Registry
