"""Explicit local election for complete chip-replicated indexer selection."""
from contextlib import contextmanager
from jinja2 import ChoiceLoader, DictLoader
from triton.flagmega.errors import CodegenError

KEY = 'kernels/dsa_indexer/_function.py.jinja'

@contextmanager
def activate():
    from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
    original = TritonTemplateRegistry.render
    def render(self, name, context):
        if name != 'module.py.jinja':
            return original(self, name, context)
        calls = [c for c in context.get('render_calls', ()) if c['family'] == 'dsa_indexer']
        for call in calls:
            if call['variant'] != 'decode' or call['writer_active'] not in ('(shard_y == 0) & (shard_x == 0)', '(shard_coord0 == 0) & (shard_coord1 == 0)', '(shard_coord1 == 0) & (shard_coord2 == 0)'):
                raise CodegenError('Single-owner indexer requires two complete broadcast block axes')
        loader = self.environment.loader
        source = loader.get_source(self.environment, KEY)[0]
        joint = any(c.get('joint_dsa_pool_scores') for c in calls)
        if joint:
            from .joint_dsa_absorbed_pool_scores import helpers, wrapper
            source = helpers(source)
        match = '    {{ call.symbol }}__select({{ call.signature }})'
        if source.count(match) != 1:
            raise CodegenError('Indexer select wrapper changed')
        start = source.index('def {{ call.symbol }}({{ call.signature }}):')
        head = source[:start]
        tail = source[start:]
        original_wrapper = ('def {{ call.symbol }}({{ call.signature }}):\n'
                    '    {{ call.symbol }}__write({{ call.signature }})\n'
                    '{{ entry.stage_barrier() }}\n'
                    '    {{ call.symbol }}__select({{ call.signature }})\n'
                    '{{ entry.stage_barrier() }}\n'
                    '    if {{ call.advance_sequence }}:\n'
                    '        {{ call.symbol }}__advance({{ call.signature }})')
        if tail.count(original_wrapper) != 1:
            raise CodegenError('Complete original write/select/advance wrapper changed')
        local = ('{% if call.dsa_absorbed_overlap | default(none) %}\n'
                 'def {{ call.symbol }}({{ call.signature }}):\n'
                 '    if tl.program_id(0) == 0:\n'
                 '        {{ call.symbol }}__write({{ call.signature }})\n'
                 '        tl.debug_barrier()\n'
                 '        {{ call.symbol }}__select({{ call.signature }})\n'
                 '        if {{ call.advance_sequence }}:\n'
                 '            {{ call.symbol }}__advance({{ call.signature }})\n'
                 '    else:\n'
                 '        {{ call.dsa_absorbed_overlap.symbol }}({{ call.dsa_absorbed_overlap.arguments }})\n'
                 '{% else %}\n' + original_wrapper + '\n{% endif %}')
        if joint:
            local = wrapper(local)
        selected_source = head + tail.replace(original_wrapper, local)
        self.environment.loader = ChoiceLoader([DictLoader({KEY: selected_source}), loader])
        try:
            return original(self, name, context)
        finally:
            self.environment.loader = loader
    TritonTemplateRegistry.render = render
    try:
        yield
    finally:
        TritonTemplateRegistry.render = original
