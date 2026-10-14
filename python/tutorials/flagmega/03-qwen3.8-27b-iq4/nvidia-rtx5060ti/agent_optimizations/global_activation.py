"""Bind declared, invocation-scoped global Q8 scratch to each CTA consumer."""
from triton.flagmega.codegen.triton import kernel_call_renderers as renderers


class GlobalActivationRenderer:

    def __init__(self, original):
        self.original = original

    def __call__(self, raw):
        call = self.original(raw)
        storage = raw.get('facts', {}).get('activation_storage')
        if storage not in ('cta_global', 'cta_global_interleaved'):
            return call
        k = call['k']
        formats = (
            ('global_qword', 'q8_words', k // 4),
            ('global_qscale', 'q8_scales', k // 32),
            ('global_residual_word', 'q8_residual_words', k // 4),
            ('global_residual_scale', 'q8_residual_scales', k // 32),
        )
        if storage == 'cta_global_interleaved':
            formats = formats[:2]
        for field, formal, width in formats:
            binding = renderers._buffer(raw, 'workspaces', formal)
            shape = renderers._static_shape(binding['abi'], 'logical_shape')
            if len(shape) != 2 or shape[1] != width:
                raise ValueError(f'Q8 global workspace {formal} has incorrect geometry: {shape}')
            call[field] = f'({renderers._pointer(binding)} + tl.program_id(0) * {width})'
        return call


def configure():
    for family in ('block_fp8', 'matmul_glu'):
        original = renderers._FAMILY_ENCODERS[family]
        if not isinstance(original, GlobalActivationRenderer):
            renderers._FAMILY_ENCODERS[family] = GlobalActivationRenderer(original)
