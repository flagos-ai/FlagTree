# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""DFlash GGUF metadata and tensor import plan.

This plan preserves packed storage and explicitly reports the missing native
noncausal decoder/injection lowering. It is not an executable IR artifact.
"""
from pathlib import Path
from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import checkpoint_from_path, GGUFCheckpoint


class DFlashImporter:
    def __init__(self, checkpoint):
        if isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            self.checkpoint = GGUFCheckpoint(path) if path.suffix.lower() == '.gguf' else checkpoint_from_path(path)
        else:
            self.checkpoint = checkpoint
        if self.checkpoint.config.get('model_type') != 'dflash':
            raise ImporterError('DFlash importer requires LLM_ARCH_DFLASH GGUF weights.')

    def import_plan(self):
        c = self.checkpoint
        t, d = c.config['text_config'], c.config['dflash']
        h, f = t['embedding_length'], t['feed_forward_length']
        dim, heads, kv = (t[k] for k in ('attention.key_length', 'attention.head_count', 'attention.head_count_kv'))
        shapes = {'fc.weight': (h, len(d['target_layers'])*h), 'enc.output_norm.weight': (h,),
                  'output_norm.weight': (h,)}
        for layer in range(t['block_count']):
            p = f'blk.{layer}.'
            shapes.update({p+'attn_norm.weight': (h,), p+'attn_q.weight': (heads*dim,h),
                           p+'attn_k.weight': (kv*dim,h), p+'attn_output.weight': (h,heads*dim),
                           p+'attn_q_norm.weight': (dim,), p+'attn_k_norm.weight': (dim,),
                           p+'ffn_norm.weight': (h,), p+'ffn_gate.weight': (f,h),
                           p+'ffn_up.weight': (f,h), p+'ffn_down.weight': (h,f)})
            if p+'attn_v.weight' in c.keys: shapes[p+'attn_v.weight']=(kv*dim,h)
        selector = 'selector_hidden.weight' in c.keys
        if selector:
            if any(d[k] <= 0 for k in ('block_size','conv_kernel_size','conv_group_size','selector_rank','selector_top_k')):
                raise ImporterError('DFlash2 model is missing conv/selector metadata.')
            if h % d['conv_group_size'] or h < d['selector_top_k']*(d['selector_top_k']+1):
                raise ImporterError('DFlash2 hidden size is incompatible with conv groups or selector lattice.')
            shapes['selector_hidden.weight']=(d['selector_rank'],h)
            projected=2*d['conv_kernel_size']*(h//d['conv_group_size'])
            for layer in range(t['block_count']):
                for role in ('attn','ffn'):
                    shapes[f'blk.{layer}.{role}_conv_base']=(2,d['conv_kernel_size'],h)
                    shapes[f'blk.{layer}.{role}_conv_proj.weight']=(projected,h)
            for key in ('selector_predecessor.weight','selector_successor.weight'):
                info=c.tensor_info(key)
                if len(info.shape)!=2 or info.shape[1]!=d['selector_rank']:
                    raise ImporterError(f'DFlash2 invalid selector tensor {key!r}.')
        for key,shape in shapes.items():
            info=c.tensor_info(key)
            if info.shape!=shape:
                raise ImporterError(f'DFlash tensor {key!r} requires shape {shape}, got {info.shape}.')
        return {'schema': 'flagmega.dflash-import-plan/v1', 'model_type': 'dflash',
                'variant': 'dflash2' if selector else 'dflash1', 'target_layers': list(d['target_layers']),
                'config': c.config, 'tensors': [{'key': k, 'shape': list(c.tensor_info(k).shape),
                    'encoding': c.tensor_info(k).quant_type, 'source': c.tensor_info(k).source} for k in c.keys
                    if k in getattr(c, '_raw', c.keys)],
                'executable': False, 'required_target_bindings': ['token_embeddings', 'lm_head', 'target_hidden_features'],
                'decoder_attention': 'noncausal', 'native_lowering': 'unavailable'}

    def import_module(self):
        self.import_plan()  # Validate the real weights before reporting the execution gap.
        raise ImporterError('DFlash GGUF import plan is valid; native decode is unavailable: '
                            'target-feature KV injection and noncausal draft-block attention need lowering.')


def import_dflash(checkpoint):
    return DFlashImporter(checkpoint).import_plan()
