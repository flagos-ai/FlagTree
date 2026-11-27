# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Dense Qwen3.5 text decode with native packed GGUF projection weights."""

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

from triton.flagmega.errors import ImporterError
from triton.flagmega.importer.checkpoint import checkpoint_from_path
from triton.flagmega.importer.qwen3_5 import Qwen35Layer0Importer, Qwen35LayerConfig, _positive_int
from triton.flagmega.importer.source import attach_import_source_locations
from triton.flagmega.ir import DType, F, Module, TupleType, VectorType, dim, effect, tensor_type, verify_module
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
from triton.flagmega.passes.shape_bucket import DECLARATION_SCHEMA


def token_extent(num_tokens):
    """Return ``(extent, range)`` for a static count or a bucketed range.

    A positive integer is a static specialization.  A ``(minimum, maximum)``
    pair becomes the symbolic ``tokens`` dimension declared to the compiler;
    the shape-bucket pass substitutes the selected segment before codegen.
    """

    if isinstance(num_tokens, bool):
        raise ImporterError('Dense Qwen3.5 token extent must be an integer or a (minimum, maximum) range.')
    if isinstance(num_tokens, int):
        if num_tokens <= 0:
            raise ImporterError('Dense Qwen3.5 token extent must be positive.')
        return num_tokens, None
    if (isinstance(num_tokens, (tuple, list)) and len(num_tokens) == 2
            and all(type(value) is int for value in num_tokens)):
        minimum, maximum = num_tokens
        if not 1 <= minimum <= maximum:
            raise ImporterError('Dense Qwen3.5 token range requires 1 <= minimum <= maximum.')
        return dim('tokens', minimum, maximum), (minimum, maximum)
    raise ImporterError('Dense Qwen3.5 token extent must be an integer or a (minimum, maximum) range.')


def rms_norm(value, weight, epsilon, *, name):
    wide = F.tensors.cast(value, 'float32')
    scale = F.tensors.cast(weight, 'float32')
    scale = F.math.add(scale, F.builtin.splat_const(scale.type, 1.0))
    stats = F.nn.norm_stats(wide, axis=-1, use_mean=False)
    zero = F.builtin.splat_const(scale.type, 0.0)
    return F.nn.norm_apply(value, stats, scale, zero, axis=-1, epsilon=epsilon, use_mean=False,
                           round_before_scale=False, output_dtype='bfloat16', name=name)


@dataclass(frozen=True)
class Qwen35DenseConfig(Qwen35LayerConfig):
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    layer_types: tuple[str, ...]

    @property
    def query_size(self):
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_key_value_heads * self.head_dim

    @classmethod
    def parse(cls, source):
        base = Qwen35Layer0Importer._parse_config(source)
        text = source.get('text_config', source)
        heads = _positive_int(text, 'num_attention_heads')
        kv_heads = _positive_int(text, 'num_key_value_heads')
        dim = _positive_int(text, 'head_dim')
        rope = text.get('rope_parameters', {})
        fraction = float(rope.get('partial_rotary_factor', text.get('partial_rotary_factor', 0.25)))
        theta = float(rope.get('rope_theta', text.get('rope_theta', 10000000.0)))
        rotary = dim * fraction
        kinds = tuple(text['layer_types'])
        if len(kinds) != base.num_hidden_layers or any(k not in ('linear_attention', 'full_attention') for k in kinds):
            raise ImporterError('Dense Qwen3.5 layer_types must describe every decoder layer.')
        if heads % kv_heads or not 0 < rotary <= dim or not rotary.is_integer() or int(rotary) % 2:
            raise ImporterError('Invalid dense Qwen3.5 attention heads or rotary dimension.')
        if not math.isfinite(theta) or theta <= 0 or not math.isfinite(base.epsilon) or base.epsilon <= 0:
            raise ImporterError('Dense Qwen3.5 RoPE theta and RMSNorm epsilon must be finite and positive.')
        if text.get('attention_bias', False) or text.get('hidden_act', 'silu') != 'silu':
            raise ImporterError('Dense Qwen3.5 requires bias-free projections and SiLU.')
        return cls(**vars(base), num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=dim,
                   rotary_dim=int(rotary), rope_theta=theta, layer_types=kinds)

    def weight_shapes(self, kind):
        h, i = self.hidden_size, self.intermediate_size
        result = {
            'input_layernorm.weight': (h, ), 'post_attention_layernorm.weight': (h, ), 'mlp.gate_proj.weight': (i, h),
            'mlp.up_proj.weight': (i, h), 'mlp.down_proj.weight': (h, i)
        }
        if kind == 'linear_attention':
            result.update({
                'linear_attn.in_proj_qkv.weight': (self.conv_dim, h), 'linear_attn.in_proj_z.weight':
                (self.value_dim, h), 'linear_attn.in_proj_b.weight': (self.num_value_heads,
                                                                      h), 'linear_attn.in_proj_a.weight':
                (self.num_value_heads, h), 'linear_attn.conv1d.weight': (self.conv_dim, 1, self.conv_kernel_size),
                'linear_attn.A_log': (self.num_value_heads, ), 'linear_attn.dt_bias': (self.num_value_heads, ),
                'linear_attn.norm.weight': (self.value_head_dim, ), 'linear_attn.out_proj.weight': (h, self.value_dim)
            })
        else:
            result.update({
                'self_attn.q_proj.weight': (2 * self.query_size, h), 'self_attn.k_proj.weight': (self.kv_size, h),
                'self_attn.v_proj.weight': (self.kv_size, h), 'self_attn.q_norm.weight': (self.head_dim, ),
                'self_attn.k_norm.weight': (self.head_dim, ), 'self_attn.o_proj.weight': (h, self.query_size)
            })
        return result


class Qwen35DenseImporter:

    def __init__(self, checkpoint, *, revision=None, layers=None, block_size=256, num_blocks=16,
                 num_tokens=1, shape_bucket_segments_count=2, shape_bucket_segment=None,
                 _mtp=False, _mtp_hidden_lanes=1, _expose_hidden=False):
        self.checkpoint = checkpoint_from_path(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
        self.config = Qwen35DenseConfig.parse(self.checkpoint.config)
        quant = self.checkpoint.config.get("quantization_config", {})
        if quant.get("quant_method") != "gguf":
            raise ImporterError("Complete dense Qwen3.5 decode currently requires a native GGUF checkpoint.")
        self.revision = revision
        self.num_tokens, self.token_range = token_extent(num_tokens)
        if (isinstance(shape_bucket_segments_count, bool)
                or not isinstance(shape_bucket_segments_count, int)
                or shape_bucket_segments_count < 1):
            raise ImporterError('Shape-bucket segments_count must be a positive integer.')
        if shape_bucket_segment is not None and (
                isinstance(shape_bucket_segment, bool) or not isinstance(shape_bucket_segment, int)
                or shape_bucket_segment < 0):
            raise ImporterError('Shape-bucket segment must be a non-negative integer.')
        if self.token_range is None and (shape_bucket_segments_count != 2 or shape_bucket_segment is not None):
            raise ImporterError('Shape-bucket options require a (minimum, maximum) token range.')
        if self.token_range is not None and shape_bucket_segments_count < 2:
            raise ImporterError('Shape-bucket segments_count must be at least two when enabled.')
        self.shape_bucket_segments_count = shape_bucket_segments_count
        self.shape_bucket_segment = shape_bucket_segment
        # Reshape keeps one inferred axis while the token extent is symbolic;
        # a static token count is spelled literally so T=1 stays byte-identical.
        self.token_leading = (self.num_tokens if isinstance(self.num_tokens, int) else -1)
        c = self.config
        self.mtp = _mtp
        self.mtp_hidden_lanes = _mtp_hidden_lanes
        self.expose_hidden = _expose_hidden and not _mtp
        if type(_mtp_hidden_lanes) is not int or _mtp_hidden_lanes not in (1, 8) or c.hidden_size % _mtp_hidden_lanes:
            raise ImporterError('MTP hidden input lanes must be one or eight and divide hidden_size.')
        if _mtp:
            from dataclasses import replace
            if self.checkpoint.config.get('gguf', {}).get('architecture') != 'qwen35':
                raise ImporterError('MTP currently requires qwen35 GGUF.')
            if self.checkpoint.config.get('gguf', {}).get('nextn_predict_layers') != 1:
                raise ImporterError('Qwen3.5 MTP requires exactly one nextn prediction layer.')
            self.config = c = replace(c, layer_types=(*c.layer_types, 'full_attention'))
            layers = (c.num_hidden_layers,)
        self.layers = tuple(range(c.num_hidden_layers)) if layers is None else tuple(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(
                type(i) is not int or not 0 <= i < c.num_hidden_layers + int(_mtp) for i in self.layers):
            raise ImporterError('Dense Qwen3.5 requested layer indices are invalid.')
        prefixes = [p for p in ('model.language_model.', 'model.') if p + 'embed_tokens.weight' in self.checkpoint.keys]
        if len(prefixes) != 1:
            raise ImporterError('Dense Qwen3.5 checkpoint requires an unambiguous embedding namespace.')
        self.model_prefix = prefixes[0]
        counts = Counter(c.layer_types[i] for i in self.layers)
        self.gdn_config = GatedDeltaNetStateConfig(max(1,
                                                       counts['linear_attention']), c.num_key_heads, c.num_value_heads,
                                                   c.key_head_dim, c.value_head_dim, c.conv_kernel_size, c.hidden_size)
        self.paged_config = PagedAttentionStateConfig(max(1, counts['full_attention']), c.num_key_value_heads,
                                                      c.head_dim, block_size=block_size, num_blocks=num_blocks)

    def info(self, key, shape):
        info = self.checkpoint.tensor_info(key)
        if info.shape != shape:
            raise ImporterError(f'Tensor {key!r} requires shape {shape}, got {info.shape}.')
        if info.quant_type not in (None, 'IQ4_XS', 'Q4_K'):
            raise ImporterError(f'Unsupported GGUF encoding {info.quant_type!r} for {key!r}.')
        return info

    def import_module(self):
        importer, c = self, self.config
        tokens, leading = self.num_tokens, self.token_leading

        class Graph(Module):

            def weight_at(self, key, shape, *, group=None):
                info = importer.info(key, shape)
                return self.weight(key, info.storage_type, key=key, source=info.source, id='w_' + key.replace('.', '_'),
                                   metadata={} if group is None else {'rdata_group': group})

            def project(self, value, weights, infos, key, *, name, output_dtype=None):
                encoding = infos[key].quant_type
                if encoding is None:
                    return F.math.matmul(value, weights[key], transpose_b=True, output_data_type=output_dtype,
                                         name=name)
                n, k = infos[key].shape
                scale = F.builtin.splat_const(tensor_type('float32', ((n + 127) // 128, (k + 127) // 128)), 1.0)
                return F.math.block_scaled_matmul(value, weights[key], scale, weight_block_n=128, weight_block_k=128,
                                                  weight_encoding=encoding, output_data_type=output_dtype, name=name)

            def linear_attention(self, value, state, layer_id, weights, infos, name):
                p = lambda key, x=value: self.project(x, weights, infos, 'linear_attn.' + key + '.weight', name=name +
                                                      '_' + key)
                view = F.nn.gated_delta_net_state_slice(state, layer_id, name=name + '_state_view')
                conv = F.nn.gated_delta_net_convolution(p('in_proj_qkv'), view, weights['linear_attn.conv1d.weight'],
                                                        conv_kernel_size=c.conv_kernel_size, name=name + '_convolution')
                convolved, updated = F.tensors.get_items(conv, 0, 1, name_prefix=name + '_conv')
                encodings = {
                    part + '_weight_encoding': infos['linear_attn.in_proj_' + part + '.weight'].quant_type
                    for part in ('b', 'a')
                    if infos['linear_attn.in_proj_' + part + '.weight'].quant_type is not None
                }
                recurrent = F.nn.gated_delta_net_recurrent_core(
                    updated, convolved, p('in_proj_z'), value, weights['linear_attn.in_proj_b.weight'],
                    weights['linear_attn.in_proj_a.weight'], weights['linear_attn.A_log'],
                    weights['linear_attn.dt_bias'], weights['linear_attn.norm.weight'], num_key_heads=c.num_key_heads,
                    num_value_heads=c.num_value_heads, key_head_dim=c.key_head_dim, value_head_dim=c.value_head_dim,
                    epsilon=c.epsilon, value_head_order=importer.checkpoint.config.get('gguf', {}).get(
                        'value_head_order', 'grouped'), **encodings, name=name + '_recurrent')
                return p('out_proj', F.tensors.get_item(recurrent, 0)), state

            def full_attention(self, value, state, layer_id, advance, cosine, sine, weights, infos, name):
                p = lambda key: self.project(value, weights, infos, 'self_attn.' + key + '_proj.weight', name=name + '_'
                                             + key)
                # Qwen stores [query, gate] rows interleaved per head. Split the
                # projected activations, leaving the original packed weights intact.
                qg = F.tensors.reshape(p('q'), (leading, c.num_attention_heads, 2, c.head_dim))
                query, gate = (F.tensors.reshape(F.tensors.slice(qg, starts=(i, ), ends=(i + 1, ), axes=(2, )),
                                                 (leading, c.num_attention_heads, c.head_dim)) for i in (0, 1))
                key, val = (F.tensors.reshape(p(part), (leading, c.num_key_value_heads, c.head_dim))
                            for part in ('k', 'v'))
                query = rms_norm(query, weights['self_attn.q_norm.weight'], c.epsilon, name=name + '_q_norm')
                key = rms_norm(key, weights['self_attn.k_norm.weight'], c.epsilon, name=name + '_k_norm')
                query, key = (F.nn.rope(x, cosine, sine, rotary_dim=c.rotary_dim, name=name + '_' + part + '_rope')
                              for part, x in (('q', query), ('k', key)))
                packed_q, packed_k, packed_v = (F.tensors.pack(x, 8, axis=-1) for x in (query, key, val))
                no_advance = F.builtin.scalar_const(tensor_type('bool', ()), False)
                layout = ('seq', 'head', 'dim')
                state = F.nn.update_paged_attention_kv_cache(packed_k, state, layer_id, no_advance, cache_kind='key',
                                                             layout=layout)
                state = F.nn.update_paged_attention_kv_cache(packed_v, state, layer_id, advance, cache_kind='value',
                                                             layout=layout)
                attended = F.nn.paged_attention(packed_q, state, layer_id, scale=c.head_dim**-0.5, layout=layout,
                                                hidden_size=c.query_size)
                gated = F.math.mul(F.tensors.unpack(attended, axis=-1), F.math.sigmoid(gate))
                output = self.project(F.tensors.reshape(gated, (leading, c.query_size)), weights, infos,
                                      'self_attn.o_proj.weight', name=name + '_attention_output')
                return output, state

            def decoder(self, kind, infos, state_type, rotary_type, index):
                name = 'decode_' + ('linear' if kind == 'linear_attention' else 'attention') + '_' + str(index)
                params = []

                def parameter(key, value_type):
                    node = self.input(key, value_type, id=name + '_' + key.replace('.', '_'),
                                      metadata={'function_parameter': name})
                    params.append(node)
                    return node

                hidden = parameter('hidden', tensor_type('bfloat16', (tokens, c.hidden_size)))
                state = parameter('state', state_type)
                layer_id = parameter('layer_id', tensor_type('int32', ()))
                if kind == 'full_attention':
                    advance = parameter('advance_sequence', tensor_type('bool', ()))
                    cosine, sine = (parameter(k, rotary_type) for k in ('rotary_cos', 'rotary_sin'))
                weights = {key: parameter(key, info.storage_type) for key, info in infos.items()}
                normalized = rms_norm(hidden, weights['input_layernorm.weight'], c.epsilon, name=name + '_input_norm')
                if kind == 'linear_attention':
                    attention, updated = self.linear_attention(normalized, state, layer_id, weights, infos, name)
                else:
                    attention, updated = self.full_attention(normalized, state, layer_id, advance, cosine, sine,
                                                             weights, infos, name)
                residual = F.math.add(hidden, attention, name=name + '_attention_residual')
                normalized = rms_norm(residual, weights['post_attention_layernorm.weight'], c.epsilon,
                                      name=name + '_post_norm')
                gate_key, up_key = 'mlp.gate_proj.weight', 'mlp.up_proj.weight'
                if infos[gate_key].quant_type is not None and infos[up_key].quant_type is not None:
                    scale = F.builtin.splat_const(
                        tensor_type('float32', ((c.intermediate_size + 127) // 128, (c.hidden_size + 127) // 128)), 1.0)
                    mlp = F.nn.matmul_glu(normalized, weights[gate_key], weights[up_key], scale, scale,
                                          activation='silu', weight_block_n=128, weight_block_k=128,
                                          gate_weight_encoding=infos[gate_key].quant_type,
                                          up_weight_encoding=infos[up_key].quant_type, name=name + '_mlp_gate_up')
                else:
                    gate = self.project(normalized, weights, infos, gate_key, name=name + '_mlp_gate')
                    up = self.project(normalized, weights, infos, up_key, name=name + '_mlp_up')
                    mlp = F.math.mul(F.math.mul(gate, F.math.sigmoid(gate)), up)
                mlp = self.project(mlp, weights, infos, 'mlp.down_proj.weight', name=name + '_mlp_down')
                output = F.math.add(residual, mlp, name=name + '_output')
                self.function(name, params, (output, updated),
                              attrs={'calling_convention': 'device', 'noinline': True, 'reusable': True})
                return name, tuple(infos)

            def forward(self):
                ids = self.input('input_ids', tensor_type('int32', (tokens, )), id='input_ids')
                entry_gdn = gdn = self.input('gated_delta_net_state', importer.gdn_config.ref_type, id='gdn_state')
                entry_paged = paged = self.input('paged_attention_state', importer.paged_config.ref_type,
                                                 id='paged_state')
                emb_key = importer.model_prefix + 'embed_tokens.weight'
                mtp_prefix = f'{importer.model_prefix}layers.{c.num_hidden_layers}.nextn.'
                if importer.mtp and mtp_prefix + 'embed_tokens.weight' in importer.checkpoint.keys:
                    emb_key = mtp_prefix + 'embed_tokens.weight'
                emb_info = importer.info(emb_key, (c.vocab_size, c.hidden_size))
                embedding = self.weight_at(emb_key, emb_info.shape)
                attrs = {} if emb_info.quant_type is None else {'weight_encoding': emb_info.quant_type}
                hidden = F.nn.embedding(ids, embedding, padding_idx=c.padding_idx, **attrs, name='token_embedding')
                if importer.mtp:
                    lanes = importer.mtp_hidden_lanes
                    h_type = (tensor_type('bfloat16', (tokens, c.hidden_size)) if lanes == 1 else
                              tensor_type(VectorType(DType.BFLOAT16, (lanes,)), (tokens, c.hidden_size // lanes)))
                    h_input = self.input('hidden_input', h_type, id='hidden_input')
                    h_value = h_input if lanes == 1 else F.tensors.unpack(h_input, axis=-1)
                    e_norm = rms_norm(hidden, self.weight_at(mtp_prefix + 'enorm.weight', (c.hidden_size,)),
                                      c.epsilon, name='mtp_enorm')
                    h_norm = rms_norm(h_value, self.weight_at(mtp_prefix + 'hnorm.weight', (c.hidden_size,)),
                                      c.epsilon, name='mtp_hnorm')
                    split_keys = tuple(mtp_prefix + 'eh_' + role + '.weight' for role in ('e', 'h'))
                    if all(key in importer.checkpoint.keys for key in split_keys):
                        parts = []
                        for value, key in zip((e_norm, h_norm), split_keys, strict=True):
                            info = importer.info(key, (c.hidden_size, c.hidden_size))
                            parts.append(self.project(value, {key: self.weight_at(key, info.shape)},
                                                      {key: info}, key, name='mtp_' + key.rsplit('.', 2)[1],
                                                      output_dtype='float32'))
                        hidden = F.tensors.cast(F.math.add(*parts), 'bfloat16', name='mtp_eh_proj')
                    else:
                        concat = F.tensors.concat(e_norm, h_norm, axis=-1, name='mtp_concat')
                        proj_key = mtp_prefix + 'eh_proj.weight'
                        proj_info = importer.info(proj_key, (c.hidden_size, 2 * c.hidden_size))
                        hidden = self.project(concat, {proj_key: self.weight_at(proj_key, proj_info.shape)},
                                              {proj_key: proj_info}, proj_key, name='mtp_eh_proj')
                positions = F.nn.rotary_embedding(hidden, paged, head_dim=c.rotary_dim, theta=c.rope_theta,
                                                  name='position_embeddings')
                cosine, sine = F.tensors.get_items(positions, 0, 1, name_prefix='main_rotary')
                counts = Counter(c.layer_types[i] for i in importer.layers)
                indices, decoders = Counter(), {}
                for layer in importer.layers:
                    kind = c.layer_types[layer]
                    shapes = c.weight_shapes(kind)
                    prefix = f'{importer.model_prefix}layers.{layer}.'
                    infos = {key: importer.info(prefix + key, shape) for key, shape in shapes.items()}
                    signature = (kind,
                                 tuple((key, str(info.storage_type), info.quant_type) for key, info in infos.items()))
                    state = gdn if kind == 'linear_attention' else paged
                    if signature not in decoders:
                        decoders[signature] = self.decoder(kind, infos, state.type, cosine.type, len(decoders))
                    name, keys = decoders[signature]
                    local_layer = indices[kind]
                    indices[kind] += 1
                    weights = {key: self.weight_at(prefix + key, shapes[key]) for key in keys}
                    args = [hidden, state, F.builtin.scalar_const(tensor_type('int32', ()), local_layer)]
                    if kind == 'full_attention':
                        advance = F.builtin.scalar_const(tensor_type('bool', ()), indices[kind] == counts[kind])
                        args.extend((advance, cosine, sine))
                    args.extend(weights[key] for key in keys)
                    call = F.builtin.call(
                        *args, callee=name, result_type=TupleType((hidden.type, state.type)), effect=effect(
                            'read_write',
                            'gated_delta_net_state' if kind == 'linear_attention' else 'paged_attention_kv_cache'),
                        name=f'layer_{layer}_call', metadata={'layer_index': layer, 'state_layer_index': local_layer})
                    hidden, updated = F.tensors.get_items(call, 0, 1, name_prefix=f'layer_{layer}')
                    if kind == 'linear_attention':
                        gdn = updated
                    else:
                        paged = updated
                norm_key = importer.model_prefix + 'norm.weight'
                if importer.mtp and mtp_prefix + 'shared_head_norm.weight' in importer.checkpoint.keys:
                    norm_key = mtp_prefix + 'shared_head_norm.weight'
                norm = self.weight_at(norm_key, (c.hidden_size, ))
                hidden = rms_norm(hidden, norm, c.epsilon, name='final_norm')
                lm_key = ('lm_head.weight' if 'lm_head.weight' in importer.checkpoint.keys else
                          importer.model_prefix + 'embed_tokens.weight')
                if importer.mtp and mtp_prefix + 'shared_head_head.weight' in importer.checkpoint.keys:
                    lm_key = mtp_prefix + 'shared_head_head.weight'
                lm_info = importer.info(lm_key, (c.vocab_size, c.hidden_size))
                head = embedding if lm_key == emb_key else self.weight_at(lm_key, lm_info.shape)
                logits = self.project(hidden, {lm_key: head}, {lm_key: lm_info}, lm_key, name='logits',
                                      output_dtype='float32')
                token = F.nn.greedy_sample(logits, name='next_token')
                if importer.mtp:
                    self.function('main', (ids, h_input, entry_paged), (logits, token, hidden, paged))
                elif importer.expose_hidden:
                    self.function('main', (ids, entry_gdn, entry_paged), (logits, token, hidden, gdn, paged))
                else:
                    self.function('main', (ids, entry_gdn, entry_paged), (logits, token, gdn, paged))

        tokens_per_call = (self.token_range[1] if self.token_range is not None
                           else int(self.num_tokens))
        metadata = {
            'architecture': 'Qwen3_5ForConditionalGeneration', 'model_type': 'qwen3_5', 'mode': 'decode-1',
            'execution_phase': 'decode', 'tokens_per_call': tokens_per_call,
            'scope': 'text-decoder', 'revision': self.revision,
            'model_prefix': self.model_prefix, 'num_hidden_layers': len(self.layers), 'source_num_hidden_layers':
            c.num_hidden_layers, 'imported_layer_indices': self.layers, 'layer_types': tuple(c.layer_types[i]
                                                                                             for i in self.layers),
            'hidden_size': c.hidden_size, 'vocab_size': c.vocab_size, 'numerical_contract': 'nncase'
        }
        if self.mtp:
            metadata.update(scope='mtp-draft', mode='mtp-1', execution_phase='mtp',
                            numerical_contract='nncase', hidden_output='post-shared-head-norm',
                            hidden_input_lanes=self.mtp_hidden_lanes)
        if self.expose_hidden:
            metadata.update(hidden_output='post-final-norm')
        if self.token_range is not None:
            shape_bucket = {
                'schema': DECLARATION_SCHEMA,
                'enable': True,
                'range_info': {'tokens': list(self.token_range)},
                'segments_count': self.shape_bucket_segments_count,
                'fix_var_map': {},
            }
            if self.shape_bucket_segment is not None:
                shape_bucket['segment'] = self.shape_bucket_segment
            metadata.update(shape_bucket=shape_bucket)
            if not self.mtp:
                metadata['mode'] = 'decode-bucket'
        graph = Graph(dialect='high_level', stage='imported', entry='main', metadata=metadata).build()
        return verify_module(
            attach_import_source_locations(graph, architecture=metadata['architecture'], revision=self.revision))
