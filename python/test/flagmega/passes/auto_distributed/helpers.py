# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import MemoryCheckpoint, Qwen3LayerImporter, TensorInfo


def packed_matmul_module(*, size: int = 1024) -> fm.IRModule:
    builder = fm.IRBuilder(dialect="high_level", stage="packed", metadata={"unit": "auto-distributed"})
    value_type = fm.tensor_type("bfloat16", [1, size])
    weight_type = fm.tensor_type(fm.vector_type("float8_e4m3fn", (2, 16)), [size, size // 32])
    scale_type = fm.tensor_type("float32", [size // 128, size // 128])
    value = builder.var("value", value_type, id="value")
    weight = builder.weight("weight", weight_type, source="unit.safetensors", key="weight", id="weight")
    scale = builder.weight("scale", scale_type, source="unit.safetensors", key="scale", id="scale")
    output = builder.call(
        "math.packed_block_scaled_matmul",
        (value, weight, scale),
        value_type,
        id="output",
        attrs={
            "weight_block_n": 128,
            "weight_block_k": 128,
            "k_pack": 2,
            "k_vector": 16,
            "packed_layout": "n_major_k_packed",
        },
    )
    builder.function("main", (value,), (output,))
    return fm.verify_module(builder.build(entry="main"))


def qwen3_packed_module(*, reusable: bool = True) -> fm.IRModule:
    """Build the metadata-only Qwen3 layer fixture consumed by this pass UT."""

    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": 151936,
        "num_hidden_layers": 1,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": True,
        "pad_token_id": None,
    }
    prefix = "model.layers.0."
    shapes = {
        "model.embed_tokens.weight": (151936, 2048),
        prefix + "input_layernorm.weight": (2048,),
        prefix + "self_attn.q_proj.weight": (2048, 2048),
        prefix + "self_attn.k_proj.weight": (1024, 2048),
        prefix + "self_attn.v_proj.weight": (1024, 2048),
        prefix + "self_attn.q_norm.weight": (128,),
        prefix + "self_attn.k_norm.weight": (128,),
        prefix + "self_attn.o_proj.weight": (2048, 2048),
        prefix + "post_attention_layernorm.weight": (2048,),
        prefix + "mlp.gate_proj.weight": (6144, 2048),
        prefix + "mlp.up_proj.weight": (6144, 2048),
        prefix + "mlp.down_proj.weight": (2048, 6144),
        "model.norm.weight": (2048,),
    }
    tensors = {
        key: TensorInfo(key, fm.DType.BFLOAT16, shape, "qwen3-unit.safetensors")
        for key, shape in shapes.items()
    }
    module = Qwen3LayerImporter(
        MemoryCheckpoint(config, tensors),
        block_size=4,
        num_blocks=2,
    ).import_module()
    if reusable:
        module = replace(
            module,
            functions=tuple(
                replace(
                    function,
                    attrs={
                        **dict(function.attrs),
                        "reusable": True,
                        "calling_convention": "device",
                        "noinline": True,
                    },
                )
                for function in module.functions
            ),
        )
    compiler = Compiler()
    for stage in (
        "decompose-gdn",
        "form-qkv-rope-with-cache",
        "propose-vectorization",
        "apply-vectorization",
        "propose-packing",
        "apply-packing",
    ):
        module = compiler.run_stage(module, stage).module
    return module
