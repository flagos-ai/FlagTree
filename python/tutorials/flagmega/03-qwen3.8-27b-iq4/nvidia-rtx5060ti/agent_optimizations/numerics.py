"""Explicit residual precision profile for the imported dense text decoder."""
from dataclasses import replace

from triton.flagmega.errors import ImporterError
from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.rules.neutral._utility import make_node

CONTRACT = 'nncase-fp32-residual'


def fp32_residual(module):
    if module.stage != 'imported' or module.metadata.get('numerical_contract') != 'nncase':
        raise ImporterError('Select FP32 residual precision from a fresh nncase import')
    decoders = {name for name in module.function_map if name.startswith('decode_')}
    if not decoders:
        raise ImporterError('The residual profile requires reusable decoder functions')
    parameters = {module.function_map[name].parameters[0] for name in decoders}
    residuals = {name + suffix for name in decoders for suffix in ('_attention_residual', '_output')}
    if any(module.node_map[name].op != 'math.add' for name in residuals):
        raise ImporterError('Unexpected decoder residual topology')
    nodes, mapped = [], {}

    def emit(op, name, inputs, attrs, metadata):
        node = make_node(op, name, inputs, attrs, {**metadata, 'numerical_contract': CONTRACT})
        nodes.append(node)
        return node

    def wide(value, name, metadata):
        if value.type.dtype == DType.FLOAT32:
            return value
        return emit('tensors.cast', name, (value, ), {'dtype': 'float32'}, metadata)

    for source in module.nodes:
        inputs = tuple(mapped[name] for name in source.inputs)
        if source.id in parameters:
            result = replace(source, type=tensor_type('float32', source.type.shape))
        elif source.op == 'builtin.call' and source.attrs.get('callee') in decoders:
            result = replace(source, inputs=tuple(value.id for value in inputs), type=TupleType(
                (inputs[0].type, source.type.fields[1])))
        elif source.id in residuals:
            operands = tuple(
                wide(value, source.id + f'.operand{index}_wide', source.metadata) for index, value in enumerate(inputs))
            result = emit(source.op, source.id, operands, source.attrs, source.metadata)
        elif source.op == 'nn.embedding':
            embedded = emit(source.op, source.id, inputs, source.attrs, source.metadata)
            result = wide(embedded, source.id + '.residual_wide', source.metadata)
        elif source.op == 'nn.rotary_embedding':
            # Only residual storage changes: retain the imported RoPE table dtype.
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, 'output_dtype': source.type.fields[0].dtype.value}, source.metadata)
        elif not source.inputs:
            result = source
        else:
            result = emit(source.op, source.id, inputs, source.attrs, source.metadata)
        if not nodes or nodes[-1] is not result:
            nodes.append(result)
        mapped[source.id] = result
    return verify_module(
        replace(
            module, nodes=tuple(nodes), metadata={
                **module.metadata, 'numerical_contract':
                CONTRACT, 'residual_dtype':
                'float32', 'projection_dtype':
                'bfloat16', 'numerical_profile_note':
                'FP32 residual accumulation; original packed weights and nncase projection/state boundaries'
            }))
