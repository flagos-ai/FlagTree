"""Value contract propagation at the normal U4 distribution boundary."""
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType
from triton.flagmega.ir.distributed_inference import all_broadcast, tensor_of
from .u4_ops import Q8_CONTRACT_KEYS, _common, q8_tuple_contract, verify_q8_provenance
from .u4_norm_q8_c1_c2_regions import CONTRACT_ID


def q8_reshard_metadata(module, source, target_type, op):
    """Derive a complete Q8 contract from real nodes, never a metadata claim."""
    if module.metadata.get("u4_input_norm_q8_c1c2") != CONTRACT_ID:
        return {}
    if "u4_q8_contract" not in source.metadata and source.op != "u4.q8_encode":
        return {}
    if (op != "distributed.sharded_view"
            or not isinstance(source.type, DistributedType)
            or not isinstance(target_type, DistributedType)
            or not all_broadcast(source.type) or not all_broadcast(target_type)
            or source.type.partial is not None or target_type.partial is not None
            or source.type.exclusive is not None or target_type.exclusive is not None
            or tensor_of(source.type) != tensor_of(target_type)):
        raise IRSchemaError("U4 Q8 boundary must preserve the complete immutable byte value.")
    root, seen = source, set()
    while root.op.startswith("distributed."):
        if root.id in seen or len(root.inputs) != 1:
            raise IRSchemaError("U4 Q8 boundary requires one acyclic actual root.")
        seen.add(root.id)
        root = module.node_map[root.inputs[0]]
    if root.op == "builtin.get_item" and len(root.inputs) == 1:
        producer = module.node_map[root.inputs[0]]
        contract = q8_tuple_contract(producer, root.attrs.get("index"))
    elif root.op == "u4.q8_encode":
        contract = _common({key: root.attrs[key] for key in Q8_CONTRACT_KEYS})
    else:
        raise IRSchemaError("U4 Q8 boundary has no registered actual producer.")
    # This verifies every intermediate node and the actual producer's full
    # inputs and source epoch. A copied/forged tag cannot authorize propagation.
    verify_q8_provenance(module, source, contract)
    return {"u4_q8_contract": contract, "u4_opaque": True, "canonical_global": True}
