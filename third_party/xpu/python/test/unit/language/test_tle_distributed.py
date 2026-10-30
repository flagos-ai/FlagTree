"""XPU TLE distributed frontend coverage that does not require device codegen."""

import pytest

import triton.experimental.tle.language as tle
from triton._C.libtriton.tle import attr, utils


def test_device_mesh_shape_flatten_and_submesh():
    mesh = tle.device_mesh(tle.MeshConfig(node=[("node_x", 2), ("node_y", 2)], device=4, block=2))
    assert mesh.shape == (2, 2, 4, 2)
    assert mesh.dim_names == ("node_x", "node_y", "device", "block")
    assert mesh.size == 32

    flat = mesh.flatten()
    assert flat.shape == (32, )
    assert flat.dim_names == ("flat", )
    assert flat.physical_ids == mesh.physical_ids

    submesh = mesh[1, :, 2, :]
    assert submesh.shape == (2, 2)
    assert submesh.dim_names == ("node_y", "block")
    assert submesh.physical_ids == (20, 21, 28, 29)
    assert submesh.launch_shape == mesh.shape


def test_device_mesh_rejects_invalid_topology():
    with pytest.raises(TypeError):
        tle.device_mesh({"node": "2"})
    with pytest.raises(ValueError):
        tle.device_mesh({"node": []})
    with pytest.raises(ValueError):
        tle.device_mesh({"node": [("x", 0)]})


def test_sharding_and_sharded_tensor_metadata():
    mesh = tle.device_mesh({"device": 4, "block": 2})
    spec = tle.sharding(mesh, split=[tle.S("device"), tle.B], partial=[tle.P("block")])
    assert spec.axis_state("device") == "S"
    assert spec.axis_state("block") == "P"
    assert spec.broadcast == ()

    tensor = tle.make_sharded_tensor("x_ptr", spec, shape=[8, 16])
    assert tensor.handle == "x_ptr"
    assert tensor.shape == (8, 16)
    assert tensor.sharding == spec

    with pytest.raises(ValueError, match="both split and partial"):
        tle.sharding(mesh, split=["device"], partial=["device"])


def test_tle_attr_compatibility_bindings():
    assert attr.FlagCXTeamKind.from_str("intra_node") == attr.FlagCXTeamKind.Intra
    assert attr.FlagCXTeamKind.from_int(2) == attr.FlagCXTeamKind.World
    assert attr.FlagCXTeamKind.from_int(3) is None
    assert attr.FlagCXCoopKind.from_str("block") == attr.FlagCXCoopKind.Block
    assert attr.SyncScope.from_str("system") == attr.SyncScope.System
    assert attr.MemoryOrder.from_str("acqrel") == attr.MemoryOrder.AcqRel
    assert utils is not None


def test_implicit_local_barrier_is_unsupported():
    with pytest.raises(NotImplementedError):
        tle.distributed_barrier.__wrapped__()


def test_cluster_remote_is_unsupported():
    with pytest.raises(NotImplementedError):
        tle.remote.__wrapped__(None, space="cluster")


def test_device_shard_requires_comm():
    mesh = tle.device_mesh(tle.MeshConfig(device=2))
    with pytest.raises(ValueError, match="device_dptr"):
        tle.shard_id.__wrapped__(mesh, "device")
