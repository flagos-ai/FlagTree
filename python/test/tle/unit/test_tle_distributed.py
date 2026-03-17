# flagtree tle
import pytest

import triton.experimental.tle.language as tle
from triton.experimental.tle.language import (
    _infer_submesh_barrier_group,
    _mesh_to_cluster_dims,
    _normalize_remote_shard_id,
    _resolve_launch_axis,
)
import triton.language.core as tlcore


class TestDeviceMesh:

    def test_device_mesh_shape_and_flatten(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
            "block_cluster": [("cluster_x", 2), ("cluster_y", 2)],
            "block": 4,
        })
        assert mesh.shape == (2, 2, 4, 2, 2, 4)
        assert mesh.ndim == 6
        assert mesh.size == 256

        flat = mesh.flatten()
        assert flat.shape == (256, )
        assert flat.dim_names == ("flat", )

    def test_device_mesh_slice_submesh(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
        })
        sub = mesh[1, :, 2]
        assert sub.shape == (2, )
        assert sub.dim_names == ("node_y", )
        assert sub.size == 2

    def test_device_mesh_invalid_topology(self):
        with pytest.raises(TypeError):
            tle.device_mesh({"node": "2"})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": []})
        with pytest.raises(ValueError):
            tle.device_mesh({"node": [("x", 0)]})


class TestShardingSpec:

    def test_sharding_spec_states(self):
        mesh = tle.device_mesh({
            "device": 4,
            "cluster": [("cluster_x", 2), ("cluster_y", 2)],
            "block": 4,
        })
        spec = tle.sharding(mesh, split=[["cluster_x", "cluster_y"], "device"], partial=["block"])
        assert spec.axis_state("cluster_x") == "S"
        assert spec.axis_state("cluster_y") == "S"
        assert spec.axis_state("device") == "S"
        assert spec.axis_state("block") == "P"
        assert spec.broadcast == tuple()

    def test_sharding_rejects_overlap(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        with pytest.raises(ValueError):
            tle.sharding(mesh, split=["device"], partial=["device"])

    def test_make_sharded_tensor(self):
        mesh = tle.device_mesh({"device": 4, "block": 4})
        spec = tle.sharding(mesh, split=["device", "block"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[4, 8])
        assert st.handle == "x_ptr"
        assert st.shape == (4, 8)
        assert st.sharding == spec

    def test_reshard_deferred(self):
        mesh = tle.device_mesh({"device": 4})
        spec = tle.sharding(mesh, split=["device"], partial=[])
        st = tle.make_sharded_tensor("x_ptr", sharding=spec, shape=[8])
        with pytest.raises(NotImplementedError):
            tle.reshard(st, spec)


class TestRemoteShardId:

    def test_normalize_remote_shard_id_scalar(self):
        assert _normalize_remote_shard_id(3, None) == 3
        with pytest.raises(ValueError):
            _normalize_remote_shard_id(-1, None)

    def test_normalize_remote_shard_id_tuple(self):
        mesh = tle.device_mesh({"cluster": [("x", 2), ("y", 4)]})
        assert _normalize_remote_shard_id((1, 3), mesh) == 7
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((2, 0), mesh)
        with pytest.raises(ValueError):
            _normalize_remote_shard_id((1, 3), None)

    def test_m3_entrypoints_are_builtins(self):
        assert tlcore.is_builtin(tle.remote)
        assert tlcore.is_builtin(tle.distributed_barrier)
        assert tlcore.is_builtin(tle.shard_id)

    def test_remote_buffered_tensor_attach_metadata(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        buf = buffered_tensor()
        remote_buf = tle.remote(buf, 1, _semantic=_FakeSemantic())

        assert remote_buf is not buf
        shard_id = getattr(remote_buf.type, "_tle_remote_shard_id", getattr(remote_buf, "_tle_remote_shard_id", None))
        scope = getattr(remote_buf.type, "_tle_remote_scope", getattr(remote_buf, "_tle_remote_scope", None))
        assert shard_id == 1
        assert scope is None
        assert not hasattr(buf, "_tle_remote_shard_id")
        assert not hasattr(buf.type, "_tle_remote_shard_id")
        assert not hasattr(buf.type, "_tle_remote_scope")

    def test_remote_buffered_tensor_rejects_duplicate_annotation(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        remote_buf = tle.remote(buffered_tensor(), 0, _semantic=_FakeSemantic())
        with pytest.raises(ValueError, match="cannot be applied twice"):
            tle.remote(remote_buf, 1, _semantic=_FakeSemantic())

    def test_remote_buffered_tensor_validates_shard_id_early(self):

        class buffered_tensor:

            def __init__(self):
                self.handle = object()
                self.type = object()

        buf = buffered_tensor()
        with pytest.raises(ValueError, match="shard_id must be >= 0"):
            tle.remote(buf, -1, _semantic=_FakeSemantic())
        with pytest.raises(ValueError, match="tuple shard_id requires scope"):
            tle.remote(buf, (0, 0), _semantic=_FakeSemantic())


class TestClusterDims:

    def test_mesh_to_cluster_dims_prefers_cluster_axes(self):
        mesh = tle.device_mesh({
            "node": [("node_x", 2), ("node_y", 2)],
            "device": 4,
            "block_cluster": [("cluster_x", 2), ("cluster_y", 1)],
            "block": 8,
        })
        assert _mesh_to_cluster_dims(mesh) == (2, 1, 1)

    def test_mesh_to_cluster_dims_fallback_to_block_axes(self):
        mesh = tle.device_mesh({"device": 4, "block": [("block_x", 2), ("block_y", 2)]})
        assert _mesh_to_cluster_dims(mesh) == (2, 2, 1)

    def test_mesh_to_cluster_dims_submesh_keeps_launch_view(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        assert sub.shape == (2, )
        assert _mesh_to_cluster_dims(sub) == (2, 2, 1)

    def test_resolve_launch_axis(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        assert _resolve_launch_axis(mesh, "cluster_x") == 0
        assert _resolve_launch_axis(mesh, "cluster_y") == 1
        assert _resolve_launch_axis(mesh, 0) == 0
        assert _resolve_launch_axis(mesh, -1) == 1
        with pytest.raises(ValueError):
            _resolve_launch_axis(mesh, "missing_axis")
        with pytest.raises(IndexError):
            _resolve_launch_axis(mesh, 2)


class _FakeOptions:

    def __init__(self):
        self.num_ctas = 1
        self.cluster_dims = (1, 1, 1)
        self.launch_cooperative_grid = False


class _FakeBuilder:

    def __init__(self):
        self.options = _FakeOptions()
        self.distributed_barrier_calls = 0
        self.distributed_barrier_group_args = []
        self.barrier_calls = 0

    def create_distributed_barrier(self, *args):
        self.distributed_barrier_calls += 1
        self.distributed_barrier_group_args.append(tuple(args))

    def create_barrier(self):
        self.barrier_calls += 1


class _LegacyDistributedBarrierBuilder(_FakeBuilder):

    def create_distributed_barrier(self):
        self.distributed_barrier_calls += 1
        self.distributed_barrier_group_args.append(tuple())


class _FakeSemantic:

    def __init__(self, builder=None):
        self.builder = _FakeBuilder() if builder is None else builder


class _LegacyBarrierSemantic:

    def __init__(self):
        self.builder = _LegacyDistributedBarrierBuilder()


class TestDistributedBarrierScope:

    def test_distributed_barrier_full_cluster_mesh(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 1)]})
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh, _semantic=semantic)
        assert semantic.builder.options.cluster_dims == (2, 1, 1)
        assert semantic.builder.distributed_barrier_calls == 1
        assert semantic.builder.distributed_barrier_group_args == [tuple()]

    def test_distributed_barrier_submesh_emits_group_descriptor(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=sub, _semantic=semantic)
        assert semantic.builder.distributed_barrier_calls == 1
        args = semantic.builder.distributed_barrier_group_args[0]
        assert args[0] == "submesh"
        assert args[1] == [2]
        assert args[2] == [1]
        assert args[3] == [0, 1]

    def test_distributed_barrier_submesh_requires_group_aware_builder(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        semantic = _LegacyBarrierSemantic()
        with pytest.raises(NotImplementedError, match="requires rebuilt TLE extension"):
            tle.distributed_barrier(mesh=sub, _semantic=semantic)

    def test_distributed_barrier_grid_mesh_enables_coop_launch(self):
        mesh = tle.device_mesh({"block": [("block_x", 4)]})
        semantic = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh, _semantic=semantic)
        assert semantic.builder.options.launch_cooperative_grid is True
        assert semantic.builder.options.cluster_dims == (1, 1, 1)
        assert semantic.builder.distributed_barrier_calls == 1
        assert semantic.builder.distributed_barrier_group_args == [("grid", [], [], [])]

    def test_distributed_barrier_grid_mesh_rejects_cluster_launch(self):
        mesh = tle.device_mesh({"block": [("block_x", 4)]})
        semantic = _FakeSemantic()
        semantic.builder.options.cluster_dims = (2, 1, 1)
        with pytest.raises(ValueError, match="requires cluster_dims=\\(1, 1, 1\\)"):
            tle.distributed_barrier(mesh=mesh, _semantic=semantic)

    def test_distributed_barrier_auto_can_pick_cluster_or_grid(self):
        mesh = tle.device_mesh({
            "block_cluster": [("cluster_x", 2)],
            "block": [("block_x", 4)],
        })

        semantic_cluster = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh[:, 0], _semantic=semantic_cluster)
        assert semantic_cluster.builder.distributed_barrier_group_args == [tuple()]

        semantic_grid = _FakeSemantic()
        tle.distributed_barrier(mesh=mesh[0, :], _semantic=semantic_grid)
        assert semantic_grid.builder.distributed_barrier_group_args == [("grid", [], [], [])]

    def test_infer_submesh_barrier_group(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        sub = mesh[0, :]
        group = _infer_submesh_barrier_group(sub, (2, 2, 1))
        assert group is not None
        assert group.kind == "submesh"
        assert group.rank == 1
        assert group.shape == (2, )
        assert group.axes == (1, )
        assert group.mask == (0, 1)

    def test_infer_submesh_barrier_group_full_mesh_returns_none(self):
        mesh = tle.device_mesh({"block_cluster": [("cluster_x", 2), ("cluster_y", 2)]})
        assert _infer_submesh_barrier_group(mesh, (2, 2, 1)) is None
