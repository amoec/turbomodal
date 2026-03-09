"""Tests for turbomodal.sensors module."""

import numpy as np
import pytest

from turbomodal.sensors import (
    SensorType,
    SensorLocation,
    SensorArrayConfig,
    VirtualSensorArray,
)


# ---- SensorType enum ----

def test_sensor_type_enum():
    assert SensorType.STRAIN_GAUGE.value == "strain_gauge"
    assert SensorType.BTT_PROBE.value == "btt_probe"
    assert SensorType.CASING_ACCELEROMETER.value == "casing_accelerometer"
    assert SensorType.DISPLACEMENT.value == "displacement"


# ---- SensorLocation defaults ----

def test_sensor_location_defaults():
    loc = SensorLocation(
        sensor_type=SensorType.BTT_PROBE,
        position=np.array([0.1, 0.0, 0.0]),
        direction=np.array([-1.0, 0.0, 0.0]),
    )
    assert loc.bandwidth == 50_000.0
    assert loc.sensitivity == 1.0
    assert loc.noise_floor == 1e-9
    assert loc.label == ""


def test_sensor_location_custom():
    loc = SensorLocation(
        sensor_type=SensorType.STRAIN_GAUGE,
        position=np.array([0.15, 0.0, 0.005]),
        direction=np.array([0.0, 0.0, 1.0]),
        label="SG_01",
        bandwidth=25_000.0,
        sensitivity=2.0,
    )
    assert loc.label == "SG_01"
    assert loc.bandwidth == 25_000.0
    assert loc.sensitivity == 2.0


# ---- SensorArrayConfig defaults ----

def test_sensor_array_config_defaults():
    cfg = SensorArrayConfig()
    assert cfg.sample_rate == 100_000.0
    assert cfg.duration == 1.0
    assert cfg.sensors == []


# ---- default_btt_array ----

def test_default_btt_array_geometry():
    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
    )
    assert len(cfg.sensors) == 4
    for s in cfg.sensors:
        assert s.sensor_type == SensorType.BTT_PROBE


def test_default_btt_array_labels():
    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
    )
    for s in cfg.sensors:
        assert s.label.startswith("BTT_")


def test_default_btt_array_directions():
    """BTT probes should point radially inward."""
    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
    )
    for s in cfg.sensors:
        r = np.linalg.norm(s.position[:2])
        if r > 1e-10:
            # Direction should be opposite to position (inward)
            radial = s.position[:2] / r
            dir_radial = s.direction[:2]
            # Dot product of direction with outward radial should be negative
            assert np.dot(dir_radial, radial) < 0


# ---- default_displacement_array ----

def test_default_displacement_array_axial():
    positions = [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.01]]
    cfg = SensorArrayConfig.default_displacement_array(positions)
    assert len(cfg.sensors) == 3
    for s in cfg.sensors:
        assert s.sensor_type == SensorType.DISPLACEMENT
        np.testing.assert_allclose(s.direction, [0, 0, 1])
        assert s.label.startswith("DISP_")


def test_default_displacement_array_radial():
    positions = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    cfg = SensorArrayConfig.default_displacement_array(positions, direction="radial")
    assert len(cfg.sensors) == 2
    # First sensor at (0.1,0,0): radial direction should be [1,0,0]
    np.testing.assert_allclose(cfg.sensors[0].direction, [1, 0, 0], atol=1e-12)
    # Second sensor at (0,0.1,0): radial direction should be [0,1,0]
    np.testing.assert_allclose(cfg.sensors[1].direction, [0, 1, 0], atol=1e-12)


def test_default_displacement_array_custom_direction():
    positions = [[0.05, 0.0, 0.01]]
    d = [1.0, 1.0, 0.0]
    cfg = SensorArrayConfig.default_displacement_array(positions, direction=d)
    assert len(cfg.sensors) == 1
    np.testing.assert_allclose(cfg.sensors[0].direction, d)


def test_default_displacement_array_bandwidth_and_noise():
    positions = [[0.05, 0.0, 0.0]]
    cfg = SensorArrayConfig.default_displacement_array(
        positions, bandwidth=20_000.0, noise_floor=5e-8,
    )
    assert cfg.sensors[0].bandwidth == 20_000.0
    assert cfg.sensors[0].noise_floor == 5e-8


def test_displacement_array_with_mesh(wedge_mesh_path):
    """Displacement sensors should work with VirtualSensorArray end-to-end."""
    from turbomodal._core import Mesh
    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    # Place sensors at a few node positions
    nodes = np.asarray(mesh.nodes)
    pick = [0, nodes.shape[0] // 2, nodes.shape[0] - 1]
    positions = nodes[pick]

    cfg = SensorArrayConfig.default_displacement_array(
        positions, direction="axial", sample_rate=100_000.0, duration=0.1,
    )
    vsa = VirtualSensorArray(mesh, cfg)
    assert vsa.n_sensors == 3

    rng = np.random.default_rng(99)
    mode = rng.standard_normal(3 * mesh.num_nodes())
    sampled = vsa.sample_mode_shape(mode)
    assert sampled.shape == (3,)
    # Axial sensor at a node should read the z-displacement at that node
    for i, node_idx in enumerate(pick):
        expected = mode[3 * node_idx + 2]  # z-DOF
        np.testing.assert_allclose(sampled[i], expected, atol=1e-10)


# ---- VirtualSensorArray (requires mesh) ----

@pytest.fixture
def sensor_array_with_mesh(wedge_mesh_path):
    from turbomodal._core import Mesh
    mesh = Mesh()
    mesh.num_sectors = 24
    mesh.load_from_gmsh(wedge_mesh_path)
    mesh.identify_cyclic_boundaries()
    mesh.match_boundary_nodes()

    cfg = SensorArrayConfig.default_btt_array(
        num_probes=4,
        casing_radius=0.15,
        axial_positions=[0.005],
        sample_rate=100_000.0,
        duration=0.1,
    )
    vsa = VirtualSensorArray(mesh, cfg)
    return vsa, mesh


def test_virtual_sensor_array_n_sensors(sensor_array_with_mesh):
    vsa, _ = sensor_array_with_mesh
    assert vsa.n_sensors == 4


def test_interpolation_matrix_shape(sensor_array_with_mesh):
    vsa, mesh = sensor_array_with_mesh
    H = vsa.interpolation_matrix
    assert H.shape[0] == vsa.n_sensors
    assert H.shape[1] == 3 * mesh.num_nodes()


def test_interpolation_matrix_sparse(sensor_array_with_mesh):
    from scipy import sparse
    vsa, _ = sensor_array_with_mesh
    H = vsa.interpolation_matrix
    assert sparse.issparse(H)


def test_sample_mode_shape_real(sensor_array_with_mesh):
    vsa, mesh = sensor_array_with_mesh
    mode = np.random.default_rng(42).standard_normal(3 * mesh.num_nodes())
    sampled = vsa.sample_mode_shape(mode)
    assert sampled.shape == (vsa.n_sensors,)
    assert np.isrealobj(sampled)


def test_sample_mode_shape_complex(sensor_array_with_mesh):
    vsa, mesh = sensor_array_with_mesh
    rng = np.random.default_rng(42)
    mode = rng.standard_normal(3 * mesh.num_nodes()) + \
           1j * rng.standard_normal(3 * mesh.num_nodes())
    sampled = vsa.sample_mode_shape(mode)
    assert sampled.shape == (vsa.n_sensors,)
    assert np.iscomplexobj(sampled)


# ---- sensor_circumferential_angles ----

def test_sensor_circumferential_angles(sensor_array_with_mesh):
    """BTT probes at 0°, 90°, 180°, 270° should have correct angles."""
    vsa, mesh = sensor_array_with_mesh
    angles = vsa.sensor_circumferential_angles()
    assert angles.shape == (4,)
    expected = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    np.testing.assert_allclose(angles, expected, atol=1e-10)


# ---- mesh property ----

def test_mesh_property(sensor_array_with_mesh):
    vsa, mesh = sensor_array_with_mesh
    assert vsa.mesh is mesh


# ---- blade_tip_profile ----

def test_blade_tip_profile(sensor_array_with_mesh):
    """blade_tip_profile should return angles within [0, sector_angle)."""
    vsa, mesh = sensor_array_with_mesh
    start, end = vsa.blade_tip_profile()
    sector_angle = 2 * np.pi / mesh.num_sectors
    assert 0 <= start < sector_angle
    assert start <= end <= sector_angle


# ---- is_stationary field ----

def test_is_stationary_default():
    """is_stationary defaults to None."""
    loc = SensorLocation(
        sensor_type=SensorType.BTT_PROBE,
        position=np.zeros(3),
        direction=np.array([1, 0, 0.0]),
    )
    assert loc.is_stationary is None


def test_is_stationary_explicit():
    """is_stationary can be set explicitly."""
    loc = SensorLocation(
        sensor_type=SensorType.DISPLACEMENT,
        position=np.zeros(3),
        direction=np.array([0, 0, 1.0]),
        is_stationary=True,
    )
    assert loc.is_stationary is True
