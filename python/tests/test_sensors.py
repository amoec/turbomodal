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
