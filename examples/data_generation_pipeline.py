#!/usr/bin/env python3
"""Data generation pipeline: mesh -> solve -> sensors -> signals -> HDF5.

Demonstrates the complete Subsystem A (FEA) and Subsystem B (signal
synthesis) workflow for generating a training dataset.  Uses the test
mesh shipped with turbomodal; replace with your own mesh or CAD file
for real analyses.

Usage:
    python data_generation_pipeline.py
"""

import numpy as np
from pathlib import Path

import turbomodal as tm
from turbomodal import (
    DatasetConfig,
    NoiseConfig,
    ParametricRange,
    ParametricSweepConfig,
    SensorArrayConfig,
    SignalGenerationConfig,
    VirtualSensorArray,
    apply_noise,
    export_modal_results,
    generate_signals_for_condition,
    load_modal_results,
    run_parametric_sweep,
)

# ---------------------------------------------------------------------------
# 1. Load mesh
# ---------------------------------------------------------------------------
TEST_DATA = Path(__file__).resolve().parent.parent / "tests" / "test_data"
MESH_FILE = TEST_DATA / "wedge_sector.msh"
NUM_SECTORS = 24

print(f"Loading mesh from {MESH_FILE}")
mesh = tm.load_mesh(str(MESH_FILE), num_sectors=NUM_SECTORS)
print(f"  {mesh.num_nodes()} nodes, {mesh.num_elements()} elements, "
      f"{NUM_SECTORS} sectors")

# ---------------------------------------------------------------------------
# 2. Define material
# ---------------------------------------------------------------------------
mat = tm.Material(E=200e9, nu=0.3, rho=7800)
print(f"  Material: E={mat.E/1e9:.0f} GPa, nu={mat.nu}, rho={mat.rho}")

# ---------------------------------------------------------------------------
# 3. Solve at a single RPM
# ---------------------------------------------------------------------------
RPM = 5000.0
print(f"\nSolving at {RPM:.0f} RPM...")
results = tm.solve(mesh, mat, rpm=RPM, num_modes=5, verbose=1)

print(f"\nNatural frequencies at {RPM:.0f} RPM:")
for r in results:
    freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:3])
    print(f"  ND={r.harmonic_index}: [{freqs}] Hz")

# ---------------------------------------------------------------------------
# 4. Create a virtual sensor array (8 BTT probes, 1 axial station)
# ---------------------------------------------------------------------------
print("\nCreating BTT sensor array...")
sensor_config = SensorArrayConfig.default_btt_array(
    num_probes=8,
    casing_radius=0.15,
    axial_positions=[0.0],
    sample_rate=100_000.0,
    duration=0.5,
)
sensor_array = VirtualSensorArray(mesh, sensor_config)
print(f"  {sensor_array.n_sensors} sensors")

# ---------------------------------------------------------------------------
# 5. Generate clean sensor signals
# ---------------------------------------------------------------------------
print("\nGenerating clean signals...")
clean = sensor_array.generate_time_signal(
    modal_results=results,
    rpm=RPM,
)
print(f"  Signal shape: {clean.shape}  "
      f"({clean.shape[0]} channels, {clean.shape[1]} samples)")

# ---------------------------------------------------------------------------
# 6. Apply noise
# ---------------------------------------------------------------------------
print("\nApplying noise...")
noise_config = NoiseConfig(
    gaussian_snr_db=30.0,
    bandwidth_hz=40_000.0,
    drift_type="random_walk",
    drift_rate=0.001,
    adc_bits=16,
    adc_range=10.0,
    dropout_probability=0.001,
)
noisy = apply_noise(clean, noise_config, sample_rate=100_000.0)
print(f"  Noisy signal shape: {noisy.shape}")

# ---------------------------------------------------------------------------
# 7. End-to-end signal generation (alternative API)
# ---------------------------------------------------------------------------
print("\nUsing end-to-end signal generation API...")
sig_config = SignalGenerationConfig(
    sample_rate=100_000.0,
    duration=0.5,
    amplitude_mode="unit",
    amplitude_scale=1e-6,
    seed=42,
)
sig_result = generate_signals_for_condition(
    sensor_array, results, rpm=RPM,
    config=sig_config, noise_config=noise_config,
)
print(f"  Keys: {list(sig_result.keys())}")
print(f"  Signal shape: {sig_result['signals'].shape}")

# ---------------------------------------------------------------------------
# 8. Run a parametric sweep and export to HDF5
# ---------------------------------------------------------------------------
import tempfile
output_dir = Path(tempfile.gettempdir()) / "turbomodal_examples"
output_dir.mkdir(exist_ok=True)
h5_path = str(output_dir / "example_dataset.h5")

print(f"\nRunning parametric sweep (10 samples)...")
sweep_config = ParametricSweepConfig(
    ranges=[
        ParametricRange(name="rpm", low=1000, high=10000),
        ParametricRange(name="temperature", low=293.15, high=573.15),
    ],
    num_samples=10,
    seed=42,
    num_modes=5,
)
dataset_config = DatasetConfig(
    output_path=h5_path,
    include_mode_shapes=False,  # keep file small for this example
    compression="gzip",
    compression_level=4,
)
output_path = run_parametric_sweep(
    mesh, base_material=mat,
    config=sweep_config,
    dataset_config=dataset_config,
    verbose=1,
)
print(f"  Dataset saved to {output_path}")

# ---------------------------------------------------------------------------
# 9. Load and verify the HDF5 roundtrip
# ---------------------------------------------------------------------------
print("\nLoading dataset back...")
mesh_data, conditions, results_dict = load_modal_results(h5_path)
print(f"  Conditions: {len(conditions)}")
print(f"  Mesh nodes: {mesh_data['nodes'].shape[0]}")
print(f"  First condition: RPM={conditions[0].rpm:.0f}, "
      f"T={conditions[0].temperature:.1f} K")

cond0 = conditions[0]
r0 = results_dict[cond0.condition_id]
print(f"  Eigenvalues shape: {r0['eigenvalues'].shape}")

print("\nDone. Full pipeline: mesh -> solve -> sensors -> signals -> HDF5 -> load")
