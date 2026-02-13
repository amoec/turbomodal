#!/usr/bin/env python3
"""End-to-end example: load mesh, solve, and visualize with turbomodal.

Usage:
    python python_example.py

This example uses the test mesh files shipped with turbomodal. For real
analyses, replace with your own CAD or mesh files.
"""

import numpy as np
from pathlib import Path

import turbomodal as tm

# Locate test mesh
TEST_DATA = Path(__file__).resolve().parent.parent / "tests" / "test_data"
MESH_FILE = TEST_DATA / "leissa_disk_sector.msh"

if not MESH_FILE.exists():
    print(f"Test mesh not found at {MESH_FILE}")
    print("Using wedge_sector.msh instead")
    MESH_FILE = TEST_DATA / "wedge_sector.msh"

# --- 1. Load mesh ---
print(f"Loading mesh from {MESH_FILE}")
mesh = tm.load_mesh(str(MESH_FILE), num_sectors=36)
print(f"  {mesh.num_nodes()} nodes, {mesh.num_elements()} elements")

# --- 2. Define material (steel) ---
mat = tm.Material(E=200e9, nu=0.3, rho=7800)
print(f"  Material: E={mat.E/1e9:.1f} GPa, nu={mat.nu}, rho={mat.rho}")

# --- 3. Inspect the mesh ---
print("\nPlotting mesh...")
plotter = tm.plot_mesh(mesh, off_screen=False)
plotter.show()

# --- 4. Solve at 0 RPM ---
print("\nSolving at 0 RPM...")
results = tm.solve(mesh, mat, rpm=0, num_modes=5)
print(f"\nResults at 0 RPM:")
for r in results:
    freqs = ", ".join(f"{f:.1f}" for f in r.frequencies[:3])
    print(f"  ND={r.harmonic_index}: [{freqs}] Hz")

# --- 5. Plot first mode of ND=0 ---
print("\nPlotting ND=0 first mode...")
plotter = tm.plot_mode(mesh, results[0], mode_index=0, scale=0.001)
plotter.show()

# --- 6. Full annulus reconstruction ---
if len(results) > 1:
    print("\nPlotting full annulus ND=1 first mode...")
    plotter = tm.plot_full_annulus(mesh, results[1], mode_index=0, scale=0.001)
    plotter.show()

# --- 7. RPM sweep and Campbell diagram ---
print("\nRunning RPM sweep (0 to 15000 RPM, 6 points)...")
rpm_vals = np.linspace(0, 15000, 6)
sweep = tm.rpm_sweep(mesh, mat, rpm_vals, num_modes=5)

print("Plotting Campbell diagram...")
fig = tm.plot_campbell(sweep, engine_orders=[1, 2, 36])
import matplotlib.pyplot as plt
plt.show()

# --- 8. ZZENF diagram at max RPM ---
print(f"\nPlotting ZZENF diagram at {rpm_vals[-1]:.0f} RPM...")
fig = tm.plot_zzenf(sweep[-1], num_sectors=36)
plt.show()

print("\nDone.")
