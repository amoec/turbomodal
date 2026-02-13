#!/usr/bin/env python3
"""Generate turbomodal visualizations for review."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

pv.OFF_SCREEN = True

import tempfile
import turbomodal as tm
from pathlib import Path

OUT = Path(tempfile.gettempdir()) / "turbomodal_viz"
OUT.mkdir(exist_ok=True)

MESH_FILE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "test_data"
    / "leissa_disk_sector.msh"
)

# --- Load mesh ---
print("Loading mesh...")
mesh = tm.load_mesh(str(MESH_FILE), num_sectors=24)
mat = tm.Material(E=200e9, nu=0.3, rho=7800)
print(
    f"  {mesh.num_nodes()} nodes, {mesh.num_elements()} elements, {mesh.num_sectors} sectors"
)

# --- 1. Mesh visualization ---
print("Generating mesh plot...")
plotter = tm.plot_mesh(mesh, off_screen=True)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "01_mesh.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '01_mesh.png'}")

# --- 2. Solve at 0 RPM ---
print("\nSolving at 0 RPM...")
results_0 = tm.solve(mesh, mat, rpm=0, num_modes=5, verbose=2)

# --- 3. Mode shape: ND=0 first mode ---
print("\nGenerating ND=0 mode 0...")
plotter = tm.plot_mode(mesh, results_0[0], mode_index=0, scale=0.002, off_screen=True)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "02_mode_nd0_m0.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '02_mode_nd0_m0.png'}")

# --- 4. Mode shape: ND=2 first mode ---
print("Generating ND=2 mode 0...")
plotter = tm.plot_mode(mesh, results_0[2], mode_index=0, scale=0.002, off_screen=True)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "03_mode_nd2_m0.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '03_mode_nd2_m0.png'}")

# --- 5. Mode shape: ND=3 first mode ---
print("Generating ND=3 mode 0...")
plotter = tm.plot_mode(mesh, results_0[3], mode_index=0, scale=0.002, off_screen=True)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "04_mode_nd3_m0.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '04_mode_nd3_m0.png'}")

# --- 6. Full annulus: ND=3 ---
print("\nGenerating full annulus ND=3 mode 0...")
plotter = tm.plot_full_annulus(
    mesh, results_0[3], mode_index=0, scale=0.002, off_screen=True
)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "05_full_annulus_nd3.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '05_full_annulus_nd3.png'}")

# --- 7. Full annulus: ND=5 ---
print("Generating full annulus ND=5 mode 0...")
plotter = tm.plot_full_annulus(
    mesh, results_0[5], mode_index=0, scale=0.002, off_screen=True
)
plotter.camera_position = "xy"
plotter.screenshot(str(OUT / "06_full_annulus_nd5.png"), window_size=[1600, 1200])
plotter.close()
print(f"  Saved {OUT / '06_full_annulus_nd5.png'}")

# --- 8. RPM sweep for Campbell diagram ---
print("\nRunning RPM sweep (0 to 15000 RPM, 16 points)...")
rpm_vals = np.linspace(0, 15000, 16)
sweep = tm.rpm_sweep(mesh, mat, rpm_vals, num_modes=5, verbose=1)

# --- 9. Campbell diagram ---
print("Generating Campbell diagram...")
fig = tm.plot_campbell(sweep, engine_orders=[1, 2, 3, 24], max_freq=8000)
fig.savefig(str(OUT / "07_campbell.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT / '07_campbell.png'}")

# --- 10. ZZENF diagram at 10000 RPM ---
print("Generating ZZENF diagram at 10000 RPM...")
# Find the closest RPM index to 10000
rpm_idx = np.argmin(np.abs(rpm_vals - 10000))
fig = tm.plot_zzenf(sweep[rpm_idx], num_sectors=24, max_freq=8000)
fig.savefig(str(OUT / "08_zzenf_10000rpm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT / '08_zzenf_10000rpm.png'}")

# --- 11. ZZENF at 0 RPM ---
print("Generating ZZENF diagram at 0 RPM...")
fig = tm.plot_zzenf(sweep[0], num_sectors=24, max_freq=8000)
fig.savefig(str(OUT / "09_zzenf_0rpm.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {OUT / '09_zzenf_0rpm.png'}")

print(f"\nAll visualizations saved to {OUT}/")
