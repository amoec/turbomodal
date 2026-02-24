#!/usr/bin/env python3
"""
Generate a blade-like sector mesh for high-RPM diagnostics.

Geometry: a single sector of an integrally bladed disk (blisk) with:
  - Hub radius: 0.04 m
  - Disk outer radius: 0.08 m (where blade starts)
  - Blade tip radius: 0.15 m
  - Disk thickness: 0.008 m (tapers toward outer edge)
  - Blade thickness: 0.003 m (varies along span)
  - Blade has lean/sweep via curved leading/trailing edges
  - 24 sectors (15-degree sector angle)
  - Rotation axis: Z

Uses Gmsh built-in kernel with transfinite meshing for matching
left/right boundary nodes (required for cyclic symmetry).
"""

import gmsh
import math
import sys

gmsh.initialize()
gmsh.model.add("blade_sector")

# ---- Geometry parameters ----
R_hub = 0.04       # hub inner radius
R_disk = 0.08      # disk outer / blade root radius
R_tip = 0.15       # blade tip radius
h_disk = 0.008     # disk thickness at hub
h_disk_tip = 0.005 # disk thickness at disk outer edge (taper)
h_blade = 0.003    # blade thickness at root
h_blade_tip = 0.002  # blade thickness at tip

N_sectors = 24
alpha_deg = 360.0 / N_sectors  # 15 degrees
alpha_rad = math.radians(alpha_deg)

# Mesh density
n_radial_disk = 4   # radial nodes on disk portion
n_radial_blade = 6  # radial nodes on blade portion
n_circum = 3        # circumferential nodes
n_disk_thick = 2    # layers through disk thickness
n_blade_thick = 2   # layers through blade thickness

geo = gmsh.model.geo

# ---- Build 2D cross-section (bottom face) ----
# We'll build the sector in two parts: disk and blade
# Both share a common edge at R_disk

center = geo.addPoint(0, 0, 0)

# -- Disk bottom face --
# 4 corner points of the disk sector (at z=0)
p_hub_left = geo.addPoint(R_hub, 0, 0)
p_disk_left = geo.addPoint(R_disk, 0, 0)
p_disk_right = geo.addPoint(R_disk * math.cos(alpha_rad),
                             R_disk * math.sin(alpha_rad), 0)
p_hub_right = geo.addPoint(R_hub * math.cos(alpha_rad),
                            R_hub * math.sin(alpha_rad), 0)

l_disk_left = geo.addLine(p_hub_left, p_disk_left)
l_disk_outer = geo.addCircleArc(p_disk_left, center, p_disk_right)
l_disk_right = geo.addLine(p_disk_right, p_hub_right)
l_disk_inner = geo.addCircleArc(p_hub_right, center, p_hub_left)

cl_disk = geo.addCurveLoop([l_disk_left, l_disk_outer, l_disk_right, l_disk_inner])
s_disk = geo.addPlaneSurface([cl_disk])

# Transfinite for disk
geo.mesh.setTransfiniteCurve(l_disk_left, n_radial_disk)
geo.mesh.setTransfiniteCurve(l_disk_right, n_radial_disk)
geo.mesh.setTransfiniteCurve(l_disk_outer, n_circum)
geo.mesh.setTransfiniteCurve(l_disk_inner, n_circum)
geo.mesh.setTransfiniteSurface(s_disk, "Left", [p_hub_left, p_disk_left, p_disk_right, p_hub_right])

# -- Blade bottom face --
# Blade extends from R_disk to R_tip, but with a slight lean
# Lean angle: blade leans circumferentially by ~3 degrees from root to tip
lean_angle = math.radians(3.0)

# Blade tip points (slightly leaned in circumferential direction)
p_tip_left = geo.addPoint(R_tip * math.cos(lean_angle),
                           R_tip * math.sin(lean_angle), 0)
p_tip_right = geo.addPoint(R_tip * math.cos(alpha_rad + lean_angle),
                            R_tip * math.sin(alpha_rad + lean_angle), 0)

# Use splines for curved blade edges (lean from root to tip)
# Left edge: from p_disk_left to p_tip_left with mid-point control
r_mid_blade = 0.5 * (R_disk + R_tip)
lean_mid = lean_angle * 0.4  # lean progresses non-linearly
p_blade_mid_left = geo.addPoint(r_mid_blade * math.cos(lean_mid),
                                 r_mid_blade * math.sin(lean_mid), 0)
p_blade_mid_right = geo.addPoint(r_mid_blade * math.cos(alpha_rad + lean_mid),
                                  r_mid_blade * math.sin(alpha_rad + lean_mid), 0)

l_blade_left = geo.addSpline([p_disk_left, p_blade_mid_left, p_tip_left])
l_blade_tip = geo.addCircleArc(p_tip_left, center, p_tip_right)
l_blade_right = geo.addSpline([p_tip_right, p_blade_mid_right, p_disk_right])
# l_disk_outer is shared (but reversed for the blade loop)

cl_blade = geo.addCurveLoop([l_blade_left, l_blade_tip, l_blade_right, -l_disk_outer])
s_blade = geo.addPlaneSurface([cl_blade])

# Transfinite for blade
geo.mesh.setTransfiniteCurve(l_blade_left, n_radial_blade)
geo.mesh.setTransfiniteCurve(l_blade_right, n_radial_blade)
geo.mesh.setTransfiniteCurve(l_blade_tip, n_circum)
# l_disk_outer already set above with n_circum
geo.mesh.setTransfiniteSurface(s_blade, "Left", [p_disk_left, p_tip_left, p_tip_right, p_disk_right])

# ---- Extrude through thickness ----
# Disk: thicker (h_disk at hub, tapering isn't directly supported in extrude,
# but we can use a uniform thickness for the disk portion)
out_disk = geo.extrude([(2, s_disk)], 0, 0, h_disk, [n_disk_thick])

# Blade: thinner
out_blade = geo.extrude([(2, s_blade)], 0, 0, h_blade, [n_blade_thick])

geo.synchronize()

# ---- Identify surfaces for physical groups ----
# Extrude output format: [(dim, tag), ...] = [top_surf, volume, side_from_curve1, ...]

vol_disk = out_disk[1][1]
surf_disk_top = out_disk[0][1]
# Side surfaces from extrusion: in order of the curve loop
# l_disk_left -> left face of disk
surf_disk_left = out_disk[2][1]
# l_disk_outer -> outer face of disk (interface with blade)
surf_disk_outer_face = out_disk[3][1]
# l_disk_right -> right face of disk
surf_disk_right = out_disk[4][1]
# l_disk_inner -> hub face
surf_hub = out_disk[5][1]

vol_blade = out_blade[1][1]
surf_blade_top = out_blade[0][1]
# Side surfaces from blade extrusion
# l_blade_left -> left face of blade
surf_blade_left = out_blade[2][1]
# l_blade_tip -> tip face
surf_blade_tip = out_blade[3][1]
# l_blade_right -> right face of blade
surf_blade_right = out_blade[4][1]
# -l_disk_outer -> interface with disk (shared face)
surf_blade_root_face = out_blade[5][1]

print(f"Disk volume: {vol_disk}, Blade volume: {vol_blade}")
print(f"Disk left: {surf_disk_left}, Disk right: {surf_disk_right}")
print(f"Blade left: {surf_blade_left}, Blade right: {surf_blade_right}")
print(f"Hub: {surf_hub}")

# Physical groups
gmsh.model.addPhysicalGroup(3, [vol_disk, vol_blade], 1, "volume")
gmsh.model.addPhysicalGroup(2, [surf_disk_left, surf_blade_left], 2, "left_boundary")
gmsh.model.addPhysicalGroup(2, [surf_disk_right, surf_blade_right], 3, "right_boundary")
gmsh.model.addPhysicalGroup(2, [surf_hub], 4, "hub_constraint")

# ---- Mesh settings ----
gmsh.option.setNumber("Mesh.ElementOrder", 2)  # TET10
gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep tets

gmsh.model.mesh.generate(3)

# Count elements
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
n_tet10 = 0
for et, tags in zip(elem_types, elem_tags):
    if et == 11:
        n_tet10 += len(tags)

node_tags, _, _ = gmsh.model.mesh.getNodes()
print(f"\nGenerated mesh: {len(node_tags)} nodes, {n_tet10} TET10 elements")

# Verify left/right node counts
left_nodes_disk = gmsh.model.mesh.getNodes(2, surf_disk_left, includeBoundary=True)
right_nodes_disk = gmsh.model.mesh.getNodes(2, surf_disk_right, includeBoundary=True)
left_nodes_blade = gmsh.model.mesh.getNodes(2, surf_blade_left, includeBoundary=True)
right_nodes_blade = gmsh.model.mesh.getNodes(2, surf_blade_right, includeBoundary=True)

# Combine and deduplicate
left_all = set(left_nodes_disk[0].tolist()) | set(left_nodes_blade[0].tolist())
right_all = set(right_nodes_disk[0].tolist()) | set(right_nodes_blade[0].tolist())
print(f"Left boundary: {len(left_all)} nodes, Right boundary: {len(right_all)} nodes")

if len(left_all) != len(right_all):
    print("WARNING: Left/right node counts don't match!")
    print("  This will cause issues with cyclic symmetry matching.")
else:
    print("Left/right node counts match. OK")

# Write MSH 2.2
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
output_path = "/Users/adam/Projects/modal-identification/turbomodal/tests/test_data/blade_sector.msh"
gmsh.write(output_path)

gmsh.finalize()
print(f"\nWrote {output_path}")
