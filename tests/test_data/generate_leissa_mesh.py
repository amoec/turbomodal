#!/usr/bin/env python3
"""
Generate a flat disk sector mesh for Leissa validation.

Uses the built-in kernel with transfinite meshing to guarantee matching
left/right boundary nodes (periodic mesh). Structured extrusion through
thickness avoids shear locking in thin plate bending with 3D solid TET10.
"""

import gmsh
import math

gmsh.initialize()
gmsh.model.add("leissa_disk")

# Geometry parameters
R_inner = 0.03    # Hub radius (b/a = 0.1, matches Leissa annular plate tables)
R_outer = 0.3     # Disk outer radius
h = 0.01          # Thickness
alpha = 15.0      # Sector angle in degrees
alpha_rad = math.radians(alpha)

# Mesh density
n_radial = 18     # nodes along radial edges
n_circum = 5      # nodes along circumferential arcs
n_layers = 3      # structured layers through thickness

geo = gmsh.model.geo

# Bottom face corner points (no mesh size - controlled by transfinite)
p1 = geo.addPoint(R_inner, 0, 0)
p2 = geo.addPoint(R_outer, 0, 0)
p3 = geo.addPoint(R_outer * math.cos(alpha_rad),
                   R_outer * math.sin(alpha_rad), 0)
p4 = geo.addPoint(R_inner * math.cos(alpha_rad),
                   R_inner * math.sin(alpha_rad), 0)
p5 = geo.addPoint(0, 0, 0)  # Center for arcs

# Curves
l1 = geo.addLine(p1, p2)              # Left radial
l2 = geo.addCircleArc(p2, p5, p3)     # Outer arc
l3 = geo.addLine(p3, p4)              # Right radial
l4 = geo.addCircleArc(p4, p5, p1)     # Inner arc

cl = geo.addCurveLoop([l1, l2, l3, l4])
s1 = geo.addPlaneSurface([cl])

# Transfinite curves: matching node counts on opposite edges
geo.mesh.setTransfiniteCurve(l1, n_radial)   # left radial
geo.mesh.setTransfiniteCurve(l3, n_radial)   # right radial (same count!)
geo.mesh.setTransfiniteCurve(l2, n_circum)   # outer arc
geo.mesh.setTransfiniteCurve(l4, n_circum)   # inner arc

# Transfinite surface: structured quad-split-to-triangles mesh
geo.mesh.setTransfiniteSurface(s1, "Left", [p1, p2, p3, p4])

# Extrude with structured layers through thickness
out = geo.extrude([(2, s1)], 0, 0, h, [n_layers])

geo.synchronize()

# Identify extruded entities
# geo.extrude returns: [top_surf, volume, side_from_l1, side_from_l2, side_from_l3, side_from_l4]
surf_top = out[0][1]
vol_tag = out[1][1]
surf_left = out[2][1]     # from l1 (left radial edge -> y=0 face)
surf_outer = out[3][1]    # from l2 (outer arc)
surf_right = out[4][1]    # from l3 (right radial edge -> theta=alpha face)
surf_inner = out[5][1]    # from l4 (inner arc)
surf_bottom = s1

print(f"Volume: {vol_tag}")
print(f"Surfaces: bottom={surf_bottom}, top={surf_top}, "
      f"left={surf_left}, right={surf_right}, "
      f"inner={surf_inner}, outer={surf_outer}")

# Physical groups
gmsh.model.addPhysicalGroup(3, [vol_tag], 1, "volume")
gmsh.model.addPhysicalGroup(2, [surf_left], 2, "left_boundary")
gmsh.model.addPhysicalGroup(2, [surf_right], 3, "right_boundary")
gmsh.model.addPhysicalGroup(2, [surf_inner], 4, "hub_constraint")

# Mesh settings
gmsh.option.setNumber("Mesh.ElementOrder", 2)

gmsh.model.mesh.generate(3)

# Count elements
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
n_tet10 = 0
for et, tags in zip(elem_types, elem_tags):
    if et == 11:
        n_tet10 += len(tags)

node_tags, _, _ = gmsh.model.mesh.getNodes()
print(f"Generated mesh: {len(node_tags)} nodes, {n_tet10} TET10 elements")

# Verify left/right surface node counts match
left_nodes = gmsh.model.mesh.getNodes(2, surf_left, includeBoundary=True)
right_nodes = gmsh.model.mesh.getNodes(2, surf_right, includeBoundary=True)
print(f"Left boundary: {len(left_nodes[0])} nodes, "
      f"Right boundary: {len(right_nodes[0])} nodes")

if len(left_nodes[0]) != len(right_nodes[0]):
    print("WARNING: Left/right node counts don't match!")

# Verify node matching by checking rotated positions
cos_a = math.cos(alpha_rad)
sin_a = math.sin(alpha_rad)
left_coords = left_nodes[1].reshape(-1, 3)
right_coords = right_nodes[1].reshape(-1, 3)

max_dist = 0
for i in range(len(left_coords)):
    lx, ly, lz = left_coords[i]
    # Find closest right node after rotating by -alpha
    best_d = 1e10
    for j in range(len(right_coords)):
        rx, ry, rz = right_coords[j]
        # Rotate right node by -alpha
        rx_rot = cos_a * rx + sin_a * ry
        ry_rot = -sin_a * rx + cos_a * ry
        d = math.sqrt((lx - rx_rot)**2 + (ly - ry_rot)**2 + (lz - rz)**2)
        best_d = min(best_d, d)
    max_dist = max(max_dist, best_d)

print(f"Max matching distance: {max_dist:.2e}")

# Write MSH 2.2 format
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write("leissa_disk_sector.msh")

gmsh.finalize()
print("Wrote leissa_disk_sector.msh")
