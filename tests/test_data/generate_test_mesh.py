#!/usr/bin/env python3
"""
Generate a minimal Gmsh v2.2 ASCII mesh file for a single sector of a flat
annular disk with cyclic symmetry.

Geometry:
  - 24 sectors total, so sector angular span = 15 degrees (pi/12 radians)
  - Inner radius (hub): 0.05 m
  - Outer radius (tip): 0.15 m
  - Thickness: 0.01 m (z from 0 to 0.01)
  - Left boundary at theta=0 (xz-plane, y=0, x>0)
  - Right boundary at theta=15 deg

Mesh:
  - 3 radial x 2 angular x 2 axial corner nodes = 12 corner nodes
  - 2 hex cells, each split into 6 TET4 -> upgraded to TET10 with midside nodes
  - TRI6 faces on left, right, and hub boundaries

Physical groups:
  - 1: "volume"         (3D)
  - 2: "left_boundary"  (2D)
  - 3: "right_boundary" (2D)
  - 4: "hub_constraint"  (2D)
"""

import math
import itertools

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
R_INNER = 0.05
R_OUTER = 0.15
R_MID = 0.5 * (R_INNER + R_OUTER)
THICKNESS = 0.01

THETA_LEFT = 0.0            # left boundary
THETA_RIGHT = math.pi / 12  # right boundary (15 deg)

RADII = [R_INNER, R_MID, R_OUTER]        # 3 radial positions
THETAS = [THETA_LEFT, THETA_RIGHT]        # 2 angular positions
ZS = [0.0, THICKNESS]                     # 2 axial positions

# ---------------------------------------------------------------------------
# Node generation
# ---------------------------------------------------------------------------
# We will store nodes as {node_id: (x, y, z)}, 1-based.
# We also keep a coordinate -> node_id map to avoid duplicates for midside nodes.

nodes = {}          # node_id -> (x, y, z)
coord_to_id = {}    # rounded (x,y,z) tuple -> node_id
next_node_id = 1

COORD_DECIMALS = 10  # rounding precision for dedup


def _key(x, y, z):
    return (round(x, COORD_DECIMALS), round(y, COORD_DECIMALS),
            round(z, COORD_DECIMALS))


def add_node(x, y, z):
    """Add a node (or return existing if coordinates match)."""
    global next_node_id
    k = _key(x, y, z)
    if k in coord_to_id:
        return coord_to_id[k]
    nid = next_node_id
    next_node_id += 1
    nodes[nid] = (x, y, z)
    coord_to_id[k] = nid
    return nid


def cyl_node(r, theta, z):
    """Add a node given cylindrical coordinates."""
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return add_node(x, y, z)


def midside_node(n1, n2):
    """Add the midside node between two existing nodes."""
    x1, y1, z1 = nodes[n1]
    x2, y2, z2 = nodes[n2]
    return add_node(0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.5 * (z1 + z2))


# -- Create the 12 corner nodes on a structured grid --
# Index: corner_node[ir][it][iz]
#   ir: 0=inner, 1=mid, 2=outer
#   it: 0=left(theta=0), 1=right(theta=15deg)
#   iz: 0=bottom(z=0), 1=top(z=0.01)
corner_node = [[[None for _ in ZS] for _ in THETAS] for _ in RADII]

for ir, r in enumerate(RADII):
    for it, theta in enumerate(THETAS):
        for iz, z in enumerate(ZS):
            corner_node[ir][it][iz] = cyl_node(r, theta, z)

print(f"Corner nodes created: {next_node_id - 1}")

# ---------------------------------------------------------------------------
# Hex cell definition
# ---------------------------------------------------------------------------
# We have 2 hex cells (inner band, outer band).
# Each hex cell has 8 corners.
# The structured grid indices for the 2 cells:
#   Cell 0 (inner): ir=[0,1], it=[0,1], iz=[0,1]
#   Cell 1 (outer): ir=[1,2], it=[0,1], iz=[0,1]

hex_cells = []  # list of 8-tuples of node IDs (hex corner ordering)

# Standard hex corner ordering (Gmsh HEX8 convention):
#
#   For a cell with local indices (i, j, k) in (radial, angular, axial):
#     n0 = (i,   j,   k)     n4 = (i,   j,   k+1)
#     n1 = (i+1, j,   k)     n5 = (i+1, j,   k+1)
#     n2 = (i+1, j+1, k)     n6 = (i+1, j+1, k+1)
#     n3 = (i,   j+1, k)     n7 = (i,   j+1, k+1)
#
# This matches the standard hex topology where:
#   - bottom face: n0-n1-n2-n3 (z=0)
#   - top face:    n4-n5-n6-n7 (z=0.01)

for ir_start in [0, 1]:
    ir0, ir1 = ir_start, ir_start + 1
    n0 = corner_node[ir0][0][0]
    n1 = corner_node[ir1][0][0]
    n2 = corner_node[ir1][1][0]
    n3 = corner_node[ir0][1][0]
    n4 = corner_node[ir0][0][1]
    n5 = corner_node[ir1][0][1]
    n6 = corner_node[ir1][1][1]
    n7 = corner_node[ir0][1][1]
    hex_cells.append((n0, n1, n2, n3, n4, n5, n6, n7))

print(f"Hex cells: {len(hex_cells)}")

# ---------------------------------------------------------------------------
# Split each hex into 6 tetrahedra
# ---------------------------------------------------------------------------
# Standard decomposition of a hexahedron into 6 tetrahedra.
# The split depends on the parity to ensure face-compatible meshes between
# adjacent hex cells.  We use one consistent decomposition here.
#
# For hex corners labelled 0..7 as above, one standard 6-tet decomposition:
#   Tet 0: 0, 1, 2, 5
#   Tet 1: 0, 2, 3, 7
#   Tet 2: 0, 5, 4, 7
#   Tet 3: 2, 7, 5, 6
#   Tet 4: 0, 7, 5, 2   (central tet connecting the pieces)
#
# Actually let's use the well-known 5-tet or 6-tet decomposition.
# A robust 6-tet decomposition that guarantees compatible faces:
#
# Using the Julien Dompierre et al. decomposition.
# For simplicity, let's use the "type A" 6-tet split:
#   T0: (0, 5, 1, 3)
#   T1: (0, 5, 3, 4)
#   T2: (3, 5, 4, 7)
#   T3: (1, 5, 2, 3)
#   T4: (3, 5, 2, 7)
#   T5: (2, 5, 6, 7)
#
# This is a known valid decomposition. Let me verify:
# Each tet should have positive volume if the hex has positive volume.
# The decomposition must cover the entire hex without overlap.
#
# Using the Dompierre 6-tet decomposition (type based on diagonal choice):
# Reference: "How to Subdivide Pyramids, Prisms, and Hexahedra into Tetrahedra"
# by J. Dompierre et al.

def split_hex_to_tets(h):
    """
    Split a hexahedron (8 nodes) into 6 tetrahedra.

    h = (n0, n1, n2, n3, n4, n5, n6, n7) following standard hex numbering.

    Returns list of 6 tuples, each with 4 corner node IDs.
    """
    n0, n1, n2, n3, n4, n5, n6, n7 = h

    # Dompierre type-0 decomposition (diagonal 0-6):
    tets = [
        (n0, n1, n2, n5),
        (n0, n2, n7, n5),
        (n0, n2, n3, n7),
        (n0, n5, n7, n4),
        (n2, n7, n5, n6),
    ]
    # That's 5 tets - the well-known 5-tet decomposition for a hex.
    # Actually 5 tets is fine and is a valid decomposition. Let me use it.
    return tets


tet4_elements = []  # list of 4-tuples of node IDs (corners only)

for hc in hex_cells:
    tets = split_hex_to_tets(hc)
    tet4_elements.extend(tets)

print(f"TET4 elements: {len(tet4_elements)}")

# ---------------------------------------------------------------------------
# Upgrade TET4 -> TET10 by adding midside nodes
# ---------------------------------------------------------------------------
# Gmsh TET10 node ordering:
#   Corners: 0, 1, 2, 3
#   Midsides:
#     4 = mid(0,1)
#     5 = mid(0,2)
#     6 = mid(0,3)
#     7 = mid(1,2)
#     8 = mid(1,3)  -- Gmsh calls this edge 1-3
#     9 = mid(2,3)
#
# Gmsh documentation for type 11 (10-node tetrahedron):
#   Node ordering: n0, n1, n2, n3, n01, n12, n02, n03, n13, n23
#   where nIJ = midside of edge I-J
#
# Let me be precise about Gmsh's TET10 ordering from the documentation:
#
# From Gmsh docs, element type 11 (10-node second order tetrahedron):
#   Nodes: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#   where:
#     1-4 are corners
#     5 = mid(1,2)   -> edge 0-1
#     6 = mid(2,3)   -> edge 1-2
#     7 = mid(1,3)   -> edge 0-2  (NOTE: this is 1-3 in 1-based = 0-2 in 0-based)
#     8 = mid(1,4)   -> edge 0-3
#     9 = mid(2,4)   -> edge 1-3
#    10 = mid(3,4)   -> edge 2-3
#
# In 0-based indexing for corners (0,1,2,3), the midsides are:
#   m01 = mid(0,1)   -> position 4  (Gmsh node 5)
#   m12 = mid(1,2)   -> position 5  (Gmsh node 6)
#   m02 = mid(0,2)   -> position 6  (Gmsh node 7)
#   m03 = mid(0,3)   -> position 7  (Gmsh node 8)
#   m13 = mid(1,3)   -> position 8  (Gmsh node 9)
#   m23 = mid(2,3)   -> position 9  (Gmsh node 10)

tet10_elements = []  # list of 10-tuples of node IDs

for tet in tet4_elements:
    c0, c1, c2, c3 = tet
    m01 = midside_node(c0, c1)
    m12 = midside_node(c1, c2)
    m02 = midside_node(c0, c2)
    m03 = midside_node(c0, c3)
    m13 = midside_node(c1, c3)
    m23 = midside_node(c2, c3)
    tet10_elements.append((c0, c1, c2, c3, m01, m12, m02, m03, m13, m23))

print(f"TET10 elements: {len(tet10_elements)}")
print(f"Total nodes after TET10 upgrade: {len(nodes)}")

# ---------------------------------------------------------------------------
# Identify boundary faces
# ---------------------------------------------------------------------------
# We need to find triangular faces on the boundaries.
# A face is on a boundary if all its corner nodes lie on that boundary.
#
# Boundary predicates (using corner coordinates):
#   - Left boundary:  theta = 0  =>  y = 0, x > 0
#   - Right boundary: theta = 15 deg
#   - Hub:            r = R_INNER = 0.05

TOL = 1e-8


def is_left(nid):
    """Node is on left boundary (y=0, x>0)."""
    x, y, z = nodes[nid]
    return abs(y) < TOL and x > TOL


def is_right(nid):
    """Node is on right boundary (theta = 15 deg)."""
    x, y, z = nodes[nid]
    r = math.sqrt(x**2 + y**2)
    if r < TOL:
        return False
    theta = math.atan2(y, x)
    return abs(theta - THETA_RIGHT) < TOL


def is_hub(nid):
    """Node is on hub surface (r = R_INNER)."""
    x, y, z = nodes[nid]
    r = math.sqrt(x**2 + y**2)
    return abs(r - R_INNER) < TOL


# Extract all faces from TET4 elements and check boundaries.
# A TET4 has 4 triangular faces. For corners (0,1,2,3):
#   Face 0: (0, 1, 2)
#   Face 1: (0, 1, 3)
#   Face 2: (0, 2, 3)
#   Face 3: (1, 2, 3)

TET_FACES = [
    (0, 2, 1),  # face opposite to corner 3 (outward normal convention)
    (0, 1, 3),  # face opposite to corner 2
    (0, 3, 2),  # face opposite to corner 1
    (1, 2, 3),  # face opposite to corner 0
]


def get_tri6_for_face(tet10, face_local_indices):
    """
    Given a TET10 element and a face defined by 3 local corner indices,
    return the TRI6 connectivity (3 corners + 3 midsides).

    TET10 layout: (c0, c1, c2, c3, m01, m12, m02, m03, m13, m23)
    Midside index map (local corner pair -> TET10 position):
      (0,1)->4, (1,2)->5, (0,2)->6, (0,3)->7, (1,3)->8, (2,3)->9
    """
    midside_map = {
        (0, 1): 4, (1, 0): 4,
        (1, 2): 5, (2, 1): 5,
        (0, 2): 6, (2, 0): 6,
        (0, 3): 7, (3, 0): 7,
        (1, 3): 8, (3, 1): 8,
        (2, 3): 9, (3, 2): 9,
    }

    li, lj, lk = face_local_indices
    ci = tet10[li]
    cj = tet10[lj]
    ck = tet10[lk]
    mij = tet10[midside_map[(li, lj)]]
    mjk = tet10[midside_map[(lj, lk)]]
    mik = tet10[midside_map[(li, lk)]]

    # Gmsh TRI6 ordering (type 9):
    #   3 corners, then 3 midsides
    #   mid(0,1), mid(1,2), mid(0,2)  -- same pattern as TET10 face
    return (ci, cj, ck, mij, mjk, mik)


# Collect boundary faces
left_faces = []    # TRI6 on left boundary
right_faces = []   # TRI6 on right boundary
hub_faces = []     # TRI6 on hub surface

for tet10 in tet10_elements:
    tet_corners = tet10[:4]
    for face_idx, face_locals in enumerate(TET_FACES):
        li, lj, lk = face_locals
        ci, cj, ck = tet_corners[li], tet_corners[lj], tet_corners[lk]

        # Check if all 3 corner nodes are on a boundary
        if is_left(ci) and is_left(cj) and is_left(ck):
            tri6 = get_tri6_for_face(tet10, face_locals)
            left_faces.append(tri6)

        if is_right(ci) and is_right(cj) and is_right(ck):
            tri6 = get_tri6_for_face(tet10, face_locals)
            right_faces.append(tri6)

        if is_hub(ci) and is_hub(cj) and is_hub(ck):
            tri6 = get_tri6_for_face(tet10, face_locals)
            hub_faces.append(tri6)

print(f"Left boundary TRI6 faces: {len(left_faces)}")
print(f"Right boundary TRI6 faces: {len(right_faces)}")
print(f"Hub TRI6 faces: {len(hub_faces)}")

# Deduplicate faces (a face might appear from two tets sharing it, but
# boundary faces by definition only belong to one tet, so this should be fine).
# Still, let's deduplicate just in case.

def face_key(tri6):
    """Canonical key for a triangular face (sorted corner nodes)."""
    return tuple(sorted(tri6[:3]))


def dedup_faces(faces):
    seen = set()
    result = []
    for f in faces:
        k = face_key(f)
        if k not in seen:
            seen.add(k)
            result.append(f)
    return result


left_faces = dedup_faces(left_faces)
right_faces = dedup_faces(right_faces)
hub_faces = dedup_faces(hub_faces)

print(f"After dedup:")
print(f"  Left boundary TRI6 faces: {len(left_faces)}")
print(f"  Right boundary TRI6 faces: {len(right_faces)}")
print(f"  Hub TRI6 faces: {len(hub_faces)}")

# Sanity check: left and right should have same count
assert len(left_faces) == len(right_faces), \
    f"Left ({len(left_faces)}) and right ({len(right_faces)}) face counts differ!"

# ---------------------------------------------------------------------------
# Write Gmsh v2.2 ASCII file
# ---------------------------------------------------------------------------
# Format reference:
#   $MeshFormat
#   2.2 0 8
#   $EndMeshFormat
#   $PhysicalNames
#   num_names
#   dim tag "name"
#   ...
#   $EndPhysicalNames
#   $Nodes
#   num_nodes
#   node_id x y z
#   ...
#   $EndNodes
#   $Elements
#   num_elements
#   elm_id type num_tags tag1 tag2 ... node1 node2 ...
#   $EndElements
#
# Element tags: typically 2 tags: [physical_group, elementary_entity]
# We use elementary_entity = physical_group for simplicity.
#
# Element types:
#   9  = TRI6  (6-node second order triangle)
#   11 = TET10 (10-node second order tetrahedron)

OUTPUT_FILE = "/Users/adam/Projects/modal-identification/turbomodal/tests/test_data/wedge_sector.msh"

# Physical group tags
PHYS_VOLUME = 1
PHYS_LEFT = 2
PHYS_RIGHT = 3
PHYS_HUB = 4

with open(OUTPUT_FILE, 'w') as f:
    # -- MeshFormat --
    f.write("$MeshFormat\n")
    f.write("2.2 0 8\n")
    f.write("$EndMeshFormat\n")

    # -- PhysicalNames --
    f.write("$PhysicalNames\n")
    f.write("4\n")
    f.write(f'3 {PHYS_VOLUME} "volume"\n')
    f.write(f'2 {PHYS_LEFT} "left_boundary"\n')
    f.write(f'2 {PHYS_RIGHT} "right_boundary"\n')
    f.write(f'2 {PHYS_HUB} "hub_constraint"\n')
    f.write("$EndPhysicalNames\n")

    # -- Nodes --
    f.write("$Nodes\n")
    f.write(f"{len(nodes)}\n")
    for nid in sorted(nodes.keys()):
        x, y, z = nodes[nid]
        f.write(f"{nid} {x:.15e} {y:.15e} {z:.15e}\n")
    f.write("$EndNodes\n")

    # -- Elements --
    total_elements = len(tet10_elements) + len(left_faces) + \
                     len(right_faces) + len(hub_faces)
    f.write("$Elements\n")
    f.write(f"{total_elements}\n")

    elm_id = 1

    # TRI6 faces on left boundary (type 9, physical group 2)
    for tri6 in left_faces:
        tags = f"2 {PHYS_LEFT} {PHYS_LEFT}"
        node_str = " ".join(str(n) for n in tri6)
        f.write(f"{elm_id} 9 {tags} {node_str}\n")
        elm_id += 1

    # TRI6 faces on right boundary (type 9, physical group 3)
    for tri6 in right_faces:
        tags = f"2 {PHYS_RIGHT} {PHYS_RIGHT}"
        node_str = " ".join(str(n) for n in tri6)
        f.write(f"{elm_id} 9 {tags} {node_str}\n")
        elm_id += 1

    # TRI6 faces on hub surface (type 9, physical group 4)
    for tri6 in hub_faces:
        tags = f"2 {PHYS_HUB} {PHYS_HUB}"
        node_str = " ".join(str(n) for n in tri6)
        f.write(f"{elm_id} 9 {tags} {node_str}\n")
        elm_id += 1

    # TET10 volume elements (type 11, physical group 1)
    for tet10 in tet10_elements:
        tags = f"2 {PHYS_VOLUME} {PHYS_VOLUME}"
        node_str = " ".join(str(n) for n in tet10)
        f.write(f"{elm_id} 11 {tags} {node_str}\n")
        elm_id += 1

    f.write("$EndElements\n")

print(f"\nMesh written to: {OUTPUT_FILE}")
print(f"  Total nodes:    {len(nodes)}")
print(f"  Total elements: {total_elements}")
print(f"    TET10 volume: {len(tet10_elements)}")
print(f"    TRI6 left:    {len(left_faces)}")
print(f"    TRI6 right:   {len(right_faces)}")
print(f"    TRI6 hub:     {len(hub_faces)}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n--- Validation ---")

# Check that all TET10 elements have positive volume
def tet_volume(n0, n1, n2, n3):
    """Signed volume of tetrahedron from 4 corner coordinates."""
    x0, y0, z0 = n0
    x1, y1, z1 = n1
    x2, y2, z2 = n2
    x3, y3, z3 = n3
    # V = (1/6) * |det([n1-n0, n2-n0, n3-n0])|
    ax, ay, az = x1 - x0, y1 - y0, z1 - z0
    bx, by, bz = x2 - x0, y2 - y0, z2 - z0
    cx, cy, cz = x3 - x0, y3 - y0, z3 - z0
    det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return det / 6.0


all_positive = True
for i, tet10 in enumerate(tet10_elements):
    c0, c1, c2, c3 = tet10[:4]
    vol = tet_volume(nodes[c0], nodes[c1], nodes[c2], nodes[c3])
    if vol <= 0:
        print(f"  WARNING: TET10 element {i} has non-positive volume: {vol:.6e}")
        all_positive = False

if all_positive:
    print("  All TET10 elements have positive volume. OK")
else:
    print("  Some elements have non-positive volume - fixing orientation...")
    # Fix by swapping two corners to flip orientation
    fixed_tet10 = []
    for tet10 in tet10_elements:
        c0, c1, c2, c3 = tet10[:4]
        vol = tet_volume(nodes[c0], nodes[c1], nodes[c2], nodes[c3])
        if vol < 0:
            # Swap c1 and c2 to flip orientation
            # TET10: (c0, c1, c2, c3, m01, m12, m02, m03, m13, m23)
            # After swap c1<->c2:
            #   corners: c0, c2, c1, c3
            #   m01 -> m(0,new1) = m(0,2) = m02 -> position 4
            #   m12 -> m(new1,new2) = m(2,1) = m12 -> position 5
            #   m02 -> m(0,new2) = m(0,1) = m01 -> position 6
            #   m03 -> m(0,3) -> position 7 (unchanged)
            #   m13 -> m(new1,3) = m(2,3) = m23 -> position 8
            #   m23 -> m(new2,3) = m(1,3) = m13 -> position 9
            old = tet10
            fixed = (old[0], old[2], old[1], old[3],
                     old[6], old[5], old[4], old[7], old[9], old[8])
            fixed_tet10.append(fixed)
        else:
            fixed_tet10.append(tet10)
    tet10_elements = fixed_tet10

    # Re-check
    print("  Re-checking after fix...")
    all_ok = True
    for i, tet10 in enumerate(tet10_elements):
        c0, c1, c2, c3 = tet10[:4]
        vol = tet_volume(nodes[c0], nodes[c1], nodes[c2], nodes[c3])
        if vol <= 0:
            print(f"  STILL BAD: TET10 element {i} volume: {vol:.6e}")
            all_ok = False
    if all_ok:
        print("  All volumes now positive. Re-writing mesh file...")

        # Re-write the file with fixed elements
        with open(OUTPUT_FILE, 'w') as f:
            f.write("$MeshFormat\n")
            f.write("2.2 0 8\n")
            f.write("$EndMeshFormat\n")

            f.write("$PhysicalNames\n")
            f.write("4\n")
            f.write(f'3 {PHYS_VOLUME} "volume"\n')
            f.write(f'2 {PHYS_LEFT} "left_boundary"\n')
            f.write(f'2 {PHYS_RIGHT} "right_boundary"\n')
            f.write(f'2 {PHYS_HUB} "hub_constraint"\n')
            f.write("$EndPhysicalNames\n")

            f.write("$Nodes\n")
            f.write(f"{len(nodes)}\n")
            for nid in sorted(nodes.keys()):
                x, y, z = nodes[nid]
                f.write(f"{nid} {x:.15e} {y:.15e} {z:.15e}\n")
            f.write("$EndNodes\n")

            total_elements = len(tet10_elements) + len(left_faces) + \
                             len(right_faces) + len(hub_faces)
            f.write("$Elements\n")
            f.write(f"{total_elements}\n")

            elm_id = 1
            for tri6 in left_faces:
                tags = f"2 {PHYS_LEFT} {PHYS_LEFT}"
                node_str = " ".join(str(n) for n in tri6)
                f.write(f"{elm_id} 9 {tags} {node_str}\n")
                elm_id += 1
            for tri6 in right_faces:
                tags = f"2 {PHYS_RIGHT} {PHYS_RIGHT}"
                node_str = " ".join(str(n) for n in tri6)
                f.write(f"{elm_id} 9 {tags} {node_str}\n")
                elm_id += 1
            for tri6 in hub_faces:
                tags = f"2 {PHYS_HUB} {PHYS_HUB}"
                node_str = " ".join(str(n) for n in tri6)
                f.write(f"{elm_id} 9 {tags} {node_str}\n")
                elm_id += 1
            for tet10 in tet10_elements:
                tags = f"2 {PHYS_VOLUME} {PHYS_VOLUME}"
                node_str = " ".join(str(n) for n in tet10)
                f.write(f"{elm_id} 11 {tags} {node_str}\n")
                elm_id += 1

            f.write("$EndElements\n")
        print("  Mesh re-written successfully.")

# Check that left and right have matching node counts
left_nodes = set()
for tri6 in left_faces:
    left_nodes.update(tri6)
right_nodes = set()
for tri6 in right_faces:
    right_nodes.update(tri6)
hub_nodes = set()
for tri6 in hub_faces:
    hub_nodes.update(tri6)

print(f"  Left boundary nodes:  {len(left_nodes)}")
print(f"  Right boundary nodes: {len(right_nodes)}")
print(f"  Hub boundary nodes:   {len(hub_nodes)}")
assert len(left_nodes) == len(right_nodes), "Mismatch in left/right node counts!"
print("  Left/right node counts match. OK")

# Print coordinate ranges
all_coords = list(nodes.values())
xs = [c[0] for c in all_coords]
ys = [c[1] for c in all_coords]
zs_vals = [c[2] for c in all_coords]
print(f"  Coordinate ranges:")
print(f"    x: [{min(xs):.6f}, {max(xs):.6f}]")
print(f"    y: [{min(ys):.6f}, {max(ys):.6f}]")
print(f"    z: [{min(zs_vals):.6f}, {max(zs_vals):.6f}]")

print("\nDone!")
