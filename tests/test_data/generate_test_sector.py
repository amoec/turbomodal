"""Generate simple test sector CAD files (STEP and BREP) using gmsh.

Creates an annular sector (1/24 of a disk) with a blade-like protrusion,
suitable for testing load_cad() with cyclic symmetry analysis.

Usage:
    python generate_test_sector.py
"""

import gmsh
import math
import sys


def create_simple_sector(
    num_sectors: int = 24,
    inner_radius: float = 0.05,    # 50 mm
    outer_radius: float = 0.15,    # 150 mm
    thickness: float = 0.01,       # 10 mm
    blade_height: float = 0.04,    # 40 mm blade
    blade_thickness: float = 0.003,  # 3 mm blade
) -> None:
    """Create an annular sector with a simple blade protrusion."""

    sector_angle = 2 * math.pi / num_sectors  # radians

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    # --- Build the disk sector ---
    # Create the sector using OpenCASCADE boolean operations
    # Full cylinder - inner cylinder, then cut to sector angle

    # Outer cylinder (thin disk)
    outer_cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, thickness, outer_radius)
    # Inner cylinder (to subtract)
    inner_cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, thickness, inner_radius)
    # Cut inner from outer to get annular disk
    annulus, _ = gmsh.model.occ.cut([(3, outer_cyl)], [(3, inner_cyl)])

    gmsh.model.occ.synchronize()

    # Get the annulus volume tag
    annulus_tag = annulus[0][1]

    # Create a wedge to cut the sector angle
    # We'll use a large box and rotate/intersect
    # Simpler approach: use a sector-shaped extrusion

    # Actually, let's use a different approach: create the sector directly
    # by defining points in cylindrical coordinates

    gmsh.model.occ.remove([(3, annulus_tag)], recursive=True)

    # Define sector boundary points
    # Bottom face (z=0)
    p1 = gmsh.model.occ.addPoint(inner_radius, 0, 0)
    p2 = gmsh.model.occ.addPoint(outer_radius, 0, 0)
    p3 = gmsh.model.occ.addPoint(
        outer_radius * math.cos(sector_angle),
        outer_radius * math.sin(sector_angle),
        0,
    )
    p4 = gmsh.model.occ.addPoint(
        inner_radius * math.cos(sector_angle),
        inner_radius * math.sin(sector_angle),
        0,
    )
    # Center for arcs
    pc = gmsh.model.occ.addPoint(0, 0, 0)

    # Lines and arcs for bottom face
    l1 = gmsh.model.occ.addLine(p1, p2)          # inner to outer at theta=0
    a1 = gmsh.model.occ.addCircleArc(p2, pc, p3)  # outer arc
    l2 = gmsh.model.occ.addLine(p3, p4)          # outer to inner at theta=sector
    a2 = gmsh.model.occ.addCircleArc(p4, pc, p1)  # inner arc

    # Create wire and surface
    wire = gmsh.model.occ.addCurveLoop([l1, a1, l2, a2])
    surf = gmsh.model.occ.addPlaneSurface([wire])

    # Extrude to create the disk sector
    extrude = gmsh.model.occ.extrude([(2, surf)], 0, 0, thickness)

    gmsh.model.occ.synchronize()

    # Find the volume
    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
        print("ERROR: No volumes created")
        gmsh.finalize()
        return

    disk_tag = volumes[0][1]

    # --- Add a simple blade protrusion at the mid-radius ---
    mid_r = (inner_radius + outer_radius) / 2
    blade_r_start = mid_r - blade_thickness
    blade_r_end = mid_r + blade_thickness
    half_angle = sector_angle / 2

    # Blade as a box-like protrusion on top of the disk
    blade = gmsh.model.occ.addBox(
        blade_r_start * math.cos(half_angle) - blade_thickness,
        blade_r_start * math.sin(half_angle) - blade_thickness,
        thickness,
        2 * blade_thickness,
        2 * blade_thickness,
        blade_height,
    )

    # Fuse blade with disk
    fused, _ = gmsh.model.occ.fuse([(3, disk_tag)], [(3, blade)])
    gmsh.model.occ.synchronize()

    # Export to STEP and BREP
    gmsh.write("test_sector.step")
    gmsh.write("test_sector.brep")
    print("Created: test_sector.step, test_sector.brep")

    # Also create a simpler version without the blade (just the sector)
    gmsh.model.occ.remove(gmsh.model.getEntities(dim=3), recursive=True)
    gmsh.model.occ.remove(gmsh.model.getEntities(dim=2), recursive=True)
    gmsh.model.occ.remove(gmsh.model.getEntities(dim=1), recursive=True)
    gmsh.model.occ.remove(gmsh.model.getEntities(dim=0), recursive=True)
    gmsh.model.occ.synchronize()

    # Recreate just the sector
    p1 = gmsh.model.occ.addPoint(inner_radius, 0, 0)
    p2 = gmsh.model.occ.addPoint(outer_radius, 0, 0)
    p3 = gmsh.model.occ.addPoint(
        outer_radius * math.cos(sector_angle),
        outer_radius * math.sin(sector_angle),
        0,
    )
    p4 = gmsh.model.occ.addPoint(
        inner_radius * math.cos(sector_angle),
        inner_radius * math.sin(sector_angle),
        0,
    )
    pc = gmsh.model.occ.addPoint(0, 0, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    a1 = gmsh.model.occ.addCircleArc(p2, pc, p3)
    l2 = gmsh.model.occ.addLine(p3, p4)
    a2 = gmsh.model.occ.addCircleArc(p4, pc, p1)

    wire = gmsh.model.occ.addCurveLoop([l1, a1, l2, a2])
    surf = gmsh.model.occ.addPlaneSurface([wire])
    gmsh.model.occ.extrude([(2, surf)], 0, 0, thickness)
    gmsh.model.occ.synchronize()

    gmsh.write("test_disk_sector.step")
    gmsh.write("test_disk_sector.brep")
    print("Created: test_disk_sector.step, test_disk_sector.brep")

    gmsh.finalize()


if __name__ == "__main__":
    create_simple_sector()
    print("\nTest files generated. Usage:")
    print("  import turbomodal as tm")
    print('  mesh = tm.load_cad("test_disk_sector.step", num_sectors=24, mesh_size=0.005)')
    print(f"  # mesh should have nodes, elements, and detected cyclic boundaries")
