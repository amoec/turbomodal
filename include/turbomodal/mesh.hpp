#pragma once

#include "turbomodal/common.hpp"
#include <map>
#include <set>

namespace turbomodal {

struct NodeSet {
    std::string name;
    std::vector<int> node_ids;
};

struct BoundaryFace {
    std::array<int, 6> nodes;      // 3 corners + 3 midsides (TRI6)
    int parent_element;
    int face_index;                // 0-3 within parent TET10
    Eigen::Vector3d outward_normal;
    double area;
};

class Mesh {
public:
    Eigen::MatrixXd nodes;           // N_nodes x 3 (x, y, z coordinates)
    Eigen::MatrixXi elements;        // N_elements x 10 (TET10 connectivity, 0-based)
    std::vector<NodeSet> node_sets;

    // Cyclic symmetry data
    std::vector<int> left_boundary;
    std::vector<int> right_boundary;
    std::vector<std::pair<int, int>> matched_pairs;  // (left, right) node pairs
    std::vector<int> free_boundary;  // Sector-face nodes not connected to adjacent sectors

    int num_sectors = 0;
    int rotation_axis = 2;  // 0=X, 1=Y, 2=Z

    void load_from_gmsh(const std::string& filename);
    void load_from_arrays(const Eigen::MatrixXd& node_coords,
                          const Eigen::MatrixXi& element_connectivity,
                          const std::vector<NodeSet>& node_sets_in,
                          int num_sectors_in,
                          int rotation_axis_in = 2);
    void identify_cyclic_boundaries(double tolerance = 1e-6);
    void match_boundary_nodes();

    // Find a node set by name, returns nullptr if not found
    const NodeSet* find_node_set(const std::string& name) const;

    // Extract all boundary TRI6 faces from TET10 elements
    std::vector<BoundaryFace> extract_boundary_faces() const;

    // Get sorted unique node IDs on the wetted surface
    // Wetted = boundary faces not on left/right/hub surfaces
    // If "wetted_surface" node set exists, uses that instead.
    std::vector<int> get_wetted_nodes() const;

    // Build ordered meridional (r,z) boundary points from wetted surface
    // rotation_axis determines which coordinates map to (r, z)
    Eigen::MatrixXd get_meridional_profile() const;

    int num_nodes() const { return static_cast<int>(nodes.rows()); }
    int num_elements() const { return static_cast<int>(elements.rows()); }
    int num_dof() const { return 3 * num_nodes(); }
};

}  // namespace turbomodal
