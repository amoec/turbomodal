#pragma once

#include "turbomodal/common.hpp"
#include <map>

namespace turbomodal {

struct NodeSet {
    std::string name;
    std::vector<int> node_ids;
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

    int num_nodes() const { return static_cast<int>(nodes.rows()); }
    int num_elements() const { return static_cast<int>(elements.rows()); }
    int num_dof() const { return 3 * num_nodes(); }
};

}  // namespace turbomodal
