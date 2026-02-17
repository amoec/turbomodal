#include "turbomodal/mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <map>

namespace turbomodal {

// Gmsh element type node counts
static int gmsh_element_node_count(int type) {
    switch (type) {
        case 1: return 2;   // 2-node line
        case 2: return 3;   // 3-node triangle
        case 3: return 4;   // 4-node quadrangle
        case 4: return 4;   // 4-node tetrahedron
        case 5: return 8;   // 8-node hexahedron
        case 8: return 3;   // 3-node second order line
        case 9: return 6;   // 6-node second order triangle (TRI6)
        case 11: return 10; // 10-node second order tetrahedron (TET10)
        case 15: return 1;  // 1-node point
        default: return -1; // unknown
    }
}

void Mesh::load_from_gmsh(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mesh file: " + filename);
    }

    // Physical name mappings: tag -> (dimension, name)
    std::map<int, std::pair<int, std::string>> physical_names;

    // Temporary storage
    std::vector<Eigen::Vector3d> node_list;
    std::map<int, int> gmsh_to_local;  // Gmsh 1-based ID -> 0-based index
    std::vector<Eigen::VectorXi> tet10_list;

    // Node sets keyed by physical group name, collecting unique node IDs
    std::map<std::string, std::set<int>> node_set_map;

    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }

        if (line == "$MeshFormat") {
            std::getline(file, line);
            std::istringstream iss(line);
            double version;
            int file_type, data_size;
            iss >> version >> file_type >> data_size;
            if (version < 2.0 || version >= 3.0) {
                throw std::runtime_error("Unsupported Gmsh format version: " + std::to_string(version) + ". Only v2.x supported.");
            }
            if (file_type != 0) {
                throw std::runtime_error("Only ASCII Gmsh files supported (file-type must be 0).");
            }
            // Read $EndMeshFormat
            std::getline(file, line);
        }
        else if (line == "$PhysicalNames") {
            std::getline(file, line);
            int num_names = std::stoi(line);
            for (int i = 0; i < num_names; i++) {
                std::getline(file, line);
                std::istringstream iss(line);
                int dim, tag;
                std::string name;
                iss >> dim >> tag >> name;
                // Remove quotes from name
                if (name.front() == '"') name = name.substr(1);
                if (name.back() == '"') name.pop_back();
                physical_names[tag] = {dim, name};
            }
            // Read $EndPhysicalNames
            std::getline(file, line);
        }
        else if (line == "$Nodes") {
            std::getline(file, line);
            int num_nodes_in_file = std::stoi(line);
            node_list.reserve(num_nodes_in_file);

            for (int i = 0; i < num_nodes_in_file; i++) {
                std::getline(file, line);
                std::istringstream iss(line);
                int node_id;
                double x, y, z;
                iss >> node_id >> x >> y >> z;
                int local_id = static_cast<int>(node_list.size());
                gmsh_to_local[node_id] = local_id;
                node_list.push_back({x, y, z});
            }
            // Read $EndNodes
            std::getline(file, line);
        }
        else if (line == "$Elements") {
            std::getline(file, line);
            int num_elements_in_file = std::stoi(line);

            for (int i = 0; i < num_elements_in_file; i++) {
                std::getline(file, line);
                std::istringstream iss(line);
                int elem_id, elem_type, num_tags;
                iss >> elem_id >> elem_type >> num_tags;

                // Read tags: first tag is physical group, second is geometric entity
                std::vector<int> tags(num_tags);
                for (int t = 0; t < num_tags; t++) {
                    iss >> tags[t];
                }
                int phys_tag = (num_tags > 0) ? tags[0] : -1;

                // Read element nodes
                int nnodes = gmsh_element_node_count(elem_type);
                if (nnodes < 0) {
                    throw std::runtime_error("Unknown Gmsh element type: " + std::to_string(elem_type));
                }

                std::vector<int> elem_nodes(nnodes);
                for (int n = 0; n < nnodes; n++) {
                    int gmsh_id;
                    iss >> gmsh_id;
                    auto it = gmsh_to_local.find(gmsh_id);
                    if (it == gmsh_to_local.end()) {
                        throw std::runtime_error("Element references unknown node ID: " + std::to_string(gmsh_id));
                    }
                    elem_nodes[n] = it->second;
                }

                // Store TET10 elements for the volume mesh
                if (elem_type == 11) {
                    // Our code expects mid-edge ordering:
                    //   4=edge(0,1), 5=edge(1,2), 6=edge(0,2), 7=edge(0,3), 8=edge(1,3), 9=edge(2,3)
                    // Gmsh MSH 2.2 documentation says:
                    //   8=edge(2,3), 9=edge(1,3)  (swapped vs our convention)
                    // However, different Gmsh meshing algorithms may output either ordering.
                    // Auto-detect by checking mid-edge distances for nodes 8 and 9.
                    Eigen::Vector3d p8 = node_list[elem_nodes[8]];
                    Eigen::Vector3d p9 = node_list[elem_nodes[9]];
                    Eigen::Vector3d mid_13 = (node_list[elem_nodes[1]] + node_list[elem_nodes[3]]) * 0.5;
                    Eigen::Vector3d mid_23 = (node_list[elem_nodes[2]] + node_list[elem_nodes[3]]) * 0.5;

                    double d8_as_13 = (p8 - mid_13).squaredNorm();
                    double d8_as_23 = (p8 - mid_23).squaredNorm();

                    if (d8_as_23 < d8_as_13) {
                        // Node 8 is at mid(2,3) but should be at mid(1,3) — swap needed
                        std::swap(elem_nodes[8], elem_nodes[9]);
                    }

                    Eigen::VectorXi conn(10);
                    for (int n = 0; n < 10; n++) {
                        conn(n) = elem_nodes[n];
                    }
                    tet10_list.push_back(conn);
                }

                // Add nodes to physical group node set
                if (phys_tag > 0 && physical_names.count(phys_tag)) {
                    const std::string& group_name = physical_names[phys_tag].second;
                    auto& ns = node_set_map[group_name];
                    for (int n : elem_nodes) {
                        ns.insert(n);
                    }
                }
            }
            // Read $EndElements
            std::getline(file, line);
        }
        // Skip unknown sections
    }

    file.close();

    // Convert node_list to Eigen matrix
    int n_nodes = static_cast<int>(node_list.size());
    nodes.resize(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        nodes.row(i) = node_list[i].transpose();
    }

    // Convert tet10_list to Eigen matrix
    int n_elems = static_cast<int>(tet10_list.size());
    elements.resize(n_elems, 10);
    for (int i = 0; i < n_elems; i++) {
        elements.row(i) = tet10_list[i].transpose();
    }

    // Convert node_set_map to node_sets vector
    node_sets.clear();
    for (auto& [name, id_set] : node_set_map) {
        NodeSet ns;
        ns.name = name;
        ns.node_ids.assign(id_set.begin(), id_set.end());
        std::sort(ns.node_ids.begin(), ns.node_ids.end());
        node_sets.push_back(std::move(ns));
    }
}

void Mesh::load_from_arrays(const Eigen::MatrixXd& node_coords,
                            const Eigen::MatrixXi& element_connectivity,
                            const std::vector<NodeSet>& node_sets_in,
                            int num_sectors_in) {
    if (node_coords.cols() != 3) {
        throw std::runtime_error("node_coords must have 3 columns (x, y, z)");
    }
    if (element_connectivity.cols() != 10) {
        throw std::runtime_error("element_connectivity must have 10 columns (TET10)");
    }
    nodes = node_coords;
    elements = element_connectivity;
    node_sets = node_sets_in;
    num_sectors = num_sectors_in;
    identify_cyclic_boundaries();
    match_boundary_nodes();
}

void Mesh::identify_cyclic_boundaries(double tolerance) {
    left_boundary.clear();
    right_boundary.clear();

    const NodeSet* left_ns = find_node_set("left_boundary");
    const NodeSet* right_ns = find_node_set("right_boundary");

    if (!left_ns || !right_ns) {
        throw std::runtime_error(
            "Mesh must have physical groups named 'left_boundary' and 'right_boundary' "
            "for cyclic symmetry.");
    }

    left_boundary = left_ns->node_ids;
    right_boundary = right_ns->node_ids;

    if (left_boundary.size() != right_boundary.size()) {
        throw std::runtime_error(
            "Left and right cyclic boundaries have different node counts (left="
            + std::to_string(left_boundary.size()) + ", right="
            + std::to_string(right_boundary.size())
            + "). This usually means gmsh could not enforce periodic meshing. "
            "The Python-side node snapping should have equalised the boundaries "
            "before reaching C++ — this indicates a bug.");
    }
}

void Mesh::match_boundary_nodes() {
    if (num_sectors <= 0) {
        throw std::runtime_error("num_sectors must be set before matching boundary nodes.");
    }
    if (left_boundary.empty() || right_boundary.empty()) {
        throw std::runtime_error("Call identify_cyclic_boundaries() before match_boundary_nodes().");
    }

    double alpha = 2.0 * PI / num_sectors;
    double cos_a = std::cos(-alpha);  // Rotate right boundary back by -alpha
    double sin_a = std::sin(-alpha);

    matched_pairs.clear();
    matched_pairs.reserve(left_boundary.size());

    std::vector<bool> used(right_boundary.size(), false);

    for (int left_id : left_boundary) {
        double lx = nodes(left_id, 0);
        double ly = nodes(left_id, 1);
        double lz = nodes(left_id, 2);

        double best_dist = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (int j = 0; j < static_cast<int>(right_boundary.size()); j++) {
            if (used[j]) continue;
            int right_id = right_boundary[j];
            double rx = nodes(right_id, 0);
            double ry = nodes(right_id, 1);
            double rz = nodes(right_id, 2);

            // Rotate right node back by -alpha to compare with left
            double rx_rot = cos_a * rx - sin_a * ry;
            double ry_rot = sin_a * rx + cos_a * ry;

            double ddx = lx - rx_rot;
            double ddy = ly - ry_rot;
            double ddz = lz - rz;
            double dist = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }

        if (best_idx < 0) {
            throw std::runtime_error(
                "No matching right boundary node found for left node " +
                std::to_string(left_id) + ". The Python-side node snapping "
                "should have ensured exact matching — this indicates a bug.");
        }

        // Tolerance: relative to radial distance of the left-side node
        double r_ref = std::sqrt(lx * lx + ly * ly);
        double tol = std::max(1e-10, 1e-6 * r_ref);

        if (best_dist > tol) {
            throw std::runtime_error(
                "Left node " + std::to_string(left_id)
                + " has no matching right boundary node within tolerance "
                "(closest distance: " + std::to_string(best_dist)
                + ", tolerance: " + std::to_string(tol) + ").");
        }

        used[best_idx] = true;
        matched_pairs.push_back({left_id, right_boundary[best_idx]});
    }

    // Reorder right_boundary to match left_boundary ordering
    for (size_t i = 0; i < matched_pairs.size(); i++) {
        right_boundary[i] = matched_pairs[i].second;
    }
}

const NodeSet* Mesh::find_node_set(const std::string& name) const {
    for (const auto& ns : node_sets) {
        if (ns.name == name) {
            return &ns;
        }
    }
    return nullptr;
}

}  // namespace turbomodal
