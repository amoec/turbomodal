#include "turbomodal/mesh.hpp"
#include "turbomodal/element.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <map>
#include <unordered_map>

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
                            int num_sectors_in,
                            int rotation_axis_in) {
    if (node_coords.cols() != 3) {
        throw std::runtime_error("node_coords must have 3 columns (x, y, z)");
    }
    if (element_connectivity.cols() != 10) {
        throw std::runtime_error("element_connectivity must have 10 columns (TET10)");
    }
    if (rotation_axis_in < 0 || rotation_axis_in > 2) {
        throw std::runtime_error("rotation_axis must be 0 (X), 1 (Y), or 2 (Z)");
    }
    nodes = node_coords;
    elements = element_connectivity;
    node_sets = node_sets_in;
    num_sectors = num_sectors_in;
    rotation_axis = rotation_axis_in;
    identify_cyclic_boundaries();
    match_boundary_nodes();
}

void Mesh::identify_cyclic_boundaries(double tolerance) {
    left_boundary.clear();
    right_boundary.clear();
    free_boundary.clear();

    const NodeSet* left_ns = find_node_set("left_boundary");
    const NodeSet* right_ns = find_node_set("right_boundary");

    if (!left_ns || !right_ns) {
        throw std::runtime_error(
            "Mesh must have physical groups named 'left_boundary' and 'right_boundary' "
            "for cyclic symmetry.");
    }

    left_boundary = left_ns->node_ids;
    right_boundary = right_ns->node_ids;

    // Load free boundary if present (sector-face nodes not connected to
    // adjacent sectors, e.g. inset blade edges)
    const NodeSet* free_ns = find_node_set("free_boundary");
    if (free_ns) {
        free_boundary = free_ns->node_ids;
    }

    // Note: left/right counts may differ for complex blade geometries.
    // match_boundary_nodes() will match what it can and move unmatched
    // nodes to free_boundary.
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

    // Determine which coordinate axes form the rotation plane.
    // rotation_axis=0 (X) → rotate in YZ, rotation_axis=1 (Y) → XZ,
    // rotation_axis=2 (Z) → XY.
    int c1, c2, c_axial;
    if (rotation_axis == 0) {
        c1 = 1; c2 = 2; c_axial = 0;
    } else if (rotation_axis == 1) {
        c1 = 0; c2 = 2; c_axial = 1;
    } else {
        c1 = 0; c2 = 1; c_axial = 2;
    }

    matched_pairs.clear();
    std::vector<int> new_left, new_right;
    std::vector<int> free_left, free_right;
    std::vector<bool> used(right_boundary.size(), false);

    for (int left_id : left_boundary) {
        double l1 = nodes(left_id, c1);
        double l2 = nodes(left_id, c2);
        double l_ax = nodes(left_id, c_axial);

        double best_dist = std::numeric_limits<double>::max();
        int best_idx = -1;

        for (int j = 0; j < static_cast<int>(right_boundary.size()); j++) {
            if (used[j]) continue;
            int right_id = right_boundary[j];
            double r1 = nodes(right_id, c1);
            double r2 = nodes(right_id, c2);
            double r_ax = nodes(right_id, c_axial);

            // Rotate right node back by -alpha in the rotation plane
            double r1_rot = cos_a * r1 - sin_a * r2;
            double r2_rot = sin_a * r1 + cos_a * r2;

            double d1 = l1 - r1_rot;
            double d2 = l2 - r2_rot;
            double d_ax = l_ax - r_ax;
            double dist = std::sqrt(d1 * d1 + d2 * d2 + d_ax * d_ax);

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }

        // Tolerance: relative to radial distance of the left-side node
        double r_ref = std::sqrt(l1 * l1 + l2 * l2);
        double tol = std::max(1e-6, 1e-4 * r_ref);

        if (best_idx < 0 || best_dist > tol) {
            // No match — this is a free boundary node (e.g. inset blade edge)
            free_left.push_back(left_id);
            continue;
        }

        used[best_idx] = true;
        new_left.push_back(left_id);
        new_right.push_back(right_boundary[best_idx]);
        matched_pairs.push_back({left_id, right_boundary[best_idx]});
    }

    // Unmatched right nodes are also free boundary
    for (int j = 0; j < static_cast<int>(right_boundary.size()); j++) {
        if (!used[j]) {
            free_right.push_back(right_boundary[j]);
        }
    }

    // Update boundaries to contain ONLY matched nodes
    left_boundary = new_left;
    right_boundary = new_right;

    // Merge free nodes from both sides with any pre-existing free_boundary
    std::set<int> free_set(free_boundary.begin(), free_boundary.end());
    free_set.insert(free_left.begin(), free_left.end());
    free_set.insert(free_right.begin(), free_right.end());
    free_boundary.assign(free_set.begin(), free_set.end());
}

const NodeSet* Mesh::find_node_set(const std::string& name) const {
    for (const auto& ns : node_sets) {
        if (ns.name == name) {
            return &ns;
        }
    }
    return nullptr;
}

// TET10 face definitions: 4 faces, each with 3 corners + 3 midsides
// Face i is opposite to vertex i (standard convention)
static const int tet10_face_corners[4][3] = {
    {0, 2, 1},  // face 0 (opp node 3)
    {0, 1, 3},  // face 1 (opp node 2)
    {0, 3, 2},  // face 2 (opp node 1)
    {1, 2, 3},  // face 3 (opp node 0)
};
static const int tet10_face_midsides[4][3] = {
    {6, 5, 4},  // face 0
    {4, 8, 7},  // face 1
    {7, 9, 6},  // face 2
    {5, 9, 8},  // face 3
};

std::vector<BoundaryFace> Mesh::extract_boundary_faces() const {
    int n_elem = num_elements();

    // Hash each face by sorted corner node IDs.
    // Key: sorted triple of global corner IDs.
    // Value: list of (element_idx, face_idx).
    struct FaceKey {
        int a, b, c;
        bool operator==(const FaceKey& o) const { return a == o.a && b == o.b && c == o.c; }
    };
    struct FaceHash {
        size_t operator()(const FaceKey& k) const {
            size_t h = std::hash<int>()(k.a);
            h ^= std::hash<int>()(k.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct FaceInfo {
        int elem_idx;
        int face_idx;
    };

    std::unordered_map<FaceKey, std::vector<FaceInfo>, FaceHash> face_map;

    for (int e = 0; e < n_elem; e++) {
        for (int f = 0; f < 4; f++) {
            int c0 = elements(e, tet10_face_corners[f][0]);
            int c1 = elements(e, tet10_face_corners[f][1]);
            int c2 = elements(e, tet10_face_corners[f][2]);
            // Sort corners for hashing
            std::array<int, 3> sorted = {c0, c1, c2};
            std::sort(sorted.begin(), sorted.end());
            FaceKey key{sorted[0], sorted[1], sorted[2]};
            face_map[key].push_back({e, f});
        }
    }

    // Boundary faces appear exactly once
    std::vector<BoundaryFace> boundary_faces;
    for (auto& [key, infos] : face_map) {
        if (infos.size() != 1) continue;

        int e = infos[0].elem_idx;
        int f = infos[0].face_idx;

        BoundaryFace bf;
        for (int i = 0; i < 3; i++) {
            bf.nodes[i] = elements(e, tet10_face_corners[f][i]);
            bf.nodes[i + 3] = elements(e, tet10_face_midsides[f][i]);
        }
        bf.parent_element = e;
        bf.face_index = f;

        // Compute outward normal from corner cross product
        Eigen::Vector3d p0 = nodes.row(bf.nodes[0]).transpose();
        Eigen::Vector3d p1 = nodes.row(bf.nodes[1]).transpose();
        Eigen::Vector3d p2 = nodes.row(bf.nodes[2]).transpose();
        Eigen::Vector3d v1 = p1 - p0;
        Eigen::Vector3d v2 = p2 - p0;
        Eigen::Vector3d normal = v1.cross(v2);
        bf.area = 0.5 * normal.norm();
        if (bf.area > 0.0) {
            normal.normalize();
        }

        // Orient outward: check against tet centroid
        Eigen::Vector3d tet_centroid = Eigen::Vector3d::Zero();
        for (int n = 0; n < 4; n++) {  // only corner nodes for centroid
            tet_centroid += nodes.row(elements(e, n)).transpose();
        }
        tet_centroid /= 4.0;

        Eigen::Vector3d face_center = (p0 + p1 + p2) / 3.0;
        if (normal.dot(face_center - tet_centroid) < 0.0) {
            normal = -normal;
        }
        bf.outward_normal = normal;

        boundary_faces.push_back(bf);
    }
    return boundary_faces;
}

std::vector<int> Mesh::get_wetted_nodes() const {
    // Check for explicit wetted_surface node set
    const NodeSet* ws = find_node_set("wetted_surface");
    if (ws) {
        return ws->node_ids;
    }

    // Auto-detect: wetted = boundary face nodes not on left/right/hub/free
    std::set<int> left_set(left_boundary.begin(), left_boundary.end());
    std::set<int> right_set(right_boundary.begin(), right_boundary.end());
    std::set<int> free_set(free_boundary.begin(), free_boundary.end());

    std::set<int> hub_set;
    const NodeSet* hub = find_node_set("hub_constraint");
    if (hub) {
        hub_set.insert(hub->node_ids.begin(), hub->node_ids.end());
    }

    auto boundary_faces = extract_boundary_faces();
    std::set<int> wetted_set;

    for (const auto& face : boundary_faces) {
        // Check corner nodes: if all 3 are in the same boundary set, skip
        bool all_left = true, all_right = true, all_hub = true, all_free = true;
        for (int i = 0; i < 3; i++) {
            if (left_set.count(face.nodes[i]) == 0) all_left = false;
            if (right_set.count(face.nodes[i]) == 0) all_right = false;
            if (hub_set.count(face.nodes[i]) == 0) all_hub = false;
            if (free_set.count(face.nodes[i]) == 0) all_free = false;
        }
        if (all_left || all_right || all_hub || all_free) continue;

        // This face is on the wetted surface — add all 6 nodes
        for (int i = 0; i < 6; i++) {
            wetted_set.insert(face.nodes[i]);
        }
    }

    return std::vector<int>(wetted_set.begin(), wetted_set.end());
}

Eigen::MatrixXd Mesh::get_meridional_profile() const {
    auto wetted = get_wetted_nodes();
    if (wetted.empty()) return Eigen::MatrixXd();

    // Determine coordinate axes
    int c1, c2, c_axial;
    if (rotation_axis == 0) {
        c1 = 1; c2 = 2; c_axial = 0;
    } else if (rotation_axis == 1) {
        c1 = 0; c2 = 2; c_axial = 1;
    } else {
        c1 = 0; c2 = 1; c_axial = 2;
    }

    // Project wetted nodes to (r, z) and merge duplicates
    struct RZPoint {
        double r, z;
        int node_id;
    };
    std::vector<RZPoint> rz_points;
    for (int nid : wetted) {
        double x1 = nodes(nid, c1);
        double x2 = nodes(nid, c2);
        double r = std::sqrt(x1 * x1 + x2 * x2);
        double z = nodes(nid, c_axial);

        // Check if a close point already exists
        bool merged = false;
        for (auto& p : rz_points) {
            if (std::abs(p.r - r) < 1e-10 && std::abs(p.z - z) < 1e-10) {
                merged = true;
                break;
            }
        }
        if (!merged) {
            rz_points.push_back({r, z, nid});
        }
    }

    if (rz_points.size() < 2) return Eigen::MatrixXd();

    // Build adjacency from wetted boundary edges.
    // Extract edges from boundary faces that are on the wetted surface.
    auto boundary_faces = extract_boundary_faces();
    std::set<int> left_set(left_boundary.begin(), left_boundary.end());
    std::set<int> right_set(right_boundary.begin(), right_boundary.end());
    std::set<int> hub_set;
    const NodeSet* hub = find_node_set("hub_constraint");
    if (hub) hub_set.insert(hub->node_ids.begin(), hub->node_ids.end());

    // Map from node_id to rz_point index
    std::map<int, int> node_to_rz;
    for (int i = 0; i < static_cast<int>(rz_points.size()); i++) {
        node_to_rz[rz_points[i].node_id] = i;
    }
    // Also map all merged nodes
    std::set<int> wetted_set(wetted.begin(), wetted.end());
    for (int nid : wetted) {
        double x1 = nodes(nid, c1);
        double x2 = nodes(nid, c2);
        double r = std::sqrt(x1 * x1 + x2 * x2);
        double z = nodes(nid, c_axial);
        for (int i = 0; i < static_cast<int>(rz_points.size()); i++) {
            if (std::abs(rz_points[i].r - r) < 1e-10 &&
                std::abs(rz_points[i].z - z) < 1e-10) {
                node_to_rz[nid] = i;
                break;
            }
        }
    }

    // Count how many wetted faces share each edge (in rz-projected space)
    // An edge on the boundary of the wetted region appears in only 1 wetted face
    struct Edge {
        int a, b;
        bool operator<(const Edge& o) const {
            return std::tie(a, b) < std::tie(o.a, o.b);
        }
    };
    std::map<Edge, int> edge_count;

    for (const auto& face : boundary_faces) {
        bool all_left = true, all_right = true, all_hub = true;
        for (int i = 0; i < 3; i++) {
            if (left_set.count(face.nodes[i]) == 0) all_left = false;
            if (right_set.count(face.nodes[i]) == 0) all_right = false;
            if (hub_set.count(face.nodes[i]) == 0) all_hub = false;
        }
        if (all_left || all_right || all_hub) continue;

        // Wetted face: count corner edges in rz space
        for (int i = 0; i < 3; i++) {
            int n0 = face.nodes[i];
            int n1 = face.nodes[(i + 1) % 3];
            auto it0 = node_to_rz.find(n0);
            auto it1 = node_to_rz.find(n1);
            if (it0 == node_to_rz.end() || it1 == node_to_rz.end()) continue;
            int a = it0->second, b = it1->second;
            if (a == b) continue;  // degenerate after merging
            if (a > b) std::swap(a, b);
            edge_count[{a, b}]++;
        }
    }

    // Edges appearing once or that connect points with different r OR z are boundary.
    // Build adjacency for the meridional boundary.
    std::map<int, std::vector<int>> adj;
    for (auto& [edge, count] : edge_count) {
        if (count == 1) {
            adj[edge.a].push_back(edge.b);
            adj[edge.b].push_back(edge.a);
        }
    }

    // If no boundary edges found, fall back to convex-hull-like approach:
    // sort all points by angle from centroid
    if (adj.empty()) {
        double r_mean = 0, z_mean = 0;
        for (auto& p : rz_points) { r_mean += p.r; z_mean += p.z; }
        r_mean /= rz_points.size();
        z_mean /= rz_points.size();

        std::vector<int> idx(rz_points.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return std::atan2(rz_points[a].z - z_mean, rz_points[a].r - r_mean) <
                   std::atan2(rz_points[b].z - z_mean, rz_points[b].r - r_mean);
        });

        Eigen::MatrixXd result(idx.size(), 2);
        for (int i = 0; i < static_cast<int>(idx.size()); i++) {
            result(i, 0) = rz_points[idx[i]].r;
            result(i, 1) = rz_points[idx[i]].z;
        }
        return result;
    }

    // Chain boundary edges into an ordered polyline
    // Find a starting point (prefer one with only 1 neighbor = endpoint)
    int start = adj.begin()->first;
    for (auto& [node, nbrs] : adj) {
        if (nbrs.size() == 1) { start = node; break; }
    }

    std::vector<int> chain;
    std::set<int> visited;
    chain.push_back(start);
    visited.insert(start);

    while (true) {
        int cur = chain.back();
        auto it = adj.find(cur);
        if (it == adj.end()) break;
        bool found = false;
        for (int next : it->second) {
            if (visited.count(next) == 0) {
                chain.push_back(next);
                visited.insert(next);
                found = true;
                break;
            }
        }
        if (!found) break;
    }

    Eigen::MatrixXd result(chain.size(), 2);
    for (int i = 0; i < static_cast<int>(chain.size()); i++) {
        result(i, 0) = rz_points[chain[i]].r;
        result(i, 1) = rz_points[chain[i]].z;
    }
    return result;
}

// ---- Negative Jacobian detection and correction ----

// TET10 mid-edge node index → (corner_a, corner_b)
static constexpr int MID_EDGE_MAP[6][3] = {
    {4, 0, 1}, {5, 1, 2}, {6, 0, 2},
    {7, 0, 3}, {8, 1, 3}, {9, 2, 3}
};

// Check if all 4 Gauss-point Jacobians are positive for element with given coords.
static bool element_jacobians_ok(const Matrix10x3d& coords) {
    TET10Element elem;
    elem.node_coords = coords;
    for (int gp = 0; gp < 4; gp++) {
        double xi   = TET10Element::gauss_points[gp](0);
        double eta  = TET10Element::gauss_points[gp](1);
        double zeta = TET10Element::gauss_points[gp](2);
        double detJ = elem.jacobian(xi, eta, zeta).determinant();
        if (detJ <= 0.0) return false;
    }
    return true;
}

// Rotate a 3D position by angle `a` about `rotation_axis` (0=X, 1=Y, 2=Z).
static Eigen::Vector3d rotate_about_axis(const Eigen::Vector3d& p, double a, int rotation_axis) {
    double ca = std::cos(a), sa = std::sin(a);
    Eigen::Vector3d r = p;
    int c1, c2;
    if (rotation_axis == 0) { c1 = 1; c2 = 2; }
    else if (rotation_axis == 1) { c1 = 0; c2 = 2; }
    else { c1 = 0; c2 = 1; }
    r(c1) = ca * p(c1) - sa * p(c2);
    r(c2) = sa * p(c1) + ca * p(c2);
    return r;
}

Mesh::MeshQualityReport Mesh::fix_negative_jacobians() {
    MeshQualityReport report;
    int n_elem = num_elements();

    // Build lookup: global node ID → matched partner node ID.
    // Also determine sector angle for rotating corrections between boundaries.
    bool has_cyclic = !matched_pairs.empty() && num_sectors > 0;
    double sector_angle = has_cyclic ? (2.0 * PI / num_sectors) : 0.0;

    // left_node → right_node, right_node → left_node
    std::map<int, int> partner_of;
    // Track which side each boundary node is on: +1 = left, -1 = right
    std::map<int, int> boundary_side;
    if (has_cyclic) {
        for (const auto& [left_id, right_id] : matched_pairs) {
            partner_of[left_id] = right_id;
            partner_of[right_id] = left_id;
            boundary_side[left_id] = +1;
            boundary_side[right_id] = -1;
        }
    }

    // Per-node blending factors: track the maximum alpha needed across all
    // elements sharing a mid-edge node.  This ensures that a node shared by
    // multiple elements gets enough correction for all of them.
    std::map<int, double> node_alpha;  // global node ID → required alpha

    // Phase 1: determine per-element blending factors
    struct ElemFix {
        int elem_idx;
        double alpha;
        Eigen::Matrix<double, 6, 3> midpoints;  // straight-edge midpoints
        Matrix10x3d original_coords;
    };
    std::vector<ElemFix> fixes;

    for (int e = 0; e < n_elem; e++) {
        Matrix10x3d coords;
        for (int n = 0; n < 10; n++) {
            coords.row(n) = nodes.row(elements(e, n));
        }

        if (element_jacobians_ok(coords))
            continue;

        report.num_negative_jacobian++;

        // Compute edge midpoints for each mid-edge node
        Eigen::Matrix<double, 6, 3> midpoints;
        for (int m = 0; m < 6; m++) {
            int ca = MID_EDGE_MAP[m][1];
            int cb = MID_EDGE_MAP[m][2];
            midpoints.row(m) = (coords.row(ca) + coords.row(cb)) * 0.5;
        }

        // First check if fully straightened (alpha=1) works
        Matrix10x3d test_coords = coords;
        for (int m = 0; m < 6; m++) {
            int mid_node = MID_EDGE_MAP[m][0];
            test_coords.row(mid_node) = midpoints.row(m);
        }
        if (!element_jacobians_ok(test_coords)) {
            report.num_unfixable++;
            report.unfixable_elements.push_back(e);
            std::cerr << "[Mesh] Element " << e
                      << " has inverted corner tetrahedron — cannot fix by "
                         "mid-node adjustment\n";
            continue;
        }

        // Binary search for minimum alpha (10 iterations → precision ~0.001)
        double lo = 0.0, hi = 1.0;
        bool fixed = false;
        for (int iter = 0; iter < 10; iter++) {
            double mid_alpha = (lo + hi) * 0.5;
            test_coords = coords;
            for (int m = 0; m < 6; m++) {
                int mid_node = MID_EDGE_MAP[m][0];
                test_coords.row(mid_node) =
                    (1.0 - mid_alpha) * coords.row(mid_node) +
                    mid_alpha * midpoints.row(m);
            }
            if (element_jacobians_ok(test_coords)) {
                hi = mid_alpha;
                fixed = true;
            } else {
                lo = mid_alpha;
            }
        }

        if (fixed) {
            // Record this element's required alpha per mid-edge node
            for (int m = 0; m < 6; m++) {
                int gid = elements(e, MID_EDGE_MAP[m][0]);
                node_alpha[gid] = std::max(node_alpha[gid], hi);
            }
            fixes.push_back({e, hi, midpoints, coords});
            report.num_fixed++;
            report.fixed_elements.push_back(e);
        }
    }

    // Phase 2: propagate blending factors to cyclic partners.
    // If a boundary node needs alpha, its matched partner needs at least
    // the same alpha so the geometry remains rotationally symmetric.
    if (has_cyclic) {
        // Iterate until stable (typically 1 pass since partners are direct)
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& [nid, alpha_val] : node_alpha) {
                auto pit = partner_of.find(nid);
                if (pit == partner_of.end()) continue;
                int partner = pit->second;
                double& partner_alpha = node_alpha[partner];
                if (alpha_val > partner_alpha) {
                    partner_alpha = alpha_val;
                    changed = true;
                }
            }
        }
    }

    // Phase 3: apply corrections.
    // For each node that needs correction, compute the blended position.
    // For cyclic boundary nodes, also correct the partner with the
    // rotationally equivalent displacement.
    std::set<int> corrected;
    for (auto& fix : fixes) {
        for (int m = 0; m < 6; m++) {
            int mid_local = MID_EDGE_MAP[m][0];
            int gid = elements(fix.elem_idx, mid_local);
            if (corrected.count(gid)) continue;

            double alpha = node_alpha[gid];
            Eigen::Vector3d old_pos = fix.original_coords.row(mid_local).transpose();
            Eigen::Vector3d straight_pos = fix.midpoints.row(m).transpose();
            Eigen::Vector3d new_pos = (1.0 - alpha) * old_pos + alpha * straight_pos;
            Eigen::Vector3d delta = new_pos - old_pos;

            nodes.row(gid) = new_pos.transpose();
            corrected.insert(gid);

            // Propagate to cyclic partner: set partner position to the
            // rotated image of the corrected position, enforcing exact symmetry.
            if (has_cyclic) {
                auto pit = partner_of.find(gid);
                if (pit != partner_of.end() && !corrected.count(pit->second)) {
                    int partner = pit->second;
                    int side = boundary_side[gid];
                    // side=+1 (left): partner is right → rotate new_pos by +sector_angle
                    // side=-1 (right): partner is left → rotate new_pos by -sector_angle
                    double rot_angle = (side > 0) ? sector_angle : -sector_angle;
                    nodes.row(partner) = rotate_about_axis(new_pos, rot_angle, rotation_axis).transpose();
                    corrected.insert(partner);
                }
            }
        }
    }

    if (report.num_negative_jacobian > 0) {
        std::cerr << "[Mesh] Negative Jacobian report: "
                  << report.num_negative_jacobian << " bad elements, "
                  << report.num_fixed << " fixed, "
                  << report.num_unfixable << " unfixable\n";
    }

    return report;
}

}  // namespace turbomodal
