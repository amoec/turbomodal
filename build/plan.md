# Interactive BC Editor — Full Redesign

## Context

The current `interactive_plane_selector()` is minimal — it shows a plane widget and returns a single BC on close. The user wants a complete in-viewer editor where you can define multiple BCs, see real-time node selection feedback, and configure all BC parameters without leaving the window.

## Design

Replace `interactive_plane_selector()` with `bc_editor(mesh)` — a single-window interactive application.

### UI Layout

```
┌──────────────────────────────────────────────────────────┐
│  BC Editor                                               │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                                                     │ │
│  │         3D Mesh View                                │ │
│  │         (surface wireframe + colored node spheres)  │ │
│  │         (plane widget overlaid)                     │ │
│  │                                                     │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  [Status text - upper left]                              │
│  Current: "bc_1" | FIXED | 42 surface nodes selected    │
│                                                          │
│  [Controls text - lower left]                            │
│  T: cycle type (FIXED→DISP→FRICT)                       │
│  X/Y/Z: toggle constrained component (DISP only)        │
│  Enter: accept BC & start next                           │
│  Backspace: undo last accepted BC                        │
│  Q: finish & return all BCs                              │
│                                                          │
│  [BC list - upper right]                                 │
│  Accepted: bc_1 (FIXED, 38 nodes)                        │
│            bc_2 (DISP uz=0, 55 nodes)                   │
│                                                          │
│  [Text slider widget at top for BC type]                 │
└──────────────────────────────────────────────────────────┘
```

### State Management

A class `_BCEditorState` holds:
- `surface_nodes: np.ndarray` — precomputed surface node IDs (from extract_boundary_faces, excluding left/right cyclic)
- `surface_coords: np.ndarray` — (N_surf, 3) coordinates for fast plane distance computation
- `current_type: str` — "fixed" / "displacement" / "frictionless"
- `current_components: list[bool]` — [True, True, True] for displacement
- `current_name: str` — auto-generated "bc_1", "bc_2", ...
- `current_origin: np.ndarray` — plane origin
- `current_normal: np.ndarray` — plane normal
- `accepted_bcs: list[BoundaryCondition]` — finalized BCs
- `accepted_node_sets: list[set[int]]` — node IDs for each accepted BC (for coloring)
- `bc_counter: int` — for auto-naming

### Plane Widget Callback (fires on `interaction_event='always'`)

1. Update `current_origin`, `current_normal` from callback args
2. Compute `selected = surface_coords . dot(normal) - origin.dot(normal) >= -tol` (vectorized numpy, fast)
3. Build point cloud of selected surface nodes → `add_mesh(cloud, name="current_selection", color="yellow", ...)`
4. Build point clouds of each accepted BC's nodes → `add_mesh(..., name=f"bc_{i}", color=colors[i], ...)`
5. Update status text: `add_text(f"Current: \"{name}\" | {type} | {n} surface nodes", name="status", ...)`

### Key Bindings

| Key | Action |
|---|---|
| `t` | Cycle BC type: FIXED → DISPLACEMENT → FRICTIONLESS → FIXED |
| `x` | Toggle constrained_components[0] (DISPLACEMENT only) |
| `y` | Toggle constrained_components[1] (DISPLACEMENT only) |
| `z` | Toggle constrained_components[2] (DISPLACEMENT only) |
| `Return` | Accept current BC, increment counter, reset plane to center |
| `BackSpace` | Undo last accepted BC |

### Surface Node Precomputation

At startup, compute surface nodes EXCLUDING left/right cyclic boundary nodes:
```python
boundary_faces = mesh.extract_boundary_faces()  # need to expose this or use existing
# Actually: use select_nodes_by_plane with a plane that selects everything,
# OR just get all surface nodes from PyVista's extract_surface
```

Better: extract surface from PyVista grid, get unique point IDs. Filter out left_boundary and right_boundary node IDs. This gives exactly the surface nodes the user cares about.

### Dynamic Coloring

- **Gray wireframe**: base mesh surface
- **Yellow spheres**: currently selected nodes (live, updates as plane moves)
- **Colored spheres** per accepted BC: red, blue, green, orange, purple, cyan (cycling)
- Accepted BC nodes are NOT shown in yellow (they're "claimed")

### Text Readouts (all updated dynamically via `name=`)

1. **Status** (upper_left): `Editing: "bc_2" | DISPLACEMENT (x,z) | 55 nodes`
2. **Controls** (lower_left): key binding help
3. **BC list** (upper_right): list of accepted BCs with names, types, node counts

### Return Value

`bc_editor(mesh) -> list[BoundaryCondition]` — returns all accepted BCs when the user closes the window or presses Q. If none were accepted but there's a current selection, it's auto-accepted.

## File to Modify

`python/turbomodal/viz.py` — replace `interactive_plane_selector()` with `bc_editor()`, keep backward-compat alias.

Also update `__init__.py` exports.

## Verification

Launch `bc_editor(mesh)` on the blade sector mesh. Verify:
- Plane widget moves, yellow nodes update in real-time
- T cycles type, status text updates
- Enter accepts BC, nodes turn to assigned color, new plane starts
- Multiple BCs can be defined
- Returned list has correct BoundaryCondition objects
