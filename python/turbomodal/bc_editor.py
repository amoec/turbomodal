"""Interactive boundary condition editor with Qt sidebar and multiple selection modes.

This module provides the ``BCEditorApp`` class, which replaces the monolithic
``bc_editor()`` function in ``viz.py`` with a PyVistaQt-based editor featuring:

- Qt sidebar with BC type selector, property fields, and accepted BCs list
- Multiple selection modes: plane, box/rubber-band, named selection, face pick
- Full undo/redo stack
- Professional color scheme with type-specific colors
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from turbomodal._core import Mesh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BC_TYPES = ["fixed", "displacement", "frictionless", "elastic_support", "cylindrical"]

BC_TYPE_LABELS = {
    "fixed": "Fixed Support",
    "displacement": "Displacement",
    "frictionless": "Frictionless",
    "elastic_support": "Elastic Support",
    "cylindrical": "Cylindrical Support",
}

BC_COLORS = {
    "fixed": "#2563EB",
    "displacement": "#059669",
    "frictionless": "#7C3AED",
    "elastic_support": "#DC2626",
    "cylindrical": "#D97706",
}

_INSTANCE_COLORS = [
    "#2563EB", "#059669", "#7C3AED", "#DC2626", "#D97706",
    "#0891B2", "#BE185D", "#4B5563",
]

SELECTION_MODES = ["plane", "box", "named", "face"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EditorCommand:
    """A reversible editor action for undo/redo."""
    action: str  # "accept", "delete"
    bc: object
    node_ids: np.ndarray
    index: int = -1


@dataclass
class EditorState:
    """Mutable state for the editor."""
    type_idx: int = 0
    components: list = field(default_factory=lambda: [True, True, True])
    counter: int = 1
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    selection_radius: float = 1.0
    spring_k: list = field(default_factory=lambda: [1e8, 1e8, 1e8])
    cyl_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    cyl_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    picked_node_ids: set = field(default_factory=set)


# ---------------------------------------------------------------------------
# Selection Modes
# ---------------------------------------------------------------------------

class SelectionMode(ABC):
    """Abstract base for node selection strategies."""

    def __init__(self, editor: "BCEditorApp"):
        self.editor = editor

    @abstractmethod
    def activate(self) -> None: ...

    @abstractmethod
    def deactivate(self) -> None: ...

    @abstractmethod
    def get_selected_ids(self) -> np.ndarray: ...

    def on_refresh(self) -> None:
        """Mode-specific visuals drawn each refresh."""

    def _exclude_claimed(self, ids: np.ndarray) -> np.ndarray:
        """Remove already-accepted node IDs from a selection."""
        if len(ids) == 0:
            return ids
        claimed = set()
        for acc_ids in self.editor._accepted_ids:
            claimed.update(acc_ids.tolist())
        if not claimed:
            return ids
        mask = np.array([int(i) not in claimed for i in ids], dtype=bool)
        return ids[mask]


class PlaneSelectionMode(SelectionMode):
    """Select surface nodes on the positive side of a cutting plane.

    Uses the interactive VTK plane widget for dragging in the 3D viewport.
    Origin/radius/normal-snap controls are Qt widgets in the sidebar.
    """

    def __init__(self, editor: "BCEditorApp"):
        super().__init__(editor)
        self._widget = None
        self._suppress = False

    def activate(self) -> None:
        e = self.editor
        p = e.plotter
        sb = e._sidebar

        if sb:
            sb.plane_controls.setVisible(True)
            self._sync_spinboxes_from_state()

        # Interactive VTK plane widget for dragging
        def _on_plane(normal, origin, widget=None):
            if widget is not None and self._widget is None:
                self._widget = widget
            if self._suppress:
                return
            e.state.normal = np.array(normal)
            e.state.origin = np.array(origin)
            # Sync sidebar spinboxes
            if sb:
                sb._suppress_plane_cb = True
                for i, spin in enumerate(sb.origin_spins):
                    spin.setValue(float(e.state.origin[i]))
                sb._suppress_plane_cb = False
            e._refresh()

        p.add_plane_widget(
            _on_plane,
            normal=e.state.normal,
            origin=e.state.origin,
            normal_rotation=True,
            interaction_event="end",
            pass_widget=True,
        )

    def deactivate(self) -> None:
        e = self.editor
        sb = e._sidebar
        if sb:
            sb.plane_controls.setVisible(False)
        e.plotter.clear_plane_widgets()
        self._widget = None
        try:
            e.plotter.remove_actor("radius_disk")
        except (KeyError, ValueError):
            pass

    def get_selected_ids(self) -> np.ndarray:
        e = self.editor
        n = np.asarray(e.state.normal, dtype=np.float64)
        n_hat = n / (np.linalg.norm(n) + 1e-30)
        o = np.asarray(e.state.origin, dtype=np.float64)
        dists = e._surface_coords @ n_hat - o @ n_hat
        sel_mask = dists >= -1e-6

        radius = e.state.selection_radius
        if radius < e._max_radius * 0.99:
            delta = e._surface_coords - o
            normal_comp = np.outer(delta @ n_hat, n_hat)
            on_plane = delta - normal_comp
            plane_dist = np.linalg.norm(on_plane, axis=1)
            sel_mask &= plane_dist <= radius

        return self._exclude_claimed(e._surface_node_ids[sel_mask])

    def on_refresh(self) -> None:
        e = self.editor
        p = e.plotter
        radius = e.state.selection_radius

        # Show translucent disc for selection radius
        if radius < e._max_radius * 0.99:
            try:
                disk = pv.Disc(
                    center=e.state.origin, normal=e.state.normal,
                    inner=0.0, outer=radius, r_res=1, c_res=36,
                )
                p.add_mesh(disk, name="radius_disk", color="#38BDF8",
                           opacity=0.10, show_edges=True,
                           edge_color="#38BDF8", line_width=1)
            except Exception:
                pass
        else:
            try:
                p.remove_actor("radius_disk")
            except (KeyError, ValueError):
                pass

    def _sync_spinboxes_from_state(self):
        sb = self.editor._sidebar
        if not sb:
            return
        sb._suppress_plane_cb = True
        for i, spin in enumerate(sb.origin_spins):
            spin.setValue(float(self.editor.state.origin[i]))
        sb.radius_spin.setValue(float(self.editor.state.selection_radius))
        sb._suppress_plane_cb = False

    def update_origin_from_sidebar(self):
        """Called when sidebar origin spinboxes change — sync the VTK widget."""
        if self._widget is not None:
            self._suppress = True
            self._widget.SetOrigin(*self.editor.state.origin)
            self._widget.UpdatePlacement()
            self._suppress = False
        self.editor._refresh()

    def update_normal(self, normal):
        """Called when sidebar normal buttons change — sync the VTK widget."""
        e = self.editor
        e.state.normal = np.array(normal, dtype=np.float64)
        nn = np.linalg.norm(e.state.normal)
        if nn > 1e-12:
            e.state.normal /= nn
        if self._widget is not None:
            self._suppress = True
            self._widget.SetNormal(*e.state.normal)
            self._widget.UpdatePlacement()
            self._suppress = False
        e._refresh()


class BoxSelectionMode(SelectionMode):
    """Rubber-band box selection of visible surface nodes."""

    def activate(self) -> None:
        e = self.editor
        e.state.picked_node_ids = set()
        sb = e._sidebar
        if sb:
            sb.clear_sel_btn.setVisible(True)
            sb.mode_hint.setText("Drag a rectangle on the 3D view to select nodes.\n"
                                 "Each drag adds to the selection.\n"
                                 "Switch to Orbit tool to rotate the view.")
            sb.mode_hint.setVisible(True)

        # Add surface as a pickable mesh for rectangle selection
        self._pickable_name = "_box_pick_surface"
        e.plotter.add_mesh(
            e._surface_mesh, name=self._pickable_name,
            color="lightgray", opacity=0.01, pickable=True,
        )

    def deactivate(self) -> None:
        e = self.editor
        try:
            e.plotter.remove_actor(self._pickable_name)
        except (KeyError, ValueError):
            pass
        e.state.picked_node_ids.clear()
        sb = e._sidebar
        if sb:
            sb.clear_sel_btn.setVisible(False)
            sb.mode_hint.setVisible(False)

    def get_selected_ids(self) -> np.ndarray:
        ids = np.array(sorted(self.editor.state.picked_node_ids), dtype=np.int64)
        return self._exclude_claimed(ids)

    def pick_in_frustum(self, frustum):
        """Select all surface nodes inside a screen-space frustum."""
        e = self.editor
        try:
            extracted = e._surface_mesh.select_enclosed_points(frustum)
            inside = extracted.point_data["SelectedPoints"].astype(bool)
            orig_ids = e._surface_mesh.point_data.get("vtkOriginalPointIds", None)
            surface_set = set(e._surface_node_ids.tolist())
            if orig_ids is not None:
                new_ids = {int(orig_ids[i]) for i in np.where(inside)[0]
                           if int(orig_ids[i]) in surface_set}
            else:
                new_ids = {int(i) for i in np.where(inside)[0] if i in surface_set}
            e.state.picked_node_ids |= new_ids
            e._refresh()
        except Exception:
            pass


class NamedSelectionMode(SelectionMode):
    """Select from pre-defined mesh node sets."""

    def activate(self) -> None:
        e = self.editor
        e.state.picked_node_ids = set()
        sb = e._sidebar
        if sb:
            combo = sb.named_combo
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("-- Select a node set --")
            surface_set = set(e._surface_node_ids.tolist())
            for ns in e.mesh.node_sets:
                n_surf = sum(1 for nid in ns.node_ids if nid in surface_set)
                if n_surf > 0:
                    combo.addItem(f"{ns.name} ({n_surf} surface nodes)")
            combo.blockSignals(False)
            combo.setVisible(True)
            combo.setEnabled(True)
            sb.mode_hint.setText("Choose a named node set from the dropdown.\n"
                                 "Nodes will be highlighted in the 3D view.")
            sb.mode_hint.setVisible(True)

    def deactivate(self) -> None:
        e = self.editor
        e.state.picked_node_ids.clear()
        sb = e._sidebar
        if sb:
            sb.named_combo.setVisible(False)
            sb.mode_hint.setVisible(False)

    def get_selected_ids(self) -> np.ndarray:
        ids = np.array(sorted(self.editor.state.picked_node_ids), dtype=np.int64)
        return self._exclude_claimed(ids)

    def on_combo_changed(self, index):
        if index <= 0:
            return
        e = self.editor
        # Filter node_sets to only those with surface nodes > 0
        surface_set = set(e._surface_node_ids.tolist())
        visible_sets = [ns for ns in e.mesh.node_sets
                        if any(nid in surface_set for nid in ns.node_ids)]
        if 0 <= index - 1 < len(visible_sets):
            ns = visible_sets[index - 1]
            e.state.picked_node_ids = {nid for nid in ns.node_ids if nid in surface_set}
            e._refresh()


class FaceSelectionMode(SelectionMode):
    """Click a surface to select connected face region via incremental flood fill.

    Uses incremental dihedral-angle check: a neighbor cell is included if
    the angle between its normal and the current cell's normal is below a
    threshold.  This naturally follows curved surfaces while stopping at
    sharp edges — exactly how commercial FEA tools work.
    """

    _DIHEDRAL_THRESHOLD = 75.0  # max angle between adjacent cell normals (< 90° to stop at sharp edges)

    def activate(self) -> None:
        e = self.editor
        e.state.picked_node_ids = set()
        sb = e._sidebar
        if sb:
            sb.clear_sel_btn.setVisible(True)
            sb.mode_hint.setText("Click on a surface to select the face region.\n"
                                 "Each click adds to the selection.\n"
                                 "Switch to Orbit tool to rotate the view.")
            sb.mode_hint.setVisible(True)

        # The extracted surface is already linear triangles (extract_surface
        # linearizes TET10 quadratic faces).  Just compute normals.
        surface = e._surface_mesh
        self._surface = surface.compute_normals(
            cell_normals=True, point_normals=False, inplace=False,
        )
        n_cells = self._surface.n_cells

        # Build edge-based neighbor map (edge = shared by 2 cells)
        pt2cells: dict[int, list[int]] = {}
        edge2cells: dict[tuple[int, int], list[int]] = {}
        for cid in range(n_cells):
            pts = self._surface.get_cell(cid).point_ids
            for pid in pts:
                pt2cells.setdefault(pid, []).append(cid)
            for i in range(len(pts)):
                p0, p1 = pts[i], pts[(i + 1) % len(pts)]
                edge = (min(p0, p1), max(p0, p1))
                edge2cells.setdefault(edge, []).append(cid)

        self._pt2cells = pt2cells

        # Identify cyclic boundary cells — cells where ALL nodes are on
        # the left or right cyclic boundary.  These should not be pickable.
        exclude_nodes = set(e.mesh.left_boundary) | set(e.mesh.right_boundary)
        self._excluded_cells: set[int] = set()
        for cid in range(n_cells):
            pts = self._surface.get_cell(cid).point_ids
            if all(int(p) in exclude_nodes for p in pts):
                self._excluded_cells.add(cid)

        self._edge_neighbors: dict[int, set[int]] = {}
        for cid in range(n_cells):
            nbrs = set()
            pts = self._surface.get_cell(cid).point_ids
            for i in range(len(pts)):
                p0, p1 = pts[i], pts[(i + 1) % len(pts)]
                edge = (min(p0, p1), max(p0, p1))
                for other in edge2cells.get(edge, []):
                    if other != cid:
                        nbrs.add(other)
            self._edge_neighbors[cid] = nbrs

        print(f"[FACE] {n_cells} surface cells, {len(self._excluded_cells)} cyclic boundary cells excluded")

    def deactivate(self) -> None:
        e = self.editor
        e.state.picked_node_ids.clear()
        sb = e._sidebar
        if sb:
            sb.clear_sel_btn.setVisible(False)
            sb.mode_hint.setVisible(False)

    def get_selected_ids(self) -> np.ndarray:
        ids = np.array(sorted(self.editor.state.picked_node_ids), dtype=np.int64)
        return self._exclude_claimed(ids)

    def flood_fill_from_cell(self, seed_cell: int):
        """Incremental flood fill: spread to neighbors whose dihedral angle
        with the current cell is below the threshold."""
        e = self.editor
        surface = self._surface
        normals = surface.cell_normals
        cos_thr = np.cos(np.radians(self._DIHEDRAL_THRESHOLD))
        surface_set = set(e._surface_node_ids.tolist())

        excluded = self._excluded_cells
        visited = set()
        queue = [seed_cell]
        selected_cells = set()

        while queue:
            cid = queue.pop()
            if cid in visited or cid in excluded:
                continue
            visited.add(cid)
            selected_cells.add(cid)

            n_cur = normals[cid]
            for nid in self._edge_neighbors.get(cid, []):
                if nid in visited or nid in excluded:
                    continue
                # Incremental check: compare THIS cell to its NEIGHBOR
                dot = float(np.dot(n_cur, normals[nid]))
                if dot >= cos_thr:  # angle < threshold → same face
                    queue.append(nid)

        # Collect node IDs
        new_nodes = set()
        for cid in selected_cells:
            for pid in surface.get_cell(cid).point_ids:
                if pid in surface_set:
                    new_nodes.add(pid)

        print(f"[FACE] flood fill: {len(selected_cells)} cells, {len(new_nodes)} nodes")
        e.state.picked_node_ids |= new_nodes
        e._refresh()


# ---------------------------------------------------------------------------
# Qt Sidebar
# ---------------------------------------------------------------------------

def _build_sidebar(editor: "BCEditorApp"):
    """Build the Qt sidebar widget with all controls."""
    from qtpy import QtWidgets, QtCore, QtGui

    sidebar = QtWidgets.QWidget()
    sidebar.setMinimumWidth(290)
    sidebar.setMaximumWidth(330)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMinimumWidth(290)
    scroll.setMaximumWidth(330)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    layout = QtWidgets.QVBoxLayout(sidebar)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(6)

    # --- Title ---
    title = QtWidgets.QLabel("Boundary Conditions")
    title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px 0;")
    layout.addWidget(title)

    # === Selection Mode ===
    mode_group = QtWidgets.QGroupBox("Selection Mode")
    mode_layout = QtWidgets.QHBoxLayout(mode_group)
    mode_layout.setContentsMargins(4, 4, 4, 4)
    mode_buttons = QtWidgets.QButtonGroup(sidebar)
    mode_labels = {"plane": "Plane", "box": "Box", "named": "Named", "face": "Face"}
    for i, mode_key in enumerate(SELECTION_MODES):
        btn = QtWidgets.QPushButton(mode_labels[mode_key])
        btn.setCheckable(True)
        btn.setChecked(i == 0)
        btn.setFixedHeight(28)
        mode_buttons.addButton(btn, i)
        mode_layout.addWidget(btn)
    layout.addWidget(mode_group)

    def _on_mode_changed(btn_id):
        editor._switch_selection_mode(SELECTION_MODES[btn_id])

    mode_buttons.idClicked.connect(_on_mode_changed)

    # === Interaction Tool (Orbit / Select) ===
    tool_row = QtWidgets.QHBoxLayout()
    tool_label = QtWidgets.QLabel("Tool:")
    tool_label.setFixedWidth(35)
    tool_label.setStyleSheet("font-weight: bold;")
    orbit_btn = QtWidgets.QPushButton("Orbit")
    orbit_btn.setCheckable(True)
    orbit_btn.setChecked(True)
    orbit_btn.setFixedHeight(28)
    orbit_btn.setStyleSheet(
        "QPushButton:checked { background-color: #2563EB; color: white; font-weight: bold; }"
    )
    select_btn = QtWidgets.QPushButton("Select")
    select_btn.setCheckable(True)
    select_btn.setFixedHeight(28)
    select_btn.setStyleSheet(
        "QPushButton:checked { background-color: #DC2626; color: white; font-weight: bold; }"
    )
    tool_group = QtWidgets.QButtonGroup(sidebar)
    tool_group.addButton(orbit_btn, 0)
    tool_group.addButton(select_btn, 1)
    tool_row.addWidget(tool_label)
    tool_row.addWidget(orbit_btn)
    tool_row.addWidget(select_btn)
    layout.addLayout(tool_row)

    def _on_tool_changed(btn_id):
        editor._set_interaction_tool("orbit" if btn_id == 0 else "select")

    tool_group.idClicked.connect(_on_tool_changed)

    # --- Mode hint label (for box/face/named) ---
    mode_hint = QtWidgets.QLabel("")
    mode_hint.setWordWrap(True)
    mode_hint.setStyleSheet("color: #6B7280; font-size: 11px; padding: 2px 4px;")
    mode_hint.setVisible(False)
    layout.addWidget(mode_hint)

    # --- Clear Selection button (for box/face) ---
    clear_sel_btn = QtWidgets.QPushButton("Clear Selection")
    clear_sel_btn.setVisible(False)
    clear_sel_btn.clicked.connect(lambda: _clear_picked())
    layout.addWidget(clear_sel_btn)

    def _clear_picked():
        editor.state.picked_node_ids.clear()
        editor._refresh()

    # --- Named selection combo (for Named mode) ---
    named_combo = QtWidgets.QComboBox()
    named_combo.setVisible(False)
    layout.addWidget(named_combo)

    def _on_named_changed(index):
        mode = editor._modes.get("named")
        if mode and editor._current_mode_name == "named":
            mode.on_combo_changed(index)

    named_combo.currentIndexChanged.connect(_on_named_changed)

    # === Plane Controls (visible only in Plane mode) ===
    plane_controls = QtWidgets.QGroupBox("Plane Controls")
    pc_layout = QtWidgets.QVBoxLayout(plane_controls)
    pc_layout.setContentsMargins(4, 4, 4, 4)
    pc_layout.setSpacing(4)

    sidebar._suppress_plane_cb = False

    # Origin X/Y/Z
    origin_spins = []
    for ax in ["X", "Y", "Z"]:
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"Origin {ax}:")
        lbl.setFixedWidth(60)
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(5)
        spin.setSingleStep(0.001)
        spin.setMinimumWidth(100)
        origin_spins.append(spin)
        row.addWidget(lbl)
        row.addWidget(spin)
        pc_layout.addLayout(row)

    def _on_origin_spin(idx):
        def _cb(value):
            if sidebar._suppress_plane_cb:
                return
            editor.state.origin[idx] = value
            plane_mode = editor._modes.get("plane")
            if plane_mode and editor._current_mode_name == "plane":
                plane_mode.update_origin_from_sidebar()
            else:
                editor._refresh()
        return _cb

    for i, spin in enumerate(origin_spins):
        spin.valueChanged.connect(_on_origin_spin(i))

    # Radius
    rad_row = QtWidgets.QHBoxLayout()
    rad_lbl = QtWidgets.QLabel("Radius:")
    rad_lbl.setFixedWidth(60)
    radius_spin = QtWidgets.QDoubleSpinBox()
    radius_spin.setRange(0, 1e6)
    radius_spin.setDecimals(4)
    radius_spin.setSingleStep(0.001)
    radius_spin.setMinimumWidth(100)
    rad_row.addWidget(rad_lbl)
    rad_row.addWidget(radius_spin)
    pc_layout.addLayout(rad_row)

    def _on_radius(value):
        if sidebar._suppress_plane_cb:
            return
        editor.state.selection_radius = value
        plane_mode = editor._modes.get("plane")
        if plane_mode and editor._current_mode_name == "plane":
            editor._refresh()
        else:
            editor._refresh()

    radius_spin.valueChanged.connect(_on_radius)

    # Normal direction buttons
    norm_lbl = QtWidgets.QLabel("Snap Normal:")
    norm_lbl.setStyleSheet("font-size: 11px; color: #6B7280; margin-top: 4px;")
    pc_layout.addWidget(norm_lbl)

    snap_row = QtWidgets.QHBoxLayout()
    for label, normal in [("+X", [1, 0, 0]), ("-X", [-1, 0, 0]),
                           ("+Y", [0, 1, 0]), ("-Y", [0, -1, 0]),
                           ("+Z", [0, 0, 1]), ("-Z", [0, 0, -1])]:
        btn = QtWidgets.QPushButton(label)
        btn.setFixedSize(38, 26)
        btn.setStyleSheet("font-size: 11px;")
        n = list(normal)
        btn.clicked.connect(lambda checked, nn=n: _snap_normal(nn))
        snap_row.addWidget(btn)
    pc_layout.addLayout(snap_row)

    flip_btn = QtWidgets.QPushButton("Flip Normal")
    flip_btn.setFixedHeight(26)
    flip_btn.clicked.connect(lambda: _snap_normal((-editor.state.normal).tolist()))
    pc_layout.addWidget(flip_btn)

    def _snap_normal(n):
        plane_mode = editor._modes.get("plane")
        if plane_mode and editor._current_mode_name == "plane":
            plane_mode.update_normal(n)
        else:
            editor.state.normal = np.array(n, dtype=np.float64)
            nn = np.linalg.norm(editor.state.normal)
            if nn > 1e-12:
                editor.state.normal /= nn
            editor._refresh()

    layout.addWidget(plane_controls)

    # === BC Type ===
    type_group = QtWidgets.QGroupBox("BC Type")
    type_layout = QtWidgets.QVBoxLayout(type_group)
    type_layout.setContentsMargins(4, 4, 4, 4)
    type_combo = QtWidgets.QComboBox()
    for t in BC_TYPES:
        type_combo.addItem(BC_TYPE_LABELS[t])
    type_layout.addWidget(type_combo)

    # Type-specific properties (stacked widget)
    props_stack = QtWidgets.QStackedWidget()

    # Fixed: no extra props
    fixed_page = QtWidgets.QWidget()
    fl = QtWidgets.QVBoxLayout(fixed_page)
    fl.addWidget(QtWidgets.QLabel("All DOFs constrained (ux=uy=uz=0)"))
    fl.addStretch()
    props_stack.addWidget(fixed_page)

    # Displacement: X/Y/Z checkboxes
    disp_page = QtWidgets.QWidget()
    dl = QtWidgets.QVBoxLayout(disp_page)
    disp_checks = []
    for label in ["Lock ux (X)", "Lock uy (Y)", "Lock uz (Z)"]:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(True)
        disp_checks.append(cb)
        dl.addWidget(cb)
    dl.addStretch()
    props_stack.addWidget(disp_page)

    def _on_disp_check(idx):
        def _cb(checked):
            editor.state.components[idx] = bool(checked)
            editor._refresh()
        return _cb
    for i, cb in enumerate(disp_checks):
        cb.stateChanged.connect(_on_disp_check(i))

    # Frictionless
    frict_page = QtWidgets.QWidget()
    frl = QtWidgets.QVBoxLayout(frict_page)
    frl.addWidget(QtWidgets.QLabel("Normal direction auto-computed\nfrom surface geometry"))
    frl.addStretch()
    props_stack.addWidget(frict_page)

    # Elastic Support: stiffness inputs
    spring_page = QtWidgets.QWidget()
    sl = QtWidgets.QFormLayout(spring_page)
    spring_inputs = []
    for label in ["kx (N/m)", "ky (N/m)", "kz (N/m)"]:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0, 1e20)
        spin.setDecimals(2)
        spin.setValue(1e8)
        spin.setMinimumWidth(120)
        spring_inputs.append(spin)
        sl.addRow(label, spin)
    props_stack.addWidget(spring_page)

    def _on_spring_k(idx):
        def _cb(value):
            editor.state.spring_k[idx] = value
        return _cb
    for i, spin in enumerate(spring_inputs):
        spin.valueChanged.connect(_on_spring_k(i))

    # Cylindrical: R/T/Z checkboxes
    cyl_page = QtWidgets.QWidget()
    cl = QtWidgets.QVBoxLayout(cyl_page)
    cyl_checks = []
    for label in ["Lock Radial", "Lock Tangential", "Lock Axial"]:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(True)
        cyl_checks.append(cb)
        cl.addWidget(cb)
    cl.addStretch()
    props_stack.addWidget(cyl_page)

    def _on_cyl_check(idx):
        def _cb(checked):
            editor.state.components[idx] = bool(checked)
            editor._refresh()
        return _cb
    for i, cb in enumerate(cyl_checks):
        cb.stateChanged.connect(_on_cyl_check(i))

    type_layout.addWidget(props_stack)
    layout.addWidget(type_group)

    def _on_type_changed(index):
        editor.state.type_idx = index
        props_stack.setCurrentIndex(index)
        if BC_TYPES[index] == "displacement":
            for i, cb in enumerate(disp_checks):
                cb.setChecked(editor.state.components[i])
        elif BC_TYPES[index] == "cylindrical":
            for i, cb in enumerate(cyl_checks):
                cb.setChecked(editor.state.components[i])
        editor._refresh()

    type_combo.currentIndexChanged.connect(_on_type_changed)

    # === Current Selection ===
    name_group = QtWidgets.QGroupBox("Current Selection")
    name_layout = QtWidgets.QFormLayout(name_group)
    name_layout.setContentsMargins(4, 4, 4, 4)
    name_edit = QtWidgets.QLineEdit(f"bc_{editor.state.counter}")
    name_layout.addRow("Name:", name_edit)
    info_label = QtWidgets.QLabel("0 nodes selected")
    info_label.setStyleSheet("color: #6B7280;")
    name_layout.addRow("", info_label)
    layout.addWidget(name_group)

    # === Action Buttons ===
    btn_layout = QtWidgets.QHBoxLayout()
    accept_btn = QtWidgets.QPushButton("Accept")
    accept_btn.setStyleSheet(
        "background-color: #2563EB; color: white; font-weight: bold; "
        "padding: 6px 16px; border-radius: 4px;"
    )
    accept_btn.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return))
    undo_btn = QtWidgets.QPushButton("Undo")
    undo_btn.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
    redo_btn = QtWidgets.QPushButton("Redo")
    redo_btn.setShortcut(QtGui.QKeySequence("Ctrl+Shift+Z"))
    btn_layout.addWidget(accept_btn)
    btn_layout.addWidget(undo_btn)
    btn_layout.addWidget(redo_btn)
    layout.addLayout(btn_layout)

    accept_btn.clicked.connect(editor.accept_current)
    undo_btn.clicked.connect(editor.undo)
    redo_btn.clicked.connect(editor.redo)

    # === Accepted BCs List ===
    list_group = QtWidgets.QGroupBox("Accepted BCs")
    list_layout = QtWidgets.QVBoxLayout(list_group)
    list_layout.setContentsMargins(4, 4, 4, 4)
    bc_list = QtWidgets.QListWidget()
    bc_list.setAlternatingRowColors(True)
    list_layout.addWidget(bc_list)

    delete_btn = QtWidgets.QPushButton("Delete Selected")
    delete_btn.setEnabled(False)
    list_layout.addWidget(delete_btn)
    layout.addWidget(list_group)

    def _on_delete():
        row = bc_list.currentRow()
        if 0 <= row < len(editor._accepted):
            editor._delete_bc(row)

    delete_btn.clicked.connect(_on_delete)
    bc_list.currentRowChanged.connect(lambda row: delete_btn.setEnabled(row >= 0))
    bc_list.itemDoubleClicked.connect(
        lambda item: editor._highlight_bc(bc_list.row(item)))

    # === Done Button ===
    done_btn = QtWidgets.QPushButton("Done")
    done_btn.setStyleSheet(
        "background-color: #059669; color: white; font-weight: bold; "
        "padding: 8px; border-radius: 4px; margin-top: 8px;"
    )
    done_btn.clicked.connect(editor._finish)
    layout.addWidget(done_btn)

    layout.addStretch()

    # Store widget references
    sidebar.type_combo = type_combo
    sidebar.props_stack = props_stack
    sidebar.name_edit = name_edit
    sidebar.info_label = info_label
    sidebar.bc_list = bc_list
    sidebar.delete_btn = delete_btn
    sidebar.mode_buttons = mode_buttons
    sidebar.named_combo = named_combo
    sidebar.disp_checks = disp_checks
    sidebar.cyl_checks = cyl_checks
    sidebar.spring_inputs = spring_inputs
    sidebar.plane_controls = plane_controls
    sidebar.origin_spins = origin_spins
    sidebar.radius_spin = radius_spin
    sidebar.clear_sel_btn = clear_sel_btn
    sidebar.mode_hint = mode_hint
    sidebar.tool_group = tool_group
    sidebar.select_btn = select_btn
    sidebar.orbit_btn = orbit_btn

    scroll.setWidget(sidebar)
    return scroll, sidebar


# ---------------------------------------------------------------------------
# Main Editor Application
# ---------------------------------------------------------------------------

class BCEditorApp:
    """Interactive BC editor with Qt sidebar and multiple selection modes."""

    def __init__(self, mesh: "Mesh"):
        self.mesh = mesh
        self.plotter = None
        self.state = EditorState()
        self._sidebar = None
        self._scroll = None

        self._accepted: list = []
        self._accepted_ids: list[np.ndarray] = []
        self._undo_stack: list[EditorCommand] = []
        self._redo_stack: list[EditorCommand] = []

        self._modes: dict[str, SelectionMode] = {}
        self._current_mode_name = "plane"
        self._current_mode: SelectionMode | None = None

        self._grid = None
        self._surface_mesh = None
        self._surface_node_ids = None
        self._surface_coords = None
        self._all_coords = None
        self._bbox_diag = 1.0
        self._max_radius = 1.0
        self._slider_lo = np.zeros(3)
        self._slider_hi = np.ones(3)
        self._sphere_sel = None
        self._sphere_acc = None
        self._finished = False

    def run(self) -> list:
        """Open the editor window and return accepted BCs when closed."""
        from turbomodal.viz import _mesh_to_pyvista

        self._precompute_geometry(_mesh_to_pyvista)
        self._setup_qt_window()
        self._setup_plotter()
        self._setup_picking()
        self._init_selection_modes()
        self._current_mode.activate()
        self._refresh()

        self._app_exec()
        return self._accepted

    def _precompute_geometry(self, mesh_to_pyvista_fn):
        self._all_coords = np.asarray(self.mesh.nodes)
        self._grid = mesh_to_pyvista_fn(self.mesh)
        self._surface_mesh = self._grid.extract_surface(algorithm="dataset_surface")

        surf_point_ids = self._surface_mesh.point_data.get("vtkOriginalPointIds", None)
        if surf_point_ids is None:
            surf_point_ids = np.array(
                self.mesh.select_nodes_by_plane(
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.0, 1e30]),
                    tolerance=1e30,
                )
            )
        else:
            surf_point_ids = np.asarray(surf_point_ids)

        exclude = set(self.mesh.left_boundary) | set(self.mesh.right_boundary)
        mask = np.array([nid not in exclude for nid in surf_point_ids])
        self._surface_node_ids = surf_point_ids[mask]
        self._surface_coords = self._all_coords[self._surface_node_ids]

        bounds = self._surface_mesh.bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        bbox_min = np.array([bounds[0], bounds[2], bounds[4]])
        bbox_max = np.array([bounds[1], bounds[3], bounds[5]])
        bbox_range = bbox_max - bbox_min
        self._bbox_diag = float(np.linalg.norm(bbox_range))
        self._slider_lo = bbox_min - 0.2 * bbox_range
        self._slider_hi = bbox_max + 0.2 * bbox_range
        self._max_radius = self._bbox_diag * 1.5

        self.state.origin = center.copy()
        self.state.selection_radius = self._max_radius

        cyl_ax = np.zeros(3)
        cyl_ax[self.mesh.rotation_axis] = 1.0
        self.state.cyl_axis = cyl_ax

        r_sel = self._bbox_diag * 0.004
        r_acc = self._bbox_diag * 0.0035
        self._sphere_sel = pv.Sphere(radius=r_sel)
        self._sphere_acc = pv.Sphere(radius=r_acc)

    def _setup_qt_window(self):
        from qtpy import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        self._app = app

        self._window = QtWidgets.QMainWindow()
        self._window.setWindowTitle("BC Editor \u2014 turbomodal")
        self._window.resize(1400, 900)

        central = QtWidgets.QWidget()
        hlayout = QtWidgets.QHBoxLayout(central)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)

        self._scroll, self._sidebar = _build_sidebar(self)

        self._vtk_frame = QtWidgets.QFrame()
        self._vtk_frame.setMinimumWidth(800)

        hlayout.addWidget(self._scroll)
        hlayout.addWidget(self._vtk_frame, stretch=1)

        self._window.setCentralWidget(central)
        self._window.closeEvent = self._on_close

        # Initialize sidebar spinbox ranges from geometry
        sb = self._sidebar
        for i, spin in enumerate(sb.origin_spins):
            spin.setRange(float(self._slider_lo[i]), float(self._slider_hi[i]))
            spin.setValue(float(self.state.origin[i]))
            spin.setSingleStep(float(self._bbox_diag * 0.01))
        sb.radius_spin.setRange(0, float(self._max_radius))
        sb.radius_spin.setValue(float(self._max_radius))
        sb.radius_spin.setSingleStep(float(self._bbox_diag * 0.01))

    def _setup_plotter(self):
        from pyvistaqt import QtInteractor
        from qtpy import QtWidgets

        vtk_layout = QtWidgets.QVBoxLayout(self._vtk_frame)
        vtk_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self._vtk_frame)
        vtk_layout.addWidget(self.plotter.interactor)

        self.plotter.add_mesh(
            self._surface_mesh, color="lightgray", opacity=0.35,
            show_edges=True, edge_color="#9CA3AF", line_width=0.5,
            name="_base_mesh", pickable=True,
        )

        # Lighting
        try:
            self.plotter.remove_all_lights()
        except Exception:
            pass
        self.plotter.add_light(pv.Light(position=(2, 1, 3), focal_point=(0, 0, 0), intensity=0.8))
        self.plotter.add_light(pv.Light(position=(-2, -1, 1), focal_point=(0, 0, 0), intensity=0.3))
        self.plotter.add_light(pv.Light(position=(0, -3, 2), focal_point=(0, 0, 0), intensity=0.4))

        self.plotter.add_axes()
        self.plotter.reset_camera()

    def _setup_picking(self):
        """Store the default orbit style and build a custom selection style."""
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser as vtkInteractorStyle

        self._iren = self.plotter.interactor.GetRenderWindow().GetInteractor()
        self._orbit_style = self._iren.GetInteractorStyle()
        self._interaction_tool = "orbit"
        self._pick_start = None
        self._rubber_band_actor = None

        # Use bare vtkInteractorStyle — it has NO default mouse behaviour,
        # so left-click will never orbit.  We handle everything ourselves.
        select_style = vtkInteractorStyle()
        editor = self

        def _on_left_down(obj, event):
            editor._pick_start = editor._iren.GetEventPosition()

        def _on_mouse_move(obj, event):
            if editor._pick_start is not None and editor._current_mode_name == "box":
                current = editor._iren.GetEventPosition()
                editor._draw_rubber_band(editor._pick_start, current)

        def _on_left_up(obj, event):
            if editor._pick_start is None:
                return
            end = editor._iren.GetEventPosition()
            start = editor._pick_start
            editor._pick_start = None
            editor._clear_rubber_band()

            dx = abs(end[0] - start[0])
            dy = abs(end[1] - start[1])

            if editor._current_mode_name == "box" and dx > 5 and dy > 5:
                editor._box_select(
                    min(start[0], end[0]), min(start[1], end[1]),
                    max(start[0], end[0]), max(start[1], end[1]),
                )
            elif editor._current_mode_name == "face":
                editor._face_pick_at_screen(end[0], end[1])

        # Allow middle-button pan and scroll zoom in select mode
        def _on_middle_down(obj, event):
            obj.OnMiddleButtonDown()

        def _on_middle_up(obj, event):
            obj.OnMiddleButtonUp()

        def _on_scroll_fwd(obj, event):
            obj.OnMouseWheelForward()

        def _on_scroll_bwd(obj, event):
            obj.OnMouseWheelBackward()

        select_style.AddObserver("LeftButtonPressEvent", _on_left_down)
        select_style.AddObserver("MouseMoveEvent", _on_mouse_move)
        select_style.AddObserver("LeftButtonReleaseEvent", _on_left_up)
        select_style.AddObserver("MiddleButtonPressEvent", _on_middle_down)
        select_style.AddObserver("MiddleButtonReleaseEvent", _on_middle_up)
        select_style.AddObserver("MouseWheelForwardEvent", _on_scroll_fwd)
        select_style.AddObserver("MouseWheelBackwardEvent", _on_scroll_bwd)

        self._select_style = select_style

    def _set_interaction_tool(self, tool: str):
        """Switch between 'orbit' and 'select' interaction tools."""
        if tool == self._interaction_tool:
            return
        self._interaction_tool = tool
        if tool == "orbit":
            self._iren.SetInteractorStyle(self._orbit_style)
        else:
            self._iren.SetInteractorStyle(self._select_style)
        # Update button state
        sb = self._sidebar
        if sb:
            sb.tool_group.blockSignals(True)
            sb.orbit_btn.setChecked(tool == "orbit")
            sb.select_btn.setChecked(tool == "select")
            sb.tool_group.blockSignals(False)

    def _draw_rubber_band(self, start, current):
        """Draw a 2D rectangle overlay on the viewport."""
        # Use a simple approach: draw a pv.Rectangle in screen space
        # Actually, use VTK 2D actor for an overlay rectangle
        self._clear_rubber_band()
        try:
            from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D, vtkCoordinate
            from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
            import vtkmodules.vtkCommonDataModel as vtk_data
            import vtkmodules.vtkCommonCore as vtk_core

            x0, y0 = start
            x1, y1 = current

            points = vtk_core.vtkPoints()
            points.InsertNextPoint(x0, y0, 0)
            points.InsertNextPoint(x1, y0, 0)
            points.InsertNextPoint(x1, y1, 0)
            points.InsertNextPoint(x0, y1, 0)

            lines = vtk_data.vtkCellArray()
            lines.InsertNextCell(5)
            for i in [0, 1, 2, 3, 0]:
                lines.InsertCellPoint(i)

            poly = vtk_data.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)

            coord = vtkCoordinate()
            coord.SetCoordinateSystemToDisplay()

            mapper = vtkPolyDataMapper2D()
            mapper.SetInputData(poly)
            mapper.SetTransformCoordinate(coord)

            actor = vtkActor2D()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.22, 0.39, 0.93)  # blue
            actor.GetProperty().SetLineWidth(2)

            self.plotter.renderer.AddActor2D(actor)
            self._rubber_band_actor = actor
            self.plotter.render()
        except Exception:
            pass

    def _clear_rubber_band(self):
        """Remove the rubber band rectangle overlay."""
        if self._rubber_band_actor is not None:
            try:
                self.plotter.renderer.RemoveActor2D(self._rubber_band_actor)
                self.plotter.render()
            except Exception:
                pass
            self._rubber_band_actor = None

    def _face_pick_at_screen(self, sx, sy):
        """Pick the front-visible face via screen-space cell-center projection."""
        face_mode = self._modes.get("face")
        if not face_mode or self._current_mode_name != "face":
            return

        renderer = self.plotter.renderer
        cam_pos = np.array(renderer.GetActiveCamera().GetPosition())
        surface = face_mode._surface
        normals = surface.cell_normals
        n_cells = surface.n_cells
        excluded = face_mode._excluded_cells

        best_cell = -1
        best_dist_sq = float("inf")

        for cid in range(n_cells):
            if cid in excluded:
                continue
            center = np.mean(surface.points[surface.get_cell(cid).point_ids], axis=0)
            view_dir = center - cam_pos
            # Back-face cull
            if np.dot(normals[cid], view_dir) >= 0:
                continue

            renderer.SetWorldPoint(center[0], center[1], center[2], 1.0)
            renderer.WorldToDisplay()
            disp = renderer.GetDisplayPoint()
            dx = disp[0] - sx
            dy = disp[1] - sy
            d2 = dx * dx + dy * dy
            if d2 < best_dist_sq:
                best_dist_sq = d2
                best_cell = cid

        if best_cell < 0 or best_dist_sq > 2500:  # >50px
            return

        print(f"[PICK] click ({sx},{sy}): cell {best_cell}, "
              f"normal={normals[best_cell]}, dist={best_dist_sq**0.5:.1f}px")
        face_mode.flood_fill_from_cell(best_cell)

    def _box_select(self, x0, y0, x1, y1):
        """Select surface nodes inside a screen-space rectangle."""
        renderer = self.plotter.renderer
        coords = self._surface_coords
        surface_ids = self._surface_node_ids

        # Project all surface points to screen coordinates
        new_ids = set()
        for i, pt in enumerate(coords):
            renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            renderer.WorldToDisplay()
            disp = renderer.GetDisplayPoint()
            sx, sy = disp[0], disp[1]
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                new_ids.add(int(surface_ids[i]))

        if new_ids:
            self.state.picked_node_ids |= new_ids
            self._refresh()

    def _init_selection_modes(self):
        self._modes = {
            "plane": PlaneSelectionMode(self),
            "box": BoxSelectionMode(self),
            "named": NamedSelectionMode(self),
            "face": FaceSelectionMode(self),
        }
        self._current_mode_name = "plane"
        self._current_mode = self._modes["plane"]

    def _switch_selection_mode(self, mode_name: str):
        if mode_name == self._current_mode_name:
            return
        if self._current_mode is not None:
            self._current_mode.deactivate()
        self._current_mode_name = mode_name
        self._current_mode = self._modes[mode_name]
        self._current_mode.activate()

        # Auto-switch interaction tool based on mode
        if mode_name in ("box", "face"):
            self._set_interaction_tool("select")
        else:
            self._set_interaction_tool("orbit")

        self._refresh()

    def _refresh(self):
        if self.plotter is None:
            return

        selected_ids = self._current_mode.get_selected_ids()
        n_selected = len(selected_ids)

        # Current selection glyphs
        if n_selected > 0:
            pts = pv.PolyData(self._all_coords[selected_ids])
            glyphs = pts.glyph(geom=self._sphere_sel, scale=False, orient=False)
            self.plotter.add_mesh(glyphs, name="current_sel", color="#38BDF8")
        else:
            try:
                self.plotter.remove_actor("current_sel")
            except (KeyError, ValueError):
                pass

        # Accepted BC glyphs
        for i, ids in enumerate(self._accepted_ids):
            bc = self._accepted[i]
            color = BC_COLORS.get(bc.type, _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)])
            if len(ids) > 0:
                pts = pv.PolyData(self._all_coords[ids])
                glyphs = pts.glyph(geom=self._sphere_acc, scale=False, orient=False)
                self.plotter.add_mesh(glyphs, name=f"accepted_{i}", color=color)

        # Mode-specific visuals
        self._current_mode.on_refresh()

        # Update sidebar
        if self._sidebar is not None:
            ct = BC_TYPES[self.state.type_idx]
            n_dofs = self._constrained_dof_count(n_selected, ct)
            self._sidebar.info_label.setText(f"{n_selected} nodes, {n_dofs} DOFs")
            self._sidebar.name_edit.setText(f"bc_{self.state.counter}")

        self.plotter.render()

        # Re-ensure the correct interactor style is set (some PyVista
        # operations like add_mesh can reset it)
        if self._interaction_tool == "select":
            current = self._iren.GetInteractorStyle()
            if current is not self._select_style:
                self._iren.SetInteractorStyle(self._select_style)

    def _constrained_dof_count(self, n_nodes: int, bc_type: str) -> int:
        if bc_type == "fixed":
            return n_nodes * 3
        elif bc_type in ("displacement", "cylindrical"):
            return n_nodes * sum(self.state.components)
        elif bc_type == "elastic_support":
            return n_nodes * sum(1 for k in self.state.spring_k if k > 0)
        return n_nodes

    def accept_current(self):
        from turbomodal.solver import BoundaryCondition

        selected_ids = self._current_mode.get_selected_ids()
        if len(selected_ids) == 0:
            return

        name = self._sidebar.name_edit.text() if self._sidebar else f"bc_{self.state.counter}"
        ct = BC_TYPES[self.state.type_idx]
        radius = self.state.selection_radius

        bc = BoundaryCondition(
            name=name, type=ct,
            plane_point=self.state.origin.copy(),
            plane_normal=self.state.normal.copy(),
            constrained_components=tuple(self.state.components),
            node_ids=selected_ids.tolist(),
            selection_radius=radius if radius < self._max_radius * 0.99 else None,
        )
        if ct == "elastic_support":
            bc.spring_stiffness = np.array(self.state.spring_k, dtype=np.float64)
        elif ct == "cylindrical":
            bc.cylinder_axis = self.state.cyl_axis.copy()
            bc.cylinder_origin = self.state.cyl_origin.copy()

        cmd = EditorCommand(
            action="accept", bc=bc, node_ids=selected_ids.copy(),
            index=len(self._accepted),
        )
        self._execute_command(cmd)
        self._redo_stack.clear()

        self.state.counter += 1
        self.state.type_idx = 0
        self.state.components = [True, True, True]
        self.state.picked_node_ids.clear()
        if self._sidebar:
            self._sidebar.type_combo.setCurrentIndex(0)
        self._refresh()

    def _execute_command(self, cmd: EditorCommand):
        if cmd.action == "accept":
            self._accepted.append(cmd.bc)
            self._accepted_ids.append(cmd.node_ids)
            self._update_bc_list()
        elif cmd.action == "delete":
            if 0 <= cmd.index < len(self._accepted):
                self._accepted.pop(cmd.index)
                self._accepted_ids.pop(cmd.index)
                try:
                    self.plotter.remove_actor(f"accepted_{cmd.index}")
                except (KeyError, ValueError):
                    pass
                self._update_bc_list()
        self._undo_stack.append(cmd)

    def undo(self):
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        if cmd.action == "accept":
            if self._accepted and self._accepted[-1] is cmd.bc:
                self._accepted.pop()
                self._accepted_ids.pop()
                try:
                    self.plotter.remove_actor(f"accepted_{cmd.index}")
                except (KeyError, ValueError):
                    pass
                self.state.counter = max(1, self.state.counter - 1)
        elif cmd.action == "delete":
            self._accepted.insert(cmd.index, cmd.bc)
            self._accepted_ids.insert(cmd.index, cmd.node_ids)
        self._redo_stack.append(cmd)
        self._update_bc_list()
        self._refresh()

    def redo(self):
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self._execute_command(cmd)
        self._refresh()

    def _delete_bc(self, index: int):
        if 0 <= index < len(self._accepted):
            bc = self._accepted[index]
            ids = self._accepted_ids[index]
            cmd = EditorCommand(action="delete", bc=bc, node_ids=ids, index=index)
            self._execute_command(cmd)
            self._redo_stack.clear()
            self._refresh()

    def _highlight_bc(self, index: int):
        if 0 <= index < len(self._accepted_ids):
            ids = self._accepted_ids[index]
            if len(ids) > 0:
                pts = pv.PolyData(self._all_coords[ids])
                glyphs = pts.glyph(geom=self._sphere_sel, scale=False, orient=False)
                self.plotter.add_mesh(glyphs, name="highlight_temp", color="white")
                self.plotter.render()

    def _update_bc_list(self):
        if self._sidebar is None:
            return
        from qtpy import QtWidgets, QtGui
        bc_list = self._sidebar.bc_list
        bc_list.clear()
        for i, bc in enumerate(self._accepted):
            n = len(self._accepted_ids[i])
            color = BC_COLORS.get(bc.type, _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)])
            tp = bc.type.upper()
            if bc.type == "displacement":
                ax = [c for c, v in zip("xyz", bc.constrained_components) if v]
                tp += f"({','.join(ax)})"
            elif bc.type == "cylindrical":
                ax = [c for c, v in zip("RTZ", bc.constrained_components) if v]
                tp += f"({','.join(ax)})"
            elif bc.type == "elastic_support":
                tp = "SPRING"
            item = QtWidgets.QListWidgetItem(f"{bc.name}: {tp}  [{n} nodes]")
            item.setForeground(QtGui.QColor(color))
            bc_list.addItem(item)

    def _finish(self):
        self._finished = True
        self._window.close()

    def _on_close(self, event):
        if self._current_mode is not None:
            self._current_mode.deactivate()
        if self.plotter is not None:
            self.plotter.close()
        event.accept()

    def _app_exec(self):
        self._window.show()
        self._app.exec()
