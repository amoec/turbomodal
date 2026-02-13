# turbomodal Data API Reference

This document covers the data management modules: HDF5 dataset export/import
(`turbomodal.dataset`) and parametric sweep orchestration
(`turbomodal.parametric`).

---

## turbomodal.dataset -- HDF5 Dataset

### OperatingCondition

A single operating condition for parametric modal analysis.

```python
@dataclass
class OperatingCondition:
    condition_id: int
    rpm: float
    temperature: float = 293.15
    pressure_ratio: float = 1.0
    inlet_distortion: float = 0.0
    tip_clearance: float = 0.0
    mistuning_pattern: Optional[np.ndarray] = None
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `condition_id` | `int` | required | Unique integer identifier |
| `rpm` | `float` | required | Rotation speed in RPM |
| `temperature` | `float` | `293.15` | Bulk temperature in Kelvin (20 C) |
| `pressure_ratio` | `float` | `1.0` | Compressor/turbine pressure ratio |
| `inlet_distortion` | `float` | `0.0` | Non-dimensional inlet distortion amplitude |
| `tip_clearance` | `float` | `0.0` | Tip clearance in metres |
| `mistuning_pattern` | `ndarray \| None` | `None` | Per-blade frequency deviation array (length N) |

### DatasetConfig

Configuration for HDF5 dataset export.

```python
@dataclass
class DatasetConfig:
    output_path: str = "turbomodal_dataset.h5"
    num_modes_per_harmonic: int = 10
    include_mode_shapes: bool = True
    include_forced_response: bool = False
    compression: str = "gzip"
    compression_level: int = 4
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_path` | `str` | `"turbomodal_dataset.h5"` | HDF5 output file path |
| `num_modes_per_harmonic` | `int` | `10` | Number of modes stored per harmonic index |
| `include_mode_shapes` | `bool` | `True` | Whether to store full complex mode shape arrays |
| `include_forced_response` | `bool` | `False` | Whether to store forced response data |
| `compression` | `str` | `"gzip"` | HDF5 compression filter (`"gzip"`, `"lzf"`) |
| `compression_level` | `int` | `4` | Compression level (1-9 for gzip) |

### HDF5 File Layout

The exported HDF5 file has the following group structure:

```
/mesh/nodes              (n_nodes, 3)  float64
/mesh/elements           (n_elem, 10)  int32
/mesh/num_sectors        scalar int    (stored as attribute)

/conditions              structured array with fields:
                         condition_id, rpm, temperature,
                         pressure_ratio, inlet_distortion, tip_clearance

/modes/eigenvalues/{cond_id}       (n_harmonics, n_modes) float64
/modes/harmonic_index/{cond_id}    (n_harmonics,) int32
/modes/whirl_direction/{cond_id}   (n_harmonics, n_modes) int32
/modes/mode_shapes/{cond_id}       (n_harmonics, n_modes, n_dof) complex128
                                   (only if include_mode_shapes is True)

/mistuning/{cond_id}               (n_sectors,) float64
                                   (only for conditions with mistuning)
```

Global attributes: `turbomodal_version`, `num_conditions`,
`num_modes_per_harmonic`, `include_mode_shapes`.

### export_modal_results

Export modal analysis results for multiple operating conditions to HDF5.

```python
def export_modal_results(
    path: str | Path,
    mesh: Any,
    conditions: list[OperatingCondition],
    all_results: dict[int, list],
    config: DatasetConfig | None = None,
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | required | Output HDF5 file path |
| `mesh` | `Mesh` | required | Mesh object (must expose `.nodes`, `.elements`, `.num_sectors`) |
| `conditions` | `list[OperatingCondition]` | required | List of solved operating conditions |
| `all_results` | `dict[int, list]` | required | Mapping from condition_id to list of `ModalResult` |
| `config` | `DatasetConfig \| None` | `None` | Dataset configuration (None = defaults) |

**Example:**

```python
import turbomodal as tm

conditions = [
    tm.OperatingCondition(condition_id=0, rpm=5000),
    tm.OperatingCondition(condition_id=1, rpm=10000),
]

all_results = {}
for cond in conditions:
    all_results[cond.condition_id] = tm.solve(mesh, mat, rpm=cond.rpm)

tm.export_modal_results(
    "dataset.h5", mesh, conditions, all_results,
    config=tm.DatasetConfig(num_modes_per_harmonic=10)
)
```

### load_modal_results

Load modal analysis results from an HDF5 dataset.

```python
def load_modal_results(
    path: str | Path,
) -> tuple[dict[str, Any], list[OperatingCondition], dict[int, dict[str, Any]]]
```

**Parameters:**

- `path` : path to the HDF5 file written by `export_modal_results`.

**Returns:** A 3-tuple:

1. `mesh_data` : dict with keys `"nodes"` (ndarray), `"elements"` (ndarray),
   `"num_sectors"` (int).
2. `conditions` : list of `OperatingCondition` reconstructed from the file.
3. `results_dict` : mapping from condition_id to dict containing:
   - `"eigenvalues"` : `(n_harmonics, n_modes)` float64
   - `"harmonic_index"` : `(n_harmonics,)` int32
   - `"whirl_direction"` : `(n_harmonics, n_modes)` int32
   - `"mode_shapes"` : `(n_harmonics, n_modes, n_dof)` complex128 (if stored)
   - `"mistuning_pattern"` : `(n_sectors,)` float64 (if stored)

**Raises:** `FileNotFoundError` if path does not exist.

**Example:**

```python
mesh_data, conditions, results_dict = tm.load_modal_results("dataset.h5")
print(f"Loaded {len(conditions)} conditions")
for cid, entry in results_dict.items():
    print(f"  Condition {cid}: {entry['eigenvalues'].shape}")
```

---

## turbomodal.parametric -- Parametric Sweep

### ParametricRange

Defines the sweep range for a single operating parameter.

```python
@dataclass
class ParametricRange:
    name: str
    low: float
    high: float
    log_scale: bool = False
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Parameter name: `"rpm"`, `"temperature"`, `"pressure_ratio"`, `"inlet_distortion"`, or `"tip_clearance"` |
| `low` | `float` | required | Lower bound |
| `high` | `float` | required | Upper bound |
| `log_scale` | `bool` | `False` | If True, sample uniformly in log-space |

### ParametricSweepConfig

Configuration for a parametric sweep study.

```python
@dataclass
class ParametricSweepConfig:
    ranges: list[ParametricRange] = field(default_factory=list)
    num_samples: int = 1000
    sampling_method: str = "lhs"
    seed: int = 42
    num_modes: int = 10
    include_mistuning: bool = False
    mistuning_sigma: float = 0.02
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ranges` | `list[ParametricRange]` | `[]` | Swept parameters |
| `num_samples` | `int` | `1000` | Total number of LHS samples |
| `sampling_method` | `str` | `"lhs"` | Sampling strategy (currently only `"lhs"`) |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `num_modes` | `int` | `10` | Modes per harmonic index |
| `include_mistuning` | `bool` | `False` | Apply random blade-to-blade mistuning |
| `mistuning_sigma` | `float` | `0.02` | Standard deviation of fractional frequency deviations |

### generate_conditions

Generate operating conditions by Latin Hypercube Sampling.

```python
def generate_conditions(
    config: ParametricSweepConfig,
) -> list[OperatingCondition]
```

**Parameters:**

- `config` : sweep configuration with parameter ranges and sample count.

**Returns:** list of `OperatingCondition`, one per sample point.

**Raises:** `ValueError` if a parameter name is not recognized or no ranges
are specified.

### run_parametric_sweep

Execute a parametric sweep over operating conditions and optionally export to HDF5.

```python
def run_parametric_sweep(
    mesh: Any,
    base_material: Material,
    config: ParametricSweepConfig,
    dataset_config: DatasetConfig | None = None,
    damping: Any | None = None,
    fluid: FluidConfig | None = None,
    verbose: int = 1,
) -> str
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mesh` | `Mesh` | required | Sector mesh |
| `base_material` | `Material` | required | Reference material (temperature-dependent) |
| `config` | `ParametricSweepConfig` | required | Sweep configuration |
| `dataset_config` | `DatasetConfig \| None` | `None` | HDF5 export settings (None = no export) |
| `damping` | `Any \| None` | `None` | Optional DampingConfig |
| `fluid` | `FluidConfig \| None` | `None` | Optional FluidConfig |
| `verbose` | `int` | `1` | 0=silent, 1=progress, 2=per-condition detail |

**Returns:** Path to the HDF5 file if `dataset_config` is provided, otherwise `""`.

For each condition the routine:
1. Adjusts material stiffness for the condition temperature via
   `Material.at_temperature`.
2. Creates a `CyclicSymmetrySolver` and solves for modal results.
3. Optionally runs the Fundamental Mistuning Model (`FMMSolver`) when
   `config.include_mistuning` is True.
4. Collects results keyed by condition_id.

**Example:**

```python
sweep_config = tm.ParametricSweepConfig(
    ranges=[
        tm.ParametricRange("rpm", 3000, 15000),
        tm.ParametricRange("temperature", 293, 1073),
    ],
    num_samples=500,
    num_modes=10,
    include_mistuning=True,
    mistuning_sigma=0.03,
)

ds_config = tm.DatasetConfig(output_path="sweep_dataset.h5")

output_path = tm.run_parametric_sweep(
    mesh, mat, sweep_config,
    dataset_config=ds_config, verbose=1,
)
print(f"Dataset saved to: {output_path}")
```

---

## See also

- [Core API](core.md) -- Mesh, Material, Solver
- [Signals API](signals.md) -- Sensor array and signal generation
- [Analysis API](analysis.md) -- Campbell and ZZENF visualization
- [ML API](ml.md) -- Feature extraction and model training
