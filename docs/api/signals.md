# turbomodal Signals API Reference

This document covers the signal generation subsystem: virtual sensor arrays
(`turbomodal.sensors`), noise models (`turbomodal.noise`), and the end-to-end
signal generation pipeline (`turbomodal.signal_gen`).

---

## turbomodal.sensors -- Virtual Sensor Array

### SensorType

Enumeration of supported virtual sensor types.

```python
class SensorType(Enum):
    STRAIN_GAUGE = "strain_gauge"
    BTT_PROBE = "btt_probe"
    CASING_ACCELEROMETER = "casing_accelerometer"
```

### SensorLocation

Placement and characteristics of a single virtual sensor.

```python
@dataclass
class SensorLocation:
    sensor_type: SensorType
    position: NDArray
    direction: NDArray
    label: str = ""
    bandwidth: float = 50_000.0
    sensitivity: float = 1.0
    noise_floor: float = 1e-9
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensor_type` | `SensorType` | required | Kind of sensor |
| `position` | `NDArray (3,)` | required | World-space coordinates |
| `direction` | `NDArray (3,)` | required | Unit vector defining measurement direction |
| `label` | `str` | `""` | Human-readable identifier (e.g. `"BTT_probe_03"`) |
| `bandwidth` | `float` | `50000.0` | Sensor bandwidth in Hz |
| `sensitivity` | `float` | `1.0` | Output per unit displacement |
| `noise_floor` | `float` | `1e-9` | Minimum detectable signal level |

### SensorArrayConfig

Configuration for an array of virtual sensors.

```python
@dataclass
class SensorArrayConfig:
    sensors: list[SensorLocation] = field(default_factory=list)
    sample_rate: float = 100_000.0
    duration: float = 1.0
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sensors` | `list[SensorLocation]` | `[]` | Ordered list of sensor locations |
| `sample_rate` | `float` | `100000.0` | Sampling rate in Hz |
| `duration` | `float` | `1.0` | Total acquisition duration in seconds |

#### SensorArrayConfig.default_btt_array (static method)

Create a BTT probe array with probes evenly spaced circumferentially.

```python
@staticmethod
def default_btt_array(
    num_probes: int,
    casing_radius: float,
    axial_positions: Sequence[float] | NDArray,
    sample_rate: float = 500_000.0,
    duration: float = 1.0,
) -> SensorArrayConfig
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_probes` | `int` | required | Number of probes per axial station |
| `casing_radius` | `float` | required | Radial distance from rotation axis to probe tip |
| `axial_positions` | `Sequence[float] \| NDArray` | required | Axial (z) coordinates of each ring |
| `sample_rate` | `float` | `500000.0` | Acquisition sampling rate in Hz |
| `duration` | `float` | `1.0` | Acquisition duration in seconds |

**Returns:** `SensorArrayConfig` with BTT probes measuring radial displacement.

#### SensorArrayConfig.default_strain_gauge_array (static method)

Create a strain-gauge array at specified radial positions.

```python
@staticmethod
def default_strain_gauge_array(
    num_gauges: int,
    radial_positions: Sequence[float] | NDArray,
    sample_rate: float = 50_000.0,
    duration: float = 1.0,
) -> SensorArrayConfig
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_gauges` | `int` | required | Number of gauges per radial ring |
| `radial_positions` | `Sequence[float] \| NDArray` | required | Radial distance of each ring |
| `sample_rate` | `float` | `50000.0` | Sampling rate in Hz |
| `duration` | `float` | `1.0` | Duration in seconds |

**Returns:** `SensorArrayConfig` with strain gauges measuring tangential displacement.

### VirtualSensorArray

Samples FE mode shapes at discrete sensor locations and synthesizes time-domain
signals.

```python
class VirtualSensorArray(mesh, config: SensorArrayConfig)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `config` | `SensorArrayConfig` | The sensor array configuration |
| `n_sensors` | `int` | Number of sensors in the array |
| `interpolation_matrix` | `csr_matrix (n_sensors, n_dof)` | Sparse interpolation matrix (built lazily) |

**Methods:**

#### build_interpolation_matrix

```python
def build_interpolation_matrix(self) -> sparse.csr_matrix
```

Build a sparse `(n_sensors, 3 * n_nodes)` interpolation matrix. For each sensor,
the nearest mesh node is located and the displacement DOFs are projected onto
the sensor measurement direction.

#### sample_mode_shape

```python
def sample_mode_shape(self, mode_shape: NDArray) -> NDArray
```

**Parameters:**

- `mode_shape` : complex mode-shape vector of length `n_dof` (i.e. `3 * n_nodes`).

**Returns:** Complex array of shape `(n_sensors,)`.

#### generate_time_signal

```python
def generate_time_signal(
    self,
    modal_results: Sequence,
    rpm: float,
    amplitudes: NDArray | None = None,
    noise_config: NoiseConfig | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray
```

Synthesize a multi-channel time-domain signal from modal results. The signal is:

    x_s(t) = sum_r A_r * Re(phi_s_r * exp(j * omega_r * t))

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modal_results` | `Sequence[ModalResult]` | required | One per harmonic index |
| `rpm` | `float` | required | Rotational speed in RPM |
| `amplitudes` | `NDArray \| None` | `None` | Per-mode amplitudes (None = unit) |
| `noise_config` | `NoiseConfig \| None` | `None` | Noise to apply to signal |
| `rng` | `Generator \| None` | `None` | Random generator for reproducibility |

**Returns:** Real array of shape `(n_sensors, n_samples)`.

**Example:**

```python
import turbomodal as tm

config = tm.SensorArrayConfig.default_btt_array(
    num_probes=8, casing_radius=0.25, axial_positions=[0.0, 0.05]
)
sensor_array = tm.VirtualSensorArray(mesh, config)

signal = sensor_array.generate_time_signal(
    modal_results=results, rpm=10000,
    noise_config=tm.NoiseConfig(gaussian_snr_db=30)
)
print(signal.shape)  # (16, 500000)
```

---

## turbomodal.noise -- Noise Models

### NoiseConfig

Configuration for all noise effects applied to a synthetic signal.

```python
@dataclass
class NoiseConfig:
    gaussian_snr_db: float = 40.0
    harmonic_interference: list[dict] = field(default_factory=list)
    drift_rate: float = 0.0
    drift_type: str = "none"
    bandwidth_hz: float = 0.0
    filter_order: int = 4
    adc_bits: int = 0
    adc_range: float = 10.0
    dropout_probability: float = 0.0
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gaussian_snr_db` | `float` | `40.0` | SNR in dB for additive white Gaussian noise. Set to `0.0` or `inf` to disable. |
| `harmonic_interference` | `list[dict]` | `[]` | List of interference tones. Each dict has keys `frequency_hz`, `amplitude_ratio`, `phase_deg`. |
| `drift_rate` | `float` | `0.0` | Drift magnitude per second |
| `drift_type` | `str` | `"none"` | `"none"`, `"linear"`, or `"random_walk"` |
| `bandwidth_hz` | `float` | `0.0` | Butterworth low-pass cutoff (0 = disabled) |
| `filter_order` | `int` | `4` | Butterworth filter order |
| `adc_bits` | `int` | `0` | ADC quantization resolution (0 = disabled) |
| `adc_range` | `float` | `10.0` | Full-scale ADC voltage range |
| `dropout_probability` | `float` | `0.0` | Per-sample probability of replacing with zero |

### apply_noise

Apply all configured noise effects in sequence.

```python
def apply_noise(
    signal: NDArray,
    config: NoiseConfig,
    sample_rate: float,
    rng: np.random.Generator | None = None,
) -> NDArray
```

Processing order:
1. Harmonic interference
2. Additive Gaussian noise
3. Bandwidth limiting (Butterworth low-pass)
4. Sensor drift
5. ADC quantization
6. Signal dropout

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `NDArray` | required | `(n_samples,)` or `(n_channels, n_samples)` real array |
| `config` | `NoiseConfig` | required | Noise configuration |
| `sample_rate` | `float` | required | Sampling rate in Hz |
| `rng` | `Generator \| None` | `None` | Random generator |

**Returns:** Corrupted signal with the same shape.

### Individual Noise Functions

- `add_gaussian_noise(signal, snr_db, rng=None)` -- Add white Gaussian noise at
  specified SNR (dB).
- `add_harmonic_interference(signal, harmonics, sample_rate, rng=None)` -- Add
  sinusoidal interference tones.
- `apply_bandwidth_limit(signal, bandwidth_hz, sample_rate, order=4)` -- Butterworth
  low-pass filter.
- `apply_drift(signal, drift_rate, drift_type, sample_rate, rng=None)` -- Add
  sensor drift (`"linear"` or `"random_walk"`).
- `apply_quantization(signal, adc_bits, adc_range=10.0)` -- Simulate ADC quantization.
- `apply_dropout(signal, probability, rng=None)` -- Simulate intermittent sensor
  dropouts.

All functions accept both 1-D `(n_samples,)` and 2-D `(n_channels, n_samples)`
signal arrays and preserve the input shape.

---

## turbomodal.signal_gen -- Signal Generation Pipeline

### SignalGenerationConfig

```python
@dataclass
class SignalGenerationConfig:
    sample_rate: float = 100000.0
    duration: float = 1.0
    num_revolutions: int = 0
    seed: int = 42
    amplitude_mode: str = "unit"
    amplitude_scale: float = 1e-6
    max_frequency: float = 0.0
    max_modes_per_harmonic: int = 0
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sample_rate` | `float` | `100000.0` | Sampling rate in Hz |
| `duration` | `float` | `1.0` | Duration in seconds |
| `num_revolutions` | `int` | `0` | If > 0, overrides duration |
| `seed` | `int` | `42` | Random seed |
| `amplitude_mode` | `str` | `"unit"` | `"unit"`, `"forced_response"`, or `"random"` |
| `amplitude_scale` | `float` | `1e-6` | Base amplitude in metres |
| `max_frequency` | `float` | `0.0` | Max frequency filter (0 = all modes) |
| `max_modes_per_harmonic` | `int` | `0` | Max modes per harmonic (0 = all) |

### generate_signals_for_condition

```python
def generate_signals_for_condition(
    sensor_array,
    modal_results: list,
    rpm: float,
    config: SignalGenerationConfig,
    noise_config=None,
    forced_response_result=None,
) -> dict
```

**Returns:** dict with keys `"signals"`, `"time"`, `"clean_signals"`.

### generate_dataset_signals

```python
def generate_dataset_signals(
    mesh,
    modal_results_per_condition: list[list],
    conditions: list,
    sensor_array,
    config: SignalGenerationConfig = SignalGenerationConfig(),
    noise_config=None,
    forced_response_results: Optional[list] = None,
    verbose: int = 1,
) -> dict
```

**Returns:** dict with keys `"signals"` `(n_cond, n_sensors, n_samples)`,
`"clean_signals"`, `"conditions"`, `"sample_rate"`, `"time"`.

---

## See also

- [Core API](core.md) -- Mesh, Material, Solver
- [Data API](data.md) -- Dataset export and parametric sweeps
- [ML API](ml.md) -- Feature extraction from signals
- [Optimization API](optimization.md) -- Sensor placement optimization
