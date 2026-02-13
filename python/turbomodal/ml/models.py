"""Model implementations for the complexity ladder (Tiers 1-6).

Each tier provides a model class that implements the ModeIDModel protocol
defined in ``turbomodal.ml.pipeline``.  The complexity ladder trains tiers
in order, stopping at the first that meets performance targets.

Tier 1 -- LinearModeIDModel   (scikit-learn: LogisticRegression + Ridge)
Tier 2 -- TreeModeIDModel     (XGBoost with RandomForest fallback)
Tier 3 -- SVMModeIDModel      (scikit-learn: SVC + SVR with scaling)
Tier 4 -- ShallowNNModeIDModel(PyTorch: 2-hidden-layer multi-task net)
Tier 5 -- CNNModeIDModel      (PyTorch: 1-D CNN on spectral features)
Tier 6 -- TemporalModeIDModel (PyTorch: Conv + BiLSTM on sequences)
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common label encoding utilities
# ---------------------------------------------------------------------------


def _encode_mode_labels(nd: np.ndarray, nc: np.ndarray) -> np.ndarray:
    """Encode (ND, NC) pairs as integer class labels.

    Parameters
    ----------
    nd : numpy.ndarray
        Nodal diameter values (int).
    nc : numpy.ndarray
        Nodal circle values (int).

    Returns
    -------
    numpy.ndarray
        Unique integer label per (ND, NC) pair.  Uses ``nd * 100 + nc``
        which is valid for nd < 100 and nc < 100.
    """
    return (np.asarray(nd, dtype=np.int64) * 100
            + np.asarray(nc, dtype=np.int64))


def _decode_mode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode integer labels back to (ND, NC) pairs.

    Parameters
    ----------
    labels : numpy.ndarray
        Encoded labels produced by :func:`_encode_mode_labels`.

    Returns
    -------
    nd : numpy.ndarray
        Nodal diameter values.
    nc : numpy.ndarray
        Nodal circle values.
    """
    labels = np.asarray(labels, dtype=np.int64)
    nd = labels // 100
    nc = labels % 100
    return nd, nc


# ---------------------------------------------------------------------------
# Shared PyTorch training infrastructure (deferred import)
# ---------------------------------------------------------------------------


def _get_device(device_str: str) -> "torch.device":
    """Resolve a device string to a ``torch.device``.

    Parameters
    ----------
    device_str : str
        One of ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.

    Returns
    -------
    torch.device
    """
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _split_validation(
    X: np.ndarray,
    y: dict[str, np.ndarray],
    val_split: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None, dict[str, np.ndarray] | None]:
    """Split arrays into train / validation sets.

    Parameters
    ----------
    X : numpy.ndarray, shape (N, ...)
    y : dict of arrays each of length N.
    val_split : fraction in (0, 1) held out for validation.

    Returns
    -------
    X_train, y_train, X_val, y_val
        *X_val* and *y_val* are ``None`` when *val_split* <= 0.
    """
    n = X.shape[0]
    if val_split <= 0.0 or n < 4:
        return X, y, None, None
    n_val = max(1, int(n * val_split))
    idx = np.random.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    y_train = {k: v[train_idx] for k, v in y.items()}
    y_val = {k: v[val_idx] for k, v in y.items()}
    return X[train_idx], y_train, X[val_idx], y_val


def _train_pytorch_model(
    model: "torch.nn.Module",
    X_train: np.ndarray,
    y_train: dict[str, np.ndarray],
    X_val: np.ndarray | None,
    y_val: dict[str, np.ndarray] | None,
    config: Any,
    device: "torch.device",
    *,
    mode_label_encoder: dict[int, int] | None = None,
    whirl_offset: int = 1,
    reshape_fn: Any | None = None,
) -> dict[str, float]:
    """Shared training loop for all PyTorch-based tiers.

    Parameters
    ----------
    model : torch.nn.Module
        The network to train (already on *device*).
    X_train, y_train : training data and labels.
    X_val, y_val : optional validation data and labels.
    config : TrainingConfig instance.
    device : torch.device.
    mode_label_encoder : mapping from raw encoded mode label to contiguous
        class index.
    whirl_offset : value added to ``whirl_direction`` so that -1/0/+1
        become 0/1/2.
    reshape_fn : optional callable ``(X_tensor) -> X_tensor`` applied
        before each forward pass (e.g. to reshape to (B, C, L)).

    Returns
    -------
    dict[str, float]
        Training metrics (last epoch).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Encode targets ----------------------------------------------------------
    mode_labels_raw = _encode_mode_labels(
        y_train["nodal_diameter"], y_train["nodal_circle"]
    )
    if mode_label_encoder is None:
        unique_labels = np.unique(mode_labels_raw)
        mode_label_encoder = {int(v): i for i, v in enumerate(unique_labels)}
    mode_targets = np.array(
        [mode_label_encoder[int(v)] for v in mode_labels_raw], dtype=np.int64
    )
    whirl_targets = np.asarray(y_train["whirl_direction"], dtype=np.int64) + whirl_offset
    amp_targets = np.asarray(y_train["amplitude"], dtype=np.float32)
    vel_targets = np.asarray(y_train["wave_velocity"], dtype=np.float32)

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    ds = TensorDataset(
        X_t,
        torch.tensor(mode_targets, dtype=torch.long),
        torch.tensor(whirl_targets, dtype=torch.long),
        torch.tensor(amp_targets, dtype=torch.float32),
        torch.tensor(vel_targets, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    # Validation tensors (optional) -------------------------------------------
    has_val = X_val is not None and y_val is not None
    if has_val:
        mode_val_raw = _encode_mode_labels(
            y_val["nodal_diameter"], y_val["nodal_circle"]
        )
        mode_val = np.array(
            [mode_label_encoder.get(int(v), 0) for v in mode_val_raw],
            dtype=np.int64,
        )
        whirl_val = np.asarray(y_val["whirl_direction"], dtype=np.int64) + whirl_offset
        amp_val = np.asarray(y_val["amplitude"], dtype=np.float32)
        vel_val = np.asarray(y_val["wave_velocity"], dtype=np.float32)

        X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)
        mode_val_t = torch.tensor(mode_val, dtype=torch.long).to(device)
        whirl_val_t = torch.tensor(whirl_val, dtype=torch.long).to(device)
        amp_val_t = torch.tensor(amp_val, dtype=torch.float32).to(device)
        vel_val_t = torch.tensor(vel_val, dtype=torch.float32).to(device)

    # Losses & optimizer ------------------------------------------------------
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Early stopping state ----------------------------------------------------
    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    # Training loop -----------------------------------------------------------
    model.train()
    train_loss = 0.0
    for epoch in range(config.epochs):
        running_loss = 0.0
        n_batches = 0
        for batch in loader:
            xb, mode_b, whirl_b, amp_b, vel_b = [b.to(device) for b in batch]
            if reshape_fn is not None:
                xb = reshape_fn(xb)
            out_mode, out_whirl, out_amp, out_vel = model(xb)[:4]
            loss = (
                ce_loss(out_mode, mode_b.long())
                + ce_loss(out_whirl, whirl_b.long())
                + mse_loss(out_amp.squeeze(-1), amp_b.float())
                + mse_loss(out_vel.squeeze(-1), vel_b.float())
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # Validation ----------------------------------------------------------
        if has_val:
            model.eval()
            with torch.no_grad():
                xv = X_val_t
                if reshape_fn is not None:
                    xv = reshape_fn(xv)
                om, ow, oa, ov = model(xv)[:4]
                val_loss = (
                    ce_loss(om, mode_val_t.long()).item()
                    + ce_loss(ow, whirl_val_t.long()).item()
                    + mse_loss(oa.squeeze(-1), amp_val_t.float()).item()
                    + mse_loss(ov.squeeze(-1), vel_val_t.float()).item()
                )
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.4f)",
                        epoch + 1,
                        best_val_loss,
                    )
                    break

    # Restore best model state ------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, float] = {"train_loss": train_loss}
    if has_val:
        metrics["val_loss"] = best_val_loss
    return metrics


# ---------------------------------------------------------------------------
# Lazy PyTorch nn.Module class cache
# ---------------------------------------------------------------------------
# The nn.Module subclasses are built once on first access via
# _get_nn_class(name).  This allows the module to be imported even when
# PyTorch is not installed; an ImportError only occurs when a
# PyTorch-based tier is actually instantiated.

_NN_CLASS_CACHE: dict[str, type] = {}


def _get_nn_class(name: str) -> type:
    """Return the nn.Module class for *name*, building it lazily if needed.

    Parameters
    ----------
    name : str
        One of ``"ShallowNet"``, ``"CNN1DNet"``, ``"TemporalNet"``.
    """
    if name not in _NN_CLASS_CACHE:
        _build_all_nn_classes()
    return _NN_CLASS_CACHE[name]


def _build_all_nn_classes() -> None:
    """Construct all nn.Module subclasses and store them in the cache."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    import math

    class ShallowNet(nn.Module):
        """Two-hidden-layer multi-task net (Tier 4)."""

        def __init__(self, n_features: int, n_mode_classes: int, n_whirl_classes: int = 3,
                     dropout: float = 0.1, heteroscedastic: bool = False):
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.backbone = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.mode_head = nn.Linear(64, n_mode_classes)
            self.whirl_head = nn.Linear(64, n_whirl_classes)
            self.amp_head = nn.Linear(64, 1)
            self.vel_head = nn.Linear(64, 1)
            if heteroscedastic:
                self.amp_logvar_head = nn.Linear(64, 1)
                self.vel_logvar_head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.backbone(x)
            outputs = (
                self.mode_head(h),
                self.whirl_head(h),
                self.amp_head(h),
                self.vel_head(h),
            )
            if self.heteroscedastic:
                outputs = outputs + (self.amp_logvar_head(h), self.vel_logvar_head(h))
            return outputs

    class CNN1DNet(nn.Module):
        """1-D convolutional multi-task net (Tier 5)."""

        def __init__(
            self,
            n_channels: int,
            n_freq_bins: int,
            n_mode_classes: int,
            n_whirl_classes: int = 3,
            dropout: float = 0.1,
            heteroscedastic: bool = False,
        ):
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.AdaptiveAvgPool1d(1),
            )
            self.mode_head = nn.Linear(64, n_mode_classes)
            self.whirl_head = nn.Linear(64, n_whirl_classes)
            self.amp_head = nn.Linear(64, 1)
            self.vel_head = nn.Linear(64, 1)
            if heteroscedastic:
                self.amp_logvar_head = nn.Linear(64, 1)
                self.vel_logvar_head = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch, n_channels, n_freq_bins)
            h = self.features(x)          # (batch, 64, 1)
            h = h.squeeze(-1)             # (batch, 64)
            outputs = (
                self.mode_head(h),
                self.whirl_head(h),
                self.amp_head(h),
                self.vel_head(h),
            )
            if self.heteroscedastic:
                outputs = outputs + (self.amp_logvar_head(h), self.vel_logvar_head(h))
            return outputs

    class ResBlock1D(nn.Module):
        """1-D residual block with two Conv1d layers."""

        def __init__(self, channels: int, dropout: float = 0.1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(channels),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x + self.block(x))

    class ResNet1DNet(nn.Module):
        """1-D ResNet multi-task net (Tier 5 variant)."""

        def __init__(
            self,
            n_channels: int,
            n_freq_bins: int,
            n_mode_classes: int,
            n_whirl_classes: int = 3,
            dropout: float = 0.1,
            heteroscedastic: bool = False,
        ):
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.stem = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.res_blocks = nn.Sequential(
                ResBlock1D(64, dropout),
                ResBlock1D(64, dropout),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.mode_head = nn.Linear(64, n_mode_classes)
            self.whirl_head = nn.Linear(64, n_whirl_classes)
            self.amp_head = nn.Linear(64, 1)
            self.vel_head = nn.Linear(64, 1)
            if heteroscedastic:
                self.amp_logvar_head = nn.Linear(64, 1)
                self.vel_logvar_head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.stem(x)
            h = self.res_blocks(h)
            h = self.pool(h).squeeze(-1)
            outputs = (
                self.mode_head(h),
                self.whirl_head(h),
                self.amp_head(h),
                self.vel_head(h),
            )
            if self.heteroscedastic:
                outputs = outputs + (self.amp_logvar_head(h), self.vel_logvar_head(h))
            return outputs

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for Transformer."""

        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model > 1:
                pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerNet(nn.Module):
        """Transformer-based multi-task net (Tier 6 variant)."""

        def __init__(
            self,
            n_channels: int,
            seq_len: int,
            n_mode_classes: int,
            n_whirl_classes: int = 3,
            dropout: float = 0.1,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            heteroscedastic: bool = False,
        ):
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.input_proj = nn.Linear(n_channels, d_model)
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=128,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 64)
            self.dropout = nn.Dropout(dropout)
            self.mode_head = nn.Linear(64, n_mode_classes)
            self.whirl_head = nn.Linear(64, n_whirl_classes)
            self.amp_head = nn.Linear(64, 1)
            self.vel_head = nn.Linear(64, 1)
            if heteroscedastic:
                self.amp_logvar_head = nn.Linear(64, 1)
                self.vel_logvar_head = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch, n_channels, seq_len)
            x = x.permute(0, 2, 1)  # (batch, seq_len, n_channels)
            x = self.input_proj(x)  # (batch, seq_len, d_model)
            x = self.pos_enc(x)
            x = self.transformer(x)  # (batch, seq_len, d_model)
            x = x.mean(dim=1)  # global average -> (batch, d_model)
            h = F.relu(self.fc(x))
            h = self.dropout(h)
            outputs = (
                self.mode_head(h),
                self.whirl_head(h),
                self.amp_head(h),
                self.vel_head(h),
            )
            if self.heteroscedastic:
                outputs = outputs + (self.amp_logvar_head(h), self.vel_logvar_head(h))
            return outputs

    class TemporalNet(nn.Module):
        """Conv + BiLSTM multi-task net (Tier 6)."""

        def __init__(
            self,
            n_channels: int,
            seq_len: int,
            n_mode_classes: int,
            n_whirl_classes: int = 3,
            dropout: float = 0.1,
            heteroscedastic: bool = False,
        ):
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, kernel_size=7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(64, 32, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(64, 64)  # 32*2 from bidirectional
            self.dropout_layer = nn.Dropout(dropout)
            self.mode_head = nn.Linear(64, n_mode_classes)
            self.whirl_head = nn.Linear(64, n_whirl_classes)
            self.amp_head = nn.Linear(64, 1)
            self.vel_head = nn.Linear(64, 1)
            if heteroscedastic:
                self.amp_logvar_head = nn.Linear(64, 1)
                self.vel_logvar_head = nn.Linear(64, 1)

        def forward(self, x):
            # x: (batch, n_channels, seq_len)
            h = self.conv(x)              # (batch, 64, seq_len)
            h = h.permute(0, 2, 1)        # (batch, seq_len, 64) for LSTM
            h, _ = self.lstm(h)            # (batch, seq_len, 64)
            h = h[:, -1, :]               # last timestep -> (batch, 64)
            h = F.relu(self.fc(h))         # (batch, 64)
            h = self.dropout_layer(h)
            outputs = (
                self.mode_head(h),
                self.whirl_head(h),
                self.amp_head(h),
                self.vel_head(h),
            )
            if self.heteroscedastic:
                outputs = outputs + (self.amp_logvar_head(h), self.vel_logvar_head(h))
            return outputs

    _NN_CLASS_CACHE["ShallowNet"] = ShallowNet
    _NN_CLASS_CACHE["CNN1DNet"] = CNN1DNet
    _NN_CLASS_CACHE["ResBlock1D"] = ResBlock1D
    _NN_CLASS_CACHE["ResNet1DNet"] = ResNet1DNet
    _NN_CLASS_CACHE["PositionalEncoding"] = PositionalEncoding
    _NN_CLASS_CACHE["TransformerNet"] = TransformerNet
    _NN_CLASS_CACHE["TemporalNet"] = TemporalNet


# ============================================================================
# Tier 1 -- LinearModeIDModel
# ============================================================================


class LinearModeIDModel:
    """Logistic Regression + Ridge/Lasso for mode identification.

    Uses four independent estimators:

    * ``_mode_clf`` -- ``LogisticRegression`` on encoded (ND, NC) labels.
    * ``_whirl_clf`` -- ``LogisticRegression`` on whirl direction.
    * ``_amp_reg`` -- ``Ridge`` or ``Lasso`` regression on amplitude.
    * ``_vel_reg`` -- ``Ridge`` or ``Lasso`` regression on wave velocity.

    Parameters
    ----------
    variant : str
        ``"ridge"`` (default) or ``"lasso"``.
    """

    def __init__(self, variant: str = "ridge") -> None:
        self._variant = variant
        self._mode_clf: Any = None
        self._whirl_clf: Any = None
        self._amp_reg: Any = None
        self._vel_reg: Any = None
        self._is_fitted: bool = False

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train all four estimators.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays, each length N.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
            Training metrics including accuracy and R-squared scores.
        """
        from sklearn.linear_model import LogisticRegression, Ridge, Lasso
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_percentage_error,
            r2_score,
        )

        mode_labels = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        whirl = np.asarray(y["whirl_direction"], dtype=np.int64)
        amplitude = np.asarray(y["amplitude"], dtype=np.float64)
        velocity = np.asarray(y["wave_velocity"], dtype=np.float64)

        self._mode_clf = LogisticRegression(max_iter=1000, C=1.0)
        self._whirl_clf = LogisticRegression(max_iter=1000)
        if self._variant == "lasso":
            self._amp_reg = Lasso(alpha=1.0, max_iter=2000)
            self._vel_reg = Lasso(alpha=1.0, max_iter=2000)
        else:
            self._amp_reg = Ridge(alpha=1.0)
            self._vel_reg = Ridge(alpha=1.0)

        self._mode_clf.fit(X, mode_labels)
        self._whirl_clf.fit(X, whirl)
        self._amp_reg.fit(X, amplitude)
        self._vel_reg.fit(X, velocity)

        self._is_fitted = True

        # Compute training metrics
        mode_pred = self._mode_clf.predict(X)
        whirl_pred = self._whirl_clf.predict(X)
        amp_pred = self._amp_reg.predict(X)
        vel_pred = self._vel_reg.predict(X)

        metrics: dict[str, float] = {
            "mode_accuracy": float(accuracy_score(mode_labels, mode_pred)),
            "mode_f1": float(
                f1_score(mode_labels, mode_pred, average="macro", zero_division=0)
            ),
            "whirl_accuracy": float(accuracy_score(whirl, whirl_pred)),
            "amplitude_r2": float(r2_score(amplitude, amp_pred)),
            "velocity_r2": float(r2_score(velocity, vel_pred)),
        }

        # MAPE can fail when true values are zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nonzero = amplitude != 0.0
            if nonzero.any():
                metrics["amplitude_mape"] = float(
                    mean_absolute_percentage_error(amplitude[nonzero], amp_pred[nonzero])
                )
            else:
                metrics["amplitude_mape"] = 0.0

        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters from features.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
            Keys: nodal_diameter, nodal_circle, frequency, whirl_direction,
            amplitude, wave_velocity, confidence.
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        mode_pred = self._mode_clf.predict(X)
        nd, nc = _decode_mode_labels(mode_pred)

        whirl_pred = self._whirl_clf.predict(X)
        amp_pred = self._amp_reg.predict(X)
        vel_pred = self._vel_reg.predict(X)

        # Confidence from mode classifier probability
        if hasattr(self._mode_clf, "predict_proba"):
            proba = self._mode_clf.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(X.shape[0], dtype=np.float64)

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_pred.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Persist model to disk via joblib.

        Parameters
        ----------
        path : str
            File path (typically ``*.joblib``).
        """
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump(
            {
                "mode_clf": self._mode_clf,
                "whirl_clf": self._whirl_clf,
                "amp_reg": self._amp_reg,
                "vel_reg": self._vel_reg,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a previously saved model.

        Parameters
        ----------
        path : str
            File path written by :meth:`save`.
        """
        import joblib

        data = joblib.load(path)
        self._mode_clf = data["mode_clf"]
        self._whirl_clf = data["whirl_clf"]
        self._amp_reg = data["amp_reg"]
        self._vel_reg = data["vel_reg"]
        self._is_fitted = True


# ============================================================================
# Tier 2 -- TreeModeIDModel
# ============================================================================


class TreeModeIDModel:
    """Random Forest / XGBoost ensemble for mode identification.

    Falls back from XGBoost to scikit-learn ``RandomForest*`` when the
    ``xgboost`` package is not installed.

    Attributes
    ----------
    feature_importances_ : numpy.ndarray
        Average feature importances across all four estimators.
        Available after :meth:`train`.
    """

    def __init__(self) -> None:
        self._mode_clf: Any = None
        self._whirl_clf: Any = None
        self._amp_reg: Any = None
        self._vel_reg: Any = None
        self._backend: str = "sklearn"  # "lightgbm", "xgboost", or "sklearn"
        self._mode_label_encoder: dict[int, int] | None = None
        self._mode_label_decoder: dict[int, int] | None = None
        self._is_fitted: bool = False

    # --------------------------------------------------------------------- #

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances across all four estimators."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted; no feature importances available.")
        imps = []
        for est in (self._mode_clf, self._whirl_clf, self._amp_reg, self._vel_reg):
            if hasattr(est, "feature_importances_"):
                imps.append(est.feature_importances_)
        if not imps:
            raise RuntimeError("Estimators do not expose feature_importances_.")
        return np.mean(imps, axis=0)

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train four tree-based estimators.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_percentage_error,
            r2_score,
        )

        # Three-level fallback: LightGBM -> XGBoost -> RandomForest
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            self._mode_clf = LGBMClassifier(n_estimators=100, verbose=-1)
            self._whirl_clf = LGBMClassifier(n_estimators=100, verbose=-1)
            self._amp_reg = LGBMRegressor(n_estimators=100, verbose=-1)
            self._vel_reg = LGBMRegressor(n_estimators=100, verbose=-1)
            self._backend = "lightgbm"
        except ImportError:
            try:
                from xgboost import XGBClassifier, XGBRegressor

                self._mode_clf = XGBClassifier(
                    n_estimators=100, use_label_encoder=False, eval_metric="mlogloss"
                )
                self._whirl_clf = XGBClassifier(
                    n_estimators=100, use_label_encoder=False, eval_metric="mlogloss"
                )
                self._amp_reg = XGBRegressor(n_estimators=100)
                self._vel_reg = XGBRegressor(n_estimators=100)
                self._backend = "xgboost"
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                self._mode_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                self._whirl_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                self._amp_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1)
                self._vel_reg = RandomForestRegressor(n_estimators=100, n_jobs=-1)
                self._backend = "sklearn"

        mode_labels_raw = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        unique_labels = np.unique(mode_labels_raw)
        self._mode_label_encoder = {int(v): i for i, v in enumerate(unique_labels)}
        self._mode_label_decoder = {i: int(v) for i, v in enumerate(unique_labels)}
        mode_labels = np.array(
            [self._mode_label_encoder[int(v)] for v in mode_labels_raw], dtype=np.int64
        )
        whirl = np.asarray(y["whirl_direction"], dtype=np.int64)
        amplitude = np.asarray(y["amplitude"], dtype=np.float64)
        velocity = np.asarray(y["wave_velocity"], dtype=np.float64)

        self._mode_clf.fit(X, mode_labels)
        self._whirl_clf.fit(X, whirl)
        self._amp_reg.fit(X, amplitude)
        self._vel_reg.fit(X, velocity)
        self._is_fitted = True

        # Metrics
        mode_pred = self._mode_clf.predict(X)
        whirl_pred = self._whirl_clf.predict(X)
        amp_pred = self._amp_reg.predict(X)
        vel_pred = self._vel_reg.predict(X)

        metrics: dict[str, float] = {
            "mode_accuracy": float(accuracy_score(mode_labels, mode_pred)),
            "mode_f1": float(
                f1_score(mode_labels, mode_pred, average="macro", zero_division=0)
            ),
            "whirl_accuracy": float(accuracy_score(whirl, whirl_pred)),
            "amplitude_r2": float(r2_score(amplitude, amp_pred)),
            "velocity_r2": float(r2_score(velocity, vel_pred)),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nonzero = amplitude != 0.0
            if nonzero.any():
                metrics["amplitude_mape"] = float(
                    mean_absolute_percentage_error(amplitude[nonzero], amp_pred[nonzero])
                )
            else:
                metrics["amplitude_mape"] = 0.0

        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        mode_pred_idx = self._mode_clf.predict(X)
        mode_pred = np.array(
            [self._mode_label_decoder[int(i)] for i in mode_pred_idx], dtype=np.int64
        )
        nd, nc = _decode_mode_labels(mode_pred)
        whirl_pred = self._whirl_clf.predict(X)
        amp_pred = self._amp_reg.predict(X)
        vel_pred = self._vel_reg.predict(X)

        # Confidence from class probability (if available)
        if hasattr(self._mode_clf, "predict_proba"):
            proba = self._mode_clf.predict_proba(X)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(X.shape[0], dtype=np.float64)

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_pred.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Persist model to disk via joblib.

        Parameters
        ----------
        path : str
        """
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump(
            {
                "mode_clf": self._mode_clf,
                "whirl_clf": self._whirl_clf,
                "amp_reg": self._amp_reg,
                "vel_reg": self._vel_reg,
                "backend": self._backend,
                "mode_label_encoder": self._mode_label_encoder,
                "mode_label_decoder": self._mode_label_decoder,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a previously saved model.

        Parameters
        ----------
        path : str
        """
        import joblib

        data = joblib.load(path)
        self._mode_clf = data["mode_clf"]
        self._whirl_clf = data["whirl_clf"]
        self._amp_reg = data["amp_reg"]
        self._vel_reg = data["vel_reg"]
        self._backend = data.get("backend", "xgboost" if data.get("use_xgb") else "sklearn")
        self._mode_label_encoder = data.get("mode_label_encoder")
        self._mode_label_decoder = data.get("mode_label_decoder")
        self._is_fitted = True


# ============================================================================
# Tier 3 -- SVMModeIDModel
# ============================================================================


class SVMModeIDModel:
    """SVM with RBF kernel for mode identification.

    Requires feature scaling via ``StandardScaler``.

    Medium interpretability via support vectors and kernel analysis.
    """

    def __init__(self) -> None:
        self._scaler: Any = None
        self._mode_clf: Any = None
        self._whirl_clf: Any = None
        self._amp_reg: Any = None
        self._vel_reg: Any = None
        self._is_fitted: bool = False

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train SVM estimators with scaled features.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_percentage_error,
            r2_score,
        )
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, SVR

        mode_labels = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        whirl = np.asarray(y["whirl_direction"], dtype=np.int64)
        amplitude = np.asarray(y["amplitude"], dtype=np.float64)
        velocity = np.asarray(y["wave_velocity"], dtype=np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._mode_clf = SVC(kernel="rbf", probability=True)
        self._whirl_clf = SVC(kernel="rbf", probability=True)
        self._amp_reg = SVR(kernel="rbf")
        self._vel_reg = SVR(kernel="rbf")

        self._mode_clf.fit(X_scaled, mode_labels)
        self._whirl_clf.fit(X_scaled, whirl)
        self._amp_reg.fit(X_scaled, amplitude)
        self._vel_reg.fit(X_scaled, velocity)
        self._is_fitted = True

        # Metrics
        mode_pred = self._mode_clf.predict(X_scaled)
        whirl_pred = self._whirl_clf.predict(X_scaled)
        amp_pred = self._amp_reg.predict(X_scaled)
        vel_pred = self._vel_reg.predict(X_scaled)

        metrics: dict[str, float] = {
            "mode_accuracy": float(accuracy_score(mode_labels, mode_pred)),
            "mode_f1": float(
                f1_score(mode_labels, mode_pred, average="macro", zero_division=0)
            ),
            "whirl_accuracy": float(accuracy_score(whirl, whirl_pred)),
            "amplitude_r2": float(r2_score(amplitude, amp_pred)),
            "velocity_r2": float(r2_score(velocity, vel_pred)),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nonzero = amplitude != 0.0
            if nonzero.any():
                metrics["amplitude_mape"] = float(
                    mean_absolute_percentage_error(amplitude[nonzero], amp_pred[nonzero])
                )
            else:
                metrics["amplitude_mape"] = 0.0

        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters (scales input first).

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X_scaled = self._scaler.transform(X)

        mode_pred = self._mode_clf.predict(X_scaled)
        nd, nc = _decode_mode_labels(mode_pred)
        whirl_pred = self._whirl_clf.predict(X_scaled)
        amp_pred = self._amp_reg.predict(X_scaled)
        vel_pred = self._vel_reg.predict(X_scaled)

        if hasattr(self._mode_clf, "predict_proba"):
            proba = self._mode_clf.predict_proba(X_scaled)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(X.shape[0], dtype=np.float64)

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_pred.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Persist model to disk via joblib.

        Parameters
        ----------
        path : str
        """
        import joblib

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        joblib.dump(
            {
                "scaler": self._scaler,
                "mode_clf": self._mode_clf,
                "whirl_clf": self._whirl_clf,
                "amp_reg": self._amp_reg,
                "vel_reg": self._vel_reg,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load a previously saved model.

        Parameters
        ----------
        path : str
        """
        import joblib

        data = joblib.load(path)
        self._scaler = data["scaler"]
        self._mode_clf = data["mode_clf"]
        self._whirl_clf = data["whirl_clf"]
        self._amp_reg = data["amp_reg"]
        self._vel_reg = data["vel_reg"]
        self._is_fitted = True


# ============================================================================
# Tier 4 -- ShallowNNModeIDModel (PyTorch)
# ============================================================================


class ShallowNNModeIDModel:
    """Shallow multi-task neural network for mode identification.

    Architecture: two hidden layers (128 -> 64) with four task-specific
    heads for mode classification, whirl classification, amplitude
    regression, and wave velocity regression.

    Medium interpretability via gradient-based attribution.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._scaler: Any = None
        self._mode_label_encoder: dict[int, int] | None = None
        self._mode_label_decoder: dict[int, int] | None = None
        self._n_features: int = 0
        self._n_mode_classes: int = 0
        self._device_str: str = "cpu"
        self._is_fitted: bool = False

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train the shallow neural network.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
        """
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)

        # Build label encoder
        mode_labels = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        unique_labels = np.unique(mode_labels)
        self._mode_label_encoder = {int(v): i for i, v in enumerate(unique_labels)}
        self._mode_label_decoder = {i: int(v) for i, v in enumerate(unique_labels)}
        self._n_mode_classes = len(unique_labels)
        self._n_features = X.shape[1]

        # Split
        X_train, y_train, X_val, y_val = _split_validation(
            X_scaled, y, config.validation_split
        )

        # Build model
        device = _get_device(config.device)
        self._device_str = str(device)

        ShallowNet = _get_nn_class("ShallowNet")
        self._model = ShallowNet(self._n_features, self._n_mode_classes).to(device)

        metrics = _train_pytorch_model(
            self._model,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            device,
            mode_label_encoder=self._mode_label_encoder,
        )
        self._is_fitted = True
        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X_scaled = self._scaler.transform(X).astype(np.float32)
        device = torch.device(self._device_str)
        self._model.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(device)
            out_mode, out_whirl, out_amp, out_vel = self._model(X_t)

        # Classification heads
        mode_proba = torch.softmax(out_mode, dim=1).cpu().numpy()
        mode_idx = np.argmax(mode_proba, axis=1)
        confidence = np.max(mode_proba, axis=1)

        # Decode mode labels
        mode_raw = np.array(
            [self._mode_label_decoder[int(i)] for i in mode_idx], dtype=np.int64
        )
        nd, nc = _decode_mode_labels(mode_raw)

        whirl_idx = torch.argmax(out_whirl, dim=1).cpu().numpy() - 1  # undo +1 offset

        amp_pred = out_amp.squeeze(-1).cpu().numpy()
        vel_pred = out_vel.squeeze(-1).cpu().numpy()

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_idx.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save model state dict plus metadata.

        Parameters
        ----------
        path : str
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "n_features": self._n_features,
                "n_mode_classes": self._n_mode_classes,
                "mode_label_encoder": self._mode_label_encoder,
                "mode_label_decoder": self._mode_label_decoder,
                "scaler_mean": self._scaler.mean_,
                "scaler_scale": self._scaler.scale_,
                "device": self._device_str,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        data = torch.load(path, map_location="cpu", weights_only=False)
        self._n_features = data["n_features"]
        self._n_mode_classes = data["n_mode_classes"]
        self._mode_label_encoder = data["mode_label_encoder"]
        self._mode_label_decoder = data["mode_label_decoder"]
        self._device_str = data.get("device", "cpu")

        self._scaler = StandardScaler()
        self._scaler.mean_ = data["scaler_mean"]
        self._scaler.scale_ = data["scaler_scale"]
        self._scaler.var_ = data["scaler_scale"] ** 2
        self._scaler.n_features_in_ = self._n_features

        device = torch.device(self._device_str)
        ShallowNet = _get_nn_class("ShallowNet")
        self._model = ShallowNet(self._n_features, self._n_mode_classes).to(device)
        self._model.load_state_dict(data["state_dict"])
        self._model.eval()
        self._is_fitted = True


# ============================================================================
# Tier 5 -- CNNModeIDModel (PyTorch)
# ============================================================================


def _infer_channel_shape(n_features: int) -> tuple[int, int]:
    """Infer (n_channels, spatial_dim) from total feature count.

    Tries common channel counts in order of preference and returns the
    first that divides *n_features* evenly.

    Parameters
    ----------
    n_features : int
        Total number of features.

    Returns
    -------
    tuple[int, int]
        (n_channels, spatial_dim).
    """
    for nc in (1, 2, 4, 8, 16, 3, 6, 12):
        if n_features % nc == 0:
            return nc, n_features // nc
    return 1, n_features


def _make_reshape_fn(n_channels: int, spatial_dim: int):
    """Return a callable that reshapes ``(B, F) -> (B, C, L)``."""
    def _reshape(x):
        return x.view(x.size(0), n_channels, spatial_dim)
    return _reshape


class CNNModeIDModel:
    """1-D CNN / ResNet on spectral inputs for mode identification.

    Input features are reshaped from ``(batch, n_features)`` to
    ``(batch, n_channels, n_freq_bins)`` before being fed to the
    convolutional backbone.

    Parameters
    ----------
    variant : str
        ``"cnn"`` (default) or ``"resnet"``.
    """

    def __init__(self, variant: str = "cnn") -> None:
        self._variant = variant
        self._model: Any = None
        self._scaler: Any = None
        self._mode_label_encoder: dict[int, int] | None = None
        self._mode_label_decoder: dict[int, int] | None = None
        self._n_channels: int = 0
        self._n_freq_bins: int = 0
        self._n_mode_classes: int = 0
        self._device_str: str = "cpu"
        self._is_fitted: bool = False
        self.last_conv_activations: Any = None
        """Activations from the last convolutional layer (for Grad-CAM)."""

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train the 1-D CNN.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
        """
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)

        mode_labels = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        unique_labels = np.unique(mode_labels)
        self._mode_label_encoder = {int(v): i for i, v in enumerate(unique_labels)}
        self._mode_label_decoder = {i: int(v) for i, v in enumerate(unique_labels)}
        self._n_mode_classes = len(unique_labels)

        self._n_channels, self._n_freq_bins = _infer_channel_shape(X.shape[1])

        X_train, y_train, X_val, y_val = _split_validation(
            X_scaled, y, config.validation_split
        )

        device = _get_device(config.device)
        self._device_str = str(device)

        if self._variant == "resnet":
            ResNet1DNet = _get_nn_class("ResNet1DNet")
            self._model = ResNet1DNet(
                self._n_channels, self._n_freq_bins, self._n_mode_classes
            ).to(device)
        else:
            CNN1DNet = _get_nn_class("CNN1DNet")
            self._model = CNN1DNet(
                self._n_channels, self._n_freq_bins, self._n_mode_classes
            ).to(device)

        # Register Grad-CAM hook
        self.last_conv_activations = None

        def _hook_fn(module, input, output):
            self.last_conv_activations = output.detach()

        if self._variant == "resnet":
            self._model.pool.register_forward_hook(_hook_fn)
        else:
            self._model.features[-1].register_forward_hook(_hook_fn)

        reshape_fn = _make_reshape_fn(self._n_channels, self._n_freq_bins)

        metrics = _train_pytorch_model(
            self._model,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            device,
            mode_label_encoder=self._mode_label_encoder,
            reshape_fn=reshape_fn,
        )
        self._is_fitted = True
        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X_scaled = self._scaler.transform(X).astype(np.float32)
        device = torch.device(self._device_str)
        self._model.eval()

        reshape_fn = _make_reshape_fn(self._n_channels, self._n_freq_bins)

        with torch.no_grad():
            X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(device)
            X_t = reshape_fn(X_t)
            out_mode, out_whirl, out_amp, out_vel = self._model(X_t)

        mode_proba = torch.softmax(out_mode, dim=1).cpu().numpy()
        mode_idx = np.argmax(mode_proba, axis=1)
        confidence = np.max(mode_proba, axis=1)

        mode_raw = np.array(
            [self._mode_label_decoder[int(i)] for i in mode_idx], dtype=np.int64
        )
        nd, nc = _decode_mode_labels(mode_raw)
        whirl_idx = torch.argmax(out_whirl, dim=1).cpu().numpy() - 1

        amp_pred = out_amp.squeeze(-1).cpu().numpy()
        vel_pred = out_vel.squeeze(-1).cpu().numpy()

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_idx.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Persist model to disk.

        Parameters
        ----------
        path : str
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "n_channels": self._n_channels,
                "n_freq_bins": self._n_freq_bins,
                "n_mode_classes": self._n_mode_classes,
                "mode_label_encoder": self._mode_label_encoder,
                "mode_label_decoder": self._mode_label_decoder,
                "scaler_mean": self._scaler.mean_,
                "scaler_scale": self._scaler.scale_,
                "device": self._device_str,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        data = torch.load(path, map_location="cpu", weights_only=False)
        self._n_channels = data["n_channels"]
        self._n_freq_bins = data["n_freq_bins"]
        self._n_mode_classes = data["n_mode_classes"]
        self._mode_label_encoder = data["mode_label_encoder"]
        self._mode_label_decoder = data["mode_label_decoder"]
        self._device_str = data.get("device", "cpu")

        self._scaler = StandardScaler()
        self._scaler.mean_ = data["scaler_mean"]
        self._scaler.scale_ = data["scaler_scale"]
        self._scaler.var_ = data["scaler_scale"] ** 2
        self._scaler.n_features_in_ = self._n_channels * self._n_freq_bins

        device = torch.device(self._device_str)
        CNN1DNet = _get_nn_class("CNN1DNet")
        self._model = CNN1DNet(
            self._n_channels, self._n_freq_bins, self._n_mode_classes
        ).to(device)
        self._model.load_state_dict(data["state_dict"])
        self._model.eval()
        self._is_fitted = True


# ============================================================================
# Tier 6 -- TemporalModeIDModel (PyTorch)
# ============================================================================


class TemporalModeIDModel:
    """Temporal CNN + bidirectional LSTM / Transformer for mode identification.

    Input features are reshaped to ``(batch, n_channels, seq_len)`` and
    processed by the selected backend.

    Parameters
    ----------
    variant : str
        ``"lstm"`` (default) or ``"transformer"``.
    """

    def __init__(self, variant: str = "lstm") -> None:
        self._variant = variant
        self._model: Any = None
        self._scaler: Any = None
        self._mode_label_encoder: dict[int, int] | None = None
        self._mode_label_decoder: dict[int, int] | None = None
        self._n_channels: int = 0
        self._seq_len: int = 0
        self._n_mode_classes: int = 0
        self._device_str: str = "cpu"
        self._is_fitted: bool = False

    # --------------------------------------------------------------------- #

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train the temporal model.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)
        y : dict of target arrays.
        config : TrainingConfig.

        Returns
        -------
        dict[str, float]
        """
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X).astype(np.float32)

        mode_labels = _encode_mode_labels(y["nodal_diameter"], y["nodal_circle"])
        unique_labels = np.unique(mode_labels)
        self._mode_label_encoder = {int(v): i for i, v in enumerate(unique_labels)}
        self._mode_label_decoder = {i: int(v) for i, v in enumerate(unique_labels)}
        self._n_mode_classes = len(unique_labels)

        self._n_channels, self._seq_len = _infer_channel_shape(X.shape[1])

        X_train, y_train, X_val, y_val = _split_validation(
            X_scaled, y, config.validation_split
        )

        device = _get_device(config.device)
        self._device_str = str(device)

        if self._variant == "transformer":
            TransformerNet = _get_nn_class("TransformerNet")
            self._model = TransformerNet(
                self._n_channels, self._seq_len, self._n_mode_classes
            ).to(device)
        else:
            TemporalNet = _get_nn_class("TemporalNet")
            self._model = TemporalNet(
                self._n_channels, self._seq_len, self._n_mode_classes
            ).to(device)

        reshape_fn = _make_reshape_fn(self._n_channels, self._seq_len)

        metrics = _train_pytorch_model(
            self._model,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            device,
            mode_label_encoder=self._mode_label_encoder,
            reshape_fn=reshape_fn,
        )
        self._is_fitted = True
        return metrics

    # --------------------------------------------------------------------- #

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict mode parameters.

        Parameters
        ----------
        X : numpy.ndarray, shape (N, n_features)

        Returns
        -------
        dict[str, numpy.ndarray]
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call train() first.")

        X_scaled = self._scaler.transform(X).astype(np.float32)
        device = torch.device(self._device_str)
        self._model.eval()

        reshape_fn = _make_reshape_fn(self._n_channels, self._seq_len)

        with torch.no_grad():
            X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(device)
            X_t = reshape_fn(X_t)
            out_mode, out_whirl, out_amp, out_vel = self._model(X_t)

        mode_proba = torch.softmax(out_mode, dim=1).cpu().numpy()
        mode_idx = np.argmax(mode_proba, axis=1)
        confidence = np.max(mode_proba, axis=1)

        mode_raw = np.array(
            [self._mode_label_decoder[int(i)] for i in mode_idx], dtype=np.int64
        )
        nd, nc = _decode_mode_labels(mode_raw)
        whirl_idx = torch.argmax(out_whirl, dim=1).cpu().numpy() - 1

        amp_pred = out_amp.squeeze(-1).cpu().numpy()
        vel_pred = out_vel.squeeze(-1).cpu().numpy()

        return {
            "nodal_diameter": nd.astype(np.int64),
            "nodal_circle": nc.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_idx.astype(np.int64),
            "amplitude": amp_pred.astype(np.float64),
            "wave_velocity": vel_pred.astype(np.float64),
            "confidence": confidence.astype(np.float64),
        }

    # --------------------------------------------------------------------- #

    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
        """
        import torch
        from sklearn.preprocessing import StandardScaler

        data = torch.load(path, map_location="cpu", weights_only=False)
        self._n_channels = data["n_channels"]
        self._seq_len = data["seq_len"]
        self._n_mode_classes = data["n_mode_classes"]
        self._mode_label_encoder = data["mode_label_encoder"]
        self._mode_label_decoder = data["mode_label_decoder"]
        self._device_str = data.get("device", "cpu")

        self._scaler = StandardScaler()
        self._scaler.mean_ = data["scaler_mean"]
        self._scaler.scale_ = data["scaler_scale"]
        self._scaler.var_ = data["scaler_scale"] ** 2
        self._scaler.n_features_in_ = self._n_channels * self._seq_len

        device = torch.device(self._device_str)
        variant = data.get("variant", self._variant)
        self._variant = variant
        if variant == "transformer":
            TransformerNet = _get_nn_class("TransformerNet")
            self._model = TransformerNet(
                self._n_channels, self._seq_len, self._n_mode_classes
            ).to(device)
        else:
            TemporalNet = _get_nn_class("TemporalNet")
            self._model = TemporalNet(
                self._n_channels, self._seq_len, self._n_mode_classes
            ).to(device)
        self._model.load_state_dict(data["state_dict"])
        self._model.eval()
        self._is_fitted = True

    def save(self, path: str) -> None:
        """Persist model to disk.

        Parameters
        ----------
        path : str
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "n_channels": self._n_channels,
                "seq_len": self._seq_len,
                "n_mode_classes": self._n_mode_classes,
                "mode_label_encoder": self._mode_label_encoder,
                "mode_label_decoder": self._mode_label_decoder,
                "scaler_mean": self._scaler.mean_,
                "scaler_scale": self._scaler.scale_,
                "device": self._device_str,
                "variant": self._variant,
            },
            path,
        )


# ============================================================================
# CompositeModel (C5: independent per-sub-task ladder)
# ============================================================================


class CompositeModel:
    """Wraps 4 independent sub-task models into a single predict interface.

    Used when the complexity ladder runs independently per sub-task,
    potentially selecting different tiers for each.
    """

    def __init__(
        self,
        mode_model: Any,
        whirl_model: Any,
        amp_model: Any,
        vel_model: Any,
        subtask_tiers: dict[str, int] | None = None,
    ) -> None:
        self._mode_model = mode_model
        self._whirl_model = whirl_model
        self._amp_model = amp_model
        self._vel_model = vel_model
        self.subtask_tiers = subtask_tiers or {}
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Merge predictions from each sub-task model."""
        mode_preds = self._mode_model.predict(X)
        whirl_preds = self._whirl_model.predict(X)
        amp_preds = self._amp_model.predict(X)
        vel_preds = self._vel_model.predict(X)
        return {
            "nodal_diameter": mode_preds["nodal_diameter"],
            "nodal_circle": mode_preds["nodal_circle"],
            "frequency": mode_preds.get("frequency", np.zeros(X.shape[0])),
            "whirl_direction": whirl_preds["whirl_direction"],
            "amplitude": amp_preds["amplitude"],
            "wave_velocity": vel_preds["wave_velocity"],
            "confidence": mode_preds.get("confidence", np.ones(X.shape[0])),
        }

    def save(self, path: str) -> None:
        """Save all sub-task models via joblib."""
        import joblib
        joblib.dump({
            "mode_model": self._mode_model,
            "whirl_model": self._whirl_model,
            "amp_model": self._amp_model,
            "vel_model": self._vel_model,
            "subtask_tiers": self.subtask_tiers,
        }, path)

    def load(self, path: str) -> None:
        """Load sub-task models from disk."""
        import joblib
        data = joblib.load(path)
        self._mode_model = data["mode_model"]
        self._whirl_model = data["whirl_model"]
        self._amp_model = data["amp_model"]
        self._vel_model = data["vel_model"]
        self.subtask_tiers = data.get("subtask_tiers", {})
        self._is_fitted = True

    def train(self, X: np.ndarray, y: dict[str, np.ndarray],
              config: Any) -> dict[str, float]:
        """No-op: sub-task models are already trained."""
        return {}


# ============================================================================
# MC Dropout (D4)
# ============================================================================


def mc_dropout_predict(
    model: Any,
    X: np.ndarray,
    n_forward_passes: int = 30,
) -> dict[str, np.ndarray]:
    """Run MC Dropout inference for epistemic uncertainty estimation.

    Enables dropout at inference time, runs *n_forward_passes* stochastic
    forward passes, and returns mean predictions plus epistemic variance.

    Parameters
    ----------
    model : PyTorch-based model (Tier 4-6).
    X : (N, n_features) input features.
    n_forward_passes : number of stochastic forward passes.

    Returns
    -------
    dict with standard prediction keys plus ``amplitude_epistemic_var``,
    ``velocity_epistemic_var``, ``mode_entropy``.
    """
    import torch

    nn_model = model._model
    device = torch.device(model._device_str)
    X_scaled = model._scaler.transform(X).astype(np.float32)
    X_t = torch.as_tensor(X_scaled, dtype=torch.float32).to(device)

    # Reshape if needed (CNN/Temporal models)
    reshape_fn = None
    if hasattr(model, "_n_channels") and hasattr(model, "_n_freq_bins"):
        reshape_fn = _make_reshape_fn(model._n_channels, model._n_freq_bins)
    elif hasattr(model, "_n_channels") and hasattr(model, "_seq_len"):
        reshape_fn = _make_reshape_fn(model._n_channels, model._seq_len)

    if reshape_fn is not None:
        X_t = reshape_fn(X_t)

    # Enable dropout
    nn_model.train()

    all_mode_probs = []
    all_amp = []
    all_vel = []

    with torch.no_grad():
        for _ in range(n_forward_passes):
            outputs = nn_model(X_t)
            out_mode, out_whirl, out_amp, out_vel = outputs[:4]
            mode_proba = torch.softmax(out_mode, dim=1).cpu().numpy()
            all_mode_probs.append(mode_proba)
            all_amp.append(out_amp.squeeze(-1).cpu().numpy())
            all_vel.append(out_vel.squeeze(-1).cpu().numpy())

    nn_model.eval()

    all_mode_probs = np.array(all_mode_probs)  # (T, N, C)
    all_amp = np.array(all_amp)  # (T, N)
    all_vel = np.array(all_vel)  # (T, N)

    mean_mode_probs = all_mode_probs.mean(axis=0)
    mode_idx = np.argmax(mean_mode_probs, axis=1)
    confidence = np.max(mean_mode_probs, axis=1)

    # Mode entropy
    mode_entropy = -np.sum(
        mean_mode_probs * np.log(np.clip(mean_mode_probs, 1e-12, 1.0)), axis=1
    )

    # Decode mode labels
    mode_raw = np.array(
        [model._mode_label_decoder[int(i)] for i in mode_idx], dtype=np.int64
    )
    nd, nc = _decode_mode_labels(mode_raw)

    return {
        "nodal_diameter": nd,
        "nodal_circle": nc,
        "frequency": np.zeros(X.shape[0], dtype=np.float64),
        "whirl_direction": np.zeros(X.shape[0], dtype=np.int64),  # averaged
        "amplitude": all_amp.mean(axis=0),
        "wave_velocity": all_vel.mean(axis=0),
        "confidence": confidence,
        "amplitude_epistemic_var": all_amp.var(axis=0),
        "velocity_epistemic_var": all_vel.var(axis=0),
        "mode_entropy": mode_entropy,
    }


# ============================================================================
# Deep Ensembles (D5)
# ============================================================================


class DeepEnsemble:
    """Ensemble of independently trained models for uncertainty estimation.

    Parameters
    ----------
    n_members : int
        Number of ensemble members (default 5).
    tier : int
        Complexity tier to use for each member (4, 5, or 6).
    """

    def __init__(self, n_members: int = 5, tier: int = 4) -> None:
        self._n_members = n_members
        self._tier = tier
        self._members: list[Any] = []
        self._is_fitted = False

    def train(
        self,
        X: np.ndarray,
        y: dict[str, np.ndarray],
        config: Any,
    ) -> dict[str, float]:
        """Train N independent models with different random seeds."""
        self._members = []
        all_metrics: list[dict[str, float]] = []
        for i in range(self._n_members):
            np.random.seed(i * 1000)
            member = TIER_MODELS[self._tier]()
            metrics = member.train(X, y, config)
            self._members.append(member)
            all_metrics.append(metrics)
        self._is_fitted = True
        # Average training metrics
        if all_metrics:
            return {k: float(np.mean([m.get(k, 0) for m in all_metrics]))
                    for k in all_metrics[0]}
        return {}

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict with ensemble: mean + variance across members."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not trained.")
        all_preds = [m.predict(X) for m in self._members]
        # Majority vote for classification
        nd_stack = np.array([p["nodal_diameter"] for p in all_preds])
        nc_stack = np.array([p["nodal_circle"] for p in all_preds])
        whirl_stack = np.array([p["whirl_direction"] for p in all_preds])
        amp_stack = np.array([p["amplitude"] for p in all_preds])
        vel_stack = np.array([p["wave_velocity"] for p in all_preds])

        from scipy.stats import mode as _scipy_mode
        nd_vote = _scipy_mode(nd_stack, axis=0, keepdims=False).mode
        nc_vote = _scipy_mode(nc_stack, axis=0, keepdims=False).mode
        whirl_vote = _scipy_mode(whirl_stack, axis=0, keepdims=False).mode

        return {
            "nodal_diameter": nd_vote.astype(np.int64),
            "nodal_circle": nc_vote.astype(np.int64),
            "frequency": np.zeros(X.shape[0], dtype=np.float64),
            "whirl_direction": whirl_vote.astype(np.int64),
            "amplitude": amp_stack.mean(axis=0),
            "wave_velocity": vel_stack.mean(axis=0),
            "confidence": np.array([p["confidence"] for p in all_preds]).mean(axis=0),
            "amplitude_epistemic_var": amp_stack.var(axis=0),
            "velocity_epistemic_var": vel_stack.var(axis=0),
        }

    def save(self, path: str) -> None:
        """Save ensemble via joblib."""
        import joblib
        joblib.dump({"members": self._members, "tier": self._tier,
                      "n_members": self._n_members}, path)

    def load(self, path: str) -> None:
        """Load ensemble from disk."""
        import joblib
        data = joblib.load(path)
        self._members = data["members"]
        self._tier = data["tier"]
        self._n_members = data["n_members"]
        self._is_fitted = True


# ============================================================================
# Uncertainty decomposition (D7)
# ============================================================================


def predict_with_uncertainty(
    model: Any,
    X: np.ndarray,
    method: str = "mc_dropout",
    n_forward_passes: int = 30,
) -> dict[str, np.ndarray]:
    """Predict with uncertainty decomposition.

    Parameters
    ----------
    model : trained model (PyTorch-based for mc_dropout, DeepEnsemble for deep_ensemble).
    X : (N, n_features) input features.
    method : ``"mc_dropout"`` or ``"deep_ensemble"``.
    n_forward_passes : number of forward passes (mc_dropout only).

    Returns
    -------
    dict with standard prediction keys plus aleatoric/epistemic variance keys.
    """
    if method == "mc_dropout":
        preds = mc_dropout_predict(model, X, n_forward_passes)
    elif method == "deep_ensemble":
        if isinstance(model, DeepEnsemble):
            preds = model.predict(X)
        else:
            raise TypeError("deep_ensemble method requires a DeepEnsemble model.")
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mc_dropout' or 'deep_ensemble'.")

    # If heteroscedastic outputs available, decompose
    result = dict(preds)
    amp_epi = result.get("amplitude_epistemic_var", np.zeros(X.shape[0]))
    vel_epi = result.get("velocity_epistemic_var", np.zeros(X.shape[0]))

    # Aleatoric from heteroscedastic heads (if available, zeros otherwise)
    amp_ale = result.pop("amplitude_aleatoric_var", np.zeros(X.shape[0]))
    vel_ale = result.pop("velocity_aleatoric_var", np.zeros(X.shape[0]))

    result["amplitude_aleatoric_var"] = amp_ale
    result["amplitude_epistemic_var"] = amp_epi
    result["amplitude_total_var"] = amp_ale + amp_epi
    result["velocity_aleatoric_var"] = vel_ale
    result["velocity_epistemic_var"] = vel_epi
    result["velocity_total_var"] = vel_ale + vel_epi
    return result


# ============================================================================
# Registry
# ============================================================================

TIER_MODELS: dict[int, type] = {
    1: LinearModeIDModel,
    2: TreeModeIDModel,
    3: SVMModeIDModel,
    4: ShallowNNModeIDModel,
    5: CNNModeIDModel,
    6: TemporalModeIDModel,
}
