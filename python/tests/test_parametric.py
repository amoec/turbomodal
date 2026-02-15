"""Tests for turbomodal.parametric module."""

import numpy as np
import pytest

from turbomodal.parametric import (
    ParametricRange,
    ParametricSweepConfig,
    generate_conditions,
)


# ---- ParametricRange ----

def test_parametric_range_fields():
    pr = ParametricRange(name="rpm", low=1000.0, high=5000.0)
    assert pr.name == "rpm"
    assert pr.low == 1000.0
    assert pr.high == 5000.0
    assert pr.log_scale is False


def test_parametric_range_log_scale():
    pr = ParametricRange(name="pressure_ratio", low=0.1, high=10.0, log_scale=True)
    assert pr.log_scale is True


# ---- ParametricSweepConfig ----

def test_sweep_config_defaults():
    cfg = ParametricSweepConfig()
    assert cfg.num_samples == 1000
    assert cfg.sampling_method == "lhs"
    assert cfg.seed == 42
    assert cfg.num_modes == 10
    assert cfg.include_mistuning is False
    assert cfg.mistuning_sigma == 0.02
    assert cfg.ranges == []


# ---- generate_conditions ----

def test_generate_conditions_count():
    cfg = ParametricSweepConfig(
        ranges=[ParametricRange("rpm", 1000.0, 5000.0)],
        num_samples=50,
    )
    conditions = generate_conditions(cfg)
    assert len(conditions) == 50


def test_generate_conditions_bounds():
    cfg = ParametricSweepConfig(
        ranges=[ParametricRange("rpm", 1000.0, 5000.0)],
        num_samples=100,
    )
    conditions = generate_conditions(cfg)
    for cond in conditions:
        assert 1000.0 <= cond.rpm <= 5000.0


def test_generate_conditions_unique_ids():
    cfg = ParametricSweepConfig(
        ranges=[ParametricRange("rpm", 1000.0, 5000.0)],
        num_samples=50,
    )
    conditions = generate_conditions(cfg)
    ids = [c.condition_id for c in conditions]
    assert len(set(ids)) == len(ids)


def test_generate_conditions_log_scale():
    cfg = ParametricSweepConfig(
        ranges=[
            ParametricRange("rpm", 1000.0, 5000.0),
            ParametricRange("pressure_ratio", 0.1, 100.0, log_scale=True),
        ],
        num_samples=100,
    )
    conditions = generate_conditions(cfg)
    for cond in conditions:
        assert cond.pressure_ratio > 0
        assert 0.1 <= cond.pressure_ratio <= 100.0


def test_generate_conditions_multiple_params():
    cfg = ParametricSweepConfig(
        ranges=[
            ParametricRange("rpm", 1000.0, 5000.0),
            ParametricRange("temperature", 293.0, 800.0),
        ],
        num_samples=50,
    )
    conditions = generate_conditions(cfg)
    assert len(conditions) == 50
    for cond in conditions:
        assert 1000.0 <= cond.rpm <= 5000.0
        assert 293.0 <= cond.temperature <= 800.0


def test_generate_conditions_invalid_param():
    cfg = ParametricSweepConfig(
        ranges=[ParametricRange("invalid_param", 0.0, 1.0)],
        num_samples=10,
    )
    with pytest.raises(ValueError, match="Unknown parameter name"):
        generate_conditions(cfg)


def test_generate_conditions_reproducible():
    cfg = ParametricSweepConfig(
        ranges=[ParametricRange("rpm", 1000.0, 5000.0)],
        num_samples=20,
        seed=42,
    )
    c1 = generate_conditions(cfg)
    c2 = generate_conditions(cfg)
    for a, b in zip(c1, c2):
        assert a.rpm == b.rpm


def test_generate_conditions_empty_ranges():
    cfg = ParametricSweepConfig(ranges=[], num_samples=10)
    with pytest.raises(ValueError, match="At least one"):
        generate_conditions(cfg)
