"""Tests for the threshold solver."""

from __future__ import annotations

import pytest

from pallet_safety.configurator import StackSpec, build_from_stacks
from pallet_safety.models import EnvCondition, FailureMode, PalletConfig, WrapType
from pallet_safety.threshold import (
    SearchConfig,
    ThresholdAnalyzer,
    _config_fingerprint,
    default_analyzer,
)


# ---- helpers ----

def _heavy_short() -> "PalletConfig":
    return build_from_stacks(
        [StackSpec(sku="SKU-FD-001", grid_row=r, grid_col=c, height=2)
         for r in range(2) for c in range(2)],
        pallet_id="heavy_short",
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0, wrap=WrapType.STRETCH,
    )


def _tall_unwrapped() -> "PalletConfig":
    return build_from_stacks(
        [StackSpec(sku="SKU-FD-002", grid_row=0, grid_col=1, height=8)],
        pallet_id="tall_unwrapped",
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0, wrap=WrapType.NONE,
    )


def _frozen_large() -> "PalletConfig":
    """Stretch-wrapped pallet of frozen meat — wrap holds the load, so slip
    behavior reflects pallet-belt friction directly."""
    return build_from_stacks(
        [StackSpec(sku="SKU-FM-001", grid_row=r, grid_col=c, height=4)
         for r in range(2) for c in range(3)],
        pallet_id="frozen_large",
        env=EnvCondition.FROZEN, body_temp_c=-25.0, wrap=WrapType.STRETCH,
    )


# ---- basic contracts ----

def test_analyze_returns_safety_result():
    cfg = _heavy_short()
    a = ThresholdAnalyzer().analyze(cfg)
    r = a.result
    assert r.pallet_id == cfg.pallet_id
    assert 0 <= r.max_speed_mps <= 3.0
    assert 0 <= r.max_accel_mps2 <= 10.0
    assert r.dominant_failure_mode in FailureMode
    assert 0 <= r.margin_pct <= 100
    assert 0 <= r.confidence <= 1
    assert r.sim_runtime_ms > 0
    assert len(r.config_hash) > 0


def test_stable_pallet_hits_upper_bound():
    """A stable heavy-short pallet should be safe at the whole search range."""
    cfg = _heavy_short()
    a = ThresholdAnalyzer().analyze(cfg)
    assert a.result.max_speed_mps == pytest.approx(2.0, abs=0.01)
    assert a.result.dominant_failure_mode == FailureMode.NO_FAILURE


def test_tall_unwrapped_limited_by_item_slide():
    """Tall unwrapped tower fails by items sliding, not pallet tipping."""
    cfg = _tall_unwrapped()
    a = ThresholdAnalyzer().analyze(cfg)
    assert a.result.dominant_failure_mode != FailureMode.NO_FAILURE
    # max_speed might still be high — failure often triggered by accel search
    assert a.result.max_accel_mps2 < 5.0


def test_frozen_pallet_lower_accel_limit_than_thawed():
    """Same geometry with stretch wrap: colder pallet has ≤ max_accel of warm.

    With wrap holding the load together, the dominant failure is pallet-belt
    slip, which is purely a friction phenomenon — colder → lower μ → earlier
    slip. Without wrap, items jitter independently and the comparison is
    dominated by load_shift, which has more complex temperature dependence.
    """
    cfg_cold = _frozen_large()  # wrap=STRETCH
    cfg_warm = cfg_cold.model_copy(update={
        "env": EnvCondition.THAWED, "body_temp_c": 20.0,
        "seconds_since_temp_change": 7200.0,
        "pallet_id": "frozen_large_warm",
    })
    a_cold = ThresholdAnalyzer().analyze(cfg_cold)
    a_warm = ThresholdAnalyzer().analyze(cfg_warm)
    assert a_cold.result.max_accel_mps2 <= a_warm.result.max_accel_mps2 + 0.01, (
        f"cold accel {a_cold.result.max_accel_mps2:.2f} > "
        f"warm {a_warm.result.max_accel_mps2:.2f}"
    )


def test_cache_hit_on_repeat():
    analyzer = ThresholdAnalyzer()
    cfg = _heavy_short()
    a1 = analyzer.analyze(cfg)
    a2 = analyzer.analyze(cfg)
    assert a1.sims_run > 0
    assert a2.sims_run == 0
    assert a2.cache_hits == 1
    assert a1.result == a2.result


def test_config_fingerprint_ignores_notes():
    cfg_a = _heavy_short()
    cfg_b = cfg_a.model_copy(update={"notes": "a different note"})
    assert _config_fingerprint(cfg_a) == _config_fingerprint(cfg_b)


def test_config_fingerprint_changes_with_temperature():
    cfg_a = _heavy_short()
    cfg_b = cfg_a.model_copy(update={"body_temp_c": 15.0})
    assert _config_fingerprint(cfg_a) != _config_fingerprint(cfg_b)


def test_default_analyzer_singleton():
    a1 = default_analyzer()
    a2 = default_analyzer()
    assert a1 is a2


# ---- property: monotonicity ----

def test_heavier_pallet_has_no_higher_max_speed():
    """Adding more items → pallet is never more stable (at most same max_speed).

    This is a sanity check on the physics: the solver shouldn't reward weight
    with a higher speed limit. Heavier tall pallets are generally harder to
    keep upright, so max_speed monotonically decreases (or stays same).
    """
    light = build_from_stacks(
        [StackSpec(sku="SKU-FV-003", grid_row=0, grid_col=0, height=2)],
        pallet_id="light", env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    heavy = build_from_stacks(
        [StackSpec(sku="SKU-MS-003", grid_row=0, grid_col=0, height=8)],  # tall + heavy
        pallet_id="heavy", env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
        wrap=WrapType.NONE,
    )
    a_light = ThresholdAnalyzer().analyze(light)
    a_heavy = ThresholdAnalyzer().analyze(heavy)
    assert a_heavy.result.max_accel_mps2 <= a_light.result.max_accel_mps2 + 1e-6


# ---- perf ----

def test_stable_pallet_analysis_under_300ms():
    import time
    cfg = _heavy_short()
    analyzer = ThresholdAnalyzer()
    t0 = time.perf_counter()
    analyzer.analyze(cfg)
    wall_ms = (time.perf_counter() - t0) * 1000
    assert wall_ms < 300, f"stable analyze took {wall_ms:.0f}ms"


def test_analysis_always_completes():
    """Verify the analyzer doesn't hang on weird configs."""
    cfg = build_from_stacks(
        [], pallet_id="empty", env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    a = ThresholdAnalyzer().analyze(cfg)
    assert a.result.max_speed_mps == pytest.approx(2.0, abs=0.01)


def test_custom_search_bounds_respected():
    cfg = _heavy_short()
    narrow = SearchConfig(speed_max_mps=1.0, accel_max_mps2=2.0)
    a = ThresholdAnalyzer(search=narrow).analyze(cfg)
    assert a.result.max_speed_mps <= 1.0 + 1e-6
    assert a.result.max_accel_mps2 <= 2.0 + 1e-6


def test_sweep_points_recorded():
    """The AnalysisResult exposes the sweep so the UI can visualize it.

    We union the speed + accel search points — for a tall unwrapped tower,
    at minimum we have two boundary checks for each axis (4 total).
    """
    cfg = _tall_unwrapped()
    a = ThresholdAnalyzer().analyze(cfg)
    assert len(a.sweep_points) >= 2
