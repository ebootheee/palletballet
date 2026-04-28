"""Failure-detection tests against known-stable and known-unstable configs."""

from __future__ import annotations

import pytest

from pallet_safety.failures import (
    FailureThresholds,
    detect_item_slide,
    detect_load_shift,
    detect_pallet_slip,
    detect_tip,
    first_failure,
    tip_angle_deg,
)
from pallet_safety.models import EnvCondition, FailureMode, Item, PalletConfig, WrapType
from pallet_safety.solver import ConveyorProfile, simulate


def _wide_low_pallet() -> PalletConfig:
    """Stable: wide footprint, low CoM, gentle motion."""
    return PalletConfig(
        pallet_id="stable",
        items=[Item(sku="X", weight_kg=30.0, dims_m=(0.6, 0.5, 0.15),
                    position=(0, 0, 0.15))],
        env=EnvCondition.REFRIGERATED, body_temp_c=5.0,
    )


def _tall_unwrapped_stack(n_items: int = 5) -> PalletConfig:
    """Unstable under high accel: tall narrow stack, no wrap."""
    items = []
    for i in range(n_items):
        items.append(Item(sku="X", weight_kg=10.0, dims_m=(0.3, 0.3, 0.2),
                          position=(0, 0, 0.15 + i * 0.2)))
    return PalletConfig(
        pallet_id="tall_unwrapped", items=items,
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
        wrap=WrapType.NONE,
    )


def _frozen_low_friction_pallet() -> PalletConfig:
    """Cold pallet: low friction, slip-prone."""
    return PalletConfig(
        pallet_id="cold",
        items=[Item(sku="X", weight_kg=40.0, dims_m=(0.5, 0.4, 0.2),
                    position=(0, 0, 0.15))],
        env=EnvCondition.FROZEN, body_temp_c=-25.0,
    )


# ---- single-detector tests ----

def test_stable_pallet_no_tip():
    trace = simulate(_wide_low_pallet(),
                     ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=2.0))
    assert detect_tip(trace, threshold_deg=8.0) is None
    assert tip_angle_deg(trace).max() < 1.0


def test_stable_pallet_no_item_slide():
    trace = simulate(_wide_low_pallet(),
                     ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=2.0))
    assert detect_item_slide(trace, threshold_m=0.05) is None


def test_stable_pallet_no_pallet_slip_at_low_accel():
    trace = simulate(_wide_low_pallet(),
                     ConveyorProfile(target_speed_mps=0.4, accel_mps2=0.5, duration_s=3.0))
    assert detect_pallet_slip(trace, threshold_m=0.30) is None


def test_high_accel_triggers_item_slide_or_slip():
    """Aggressive accel on a tall stack: SOMETHING fails — either items slide off
    or the pallet itself slips."""
    trace = simulate(_tall_unwrapped_stack(n_items=5),
                     ConveyorProfile(target_speed_mps=2.0, accel_mps2=5.0, duration_s=2.0))
    mode, t = first_failure(trace)
    assert mode != FailureMode.NO_FAILURE
    assert t is not None and t < 2.0


def test_frozen_pallet_slips_more_than_warm():
    """Same accel; cold pallet should slip MORE than warm one."""
    profile = ConveyorProfile(target_speed_mps=1.0, accel_mps2=2.5, duration_s=2.0)
    cold = simulate(_frozen_low_friction_pallet(), profile)
    warm_cfg = _frozen_low_friction_pallet().model_copy(
        update={"env": EnvCondition.THAWED, "body_temp_c": 20.0,
                "seconds_since_temp_change": 7200.0},
    )
    warm = simulate(warm_cfg, profile)
    # Pallet displacement should lag the belt MORE for cold
    cold_pallet_disp = cold.pallet_pos[-1, 0] - cold.pallet_pos[0, 0]
    warm_pallet_disp = warm.pallet_pos[-1, 0] - warm.pallet_pos[0, 0]
    cold_belt_disp = cold.conveyor_vel.mean() * cold.times[-1]
    warm_belt_disp = warm.conveyor_vel.mean() * warm.times[-1]
    cold_lag = cold_belt_disp - cold_pallet_disp
    warm_lag = warm_belt_disp - warm_pallet_disp
    assert cold_lag > warm_lag, f"cold lag {cold_lag:.3f} should exceed warm lag {warm_lag:.3f}"


# ---- first_failure routing ----

def test_first_failure_returns_no_failure_for_stable():
    trace = simulate(_wide_low_pallet(),
                     ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=2.0))
    mode, t = first_failure(trace)
    assert mode == FailureMode.NO_FAILURE
    assert t is None


def test_first_failure_returns_earliest_event():
    """Synthetic — verify that when multiple modes fire, the EARLIEST wins."""
    trace = simulate(_tall_unwrapped_stack(n_items=6),
                     ConveyorProfile(target_speed_mps=2.0, accel_mps2=8.0, duration_s=1.0))
    mode, t = first_failure(trace)
    assert mode != FailureMode.NO_FAILURE
    # Manually compute each detector's time and check we got the smallest
    th = FailureThresholds()
    candidates = {
        FailureMode.TIP_OVER: detect_tip(trace, th.tip_angle_deg),
        FailureMode.TOP_ITEM_SLIDE: detect_item_slide(trace, th.item_slide_m),
        FailureMode.PALLET_SLIP: detect_pallet_slip(trace, th.pallet_slip_m),
        FailureMode.LOAD_SHIFT: detect_load_shift(trace, th.load_shift_m),
    }
    finite = [(m, ti) for m, ti in candidates.items() if ti is not None]
    assert finite, "test setup expected at least one failure"
    expected_mode, expected_t = min(finite, key=lambda x: x[1])
    assert mode == expected_mode
    assert t == pytest.approx(expected_t)


def test_thresholds_are_configurable():
    """Per PLAN_V2 §13, thresholds must be configurable per scenario."""
    cfg = _tall_unwrapped_stack(3)
    trace = simulate(cfg, ConveyorProfile(target_speed_mps=1.0, accel_mps2=2.0, duration_s=2.0))
    strict = FailureThresholds(item_slide_m=0.001, tip_angle_deg=0.5)
    lax = FailureThresholds(item_slide_m=1.0, tip_angle_deg=45.0,
                              pallet_slip_m=10.0, load_shift_m=1.0)
    strict_mode, _ = first_failure(trace, strict)
    lax_mode, _ = first_failure(trace, lax)
    # Strict thresholds should detect at least as much as lax
    assert lax_mode == FailureMode.NO_FAILURE
    assert strict_mode != FailureMode.NO_FAILURE


def test_load_shift_zero_for_single_item():
    cfg = _wide_low_pallet()  # 1 item
    trace = simulate(cfg, ConveyorProfile(target_speed_mps=0.5, duration_s=1.0))
    assert detect_load_shift(trace, threshold_m=0.001) is None


def test_pallet_slip_returns_none_for_motionless_belt():
    cfg = _wide_low_pallet()
    profile = ConveyorProfile(target_speed_mps=0.0, duration_s=1.0)
    trace = simulate(cfg, profile)
    assert detect_pallet_slip(trace, threshold_m=0.001) is None
