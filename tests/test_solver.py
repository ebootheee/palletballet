"""Tests for the MuJoCo solver — runs pallets through conveyor profiles and
captures state traces. The failure detectors are tested separately in
test_failures.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pallet_safety.models import EnvCondition, Item, PalletConfig
from pallet_safety.solver import ConveyorProfile, simulate


# ---- helpers ----

def _stable_pallet() -> PalletConfig:
    """Wide low pallet — should not tip or shift under normal conveyor motion."""
    return PalletConfig(
        pallet_id="stable",
        items=[Item(sku="X", weight_kg=20.0, dims_m=(0.5, 0.4, 0.15),
                    position=(0, 0, 0.15))],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )


def _empty_pallet() -> PalletConfig:
    return PalletConfig(
        pallet_id="empty", items=[],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )


# ---- ConveyorProfile ----

def test_profile_accel_hold_ramps_then_holds():
    p = ConveyorProfile(target_speed_mps=1.0, accel_mps2=0.5)
    assert p.velocity_at(0.0) == 0.0
    assert p.velocity_at(1.0) == pytest.approx(0.5)  # mid-ramp
    assert p.velocity_at(2.0) == pytest.approx(1.0)  # end of ramp
    assert p.velocity_at(5.0) == pytest.approx(1.0)  # holding


def test_profile_constant_jumps_to_target():
    p = ConveyorProfile(target_speed_mps=1.0, shape="constant")
    assert p.velocity_at(0.0) == 0.0
    assert p.velocity_at(0.001) == 1.0


def test_profile_ramp_decel():
    p = ConveyorProfile(target_speed_mps=1.0, accel_mps2=1.0,
                         shape="ramp_decel", decel_start_s=2.0)
    assert p.velocity_at(1.0) == pytest.approx(1.0)  # holding
    assert p.velocity_at(2.5) == pytest.approx(0.5)  # 0.5s into decel
    assert p.velocity_at(3.5) == pytest.approx(0.0)  # fully stopped


# ---- simulate() ----

def test_simulate_stable_returns_trace():
    cfg = _stable_pallet()
    profile = ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=2.0)
    trace = simulate(cfg, profile)
    assert trace.n_steps > 100
    assert trace.n_items == 1
    assert len(trace.times) == trace.n_steps
    assert trace.pallet_pos.shape == (trace.n_steps, 3)
    assert trace.item_pallet_pos.shape == (trace.n_steps, 1, 3)


def test_simulate_pallet_velocity_matches_belt_at_end():
    """Stable pallet with sufficient run time should be moving with the belt."""
    cfg = _stable_pallet()
    profile = ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=3.0)
    trace = simulate(cfg, profile)
    # Final pallet x-velocity should be close to belt velocity
    final_vel = trace.pallet_lin_vel[-1, 0]
    final_belt = trace.conveyor_vel[-1]
    assert abs(final_vel - final_belt) < 0.05  # within 5cm/s


def test_simulate_empty_pallet_does_not_crash():
    cfg = _empty_pallet()
    profile = ConveyorProfile(target_speed_mps=0.3, duration_s=1.0)
    trace = simulate(cfg, profile)
    assert trace.n_items == 0
    assert trace.n_steps > 50


def test_simulate_runtime_per_step_under_budget():
    """Performance gate: < 1ms per simulated millisecond for a small pallet."""
    cfg = _stable_pallet()
    profile = ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=2.0)
    t0 = time.perf_counter()
    simulate(cfg, profile)
    wall_ms = (time.perf_counter() - t0) * 1000
    sim_ms = profile.duration_s * 1000
    # On dev machine, 2-second sim of 1-item pallet should be under 100ms wall
    assert wall_ms < 250, f"sim took {wall_ms:.0f}ms wall for {sim_ms:.0f}ms sim"


def test_simulate_initial_item_position_captured():
    cfg = PalletConfig(
        pallet_id="multi",
        items=[
            Item(sku="A", weight_kg=10, dims_m=(0.3, 0.3, 0.2), position=(-0.2, 0, 0.15)),
            Item(sku="B", weight_kg=10, dims_m=(0.3, 0.3, 0.2), position=(0.2, 0, 0.15)),
        ],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    profile = ConveyorProfile(target_speed_mps=0.3, accel_mps2=0.5, duration_s=1.0)
    trace = simulate(cfg, profile)
    assert trace.item_initial_pallet_pos.shape == (2, 3)
    # Items started symmetric about x=0
    assert trace.item_initial_pallet_pos[0, 0] < 0
    assert trace.item_initial_pallet_pos[1, 0] > 0


def test_simulate_downsample():
    cfg = _stable_pallet()
    profile = ConveyorProfile(target_speed_mps=0.5, duration_s=2.0)
    trace = simulate(cfg, profile)
    ds = trace.downsample(hz=50)
    assert ds.n_steps < trace.n_steps
    assert ds.times.shape == (ds.n_steps,)
    # Spacing should be ~1/50 = 20ms
    spacings = np.diff(ds.times)
    assert spacings.mean() == pytest.approx(0.02, abs=0.005)


def test_simulate_belt_actually_moves():
    """Without an actuator, conveyor_vel would stay 0. This proves it's wired."""
    cfg = _stable_pallet()
    profile = ConveyorProfile(target_speed_mps=0.8, accel_mps2=0.5, duration_s=3.0)
    trace = simulate(cfg, profile)
    assert trace.conveyor_vel[-1] > 0.5
    assert trace.conveyor_vel[-1] < 1.0
