"""Failure-mode detectors operating on a `SimulationTrace`.

Each detector returns the time of the first failure event (in seconds, post-settle)
or `None` if the failure mode never occurred during the trace. `first_failure`
returns the earliest failure across all detectors.

Default thresholds are starting points; per PLAN_V2 §13 they should be
configurable per scenario when calibrating against customer data.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .models import FailureMode
from .solver import SimulationTrace


class FailureThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    tip_angle_deg: float = Field(default=8.0, gt=0, le=45.0)
    item_slide_m: float = Field(default=0.05, gt=0, le=1.0)
    pallet_slip_m: float = Field(default=0.30, gt=0, le=10.0)
    load_shift_m: float = Field(default=0.02, gt=0, le=1.0)


def tip_angle_deg(trace: SimulationTrace) -> np.ndarray:
    """Pallet rotation off vertical at each timestep, degrees.

    quat is (w, x, y, z); the world-Z component of the body-Z axis is
    1 - 2*(x^2 + y^2). arccos(that) is the tilt angle.
    """
    q = trace.pallet_quat
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


def detect_tip(trace: SimulationTrace, threshold_deg: float) -> float | None:
    angles = tip_angle_deg(trace)
    over = np.where(angles > threshold_deg)[0]
    return float(trace.times[over[0]]) if len(over) > 0 else None


def detect_item_slide(trace: SimulationTrace, threshold_m: float) -> float | None:
    if trace.n_items == 0:
        return None
    # delta[t, k] = ||item_pallet_pos[t,k] - item_pallet_pos[0,k]||
    deltas = np.linalg.norm(
        trace.item_pallet_pos - trace.item_initial_pallet_pos[None, :, :],
        axis=2,
    )  # shape (n_steps, n_items)
    max_per_step = deltas.max(axis=1)
    over = np.where(max_per_step > threshold_m)[0]
    return float(trace.times[over[0]]) if len(over) > 0 else None


def detect_pallet_slip(trace: SimulationTrace, threshold_m: float) -> float | None:
    """Slip = abs( belt-displacement - pallet-displacement ) integrated over time.

    Belt displacement at t = ∫ conveyor_vel from 0 to t (numerical).
    Pallet displacement at t = pallet_pos.x - pallet_pos.x[0].
    """
    if len(trace.times) < 2:
        return None
    dt = np.diff(trace.times, prepend=trace.times[0])
    belt_disp = np.cumsum(trace.conveyor_vel * dt)
    pallet_disp = trace.pallet_pos[:, 0] - trace.pallet_pos[0, 0]
    # Note: conveyor moves +X, so as belt moves the pallet should follow.
    # In our model the conveyor body translates instead — so the apparent
    # relative slip is (conveyor_disp - pallet_disp). Both grow positive.
    slip = np.abs(belt_disp - pallet_disp)
    over = np.where(slip > threshold_m)[0]
    return float(trace.times[over[0]]) if len(over) > 0 else None


def detect_load_shift(trace: SimulationTrace, threshold_m: float) -> float | None:
    """Item-to-item pairwise distance change. O(n^2) per step but n is small."""
    if trace.n_items < 2:
        return None
    initial = trace.item_initial_pallet_pos
    n = trace.n_items
    # Initial pairwise distances
    init_d = np.linalg.norm(initial[:, None, :] - initial[None, :, :], axis=2)
    for step in range(trace.n_steps):
        cur = trace.item_pallet_pos[step]
        d = np.linalg.norm(cur[:, None, :] - cur[None, :, :], axis=2)
        if np.max(np.abs(d - init_d)) > threshold_m:
            return float(trace.times[step])
    return None


def first_failure(
    trace: SimulationTrace,
    thresholds: FailureThresholds = FailureThresholds(),
) -> tuple[FailureMode, float | None]:
    """Return (mode, time) of the earliest failure event."""
    candidates: list[tuple[FailureMode, float | None]] = [
        (FailureMode.TIP_OVER, detect_tip(trace, thresholds.tip_angle_deg)),
        (FailureMode.TOP_ITEM_SLIDE, detect_item_slide(trace, thresholds.item_slide_m)),
        (FailureMode.PALLET_SLIP, detect_pallet_slip(trace, thresholds.pallet_slip_m)),
        (FailureMode.LOAD_SHIFT, detect_load_shift(trace, thresholds.load_shift_m)),
    ]
    finite = [(mode, t) for mode, t in candidates if t is not None]
    if not finite:
        return FailureMode.NO_FAILURE, None
    return min(finite, key=lambda mt: mt[1])
