"""Temperature-dependent friction model for pallet-on-conveyor surfaces.

Loads a calibration table from `data/friction_table.json` once. Provides:

    friction_coefficient(body_temp_c, seconds_since_temp_change, surface_pair)
        -> (mu_static, mu_dynamic)

Steady-state values are linearly interpolated against the table. A "transition
penalty" multiplier captures the frost-melt regime: a recently-warmed pallet in
the dangerous near-freezing range has dramatically reduced friction.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import numpy as np

DEFAULT_PAIR = ("wood_pallet", "rubber_belt")
_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "friction_table.json"


@cache
def _load_table() -> dict:
    with _DATA_PATH.open() as f:
        return json.load(f)


def _key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{pair[1]}"


def steady_state_mu(temp_c: float, surface_pair: tuple[str, str] = DEFAULT_PAIR) -> tuple[float, float]:
    """Steady-state (mu_s, mu_d) at the given temperature, no transition effect."""
    table = _load_table()
    pairs = table["surface_pairs"]
    key = _key(surface_pair)
    if key not in pairs:
        raise KeyError(f"unknown surface pair {key!r}; have {list(pairs)}")
    pts = pairs[key]["control_points"]
    temps = np.array([p["temp_c"] for p in pts])
    mu_s = np.array([p["mu_s"] for p in pts])
    mu_d = np.array([p["mu_d"] for p in pts])
    return float(np.interp(temp_c, temps, mu_s)), float(np.interp(temp_c, temps, mu_d))


def transition_penalty(temp_c: float, seconds_since_temp_change: float) -> float:
    """Multiplier in (0, 1] capturing frost-melt friction loss.

    1.0 outside the danger zone or after recovery. Drops linearly toward
    `min_multiplier` at the center of the danger zone with zero recovery time.
    """
    cfg = _load_table()["transition_penalty"]
    lo, hi = cfg["danger_zone_temp_c"]
    min_mult = cfg["min_multiplier"]
    recovery_s = cfg["recovery_seconds"]

    if temp_c < lo or temp_c > hi:
        return 1.0
    # Position within the danger zone, 0 at edges, 1 at center
    center = (lo + hi) / 2.0
    half_width = (hi - lo) / 2.0
    zone_factor = 1.0 - abs(temp_c - center) / half_width  # 0..1
    # Recovery: 0 seconds = full penalty, recovery_s = no penalty
    recovery_factor = max(0.0, 1.0 - seconds_since_temp_change / recovery_s)  # 0..1
    severity = zone_factor * recovery_factor  # 0..1
    return 1.0 - severity * (1.0 - min_mult)


def friction_coefficient(
    body_temp_c: float,
    seconds_since_temp_change: float = 3600.0,
    surface_pair: tuple[str, str] = DEFAULT_PAIR,
) -> tuple[float, float]:
    """Return (mu_static, mu_dynamic) including the transition-zone penalty."""
    mu_s, mu_d = steady_state_mu(body_temp_c, surface_pair)
    mult = transition_penalty(body_temp_c, seconds_since_temp_change)
    return mu_s * mult, mu_d * mult


def available_surface_pairs() -> list[tuple[str, str]]:
    """All pairs defined in the calibration table."""
    return [tuple(k.split("/")) for k in _load_table()["surface_pairs"]]
