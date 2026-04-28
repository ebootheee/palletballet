"""Tests for pallet_safety.friction — temperature → friction lookup."""

from __future__ import annotations

import pytest

from pallet_safety.friction import (
    DEFAULT_PAIR,
    available_surface_pairs,
    friction_coefficient,
    steady_state_mu,
    transition_penalty,
)


# --- steady-state interpolation ---

def test_steady_state_returns_two_floats():
    mu_s, mu_d = steady_state_mu(0.0)
    assert isinstance(mu_s, float)
    assert isinstance(mu_d, float)


def test_steady_state_static_exceeds_dynamic():
    """Across the operating range, mu_s > mu_d (always, by physics)."""
    for t in [-25, -15, -5, 0, 5, 15, 25]:
        mu_s, mu_d = steady_state_mu(float(t))
        assert mu_s > mu_d, f"failed at T={t}: mu_s={mu_s}, mu_d={mu_d}"


def test_steady_state_thawed_exceeds_frozen():
    """Outside the transition dip, warmer = more friction for wood/rubber."""
    mu_frozen_s, _ = steady_state_mu(-25.0)
    mu_thawed_s, _ = steady_state_mu(25.0)
    assert mu_thawed_s > mu_frozen_s


def test_steady_state_dip_at_zero():
    """The 0°C 'wet frost' point should show a local minimum vs. -5°C.

    This is the published, observed behavior of frost-melt friction loss and
    is encoded into the calibration table.
    """
    mu_neg5, _ = steady_state_mu(-5.0)
    mu_zero, _ = steady_state_mu(0.0)
    assert mu_zero < mu_neg5


def test_steady_state_interpolates_between_control_points():
    """At T = -10 (midpoint of -15 and -5), mu should be roughly average."""
    mu_neg15, _ = steady_state_mu(-15.0)
    mu_neg5, _ = steady_state_mu(-5.0)
    mu_neg10, _ = steady_state_mu(-10.0)
    assert mu_neg10 == pytest.approx((mu_neg15 + mu_neg5) / 2.0, abs=0.005)


def test_unknown_surface_pair_raises():
    with pytest.raises(KeyError):
        steady_state_mu(0.0, ("aluminum_pallet", "ice"))


# --- transition penalty ---

def test_transition_penalty_outside_zone_is_one():
    assert transition_penalty(-25.0, 0.0) == 1.0
    assert transition_penalty(20.0, 0.0) == 1.0


def test_transition_penalty_max_severity_at_zone_center_zero_time():
    # Zone is [-2, 8], center = 3; min_multiplier = 0.65 in the table
    assert transition_penalty(3.0, 0.0) == pytest.approx(0.65, abs=0.001)


def test_transition_penalty_recovers_with_time():
    p_fresh = transition_penalty(3.0, 0.0)
    p_aged = transition_penalty(3.0, 600.0)  # full recovery window
    assert p_aged > p_fresh
    assert p_aged == pytest.approx(1.0)


def test_transition_penalty_partial_recovery():
    p_full = transition_penalty(3.0, 0.0)
    p_half = transition_penalty(3.0, 300.0)
    assert p_full < p_half < 1.0


# --- combined friction_coefficient ---

def test_friction_coefficient_returns_pair():
    result = friction_coefficient(2.0)
    assert isinstance(result, tuple) and len(result) == 2


def test_friction_coefficient_applies_penalty():
    """A frozen pallet just placed in the danger zone has lower mu than at steady state."""
    fresh = friction_coefficient(3.0, seconds_since_temp_change=0.0)
    aged = friction_coefficient(3.0, seconds_since_temp_change=3600.0)
    assert fresh[0] < aged[0]
    assert fresh[1] < aged[1]


def test_friction_coefficient_steady_outside_zone():
    """Outside the transition zone the time argument is irrelevant."""
    a = friction_coefficient(-25.0, 0.0)
    b = friction_coefficient(-25.0, 99999.0)
    assert a == b


def test_default_pair_is_wood_rubber():
    assert DEFAULT_PAIR == ("wood_pallet", "rubber_belt")


def test_available_pairs_includes_default():
    pairs = available_surface_pairs()
    assert DEFAULT_PAIR in pairs
    assert len(pairs) >= 3  # wood/rubber, plastic/rubber, wood/steel


# --- realism sanity ---

def test_published_frozen_value_in_range():
    """Frozen wood-on-rubber should land in the literature range ~0.2-0.3."""
    mu_s, _ = friction_coefficient(-20.0, seconds_since_temp_change=3600.0)
    assert 0.20 <= mu_s <= 0.32


def test_published_thawed_value_in_range():
    """Thawed wood-on-rubber should land ~0.4-0.55."""
    mu_s, _ = friction_coefficient(20.0, seconds_since_temp_change=3600.0)
    assert 0.40 <= mu_s <= 0.55
