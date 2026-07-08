"""Tests for the canonical scenario registry and its API surface,
plus the /solve replay payload the web demo renders from.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from pallet_safety.scenarios import (
    all_scenarios,
    get_scenario,
    get_scenario_by_name,
    scenario_slugs,
)
from pallet_safety.service.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---- registry ----

def test_scenario_slugs_unique_and_stable():
    slugs = scenario_slugs()
    assert len(slugs) == len(set(slugs))
    # The web demo hardcodes these; renaming is a breaking change.
    assert "stable-dairy-slab" in slugs
    assert "tall-unwrapped-tower" in slugs
    assert "random-scanner-feed" in slugs


def test_all_scenarios_build_valid_pallets():
    for s in all_scenarios():
        assert s.pallet.total_mass_kg > 0
        assert s.pallet.stack_height_m > 0
        assert s.suggested_profile.target_speed_mps > 0


def test_random_scenario_is_deterministic():
    a = get_scenario("random-scanner-feed").pallet
    b = get_scenario("random-scanner-feed").pallet
    assert a.model_dump() == b.model_dump()


def test_tower_is_actually_tall_and_unwrapped():
    tower = get_scenario("tall-unwrapped-tower").pallet
    assert len(tower.items) == 8
    assert tower.wrap.value == "none"
    assert tower.stack_height_m > 0.9


def test_get_scenario_by_name_matches_slug():
    assert get_scenario_by_name("Stable dairy slab").slug == "stable-dairy-slab"
    with pytest.raises(KeyError):
        get_scenario("not-a-scenario")


# ---- API: GET /scenarios ----

def test_list_scenarios_endpoint(client):
    r = client.get("/scenarios")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == len(all_scenarios())
    for row in rows:
        assert row["slug"]
        assert row["item_count"] >= 1
        assert row["total_mass_kg"] > 0


def test_scenario_detail_endpoint(client):
    r = client.get("/scenarios/tall-unwrapped-tower")
    assert r.status_code == 200
    s = r.json()
    assert s["name"] == "Tall unwrapped tower"
    assert len(s["pallet"]["items"]) == 8
    assert s["suggested_profile"]["target_speed_mps"] == 2.0


def test_scenario_detail_404(client):
    assert client.get("/scenarios/nope").status_code == 404


# ---- API: /solve replay ----

def test_solve_without_replay_omits_it(client):
    s = client.get("/scenarios/stable-dairy-slab").json()
    r = client.post("/solve", json={
        "pallet": s["pallet"],
        "profile": {"target_speed_mps": 0.5, "accel_mps2": 0.5, "duration_s": 1.0},
    })
    assert r.status_code == 200
    assert r.json()["replay"] is None


def test_solve_replay_shapes_are_consistent(client):
    s = client.get("/scenarios/tall-unwrapped-tower").json()
    r = client.post("/solve", json={
        "pallet": s["pallet"],
        "profile": s["suggested_profile"],
        "include_replay": True,
        "output_hz": 25,
    })
    assert r.status_code == 200
    body = r.json()
    replay = body["replay"]
    assert replay is not None

    n_frames = len(replay["times_s"])
    n_items = len(replay["items"])
    assert n_frames == len(body["trace"]["times_s"])
    assert n_items == len(s["pallet"]["items"])
    assert len(replay["belt_disp_m"]) == n_frames
    assert len(replay["pallet_pos_m"]) == n_frames
    assert len(replay["pallet_quat_wxyz"]) == n_frames
    assert len(replay["item_pos_m"]) == n_frames
    assert len(replay["item_quat_wxyz"]) == n_frames
    assert all(len(frame) == n_items for frame in replay["item_pos_m"])
    assert all(len(frame) == n_items for frame in replay["item_quat_wxyz"])

    # Quaternions come straight from MuJoCo — unit norm within tolerance.
    for frame in replay["item_quat_wxyz"][:: max(1, n_frames // 5)]:
        for q in frame:
            assert math.isclose(sum(c * c for c in q), 1.0, abs_tol=1e-3)

    # Belt displacement is monotone non-decreasing under a forward profile.
    disp = replay["belt_disp_m"]
    assert all(b - a >= -1e-6 for a, b in zip(disp, disp[1:]))

    # Item geometry carries render metadata.
    assert replay["items"][0]["sku"] == "SKU-FD-002"
    assert replay["items"][0]["category"] == "fresh_dairy"


def test_solve_replay_tower_actually_fails(client):
    """The tower at its suggested (aggressive) profile must fail — that's the demo."""
    s = client.get("/scenarios/tall-unwrapped-tower").json()
    r = client.post("/solve", json={
        "pallet": s["pallet"],
        "profile": s["suggested_profile"],
        "include_replay": True,
    })
    assert r.json()["failure"]["mode"] != "no_failure"
