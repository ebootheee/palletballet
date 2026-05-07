"""Tests for the FastAPI service.

Uses TestClient (in-process) so they run fast and don't need a port.
"""

from __future__ import annotations

import mujoco
import pytest
from fastapi.testclient import TestClient

from pallet_safety.service.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_skus(client):
    r = client.get("/catalog/skus")
    assert r.status_code == 200
    skus = r.json()
    assert len(skus) >= 25
    assert all("sku" in s and "weight_kg" in s for s in skus)


def test_get_sku(client):
    r = client.get("/catalog/skus/SKU-FD-001")
    assert r.status_code == 200
    assert r.json()["sku"] == "SKU-FD-001"


def test_get_unknown_sku_404(client):
    r = client.get("/catalog/skus/NOT-REAL")
    assert r.status_code == 404


def test_random_raw_seeded_deterministic(client):
    a = client.post("/raw/random", json={"seed": 42}).json()
    b = client.post("/raw/random", json={"seed": 42}).json()
    assert a == b


def test_random_raw_unseeded_varies(client):
    a = client.post("/raw/random", json={"anomaly_rate": 0.0}).json()
    b = client.post("/raw/random", json={"anomaly_rate": 0.0}).json()
    # With None seed, results should differ
    assert a != b


def test_pallet_from_raw(client):
    raw = client.post("/raw/random", json={"seed": 1}).json()
    r = client.post("/pallet/from-raw", json=raw)
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["pallet_id"] == raw["pallet_id"]
    assert len(cfg["items"]) == raw["vision"]["layers"] * raw["vision"]["items_per_layer"]


def test_pallet_random_one_shot(client):
    r = client.post("/pallet/random", json={"seed": 5})
    assert r.status_code == 200
    cfg = r.json()
    assert "items" in cfg
    assert "total_mass_kg" in cfg


def test_pallet_validate_accepts_valid(client):
    cfg = client.post("/pallet/random", json={"seed": 7}).json()
    r = client.post("/pallet/validate", json=cfg)
    assert r.status_code == 200


def test_pallet_validate_rejects_invalid(client):
    r = client.post("/pallet/validate", json={"pallet_id": "x", "items": "not_a_list"})
    assert r.status_code == 422


def test_mjcf_build(client):
    cfg = client.post("/pallet/random", json={"seed": 9}).json()
    r = client.post("/mjcf/build", json=cfg)
    assert r.status_code == 200
    body = r.json()
    assert body["pallet_id"] == cfg["pallet_id"]
    xml = body["mjcf_xml"]
    assert "<mujoco" in xml
    # Must load in MuJoCo
    mujoco.MjModel.from_xml_string(xml)


def test_friction_lookup(client):
    r = client.get("/friction", params={"temp_c": 0.0, "seconds_since_temp_change": 0.0})
    assert r.status_code == 200
    pt = r.json()
    assert pt["mu_static"] < pt["mu_static"] + 0.1  # sanity: it's a number
    # 0°C with zero recovery should be lower than 0°C steady-state
    steady = client.get("/friction", params={"temp_c": 0.0, "seconds_since_temp_change": 3600}).json()
    assert pt["mu_static"] <= steady["mu_static"]


def test_friction_curve(client):
    r = client.get("/friction/curve", params={"n": 20})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) == 20
    assert body["points"][0]["temp_c"] < body["points"][-1]["temp_c"]


def test_friction_pairs(client):
    r = client.get("/friction/pairs")
    assert r.status_code == 200
    pairs = r.json()
    assert "wood_pallet/rubber_belt" in pairs
    assert len(pairs) >= 3


def test_random_request_validation(client):
    # max < min should 400
    r = client.post("/raw/random", json={"min_layers": 5, "max_layers": 1})
    assert r.status_code == 400


def test_solve_endpoint_stable_pallet(client):
    """End-to-end: random pallet → /solve → SafetyResponse with no failure."""
    cfg = client.post("/pallet/random", json={"seed": 11, "anomaly_rate": 0.0}).json()
    body = {
        "pallet": cfg,
        "profile": {"target_speed_mps": 0.5, "accel_mps2": 0.5, "duration_s": 2.0},
        "output_hz": 50,
    }
    r = client.post("/solve", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["pallet_id"] == cfg["pallet_id"]
    assert "failure" in out and "trace" in out
    assert len(out["trace"]["times_s"]) > 50
    assert out["runtime_ms"] > 0


def test_solve_endpoint_aggressive_accel_triggers_failure(client):
    """High accel on tall pallet: /solve should report SOME failure."""
    pallet = {
        "pallet_id": "tall-test",
        "items": [
            {"sku": "X", "weight_kg": 20.0, "dims_m": [0.3, 0.3, 0.25],
             "fragility": "rigid", "position": [0, 0, 0.15 + i * 0.25],
             "orientation_deg": 0.0}
            for i in range(5)
        ],
        "env": "refrigerated", "body_temp_c": 2.0, "wrap": "none",
    }
    body = {
        "pallet": pallet,
        "profile": {"target_speed_mps": 2.0, "accel_mps2": 6.0, "duration_s": 1.5},
    }
    r = client.post("/solve", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["failure"]["mode"] != "no_failure"
    assert out["failure"]["time_s"] is not None


def test_safety_analyze_stable_pallet(client):
    # Small homogeneous stable pallet
    cfg = {
        "pallet_id": "api-stable",
        "items": [
            {"sku": "X", "weight_kg": 15, "dims_m": [0.4, 0.4, 0.15],
             "fragility": "rigid", "position": [0, 0, 0.15], "orientation_deg": 0.0}
        ],
        "env": "refrigerated", "body_temp_c": 2.0, "wrap": "stretch",
    }
    r = client.post("/safety/analyze", json=cfg)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "result" in body
    assert body["result"]["max_speed_mps"] > 0
    assert "sweep_points" in body and len(body["sweep_points"]) >= 2


def test_safety_analyze_caches(client):
    """Two identical requests — second should have cache_hits >= 1."""
    cfg = client.post("/pallet/random", json={"seed": 7, "anomaly_rate": 0.0}).json()
    client.post("/safety/analyze", json=cfg).json()  # populate cache
    r2 = client.post("/safety/analyze", json=cfg).json()
    assert r2["cache_hits"] >= 1
    assert r2["sims_run"] == 0


def test_safety_batch(client):
    cfgs = [
        client.post("/pallet/random", json={"seed": s, "anomaly_rate": 0.0}).json()
        for s in [1, 2, 3]
    ]
    r = client.post("/safety/batch", json=cfgs)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 3
    assert all("result" in item for item in body)


def test_round_trip_random_to_mjcf(client):
    """Critical pluggability test: random adapter → configurator → MJCF, all via API."""
    raw = client.post("/raw/random", json={"seed": 100, "anomaly_rate": 0.0}).json()
    cfg = client.post("/pallet/from-raw", json=raw).json()
    mjcf = client.post("/mjcf/build", json=cfg).json()
    model = mujoco.MjModel.from_xml_string(mjcf["mjcf_xml"])
    expected_n_items = raw["vision"]["layers"] * raw["vision"]["items_per_layer"]
    # world + conveyor_body + pallet_base + items
    assert model.nbody == 3 + expected_n_items
