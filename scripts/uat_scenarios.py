"""Programmatic loadmaster UAT — known industry archetypes through the API.

This is the "I'm a loadmaster, here are the 6 pallets I see every day" check.
Each scenario walks the full input-engine path and asserts the output is
loadable and physically reasonable.
"""

from __future__ import annotations

import sys

import requests

API = "http://127.0.0.1:8000"

# Loadmaster scenarios drawn from typical 3PL / cold-storage operations
SCENARIOS = [
    {
        "name": "Frozen meat — realistic Ti=6, Hi=5 column on EUR (40cm cases fit 3x2)",
        "raw": {
            "barcode_skus": ["SKU-FM-001"] * 30,
            "vision": {"pattern": "column", "layers": 5, "items_per_layer": 6},
            "env": "frozen", "body_temp_c": -22.0, "seconds_since_temp_change": 3600.0,
            "pallet_id": "scenario-frozen-meat",
        },
        "expect": {"min_mass_kg": 300, "max_mass_kg": 360, "max_overhang_m": 0.05},
    },
    {
        "name": "Overcrowded pallet (anomalous) — 8 cases brick, expect detected overhang",
        "raw": {
            "barcode_skus": ["SKU-FM-001"] * 16,
            "vision": {"pattern": "brick", "layers": 2, "items_per_layer": 8},
            "env": "frozen", "body_temp_c": -22.0, "seconds_since_temp_change": 3600.0,
            "pallet_id": "scenario-overcrowded",
        },
        "expect": {"min_mass_kg": 160, "max_mass_kg": 200, "min_overhang_m": 0.05},
    },
    {
        "name": "Refrigerated dairy column — 25kg cheese wheels",
        "raw": {
            "barcode_skus": ["SKU-FD-001"] * 9,
            "vision": {"pattern": "column", "layers": 3, "items_per_layer": 3},
            "env": "refrigerated", "body_temp_c": 2.5, "seconds_since_temp_change": 7200.0,
            "pallet_id": "scenario-cheese-column",
        },
        "expect": {"min_mass_kg": 240, "max_mass_kg": 270},
    },
    {
        "name": "Transitioning frozen veg — danger zone friction (just out of freezer)",
        "raw": {
            "barcode_skus": ["SKU-FV-001"] * 24,
            "vision": {"pattern": "brick", "layers": 4, "items_per_layer": 6,
                       "lean_angle_deg": 5.0, "max_overhang_m": 0.04},
            "env": "transitioning", "body_temp_c": 4.0, "seconds_since_temp_change": 60.0,
            "pallet_id": "scenario-transitioning-veg",
        },
        "expect": {"min_mass_kg": 140, "max_mass_kg": 175, "max_mu_static": 0.30},
    },
    {
        "name": "Heavy cheese wheels — near-cap weight",
        "raw": {
            "barcode_skus": ["SKU-MS-003"] * 24,
            "vision": {"pattern": "column", "layers": 6, "items_per_layer": 4},
            "env": "refrigerated", "body_temp_c": 3.0, "seconds_since_temp_change": 7200.0,
            "pallet_id": "scenario-heavy-cheese",
        },
        "expect": {"min_mass_kg": 800, "max_mass_kg": 900},
    },
    {
        "name": "Mixed beverage pallet — pinwheel, ambient",
        "raw": {
            "barcode_skus": (["SKU-BV-001", "SKU-BV-002", "SKU-BV-003"] * 4)[:12],
            "vision": {"pattern": "pinwheel", "layers": 3, "items_per_layer": 4},
            "env": "thawed", "body_temp_c": 18.0, "seconds_since_temp_change": 7200.0,
            "pallet_id": "scenario-beverage-mixed",
        },
        "expect": {"min_mass_kg": 150, "max_mass_kg": 200, "min_mu_static": 0.45},
    },
    {
        "name": "Single-item lightweight — minimum case",
        "raw": {
            "barcode_skus": ["SKU-FV-003"],
            "vision": {"pattern": "column", "layers": 1, "items_per_layer": 1},
            "env": "frozen", "body_temp_c": -20.0, "seconds_since_temp_change": 3600.0,
            "pallet_id": "scenario-minimum",
        },
        "expect": {"min_mass_kg": 28, "max_mass_kg": 32},
    },
]


def run() -> int:
    failed = []
    for scn in SCENARIOS:
        try:
            ok = check(scn)
            if not ok:
                failed.append(scn["name"])
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed.append(scn["name"])

    print(f"\n{len(SCENARIOS) - len(failed)}/{len(SCENARIOS)} scenarios PASSED")
    if failed:
        print("\nFAILED:")
        for n in failed:
            print(f"  - {n}")
    return 0 if not failed else 1


def check(scn: dict) -> bool:
    print(f"\n[scenario] {scn['name']}")
    raw = scn["raw"]
    expect = scn["expect"]

    # 1. Build pallet via API
    r = requests.post(f"{API}/pallet/from-raw", json=raw)
    if r.status_code != 200:
        print(f"  FAIL: /pallet/from-raw {r.status_code}: {r.text[:200]}")
        return False
    cfg = r.json()
    print(f"  pallet built: {len(cfg['items'])} items, "
          f"{cfg['total_mass_kg']:.1f} kg, height {cfg['stack_height_m']:.2f} m")

    # 2. Mass bound checks
    mass = cfg["total_mass_kg"]
    if "min_mass_kg" in expect and mass < expect["min_mass_kg"]:
        print(f"  FAIL: mass {mass:.1f} < min {expect['min_mass_kg']}")
        return False
    if "max_mass_kg" in expect and mass > expect["max_mass_kg"]:
        print(f"  FAIL: mass {mass:.1f} > max {expect['max_mass_kg']}")
        return False

    # 3. Overhang check
    if "max_overhang_m" in expect and cfg["overhang_m"] > expect["max_overhang_m"]:
        print(f"  FAIL: overhang {cfg['overhang_m']*1000:.0f}mm > {expect['max_overhang_m']*1000:.0f}mm")
        return False
    if "min_overhang_m" in expect and cfg["overhang_m"] < expect["min_overhang_m"]:
        print(f"  FAIL: overhang {cfg['overhang_m']*1000:.0f}mm < expected min {expect['min_overhang_m']*1000:.0f}mm")
        return False
    if cfg["overhang_m"] > 0:
        print(f"  overhang detected: {cfg['overhang_m']*1000:.0f}mm")

    # 4. Friction check
    fr = requests.get(
        f"{API}/friction",
        params={"temp_c": cfg["body_temp_c"],
                "seconds_since_temp_change": cfg["seconds_since_temp_change"]},
    ).json()
    print(f"  friction: mu_s={fr['mu_static']:.3f}, mu_d={fr['mu_dynamic']:.3f}")
    if "max_mu_static" in expect and fr["mu_static"] > expect["max_mu_static"]:
        print(f"  FAIL: mu_s {fr['mu_static']:.3f} > expected max {expect['max_mu_static']}")
        return False
    if "min_mu_static" in expect and fr["mu_static"] < expect["min_mu_static"]:
        print(f"  FAIL: mu_s {fr['mu_static']:.3f} < expected min {expect['min_mu_static']}")
        return False

    # 5. MJCF builds and is loadable
    r = requests.post(f"{API}/mjcf/build", json=cfg)
    if r.status_code != 200:
        print(f"  FAIL: /mjcf/build {r.status_code}")
        return False
    print(f"  MJCF: {r.json()['bytes']} bytes")

    print("  PASS")
    return True


if __name__ == "__main__":
    sys.exit(run())
