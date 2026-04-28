"""Persona-driven UAT: walks the full system through the eyes of 4 users.

Personas:
  1. Loadmaster — builds pallets by hand, uses the UI to check safety
  2. Batch analyst — hits /safety/batch for dozens of pallets, analyzes stats
  3. Operations — single-pallet /safety/analyze, measures latency
  4. Physics validator — compares related configs to check monotonicity

Each persona either exercises the live UI or the live API and reports findings.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_personas"
URL = "http://localhost:8501"
API = "http://127.0.0.1:8000"


class Report:
    def __init__(self, persona: str):
        self.persona = persona
        self.findings: list[tuple[str, bool, str]] = []

    def check(self, description: str, condition: bool, detail: str = ""):
        self.findings.append((description, condition, detail))
        status = "PASS" if condition else "FAIL"
        line = f"  [{status}] {description}"
        if detail:
            line += f" — {detail}"
        print(line)

    def summary(self) -> tuple[int, int]:
        passed = sum(1 for _, ok, _ in self.findings if ok)
        failed = len(self.findings) - passed
        return passed, failed


# -----------------------------------------------------------------------------
# Persona 1: Loadmaster
# -----------------------------------------------------------------------------

def persona_loadmaster(page) -> Report:
    r = Report("Loadmaster")
    print(f"\n=== Persona: {r.persona} ===")

    # Navigate to Manual configurator
    page.locator('label:has-text("Manual configurator")').first.click()
    page.locator('h1:has-text("Manual configurator")').first.wait_for(timeout=15000)
    page.locator('button:has-text("fill all cells")').first.wait_for(timeout=15000)
    page.wait_for_timeout(2000)

    # Clear any old state
    page.locator('button:has-text("clear")').first.click()
    page.wait_for_timeout(500)

    # Fill all cells with the default SKU — exercises the grid sizing
    page.locator('button:has-text("fill all cells")').first.click()
    page.wait_for_timeout(2500)

    shot(page, "persona1_loadmaster_filled")

    # Check: the pallet has stacks and numbers make sense
    total_mass_elem = page.locator('text=/^\\d+\\.\\d+ kg$/').first
    mass_text = total_mass_elem.text_content() or "0 kg"
    mass_kg = float(mass_text.replace(" kg", "").strip())
    r.check("fill all cells produces realistic weight", 200 <= mass_kg <= 1200,
            f"got {mass_kg:.0f}kg")

    # Go to Safety Analysis and set source = From Manual page
    page.locator('label:has-text("Safety Analysis")').first.click()
    page.locator('h1:has-text("Safety Analysis")').first.wait_for(timeout=15000)
    page.wait_for_timeout(2000)

    # Click "From Manual page" radio
    try:
        page.locator('label:has-text("From Manual page")').first.click()
        page.wait_for_timeout(1500)
    except Exception:
        pass

    # Trigger analysis (auto-analyze should fire but click Analyze to be sure)
    try:
        page.locator('button:has-text("Analyze now")').first.click()
    except Exception:
        pass
    # Wait for result metrics
    page.locator('text=/Max speed/').first.wait_for(timeout=30000)
    page.wait_for_timeout(2000)
    shot(page, "persona1_loadmaster_analyzed")

    # Verify max_speed value is displayed
    page_text = page.content()
    r.check("Safety Analysis shows a max-speed metric", "Max speed" in page_text)
    r.check("Safety Analysis shows a max-accel metric", "Max accel" in page_text)

    return r


# -----------------------------------------------------------------------------
# Persona 2: Batch Analyst
# -----------------------------------------------------------------------------

def persona_batch_analyst() -> Report:
    r = Report("Batch Analyst")
    print(f"\n=== Persona: {r.persona} ===")

    # Generate 15 random pallets via API
    pallets = []
    for seed in range(100, 115):
        resp = requests.post(f"{API}/pallet/random", json={"seed": seed, "anomaly_rate": 0.0})
        pallets.append(resp.json())
    r.check("15 random pallets generated", len(pallets) == 15)

    # Submit to /safety/batch
    t0 = time.perf_counter()
    batch_resp = requests.post(f"{API}/safety/batch", json=pallets)
    batch_ms = (time.perf_counter() - t0) * 1000
    r.check("batch endpoint returns 200", batch_resp.status_code == 200, f"{batch_ms:.0f}ms total")
    results = batch_resp.json()

    speeds = [item["result"]["max_speed_mps"] for item in results]
    accels = [item["result"]["max_accel_mps2"] for item in results]
    runtimes = [item["result"]["sim_runtime_ms"] for item in results]

    r.check("all results have max_speed > 0", all(s > 0 for s in speeds),
            f"min={min(speeds):.2f}, max={max(speeds):.2f}")
    r.check("all results have max_accel > 0", all(a > 0 for a in accels))

    p50 = statistics.median(runtimes)
    p95 = sorted(runtimes)[int(0.95 * len(runtimes))]
    r.check("p50 per-pallet latency ≤ 800ms", p50 <= 800, f"p50={p50:.0f}ms")
    r.check("p95 per-pallet latency ≤ 1500ms", p95 <= 1500, f"p95={p95:.0f}ms")

    print(f"    speed distribution: min={min(speeds):.2f} median={statistics.median(speeds):.2f} max={max(speeds):.2f}")
    print(f"    accel distribution: min={min(accels):.2f} median={statistics.median(accels):.2f} max={max(accels):.2f}")

    # Re-submit the same batch — should hit cache (near-zero latency)
    t0 = time.perf_counter()
    batch_resp2 = requests.post(f"{API}/safety/batch", json=pallets)
    cached_ms = (time.perf_counter() - t0) * 1000
    results2 = batch_resp2.json()
    total_cache_hits = sum(item["cache_hits"] for item in results2)
    r.check("batch cache works — all 15 cached", total_cache_hits >= 15,
            f"{total_cache_hits} hits, {cached_ms:.0f}ms total")

    return r


# -----------------------------------------------------------------------------
# Persona 3: Operations (real-time inference)
# -----------------------------------------------------------------------------

def persona_operations() -> Report:
    r = Report("Operations")
    print(f"\n=== Persona: {r.persona} ===")

    # Use a seed chosen from the current time to guarantee a fresh config.
    # In production this is what a real pallet-scan event looks like: each
    # pallet is a unique combination that's never been analyzed before.
    fresh_seed = int(time.time() * 1000) % 1_000_000
    cfg = requests.post(
        f"{API}/pallet/random",
        json={"seed": fresh_seed, "anomaly_rate": 0.0},
    ).json()

    t0 = time.perf_counter()
    resp = requests.post(f"{API}/safety/analyze", json=cfg)
    cold_ms = (time.perf_counter() - t0) * 1000
    r.check("/safety/analyze returns 200", resp.status_code == 200)
    body = resp.json()
    r.check("SafetyResult has required fields",
            all(k in body["result"] for k in [
                "max_speed_mps", "max_accel_mps2", "dominant_failure_mode",
                "margin_pct", "confidence", "config_hash",
            ]))
    r.check("cold-cache latency ≤ 2s p99 (realistic for fresh pallet)",
            cold_ms <= 2000, f"{cold_ms:.0f}ms")
    r.check("analysis did actual work (cache_hits == 0 on fresh config)",
            body["cache_hits"] == 0)

    # Warm-cache hit
    t0 = time.perf_counter()
    resp2 = requests.post(f"{API}/safety/analyze", json=cfg)
    warm_ms = (time.perf_counter() - t0) * 1000
    body2 = resp2.json()
    r.check("warm-cache latency ≤ 50ms", warm_ms <= 50, f"{warm_ms:.1f}ms")
    r.check("cache hit recorded on repeat", body2["cache_hits"] >= 1)

    # Operational reality: 10 unique pallets in sequence (typical 30s conveyor window)
    latencies = []
    for i in range(10):
        s = int(time.time() * 1000 + i * 7919) % 1_000_000
        c = requests.post(f"{API}/pallet/random",
                           json={"seed": s, "anomaly_rate": 0.0}).json()
        t0 = time.perf_counter()
        requests.post(f"{API}/safety/analyze", json=c)
        latencies.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    r.check("operational p50 ≤ 600ms (10 unique fresh pallets)", p50 <= 600,
            f"p50={p50:.0f}ms")
    r.check("operational p95 ≤ 1500ms", p95 <= 1500, f"p95={p95:.0f}ms")

    return r


# -----------------------------------------------------------------------------
# Persona 4: Physics Validator
# -----------------------------------------------------------------------------

def persona_physics_validator() -> Report:
    r = Report("Physics Validator")
    print(f"\n=== Persona: {r.persona} ===")

    # Use configs calibrated to FAIL in the search range so we can observe
    # monotonicity. Pallets that all hit upper bounds reveal nothing.

    from pallet_safety.configurator import StackSpec, build_from_stacks
    from pallet_safety.models import EnvCondition, WrapType

    def _analyze_cfg(cfg_obj) -> dict:
        return requests.post(f"{API}/safety/analyze",
                              json=cfg_obj.model_dump(mode="json")).json()["result"]

    # -- 1. Wrap comparison: same pallet with wrap=NONE vs wrap=STRETCH --
    stacks = [StackSpec(sku="SKU-FM-002", grid_row=0, grid_col=0, height=6)]  # 6-tall tower
    no_wrap = build_from_stacks(stacks, pallet_id="phys_no_wrap",
                                 env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
                                 wrap=WrapType.NONE)
    stretched = no_wrap.model_copy(update={"pallet_id": "phys_stretched", "wrap": "stretch"})
    r_nw = _analyze_cfg(no_wrap)
    r_st = _analyze_cfg(stretched)
    print(f"    wrap NONE    : speed={r_nw['max_speed_mps']:.2f} accel={r_nw['max_accel_mps2']:.2f} mode={r_nw['dominant_failure_mode']}")
    print(f"    wrap STRETCH : speed={r_st['max_speed_mps']:.2f} accel={r_st['max_accel_mps2']:.2f} mode={r_st['dominant_failure_mode']}")
    r.check("stretch wrap doesn't reduce max_speed vs no wrap",
            r_st["max_speed_mps"] >= r_nw["max_speed_mps"] - 0.01)
    r.check("stretch wrap doesn't reduce max_accel vs no wrap",
            r_st["max_accel_mps2"] >= r_nw["max_accel_mps2"] - 0.01)

    # -- 2. Temperature sweep on stretch-wrapped heavy pallet --
    heavy_stretch = build_from_stacks(
        [StackSpec(sku="SKU-MS-003", grid_row=r_, grid_col=c_, height=4)
         for r_ in range(2) for c_ in range(2)],
        pallet_id="phys_T", env=EnvCondition.FROZEN, body_temp_c=-25.0,
        wrap=WrapType.STRETCH,
    )
    warm = heavy_stretch.model_copy(update={
        "pallet_id": "phys_T_warm", "env": "thawed", "body_temp_c": 20.0,
        "seconds_since_temp_change": 7200.0,
    })
    r_cold = _analyze_cfg(heavy_stretch)
    r_warm = _analyze_cfg(warm)
    print(f"    heavy FROZEN  : accel={r_cold['max_accel_mps2']:.2f}")
    print(f"    heavy THAWED  : accel={r_warm['max_accel_mps2']:.2f}")
    r.check("cold heavy pallet max_accel ≤ warm (monotone in μ)",
            r_cold["max_accel_mps2"] <= r_warm["max_accel_mps2"] + 0.21,
            f"cold={r_cold['max_accel_mps2']:.2f} warm={r_warm['max_accel_mps2']:.2f}")

    # -- 3. Height comparison: 2 vs 6 stacked --
    short_stack = build_from_stacks(
        [StackSpec(sku="SKU-FD-002", grid_row=0, grid_col=0, height=2)],
        pallet_id="phys_short",
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0, wrap=WrapType.NONE,
    )
    tall_stack = short_stack.model_copy(update={
        "pallet_id": "phys_tall",
        "items": [  # rebuild items list for height=6
            *[{"sku": "SKU-FD-002", "weight_kg": 6.0, "dims_m": [0.35, 0.25, 0.15],
               "fragility": "rigid", "position": [0.0, 0.0, 0.15 + i * 0.15],
               "orientation_deg": 0.0} for i in range(6)],
        ],
    })
    # Actually, use build_from_stacks for the tall one to be safe
    tall_stack = build_from_stacks(
        [StackSpec(sku="SKU-FD-002", grid_row=0, grid_col=0, height=6)],
        pallet_id="phys_tall",
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0, wrap=WrapType.NONE,
    )
    r_short = _analyze_cfg(short_stack)
    r_tall = _analyze_cfg(tall_stack)
    print(f"    short stack : speed={r_short['max_speed_mps']:.2f} accel={r_short['max_accel_mps2']:.2f}")
    print(f"    tall stack  : speed={r_tall['max_speed_mps']:.2f} accel={r_tall['max_accel_mps2']:.2f}")
    r.check("tall stack max_accel ≤ short stack (height reduces stability)",
            r_tall["max_accel_mps2"] <= r_short["max_accel_mps2"] + 0.21,
            f"short={r_short['max_accel_mps2']:.2f} tall={r_tall['max_accel_mps2']:.2f}")

    # -- 4. Empty pallet — should be totally safe --
    empty = build_from_stacks(
        [], pallet_id="phys_empty", env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    r_empty = _analyze_cfg(empty)
    r.check("empty pallet hits upper speed bound",
            r_empty["max_speed_mps"] >= 1.8,
            f"got {r_empty['max_speed_mps']:.2f}")

    return r


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def shot(page, name: str) -> None:
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)


def main() -> int:
    SHOTS.mkdir(exist_ok=True)

    all_reports: list[Report] = []

    # UI-driven persona
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 2200})
        page = ctx.new_page()
        page.on("pageerror", lambda e: print(f"  [JS ERROR] {e}"))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)
        all_reports.append(persona_loadmaster(page))
        browser.close()

    # API-driven personas
    all_reports.append(persona_batch_analyst())
    all_reports.append(persona_operations())
    all_reports.append(persona_physics_validator())

    print("\n" + "=" * 60)
    print("PERSONA UAT SUMMARY")
    print("=" * 60)
    total_pass = 0
    total_fail = 0
    for rep in all_reports:
        p, f = rep.summary()
        total_pass += p
        total_fail += f
        status = "✓" if f == 0 else "✗"
        print(f"  {status} {rep.persona:<20s}: {p} pass, {f} fail")
    print(f"\n  TOTAL: {total_pass} pass, {total_fail} fail")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
