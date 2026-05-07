"""Cycle 3 UAT: capture the Safety Analysis page for both STABLE and UNSTABLE
pallets so we can see the improved sweep chart and verify discrimination.
"""

from __future__ import annotations

import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_cycle3"
URL = "http://localhost:8501"


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 2400})
        page = ctx.new_page()
        page.on("pageerror", lambda e: print(f"JS ERR: {e}"))
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # --- Scenario A: stable pallet via Manual page with fill all cells ---
        print("[A] Stable homogeneous pallet — expect green envelope")
        page.locator('label:has-text("Manual configurator")').first.click()
        page.locator('h1:has-text("Manual configurator")').first.wait_for(timeout=15000)
        page.locator('button:has-text("fill all cells")').first.wait_for(timeout=15000)
        page.wait_for_timeout(2000)
        page.locator('button:has-text("clear")').first.click()
        page.wait_for_timeout(500)
        page.locator('button:has-text("fill all cells")').first.click()
        page.wait_for_timeout(2500)

        page.locator('label:has-text("Safety Analysis")').first.click()
        page.locator('h1:has-text("Safety Analysis")').first.wait_for(timeout=15000)
        page.wait_for_timeout(2000)
        page.locator('label:has-text("From Manual page")').first.click()
        page.wait_for_timeout(1500)
        page.locator('text=/Max speed/').first.wait_for(timeout=30000)
        page.wait_for_timeout(2500)
        (SHOTS / "A_stable.png").unlink(missing_ok=True)
        page.screenshot(path=str(SHOTS / "A_stable.png"), full_page=True)
        print("  [shot] A_stable.png")

        # --- Scenario B: crash preset via Manual page → analyze → see the failure ---
        print("\n[B] Crash-test preset — expect red X + narrower green envelope")
        page.locator('label:has-text("Manual configurator")').first.click()
        page.locator('h1:has-text("Manual configurator")').first.wait_for(timeout=15000)
        page.wait_for_timeout(1500)
        # Click the crash-preset selectbox (contains "(none)")
        cb = page.locator('div[data-baseweb="select"]').filter(has_text="(none)").first
        cb.click()
        page.wait_for_timeout(500)
        page.locator('li:has-text("Tall unwrapped tower")').first.click()
        page.wait_for_timeout(500)
        page.locator('button:has-text("load crash test")').first.click()
        page.wait_for_timeout(2500)

        page.locator('label:has-text("Safety Analysis")').first.click()
        page.locator('h1:has-text("Safety Analysis")').first.wait_for(timeout=15000)
        page.wait_for_timeout(1500)
        page.locator('label:has-text("From Manual page")').first.click()
        page.wait_for_timeout(1500)
        # Analyze now button (auto-analyze should fire but click to be explicit)
        try:
            page.locator('button:has-text("Analyze now")').first.click()
        except Exception:
            pass
        page.locator('text=/Max speed/').first.wait_for(timeout=60000)
        page.wait_for_timeout(3000)
        (SHOTS / "B_crash.png").unlink(missing_ok=True)
        page.screenshot(path=str(SHOTS / "B_crash.png"), full_page=True)
        print("  [shot] B_crash.png")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
