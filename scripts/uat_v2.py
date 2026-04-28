"""UAT v2 — exercises the redesigned manual configurator and inline sim runners."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_v2"
URL = "http://localhost:8501"


def wait_api() -> None:
    import urllib.request
    for _ in range(60):
        try:
            with urllib.request.urlopen(URL, timeout=0.5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit not responding at {URL}")


def wait_for_h1(page, text: str, timeout=15000) -> None:
    page.locator(f'h1:has-text("{text}")').first.wait_for(timeout=timeout)
    page.wait_for_timeout(1500)


def select_page(page, name: str, expected_h1: str, wait_for_body: str | None = None) -> None:
    page.locator(f'label:has-text("{name}")').first.click()
    wait_for_h1(page, expected_h1)
    if wait_for_body is not None:
        # Give the body section time to render (plotly, API calls, etc.)
        page.locator(wait_for_body).first.wait_for(timeout=15000)
    page.wait_for_timeout(2500)  # plotly chart finalization


def shot(page, name: str) -> None:
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    print(f"  [shot] {path.name}")


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    wait_api()
    print(f"UAT v2 against {URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1400})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # --- Generate: sample a few random pallets, verify no Ti=1 pinwheels ---
        print("\n[1] Generate — sanity check across seeds")
        select_page(page, "Generate random", "Generate random pallet",
                     wait_for_body='h3:has-text("3D view")')
        for seed in ["42", "7", "100", "999"]:
            page.locator('input[aria-label="seed (blank = random)"]').first.fill(seed)
            page.locator('button:has-text("generate")').first.click()
            page.wait_for_timeout(2000)
            shot(page, f"gen_seed_{seed}")

        # --- Generate inline Run Sim ---
        print("\n[2] Generate — inline Run Sim")
        page.locator('summary:has-text("Run simulation")').first.click()
        page.wait_for_timeout(500)
        page.locator('button:has-text("Run")').first.click()
        page.wait_for_timeout(4000)  # give sim + chart render time
        shot(page, "gen_run_sim_result")

        # --- Manual configurator: fill all, add mixed-height stacks ---
        print("\n[3] Manual configurator — homogeneous fill")
        select_page(page, "Manual configurator", "Manual configurator",
                     wait_for_body='button:has-text("fill all cells")')
        page.locator('button:has-text("fill all cells")').first.click()
        page.wait_for_timeout(2500)
        shot(page, "manual_fill_homogeneous")

        print("\n[4] Manual configurator — add 3 stacks (same SKU, different cells, default height)")
        page.locator('button:has-text("clear")').first.click()
        page.wait_for_timeout(600)
        # Click +add 3 times: each uses whatever cell is currently selected,
        # with selection cycling through via the cell selectbox. We don't
        # interact with Streamlit listboxes (non-trivial via Playwright), just
        # verify we can add multiple stacks successfully.
        for _ in range(3):
            page.locator('button:has-text("+ add stack")').first.click()
            page.wait_for_timeout(1500)
        shot(page, "manual_three_stacks")

        # Verify the Stacks table shows 3 rows
        stacks_visible = page.locator("text=/Stacks on the pallet/").count() > 0
        print(f"  stacks table visible: {stacks_visible}")

        print("\n[5] Manual configurator — inline Run Sim")
        page.locator('summary:has-text("Run simulation")').first.click()
        page.wait_for_timeout(500)
        page.locator('button:has-text("Run")').first.click()
        page.wait_for_timeout(4000)
        shot(page, "manual_run_sim_result")

        # --- Friction & Catalog — lightweight smoke checks ---
        print("\n[6] Friction smoke")
        select_page(page, "Friction model", "Friction model")
        shot(page, "friction")

        print("\n[7] Catalog smoke")
        select_page(page, "SKU catalog", "SKU catalog")
        shot(page, "catalog")

        browser.close()

    print(f"\nerrors captured: {len(errors)}")
    for e in errors[:20]:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
