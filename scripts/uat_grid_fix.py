"""Verify the auto-grid fix for the SKU-DG-001 fill-all explosion."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_grid_fix"
URL = "http://localhost:8501"


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 2200})
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Manual configurator
        page.locator('label:has-text("Manual configurator")').first.click()
        page.locator('h1:has-text("Manual configurator")').first.wait_for(timeout=15000)
        page.locator('button:has-text("fill all cells")').first.wait_for(timeout=15000)
        page.wait_for_timeout(2500)

        # Pick SKU-DG-001 (Flour Sacks)
        # Streamlit selectbox: click then pick option containing the text
        sku_box = page.locator('div[data-baseweb="select"]').filter(
            has_text="SKU-").first  # first SKU listbox in sidebar
        sku_box.click()
        page.wait_for_timeout(800)
        page.locator('li:has-text("Flour Sacks")').first.click()
        page.wait_for_timeout(800)

        path = SHOTS / "01_sku_chosen.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        # Fill all
        page.locator('button:has-text("fill all cells")').first.click()
        page.wait_for_timeout(2500)
        path = SHOTS / "02_after_fill_all.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        # Open Run Sim and click Run
        page.locator('summary:has-text("Run simulation")').first.click()
        page.wait_for_timeout(1000)
        page.locator('button:has-text("Run")').first.click()
        try:
            page.locator('text="▶ Play"').first.wait_for(timeout=30000)
        except Exception:
            pass
        page.wait_for_timeout(3000)
        path = SHOTS / "03_after_run_sim.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
