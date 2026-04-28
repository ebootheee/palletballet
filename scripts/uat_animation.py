"""UAT specifically for the new animation widget."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_anim"
URL = "http://localhost:8501"


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    print(f"Animation UAT against {URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 2000})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Generate page
        print("\nGenerate page → Run Sim → Animation")
        page.locator('label:has-text("Generate random")').first.click()
        page.locator('h1:has-text("Generate random pallet")').first.wait_for(timeout=15000)
        page.locator('h3:has-text("3D view")').first.wait_for(timeout=15000)
        page.wait_for_timeout(2500)

        # Open Run Sim expander
        page.locator('summary:has-text("Run simulation")').first.click()
        page.wait_for_timeout(1000)

        # Click Run
        page.locator('button:has-text("Run")').first.click()

        # Wait for sim + chart + animation to render
        # Look for the animation's "Play" button as the signal
        try:
            page.locator('text="▶ Play"').first.wait_for(timeout=30000)
        except Exception:
            page.locator('text=/Play/').first.wait_for(timeout=10000)
        page.wait_for_timeout(2500)
        path = SHOTS / "anim_after_run.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        # Click Play button to start animation
        try:
            play_btn = page.locator('text="▶ Play"').first
            play_btn.click()
            page.wait_for_timeout(2500)  # let animation play a bit
            path = SHOTS / "anim_playing.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"  [shot] {path.name}")
        except Exception as e:
            print(f"  could not click Play: {e}")

        # Toggle follow-pallet
        try:
            page.locator('label:has-text("follow pallet")').first.click()
            page.wait_for_timeout(2500)
            path = SHOTS / "anim_follow_pallet.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"  [shot] {path.name}")
        except Exception as e:
            print(f"  could not toggle follow_pallet: {e}")

        browser.close()

    print(f"\nerrors: {len(errors)}")
    for e in errors[:10]:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
