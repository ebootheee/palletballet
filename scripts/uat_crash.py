"""Visual UAT for the crash-test presets — load a preset and capture the animation
mid-failure to verify items are actually visualized as flying/rotating."""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "uat_screenshots_crash"
URL = "http://localhost:8501"


def main() -> int:
    SHOTS.mkdir(exist_ok=True)
    print(f"Crash UAT against {URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1700, "height": 2400})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # Manual configurator
        page.locator('label:has-text("Manual configurator")').first.click()
        page.locator('h1:has-text("Manual configurator")').first.wait_for(timeout=15000)
        page.locator('button:has-text("fill all cells")').first.wait_for(timeout=15000)
        page.wait_for_timeout(2000)

        # Find the Crash test preset selectbox
        # In Streamlit listboxes, select by clicking the selectbox then the option.
        # Easier: use the locator chain via aria attributes.
        print("\nSelecting 'Tall unwrapped tower (TIP)' preset...")
        # The selectbox's combobox role contains the current value
        cb = page.locator('div[data-baseweb="select"]').filter(
            has_text="(none)").first
        cb.click()
        page.wait_for_timeout(800)
        page.locator('li:has-text("Tall unwrapped tower")').first.click()
        page.wait_for_timeout(800)

        # Click "load crash test"
        page.locator('button:has-text("load crash test")').first.click()
        page.wait_for_timeout(2000)
        path = SHOTS / "01_preset_loaded.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        # Open Run Sim expander
        print("\nOpening Run Sim and clicking Run...")
        page.locator('summary:has-text("Run simulation")').first.click()
        page.wait_for_timeout(1000)
        page.locator('button:has-text("Run")').first.click()

        # Wait for animation to render (look for Play button)
        try:
            page.locator('text="▶ Play"').first.wait_for(timeout=30000)
        except Exception:
            page.locator('text=/Play/').first.wait_for(timeout=10000)
        page.wait_for_timeout(2500)

        path = SHOTS / "02_after_run.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        # Set loop to 10 cycles for endless replay
        try:
            loop_cb = page.locator('div[data-baseweb="select"]').filter(has_text="3").first
            loop_cb.click()
            page.wait_for_timeout(500)
            page.locator('li:has-text("10")').first.click()
            page.wait_for_timeout(500)
        except Exception as e:
            print(f"  loop selection failed: {e}")

        # Click play
        try:
            page.locator('text="▶ Play"').first.click()
            print("  Animation playing... capturing mid-flight frames")
            for i, dt_ms in enumerate([400, 800, 1500, 2500]):
                page.wait_for_timeout(dt_ms)
                path = SHOTS / f"03_anim_t{i}.png"
                page.screenshot(path=str(path), full_page=True)
                print(f"  [shot] {path.name}")
        except Exception as e:
            print(f"  could not play: {e}")

        # Try the asymmetric load preset too
        print("\nTrying 'Asymmetric load (TIP)'...")
        cb = page.locator('div[data-baseweb="select"]').filter(
            has_text="Tall unwrapped tower").first
        cb.click()
        page.wait_for_timeout(500)
        page.locator('li:has-text("Asymmetric")').first.click()
        page.wait_for_timeout(500)
        page.locator('button:has-text("load crash test")').first.click()
        page.wait_for_timeout(2500)

        page.locator('button:has-text("Run")').first.click()
        try:
            page.locator('text="▶ Play"').first.wait_for(timeout=30000)
        except Exception:
            pass
        page.wait_for_timeout(2500)
        path = SHOTS / "04_asymmetric_after_run.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        page.locator('text="▶ Play"').first.click()
        page.wait_for_timeout(2500)
        path = SHOTS / "05_asymmetric_playing.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"  [shot] {path.name}")

        browser.close()

    print(f"\nerrors: {len(errors)}")
    for e in errors[:10]:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
