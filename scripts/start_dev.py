"""One-shot dev launcher: starts the FastAPI service and the Streamlit UI together.

Usage:
    python -m uv run python scripts/start_dev.py

Visit:
    http://localhost:8501  ← Streamlit UI
    http://localhost:8000/docs ← FastAPI Swagger
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

API_CMD = [sys.executable, "-m", "uvicorn", "pallet_safety.service.api:app",
           "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
UI_CMD = [sys.executable, "-m", "streamlit", "run",
          str(ROOT / "pallet_safety" / "viz" / "streamlit_app.py"),
          "--server.port", "8501",
          "--server.headless", "true",
          "--browser.gatherUsageStats", "false"]


def main() -> int:
    print("Starting FastAPI on http://localhost:8000 ...")
    api = subprocess.Popen(API_CMD, cwd=str(ROOT))
    time.sleep(1.5)
    print("Starting Streamlit on http://localhost:8501 ...")
    ui = subprocess.Popen(UI_CMD, cwd=str(ROOT))

    def shutdown(_sig=None, _frame=None):
        print("\nShutting down...")
        for p in [ui, api]:
            try:
                p.terminate()
            except Exception:
                pass
        for p in [ui, api]:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        ui.wait()
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
