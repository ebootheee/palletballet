"""Batch safety study: N random pallets → safe-envelope distribution.

This is the "batch" workflow from the launch post, runnable against the
library directly (no HTTP). Each worker process owns a ThresholdAnalyzer so
MuJoCo models compile once per pallet and the OS scheduler does the fanout.

Usage:
    uv run python scripts/batch_study.py --n 300 --out /tmp/batch_study.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from pallet_safety.configurator import Configurator
from pallet_safety.inputs import MockRandomAdapter
from pallet_safety.threshold import ThresholdAnalyzer


def analyze_seed(seed: int) -> dict:
    adapter = MockRandomAdapter(seed=seed, anomaly_rate=0.15)
    cfg = Configurator().build(adapter.read())
    analyzer = ThresholdAnalyzer()
    t0 = time.perf_counter()
    analysis = analyzer.analyze(cfg)
    wall_s = time.perf_counter() - t0
    r = analysis.result
    com = cfg.composite_com_m
    return {
        "seed": seed,
        "items": len(cfg.items),
        "total_mass_kg": round(cfg.total_mass_kg, 1),
        "stack_height_m": round(cfg.stack_height_m, 3),
        "com_height_m": round(com[2], 3),
        "com_offset_m": round((com[0] ** 2 + com[1] ** 2) ** 0.5, 3),
        "overhang_m": round(cfg.overhang_m, 3),
        "wrap": cfg.wrap.value,
        "env": cfg.env.value,
        "body_temp_c": cfg.body_temp_c,
        "max_speed_mps": r.max_speed_mps,
        "max_accel_mps2": r.max_accel_mps2,
        "dominant_failure_mode": r.dominant_failure_mode.value,
        "confidence": r.confidence,
        "sims_run": analysis.sims_run,
        "wall_s": round(wall_s, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="batch_study.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(analyze_seed, range(args.n), chunksize=4))
    total_s = time.perf_counter() - t0

    total_sims = sum(r["sims_run"] for r in rows)
    summary = {
        "n_pallets": len(rows),
        "total_sims": total_sims,
        "wall_s": round(total_s, 1),
        "pallets_per_s": round(len(rows) / total_s, 2),
        "workers": args.workers,
    }
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
