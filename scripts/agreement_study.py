"""PLAN_GPU Phase 1: fp64-CPU vs fp32-GPU verdict agreement study.

Runs every corpus pallet (all registry scenarios + N random pallets) across a
speed x accel profile grid on both backends, feeds both traces through the
same `failures.py` detectors, and reports:

  - verdict + failure-mode agreement, overall and away from decision
    boundaries (cells whose closest metric is within BOUNDARY_BAND of its
    threshold are classed "boundary" — legitimate flip territory)
  - per-accel max-safe-speed deltas between backends, in grid steps

CPU arm = production configuration (implicitfast, fp64), fanned out over a
process pool. GPU arm = mujoco_warp (euler, fp32), one batch per pallet.

Usage:
    uv run --extra gpu python scripts/agreement_study.py --n 300 \
        --out data/agreement_study.parquet
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pallet_safety.configurator import Configurator
from pallet_safety.failures import FailureThresholds, first_failure
from pallet_safety.inputs import MockRandomAdapter
from pallet_safety.models import PalletConfig
from pallet_safety.scenarios import all_scenarios
from pallet_safety.solver import ConveyorProfile, SimulationTrace, build_model, simulate

SPEEDS = np.linspace(0.25, 2.5, 12)
ACCELS = np.linspace(0.5, 8.0, 6)
DURATION_S = 2.5
SETTLE_S = 0.5
THRESH = FailureThresholds()
# A cell is "boundary" when its most critical metric lands within this band
# around the detector threshold (ratio units): 3/4x .. 4/3x.
BOUNDARY_BAND = (0.75, 4.0 / 3.0)


def profiles() -> list[ConveyorProfile]:
    return [
        ConveyorProfile(target_speed_mps=float(s), accel_mps2=float(a), duration_s=DURATION_S)
        for s in SPEEDS for a in ACCELS
    ]


def corpus(n_random: int) -> list[tuple[str, PalletConfig]]:
    out = [(s.slug, s.pallet) for s in all_scenarios()]
    configurator = Configurator()
    for seed in range(n_random):
        raw = MockRandomAdapter(seed=seed, anomaly_rate=0.15).read()
        out.append((f"random-{seed:04d}", configurator.build(raw)))
    return out


def trace_metrics(trace: SimulationTrace) -> dict:
    """Verdict plus the raw per-detector maxima (for boundary classification)."""
    mode, t_fail = first_failure(trace, THRESH)

    q = trace.pallet_quat
    x, y = q[:, 1], q[:, 2]
    max_tip = float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))).max())

    if trace.n_items:
        deltas = np.linalg.norm(
            trace.item_pallet_pos - trace.item_initial_pallet_pos[None], axis=2)
        max_slide = float(deltas.max())
    else:
        max_slide = 0.0

    dt = np.diff(trace.times, prepend=trace.times[0])
    belt = np.cumsum(trace.conveyor_vel * dt)
    max_slip = float(np.abs(belt - (trace.pallet_pos[:, 0] - trace.pallet_pos[0, 0])).max())

    if trace.n_items >= 2:
        init = trace.item_initial_pallet_pos
        init_d = np.linalg.norm(init[:, None] - init[None, :], axis=2)
        cur = trace.item_pallet_pos
        d = np.linalg.norm(cur[:, :, None, :] - cur[:, None, :, :], axis=3)
        max_shift = float(np.abs(d - init_d[None]).max())
    else:
        max_shift = 0.0

    # How close this run came to the nearest failure threshold (1.0 = exactly at it)
    criticality = max(
        max_tip / THRESH.tip_angle_deg,
        max_slide / THRESH.item_slide_m,
        max_slip / THRESH.pallet_slip_m,
        max_shift / THRESH.load_shift_m,
    )
    return {
        "mode": mode.value,
        "fail_time_s": t_fail if t_fail is not None else np.nan,
        "unsafe": mode.value != "no_failure",
        "criticality": criticality,
        "boundary": BOUNDARY_BAND[0] < criticality < BOUNDARY_BAND[1],
    }


def cpu_pallet(args: tuple[str, PalletConfig]) -> list[dict]:
    """One worker task: all grid cells for one pallet, production CPU config."""
    name, config = args
    model = build_model(config)
    rows = []
    for p in profiles():
        trace = simulate(config, p, model=model, settle_s=SETTLE_S).downsample(33.4)
        rows.append({
            "pallet": name, "speed": p.target_speed_mps, "accel": p.accel_mps2,
            **{f"cpu_{k}": v for k, v in trace_metrics(trace).items()},
        })
    return rows


def max_safe_speed_by_accel(sub: pd.DataFrame, col: str) -> dict[float, float]:
    out = {}
    for a in ACCELS:
        cells = sub[np.isclose(sub["accel"], a)]
        safe = cells[~cells[col]]["speed"]
        out[float(a)] = float(safe.max()) if len(safe) else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="random pallets in addition to scenarios")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="data/agreement_study.parquet")
    args = ap.parse_args()

    pallets = corpus(args.n)
    n_cells = len(SPEEDS) * len(ACCELS)
    print(f"corpus: {len(pallets)} pallets x {n_cells} cells "
          f"= {len(pallets) * n_cells} sims per backend")

    t0 = time.perf_counter()
    cpu_rows: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(cpu_pallet, p): p[0] for p in pallets}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            cpu_rows[rows[0]["pallet"]] = rows
            if i % 50 == 0:
                print(f"  cpu: {i}/{len(pallets)} pallets")
    print(f"cpu arm: {time.perf_counter() - t0:.0f} s")

    from pallet_safety.gpu.grid import run_profile_grid

    t0 = time.perf_counter()
    all_rows: list[dict] = []
    prof = profiles()
    for i, (name, config) in enumerate(pallets, 1):
        traces = run_profile_grid(config, prof, settle_s=SETTLE_S)
        for row, trace in zip(cpu_rows[name], traces, strict=True):
            all_rows.append({
                **row,
                **{f"gpu_{k}": v for k, v in trace_metrics(trace).items()},
            })
        if i % 25 == 0:
            print(f"  gpu: {i}/{len(pallets)} pallets ({time.perf_counter() - t0:.0f} s)")
    print(f"gpu arm: {time.perf_counter() - t0:.0f} s")

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.out)
    print(f"wrote {len(df)} rows -> {args.out}\n")

    # ---- summary ----
    verdict_match = df["cpu_unsafe"] == df["gpu_unsafe"]
    mode_match = df["cpu_mode"] == df["gpu_mode"]
    interior = ~(df["cpu_boundary"] | df["gpu_boundary"])
    print(f"verdict agreement, all cells:      {verdict_match.mean():.4f}")
    print(f"verdict agreement, interior cells: {verdict_match[interior].mean():.4f} "
          f"({interior.sum()}/{len(df)} interior)")
    print(f"mode agreement on shared-unsafe:   "
          f"{mode_match[df.cpu_unsafe & df.gpu_unsafe].mean():.4f}")

    speed_step = float(SPEEDS[1] - SPEEDS[0])
    deltas = []
    for name, sub in df.groupby("pallet"):
        c = max_safe_speed_by_accel(sub, "cpu_unsafe")
        g = max_safe_speed_by_accel(sub, "gpu_unsafe")
        deltas.extend(abs(c[a] - g[a]) / speed_step for a in c)
    deltas = np.array(deltas)
    print(f"\nmax-safe-speed delta (grid steps of {speed_step:.2f} m/s), "
          f"n={len(deltas)} pallet-accel rows:")
    for lim in (0.0, 1.0, 2.0):
        print(f"  |delta| <= {lim:.0f}: {(deltas <= lim).mean():.4f}")
    print(f"  worst: {deltas.max():.1f} steps")

    disagree = df[~verdict_match & interior]
    if len(disagree):
        print(f"\ninterior disagreements ({len(disagree)}):")
        print(disagree[["pallet", "speed", "accel", "cpu_mode", "gpu_mode",
                        "cpu_criticality", "gpu_criticality"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
