"""PLAN_GPU Phase 0 spike: does our pallet model survive on MuJoCo Warp?

Runs the tall-unwrapped-tower scenario at 8 conveyor speeds three ways:

  1. CPU / implicitfast / fp64  — the production baseline
  2. CPU / euler / fp64         — integrator control (mjwarp lacks implicitfast)
  3. GPU / euler / fp32         — mujoco_warp, all 8 speeds as one batch

and compares failure verdicts from the same `failures.py` detectors. Go/no-go:
the weld wrap constraints, freejoints, and velocity-actuated conveyor must
step without NaNs, and GPU verdicts should broadly match CPU-euler.

Usage: uv run --extra gpu python scripts/gpu_spike.py
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from pallet_safety.failures import FailureThresholds, first_failure
from pallet_safety.scenarios import get_scenario
from pallet_safety.solver import ConveyorProfile, SimulationTrace, build_model, simulate

SLUG = "tall-unwrapped-tower"
SPEEDS = np.linspace(0.25, 2.0, 8)
SETTLE_S = 0.5
RECORD_EVERY = 15  # steps between recorded frames (~33 Hz at 2 ms timestep)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz quaternions -> (..., 3, 3) rotation matrices."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def cpu_arm(config, profiles, integrator: int, label: str):
    model = build_model(config)
    model.opt.integrator = integrator
    verdicts, total = [], 0.0
    for p in profiles:
        t0 = time.perf_counter()
        trace = simulate(config, p, model=model, settle_s=SETTLE_S)
        total += time.perf_counter() - t0
        verdicts.append(first_failure(trace, FailureThresholds()))
    print(f"  {label}: {total * 1000 / len(profiles):.0f} ms/sim")
    return verdicts


def gpu_arm(config, profiles):
    import mujoco_warp as mjw
    import warp as wp

    wp.init()
    nworld = len(profiles)
    mjm = build_model(config)
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    # Generous constraint/contact headroom: defaults overflowed at njmax=144
    # for this 8-item scene (dropped constraints -> NaNs). Scenes are small,
    # so 4x headroom costs little memory.
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=256, njmax=1024)
    print(f"  gpu: put_model/put_data {time.perf_counter() - t0:.1f} s")

    pallet_id = int(mjm.body("pallet_base").id)
    item_ids = np.array([int(mjm.body(f"item_{i}").id) for i in range(len(config.items))])
    conveyor_qv0 = int(mjm.joint("conveyor_slide").dofadr[0])

    timestep = float(mjm.opt.timestep)
    n_settle = int(SETTLE_S / timestep)
    n_run = int(profiles[0].duration_s / timestep)
    total_steps = n_settle + n_run

    # Per-world control lookup table, built on host once: (total_steps, nworld)
    ctrl_table = np.zeros((total_steps, nworld), dtype=np.float32)
    for w_i, p in enumerate(profiles):
        for s_i in range(n_settle, total_steps):
            ctrl_table[s_i, w_i] = p.velocity_at((s_i - n_settle) * timestep)

    rec_steps = list(range(0, total_steps, RECORD_EVERY))
    n_rec = len(rec_steps)
    r_times = np.zeros(n_rec)
    r_conveyor = np.zeros((n_rec, nworld))
    r_ppos = np.zeros((n_rec, nworld, 3))
    r_pquat = np.zeros((n_rec, nworld, 4))
    r_ipos = np.zeros((n_rec, nworld, len(item_ids), 3))
    r_iquat = np.zeros((n_rec, nworld, len(item_ids), 4))

    ctrl_host = np.zeros((nworld, mjm.nu), dtype=np.float32)

    t0 = time.perf_counter()
    rec_i = 0
    for step in range(total_steps):
        ctrl_host[:, 0] = ctrl_table[step]
        wp.copy(d.ctrl, wp.array(ctrl_host, dtype=wp.float32, device=d.ctrl.device))
        mjw.step(m, d)
        if rec_i < n_rec and step == rec_steps[rec_i]:
            xpos = d.xpos.numpy()
            xquat = d.xquat.numpy()
            qvel = d.qvel.numpy()
            r_times[rec_i] = max((step - n_settle) * timestep, 0.0)
            r_conveyor[rec_i] = qvel[:, conveyor_qv0]
            r_ppos[rec_i] = xpos[:, pallet_id]
            r_pquat[rec_i] = xquat[:, pallet_id]
            r_ipos[rec_i] = xpos[:, item_ids]
            r_iquat[rec_i] = xquat[:, item_ids]
            rec_i += 1
    wp.synchronize()
    wall = time.perf_counter() - t0
    print(f"  gpu: {total_steps} steps x {nworld} worlds in {wall:.2f} s "
          f"({total_steps * nworld / wall:,.0f} steps/s incl. recording)")

    if np.isnan(r_ppos).any() or np.isnan(r_ipos).any():
        print("  gpu: *** NaNs in trace — UNSTABLE, no-go as configured ***")
        return None

    # Item positions in pallet frame: R^T (item - pallet), vectorized over frames+worlds
    R = quat_to_mat(r_pquat)  # (n_rec, nworld, 3, 3)
    rel = r_ipos - r_ppos[:, :, None, :]
    ipal = np.einsum("fwji,fwkj->fwki", R, rel)  # R^T applied to each item vector

    settle_rec = next(i for i, s in enumerate(rec_steps) if s >= n_settle)
    verdicts = []
    for w_i, p in enumerate(profiles):
        sl = slice(settle_rec, n_rec)
        trace = SimulationTrace(
            times=r_times[sl], conveyor_vel=r_conveyor[sl, w_i],
            pallet_pos=r_ppos[sl, w_i], pallet_quat=r_pquat[sl, w_i],
            pallet_lin_vel=np.zeros((n_rec - settle_rec, 3)),
            pallet_ang_vel=np.zeros((n_rec - settle_rec, 3)),
            item_world_pos=r_ipos[sl, w_i], item_world_quat=r_iquat[sl, w_i],
            item_pallet_pos=ipal[sl, w_i],
            item_initial_pallet_pos=ipal[settle_rec, w_i].copy(),
            config=config, profile=p,
            n_items=len(item_ids), n_steps=n_rec - settle_rec,
        )
        verdicts.append(first_failure(trace, FailureThresholds()))
    return verdicts


def grid_test(config, base, n_speeds=32, n_accels=16, verbose=True):
    """Throughput: full speed x accel grid in one batch.

    Control table and trace recording live on device; the whole rollout is one
    CUDA graph and one host sync. This is the Phase 2 GridAnalyzer workload,
    minus polish.
    """
    import mujoco_warp as mjw
    import warp as wp

    speeds = np.linspace(0.1, 2.5, n_speeds)
    accels = np.linspace(0.25, 8.0, n_accels)
    nworld = len(speeds) * len(accels)
    grid_speed = np.repeat(speeds, len(accels))   # (nworld,)
    grid_accel = np.tile(accels, len(speeds))     # (nworld,)

    mjm = build_model(config)
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=256, njmax=1024)

    pallet_id = int(mjm.body("pallet_base").id)
    item_ids = np.array([int(mjm.body(f"item_{i}").id) for i in range(len(config.items))])

    timestep = float(mjm.opt.timestep)
    n_settle = int(SETTLE_S / timestep)
    n_run = int(base.duration_s / timestep)
    total_steps = n_settle + n_run

    # Vectorized accel_hold control table -> device, (total_steps, nworld, nu)
    t = (np.arange(total_steps) - n_settle) * timestep  # negative during settle
    v = np.minimum(np.maximum(t[:, None], 0.0) * grid_accel[None, :],
                   grid_speed[None, :]).astype(np.float32)
    ctrl_dev = wp.array(v[:, :, None], dtype=wp.float32, device=d.ctrl.device)

    rec_steps = list(range(0, total_steps, RECORD_EVERY))
    n_rec = len(rec_steps)
    rec_set = set(rec_steps)

    # Device-resident trace buffers: one wp.copy per recorded frame, one host
    # sync for the entire rollout. ~10 MB total for this scene.
    rec_xpos = wp.empty((n_rec,) + d.xpos.shape, dtype=d.xpos.dtype, device=d.xpos.device)
    rec_xquat = wp.empty((n_rec,) + d.xquat.shape, dtype=d.xquat.dtype, device=d.xquat.device)
    rec_qvel = wp.empty((n_rec,) + d.qvel.shape, dtype=d.qvel.dtype, device=d.qvel.device)

    def rollout():
        rec_i = 0
        for step in range(total_steps):
            wp.copy(d.ctrl, ctrl_dev[step])
            mjw.step(m, d)
            if step in rec_set:
                wp.copy(rec_xpos[rec_i], d.xpos)
                wp.copy(rec_xquat[rec_i], d.xquat)
                wp.copy(rec_qvel[rec_i], d.qvel)
                rec_i += 1

    # Capture the whole rollout as one CUDA graph (pointers are baked in, and
    # every op above is device-side, so the loop is capturable end to end).
    t0 = time.perf_counter()
    with wp.ScopedCapture() as capture:
        rollout()
    capture_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    wp.capture_launch(capture.graph)
    wp.synchronize()
    wall = time.perf_counter() - t0

    xpos_h = rec_xpos.numpy()    # (n_rec, nworld, nbody, 3)
    xquat_h = rec_xquat.numpy()
    r_ppos = xpos_h[:, :, pallet_id].astype(np.float64)
    r_pquat = xquat_h[:, :, pallet_id].astype(np.float64)
    r_ipos = xpos_h[:, :, item_ids].astype(np.float64)

    print(f"\ngrid: {nworld} worlds x {total_steps} steps: "
          f"graph capture {capture_s:.2f} s (one-time per pallet), "
          f"rollout {wall:.2f} s = {nworld * total_steps / wall:,.0f} steps/s "
          f"({nworld / wall:,.0f} sims/s equivalent)")
    cpu_equiv = nworld * 0.055
    print(f"grid: same work on CPU ~{cpu_equiv:.0f} s single-core "
          f"(~{cpu_equiv / 16:.1f} s on 16 cores) -> "
          f"{cpu_equiv / wall:.0f}x single-core, {cpu_equiv / 16 / wall:.1f}x 16-core")

    if np.isnan(r_ppos).any() or np.isnan(r_ipos).any():
        print("grid: *** NaNs present ***")
        return
    if not verbose:
        return

    # Verdict per world via the tip + slide detectors (vectorized, no per-world traces)
    R = quat_to_mat(r_pquat)
    rel = r_ipos - r_ppos[:, :, None, :]
    ipal = np.einsum("fwji,fwkj->fwki", R, rel)
    settle_rec = next(i for i, s in enumerate(rec_steps) if s >= n_settle)
    x, y = r_pquat[settle_rec:, :, 1], r_pquat[settle_rec:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))  # (f, w)
    tipped = (tilt > 8.0).any(axis=0)
    slide = np.linalg.norm(ipal[settle_rec:] - ipal[settle_rec][None], axis=3).max(axis=2)
    slid = (slide > 0.05).any(axis=0)
    unsafe = (tipped | slid).reshape(len(speeds), len(accels))

    print("\nmax safe speed by accel (tip/slide detectors only):")
    for a_i, a in enumerate(accels):
        safe_speeds = speeds[~unsafe[:, a_i]]
        ms = f"{safe_speeds.max():.2f} m/s" if len(safe_speeds) else "NONE"
        print(f"  accel {a:5.2f} m/s^2 -> {ms}")


def main():
    s = get_scenario(SLUG)
    config = s.pallet
    base = s.suggested_profile
    profiles = [
        ConveyorProfile(target_speed_mps=float(v), accel_mps2=base.accel_mps2,
                        duration_s=base.duration_s, shape=base.shape)
        for v in SPEEDS
    ]
    print(f"scenario: {SLUG} ({len(config.items)} items, "
          f"accel {base.accel_mps2} m/s^2, {base.duration_s} s)")

    cpu_prod = cpu_arm(config, profiles, mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                       "cpu/implicitfast/fp64")
    cpu_euler = cpu_arm(config, profiles, mujoco.mjtIntegrator.mjINT_EULER,
                        "cpu/euler/fp64      ")
    gpu = gpu_arm(config, profiles)
    if gpu is None:
        return

    print(f"\n{'speed':>6} | {'cpu prod (implicitfast)':>28} | "
          f"{'cpu euler':>28} | {'gpu euler fp32':>28}")
    agree = 0
    for i, v in enumerate(SPEEDS):
        def fmt(m, t):
            return f"{m.value}@{t:.2f}s" if t is not None else m.value
        row = [fmt(*cpu_prod[i]), fmt(*cpu_euler[i]), fmt(*gpu[i])]
        mark = "" if cpu_euler[i][0] == gpu[i][0] else "  <-- MISMATCH vs cpu-euler"
        agree += cpu_euler[i][0] == gpu[i][0]
        print(f"{v:6.2f} | {row[0]:>28} | {row[1]:>28} | {row[2]:>28}{mark}")
    print(f"\ngpu vs cpu-euler mode agreement: {agree}/{len(SPEEDS)}")

    grid_test(config, base)

    # World-count scaling: same physics, more parallelism per kernel launch.
    for ns, na in ((64, 32), (128, 32), (128, 64)):
        grid_test(config, base, n_speeds=ns, n_accels=na, verbose=False)


if __name__ == "__main__":
    main()
