"""Batched conveyor-profile rollouts on GPU via MuJoCo Warp.

One pallet, many profiles: every world shares the model topology and only the
per-world belt-velocity control differs. Results come back as downsampled
`SimulationTrace`s so the standard `failures.py` detectors run unchanged on
either backend.

Numerical caveats vs the CPU solver (see PLAN_GPU.md):
  - fp32 instead of fp64
  - Euler integrator (mjwarp does not support implicitfast)
  - no per-world early termination: every world runs the full duration

PLAN_GPU Phase 2 core. Requires the `gpu` extra: `uv sync --extra gpu`.
"""

from __future__ import annotations

try:
    import mujoco_warp as mjw
    import warp as wp
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "pallet_safety.gpu requires the 'gpu' extra: uv sync --extra gpu"
    ) from e

import mujoco
import numpy as np

from ..friction import DEFAULT_PAIR
from ..models import PalletConfig
from ..solver import ConveyorProfile, SimulationTrace, build_model

# Constraint/contact headroom. mjwarp's defaults overflow on our scenes
# (njmax 144 at 8 items; nconmax 546 at 25 items), silently dropping
# constraints until the sim goes NaN. Scale with item count — the cost is
# only megabytes.


def _buffer_sizes(n_items: int) -> tuple[int, int]:
    nconmax = max(1024, 96 * n_items)
    return nconmax, 4 * nconmax


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
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


def _ctrl_table(
    profiles: list[ConveyorProfile], n_settle: int, total_steps: int, timestep: float,
) -> np.ndarray:
    """Belt-velocity lookup table, (total_steps, nworld) float32."""
    t = (np.arange(total_steps) - n_settle) * timestep
    run_t = np.maximum(t, 0.0)
    out = np.zeros((total_steps, len(profiles)), dtype=np.float32)
    for w_i, p in enumerate(profiles):
        if p.shape == "accel_hold":
            out[:, w_i] = np.minimum(run_t * p.accel_mps2, p.target_speed_mps)
        elif p.shape == "constant":
            out[:, w_i] = np.where(t > 0, p.target_speed_mps, 0.0)
        else:  # ramp_decel or future shapes: defer to the scalar reference impl
            out[:, w_i] = [p.velocity_at(ti) if ti > 0 else 0.0 for ti in t]
    out[:n_settle] = 0.0
    return out


def run_profile_grid(
    config: PalletConfig,
    profiles: list[ConveyorProfile],
    *,
    surface_pair: tuple[str, str] = DEFAULT_PAIR,
    settle_s: float = 0.5,
    record_every: int = 15,
    use_graph: bool = False,
    nconmax: int | None = None,
    njmax: int | None = None,
) -> list[SimulationTrace]:
    """Simulate one pallet under many conveyor profiles in a single GPU batch.

    All profiles must share `duration_s` (one time axis per batch). Returns a
    post-settle `SimulationTrace` per profile at ~1/(record_every*timestep) Hz.

    `use_graph` captures the whole rollout as a CUDA graph: ~3x faster
    stepping, but capture itself costs seconds — worth it for large `nworld`,
    not for one-shot small batches.
    """
    durations = {p.duration_s for p in profiles}
    if len(durations) != 1:
        raise ValueError(f"all profiles must share duration_s, got {sorted(durations)}")

    wp.init()
    nworld = len(profiles)
    mjm = build_model(config, surface_pair=surface_pair)
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    default_ncon, default_nj = _buffer_sizes(len(config.items))
    m = mjw.put_model(mjm)
    d = mjw.put_data(
        mjm, mjd, nworld=nworld,
        nconmax=nconmax or default_ncon, njmax=njmax or default_nj,
    )

    pallet_id = int(mjm.body("pallet_base").id)
    item_ids = np.array(
        [int(mjm.body(f"item_{i}").id) for i in range(len(config.items))], dtype=np.int64
    )
    pallet_qv0 = int(mjm.joint("pallet_joint").dofadr[0])
    conveyor_qv0 = int(mjm.joint("conveyor_slide").dofadr[0])

    timestep = float(mjm.opt.timestep)
    n_settle = int(settle_s / timestep)
    total_steps = n_settle + int(profiles[0].duration_s / timestep)

    ctrl_dev = wp.array(
        _ctrl_table(profiles, n_settle, total_steps, timestep)[:, :, None],
        dtype=wp.float32,
    )

    rec_steps = list(range(0, total_steps, record_every))
    rec_set = set(rec_steps)
    n_rec = len(rec_steps)
    rec_xpos = wp.empty((n_rec,) + d.xpos.shape, dtype=d.xpos.dtype)
    rec_xquat = wp.empty((n_rec,) + d.xquat.shape, dtype=d.xquat.dtype)
    rec_qvel = wp.empty((n_rec,) + d.qvel.shape, dtype=d.qvel.dtype)

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

    import time as _time

    t0 = _time.perf_counter()
    if use_graph:
        with wp.ScopedCapture() as capture:
            rollout()
        wp.capture_launch(capture.graph)
    else:
        rollout()
    wp.synchronize()
    runtime_s = _time.perf_counter() - t0

    xpos = rec_xpos.numpy().astype(np.float64)   # (n_rec, nworld, nbody, 3)
    xquat = rec_xquat.numpy().astype(np.float64)
    qvel = rec_qvel.numpy().astype(np.float64)

    if np.isnan(xpos).any() or np.isnan(xquat).any():
        raise FloatingPointError(
            f"NaN in GPU rollout for pallet {config.pallet_id!r} "
            f"(nworld={nworld}); check njmax/nconmax and integrator stability"
        )

    times = np.maximum((np.array(rec_steps) - n_settle) * timestep, 0.0)
    ppos = xpos[:, :, pallet_id]
    pquat = xquat[:, :, pallet_id]
    ipos = xpos[:, :, item_ids]
    iquat = xquat[:, :, item_ids]
    # Item positions in the pallet frame: R^T (item - pallet)
    R = _quat_to_mat(pquat)
    rel = ipos - ppos[:, :, None, :]
    ipal = np.einsum("fwji,fwkj->fwki", R, rel)

    first_rec = next(i for i, s in enumerate(rec_steps) if s >= n_settle)
    sl = slice(first_rec, n_rec)
    n_kept = n_rec - first_rec

    traces: list[SimulationTrace] = []
    for w_i, p in enumerate(profiles):
        traces.append(SimulationTrace(
            times=times[sl],
            conveyor_vel=qvel[sl, w_i, conveyor_qv0],
            pallet_pos=ppos[sl, w_i],
            pallet_quat=pquat[sl, w_i],
            pallet_lin_vel=qvel[sl, w_i, pallet_qv0:pallet_qv0 + 3],
            pallet_ang_vel=qvel[sl, w_i, pallet_qv0 + 3:pallet_qv0 + 6],
            item_world_pos=ipos[sl, w_i],
            item_world_quat=iquat[sl, w_i],
            item_pallet_pos=ipal[sl, w_i],
            item_initial_pallet_pos=ipal[first_rec, w_i].copy(),
            config=config,
            profile=p,
            runtime_s=runtime_s,
            n_items=len(item_ids),
            n_steps=n_kept,
        ))
    return traces
