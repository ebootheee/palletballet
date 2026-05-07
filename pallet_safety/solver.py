"""Run a pallet through a conveyor profile in MuJoCo and capture a state trace.

Phase C of PLAN_V2. The solver does not decide pass/fail — that's done by
`failures.py` against the resulting `SimulationTrace`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .friction import DEFAULT_PAIR
from .mjcf_builder import build_mjcf
from .models import PalletConfig

ProfileShape = Literal["accel_hold", "constant", "ramp_decel"]


class ConveyorProfile(BaseModel):
    """Belt-velocity profile over time.

    Shapes:
      - accel_hold: ramp 0 → target over `target/accel` seconds, then hold.
      - constant:   instant target speed (worst-case slip transient).
      - ramp_decel: hold target, then ramp target → 0 starting at `decel_start_s`.
    """
    model_config = ConfigDict(frozen=True)

    target_speed_mps: float = Field(default=0.5, ge=0, le=5.0)
    accel_mps2: float = Field(default=0.5, gt=0, le=10.0)
    duration_s: float = Field(default=5.0, gt=0, le=30.0)
    shape: ProfileShape = "accel_hold"
    decel_start_s: float = Field(default=3.0, ge=0)

    def velocity_at(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if self.shape == "constant":
            return self.target_speed_mps
        if self.shape == "ramp_decel":
            ramp_up = self.target_speed_mps / max(self.accel_mps2, 1e-9)
            if t < ramp_up:
                return self.accel_mps2 * t
            if t < self.decel_start_s:
                return self.target_speed_mps
            decel_t = t - self.decel_start_s
            return max(0.0, self.target_speed_mps - self.accel_mps2 * decel_t)
        # accel_hold (default)
        ramp_up = self.target_speed_mps / max(self.accel_mps2, 1e-9)
        if t < ramp_up:
            return self.accel_mps2 * t
        return self.target_speed_mps


@dataclass
class SimulationTrace:
    """Time series of pallet + item state during a single sim run."""
    times: np.ndarray                # (n_steps,)
    conveyor_vel: np.ndarray         # (n_steps,) belt velocity
    pallet_pos: np.ndarray           # (n_steps, 3)
    pallet_quat: np.ndarray          # (n_steps, 4)
    pallet_lin_vel: np.ndarray       # (n_steps, 3)
    pallet_ang_vel: np.ndarray       # (n_steps, 3)
    item_world_pos: np.ndarray       # (n_steps, n_items, 3)
    item_world_quat: np.ndarray      # (n_steps, n_items, 4) — orientation per item
    item_pallet_pos: np.ndarray      # (n_steps, n_items, 3) in pallet frame
    item_initial_pallet_pos: np.ndarray  # (n_items, 3)
    config: PalletConfig
    profile: ConveyorProfile
    runtime_s: float = 0.0
    n_items: int = 0
    n_steps: int = 0

    def downsample(self, hz: float) -> "SimulationTrace":
        """Return a trace subsampled to the given output rate."""
        if hz <= 0 or len(self.times) == 0:
            return self
        timestep = 1.0 / hz
        kept_idx = [0]
        last_t = self.times[0]
        for i, t in enumerate(self.times):
            if t - last_t >= timestep:
                kept_idx.append(i)
                last_t = t
        idx = np.array(kept_idx)
        return SimulationTrace(
            times=self.times[idx], conveyor_vel=self.conveyor_vel[idx],
            pallet_pos=self.pallet_pos[idx], pallet_quat=self.pallet_quat[idx],
            pallet_lin_vel=self.pallet_lin_vel[idx],
            pallet_ang_vel=self.pallet_ang_vel[idx],
            item_world_pos=self.item_world_pos[idx],
            item_world_quat=self.item_world_quat[idx],
            item_pallet_pos=self.item_pallet_pos[idx],
            item_initial_pallet_pos=self.item_initial_pallet_pos,
            config=self.config, profile=self.profile, runtime_s=self.runtime_s,
            n_items=self.n_items, n_steps=len(idx),
        )


def build_model(
    config: PalletConfig,
    surface_pair: tuple[str, str] = DEFAULT_PAIR,
) -> mujoco.MjModel:
    """Compile the MJCF for a pallet config into a reusable MuJoCo model.

    Intended for callers (e.g., the threshold solver) that need to run many
    simulations on the same physical configuration and would otherwise pay the
    XML compile cost per run.
    """
    xml = build_mjcf(config, surface_pair=surface_pair, actuated_conveyor=True)
    return mujoco.MjModel.from_xml_string(xml)


def simulate(
    config: PalletConfig,
    profile: ConveyorProfile,
    surface_pair: tuple[str, str] = DEFAULT_PAIR,
    settle_s: float = 0.5,
    model: mujoco.MjModel | None = None,
    fail_fast_tip_deg: float | None = None,
    fail_fast_slide_m: float | None = None,
) -> SimulationTrace:
    """Simulate the pallet on the conveyor under the given speed profile.

    A short `settle_s` window is run with conveyor at rest to let the pallet
    drop and settle on the belt before the velocity ramp begins.

    Pass a pre-built `model` (from `build_model()`) to skip the XML compile
    step — useful when calling `simulate` many times for the same pallet.

    `fail_fast_tip_deg`: abort sim when the pallet tilts past this angle.
    `fail_fast_slide_m`: abort when any item has moved this far (in pallet
    frame) from its initial position. Both are used by the threshold solver
    to skip the rest of a clearly-failing run.
    """
    if model is None:
        model = build_model(config, surface_pair=surface_pair)
    data = mujoco.MjData(model)

    # Cache addresses outside the loop — saves O(per-step) dict lookups
    pallet_id = int(model.body("pallet_base").id)
    item_ids = np.array([int(model.body(f"item_{i}").id) for i in range(len(config.items))],
                        dtype=np.int32)
    n_items = len(item_ids)
    pallet_qv0 = int(model.joint("pallet_joint").dofadr[0])
    conveyor_qv0 = int(model.joint("conveyor_slide").dofadr[0])

    timestep = float(model.opt.timestep)
    n_settle = int(settle_s / timestep)
    n_run = int(profile.duration_s / timestep)
    total_steps = n_settle + n_run

    times = np.zeros(total_steps)
    conveyor_vel = np.zeros(total_steps)
    pallet_pos = np.zeros((total_steps, 3))
    pallet_quat = np.zeros((total_steps, 4))
    pallet_lin_vel = np.zeros((total_steps, 3))
    pallet_ang_vel = np.zeros((total_steps, 3))
    item_world_pos = np.zeros((total_steps, n_items, 3))
    item_world_quat = np.zeros((total_steps, n_items, 4))
    item_pallet_pos = np.zeros((total_steps, n_items, 3))

    t0 = time.perf_counter()
    qvel = data.qvel
    xpos = data.xpos
    xquat = data.xquat
    xmat = data.xmat
    ctrl = data.ctrl

    cos_tip_limit = (
        float(np.cos(np.radians(fail_fast_tip_deg)))
        if fail_fast_tip_deg is not None
        else None
    )
    slide_limit_sq = (
        fail_fast_slide_m * fail_fast_slide_m
        if fail_fast_slide_m is not None else None
    )
    initial_item_pallet_pos: np.ndarray | None = None
    stopped_at = total_steps

    for step in range(total_steps):
        sim_t = (step - n_settle) * timestep
        ctrl[0] = profile.velocity_at(sim_t) if step >= n_settle else 0.0
        mujoco.mj_step(model, data)

        times[step] = max(sim_t, 0.0)
        conveyor_vel[step] = qvel[conveyor_qv0]
        pallet_pos[step] = xpos[pallet_id]
        pq = xquat[pallet_id]
        pallet_quat[step] = pq
        pallet_lin_vel[step] = qvel[pallet_qv0:pallet_qv0 + 3]
        pallet_ang_vel[step] = qvel[pallet_qv0 + 3:pallet_qv0 + 6]

        if n_items > 0:
            # Vectorized: gather all item positions + orientations at once
            iw = xpos[item_ids]
            item_world_pos[step] = iw
            item_world_quat[step] = xquat[item_ids]
            # Pallet-frame: R^T @ (item_world - pallet_world)
            R = xmat[pallet_id].reshape(3, 3)
            rel = iw - xpos[pallet_id]
            item_pallet_pos[step] = rel @ R  # (rel @ R) == (R.T @ rel.T).T for row vectors

        # Early-termination checks (only post-settle)
        if step >= n_settle:
            if step == n_settle and n_items > 0:
                initial_item_pallet_pos = item_pallet_pos[step].copy()
            if cos_tip_limit is not None:
                cos_tilt = 1.0 - 2.0 * (pq[1] * pq[1] + pq[2] * pq[2])
                if cos_tilt < cos_tip_limit:
                    stopped_at = step + 1
                    break
            if slide_limit_sq is not None and n_items > 0 \
                    and initial_item_pallet_pos is not None:
                delta = item_pallet_pos[step] - initial_item_pallet_pos
                max_sq = np.max(np.sum(delta * delta, axis=1))
                if max_sq > slide_limit_sq:
                    stopped_at = step + 1
                    break

    runtime_s = time.perf_counter() - t0

    initial = item_pallet_pos[n_settle].copy() if n_settle < total_steps else item_pallet_pos[0].copy()

    # Slice off the settle period AND any unused tail from early-termination
    sl = slice(n_settle, stopped_at)
    return SimulationTrace(
        times=times[sl], conveyor_vel=conveyor_vel[sl],
        pallet_pos=pallet_pos[sl], pallet_quat=pallet_quat[sl],
        pallet_lin_vel=pallet_lin_vel[sl], pallet_ang_vel=pallet_ang_vel[sl],
        item_world_pos=item_world_pos[sl],
        item_world_quat=item_world_quat[sl],
        item_pallet_pos=item_pallet_pos[sl],
        item_initial_pallet_pos=initial.copy(),
        config=config, profile=profile, runtime_s=runtime_s,
        n_items=n_items, n_steps=int(sl.stop - sl.start),
    )
