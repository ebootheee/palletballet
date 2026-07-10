"""Repro: weld-constrained stacked boxes oscillate/drift in mujoco_warp
while classic MuJoCo (same model, same Euler integrator) stays still.

Scene: a pallet-like base box on a static plane, 12 boxes stacked in two
layers, every box welded to the base (modeling stretch wrap). No actuators,
nothing moves the scene: after settling, everything should be at rest.

Classic MuJoCo (fp64, Euler): max item drift ~0.05 mm over 3 s.  mjwarp:
items keep vibrating/walking, mm–cm scale. Deleting items makes it vanish
below ~8-10 boxes.

Tested: mujoco 3.10.0, mujoco-warp 3.10.0.1, warp-lang 1.15.0, RTX 5090
(sm_120), driver 595.71.05 / CUDA 13.2, Ubuntu.
"""

from pathlib import Path

import mujoco
import numpy as np

XML_PATH = str(Path(__file__).parent / "weld_ring_12item.xml")
DURATION_S = 3.0
SKIP_S = 1.0  # ignore initial settling


def item_drift_mm(xpos_series: np.ndarray, base_idx: int, item_idx: list[int],
                  skip: int) -> float:
    """Max displacement of any item relative to the base, vs its post-settle pose."""
    rel = xpos_series[:, item_idx] - xpos_series[:, [base_idx]]
    return float(np.linalg.norm(rel[skip:] - rel[skip], axis=-1).max() * 1000)


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER  # same on both backends
    n_steps = int(DURATION_S / model.opt.timestep)
    skip = int(SKIP_S / model.opt.timestep)
    base = model.body("pallet_base").id
    items = [model.body(f"item_{i}").id for i in range(12)]

    # ---- classic MuJoCo ----
    data = mujoco.MjData(model)
    xpos = np.zeros((n_steps, model.nbody, 3))
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        xpos[i] = data.xpos
    print(f"classic mujoco (fp64/euler): max item drift "
          f"{item_drift_mm(xpos, base, items, skip):.3f} mm")

    # ---- mujoco_warp ----
    import mujoco_warp as mjw
    import warp as wp
    wp.init()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    m = mjw.put_model(model)
    d = mjw.put_data(model, data, nworld=1, nconmax=512, njmax=2048)
    xpos = np.zeros((n_steps, model.nbody, 3))
    for i in range(n_steps):
        mjw.step(m, d)
        xpos[i] = d.xpos.numpy()[0]
    print(f"mujoco_warp   (fp32/euler): max item drift "
          f"{item_drift_mm(xpos, base, items, skip):.3f} mm")


if __name__ == "__main__":
    main()
