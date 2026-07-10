# mjwarp weld-ring reproduction

Minimal reproduction of a fidelity gap in [mujoco_warp](https://github.com/google-deepmind/mujoco_warp):
**weld-constrained stacked boxes never settle** — they oscillate/drift at
mm scale on GPU while classic MuJoCo (same model, same Euler integrator,
fp64) sits still at micron scale.

## The scene

`weld_ring_12item.xml`: a pallet-like base box on a static plane, 12 boxes
stacked in two layers, every box welded to the base
(`solref="0.02 1" solimp="0.95 0.99 0.001"` — this project's stretch-wrap
model). No actuators. Nothing external moves the scene. It was generated
from PalletBallet's MJCF builder (seed-0 random pallet, first 12 items) and
then made standalone.

## Run it

```bash
pip install mujoco mujoco-warp[cuda]   # tested: mujoco 3.10.0, mujoco-warp 3.10.0.1, warp-lang 1.15.0
python repro.py
```

Output on an RTX 5090 (sm_120, driver 595.71.05, CUDA 13.2):

```
classic mujoco (fp64/euler): max item drift 0.040 mm
mujoco_warp   (fp32/euler): max item drift 8.600 mm
```

The mjwarp number varies run to run (we've seen 4–16 mm on the same box —
the oscillation itself is nondeterministic); the classic-MuJoCo number is
stable at 0.040 mm. The gap is always two orders of magnitude or more.

That's the max displacement of any item relative to the base over 3 s,
measured after a 1 s settle. The integrator is forced to Euler on **both**
backends, so the difference is in the constraint solve, not the integration
scheme.

## What we ruled out while isolating this

- **Integrator** — classic MuJoCo with Euler+fp64 is as quiet as implicitfast.
- **Buffer overflow** — `nconmax`/`njmax` passed explicitly with headroom, no
  overflow warnings.
- **Contact divergence** — post-settle contact counts/depths are close
  between backends on related scenes.
- **Settle time** — extending settle from 0.5 s to 2 s doesn't reduce it
  (steady-state oscillation, not compaction).
- **solref softening** — behaves non-monotonically: `timeconst=0.05` rings
  *worse* than the 0.02 default; 0.1 goes quiet but changes the modeled
  physics.

Item count matters: the ring survives at 12 boxes, vanishes below ~8, and
reaches 2–3.5 cm on the full 48-box pallet — it appears to need the weld
network coupled through box-box contacts. Batched identical worlds also
diverge from each other (~2× spread in our failure metric over 72 copies of
a knife-edge scene).

## Context

Found while validating a GPU envelope-search backend for
[PalletBallet](https://boothe.io/palletballet) against its fp64 CPU solver:
a 22,104-run agreement study (307 pallets × 72 conveyor profiles, both
backends, identical failure detectors) came back at 55% verdict agreement,
almost entirely load-shift false-positives on wrapped (welded) pallets.
Write-up: [boothe.io/posts/palletballet-gpu-experiment](https://boothe.io/posts/palletballet-gpu-experiment).
Study harness: `scripts/agreement_study.py` on the
[`gpu-spike`](https://github.com/ebootheee/palletballet/tree/gpu-spike) branch.
