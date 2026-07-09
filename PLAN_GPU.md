# PLAN_GPU — MuJoCo Warp envelope grids on consumer GPUs

**Thesis.** [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (NVIDIA +
Google DeepMind, part of the Newton project) is a GPU reimplementation of the exact
engine this project runs. It consumes the same compiled `MjModel` our
`mjcf_builder` produces and steps `nworld` independent copies of it per call —
with per-world control inputs. Our envelope search is the perfect shape for
this: **one pallet, many conveyor profiles**. Every world shares the model
topology; only `ctrl` differs.

If it works, a single consumer GPU handles the simulations behind hundreds of
simultaneous safety-margin decisions, on premise, with the float64 CPU solver
as complete redundancy.

**Hardware on hand:** 2× RTX 5090 (32GB each), driver 595, CUDA 13.2 — exceeds
mujoco-warp's requirements. The dev box is also the production home server.

**Prime directive:** classic CPU MuJoCo stays the source of truth and the
serving engine for `/solve`. The GPU backend must *earn* trust through the
agreement study (Phase 1) before anything user-facing depends on it.

---

## Phase 0 — Spike (half a day)

Goal: one canonical scenario stepping on the GPU, end to end.

- [ ] Add optional extra in `pyproject.toml`: `[project.optional-dependencies] gpu = ["mujoco-warp[cuda]>=3.10", "warp-lang"]`. Pin the minor version — the package is alpha and the API churns (Newton went 1.0→1.3 in 3 months).
- [ ] `uv sync --extra gpu`; smoke script in `scripts/gpu_spike.py`:
  - `build_model(get_scenario("tall-unwrapped-tower").pallet)` → `mjw.put_model` → `make_data(nworld=8)` → step loop with per-world `ctrl` (8 different target speeds).
  - Verify: the velocity actuator (`conveyor_slide`), freejoints, weld wrap constraints, and box-box contacts all behave. These are core MuJoCo features but the mjwarp support matrix is incomplete — **this is the go/no-go gate**. If welds or the velocity actuator are unsupported, stop and reassess.
- [ ] Note first-call Warp kernel-compile time (cached afterward) and steady-state steps/sec vs CPU.

Known constraints to design around, from the MJWarp docs:
- fp32 (CPU is fp64) — the reason Phase 1 exists.
- `nconmax`/`njmax` must be sized up front; our scenes are small (≤ ~30 bodies) so generous limits are cheap.
- No per-world early termination — worlds run the full duration. The grid doesn't need fail-fast: wall-clock is set by the slowest world, which is the same duration for all.
- `IMPLICITFAST` integrator and `PGS` solver unsupported — check what `mjcf_builder` emits in `<option>` and confirm the fallback (Euler/Newton solver) matches CPU settings, or align both sides explicitly.

### Phase 0 results (2026-07-08, `scripts/gpu_spike.py`, 1× RTX 5090)

**GO.** All load-bearing MJCF features work on mjwarp 3.10: weld wrap
constraints, freejoints, box-box contacts, velocity-actuated conveyor.

- Verdicts: 8/8 failure-mode agreement (GPU/euler/fp32 vs CPU/euler/fp64) on
  tall-unwrapped-tower across 8 speeds; failure times within one 33 Hz frame.
  CPU implicitfast vs euler also agreed 8/8 — integrator confound looks small.
- Gotchas found: default `njmax` overflows at 144 for an 8-item scene (dropped
  constraints → NaNs) — pass `nconmax=256, njmax=1024` explicitly.
  `implicitfast` unsupported → override integrator to Euler before `put_model`.
  First-process Warp kernel compile ≈ 26 s (disk-cached afterward).
- Throughput (1000-step rollouts, device-side ctrl table + trace recording,
  whole rollout as one CUDA graph, single sync):
  | worlds | rollout | steps/s | vs 16-core CPU |
  |---|---|---|---|
  | 512 | 1.8 s | 279k | 1.0× |
  | 2048 | 2.6 s | 790k | 2.7× |
  | 4096 | 4.0 s | 1.01M | 3.5× |
  | 8192 | 6.8 s | 1.20M | 4.1× |
- **Bottleneck is not physics: CUDA-graph capture is ~5–6 s per pallet**
  (scales with kernel-launch count, ~70/step × 1000 steps). Phase 2 must
  either (a) capture a 1-step graph with a device-side step counter indexing
  the ctrl table, replayed N times, or (b) reuse a whole-rollout graph across
  same-topology pallets by overwriting `m` field arrays in place. Without a
  graph, per-step Python launch overhead caps everything at ~88–300 sims/s.
- fp32 boundary jitter is visible: max-safe-speed rows show ±1-grid-cell
  non-monotonicity near the failure edge — exactly the flip band Phase 1
  quantifies.

## Phase 1 — Agreement study (the gate)

Goal: quantify fp32-GPU vs fp64-CPU disagreement before trusting any GPU verdict.

- [ ] `scripts/agreement_study.py`:
  - Corpus: all 7 registry scenarios + 300 random pallets (`MockRandomAdapter`, seeds 0–299).
  - For each pallet, a coarse profile grid (e.g. 12 speeds × 6 accels = 72 profiles).
  - Run every (pallet, profile) on both backends; feed both traces through the *same* `failures.py` detectors (GPU traces copied back to numpy — detectors are backend-agnostic array reductions).
  - Record: verdict (safe/fail), failure mode, failure time.
- [ ] Metrics + acceptance criteria:
  - **Verdict agreement ≥ 99%** on cells whose CPU failure margin is comfortably away from the threshold (e.g. tip angle < 6° or > 10° when the limit is 8°).
  - Near-boundary cells may legitimately flip — report the flip band width per failure mode instead of pass/fail.
  - **Derived envelope agreement:** max safe speed from the GPU grid within one search-precision step (0.05 m/s) of CPU bisection for ≥ 95% of pallets.
- [ ] Output: `data/agreement_study.parquet` + a summary table. This becomes the centerpiece chart of the follow-up blog post regardless of outcome — "here is exactly where fp32 physics diverges" is a good post even if the answer is "too much."

### Phase 1 findings so far (2026-07-08, smoke corpus: 7 scenarios + 2 random)

The naive gate **fails**, and the failure is informative:

- Verdict agreement only ~85% (gate: ≥99%). But 92 of 97 disagreements are
  **safe-side** — GPU fails cells the CPU passes. The GPU backend
  systematically over-predicts `top_item_slide` on long rollouts (item
  "walking" under sustained belt motion: cell criticality 1.42 on GPU vs 0.53
  on CPU at 2 ms; halving/quartering the timestep converges it down to ~1.1,
  never to CPU). Not an integrator artifact: CPU-euler-fp64 matches
  CPU-implicitfast-fp64 (0.61 vs 0.53). Not gross collision divergence either
  (post-settle manifolds: 42 vs 36 contacts, similar depths). Residual suspects:
  fp32 friction stick-slip creep and solver behavior under mjwarp's Newton
  implementation.
- 5 of 97 disagreements are **dangerous-side** (CPU-unsafe, GPU-safe): 2 are
  knife-edge flips (criticality ≈ 1.0 both sides), 3 are `load_shift` misses
  on frozen-pallet-jerk-start — a *wrapped* pallet, so the weld-constraint
  solve is implicated. Track upstream; retest each mjwarp release.
- Batched worlds are **not deterministic**: 72 identical worlds spread
  criticality 0.98–2.17 on a knife-edge cell (fp32 atomics × stick-slip
  chaos). Near-edge GPU verdicts are samples, not answers.

**Architectural consequence — two-stage hybrid.** The GPU cannot be the sole
verdict source at mjwarp 3.10 fidelity. It doesn't need to be: GPU sweeps the
full profile grid (the map: modes, gradients, cliffs); the fp64 CPU solver
re-verifies the envelope edge — the boundary band the decision actually rests
on (~30–60 cells, well under a second across cores). CPU stays the authority
on every published number; the GPU buys the map and a ~10× reduction in what
the CPU must simulate. The safe-side bias means the GPU map errs toward
tighter envelopes, never looser ones, which is the correct failure direction
for a safety product.

Full 300-pallet study: `data/agreement_study.parquet` (see summary in repo).

## Phase 2 — Grid backend

Goal: `GridAnalyzer` — the GPU-native replacement for double bisection.

- [ ] `pallet_safety/gpu/grid.py` (import guarded; raises a clear error without the `gpu` extra):
  - `GridAnalyzer.analyze_grid(config, speeds, accels) -> EnvelopeGrid`
  - One `put_model` per pallet; `nworld = len(speeds) × len(accels)` (default 32×16 = 512); per-world `ctrl` written each step from precomputed `ConveyorProfile.velocity_at` lookup tables (profiles are known ahead of time — build a `(nworld, nsteps)` control array once, on device).
  - Record downsampled traces (30 Hz) into batched device arrays; copy back once at the end; run existing detectors per world. (Optimization pass later: move detector reductions into Warp kernels and copy back only verdicts — only if the trace copy shows up in profiles.)
- [ ] `EnvelopeGrid` model: speeds, accels, per-cell (mode, failure_time), plus derived `max_speed_at_accel` curve — a strict superset of today's `SafetyResult` numbers.
- [ ] Derive a `SafetyResult` from the grid so downstream consumers don't care which backend produced it. Set `confidence` from Phase 1 flip-band data — the GPU backend finally gives that field an empirical meaning.
- [ ] Tests: skip-if-no-CUDA; grid-derived SafetyResult vs CPU analyzer on the 7 scenarios within tolerance; determinism (same seed, same grid, run twice → identical verdicts).

## Phase 3 — Wire-up

- [ ] API: `POST /safety/grid` (new endpoint, not a mode flag — the response shape is different). CPU fallback: if the GPU extra isn't installed, 501 with a clear message. `/safety/analyze` unchanged.
- [ ] `scripts/batch_study.py --backend gpu`: loop pallets, one grid call each; `--gpus 2` runs one worker process per GPU (`CUDA_VISIBLE_DEVICES` pinning). Target: rerun the 400-pallet study, then scale to 40k pallets and see if the bimodal split (60/33/7) holds.
- [ ] Benchmarks to publish: sims/sec and pallets/sec GPU vs 16-core CPU; VRAM per 512-world grid; warmup cost; single-grid latency (the "could this serve interactive requests?" number).

## Phase 4 — Product surface (only after 1–3 hold up)

- [ ] Game reveal panel: render the failure-mode heatmap behind the envelope reveal — "here's the whole map; you were here."
- [ ] Follow-up blog post: the agreement study, the heatmaps, the 40k-pallet rerun, benchmark tables, and the on-premise redundancy argument.
- [ ] Optional: GPU serving from the home server (nvidia-container-toolkit in `infra/docker-compose.yml`) — decide only when there's a consumer for sub-second grid calls.

## Risks / escape hatches

| Risk | Detection | Response |
| --- | --- | --- |
| Welds / velocity actuator unsupported in mjwarp | Phase 0 spike | Stop; revisit on a later mjwarp release (it's under heavy active development) |
| fp32 verdict flips too wide | Phase 1 metrics | Publish the divergence study; keep GPU for coarse screening + CPU for final verdicts (two-stage) |
| Alpha API churn breaks us | Pinned minor version; CI job with `--extra gpu` on the GPU box | Bump deliberately, rerun agreement study on every bump |
| VRAM blowup from trace recording | Phase 2 benchmark | Downsample harder, or reduce to on-device running detector state |

## Sequencing

Phase 0 and 1 are one branch (`gpu-spike`) and answer the only real question.
Phases 2–3 are a second PR. Phase 4 is its own effort. Total estimate: 0+1 in a
day of focused work, 2+3 in another day or two — most of it in the agreement
harness, which is deliberately the most careful part.
