# Pallet Safety Command

Per-pallet physics inference for cold-storage conveyor speed governance.

**Play it live:** [boothe.io/palletballet](https://boothe.io/palletballet) — dial a belt speed, dispatch a real MuJoCo sim, watch the 3D replay, then see the envelope the solver would have set.
**API:** [palletballet-api.boothe.io/docs](https://palletballet-api.boothe.io/docs)
**Deep-dive:** [boothe.io/posts/launching-palletballet](https://boothe.io/posts/launching-palletballet)

Running your own instance? The live game page plays against any clone:
`https://boothe.io/palletballet?api=http://localhost:8000` (the default CORS
allowlist already permits it).

This project turns a scanned or configured pallet into a safe conveyor profile:
maximum speed, maximum acceleration, dominant failure mode, confidence, and a
trace explaining why. The Streamlit UI is intentionally visual and game-like so
operators can build trust by toppling known-bad loads in real time. The real
product path is the API: clone this repo, feed it pallet configs, and run
simulation batches at scale.

## Why this exists

Cold-storage pallets are not uniform rigid boxes. They are stacks of products
with different mass, temperature, geometry, wrap, overhang, and center of mass.
A static conveyor speed table is conservative on good pallets and still blind to
weird ones.

Pallet Safety Command answers a narrower, more operational question:

> Given this actual pallet, in this thermal state, what conveyor motion profile
> is safe right now?

The system models pallet geometry, cold-chain friction, contact dynamics, and
failure modes such as pallet slip, top-item slide, load shift, and tip-over.

## What is working now

- Streamlit operator console at `http://localhost:8501`
- FastAPI service at `http://localhost:8000/docs`
- Canonical scenario registry (`GET /scenarios`) shared by the console, the
  API, and the public web game
- `POST /solve` with `include_replay`: per-frame 6-DoF poses for the pallet
  and every item — a browser can replay the sim in 3D (~60 KB at 30 Hz)
- SKU catalog with cold-storage product templates
- Mock scanner/random adapter for repeatable dev and batch testing
- Manual stack-by-stack pallet builder
- Known failure scenarios for quick demo topples
- MuJoCo-backed solver and Plotly replay/trace views
- Threshold analyzer for max safe speed and acceleration
- Batch runner (`scripts/batch_study.py`): ~7.5 pallets/s on a 16-core box
- Pydantic API contracts for `RawInputs`, `PalletConfig`, and `SafetyResult`
- Test suite covering catalog, configurator, friction, MJCF, solver, API,
  scenarios, replay, and threshold behavior

## Product flow

```text
Scanner / WMS / manual stack builder
        |
        v
RawInputs
        |
        v
Configurator -> PalletConfig
        |
        +--> UI replay and trust-building demos
        |
        +--> MuJoCo solver -> failure trace
        |
        +--> Threshold analyzer -> SafetyResult
        |
        v
API client / future PLC integration / batch studies
```

## UI tour

- **Mission Control**: pick a scenario, run the conveyor hit, inspect the load
  story, replay, signal trace, and safety envelope.
- **Build Pallet**: compose a pallet stack-by-stack, load known failure presets,
  and run the same solver against the manual load.
- **Scanner Feed**: generate scanner-like payloads through the adapter path.
- **Safety Envelope**: analyze random, scanner, manual, or Mission Control
  pallets and calculate deployable speed/accel limits.
- **Live Solver**: run one configurable conveyor profile and inspect the trace.
- **API Batch**: run multiple randomized pallets and return useful pallet
  profile descriptions, not just random technical IDs.
- **Friction Lab**: inspect temperature/surface-pair friction curves.
- **SKU Catalog**: review the seed cold-storage SKU library.
- **System**: health, runtime, and local test runner.

## Known demo failures

These are built to be fast, visible trust checks:

| Scenario | What it shows |
|---|---|
| Stable dairy slab | Low, wrapped dairy load that should pass |
| Tall unwrapped tower | Top-item slide from a narrow unwrapped stack |
| Frozen pallet jerk-start | Frozen load that fails under an aggressive start |
| Top-heavy surprise | Heavy vertical stack without wrap |
| Asymmetric load | Offset center of mass with a tall heavy side |

## Quickstart

The project is managed with `uv`.

```bash
# install/sync dependencies
python -m uv sync

# run tests
python -m uv run pytest

# start API and UI together
python -m uv run python scripts/start_dev.py
```

Then open:

- Streamlit UI: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/healthz`

You can also run the services separately:

```bash
python -m uv run uvicorn pallet_safety.service.api:app --reload --port 8000
python -m uv run streamlit run pallet_safety/viz/streamlit_app.py --server.port 8501
```

## API examples

Create a random pallet:

```bash
curl -X POST http://localhost:8000/pallet/random \
  -H "content-type: application/json" \
  -d '{"seed": 42, "anomaly_rate": 0.1, "min_layers": 2, "max_layers": 5}'
```

Analyze one pallet:

```bash
curl -X POST http://localhost:8000/safety/analyze \
  -H "content-type: application/json" \
  -d @pallet.json
```

Run a single conveyor profile:

```bash
curl -X POST http://localhost:8000/solve \
  -H "content-type: application/json" \
  -d '{
    "pallet": { "...": "PalletConfig JSON" },
    "profile": {
      "target_speed_mps": 1.2,
      "accel_mps2": 4.0,
      "duration_s": 2.0
    }
  }'
```

Batch analyze pallets:

```bash
curl -X POST http://localhost:8000/safety/batch \
  -H "content-type: application/json" \
  -d @pallets.json
```

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Service health |
| `GET` | `/catalog/skus` | List SKU catalog |
| `POST` | `/raw/random` | Generate mock scanner input |
| `POST` | `/pallet/from-raw` | Convert `RawInputs` to `PalletConfig` |
| `POST` | `/pallet/random` | Random pallet convenience endpoint |
| `POST` | `/pallet/validate` | Validate a pallet payload |
| `POST` | `/mjcf/build` | Build MuJoCo XML |
| `POST` | `/solve` | Run one conveyor profile |
| `POST` | `/safety/analyze` | Compute safe operating envelope |
| `POST` | `/safety/batch` | Analyze many pallets with shared cache |
| `GET` | `/friction` | Single friction lookup |
| `GET` | `/friction/curve` | Temperature/friction curve |
| `GET` | `/friction/pairs` | Available surface pairs |

## Project layout

```text
pallet_safety/
  models.py           # Pydantic domain types
  catalog.py          # CSV-backed SKU catalog
  configurator.py     # RawInputs / StackSpec -> PalletConfig
  friction.py         # Temperature and surface-pair friction model
  mjcf_builder.py     # PalletConfig -> MuJoCo XML
  solver.py           # Conveyor simulation
  failures.py         # Failure detectors
  threshold.py        # Safe speed/accel analyzer
  inputs/
    base.py           # Adapter contracts
    mock_random.py    # Deterministic mock adapter
  service/
    api.py            # FastAPI HTTP service
  viz/
    pallet_3d.py      # Plotly 3D pallet/replay rendering
    streamlit_app.py  # Operator console

data/
  sku_catalog.csv
  friction_table.json

tests/
  ...

scripts/
  start_dev.py
```

## Competitive landscape

This project is adjacent to palletization and warehouse simulation products, but
it is not trying to replace them.

| Product/category | What it is good at | How Pallet Safety Command differs |
|---|---|---|
| [Esko Cape Pack](https://www.esko.com/en/products/cape-pack) | Cloud palletization optimization: product sizing, case sizing, pallet patterns, truck loading, and carbon impact. | Cape Pack is design/logistics optimization. This project is runtime physics inference for the actual pallet entering a conveyor zone. |
| [TOPS Pro / MaxLoad](https://topseng.com/pallet-layout-configuration-software/) | Pallet layout, skid configuration, packaging design, load accessories, truck/container loading, and 3D diagrams. | TOPS optimizes how to pack and ship. This project predicts how a packed pallet behaves dynamically under speed and acceleration. |
| [StackBuilder](https://www.treedim.com/stackbuilder/en/features) | Free packaging, palletizing, packing, and truck-loading design software. | StackBuilder is a planning calculator. This project is API-first and built for automated per-pallet safety decisions. |
| [Autodesk FlexSim warehouse simulation](https://www.autodesk.com/solutions/design-manufacturing/warehouse-simulation) | Discrete-event warehouse simulation with 3D models for docks, conveyors, AGVs, personnel, and throughput experiments. | FlexSim models facility flow. This project drills into contact physics and failure thresholds for one pallet at a time. |
| [Siemens Tecnomatix Plant Simulation](https://www.siemens.com/en-us/products/tecnomatix/plant-simulation-software/) | Production/logistics digital validation, material flow, resource utilization, 3D plant models, and experiment execution. | Tecnomatix is a broad planning and validation suite. This project is a focused edge inference service for cold-storage conveyor safety. |
| WMS/WES/PLC control systems | Inventory, routing, execution, and machine control. | These systems decide where pallets go. Pallet Safety Command can provide the per-pallet speed/accel limits they should respect. |

## Differentiation

- **Per-pallet runtime decisioning**: each pallet receives its own safety result.
- **Cold-chain friction modeling**: frozen, refrigerated, thawed, and
  transitioning conditions affect the physics.
- **Failure-mode explanation**: results identify whether the governing risk is
  slip, top-item slide, load shift, tip-over, or no failure.
- **Trust-first UI**: users can intentionally topple bad scenarios and compare
  the replay with the signal trace.
- **API-first batch path**: the same payload can be simulated once, thresholded,
  or run through batch analysis.
- **Adapter-friendly design**: scanner, WMS, manual UI, and mock random inputs
  all converge on the same `PalletConfig` contract.

## Deployment

The live demo runs on a home Ubuntu server fronted by a Cloudflare Tunnel. The
service is published as a Docker image to GHCR on every push to `main`, and
Watchtower on the server pulls and restarts within 5 minutes. See
[`infra/README.md`](infra/README.md) for the full setup.

If you want to host your own copy: clone, `docker compose up -d` in `infra/`,
done. Cloudflare Containers (GA April 2026) is also a viable target for the
same Dockerfile if you'd rather not self-host — it skips the cold-start
problem at the cost of $5–20/mo and a known WebSocket-timeout bug on long
sessions.

## Validation status

Latest local validation:

- `pytest -q --tb=short`: 168 passed
- Streamlit root: HTTP 200
- API `/healthz`: HTTP 200
- UAT flows checked:
  - stable dairy slab passes
  - tall unwrapped tower fails by top-item slide
  - frozen pallet jerk-start fails under aggressive accel
  - top-heavy surprise fails by top-item slide
  - asymmetric load exposes offset-driven failure behavior
  - batch output includes pallet profile descriptions

## Limitations

- This is not a certified safety controller.
- Real deployment needs calibration against physical tip/slip tests.
- Friction tables are seed data, not a measured facility-specific model.
- Vision, WMS, Modbus, and OPC-UA adapters are architectural targets, not
  production integrations yet.
- Streamlit is the current demo/operator console; a hardened production UI would
  likely move to a React/Three.js front end.

## Roadmap

- Add real scanner/WMS adapters.
- Add facility-specific friction calibration workflow.
- Add lateral acceleration and transfer/curve scenarios.
- Add PLC integration stubs for Modbus TCP and OPC-UA.
- Add persistent run history and comparison reports.
- Add container packaging for demo deployment.
- Promote the trust UI into a production-ready frontend if the demo validates.

