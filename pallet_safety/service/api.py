"""HTTP API for the pluggable input engine.

Adapters (UI, batch script, future real machine vision) call these endpoints
to drive the pipeline. The API itself owns no adapter — adapters are clients.

Endpoints:
    GET    /healthz                       liveness
    GET    /catalog/skus                  list catalog SKUs
    GET    /catalog/skus/{sku}            one SKU detail
    POST   /raw/random                    invoke MockRandomAdapter, return RawInputs
    POST   /pallet/from-raw               RawInputs → PalletConfig
    POST   /pallet/random                 convenience: random → PalletConfig in one call
    POST   /pallet/validate               echo a PalletConfig if valid (422 otherwise)
    POST   /mjcf/build                    PalletConfig → MJCF XML
    GET    /friction                      look up mu at one (T, seconds-since, pair)
    GET    /friction/curve                full mu(T) curve for a surface pair
    GET    /friction/pairs                list available surface pairs
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..catalog import all_skus, get as get_template
from ..configurator import Configurator
from ..failures import FailureThresholds, first_failure, tip_angle_deg
from ..friction import (
    DEFAULT_PAIR,
    available_surface_pairs,
    friction_coefficient,
    steady_state_mu,
)
from ..inputs import MockRandomAdapter
from ..inputs.base import RawInputs
from ..mjcf_builder import build_mjcf
from ..models import EnvCondition, FailureMode, FragilityClass, PalletConfig, SafetyResult
from ..solver import ConveyorProfile, simulate
from ..threshold import AnalysisResult, default_analyzer

app = FastAPI(
    title="Pallet Safety Service",
    description="Pluggable input engine + physics inference for cold-storage conveyor systems.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_configurator = Configurator()


# ---- response models ----

class HealthResponse(BaseModel):
    status: str
    version: str


class SkuInfo(BaseModel):
    sku: str
    name: str
    weight_kg: float
    dims_m: tuple[float, float, float]
    fragility: FragilityClass
    category: str
    default_env: EnvCondition


class FrictionPoint(BaseModel):
    temp_c: float
    mu_static: float
    mu_dynamic: float


class FrictionCurve(BaseModel):
    surface_pair: tuple[str, str]
    seconds_since_temp_change: float
    points: list[FrictionPoint]


class MjcfResponse(BaseModel):
    pallet_id: str
    mjcf_xml: str
    bytes: int


class RandomRequest(BaseModel):
    seed: int | None = None
    anomaly_rate: float = Field(default=0.10, ge=0, le=1)
    min_layers: int = Field(default=1, ge=0, le=10)
    max_layers: int = Field(default=5, ge=0, le=20)
    min_items_per_layer: int = Field(default=1, ge=0, le=10)
    max_items_per_layer: int = Field(default=4, ge=0, le=20)


# ---- routes ----

@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz():
    from .. import __version__
    return HealthResponse(status="ok", version=__version__)


@app.get("/catalog/skus", response_model=list[SkuInfo], tags=["catalog"])
def list_skus():
    return [_template_to_info(get_template(s)) for s in all_skus()]


@app.get("/catalog/skus/{sku}", response_model=SkuInfo, tags=["catalog"])
def get_sku(sku: str):
    try:
        return _template_to_info(get_template(sku))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/raw/random", response_model=RawInputs, tags=["inputs"])
def random_raw(req: RandomRequest = Body(default_factory=RandomRequest)):
    """Invoke the MockRandomAdapter with the supplied parameters and return RawInputs."""
    if req.max_layers < req.min_layers or req.max_items_per_layer < req.min_items_per_layer:
        raise HTTPException(400, "max bounds must be >= min bounds")
    adapter = MockRandomAdapter(
        seed=req.seed,
        anomaly_rate=req.anomaly_rate,
        min_layers=req.min_layers,
        max_layers=req.max_layers,
        min_items_per_layer=req.min_items_per_layer,
        max_items_per_layer=req.max_items_per_layer,
    )
    return adapter.read()


@app.post("/pallet/from-raw", response_model=PalletConfig, tags=["pallet"])
def pallet_from_raw(raw: RawInputs):
    try:
        return _configurator.build(raw)
    except KeyError as e:
        raise HTTPException(404, f"unknown SKU: {e}")


@app.post("/pallet/random", response_model=PalletConfig, tags=["pallet"])
def pallet_random(req: RandomRequest = Body(default_factory=RandomRequest)):
    raw = random_raw(req)
    return pallet_from_raw(raw)


@app.post("/pallet/validate", response_model=PalletConfig, tags=["pallet"])
def pallet_validate(cfg: PalletConfig):
    """Pydantic does the work — if it deserialized into PalletConfig, it's valid."""
    return cfg


@app.post("/mjcf/build", response_model=MjcfResponse, tags=["mjcf"])
def mjcf_build(
    cfg: PalletConfig,
    surface_pair: Annotated[str, Query(description='e.g. "wood_pallet/rubber_belt"')] = "/".join(DEFAULT_PAIR),
):
    pair_tuple = _parse_pair(surface_pair)
    xml = build_mjcf(cfg, surface_pair=pair_tuple)
    return MjcfResponse(pallet_id=cfg.pallet_id, mjcf_xml=xml, bytes=len(xml.encode("utf-8")))


@app.get("/friction", response_model=FrictionPoint, tags=["friction"])
def friction(
    temp_c: float = Query(..., ge=-40, le=40),
    seconds_since_temp_change: float = Query(3600.0, ge=0),
    pair: str = Query("/".join(DEFAULT_PAIR)),
):
    pair_tuple = _parse_pair(pair)
    mu_s, mu_d = friction_coefficient(temp_c, seconds_since_temp_change, pair_tuple)
    return FrictionPoint(temp_c=temp_c, mu_static=mu_s, mu_dynamic=mu_d)


@app.get("/friction/curve", response_model=FrictionCurve, tags=["friction"])
def friction_curve(
    pair: str = Query("/".join(DEFAULT_PAIR)),
    seconds_since_temp_change: float = Query(3600.0, ge=0),
    t_min: float = -30.0,
    t_max: float = 25.0,
    n: int = Query(56, ge=2, le=500),
):
    pair_tuple = _parse_pair(pair)
    step = (t_max - t_min) / (n - 1)
    points: list[FrictionPoint] = []
    for i in range(n):
        t = t_min + i * step
        mu_s, mu_d = friction_coefficient(t, seconds_since_temp_change, pair_tuple)
        points.append(FrictionPoint(temp_c=t, mu_static=mu_s, mu_dynamic=mu_d))
    return FrictionCurve(
        surface_pair=pair_tuple,
        seconds_since_temp_change=seconds_since_temp_change,
        points=points,
    )


@app.get("/friction/pairs", response_model=list[str], tags=["friction"])
def friction_pairs():
    return ["/".join(p) for p in available_surface_pairs()]


# ---- /solve : Phase C ----

class SolveRequest(BaseModel):
    pallet: PalletConfig
    profile: ConveyorProfile = Field(default_factory=ConveyorProfile)
    surface_pair: str = Field(default="/".join(DEFAULT_PAIR))
    thresholds: FailureThresholds = Field(default_factory=FailureThresholds)
    output_hz: float = Field(default=50.0, ge=1, le=1000,
                              description="Trace downsample rate.")


class FailureSummary(BaseModel):
    mode: FailureMode
    time_s: float | None
    max_tip_angle_deg: float


class TraceSeries(BaseModel):
    times_s: list[float]
    conveyor_vel_mps: list[float]
    pallet_vel_x_mps: list[float]
    pallet_pos_x_m: list[float]
    tip_angle_deg: list[float]


class SolveResponse(BaseModel):
    pallet_id: str
    failure: FailureSummary
    trace: TraceSeries
    runtime_ms: float
    n_steps_simulated: int


# ---- /safety : Phase D ----

class SweepPoint(BaseModel):
    axis: str  # "speed" or "accel"
    value: float
    safe: bool
    failure_mode: FailureMode


class SafetyAnalyzeResponse(BaseModel):
    result: SafetyResult
    sims_run: int
    cache_hits: int
    sweep_points: list[SweepPoint]


@app.post("/safety/analyze", response_model=SafetyAnalyzeResponse, tags=["safety"])
def safety_analyze(cfg: PalletConfig):
    """Compute the maximum safe operating envelope for a pallet.

    Returns `max_speed_mps`, `max_accel_mps2`, dominant failure mode, and the
    binary-search sweep points so the caller can visualize the search.
    Result is cached by config fingerprint — repeat requests are ~1ms.
    """
    analysis: AnalysisResult = default_analyzer().analyze(cfg)
    return _analysis_to_response(analysis)


@app.post("/safety/batch", response_model=list[SafetyAnalyzeResponse], tags=["safety"])
def safety_batch(cfgs: list[PalletConfig]):
    """Analyze a list of pallets. Cache is shared — duplicates are free."""
    analyzer = default_analyzer()
    return [_analysis_to_response(analyzer.analyze(c)) for c in cfgs]


def _analysis_to_response(analysis: AnalysisResult) -> SafetyAnalyzeResponse:
    return SafetyAnalyzeResponse(
        result=analysis.result,
        sims_run=analysis.sims_run,
        cache_hits=analysis.cache_hits,
        sweep_points=[SweepPoint(axis=ax, value=v, safe=s, failure_mode=m)
                       for ax, v, s, m in analysis.sweep_points],
    )


@app.post("/solve", response_model=SolveResponse, tags=["solver"])
def solve(req: SolveRequest):
    pair = _parse_pair(req.surface_pair)
    trace = simulate(req.pallet, req.profile, surface_pair=pair)
    mode, t = first_failure(trace, req.thresholds)
    angles = tip_angle_deg(trace)
    ds = trace.downsample(hz=req.output_hz)
    ds_angles = tip_angle_deg(ds)
    return SolveResponse(
        pallet_id=req.pallet.pallet_id,
        failure=FailureSummary(
            mode=mode, time_s=t,
            max_tip_angle_deg=float(angles.max() if len(angles) else 0.0),
        ),
        trace=TraceSeries(
            times_s=ds.times.tolist(),
            conveyor_vel_mps=ds.conveyor_vel.tolist(),
            pallet_vel_x_mps=ds.pallet_lin_vel[:, 0].tolist(),
            pallet_pos_x_m=ds.pallet_pos[:, 0].tolist(),
            tip_angle_deg=ds_angles.tolist(),
        ),
        runtime_ms=trace.runtime_s * 1000.0,
        n_steps_simulated=trace.n_steps,
    )


# ---- helpers ----

def _template_to_info(t) -> SkuInfo:
    return SkuInfo(
        sku=t.sku, name=t.name, weight_kg=t.weight_kg,
        dims_m=t.dims_m, fragility=t.fragility, category=t.category,
        default_env=t.default_env,
    )


def _parse_pair(s: str) -> tuple[str, str]:
    parts = s.split("/")
    if len(parts) != 2:
        raise HTTPException(400, f'surface_pair must be "X/Y", got {s!r}')
    return parts[0], parts[1]
