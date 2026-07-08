"""Streamlit demo console for the pallet safety simulator.

Run:
    streamlit run pallet_safety/viz/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Allow importing from the package when run as a script.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pallet_safety.catalog import all_skus, get as get_template  # noqa: E402
from pallet_safety.configurator import (  # noqa: E402
    Configurator,
    StackSpec,
    build_from_stacks,
    compute_grid_shape,
)
from pallet_safety.failures import first_failure, tip_angle_deg  # noqa: E402
from pallet_safety.friction import (  # noqa: E402
    DEFAULT_PAIR,
    available_surface_pairs,
    friction_coefficient,
)
from pallet_safety.inputs import MockRandomAdapter  # noqa: E402
from pallet_safety.mjcf_builder import build_mjcf  # noqa: E402
from pallet_safety.models import EnvCondition, FailureMode, PalletConfig, WrapType  # noqa: E402
from pallet_safety.scenarios import all_scenarios, get_scenario_by_name  # noqa: E402
from pallet_safety.solver import ConveyorProfile, simulate  # noqa: E402
from pallet_safety.threshold import AnalysisResult, default_analyzer  # noqa: E402
from pallet_safety.viz.pallet_3d import animate_trace, render as render_pallet  # noqa: E402

API_URL = os.getenv("PALLET_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Pallet Safety Command",
    page_icon="PS",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- visual system ----

def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #f5f1e7;
          --muted: rgba(245, 241, 231, 0.68);
          --panel: rgba(31, 32, 27, 0.94);
          --panel-2: rgba(42, 42, 35, 0.92);
          --line: rgba(245, 241, 231, 0.12);
          --amber: #eaae46;
          --green: #68d391;
          --cyan: #4aa9c8;
          --red: #f25852;
          --steel: #88929a;
        }
        .stApp {
          background:
            linear-gradient(135deg, rgba(234,174,70,0.12), transparent 28%),
            linear-gradient(315deg, rgba(74,169,200,0.10), transparent 35%),
            #11120f;
          color: var(--ink);
        }
        [data-testid="stHeader"] {
          display: none;
        }
        [data-testid="stDecoration"] {
          display: none;
        }
        [data-testid="stToolbar"] {
          display: none;
        }
        [data-testid="stStatusWidget"] {
          display: none;
        }
        [data-testid="stSidebar"] {
          background: #171814;
          border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
          color: var(--ink);
        }
        .block-container {
          max-width: 1500px;
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        h1 {
          font-size: clamp(2rem, 4vw, 4.4rem);
          line-height: 0.95;
          margin-bottom: 0.25rem;
        }
        h2 {
          font-size: 1.5rem;
          margin-top: 0.25rem;
        }
        h3 {
          font-size: 1.05rem;
        }
        p, label, span, div {
          letter-spacing: 0;
        }
        [data-testid="stCaptionContainer"], .stMarkdown p {
          color: var(--muted);
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="InputInstructions"],
        .stSlider label,
        .stSlider [data-testid="stTickBar"] *,
        .stSelectbox label,
        .stRadio label,
        .stMultiSelect label,
        .stNumberInput label,
        .stTextInput label {
          color: var(--ink) !important;
        }
        .stSlider [data-baseweb="slider"] div,
        .stSlider [data-baseweb="slider"] span {
          color: var(--ink) !important;
        }
        div[data-baseweb="tooltip"],
        div[data-baseweb="popover"] {
          color: #11120f;
        }
        input, textarea, [role="combobox"], [role="listbox"], [role="option"] {
          color: var(--ink) !important;
        }
        [data-baseweb="select"] svg {
          color: var(--ink);
        }
        [data-testid="stMetric"] {
          background: rgba(31, 32, 27, 0.82);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.9rem 1rem;
        }
        [data-testid="stMetric"] label {
          color: var(--muted);
        }
        [data-testid="stMetricValue"] {
          color: var(--ink);
        }
        .stButton button, .stDownloadButton button {
          border-radius: 8px;
          border: 1px solid rgba(234, 174, 70, 0.45);
          background: rgba(234, 174, 70, 0.12);
          color: var(--ink);
          font-weight: 700;
        }
        .stButton button:hover, .stDownloadButton button:hover {
          border-color: var(--amber);
          background: rgba(234, 174, 70, 0.22);
          color: var(--ink);
        }
        .stButton button[kind="primary"] {
          background: linear-gradient(135deg, #eaae46, #f25852);
          border-color: transparent;
          color: #11120f;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
          background: rgba(245, 241, 231, 0.06);
          border-color: var(--line);
          border-radius: 8px;
          color: var(--ink);
        }
        div[data-baseweb="popover"] {
          background: #202119 !important;
          border: 1px solid rgba(245, 241, 231, 0.18) !important;
          border-radius: 8px !important;
          box-shadow: 0 22px 60px rgba(0, 0, 0, 0.45) !important;
        }
        div[data-baseweb="popover"] *,
        div[data-baseweb="menu"] *,
        [role="listbox"] *,
        [role="option"] {
          color: var(--ink) !important;
        }
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"],
        [role="listbox"],
        [role="option"] {
          background: #202119 !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] [role="option"]:hover,
        [aria-selected="true"] {
          background: rgba(234, 174, 70, 0.18) !important;
          color: #fff8e7 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.35rem;
          border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 8px 8px 0 0;
          color: var(--muted);
          background: rgba(245, 241, 231, 0.04);
          padding: 0.65rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
          color: var(--ink);
          background: rgba(234, 174, 70, 0.14);
          border-bottom: 2px solid var(--amber);
        }
        .hero-shell {
          border: 1px solid var(--line);
          background:
            linear-gradient(135deg, rgba(234,174,70,0.18), transparent 32%),
            linear-gradient(315deg, rgba(104,211,145,0.10), transparent 38%),
            rgba(24, 25, 21, 0.86);
          border-radius: 8px;
          padding: 1.05rem 1.15rem;
          margin: 0.35rem 0 1rem;
        }
        .hero-row {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 1rem;
          flex-wrap: wrap;
        }
        .eyebrow {
          color: var(--amber);
          font-size: 0.78rem;
          text-transform: uppercase;
          font-weight: 800;
          margin: 0 0 0.4rem 0;
        }
        .lede {
          color: var(--muted);
          max-width: 760px;
          margin: 0.35rem 0 0 0;
          font-size: 1.02rem;
        }
        .pill {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 0.34rem 0.62rem;
          color: var(--ink);
          background: rgba(245,241,231,0.06);
          font-size: 0.78rem;
          font-weight: 800;
          white-space: nowrap;
        }
        .pill.good { border-color: rgba(104,211,145,0.42); color: var(--green); }
        .pill.warn { border-color: rgba(234,174,70,0.45); color: var(--amber); }
        .pill.bad { border-color: rgba(242,88,82,0.45); color: var(--red); }
        .kpi-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 0.75rem;
          margin: 0.75rem 0 1rem;
        }
        .metric-card {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(31, 32, 27, 0.90);
          padding: 0.85rem 0.95rem;
          min-height: 94px;
        }
        .metric-card .label {
          color: var(--muted);
          text-transform: uppercase;
          font-size: 0.72rem;
          font-weight: 800;
          margin-bottom: 0.25rem;
        }
        .metric-card .value {
          color: var(--ink);
          font-size: 1.75rem;
          line-height: 1.05;
          font-weight: 850;
        }
        .metric-card .caption {
          color: var(--muted);
          margin-top: 0.25rem;
          font-size: 0.78rem;
        }
        .metric-card.good .value { color: var(--green); }
        .metric-card.warn .value { color: var(--amber); }
        .metric-card.bad .value { color: var(--red); }
        .panel {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: rgba(31, 32, 27, 0.84);
          padding: 1rem;
          margin-bottom: 1rem;
        }
        .panel-title {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 850;
          margin-bottom: 0.5rem;
        }
        .story-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 0.65rem;
          margin-top: 0.75rem;
        }
        .story-item {
          border: 1px solid rgba(245, 241, 231, 0.10);
          border-radius: 8px;
          background: rgba(245, 241, 231, 0.045);
          padding: 0.65rem;
          min-width: 0;
        }
        .story-item .label {
          color: var(--muted);
          font-size: 0.72rem;
          font-weight: 800;
          text-transform: uppercase;
          margin-bottom: 0.25rem;
        }
        .story-item .value {
          color: var(--ink);
          font-size: 0.9rem;
          line-height: 1.35;
        }
        .story-text {
          color: var(--muted);
          font-size: 0.88rem;
          line-height: 1.45;
        }
        .sidebar-brand {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 0.85rem;
          margin-bottom: 0.8rem;
          background: rgba(234,174,70,0.08);
        }
        .sidebar-brand .name {
          font-weight: 900;
          font-size: 1.05rem;
          color: var(--ink);
        }
        .sidebar-brand .sub {
          color: var(--muted);
          font-size: 0.76rem;
          margin-top: 0.2rem;
        }
        .code-note {
          color: var(--muted);
          font-size: 0.82rem;
          margin-top: -0.2rem;
          margin-bottom: 0.5rem;
        }
        @media (max-width: 900px) {
          .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .story-grid { grid-template-columns: 1fr; }
          h1 { font-size: 2.3rem; }
        }
        @media (max-width: 560px) {
          .kpi-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


# ---- API helpers, with in-process fallback ----

@st.cache_data(ttl=3, show_spinner=False)
def _api_alive() -> bool:
    try:
        r = requests.get(f"{API_URL}/healthz", timeout=0.5)
        return r.ok
    except Exception:
        return False


def random_pallet(
    seed: int | None,
    anomaly_rate: float,
    min_l: int,
    max_l: int,
    min_ipl: int,
    max_ipl: int,
) -> PalletConfig:
    if _api_alive():
        body = {
            "seed": seed,
            "anomaly_rate": anomaly_rate,
            "min_layers": min_l,
            "max_layers": max_l,
            "min_items_per_layer": min_ipl,
            "max_items_per_layer": max_ipl,
        }
        r = requests.post(f"{API_URL}/pallet/random", json=body, timeout=10)
        r.raise_for_status()
        return PalletConfig.model_validate(r.json())
    adapter = MockRandomAdapter(
        seed=seed,
        anomaly_rate=anomaly_rate,
        min_layers=min_l,
        max_layers=max_l,
        min_items_per_layer=min_ipl,
        max_items_per_layer=max_ipl,
    )
    return Configurator().build(adapter.read())


def mjcf_for(cfg: PalletConfig, surface_pair: tuple[str, str]) -> str:
    if _api_alive():
        r = requests.post(
            f"{API_URL}/mjcf/build",
            params={"surface_pair": "/".join(surface_pair)},
            json=cfg.model_dump(mode="json"),
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["mjcf_xml"]
    return build_mjcf(cfg, surface_pair=surface_pair)


# ---- shared data and helpers ----

BASE_DEFAULTS = {
    "EUR": {"dims": (1.2, 0.8, 0.15), "mass": 25.0},
    "GMA": {"dims": (1.22, 1.02, 0.15), "mass": 30.0},
    "CHEP": {"dims": (1.2, 0.8, 0.15), "mass": 18.0},
}

# Canonical scenario data lives in pallet_safety.scenarios (shared with the
# API's GET /scenarios and the public web demo). This view adapts it to the
# dict shape the console's widgets read.
DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    s.name: {
        "slug": s.slug,
        "tag": s.tag,
        "failure_type": s.expected_failure,
        "description": s.description,
        "profile": s.suggested_profile.model_dump(),
    }
    for s in all_scenarios()
}

CRASH_PRESETS = {
    "Tall unwrapped tower": DEMO_SCENARIOS["Tall unwrapped tower"],
    "Frozen pallet jerk-start": DEMO_SCENARIOS["Frozen pallet jerk-start"],
    "Top-heavy stack": DEMO_SCENARIOS["Top-heavy surprise"],
    "Asymmetric load": DEMO_SCENARIOS["Asymmetric load"],
}

TOPPLE_LAB = {
    "Tip": "Tall unwrapped tower",
    "Slip": "Frozen pallet jerk-start",
    "Shift": "Top-heavy surprise",
    "Offset": "Asymmetric load",
}


def _fmt_sku(sku: str) -> str:
    try:
        t = get_template(sku)
        return f"{sku} - {t.name}"
    except Exception:
        return sku


def _build_demo_scenario(name: str) -> tuple[PalletConfig, ConveyorProfile]:
    s = get_scenario_by_name(name)
    return s.pallet.model_copy(deep=True), s.suggested_profile


def _pallet_hash(cfg: PalletConfig) -> str:
    from pallet_safety.threshold import _config_fingerprint

    return _config_fingerprint(cfg)


def _failure_tone(mode: FailureMode | str) -> str:
    value = mode.value if isinstance(mode, FailureMode) else str(mode)
    return "good" if value == FailureMode.NO_FAILURE.value else "bad"


def _scenario_failure_type(name: str | None) -> str:
    if not name:
        return "unclassified until simulation"
    return str(DEMO_SCENARIOS.get(name, {}).get("failure_type", "unclassified until simulation"))


def _scenario_description(name: str | None) -> str:
    if not name:
        return ""
    return str(DEMO_SCENARIOS.get(name, {}).get("description", ""))


def _dominant_profile_piece(cfg: PalletConfig) -> str:
    if not cfg.items:
        return "empty pallet"
    by_sku: dict[str, tuple[int, float]] = {}
    for item in cfg.items:
        count, mass = by_sku.get(item.sku, (0, 0.0))
        by_sku[item.sku] = (count + 1, mass + item.weight_kg)
    sku, (count, mass) = max(by_sku.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    try:
        tpl = get_template(sku)
        return f"{tpl.name} ({count} units, {mass:.0f} kg product mass)"
    except Exception:
        return f"{sku} ({count} units, {mass:.0f} kg product mass)"


def _pallet_profile_label(cfg: PalletConfig) -> str:
    return (
        f"{_dominant_profile_piece(cfg)}; {len(cfg.items)} total items, "
        f"{cfg.total_mass_kg:.0f} kg gross, {cfg.stack_height_m:.2f} m high, "
        f"{cfg.env.value} / {cfg.wrap.value}"
    )


def _motion_profile_label(profile: ConveyorProfile | None) -> str:
    if profile is None:
        return "No conveyor hit selected yet."
    return (
        f"{profile.target_speed_mps:.2f} m/s target, "
        f"{profile.accel_mps2:.2f} m/s^2 accel, {profile.duration_s:.1f}s run"
    )


def _source_explanation(source: str) -> str:
    notes = {
        "Mission control": (
            "Mission Control keeps one PalletConfig plus one ConveyorProfile in session state. "
            "Running Safe limits sends that exact load to the threshold analyzer."
        ),
        "Manual build": (
            "Build Pallet stores stack specs. Downstream pages rebuild those specs through the same "
            "configurator, so grid placement, SKU dimensions, temperature, and wrap move together."
        ),
        "Scanner feed": (
            "Scanner Feed mocks the adapter contract: RawInputs become a PalletConfig, then the same "
            "object can be simulated, exported, or batch-analyzed."
        ),
        "Random pallet": (
            "Random pallets come from the mock adapter so batch tests exercise the same public payload "
            "shape a camera, barcode, or WMS integration would produce."
        ),
        "API batch": (
            "The batch runner creates many PalletConfigs, analyzes each one, and returns deployable "
            "speed/accel limits with a human-readable pallet profile beside the technical ID."
        ),
    }
    return notes.get(source, "This view carries the current PalletConfig into the next solver step.")


def _result_failure_label(result_key: str | None) -> str | None:
    if not result_key:
        return None
    result = st.session_state.get(f"{result_key}_result")
    if not result:
        return None
    mode = result["mode"]
    if isinstance(mode, FailureMode):
        return mode.value
    return str(mode)


def render_load_story(
    cfg: PalletConfig,
    *,
    source: str,
    scenario_name: str | None = None,
    profile: ConveyorProfile | None = None,
    result_key: str | None = None,
    expected_failure: str | None = None,
) -> None:
    actual_failure = _result_failure_label(result_key)
    failure = actual_failure or expected_failure or _scenario_failure_type(scenario_name)
    failure_label = "Actual failure type" if actual_failure else "Expected failure type"
    description = _scenario_description(scenario_name) or _source_explanation(source)
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-title">Load story</div>
          <div class="story-text">{escape(description)}</div>
          <div class="story-grid">
            <div class="story-item">
              <div class="label">Pallet profile</div>
              <div class="value">{escape(_pallet_profile_label(cfg))}</div>
            </div>
            <div class="story-item">
              <div class="label">Motion profile</div>
              <div class="value">{escape(_motion_profile_label(profile))}</div>
            </div>
            <div class="story-item">
              <div class="label">{escape(failure_label)}</div>
              <div class="value">{escape(failure)}</div>
            </div>
            <div class="story-item">
              <div class="label">How it moves</div>
              <div class="value">{escape(_source_explanation(source))}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pallet_score(cfg: PalletConfig) -> tuple[int, str, str]:
    cx, cy, _ = cfg.composite_com_m
    half_l = cfg.base_dims_m[0] / 2.0
    half_w = cfg.base_dims_m[1] / 2.0
    com_norm = math.sqrt((cx / half_l) ** 2 + (cy / half_w) ** 2)
    height_ratio = cfg.stack_height_m / max(min(cfg.base_dims_m[0], cfg.base_dims_m[1]), 0.01)
    overhang_mm = max(0.0, cfg.overhang_m * 1000.0)

    risk = 0.0
    risk += min(26.0, overhang_mm * 0.45)
    risk += min(30.0, max(0.0, height_ratio - 1.1) * 23.0)
    risk += min(24.0, com_norm * 28.0)
    risk += {"none": 16.0, "banded": 5.0, "shrink": 3.0, "stretch": 0.0}[cfg.wrap.value]
    if -2.0 <= cfg.body_temp_c <= 8.0 and cfg.seconds_since_temp_change < 1800:
        risk += 10.0
    score = int(max(0.0, min(100.0, 100.0 - risk)))
    if score >= 82:
        return score, "A", "good"
    if score >= 62:
        return score, "B", "warn"
    if score >= 42:
        return score, "C", "warn"
    return score, "D", "bad"


def _metric_html(label: str, value: str, caption: str = "", tone: str = "") -> str:
    return (
        f'<div class="metric-card {escape(tone)}">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'<div class="caption">{escape(caption)}</div>'
        "</div>"
    )


def render_kpi_strip(cfg: PalletConfig) -> None:
    score, grade, tone = _pallet_score(cfg)
    cx, cy, cz = cfg.composite_com_m
    cards = [
        _metric_html("Risk grade", f"{grade} / {score}", f"CoM z {cz:.2f} m", tone),
        _metric_html("Mass", f"{cfg.total_mass_kg:.0f} kg", f"{len(cfg.items)} items"),
        _metric_html("Stack height", f"{cfg.stack_height_m:.2f} m", f"wrap {cfg.wrap.value}"),
        _metric_html(
            "Offset",
            f"{math.sqrt(cx * cx + cy * cy) * 1000:.0f} mm",
            f"overhang {max(0, cfg.overhang_m) * 1000:.0f} mm",
        ),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _header(title: str, eyebrow: str, lede: str = "") -> None:
    api_tone = "good" if _api_alive() else "warn"
    api_text = "API connected" if _api_alive() else "local engine"
    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-row">
            <div>
              <p class="eyebrow">{escape(eyebrow)}</p>
              <h1>{escape(title)}</h1>
              <p class="lede">{escape(lede)}</p>
            </div>
            <div class="pill {api_tone}">{escape(api_text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pallet_dataframe(cfg: PalletConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sku": i.sku,
                "weight_kg": i.weight_kg,
                "L_m": i.dims_m[0],
                "W_m": i.dims_m[1],
                "H_m": i.dims_m[2],
                "x": i.position[0],
                "y": i.position[1],
                "z": i.position[2],
                "fragility": i.fragility.value,
                "rot_deg": i.orientation_deg,
            }
            for i in cfg.items
        ]
    )


def _trace_figure(trace) -> go.Figure:
    ds = trace.downsample(hz=80)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ds.times.tolist(),
            y=ds.conveyor_vel.tolist(),
            name="belt velocity",
            line=dict(color="#4aa9c8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds.times.tolist(),
            y=ds.pallet_lin_vel[:, 0].tolist(),
            name="pallet velocity",
            line=dict(color="#68d391", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ds.times.tolist(),
            y=tip_angle_deg(ds).tolist(),
            name="tip angle",
            yaxis="y2",
            line=dict(color="#f25852", width=2, dash="dot"),
        )
    )
    fig.update_layout(
        xaxis=dict(title="time (s)", gridcolor="rgba(255,255,255,0.10)"),
        yaxis=dict(title="velocity (m/s)", gridcolor="rgba(255,255,255,0.10)"),
        yaxis2=dict(title="tip angle (deg)", overlaying="y", side="right"),
        height=330,
        margin=dict(l=30, r=30, t=48, b=30),
        legend=dict(
            orientation="h",
            x=0,
            y=1.18,
            bgcolor="rgba(17,18,15,0.94)",
            bordercolor="rgba(245,241,231,0.22)",
            borderwidth=1,
            font=dict(color="#f5f1e7", size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(31,32,27,0.80)",
        font=dict(color="#f5f1e7"),
    )
    return fig


def _sweep_figure(points: list[tuple[str, float, bool, object]]) -> go.Figure:
    fig = go.Figure()
    speed_pts = [p for p in points if p[0] == "speed"]
    accel_pts = [p for p in points if p[0] == "accel"]
    speed_xs = [p[1] for p in speed_pts] or [0.1, 2.0]
    accel_xs = [p[1] for p in accel_pts] or [0.1, 5.0]
    max_safe_speed = max((p[1] for p in speed_pts if p[2]), default=min(speed_xs))
    max_safe_accel = max((p[1] for p in accel_pts if p[2]), default=min(accel_xs))

    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=min(speed_xs) - 0.05,
        x1=max_safe_speed,
        y0=1.75,
        y1=2.25,
        fillcolor="rgba(104, 211, 145, 0.16)",
        line=dict(width=0),
    )
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=min(accel_xs) - 0.05,
        x1=max_safe_accel,
        y0=0.75,
        y1=1.25,
        fillcolor="rgba(104, 211, 145, 0.16)",
        line=dict(width=0),
    )

    def add_row(axis: str, y: float, label: str) -> None:
        pts = [p for p in points if p[0] == axis]
        safe_x = [p[1] for p in pts if p[2]]
        fail_x = [p[1] for p in pts if not p[2]]
        fail_labels = [getattr(p[3], "value", str(p[3])) for p in pts if not p[2]]
        fig.add_trace(
            go.Scatter(
                x=safe_x,
                y=[y] * len(safe_x),
                mode="markers",
                name=f"{label} safe",
                marker=dict(color="#68d391", size=16, symbol="circle"),
                hovertemplate=f"{label}: %{{x:.2f}} safe<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fail_x,
                y=[y] * len(fail_x),
                mode="markers",
                name=f"{label} fail",
                marker=dict(color="#f25852", size=16, symbol="x"),
                text=fail_labels,
                hovertemplate=f"{label}: %{{x:.2f}} -> %{{text}}<extra></extra>",
            )
        )

    add_row("speed", 2.0, "speed")
    add_row("accel", 1.0, "accel")
    x_min = min(speed_xs + accel_xs) - 0.2
    x_max = max(speed_xs + accel_xs) + 0.3
    fig.update_layout(
        xaxis=dict(title="value", range=[x_min, x_max], gridcolor="rgba(255,255,255,0.10)"),
        yaxis=dict(
            tickmode="array",
            tickvals=[1.0, 2.0],
            ticktext=["accel m/s^2", "speed m/s"],
            range=[0.4, 2.6],
            showgrid=False,
        ),
        height=270,
        margin=dict(l=90, r=20, t=15, b=35),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(31,32,27,0.80)",
        font=dict(color="#f5f1e7"),
    )
    return fig


def _store_sim_result(key_prefix: str, cfg: PalletConfig, profile: ConveyorProfile) -> None:
    trace = simulate(cfg, profile)
    mode, t_fail = first_failure(trace)
    angles = tip_angle_deg(trace)
    st.session_state[f"{key_prefix}_result"] = {
        "mode": mode,
        "t_fail": t_fail,
        "max_tip": float(angles.max() if len(angles) else 0.0),
        "runtime_ms": trace.runtime_s * 1000.0,
        "trace": trace.downsample(hz=80),
        "anim_trace": trace.downsample(hz=24),
        "profile": profile,
        "hash": _pallet_hash(cfg),
    }


def render_sim_result(
    key_prefix: str,
    *,
    compact: bool = False,
    show_replay: bool = True,
) -> None:
    result = st.session_state.get(f"{key_prefix}_result")
    if result is None:
        st.info("Run a simulation to unlock replay and signal traces.")
        return

    mode = result["mode"]
    t_fail = result["t_fail"]
    tone = _failure_tone(mode)
    status = "PASS" if mode == FailureMode.NO_FAILURE else "FAIL"
    when = "no failure" if t_fail is None else f"{t_fail:.2f}s"
    cards = [
        _metric_html("Run result", status, mode.value, tone),
        _metric_html("Failure time", when, "simulation clock"),
        _metric_html("Max tip", f"{result['max_tip']:.2f} deg", "pallet tilt"),
        _metric_html("Runtime", f"{result['runtime_ms']:.0f} ms", "wall clock"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    if not show_replay:
        st.plotly_chart(_trace_figure(result["trace"]), use_container_width=True)
        return

    tabs = st.tabs(["Replay", "Signals"])
    with tabs[0]:
        render_animation_surface(key_prefix, compact=compact)
    with tabs[1]:
        st.plotly_chart(_trace_figure(result["trace"]), use_container_width=True)


def render_animation_surface(key_prefix: str, *, compact: bool = False) -> bool:
    result = st.session_state.get(f"{key_prefix}_result")
    if result is None:
        return False
    if compact:
        follow = False
        loop_cycles = 3
        frames = 40
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        follow = c1.checkbox("Follow pallet", value=False, key=f"{key_prefix}_follow")
        loop_cycles = c2.selectbox("Loops", [1, 3, 10, 50], index=1, key=f"{key_prefix}_loop")
        frames = c3.slider("Frames", 10, 80, 45, step=5, key=f"{key_prefix}_frames")
    anim_fig = animate_trace(
        result["anim_trace"],
        max_frames=int(frames),
        follow_pallet=bool(follow),
        loop_cycles=int(loop_cycles),
        theme="dark",
        height=500 if compact else 560,
        show_bay=True,
    )
    st.plotly_chart(anim_fig, use_container_width=True, key=f"{key_prefix}_anim")
    return True


def render_mission_scene(cfg: PalletConfig) -> None:
    if render_animation_surface("mission", compact=True):
        return
    st.plotly_chart(
        render_pallet(
            cfg,
            theme="dark",
            show_floor=True,
            show_legend=False,
            height=560,
        ),
        use_container_width=True,
        key="mission_static",
    )


def _json_payload(cfg: PalletConfig) -> str:
    return json.dumps(cfg.model_dump(mode="json"), indent=2)


def _curl_snippet() -> str:
    return (
        'export PALLET_API_URL="http://localhost:8000"\n'
        'curl -s -X POST "$PALLET_API_URL/safety/analyze" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        "  --data @pallet.json"
    )


def _python_batch_snippet() -> str:
    return """import json
import requests

api = "http://localhost:8000"
with open("pallets.json", "r", encoding="utf-8") as f:
    pallets = json.load(f)

results = requests.post(f"{api}/safety/batch", json=pallets, timeout=60).json()
for item in results:
    r = item["result"]
    print(r["pallet_id"], r["max_speed_mps"], r["max_accel_mps2"], r["dominant_failure_mode"])
"""


def safety_analysis_inline(cfg: PalletConfig, *, key_prefix: str = "sa") -> None:
    st.subheader("Safety envelope")
    c1, c2, c3 = st.columns([1, 1, 2])
    auto_run = c1.checkbox("Auto analyze", value=True, key=f"{key_prefix}_auto")
    manual_run = c2.button("Analyze now", type="primary", key=f"{key_prefix}_run")
    cache_key = f"{key_prefix}_analysis"
    cfg_hash = _pallet_hash(cfg)

    should_run = manual_run or (
        auto_run and st.session_state.get(f"{key_prefix}_last_hash") != cfg_hash
    )
    if should_run:
        with st.spinner("Searching safe operating envelope..."):
            st.session_state[cache_key] = default_analyzer().analyze(cfg)
            st.session_state[f"{key_prefix}_last_hash"] = cfg_hash

    analysis: AnalysisResult | None = st.session_state.get(cache_key)
    if analysis is None:
        st.info("Analyze this pallet to see speed and acceleration limits.")
        return

    r = analysis.result
    tone = _failure_tone(r.dominant_failure_mode)
    cards = [
        _metric_html("Max speed", f"{r.max_speed_mps:.2f} m/s", "sustained belt", tone),
        _metric_html("Max accel", f"{r.max_accel_mps2:.2f} m/s^2", "ramp limit", tone),
        _metric_html("Margin", f"{r.margin_pct:.0f}%", "weakest axis"),
        _metric_html("Confidence", f"{r.confidence:.2f}", f"{analysis.sims_run} sims"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.6])
    with left:
        status_class = "good" if r.dominant_failure_mode == FailureMode.NO_FAILURE else "bad"
        st.markdown(
            f'<div class="panel"><div class="panel-title">Dominant mode</div>'
            f'<span class="pill {status_class}">{escape(r.dominant_failure_mode.value)}</span>'
            f'<div class="code-note">hash {escape(r.config_hash)} | '
            f'{analysis.cache_hits} cache hits | {r.sim_runtime_ms:.0f} ms total</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(_sweep_figure(analysis.sweep_points), use_container_width=True)


# ---- pages ----

def _load_mission_scenario(name: str) -> None:
    cfg, profile = _build_demo_scenario(name)
    st.session_state.mission_cfg = cfg
    st.session_state.mission_speed = profile.target_speed_mps
    st.session_state.mission_accel = profile.accel_mps2
    st.session_state.mission_duration = profile.duration_s
    st.session_state.mission_env = cfg.env.value
    st.session_state.mission_wrap = cfg.wrap.value
    st.session_state.mission_temp = cfg.body_temp_c
    st.session_state.mission_scenario_applied = name
    st.session_state.mission_result = None
    st.session_state.mission_analysis = None


def _ensure_mission_state(default_scenario: str) -> None:
    if "mission_scenario" not in st.session_state:
        st.session_state.mission_scenario = default_scenario
    if st.session_state.mission_scenario not in DEMO_SCENARIOS:
        st.session_state.mission_scenario = default_scenario
    if "mission_cfg" not in st.session_state:
        _load_mission_scenario(st.session_state.mission_scenario)
        return

    scenario = st.session_state.get("mission_scenario", default_scenario)
    spec = DEMO_SCENARIOS.get(scenario, DEMO_SCENARIOS[default_scenario])
    profile = ConveyorProfile(**spec["profile"])
    cfg: PalletConfig = st.session_state.mission_cfg
    defaults = {
        "mission_speed": profile.target_speed_mps,
        "mission_accel": profile.accel_mps2,
        "mission_duration": profile.duration_s,
        "mission_env": cfg.env.value,
        "mission_wrap": cfg.wrap.value,
        "mission_temp": cfg.body_temp_c,
        "mission_scenario_applied": scenario,
        "mission_result": None,
        "mission_analysis": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_mission_control() -> None:
    default_scenario = "Frozen meat sprint"
    if "pending_topple" in st.session_state:
        topple_name = st.session_state.pop("pending_topple")
        st.session_state.mission_scenario = topple_name
        _load_mission_scenario(topple_name)
        topple_profile = ConveyorProfile(**DEMO_SCENARIOS[topple_name]["profile"])
        with st.spinner(f"Toppling {topple_name}..."):
                _store_sim_result("mission", st.session_state.mission_cfg, topple_profile)
    _ensure_mission_state(default_scenario)

    _header(
        "Pallet Safety Command",
        "Real-time trust loop",
        "Pick a pallet, hit it with conveyor motion, then hand the same payload to the batch API.",
    )

    view_col, control_col = st.columns([1.55, 1.0], gap="large")
    with control_col:
        st.subheader("Runbook")
        scenario = st.selectbox(
            "Scenario",
            list(DEMO_SCENARIOS.keys()),
            key="mission_scenario",
            help="Demo pallets tuned to show stable and failing behavior.",
        )
        if st.session_state.get("mission_scenario_applied") != scenario:
            _load_mission_scenario(scenario)

        scenario_spec = DEMO_SCENARIOS[scenario]
        st.markdown(
            f'<span class="pill warn">{escape(scenario_spec["tag"])}</span>',
            unsafe_allow_html=True,
        )
        quick_profile = ConveyorProfile(
            target_speed_mps=float(st.session_state.mission_speed),
            accel_mps2=float(st.session_state.mission_accel),
            duration_s=float(st.session_state.mission_duration),
        )
        render_load_story(
            st.session_state.mission_cfg,
            source="Mission control",
            scenario_name=scenario,
            profile=quick_profile,
            result_key="mission",
        )
        q1, q2 = st.columns(2)
        if q1.button("Run sim", type="primary", use_container_width=True, key="mission_run_top"):
            with st.spinner("Running MuJoCo..."):
                _store_sim_result("mission", st.session_state.mission_cfg, quick_profile)
        if q2.button("Safe limits", use_container_width=True, key="mission_limits_top"):
            with st.spinner("Searching envelope..."):
                st.session_state.mission_analysis = default_analyzer().analyze(
                    st.session_state.mission_cfg
                )
        st.markdown('<div class="code-note">Topple lab: load a known bad pallet and run it hard.</div>', unsafe_allow_html=True)
        topple_cols = st.columns(4)
        for idx, (label, scenario_name) in enumerate(TOPPLE_LAB.items()):
            if topple_cols[idx].button(
                label,
                use_container_width=True,
                key=f"topple_{idx}",
                help=scenario_name,
            ):
                st.session_state.pending_topple = scenario_name
                st.rerun()
        st.divider()

        cfg: PalletConfig = st.session_state.mission_cfg
        env = st.selectbox(
            "Environment",
            [e.value for e in EnvCondition],
            index=[e.value for e in EnvCondition].index(st.session_state.mission_env),
            key="mission_env",
        )
        wrap = st.selectbox(
            "Wrap",
            [w.value for w in WrapType],
            index=[w.value for w in WrapType].index(st.session_state.mission_wrap),
            key="mission_wrap",
        )
        temp = st.slider(
            "Body temp (C)",
            -30.0,
            25.0,
            value=float(st.session_state.mission_temp),
            step=0.5,
            key="mission_temp",
        )
        updated = cfg.model_copy(
            update={"env": EnvCondition(env), "wrap": WrapType(wrap), "body_temp_c": temp}
        )
        if _pallet_hash(updated) != _pallet_hash(cfg):
            st.session_state.mission_cfg = updated
            st.session_state.mission_result = None
            st.session_state.mission_analysis = None
            cfg = updated

        st.divider()
        st.subheader("Conveyor hit")
        st.slider(
            "Target speed (m/s)",
            0.1,
            3.0,
            value=float(st.session_state.mission_speed),
            step=0.05,
            key="mission_speed",
        )
        st.slider(
            "Acceleration (m/s^2)",
            0.1,
            8.0,
            value=float(st.session_state.mission_accel),
            step=0.05,
            key="mission_accel",
        )
        st.slider(
            "Duration (s)",
            0.5,
            6.0,
            value=float(st.session_state.mission_duration),
            step=0.25,
            key="mission_duration",
        )

    with view_col:
        cfg = st.session_state.mission_cfg
        render_kpi_strip(cfg)
        render_mission_scene(cfg)

    tabs = st.tabs(["Signals", "Safety envelope", "Clone/API"])
    with tabs[0]:
        render_sim_result("mission", show_replay=False)
    with tabs[1]:
        analysis: AnalysisResult | None = st.session_state.get("mission_analysis")
        if analysis is None:
            st.info("Run Safe limits to calculate deployable thresholds for this pallet.")
        else:
            r = analysis.result
            cards = [
                _metric_html("Max speed", f"{r.max_speed_mps:.2f} m/s", "API output"),
                _metric_html("Max accel", f"{r.max_accel_mps2:.2f} m/s^2", "API output"),
                _metric_html("Margin", f"{r.margin_pct:.0f}%", r.dominant_failure_mode.value),
                _metric_html("Sims", f"{analysis.sims_run}", f"{r.sim_runtime_ms:.0f} ms"),
            ]
            st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
            st.plotly_chart(_sweep_figure(analysis.sweep_points), use_container_width=True)
    with tabs[2]:
        st.markdown('<div class="code-note">Save the current payload as pallet.json.</div>', unsafe_allow_html=True)
        st.download_button(
            "Download pallet JSON",
            data=_json_payload(cfg),
            file_name="pallet.json",
            mime="application/json",
        )
        c1, c2 = st.columns(2)
        with c1:
            st.code(_curl_snippet(), language="bash")
        with c2:
            st.code(_python_batch_snippet(), language="python")


def page_scanner_feed() -> None:
    _header("Scanner Feed", "Mock machine-vision adapter", "Generate the same input contract the API accepts.")

    with st.sidebar:
        st.subheader("Generator")
        seed_str = st.text_input("Seed", value="42", key="gen_seed")
        seed = int(seed_str) if seed_str.strip().isdigit() else None
        anomaly_rate = st.slider("Anomaly rate", 0.0, 1.0, 0.10, step=0.05, key="gen_anom")
        col_l, col_r = st.columns(2)
        min_layers = col_l.number_input("Min layers", 0, 10, 1, key="gen_min_l")
        max_layers = col_r.number_input("Max layers", 0, 20, 5, key="gen_max_l")
        min_ipl = col_l.number_input("Min items/layer", 0, 10, 1, key="gen_min_i")
        max_ipl = col_r.number_input("Max items/layer", 0, 20, 4, key="gen_max_i")
        surface_pair_s = st.selectbox(
            "Surface pair",
            ["/".join(p) for p in available_surface_pairs()],
            index=0,
            key="gen_surface",
        )
        regen = st.button("Generate pallet", type="primary", use_container_width=True)

    pair = tuple(surface_pair_s.split("/"))
    if regen or "current_cfg" not in st.session_state:
        cfg = random_pallet(seed, anomaly_rate, int(min_layers), int(max_layers), int(min_ipl), int(max_ipl))
        st.session_state.current_cfg = cfg
        st.session_state.surface_pair = pair
        st.session_state.gen_result = None

    cfg = st.session_state.current_cfg
    render_kpi_strip(cfg)
    render_load_story(cfg, source="Scanner feed")
    col_a, col_b = st.columns([1.5, 1.0], gap="large")
    with col_a:
        st.plotly_chart(
            render_pallet(cfg, theme="dark", show_floor=True, show_legend=False),
            use_container_width=True,
            key="gen_static",
        )
    with col_b:
        mu_s, mu_d = friction_coefficient(cfg.body_temp_c, cfg.seconds_since_temp_change, pair)
        st.metric("Static friction", f"{mu_s:.3f}", help="mu_static")
        st.metric("Dynamic friction", f"{mu_d:.3f}", help="mu_dynamic")
        st.metric("Config hash", _pallet_hash(cfg))
        st.download_button(
            "Download payload",
            data=_json_payload(cfg),
            file_name="pallet.json",
            mime="application/json",
            use_container_width=True,
        )

    run_sim_inline(cfg, key_prefix="gen")

    tabs = st.tabs(["Items", "PalletConfig JSON", "MJCF"])
    with tabs[0]:
        st.dataframe(_pallet_dataframe(cfg), use_container_width=True, height=360)
    with tabs[1]:
        st.json(json.loads(cfg.model_dump_json()))
    with tabs[2]:
        st.code(mjcf_for(cfg, st.session_state.surface_pair), language="xml")


def page_manual() -> None:
    _header("Build Pallet", "Stack-by-stack configurator", "Compose a real pallet pattern and test it immediately.")

    if "pending_crash" in st.session_state:
        preset_name = st.session_state.pop("pending_crash")
        preset = CRASH_PRESETS[preset_name]
        st.session_state.manual_loaded_preset = preset_name
        st.session_state.stacks = list(preset["stacks"])
        st.session_state.mc_env = preset["env"]
        st.session_state.mc_env_synced = preset["env"]
        st.session_state.mc_t = preset["body_temp"]
        st.session_state.mc_t_synced = preset["body_temp"]
        st.session_state.mc_wrap = preset["wrap"]
        st.session_state.mc_wrap_synced = preset["wrap"]
        st.session_state.manual_qr_v = preset["profile"]["target_speed_mps"]
        st.session_state.manual_qr_a = preset["profile"]["accel_mps2"]
        st.session_state.manual_qr_d = preset["profile"]["duration_s"]
        st.session_state.manual_qr_result = None

    if "stacks" not in st.session_state:
        st.session_state.stacks = []

    with st.sidebar:
        st.subheader("Pallet base")
        base_type = st.selectbox("Type", list(BASE_DEFAULTS), index=0, key="mc_base")
        env_opts = [e.value for e in EnvCondition]
        wrap_opts = [w.value for w in WrapType]
        env_default = env_opts.index(st.session_state.get("mc_env_synced", "refrigerated"))
        wrap_default = wrap_opts.index(st.session_state.get("mc_wrap_synced", "stretch"))
        env = st.selectbox("Environment", env_opts, index=env_default, key="mc_env")
        body_temp = st.slider(
            "Body temp (C)",
            -30.0,
            25.0,
            value=float(st.session_state.get("mc_t_synced", 2.0)),
            step=0.5,
            key="mc_t",
        )
        wrap = st.selectbox("Wrap", wrap_opts, index=wrap_default, key="mc_wrap")
        st.session_state.mc_env_synced = env
        st.session_state.mc_wrap_synced = wrap
        st.session_state.mc_t_synced = body_temp

        st.divider()
        st.subheader("Add stack")
        sku_to_add = st.selectbox("SKU", all_skus(), key="mc_sku", format_func=_fmt_sku)
        base_dims = BASE_DEFAULTS[base_type]["dims"]
        existing_specs = [StackSpec(**s) for s in st.session_state.stacks]
        planned = existing_specs + [StackSpec(sku=sku_to_add, grid_row=0, grid_col=0, height=1)]
        rows, cols = compute_grid_shape(planned, base_dims)
        st.caption(f"Auto-grid: {rows}x{cols} cells")
        cell_labels = [f"R{r}C{c}" for r in range(rows) for c in range(cols)]
        grid_choice = st.selectbox("Grid cell", cell_labels, key="mc_cell")
        height = st.slider("Height", 1, 10, 5, key="mc_h")

        if st.button("Add stack", type="primary", use_container_width=True):
            r = int(grid_choice[1])
            c = int(grid_choice[3])
            st.session_state.stacks.append(
                {"sku": sku_to_add, "grid_row": r, "grid_col": c, "height": height}
            )
            st.session_state.manual_loaded_preset = None

        c1, c2 = st.columns(2)
        if c1.button("Fill cells", use_container_width=True):
            fill_rows, fill_cols = compute_grid_shape(
                [StackSpec(sku=sku_to_add, grid_row=0, grid_col=0, height=1)],
                base_dims,
            )
            st.session_state.stacks = [
                {"sku": sku_to_add, "grid_row": r, "grid_col": c, "height": height}
                for r in range(fill_rows)
                for c in range(fill_cols)
            ]
            st.session_state.manual_loaded_preset = None
            st.session_state.manual_qr_result = None
        if c2.button("Clear", use_container_width=True):
            st.session_state.stacks = []
            st.session_state.manual_loaded_preset = None
            st.session_state.manual_qr_result = None

        st.divider()
        st.subheader("Failure presets")
        crash_choice = st.selectbox("Preset", ["(none)", *CRASH_PRESETS.keys()], key="mc_crash")
        if st.button("Load preset", use_container_width=True, disabled=(crash_choice == "(none)")):
            st.session_state.pending_crash = crash_choice
            st.rerun()

    base_cfg = BASE_DEFAULTS[base_type]
    specs = [StackSpec(**s) for s in st.session_state.stacks]
    cfg = build_from_stacks(
        specs,
        pallet_id="manual-build",
        env=EnvCondition(env),
        body_temp_c=body_temp,
        wrap=WrapType(wrap),
        base_type=base_type,
        base_dims_m=base_cfg["dims"],
        base_mass_kg=base_cfg["mass"],
    )

    grid_rows, grid_cols = compute_grid_shape(specs, base_cfg["dims"])
    fitting = sum(1 for s in specs if s.grid_row < grid_rows and s.grid_col < grid_cols)
    if fitting < len(specs):
        st.warning(
            f"{len(specs) - fitting} stack(s) fall outside the auto-grid "
            f"({grid_rows}x{grid_cols}) and were skipped."
        )

    render_kpi_strip(cfg)
    manual_profile = ConveyorProfile(
        target_speed_mps=float(st.session_state.get("manual_qr_v", 0.8)),
        accel_mps2=float(st.session_state.get("manual_qr_a", 1.0)),
        duration_s=float(st.session_state.get("manual_qr_d", 2.0)),
    )
    manual_preset = st.session_state.get("manual_loaded_preset")
    manual_expected = None
    if manual_preset in CRASH_PRESETS:
        manual_expected = str(CRASH_PRESETS[manual_preset].get("failure_type"))
    render_load_story(
        cfg,
        source="Manual build",
        scenario_name=manual_preset if manual_preset in DEMO_SCENARIOS else None,
        profile=manual_profile,
        result_key="manual_qr",
        expected_failure=manual_expected,
    )
    col_a, col_b = st.columns([1.45, 1.0], gap="large")
    with col_a:
        if cfg.items:
            st.plotly_chart(
                render_pallet(cfg, theme="dark", show_floor=True, show_legend=False),
                use_container_width=True,
                key="manual_static",
            )
        else:
            st.info("Add stacks from the sidebar to create a pallet.")
    with col_b:
        if st.session_state.stacks:
            st.subheader(f"Stacks ({len(st.session_state.stacks)})")
            _render_stacks_table(grid_rows, grid_cols)
        else:
            st.markdown('<div class="panel"><div class="panel-title">No stacks yet</div></div>', unsafe_allow_html=True)

    run_sim_inline(cfg, key_prefix="manual")

    with st.expander("MJCF preview"):
        if cfg.items:
            st.code(build_mjcf(cfg), language="xml")
        else:
            st.caption("Nothing to build yet.")


def _render_stacks_table(rows: int, cols: int) -> None:
    header = st.columns([3, 1, 1, 1])
    header[0].markdown("**SKU**")
    header[1].markdown("**Cell**")
    header[2].markdown("**Height**")
    header[3].markdown(" ")
    to_remove: int | None = None
    for i, s in enumerate(st.session_state.stacks):
        c = st.columns([3, 1, 1, 1])
        in_grid = s["grid_row"] < rows and s["grid_col"] < cols
        sku_label = _fmt_sku(s["sku"])
        if not in_grid:
            sku_label = "! " + sku_label
        c[0].text(sku_label)
        c[1].text(f"R{s['grid_row']}C{s['grid_col']}")
        c[2].text(str(s["height"]))
        if c[3].button("Remove", key=f"rm_stack_{i}"):
            to_remove = i
    if to_remove is not None:
        st.session_state.stacks.pop(to_remove)
        st.session_state.manual_loaded_preset = None
        st.session_state.manual_qr_result = None
        st.rerun()


def run_sim_inline(cfg: PalletConfig, key_prefix: str = "") -> None:
    with st.expander("Run live simulation", expanded=False):
        if not cfg.items:
            st.caption("Add items first.")
            return
        c1, c2, c3 = st.columns(3)
        target_v = c1.slider(
            "Target speed (m/s)",
            0.1,
            3.0,
            0.8,
            step=0.1,
            key=f"{key_prefix}_qr_v",
        )
        accel = c2.slider(
            "Accel (m/s^2)",
            0.1,
            8.0,
            1.0,
            step=0.1,
            key=f"{key_prefix}_qr_a",
        )
        duration = c3.slider(
            "Duration (s)",
            0.5,
            5.0,
            2.0,
            step=0.5,
            key=f"{key_prefix}_qr_d",
        )
        if st.button("Run", type="primary", key=f"{key_prefix}_qr_go"):
            profile = ConveyorProfile(
                target_speed_mps=target_v,
                accel_mps2=accel,
                duration_s=duration,
            )
            with st.spinner("Simulating..."):
                _store_sim_result(f"{key_prefix}_qr", cfg, profile)

        render_sim_result(f"{key_prefix}_qr")


def page_solver() -> None:
    _header("Live Solver", "Single profile run", "Dial the conveyor motion and inspect failure traces.")

    with st.sidebar:
        st.subheader("Pallet source")
        source = st.radio("Source", ["Random pallet", "Current scanner feed"], index=0)
        if source == "Random pallet":
            seed_str = st.text_input("Seed", value="42", key="solver_seed")
            seed = int(seed_str) if seed_str.strip().isdigit() else None
            if st.button("Regenerate", use_container_width=True):
                st.session_state.solver_cfg = random_pallet(seed, 0.0, 3, 5, 4, 6)
                st.session_state.solver_result = None

        st.subheader("Conveyor profile")
        target_v = st.slider("Target speed (m/s)", 0.0, 3.0, 0.8, step=0.05, key="solver_v")
        accel = st.slider("Acceleration (m/s^2)", 0.05, 8.0, 1.0, step=0.05, key="solver_a")
        duration = st.slider("Duration (s)", 0.5, 10.0, 3.0, step=0.5, key="solver_d")

    cfg: PalletConfig | None = st.session_state.get("solver_cfg")
    if cfg is None:
        cfg = random_pallet(42, 0.0, 3, 5, 4, 6)
        st.session_state.solver_cfg = cfg

    if source == "Current scanner feed":
        cfg = st.session_state.get("current_cfg")
        if cfg is None:
            st.warning("Generate a pallet on Scanner Feed first.")
            return

    render_kpi_strip(cfg)
    col_a, col_b = st.columns([1.4, 1.0], gap="large")
    with col_a:
        st.plotly_chart(
            render_pallet(cfg, theme="dark", show_floor=True, show_legend=False),
            use_container_width=True,
            key="solver_static",
        )
    with col_b:
        profile = ConveyorProfile(target_speed_mps=target_v, accel_mps2=accel, duration_s=duration)
        if st.button("Run solver", type="primary", use_container_width=True):
            with st.spinner("Running MuJoCo..."):
                _store_sim_result("solver", cfg, profile)
        st.code(json.dumps(profile.model_dump(mode="json"), indent=2), language="json")

    render_sim_result("solver")


def page_safety() -> None:
    _header("Safety Envelope", "Threshold search", "Find the operating limits that get sent downstream.")

    with st.sidebar:
        st.subheader("Pallet source")
        source = st.radio(
            "Source",
            ["Random pallet", "Scanner feed", "Manual build", "Mission control"],
            key="sa_src",
        )
        if source == "Random pallet":
            seed_str = st.text_input("Seed", value="42", key="sa_seed")
            seed = int(seed_str) if seed_str.strip().isdigit() else None
            if st.button("Regenerate", use_container_width=True):
                st.session_state.sa_cfg = random_pallet(seed, 0.0, 3, 5, 4, 6)
                st.session_state.sa_analysis = None

    cfg: PalletConfig | None = None
    if source == "Random pallet":
        if "sa_cfg" not in st.session_state:
            st.session_state.sa_cfg = random_pallet(42, 0.0, 3, 5, 4, 6)
        cfg = st.session_state.sa_cfg
    elif source == "Scanner feed":
        cfg = st.session_state.get("current_cfg")
        if cfg is None:
            st.warning("Generate a pallet on Scanner Feed first.")
            return
    elif source == "Manual build":
        specs = [StackSpec(**s) for s in st.session_state.get("stacks", [])]
        if not specs:
            st.warning("Build stacks on Build Pallet first.")
            return
        cfg = build_from_stacks(
            specs,
            pallet_id="manual-build",
            env=EnvCondition(st.session_state.get("mc_env_synced", "refrigerated")),
            body_temp_c=float(st.session_state.get("mc_t_synced", 2.0)),
            wrap=WrapType(st.session_state.get("mc_wrap_synced", "stretch")),
        )
    else:
        _ensure_mission_state("Frozen meat sprint")
        cfg = st.session_state.get("mission_cfg")
        if cfg is None:
            cfg, _ = _build_demo_scenario("Frozen meat sprint")

    render_kpi_strip(cfg)
    story_profile = None
    story_scenario = None
    expected_failure = None
    if source == "Mission control":
        story_scenario = st.session_state.get("mission_scenario", "Frozen meat sprint")
        story_profile = ConveyorProfile(
            target_speed_mps=float(st.session_state.get("mission_speed", 1.4)),
            accel_mps2=float(st.session_state.get("mission_accel", 3.2)),
            duration_s=float(st.session_state.get("mission_duration", 2.0)),
        )
    elif source == "Manual build":
        manual_preset = st.session_state.get("manual_loaded_preset")
        if manual_preset in CRASH_PRESETS:
            expected_failure = str(CRASH_PRESETS[manual_preset].get("failure_type"))
    render_load_story(
        cfg,
        source=source,
        scenario_name=story_scenario,
        profile=story_profile,
        expected_failure=expected_failure,
    )
    col_a, col_b = st.columns([1.45, 1.0], gap="large")
    with col_a:
        st.plotly_chart(
            render_pallet(cfg, theme="dark", show_floor=True, show_legend=False),
            use_container_width=True,
            key="sa_static",
        )
    with col_b:
        st.metric("Total mass", f"{cfg.total_mass_kg:.1f} kg")
        st.metric("Items", len(cfg.items))
        st.metric("Stack height", f"{cfg.stack_height_m:.2f} m")
        st.metric("Hash", _pallet_hash(cfg))

    safety_analysis_inline(cfg, key_prefix="sa")


def page_api_launchpad() -> None:
    _header("API Batch", "Clone-and-run power path", "The UI earns trust; the API does the real work at scale.")

    source_col, action_col = st.columns([1, 1], gap="large")
    with source_col:
        st.subheader("Current payload")
        cfg = st.session_state.get("mission_cfg")
        if cfg is None:
            cfg, _ = _build_demo_scenario("Stable dairy slab")
        cfg_source = st.radio(
            "Payload source",
            ["Mission control", "Scanner feed", "Manual build"],
            horizontal=True,
            key="api_source",
        )
        if cfg_source == "Scanner feed" and st.session_state.get("current_cfg") is not None:
            cfg = st.session_state.current_cfg
        elif cfg_source == "Manual build" and st.session_state.get("stacks"):
            specs = [StackSpec(**s) for s in st.session_state.get("stacks", [])]
            cfg = build_from_stacks(
                specs,
                pallet_id="manual-build",
                env=EnvCondition(st.session_state.get("mc_env_synced", "refrigerated")),
                body_temp_c=float(st.session_state.get("mc_t_synced", 2.0)),
                wrap=WrapType(st.session_state.get("mc_wrap_synced", "stretch")),
            )
        render_kpi_strip(cfg)
        story_profile = None
        story_scenario = None
        expected_failure = None
        if cfg_source == "Mission control":
            story_scenario = st.session_state.get("mission_scenario", "Stable dairy slab")
            story_profile = ConveyorProfile(
                target_speed_mps=float(st.session_state.get("mission_speed", 0.9)),
                accel_mps2=float(st.session_state.get("mission_accel", 0.8)),
                duration_s=float(st.session_state.get("mission_duration", 2.5)),
            )
        elif cfg_source == "Manual build":
            manual_preset = st.session_state.get("manual_loaded_preset")
            if manual_preset in CRASH_PRESETS:
                expected_failure = str(CRASH_PRESETS[manual_preset].get("failure_type"))
        render_load_story(
            cfg,
            source=cfg_source,
            scenario_name=story_scenario,
            profile=story_profile,
            expected_failure=expected_failure,
        )
        st.download_button(
            "Download pallet.json",
            data=_json_payload(cfg),
            file_name="pallet.json",
            mime="application/json",
            use_container_width=True,
        )
        with st.expander("Payload JSON", expanded=False):
            st.code(_json_payload(cfg), language="json")

    with action_col:
        st.subheader("Batch runner")
        batch_n = st.slider("Pallet count", 2, 16, 6, step=1, key="api_batch_n")
        start_seed = st.number_input("Start seed", value=100, step=1, key="api_seed")
        if st.button("Run local batch", type="primary", use_container_width=True):
            with st.spinner("Analyzing batch..."):
                analyzer = default_analyzer()
                rows = []
                for idx in range(int(batch_n)):
                    batch_cfg = random_pallet(int(start_seed) + idx, 0.08, 2, 5, 3, 6)
                    analysis = analyzer.analyze(batch_cfg)
                    r = analysis.result
                    rows.append(
                        {
                            "pallet_profile": _pallet_profile_label(batch_cfg),
                            "technical_id": r.pallet_id,
                            "max_speed_mps": r.max_speed_mps,
                            "max_accel_mps2": r.max_accel_mps2,
                            "mode": r.dominant_failure_mode.value,
                            "margin_pct": r.margin_pct,
                            "runtime_ms": r.sim_runtime_ms,
                            "sims": analysis.sims_run,
                        }
                    )
                st.session_state.api_batch_df = pd.DataFrame(rows)

        df = st.session_state.get("api_batch_df")
        if df is not None:
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.info("Run a local batch to see the API output shape.")

    tabs = st.tabs(["curl", "Python", "Endpoints"])
    with tabs[0]:
        st.code(_curl_snippet(), language="bash")
    with tabs[1]:
        st.code(_python_batch_snippet(), language="python")
    with tabs[2]:
        rows = [
            {"method": "POST", "path": "/safety/analyze", "use": "single pallet threshold result"},
            {"method": "POST", "path": "/safety/batch", "use": "many pallets, shared cache"},
            {"method": "POST", "path": "/solve", "use": "single conveyor profile trace"},
            {"method": "POST", "path": "/pallet/random", "use": "mock adapter payload"},
            {"method": "POST", "path": "/mjcf/build", "use": "MuJoCo XML for a config"},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)


def page_friction() -> None:
    _header("Friction Lab", "Temperature and surface pair", "Inspect the mu curve that drives slip behavior.")

    with st.sidebar:
        st.subheader("Plot controls")
        pair_s = st.selectbox(
            "Surface pair",
            ["/".join(p) for p in available_surface_pairs()],
            index=0,
            key="fr_pair",
        )
        seconds = st.slider("Seconds since temp change", 0, 1800, 0, step=30, key="fr_sec")
        t_min = st.number_input("T min (C)", value=-30, key="fr_min")
        t_max = st.number_input("T max (C)", value=25, key="fr_max")

    pair = tuple(pair_s.split("/"))
    temps = [t_min + i * (t_max - t_min) / 199 for i in range(200)]
    mu_s_vals, mu_d_vals = [], []
    for t in temps:
        mu_s, mu_d = friction_coefficient(t, float(seconds), pair)
        mu_s_vals.append(mu_s)
        mu_d_vals.append(mu_d)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temps, y=mu_s_vals, name="mu_static", line=dict(color="#4aa9c8", width=3)))
    fig.add_trace(go.Scatter(x=temps, y=mu_d_vals, name="mu_dynamic", line=dict(color="#f25852", width=3, dash="dash")))
    fig.add_vrect(x0=-2, x1=8, fillcolor="rgba(234,174,70,0.18)", line_width=0)
    fig.update_layout(
        xaxis=dict(title="body temperature (C)", gridcolor="rgba(255,255,255,0.10)"),
        yaxis=dict(title="mu", gridcolor="rgba(255,255,255,0.10)"),
        height=500,
        margin=dict(l=30, r=30, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(31,32,27,0.80)",
        font=dict(color="#f5f1e7"),
        legend=dict(
            orientation="h",
            x=0,
            y=1.12,
            bgcolor="rgba(17,18,15,0.94)",
            bordercolor="rgba(245,241,231,0.22)",
            borderwidth=1,
            font=dict(color="#f5f1e7", size=12),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    spot_t = c1.number_input("Spot temperature (C)", value=0.0, key="spot_t")
    mu_s, mu_d = friction_coefficient(spot_t, float(seconds), pair)
    c2.metric("mu_static", f"{mu_s:.4f}")
    c3.metric("mu_dynamic", f"{mu_d:.4f}")


def page_catalog() -> None:
    _header("SKU Catalog", "Seed product library", "Browse the product mix used by the simulator.")

    rows = []
    for s in all_skus():
        t = get_template(s)
        rows.append(
            {
                "sku": t.sku,
                "name": t.name,
                "weight_kg": t.weight_kg,
                "L_m": t.dims_m[0],
                "W_m": t.dims_m[1],
                "H_m": t.dims_m[2],
                "fragility": t.fragility.value,
                "category": t.category,
                "default_env": t.default_env.value,
            }
        )
    df = pd.DataFrame(rows)
    f1, f2 = st.columns(2)
    cat_filter = f1.multiselect("Category", sorted(df["category"].unique()))
    env_filter = f2.multiselect("Environment", sorted(df["default_env"].unique()))
    if cat_filter:
        df = df[df["category"].isin(cat_filter)]
    if env_filter:
        df = df[df["default_env"].isin(env_filter)]
    st.dataframe(df, use_container_width=True, height=620)


def page_system() -> None:
    _header("System", "Runtime and validation", "Health, catalog coverage, and local test runner.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API status", "connected" if _api_alive() else "local")
    col2.metric("Catalog SKUs", len(all_skus()))
    col3.metric("Surface pairs", len(available_surface_pairs()))
    col4.metric("Default pair", "/".join(DEFAULT_PAIR))

    st.subheader("Local tests")
    if st.button("Run pytest", type="primary"):
        with st.spinner("running pytest..."):
            res = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=short"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
        st.code(res.stdout + ("\nSTDERR:\n" + res.stderr if res.stderr else ""))

    st.subheader("API URL")
    st.code(API_URL, language="text")


PAGES = {
    "Mission Control": page_mission_control,
    "Build Pallet": page_manual,
    "Scanner Feed": page_scanner_feed,
    "Safety Envelope": page_safety,
    "Live Solver": page_solver,
    "API Batch": page_api_launchpad,
    "Friction Lab": page_friction,
    "SKU Catalog": page_catalog,
    "System": page_system,
}


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="name">Pallet Safety</div>
          <div class="sub">physics inference console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    api_class = "good" if _api_alive() else "warn"
    api_text = "connected" if _api_alive() else "in-process"
    st.markdown(f'<span class="pill {api_class}">API {api_text}</span>', unsafe_allow_html=True)
    st.caption(API_URL)

PAGES[page]()
