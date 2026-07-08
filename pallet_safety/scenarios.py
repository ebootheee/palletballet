"""Curated demo scenarios — the canonical trust-building set.

One known-good baseline, four engineered failures, and one randomized
scanner-style payload. The Streamlit UI, the HTTP API (`GET /scenarios`), and
the public demo at boothe.io/palletballet all read from this table so the
pallet a visitor topples in the browser is byte-identical to the one the
operator console and the test suite use.

Scenario pallets are deterministic: the random scanner feed uses a fixed seed
through the same MockRandomAdapter → Configurator path a real adapter would.
"""

from __future__ import annotations

from functools import cache

from pydantic import BaseModel, ConfigDict

from .configurator import Configurator, StackSpec, build_from_stacks
from .inputs import MockRandomAdapter
from .models import EnvCondition, PalletConfig, WrapType
from .solver import ConveyorProfile


class Scenario(BaseModel):
    """A named, reproducible pallet + the conveyor profile that tells its story."""

    model_config = ConfigDict(frozen=True)

    slug: str
    name: str
    tag: str
    description: str
    expected_failure: str
    pallet: PalletConfig
    suggested_profile: ConveyorProfile


# Raw specs. `stacks` entries are StackSpec kwargs; `random` scenarios go
# through the adapter path instead.
_SPECS: list[dict] = [
    {
        "slug": "stable-dairy-slab",
        "name": "Stable dairy slab",
        "tag": "High confidence baseline",
        "expected_failure": "none expected",
        "description": (
            "Low, even dairy cases with stretch wrap. This is the trust-building "
            "baseline: fast enough to be useful, boring enough to pass."
        ),
        "stacks": [
            {"sku": "SKU-FD-001", "grid_row": r, "grid_col": c, "height": 2}
            for r in range(2) for c in range(2)
        ],
        "wrap": "stretch",
        "env": "refrigerated",
        "body_temp": 2.0,
        "profile": {"target_speed_mps": 0.9, "accel_mps2": 0.8, "duration_s": 2.5},
    },
    {
        "slug": "frozen-meat-sprint",
        "name": "Frozen meat sprint",
        "tag": "Friction-limited cold run",
        "expected_failure": "pallet slip at aggressive accel",
        "description": (
            "Dense frozen beef cartons with wrap in a frozen room. The load itself "
            "is solid; the interesting limit is belt-to-pallet friction during a "
            "hard start."
        ),
        "stacks": [
            {"sku": "SKU-FM-001", "grid_row": r, "grid_col": c, "height": 4}
            for r in range(2) for c in range(3)
        ],
        "wrap": "stretch",
        "env": "frozen",
        "body_temp": -25.0,
        "profile": {"target_speed_mps": 1.4, "accel_mps2": 3.2, "duration_s": 2.0},
    },
    {
        "slug": "tall-unwrapped-tower",
        "name": "Tall unwrapped tower",
        "tag": "Visible failure demo",
        "expected_failure": "tip over / top item slide",
        "description": (
            "A narrow yogurt tower with no wrap. It is intentionally bad so the "
            "replay shows the solver crossing the stability boundary."
        ),
        "stacks": [{"sku": "SKU-FD-002", "grid_row": 0, "grid_col": 1, "height": 8}],
        "wrap": "none",
        "env": "refrigerated",
        "body_temp": 2.0,
        "profile": {"target_speed_mps": 2.0, "accel_mps2": 6.0, "duration_s": 1.5},
    },
    {
        "slug": "top-heavy-surprise",
        "name": "Top-heavy surprise",
        "tag": "Load-shift trap",
        "expected_failure": "top item slide / load shift",
        "description": (
            "A tall heavy-cheese column rides without wrap. It looks compact, but "
            "the vertical mass stack becomes the weak point during a hard conveyor "
            "start."
        ),
        "stacks": [{"sku": "SKU-MS-003", "grid_row": 0, "grid_col": 1, "height": 6}],
        "wrap": "none",
        "env": "refrigerated",
        "body_temp": 2.0,
        "profile": {"target_speed_mps": 1.2, "accel_mps2": 4.0, "duration_s": 2.0},
    },
    {
        "slug": "frozen-pallet-jerk-start",
        "name": "Frozen pallet jerk-start",
        "tag": "Pallet-slip failure",
        "expected_failure": "pallet slip",
        "description": (
            "The same dense frozen-meat footprint without wrap and with a sharper "
            "conveyor hit. Use it to see floor friction become the governing "
            "constraint."
        ),
        "stacks": [
            {"sku": "SKU-FM-001", "grid_row": r, "grid_col": c, "height": 4}
            for r in range(2) for c in range(3)
        ],
        "wrap": "none",
        "env": "frozen",
        "body_temp": -25.0,
        "profile": {"target_speed_mps": 1.5, "accel_mps2": 4.0, "duration_s": 1.5},
    },
    {
        "slug": "asymmetric-load",
        "name": "Asymmetric load",
        "tag": "Off-center tip test",
        "expected_failure": "offset-driven top item slide",
        "description": (
            "A low dairy stack counterbalances a much taller cheese column on the "
            "opposite side. The center-of-mass offset should be obvious before the "
            "replay proves it."
        ),
        "stacks": [
            {"sku": "SKU-FD-001", "grid_row": 0, "grid_col": 0, "height": 1},
            {"sku": "SKU-MS-003", "grid_row": 0, "grid_col": 1, "height": 5},
        ],
        "wrap": "none",
        "env": "refrigerated",
        "body_temp": 2.0,
        "profile": {"target_speed_mps": 1.5, "accel_mps2": 5.0, "duration_s": 1.5},
    },
    {
        "slug": "random-scanner-feed",
        "name": "Random scanner feed",
        "tag": "Mock adapter payload",
        "expected_failure": "unknown until analyzed",
        "description": (
            "A scanner-like randomized pallet generated through the same RawInputs "
            "to PalletConfig path that a camera or WMS adapter would use."
        ),
        "random": True,
        "seed": 42,
        "profile": {"target_speed_mps": 0.85, "accel_mps2": 1.0, "duration_s": 2.5},
    },
]


def _build(spec: dict) -> Scenario:
    profile = ConveyorProfile(**spec["profile"])
    if spec.get("random"):
        adapter = MockRandomAdapter(
            seed=spec.get("seed"), anomaly_rate=0.10,
            min_layers=2, max_layers=5,
            min_items_per_layer=2, max_items_per_layer=5,
        )
        pallet = Configurator().build(adapter.read())
    else:
        pallet = build_from_stacks(
            [StackSpec(**s) for s in spec["stacks"]],
            pallet_id=spec["slug"],
            env=EnvCondition(spec["env"]),
            body_temp_c=float(spec["body_temp"]),
            wrap=WrapType(spec["wrap"]),
        )
    return Scenario(
        slug=spec["slug"], name=spec["name"], tag=spec["tag"],
        description=spec["description"], expected_failure=spec["expected_failure"],
        pallet=pallet, suggested_profile=profile,
    )


@cache
def all_scenarios() -> tuple[Scenario, ...]:
    """Every scenario, in curated display order."""
    return tuple(_build(s) for s in _SPECS)


def scenario_slugs() -> list[str]:
    return [s.slug for s in all_scenarios()]


def get_scenario(slug: str) -> Scenario:
    for s in all_scenarios():
        if s.slug == slug:
            return s
    raise KeyError(f"unknown scenario: {slug}")


def get_scenario_by_name(name: str) -> Scenario:
    for s in all_scenarios():
        if s.name == name:
            return s
    raise KeyError(f"unknown scenario: {name}")
