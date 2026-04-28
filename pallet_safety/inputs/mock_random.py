"""Seeded random adapter. Produces realistic-but-synthetic RawInputs.

Calibrated against published 3PL / cold-storage pallet-build conventions:
  - Ti (items per layer): typically 4–10
  - Hi (layers high): typically 4–7
  - 70% of distribution pallets are *homogeneous* (single SKU end-to-end),
    20% are layer-homogeneous, 10% are mixed.
  - Loaded weight typically 600–1000kg, hard cap 1500kg per pallet rating.
  - Stack height capped at ~1.8m for safe handling (ANSI 84" max).
  - Brick/interlock is the dominant pattern; pinwheel for cardboard cases;
    column for crushable goods; irregular is anomalous.

Used for development, batch sweeps, and CI. Same seed → identical output.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from enum import Enum

from ..catalog import all_skus, by_env, get
from ..models import EnvCondition
from .base import InputAdapter, RawInputs, StackPattern, VisionLayout


class Homogeneity(str, Enum):
    HOMOGENEOUS = "homogeneous"            # one SKU all the way
    LAYER_HOMOGENEOUS = "layer_homogeneous"  # one SKU per layer, varies across layers
    MIXED = "mixed"                          # truly mixed (rare in real distribution)


# --- defaults calibrated to industry norms ---

DEFAULT_PATTERN_WEIGHTS = {
    StackPattern.BRICK: 0.55,    # interlocking, dominant for distribution
    StackPattern.COLUMN: 0.20,   # crushable / display goods
    StackPattern.PINWHEEL: 0.20, # cardboard cases for stability
    StackPattern.IRREGULAR: 0.05,
}

DEFAULT_HOMOGENEITY_WEIGHTS = {
    Homogeneity.HOMOGENEOUS: 0.70,
    Homogeneity.LAYER_HOMOGENEOUS: 0.20,
    Homogeneity.MIXED: 0.10,
}

DEFAULT_ENV_TEMP_RANGES = {
    EnvCondition.FROZEN: (-25.0, -18.0),
    EnvCondition.REFRIGERATED: (0.0, 5.0),
    EnvCondition.THAWED: (15.0, 22.0),
    EnvCondition.TRANSITIONING: (-2.0, 8.0),
}

DEFAULT_ENV_WEIGHTS = {
    EnvCondition.FROZEN: 0.45,
    EnvCondition.REFRIGERATED: 0.35,
    EnvCondition.THAWED: 0.10,
    EnvCondition.TRANSITIONING: 0.10,
}

DEFAULT_BASE_PALLET_WEIGHTS = {
    "EUR": 0.55,    # 1.2 × 0.8 m, common in EU and global cold chain
    "GMA": 0.35,    # 1.219 × 1.016 m, dominant in US grocery
    "CHEP": 0.10,   # plastic / pool, lower friction
}

# Hard safety caps from ANSI MH1 / ISO 6780 pallet ratings
MAX_PALLET_WEIGHT_KG = 1200.0
MAX_STACK_HEIGHT_M = 1.85


@dataclass
class MockRandomAdapter(InputAdapter):
    """Generates realistic random pallet RawInputs for development.

    Defaults reflect typical 3PL / cold-storage distribution practices.
    Override any field to bias toward edge cases.
    """

    seed: int | None = None
    min_layers: int = 3                        # bumped from 1; real Hi typically ≥ 3
    max_layers: int = 7
    min_items_per_layer: int = 4               # bumped from 1; real Ti typically ≥ 4
    max_items_per_layer: int = 10
    anomaly_rate: float = 0.08                 # 8% of real pallets show some defect
    pattern_weights: dict[StackPattern, float] = field(
        default_factory=lambda: dict(DEFAULT_PATTERN_WEIGHTS),
    )
    env_weights: dict[EnvCondition, float] = field(
        default_factory=lambda: dict(DEFAULT_ENV_WEIGHTS),
    )
    homogeneity_weights: dict[Homogeneity, float] = field(
        default_factory=lambda: dict(DEFAULT_HOMOGENEITY_WEIGHTS),
    )

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def read(self) -> RawInputs:
        env = self._weighted_choice(self.env_weights)
        pattern = self._weighted_choice(self.pattern_weights)
        homogeneity = self._weighted_choice(self.homogeneity_weights)
        base_type = self._weighted_choice(DEFAULT_BASE_PALLET_WEIGHTS)

        layers, items_per_layer = self._sample_ti_hi(env, pattern)

        is_anomaly = self._rng.random() < self.anomaly_rate
        lean_angle = self._rng.uniform(2.0, 8.0) if is_anomaly else 0.0
        overhang = self._rng.uniform(0.025, 0.10) if is_anomaly else 0.0

        # Pick SKUs respecting homogeneity
        skus_pool = self._skus_for_env(env) or all_skus()
        chosen = self._pick_skus(skus_pool, layers, items_per_layer, homogeneity)

        # Enforce weight + height caps by trimming layers if needed
        chosen, layers = self._enforce_caps(chosen, layers, items_per_layer)

        body_temp = self._rng.uniform(*DEFAULT_ENV_TEMP_RANGES[env])
        if env == EnvCondition.TRANSITIONING:
            sec_since = self._rng.uniform(30, 300)
        else:
            sec_since = self._rng.uniform(1800, 7200)

        vision = VisionLayout(
            pattern=pattern,
            layers=layers,
            items_per_layer=items_per_layer,
            lean_angle_deg=lean_angle,
            max_overhang_m=overhang,
        )
        return RawInputs(
            barcode_skus=chosen,
            vision=vision,
            env=env,
            body_temp_c=body_temp,
            seconds_since_temp_change=sec_since,
            pallet_id=f"P-{uuid.UUID(int=self._rng.getrandbits(128))!s:.8s}",
            base_pallet_type=base_type,
        )

    # ---- helpers ----

    # Minimum Ti per pattern — patterns only "work" at these counts.
    # Pinwheel is a 4-corner rotation pattern; single-item pinwheel is meaningless.
    # Brick needs ≥4 items to form the interlocking offset. Column can be Ti=1.
    _MIN_TI_BY_PATTERN = {
        StackPattern.COLUMN: 1,
        StackPattern.BRICK: 4,
        StackPattern.PINWHEEL: 4,
        StackPattern.IRREGULAR: 3,
    }

    def _sample_ti_hi(self, env: EnvCondition, pattern: StackPattern) -> tuple[int, int]:
        """Sample (Hi, Ti) constrained to what the requested pattern physically
        needs. Ti=1 pinwheel would be a twisted-tower monstrosity; we disallow it."""
        hard_min = self._MIN_TI_BY_PATTERN[pattern]
        ti_lo = max(hard_min, self.min_items_per_layer)
        ti_hi = max(ti_lo, self.max_items_per_layer)
        if pattern == StackPattern.COLUMN:
            ti_hi = min(ti_hi, 2)  # column is a tower — not wide
            ti_lo = min(ti_lo, ti_hi)
        ti = self._rng.randint(ti_lo, ti_hi)
        hi = self._rng.randint(self.min_layers, self.max_layers)
        return hi, ti

    def _pick_skus(
        self, pool: list[str], layers: int, items_per_layer: int, mode: Homogeneity,
    ) -> list[str]:
        n = layers * items_per_layer
        if not pool:
            return []
        if mode == Homogeneity.HOMOGENEOUS:
            sku = self._rng.choice(pool)
            return [sku] * n
        if mode == Homogeneity.LAYER_HOMOGENEOUS:
            out: list[str] = []
            for _ in range(layers):
                sku = self._rng.choice(pool)
                out.extend([sku] * items_per_layer)
            return out
        return [self._rng.choice(pool) for _ in range(n)]

    def _enforce_caps(
        self, skus: list[str], layers: int, items_per_layer: int,
    ) -> tuple[list[str], int]:
        """Trim layers from the top until the pallet is under weight + height caps.

        Real ops would refuse to build the pallet, but for sim-data generation
        we just produce a valid (under-cap) pallet representing what would
        actually arrive at the conveyor.
        """
        # Include the standard 0.15m pallet base in the height check so this
        # matches PalletConfig.stack_height_m (which sums from base origin).
        BASE_H = 0.15
        cur_layers = layers
        while cur_layers > 1:
            n = cur_layers * items_per_layer
            templates = [get(s) for s in skus[:n]]
            weight = sum(t.weight_kg for t in templates)
            stack_h = BASE_H + sum(
                max(t.dims_m[2] for t in templates[i*items_per_layer:(i+1)*items_per_layer])
                for i in range(cur_layers)
            )
            if weight <= MAX_PALLET_WEIGHT_KG and stack_h <= MAX_STACK_HEIGHT_M:
                break
            cur_layers -= 1
        return skus[: cur_layers * items_per_layer], cur_layers

    def _weighted_choice(self, weights: dict):
        total = sum(weights.values())
        r = self._rng.random() * total
        cum = 0.0
        for k, w in weights.items():
            cum += w
            if r <= cum:
                return k
        return list(weights)[-1]

    def _skus_for_env(self, env: EnvCondition) -> list[str]:
        return [t.sku for t in by_env(env)]
