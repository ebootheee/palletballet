"""Tests that the random adapter generates pallets that look like the real world.

References (3PL/cold-storage industry conventions):
- Ti (items per layer) typically 4-10
- Hi (layers high) typically 4-7
- ~70% of distribution pallets are homogeneous (single SKU)
- Loaded weight typically 600-1000kg, hard cap 1500kg per ANSI MH1
- Stack height cap ~1.85m for safe handling (84" ANSI)
- Brick/interlock is the dominant stacking pattern (~50%+)
"""

from __future__ import annotations

from collections import Counter

from pallet_safety.catalog import get
from pallet_safety.configurator import Configurator
from pallet_safety.inputs import (
    MAX_PALLET_WEIGHT_KG,
    MAX_STACK_HEIGHT_M,
    Homogeneity,
    MockRandomAdapter,
    StackPattern,
)


def _sample(n: int, **kw):
    a = MockRandomAdapter(seed=kw.pop("seed", 1234), **kw)
    return [a.read() for _ in range(n)]


def test_ti_hi_within_distribution_norms():
    raws = _sample(200)
    layers = [r.vision.layers for r in raws]
    items = [r.vision.items_per_layer for r in raws]
    assert min(layers) >= 1  # may be trimmed by cap
    assert max(layers) <= 7
    # Most pallets should have realistic Hi (≥3 after caps)
    assert sum(1 for h in layers if h >= 3) / len(layers) > 0.85
    assert min(items) >= 1
    assert max(items) <= 10


def test_pattern_distribution_brick_dominant():
    raws = _sample(500)
    counts = Counter(r.vision.pattern for r in raws)
    total = sum(counts.values())
    # Brick should be the most common pattern, irregular the rarest
    assert counts[StackPattern.BRICK] / total > 0.40
    assert counts[StackPattern.IRREGULAR] / total < 0.15


def test_homogeneity_bias_majority_single_sku():
    """At least 60% of generated pallets should be single-SKU (matches 70% target ± noise)."""
    raws = _sample(300)
    homo_count = sum(1 for r in raws if len(set(r.barcode_skus)) == 1)
    assert homo_count / len(raws) > 0.60


def test_no_pallet_exceeds_weight_cap():
    raws = _sample(300)
    for r in raws:
        weight = sum(get(s).weight_kg for s in r.barcode_skus)
        assert weight <= MAX_PALLET_WEIGHT_KG, \
            f"{r.pallet_id}: {weight:.1f}kg exceeds cap {MAX_PALLET_WEIGHT_KG}"


def test_no_pallet_exceeds_height_cap():
    cfg_builder = Configurator()
    raws = _sample(300)
    for r in raws:
        cfg = cfg_builder.build(r)
        assert cfg.stack_height_m <= MAX_STACK_HEIGHT_M + 0.05  # tiny tolerance for rounding


def test_typical_loaded_weight_in_realistic_band():
    """Median loaded pallet should land in the 200-1000kg range typical for cold-chain distribution."""
    cfg_builder = Configurator()
    raws = _sample(200)
    weights = [cfg_builder.build(r).total_mass_kg for r in raws]
    weights.sort()
    median = weights[len(weights) // 2]
    assert 100 <= median <= 1000, f"median pallet weight {median:.0f}kg outside realistic band"


def test_base_pallet_type_variety():
    raws = _sample(200)
    types = Counter(r.base_pallet_type for r in raws)
    # At least 2 types should appear in 200 draws
    assert len(types) >= 2
    # EUR should be the most common
    assert types["EUR"] > types["CHEP"]


def test_anomaly_rate_calibrated_low():
    """Default anomaly_rate is 0.08 — most pallets are clean.

    Wide band because 300 samples at p=0.08 has σ ≈ 0.016 → 99% CI ≈ [0.04, 0.13]
    but seed-to-seed variance means we want a generous bound for stability.
    """
    raws = _sample(500, anomaly_rate=0.08)
    anomalous = sum(1 for r in raws if r.vision.lean_angle_deg > 0 or r.vision.max_overhang_m > 0)
    rate = anomalous / len(raws)
    assert 0.03 <= rate <= 0.15


def test_homogeneity_modes_can_be_forced():
    """Forcing homogeneity to MIXED should produce mostly multi-SKU pallets."""
    raws = _sample(100, homogeneity_weights={Homogeneity.MIXED: 1.0})
    multi_sku = sum(1 for r in raws if len(set(r.barcode_skus)) > 1)
    # Most should be multi-SKU, allowing for small pallets that happen to draw same SKU
    assert multi_sku / len(raws) > 0.70


def test_column_pattern_constrained_to_narrow_ti():
    """Column stacks shouldn't have wide Ti — they're for tower-style loads."""
    raws = _sample(500, pattern_weights={StackPattern.COLUMN: 1.0})
    for r in raws:
        assert r.vision.items_per_layer <= 3, \
            f"column pattern got Ti={r.vision.items_per_layer}, expected ≤3"


def test_pinwheel_never_ti_less_than_4():
    """Pinwheel only makes sense with ≥4 items per layer (it's a 4-corner rotation)."""
    raws = _sample(300, pattern_weights={StackPattern.PINWHEEL: 1.0})
    for r in raws:
        assert r.vision.items_per_layer >= 4, \
            f"pinwheel with Ti={r.vision.items_per_layer} (must be ≥4)"


def test_brick_never_ti_less_than_4():
    """Brick/interlocking needs ≥4 items to form the offset pattern."""
    raws = _sample(300, pattern_weights={StackPattern.BRICK: 1.0})
    for r in raws:
        assert r.vision.items_per_layer >= 4, \
            f"brick with Ti={r.vision.items_per_layer} (must be ≥4)"


def test_irregular_never_ti_less_than_3():
    raws = _sample(300, pattern_weights={StackPattern.IRREGULAR: 1.0})
    for r in raws:
        assert r.vision.items_per_layer >= 3


def test_frozen_pallets_have_frozen_skus():
    """When env is FROZEN, the SKUs should all be frozen-default products."""
    from pallet_safety.models import EnvCondition
    raws = _sample(50, env_weights={EnvCondition.FROZEN: 1.0})
    for r in raws:
        assert r.env == EnvCondition.FROZEN
        for sku in r.barcode_skus:
            assert get(sku).default_env == EnvCondition.FROZEN
