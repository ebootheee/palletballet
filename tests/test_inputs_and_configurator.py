"""Tests for input adapters and the configurator.

The whole pluggable-input pattern hinges on these contracts:
  - RawInputs is the canonical wire format
  - Any InputAdapter produces validatable RawInputs
  - Configurator.build() always produces a valid PalletConfig
"""

from __future__ import annotations

import pytest

from pallet_safety.catalog import all_skus
from pallet_safety.configurator import Configurator
from pallet_safety.inputs import MockRandomAdapter
from pallet_safety.inputs.base import RawInputs, StackPattern, VisionLayout
from pallet_safety.models import EnvCondition, PalletConfig


# ---- adapter contract ----

def test_mock_adapter_implements_interface():
    a = MockRandomAdapter(seed=1)
    assert hasattr(a, "read")
    raw = a.read()
    assert isinstance(raw, RawInputs)


def test_mock_adapter_seeded_is_deterministic():
    a = MockRandomAdapter(seed=42)
    b = MockRandomAdapter(seed=42)
    for _ in range(10):
        ra = a.read()
        rb = b.read()
        assert ra == rb


def test_mock_adapter_unseeded_varies():
    a = MockRandomAdapter(seed=1)
    b = MockRandomAdapter(seed=2)
    ra = a.read()
    rb = b.read()
    assert ra != rb


def test_mock_adapter_skus_in_catalog():
    a = MockRandomAdapter(seed=7)
    catalog_skus = set(all_skus())
    for _ in range(20):
        raw = a.read()
        for sku in raw.barcode_skus:
            assert sku in catalog_skus


def test_mock_adapter_temp_matches_env():
    a = MockRandomAdapter(seed=11)
    for _ in range(50):
        raw = a.read()
        if raw.env == EnvCondition.FROZEN:
            assert raw.body_temp_c <= -10.0
        elif raw.env == EnvCondition.THAWED:
            assert raw.body_temp_c >= 10.0


def test_mock_adapter_transitioning_has_recent_change():
    a = MockRandomAdapter(seed=3, anomaly_rate=0.0)
    found_transition = False
    for _ in range(200):
        raw = a.read()
        if raw.env == EnvCondition.TRANSITIONING:
            found_transition = True
            assert raw.seconds_since_temp_change <= 300
    # At default 0.10 weight, in 200 samples we expect ~20 transitions
    assert found_transition, "transitioning env never sampled in 200 draws"


def test_mock_adapter_anomaly_rate_zero_means_clean():
    a = MockRandomAdapter(seed=9, anomaly_rate=0.0)
    for _ in range(20):
        raw = a.read()
        assert raw.vision.lean_angle_deg == 0.0
        assert raw.vision.max_overhang_m == 0.0


def test_mock_adapter_anomaly_rate_one_always_anomalous():
    a = MockRandomAdapter(seed=9, anomaly_rate=1.0)
    for _ in range(20):
        raw = a.read()
        assert raw.vision.lean_angle_deg > 0.0 or raw.vision.max_overhang_m > 0.0


# ---- configurator ----

def _raw(layers=2, items_per_layer=2, pattern=StackPattern.BRICK,
         env=EnvCondition.REFRIGERATED, temp=2.0, sku="SKU-FD-002"):
    return RawInputs(
        barcode_skus=[sku] * (layers * items_per_layer),
        vision=VisionLayout(pattern=pattern, layers=layers, items_per_layer=items_per_layer),
        env=env, body_temp_c=temp,
    )


def test_configurator_produces_valid_pallet_config():
    raw = _raw()
    cfg = Configurator().build(raw)
    assert isinstance(cfg, PalletConfig)
    assert len(cfg.items) == 4


def test_configurator_empty_skus_produces_empty_pallet():
    raw = RawInputs(
        barcode_skus=[],
        vision=VisionLayout(pattern=StackPattern.COLUMN, layers=0, items_per_layer=0),
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    cfg = Configurator().build(raw)
    assert len(cfg.items) == 0


def test_configurator_column_stacks_vertically():
    raw = _raw(layers=3, items_per_layer=1, pattern=StackPattern.COLUMN)
    cfg = Configurator().build(raw)
    z_positions = [item.position[2] for item in cfg.items]
    # Should be strictly increasing
    assert z_positions == sorted(z_positions)
    assert len(set(z_positions)) == 3


def test_configurator_brick_offsets_alternating_layers():
    raw = _raw(layers=2, items_per_layer=4, pattern=StackPattern.BRICK)
    cfg = Configurator().build(raw)
    layer1_xs = sorted({i.position[0] for i in cfg.items[:4]})
    layer2_xs = sorted({i.position[0] for i in cfg.items[4:]})
    # Brick pattern: layer 2 should be shifted in x relative to layer 1
    assert layer1_xs != layer2_xs


def test_configurator_pinwheel_rotates_alternating_layers():
    raw = _raw(layers=3, items_per_layer=2, pattern=StackPattern.PINWHEEL)
    cfg = Configurator().build(raw)
    layer_orientations = [
        cfg.items[0].orientation_deg,
        cfg.items[2].orientation_deg,
        cfg.items[4].orientation_deg,
    ]
    assert layer_orientations[0] == 0.0
    assert layer_orientations[1] == 90.0
    assert layer_orientations[2] == 0.0


def test_configurator_propagates_env_and_temp():
    raw = _raw(env=EnvCondition.FROZEN, temp=-22.0)
    cfg = Configurator().build(raw)
    assert cfg.env == EnvCondition.FROZEN
    assert cfg.body_temp_c == -22.0


def test_configurator_handles_undersupplied_skus_gracefully():
    """Vision says 4 items but only 2 SKUs scanned — configurator should still build."""
    raw = RawInputs(
        barcode_skus=["SKU-FD-002", "SKU-FD-003"],
        vision=VisionLayout(pattern=StackPattern.BRICK, layers=2, items_per_layer=2),
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    cfg = Configurator().build(raw)
    assert len(cfg.items) == 4


def test_configurator_unknown_sku_raises():
    raw = RawInputs(
        barcode_skus=["SKU-DOES-NOT-EXIST"],
        vision=VisionLayout(pattern=StackPattern.COLUMN, layers=1, items_per_layer=1),
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    with pytest.raises(KeyError):
        Configurator().build(raw)


def test_end_to_end_mock_adapter_through_configurator():
    """Pluggability check: any adapter → Configurator → valid PalletConfig."""
    adapter = MockRandomAdapter(seed=123, anomaly_rate=0.0)
    configurator = Configurator()
    for _ in range(30):
        raw = adapter.read()
        cfg = configurator.build(raw)
        assert cfg.total_mass_kg > cfg.base_mass_kg  # at least one item present
        # Also verifies it serializes
        cfg.model_dump_json()
