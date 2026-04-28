"""Tests for pallet_safety.models — the canonical domain types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pallet_safety.models import (
    EnvCondition,
    FragilityClass,
    Item,
    PalletConfig,
    WrapType,
)


def _item(weight=10.0, dims=(0.4, 0.3, 0.2), pos=(0.0, 0.0, 0.15), sku="X-1"):
    return Item(sku=sku, weight_kg=weight, dims_m=dims, position=pos)


def test_item_rejects_negative_weight():
    with pytest.raises(ValidationError):
        Item(sku="x", weight_kg=-1.0, dims_m=(0.1, 0.1, 0.1), position=(0, 0, 0))


def test_item_rejects_zero_dims():
    with pytest.raises(ValidationError):
        Item(sku="x", weight_kg=1.0, dims_m=(0.0, 0.1, 0.1), position=(0, 0, 0))


def test_item_rejects_oversize_weight():
    with pytest.raises(ValidationError):
        Item(sku="x", weight_kg=600.0, dims_m=(0.5, 0.5, 0.5), position=(0, 0, 0))


def test_item_center_of_mass_is_geometric_center():
    item = _item(dims=(0.4, 0.3, 0.2), pos=(0.5, 0.2, 0.15))
    assert item.center_of_mass == (0.5, 0.2, 0.25)  # z = pos_z + height/2


def test_pallet_total_mass_includes_base():
    p = PalletConfig(
        pallet_id="P1",
        items=[_item(weight=10), _item(weight=15, sku="X-2")],
        env=EnvCondition.REFRIGERATED,
        body_temp_c=2.0,
        base_mass_kg=25.0,
    )
    assert p.total_mass_kg == 50.0


def test_pallet_empty_items_is_valid():
    p = PalletConfig(
        pallet_id="empty",
        items=[],
        env=EnvCondition.THAWED,
        body_temp_c=20.0,
    )
    assert p.total_mass_kg == p.base_mass_kg
    assert p.stack_height_m == p.base_dims_m[2]


def test_composite_com_centered_for_symmetric_load():
    # Two identical items placed symmetrically about origin in x
    p = PalletConfig(
        pallet_id="sym",
        items=[
            _item(weight=10, pos=(-0.3, 0.0, 0.15), sku="A"),
            _item(weight=10, pos=(+0.3, 0.0, 0.15), sku="B"),
        ],
        env=EnvCondition.REFRIGERATED,
        body_temp_c=2.0,
    )
    cx, cy, cz = p.composite_com_m
    assert abs(cx) < 1e-9
    assert abs(cy) < 1e-9
    assert cz > 0  # above the pallet base


def test_composite_com_shifts_with_asymmetry():
    p = PalletConfig(
        pallet_id="asym",
        items=[
            _item(weight=10, pos=(-0.3, 0.0, 0.15), sku="A"),
            _item(weight=30, pos=(+0.3, 0.0, 0.15), sku="B"),  # 3x heavier
        ],
        env=EnvCondition.REFRIGERATED,
        body_temp_c=2.0,
    )
    cx, _, _ = p.composite_com_m
    # CoM should be biased toward the heavier side (+x)
    assert cx > 0.05


def test_composite_com_matches_manual_calc():
    # Single 10kg item at z=0.5, 25kg base centered at z=0.075
    p = PalletConfig(
        pallet_id="manual",
        items=[_item(weight=10, dims=(0.4, 0.3, 0.2), pos=(0, 0, 0.5))],
        env=EnvCondition.THAWED,
        body_temp_c=20.0,
        base_mass_kg=25.0,
        base_dims_m=(1.2, 0.8, 0.15),
    )
    expected_cz = (25.0 * 0.075 + 10.0 * 0.6) / 35.0
    _, _, cz = p.composite_com_m
    assert abs(cz - expected_cz) < 1e-9


def test_stack_height_uses_max_top():
    p = PalletConfig(
        pallet_id="tall",
        items=[
            _item(dims=(0.3, 0.3, 0.2), pos=(0, 0, 0.15)),    # top at 0.35
            _item(dims=(0.3, 0.3, 0.4), pos=(0.3, 0, 0.35), sku="x2"),  # top at 0.75
        ],
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    assert p.stack_height_m == pytest.approx(0.75)


def test_overhang_zero_when_within_footprint():
    p = PalletConfig(
        pallet_id="ok",
        items=[_item(dims=(0.4, 0.3, 0.2), pos=(0, 0, 0.15))],  # well within 1.2 x 0.8
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    assert p.overhang_m <= 0.0


def test_overhang_detects_overhanging_item():
    # Pallet is 1.2m long in X; item placed at +0.5, 0.4m long → extends to +0.7
    p = PalletConfig(
        pallet_id="hang",
        items=[_item(dims=(0.4, 0.3, 0.2), pos=(0.5, 0, 0.15))],
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    # half pallet = 0.6m; item extent = 0.5+0.2 = 0.7m; overhang = 0.1
    assert p.overhang_m == pytest.approx(0.1)


def test_pallet_json_roundtrip():
    p = PalletConfig(
        pallet_id="trip",
        items=[_item(), _item(sku="x2", pos=(0, 0, 0.4))],
        env=EnvCondition.FROZEN,
        body_temp_c=-22.0,
        wrap=WrapType.STRETCH,
    )
    blob = p.model_dump_json()
    restored = PalletConfig.model_validate_json(blob)
    assert restored == p


def test_temp_bounds():
    with pytest.raises(ValidationError):
        PalletConfig(
            pallet_id="cold", items=[], env=EnvCondition.FROZEN, body_temp_c=-100.0,
        )
    with pytest.raises(ValidationError):
        PalletConfig(
            pallet_id="hot", items=[], env=EnvCondition.THAWED, body_temp_c=100.0,
        )


def test_fragility_default_is_rigid():
    item = _item()
    assert item.fragility == FragilityClass.RIGID
