"""Tests for the stack-based pallet builder (`build_from_stacks`)."""

from __future__ import annotations

import pytest

from pallet_safety.configurator import StackSpec, build_from_stacks, compute_grid_shape
from pallet_safety.models import EnvCondition, PalletConfig, WrapType


def _stack(sku="SKU-FD-002", r=0, c=0, h=5):
    return StackSpec(sku=sku, grid_row=r, grid_col=c, height=h)


def test_empty_stacks_produces_empty_pallet():
    cfg = build_from_stacks([], pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    assert isinstance(cfg, PalletConfig)
    assert len(cfg.items) == 0


def test_single_stack_produces_right_item_count():
    cfg = build_from_stacks([_stack(h=5)], pallet_id="x",
                              env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    assert len(cfg.items) == 5


def test_items_stacked_vertically():
    cfg = build_from_stacks([_stack(h=4)], pallet_id="x",
                              env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    zs = [it.position[2] for it in cfg.items]
    assert zs == sorted(zs)  # strictly increasing
    assert len(set(zs)) == 4


def test_different_cells_different_xy():
    cfg = build_from_stacks([
        _stack(r=0, c=0, h=1),
        _stack(r=0, c=2, h=1),
        _stack(r=1, c=0, h=1),
    ], pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    positions = {(round(i.position[0], 3), round(i.position[1], 3)) for i in cfg.items}
    assert len(positions) == 3


def test_uneven_heights_different_top_z():
    cfg = build_from_stacks([
        _stack(r=0, c=0, h=6),
        _stack(r=0, c=1, h=3),
        _stack(r=0, c=2, h=4),
    ], pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    # Stack heights differ → top-z varies across cells
    tops_by_col = {}
    for it in cfg.items:
        col = round(it.position[0], 2)
        top = it.position[2] + it.dims_m[2]
        tops_by_col[col] = max(tops_by_col.get(col, 0), top)
    assert len(set(tops_by_col.values())) == 3  # 3 different stack tops


def test_mixed_sku_per_cell():
    cfg = build_from_stacks([
        _stack(sku="SKU-FM-001", r=0, c=0, h=3),
        _stack(sku="SKU-FD-003", r=0, c=1, h=3),
    ], pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    skus = {it.sku for it in cfg.items}
    assert skus == {"SKU-FM-001", "SKU-FD-003"}


def test_cell_positions_within_pallet_footprint():
    """Every item center should be within the pallet base dims (plus/minus half item)."""
    cfg = build_from_stacks([
        _stack(r=0, c=0, h=1), _stack(r=0, c=1, h=1), _stack(r=0, c=2, h=1),
        _stack(r=1, c=0, h=1), _stack(r=1, c=1, h=1), _stack(r=1, c=2, h=1),
    ], pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    for it in cfg.items:
        x, y, _ = it.position
        assert abs(x) <= cfg.base_dims_m[0] / 2.0
        assert abs(y) <= cfg.base_dims_m[1] / 2.0


def test_unknown_sku_raises():
    with pytest.raises(KeyError):
        build_from_stacks([_stack(sku="NOPE")], pallet_id="x",
                            env=EnvCondition.THAWED, body_temp_c=20.0)


def test_grid_shape_3x3_for_gma():
    """GMA pallets use 3x3 grid. Cells should be within pallet."""
    cfg = build_from_stacks(
        [_stack(r=r, c=c, h=1) for r in range(3) for c in range(3)],
        pallet_id="x", env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
        base_type="GMA", base_dims_m=(1.22, 1.02, 0.15), grid_shape=(3, 3),
    )
    assert len(cfg.items) == 9


def test_stackspec_validates_height():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        StackSpec(sku="x", grid_row=0, grid_col=0, height=0)
    with pytest.raises(ValidationError):
        StackSpec(sku="x", grid_row=0, grid_col=0, height=-1)


# ---- auto-grid (regression: SKU-DG-001 fill-all explosion) ----

def test_compute_grid_shape_for_oversized_item():
    """SKU-DG-001 (Flour Sacks 0.55x0.35) on EUR (1.2x0.8) only fits 2x2."""
    stacks = [StackSpec(sku="SKU-DG-001", grid_row=0, grid_col=0, height=1)]
    rows, cols = compute_grid_shape(stacks, (1.2, 0.8, 0.15))
    assert (rows, cols) == (2, 2)


def test_compute_grid_shape_for_small_item():
    """SKU-FV-003 (0.35x0.25) fits 3x3 on EUR."""
    stacks = [StackSpec(sku="SKU-FV-003", grid_row=0, grid_col=0, height=1)]
    rows, cols = compute_grid_shape(stacks, (1.2, 0.8, 0.15))
    assert (rows, cols) == (3, 3)


def test_compute_grid_shape_uses_largest_item_in_mixed_pallet():
    stacks = [
        StackSpec(sku="SKU-FV-003", grid_row=0, grid_col=0, height=1),  # small
        StackSpec(sku="SKU-DG-001", grid_row=0, grid_col=1, height=1),  # large
    ]
    rows, cols = compute_grid_shape(stacks, (1.2, 0.8, 0.15))
    # Should pick the smaller grid that fits the LARGER item
    assert (rows, cols) == (2, 2)


def test_build_from_stacks_skips_out_of_grid_stacks():
    """Regression: user supplies 6 stacks for 0.55m item; auto-grid only fits 4.
    The other 2 must be skipped, NOT clamped on top of others (which would
    cause MuJoCo to violently expel overlapping items)."""
    stacks = [
        StackSpec(sku="SKU-DG-001", grid_row=r, grid_col=c, height=5)
        for r in range(2) for c in range(3)
    ]
    cfg = build_from_stacks(stacks, pallet_id="x",
                              env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    # 4 cells × 5 high = 20 items (not 30)
    assert len(cfg.items) == 20
    # No two items at the same (x, y, z) — the bug we're protecting against
    positions = {(round(i.position[0], 3), round(i.position[1], 3), round(i.position[2], 3))
                 for i in cfg.items}
    assert len(positions) == 20


def test_build_from_stacks_no_overlap_for_oversized_items():
    """No item should overlap any other in the X-Y plane at the same Z."""
    stacks = [StackSpec(sku="SKU-DG-001", grid_row=r, grid_col=c, height=1)
              for r in range(2) for c in range(2)]
    cfg = build_from_stacks(stacks, pallet_id="x",
                              env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    for i, a in enumerate(cfg.items):
        for b in cfg.items[i + 1:]:
            if abs(a.position[2] - b.position[2]) < 1e-3:
                # Same layer — check footprint overlap
                ax_lo, ax_hi = a.position[0] - a.dims_m[0]/2, a.position[0] + a.dims_m[0]/2
                bx_lo, bx_hi = b.position[0] - b.dims_m[0]/2, b.position[0] + b.dims_m[0]/2
                ay_lo, ay_hi = a.position[1] - a.dims_m[1]/2, a.position[1] + a.dims_m[1]/2
                by_lo, by_hi = b.position[1] - b.dims_m[1]/2, b.position[1] + b.dims_m[1]/2
                x_overlap = max(0, min(ax_hi, bx_hi) - max(ax_lo, bx_lo))
                y_overlap = max(0, min(ay_hi, by_hi) - max(ay_lo, by_lo))
                assert x_overlap == 0 or y_overlap == 0, \
                    f"items overlap: {a.position} vs {b.position}"
