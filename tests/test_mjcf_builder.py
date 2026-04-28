"""Tests for pallet_safety.mjcf_builder.

Strategy: build MJCF, load it in MuJoCo, then verify model topology, total
mass, and composite CoM match the PalletConfig (within 1mm / 0.5%).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from pallet_safety.mjcf_builder import build_mjcf, load_model
from pallet_safety.models import EnvCondition, Item, PalletConfig, WrapType


def _item(weight=10.0, dims=(0.4, 0.3, 0.2), pos=(0.0, 0.0, 0.15), sku="X"):
    return Item(sku=sku, weight_kg=weight, dims_m=dims, position=pos)


def _config_n_items(n: int, **kw) -> PalletConfig:
    items = []
    for i in range(n):
        # Stack items in a column going up
        items.append(_item(weight=10.0, pos=(0.0, 0.0, 0.15 + i * 0.2), sku=f"X-{i}"))
    return PalletConfig(
        pallet_id=f"P-{n}",
        items=items,
        env=EnvCondition.REFRIGERATED,
        body_temp_c=2.0,
        wrap=kw.get("wrap", WrapType.NONE),
    )


def _composite_com(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float, float]:
    """World-frame composite CoM of pallet + items (skip worldbody and conveyor)."""
    mujoco.mj_kinematics(model, data)
    total = 0.0
    accum = np.zeros(3)
    for i in range(1, model.nbody):
        name = model.body(i).name
        if name == "conveyor_body":
            continue
        m = float(model.body_mass[i])
        accum += m * data.xipos[i]
        total += m
    return tuple(accum / total)


def test_empty_pallet_loads():
    cfg = PalletConfig(pallet_id="empty", items=[], env=EnvCondition.THAWED, body_temp_c=20.0)
    model = load_model(cfg)
    # world + conveyor_body + pallet_base = 3 bodies (default actuated_conveyor=True)
    assert model.nbody == 3


def test_single_item_loads():
    cfg = _config_n_items(1)
    model = load_model(cfg)
    # world + conveyor_body + pallet_base + 1 item = 4 bodies
    assert model.nbody == 4


def test_n_items_topology():
    for n in [0, 1, 3, 5, 10]:
        cfg = _config_n_items(n)
        model = load_model(cfg)
        assert model.nbody == 3 + n  # world + conveyor + pallet_base + items


def test_total_mass_matches_config():
    cfg = _config_n_items(4)
    model = load_model(cfg)
    # Sum body masses excluding world and conveyor
    sim_mass = 0.0
    for i in range(1, model.nbody):
        if model.body(i).name == "conveyor_body":
            continue
        sim_mass += float(model.body_mass[i])
    assert sim_mass == pytest.approx(cfg.total_mass_kg, abs=1e-6)


def test_static_conveyor_mode_excludes_actuator():
    cfg = _config_n_items(2)
    xml = build_mjcf(cfg, actuated_conveyor=False)
    assert "<actuator>" not in xml
    assert "conveyor_body" not in xml
    assert 'type="plane"' in xml


def test_actuated_conveyor_default():
    cfg = _config_n_items(2)
    xml = build_mjcf(cfg)
    assert "<actuator>" in xml
    assert "conveyor_motor" in xml
    assert "conveyor_slide" in xml


def test_composite_com_matches_config():
    cfg = _config_n_items(3)
    model = load_model(cfg)
    data = mujoco.MjData(model)
    sim_com = _composite_com(model, data)
    cfg_com = cfg.composite_com_m
    for i, axis in enumerate("xyz"):
        assert sim_com[i] == pytest.approx(cfg_com[i], abs=1e-3), \
            f"CoM-{axis}: sim={sim_com[i]:.4f} cfg={cfg_com[i]:.4f}"


def test_asymmetric_load_com_shifts_in_sim():
    """Heavy item on +X side → sim CoM has positive x."""
    cfg = PalletConfig(
        pallet_id="asym",
        items=[
            _item(weight=5.0, pos=(-0.3, 0.0, 0.15), sku="A"),
            _item(weight=40.0, pos=(0.3, 0.0, 0.15), sku="B"),
        ],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    model = load_model(cfg)
    data = mujoco.MjData(model)
    sim_com = _composite_com(model, data)
    assert sim_com[0] > 0.05


def test_friction_reflected_in_geom():
    """Frozen pallet should yield a noticeably lower geom friction value."""
    frozen = PalletConfig(
        pallet_id="cold", items=[_item()], env=EnvCondition.FROZEN, body_temp_c=-25.0,
    )
    thawed = PalletConfig(
        pallet_id="warm", items=[_item()], env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    fz_xml = build_mjcf(frozen)
    th_xml = build_mjcf(thawed)
    # Crude but sufficient: extract first friction value from the conveyor geom
    fz_fric = float(fz_xml.split('friction="')[1].split()[0])
    th_fric = float(th_xml.split('friction="')[1].split()[0])
    assert th_fric > fz_fric


def test_wrap_emits_equalities_when_stretchwrapped():
    cfg = _config_n_items(3, wrap=WrapType.STRETCH)
    xml = build_mjcf(cfg)
    assert "<equality>" in xml
    assert xml.count('<weld ') == 3


def test_wrap_none_emits_no_equalities():
    cfg = _config_n_items(3, wrap=WrapType.NONE)
    xml = build_mjcf(cfg)
    assert "<equality>" not in xml


def test_invalid_xml_would_fail_to_load():
    """If the builder produces malformed XML, MuJoCo will raise — this protects future edits."""
    cfg = _config_n_items(2)
    xml = build_mjcf(cfg)
    # Should load fine
    mujoco.MjModel.from_xml_string(xml)
    # Tamper and ensure failure (regression sentinel for future builder changes)
    bad = xml.replace("<mujoco", "<mujoc0")
    with pytest.raises(Exception):
        mujoco.MjModel.from_xml_string(bad)


def test_special_chars_in_pallet_id_escaped():
    cfg = PalletConfig(
        pallet_id="P&123<x>", items=[], env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    xml = build_mjcf(cfg)
    # XML should still parse
    mujoco.MjModel.from_xml_string(xml)


def test_large_pallet_loads_quickly():
    """20 items shouldn't take more than ~100ms to build + load."""
    import time
    cfg = _config_n_items(20)
    t0 = time.perf_counter()
    load_model(cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100, f"build+load took {elapsed_ms:.1f}ms"


def test_geometry_dimensions_round_trip():
    """A 0.4 x 0.3 x 0.2 item should have geom size 0.2 x 0.15 x 0.1 (half-extents)."""
    cfg = PalletConfig(
        pallet_id="dim",
        items=[_item(dims=(0.4, 0.3, 0.2), pos=(0, 0, 0.15))],
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    model = load_model(cfg)
    # Find the item geom (last one)
    item_geom_id = model.geom("item_0_geom").id
    size = model.geom_size[item_geom_id]
    assert size[0] == pytest.approx(0.2)
    assert size[1] == pytest.approx(0.15)
    assert size[2] == pytest.approx(0.1)
