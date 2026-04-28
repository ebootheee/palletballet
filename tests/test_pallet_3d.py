"""Regression tests for pallet 3D rendering.

This is here because the original Mesh3d face indices had degenerate triangles
and missing faces, causing visibly broken rendering. This test sweeps the
geometry by inspecting the trace data to make sure every triangle is valid.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from pallet_safety.models import EnvCondition, Item, PalletConfig
from pallet_safety.solver import ConveyorProfile, simulate
from pallet_safety.viz.pallet_3d import _box_mesh, animate_trace, render


def _item(weight=10.0, dims=(0.4, 0.3, 0.2), pos=(0.0, 0.0, 0.15), sku="X"):
    return Item(sku=sku, weight_kg=weight, dims_m=dims, position=pos)


def test_box_mesh_has_8_vertices_12_triangles():
    box = _box_mesh(0, 0, 0, 1, 1, 1, color="red", name="t")
    assert len(box.x) == 8
    assert len(box.y) == 8
    assert len(box.z) == 8
    assert len(box.i) == 12
    assert len(box.j) == 12
    assert len(box.k) == 12


def test_no_degenerate_triangles():
    """Every triangle must have 3 distinct vertices (no i==j, j==k, i==k)."""
    box = _box_mesh(0, 0, 0, 2, 1.5, 0.8, color="red", name="t")
    for tri_idx, (a, b, c) in enumerate(zip(box.i, box.j, box.k, strict=True)):
        assert len({a, b, c}) == 3, f"degenerate triangle {tri_idx}: ({a},{b},{c})"


def test_all_vertices_used():
    """All 8 vertices of the box must appear in some triangle."""
    box = _box_mesh(0, 0, 0, 1, 1, 1, color="red", name="t")
    used = set(box.i) | set(box.j) | set(box.k)
    assert used == set(range(8)), f"unused vertices: {set(range(8)) - used}"


def test_each_face_covered_by_two_triangles():
    """A box has 6 faces. Each face should contribute exactly 2 triangles.

    Identify a face by the constant axis: for each triangle, find which axis
    is constant across all 3 vertices — that's the face it belongs to.
    """
    box = _box_mesh(0, 0, 0, 1, 1, 1, color="red", name="t")
    # The 8 vertices, indexed
    verts = list(zip(box.x, box.y, box.z, strict=True))
    face_counts: dict[tuple[str, float], int] = {}
    for a, b, c in zip(box.i, box.j, box.k, strict=True):
        va, vb, vc = verts[a], verts[b], verts[c]
        for axis_idx, axis_name in enumerate("xyz"):
            if va[axis_idx] == vb[axis_idx] == vc[axis_idx]:
                key = (axis_name, va[axis_idx])
                face_counts[key] = face_counts.get(key, 0) + 1
                break
    assert len(face_counts) == 6, f"expected 6 distinct faces, got {face_counts}"
    for face, count in face_counts.items():
        assert count == 2, f"face {face} has {count} triangles (expected 2)"


def test_render_produces_figure():
    cfg = PalletConfig(
        pallet_id="test", items=[_item()],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    fig = render(cfg)
    assert isinstance(fig, go.Figure)


def test_render_includes_one_trace_per_item_plus_base_plus_com():
    cfg = PalletConfig(
        pallet_id="multi",
        items=[_item(sku="A"), _item(sku="B", pos=(0.3, 0, 0.15))],
        env=EnvCondition.REFRIGERATED, body_temp_c=2.0,
    )
    fig = render(cfg, show_com=True)
    # 1 base + 2 items + 1 CoM marker = 4 traces
    assert len(fig.data) == 4


def test_render_empty_pallet_only_base():
    cfg = PalletConfig(
        pallet_id="empty", items=[],
        env=EnvCondition.THAWED, body_temp_c=20.0,
    )
    fig = render(cfg, show_com=True)
    # base + CoM (no items)
    assert len(fig.data) == 2


# ---- animation ----

def _short_trace(n_items: int = 2):
    items = [_item(sku=f"X-{i}", pos=(0, 0, 0.15 + i * 0.2)) for i in range(n_items)]
    cfg = PalletConfig(pallet_id="anim", items=items,
                        env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    return simulate(cfg, ConveyorProfile(target_speed_mps=0.5, accel_mps2=0.5, duration_s=1.0))


def test_animate_trace_returns_figure():
    trace = _short_trace(2)
    fig = animate_trace(trace, max_frames=10)
    assert isinstance(fig, go.Figure)


def test_animate_trace_traces_per_frame_match_initial():
    """Every frame must have the same number of traces as the initial figure
    or plotly's frame swap fails silently."""
    trace = _short_trace(3)
    fig = animate_trace(trace, max_frames=15)
    initial_count = len(fig.data)
    for f in fig.frames:
        assert len(f.data) == initial_count


def test_animate_trace_one_trace_per_body():
    """Pallet base + N items = N+1 box meshes per frame."""
    trace = _short_trace(4)
    fig = animate_trace(trace, max_frames=8)
    assert len(fig.data) == 5  # 1 base + 4 items


def test_animate_trace_has_play_pause_buttons():
    trace = _short_trace(2)
    fig = animate_trace(trace, max_frames=10)
    btns = fig.layout.updatemenus[0].buttons
    labels = [b.label for b in btns]
    assert any("Play" in l for l in labels)
    assert any("Pause" in l for l in labels)


def test_animate_trace_has_slider():
    trace = _short_trace(2)
    fig = animate_trace(trace, max_frames=10)
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == 10


def test_animate_trace_empty_returns_empty_figure():
    """Trace with 0 items should produce a degenerate figure, not crash."""
    cfg = PalletConfig(pallet_id="empty", items=[],
                        env=EnvCondition.REFRIGERATED, body_temp_c=2.0)
    trace = simulate(cfg, ConveyorProfile(target_speed_mps=0.5, duration_s=0.5))
    fig = animate_trace(trace)
    assert len(fig.data) == 0


def test_animate_trace_loop_duplicates_frames():
    trace = _short_trace(2)
    fig_one = animate_trace(trace, max_frames=10, loop_cycles=1)
    fig_loop = animate_trace(trace, max_frames=10, loop_cycles=3)
    assert len(fig_loop.frames) == 3 * len(fig_one.frames)


def test_animate_trace_frame_names_unique_when_looping():
    """Plotly requires distinct frame names — test the cycle prefix works."""
    trace = _short_trace(2)
    fig = animate_trace(trace, max_frames=10, loop_cycles=4)
    names = [f.name for f in fig.frames]
    assert len(set(names)) == len(names)


def test_box_mesh_rotation_changes_vertices():
    """A rotated box should have different vertex coordinates than an unrotated one."""
    import numpy as np
    R_90deg_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    static = _box_mesh(0, 0, 0, 1, 0.5, 0.3, color="red", name="s")
    rotated = _box_mesh(0, 0, 0, 1, 0.5, 0.3, color="red", name="r", rotation=R_90deg_z)
    # 90° about Z swaps X and Y extents
    assert max(static.x) > max(rotated.x)  # rotated box is narrower in X
    assert max(rotated.y) > max(static.y)  # and wider in Y


def test_animate_trace_uses_item_quat_when_available():
    """Verify item rotations propagate into the animation frame meshes."""
    import numpy as np
    trace = _short_trace(2)
    # Manually inject a non-trivial rotation for one frame to verify the path
    trace.item_world_quat[0, 0] = np.array([0.7071, 0, 0.7071, 0])  # 90° about Y
    fig = animate_trace(trace, max_frames=2)
    # First mesh = pallet base; second = item 0 — vertices should not be axis-aligned
    item_mesh = fig.frames[0].data[1]
    # If unrotated, the item's z-extent would equal item dim Z.
    # With a 90° Y rotation, X and Z dims swap in world frame.
    z_extent = max(item_mesh.z) - min(item_mesh.z)
    x_extent = max(item_mesh.x) - min(item_mesh.x)
    # Item dims were (0.4, 0.3, 0.2). After 90° Y: x_extent should be ~0.2, z_extent ~0.4
    assert z_extent > x_extent


def test_animate_trace_follow_pallet_renders_base_at_origin():
    """In follow mode, the pallet base x/y should be near 0 across frames."""
    trace = _short_trace(2)
    fig = animate_trace(trace, max_frames=10, follow_pallet=True)
    # First trace in each frame is the pallet base
    for f in fig.frames:
        base = f.data[0]
        # mesh xs are 8 vertices around the base center; mean should be ~0
        cx_mean = sum(base.x) / len(base.x)
        cy_mean = sum(base.y) / len(base.y)
        assert abs(cx_mean) < 1e-6
        assert abs(cy_mean) < 1e-6


@pytest.mark.parametrize("dims,pos", [
    ((1.0, 1.0, 1.0), (0, 0, 0)),
    ((0.4, 0.3, 0.2), (0.5, -0.3, 0.15)),
    ((2.0, 0.1, 5.0), (-1, 1, 2)),
])
def test_box_centered_correctly(dims, pos):
    """Box vertices must be symmetric around the given center."""
    cx, cy, cz = pos
    lx, ly, lz = dims
    box = _box_mesh(cx, cy, cz, lx, ly, lz, color="red", name="x")
    assert min(box.x) == pytest.approx(cx - lx / 2)
    assert max(box.x) == pytest.approx(cx + lx / 2)
    assert min(box.y) == pytest.approx(cy - ly / 2)
    assert max(box.y) == pytest.approx(cy + ly / 2)
    assert min(box.z) == pytest.approx(cz - lz / 2)
    assert max(box.z) == pytest.approx(cz + lz / 2)
