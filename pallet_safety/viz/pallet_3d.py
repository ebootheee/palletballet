"""Render a PalletConfig as an interactive plotly 3D scene.

Two entry points:
  - `render(cfg)`: static snapshot of a pallet
  - `animate_trace(trace)`: plotly animation replaying a SimulationTrace,
    with play/pause and a scrub slider
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..models import FragilityClass, PalletConfig

_FRAGILITY_COLORS = {
    FragilityClass.RIGID: "rgb(70, 156, 184)",
    FragilityClass.SEMI_RIGID: "rgb(234, 174, 70)",
    FragilityClass.DEFORMABLE: "rgb(218, 86, 83)",
}

_BASE_COLOR = "rgb(96, 70, 44)"
_BELT_COLOR = "rgb(31, 38, 39)"
_BELT_EDGE_COLOR = "rgb(204, 142, 36)"
_BAY_LINE = "rgba(245, 241, 231, 0.28)"


def _quat_to_R(q) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _box_mesh(
    cx: float, cy: float, cz: float,
    lx: float, ly: float, lz: float,
    color: str, name: str, opacity: float = 0.95,
    rotation: np.ndarray | None = None,
) -> go.Mesh3d:
    """Single box from center (cx,cy,cz) and full-extents (lx,ly,lz).

    Vertex layout (in box-local frame, before rotation):
        0: (-x, -y, -z)  1: (+x, -y, -z)  2: (+x, +y, -z)  3: (-x, +y, -z)
        4: (-x, -y, +z)  5: (+x, -y, +z)  6: (+x, +y, +z)  7: (-x, +y, +z)
    Each face = 2 triangles, consistent CCW winding from outside.
    `rotation` (3x3 R) is applied to vertex offsets before translation, so the
    box can be tipped, tumbled, etc.
    """
    hx, hy, hz = lx / 2, ly / 2, lz / 2
    offsets = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz],  [hx, -hy, hz],  [hx, hy, hz],  [-hx, hy, hz],
    ])
    if rotation is not None:
        offsets = offsets @ rotation.T  # R @ v for each row v
    center = np.array([cx, cy, cz])
    verts = offsets + center
    xs, ys, zs = verts[:, 0].tolist(), verts[:, 1].tolist(), verts[:, 2].tolist()
    # 6 faces * 2 triangles = 12 triangles, all 8 vertices used, no degenerates.
    #          -Z bot      +Z top      -Y front    +Y back     -X left     +X right
    i = [0, 0,   4, 4,   0, 0,   3, 3,   0, 0,   1, 1]
    j = [1, 2,   6, 7,   5, 4,   2, 6,   3, 7,   5, 6]
    k = [2, 3,   5, 6,   1, 5,   6, 7,   7, 4,   6, 2]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=opacity, name=name,
        flatshading=True, hoverinfo="name",
        lighting=dict(ambient=0.45, diffuse=0.7, specular=0.2, roughness=0.6),
        lightposition=dict(x=2, y=-2, z=4),
    )


def _add_conveyor_bay(
    fig: go.Figure,
    *,
    x_min: float,
    x_max: float,
    y_half: float,
    z: float = -0.025,
    show_flow: bool = True,
) -> None:
    belt_width = y_half * 2.65
    belt_y0 = -belt_width / 2
    belt_y1 = belt_width / 2
    fig.add_trace(go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[belt_y0, belt_y0, belt_y1, belt_y1],
        z=[z, z, z, z],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=_BELT_COLOR,
        opacity=0.88,
        name="rubber conveyor belt",
        hoverinfo="skip",
        flatshading=True,
        lighting=dict(ambient=0.65, diffuse=0.55, specular=0.15, roughness=0.7),
    ))
    for y in [belt_y0, belt_y1]:
        fig.add_trace(go.Scatter3d(
            x=[x_min, x_max],
            y=[y, y],
            z=[z + 0.006, z + 0.006],
            mode="lines",
            line=dict(color=_BELT_EDGE_COLOR, width=7),
            name="belt edge rail",
            hoverinfo="skip",
            showlegend=False,
        ))

    stripe_count = 12
    stripe_xs = np.linspace(x_min + 0.18, x_max - 0.18, stripe_count)
    for idx, x in enumerate(stripe_xs):
        color = "rgba(245, 241, 231, 0.24)" if idx % 2 == 0 else "rgba(74, 169, 200, 0.25)"
        fig.add_trace(go.Scatter3d(
            x=[x, x + 0.10],
            y=[belt_y0 + 0.10, belt_y1 - 0.10],
            z=[z + 0.01, z + 0.01],
            mode="lines",
            line=dict(color=color, width=3),
            name="belt tread",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Cold-storage bay backdrop: two simple wall grids make the scene feel grounded.
    wall_y = belt_y1 + 0.22
    wall_z = [z, z, 1.95, 1.95]
    fig.add_trace(go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[wall_y, wall_y, wall_y, wall_y],
        z=wall_z,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color="rgb(22, 28, 29)",
        opacity=0.34,
        name="cold storage backdrop",
        hoverinfo="skip",
        flatshading=True,
        lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
    ))
    for x in np.linspace(x_min, x_max, 7):
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[wall_y + 0.002, wall_y + 0.002],
            z=[z, 1.95],
            mode="lines",
            line=dict(color=_BAY_LINE, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))
    for zz in np.linspace(0.25, 1.75, 5):
        fig.add_trace(go.Scatter3d(
            x=[x_min, x_max],
            y=[wall_y + 0.002, wall_y + 0.002],
            z=[zz, zz],
            mode="lines",
            line=dict(color=_BAY_LINE, width=1.5),
            hoverinfo="skip",
            showlegend=False,
        ))

    if show_flow:
        arrow_y = belt_y0 - 0.22
        for x in np.linspace(x_min + 0.28, x_max - 0.4, 5):
            fig.add_trace(go.Scatter3d(
                x=[x, x + 0.28, x + 0.20, x + 0.28, x + 0.20],
                y=[arrow_y, arrow_y, arrow_y + 0.07, arrow_y, arrow_y - 0.07],
                z=[0.045, 0.045, 0.045, 0.045, 0.045],
                mode="lines",
                line=dict(color="rgba(104, 211, 145, 0.78)", width=5),
                name="belt travel direction",
                hoverinfo="skip",
                showlegend=False,
            ))


def _bay_traces(
    *,
    x_min: float,
    x_max: float,
    y_half: float,
    z: float = -0.025,
    show_flow: bool = True,
) -> list[go.BaseTraceType]:
    bay_fig = go.Figure()
    _add_conveyor_bay(
        bay_fig,
        x_min=x_min,
        x_max=x_max,
        y_half=y_half,
        z=z,
        show_flow=show_flow,
    )
    return list(bay_fig.data)


def _add_com_projection(fig: go.Figure, config: PalletConfig) -> None:
    cx, cy, cz = config.composite_com_m
    fig.add_trace(go.Scatter3d(
        x=[cx, cx],
        y=[cy, cy],
        z=[0.0, cz],
        mode="lines",
        line=dict(color="rgba(242, 88, 82, 0.72)", width=5, dash="dot"),
        name="CoM projection",
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[cx],
        y=[cy],
        z=[0.012],
        mode="markers",
        marker=dict(size=7, color="rgba(242, 88, 82, 0.95)", symbol="circle"),
        name="CoM floor projection",
        hoverinfo="skip",
        showlegend=False,
    ))


def render(
    config: PalletConfig,
    show_com: bool = True,
    show_axes: bool = False,
    *,
    theme: str = "light",
    height: int = 520,
    show_floor: bool = False,
    show_legend: bool = True,
    show_bay: bool = True,
) -> go.Figure:
    """Build a plotly Figure for the given pallet."""
    fig = go.Figure()

    dark = theme == "dark"
    bl, bw, bh = config.base_dims_m
    x_extent = max(bl * 1.25, 1.8)
    y_half = max(bw * 0.75, 0.55)
    if show_bay and dark:
        _add_conveyor_bay(
            fig,
            x_min=-x_extent,
            x_max=x_extent,
            y_half=y_half,
            show_flow=True,
        )
    if show_floor:
        floor_x = [-bl / 2, bl / 2, bl / 2, -bl / 2, -bl / 2]
        floor_y = [-bw / 2, -bw / 2, bw / 2, bw / 2, -bw / 2]
        floor_z = [0.01, 0.01, 0.01, 0.01, 0.01]
        fig.add_trace(go.Scatter3d(
            x=floor_x, y=floor_y, z=floor_z,
            mode="lines",
            line=dict(color="rgba(236, 174, 70, 0.85)", width=6),
            name="safety footprint",
            hoverinfo="skip",
        ))

    # Pallet base
    fig.add_trace(_box_mesh(
        cx=0, cy=0, cz=bh / 2,
        lx=bl, ly=bw, lz=bh,
        color=_BASE_COLOR, name="pallet_base", opacity=0.95,
    ))

    # Items
    for i, item in enumerate(config.items):
        ix, iy, iz = item.position
        il, iw, ih = item.dims_m
        color = _FRAGILITY_COLORS[item.fragility]
        fig.add_trace(_box_mesh(
            cx=ix, cy=iy, cz=iz + ih / 2,
            lx=il, ly=iw, lz=ih,
            color=color, name=f"{item.sku} ({item.weight_kg:.1f}kg)",
        ))

    # Composite CoM marker
    if show_com:
        if dark:
            _add_com_projection(fig, config)
        cx, cy, cz = config.composite_com_m
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="markers",
            marker=dict(size=8, color="rgb(242, 88, 82)", symbol="diamond"),
            name=f"CoM ({cx:.2f}, {cy:.2f}, {cz:.2f})",
        ))

    # Axes hint
    if show_axes:
        for axis, color in [("x", "red"), ("y", "green"), ("z", "blue")]:
            xs = [0, 1.0 if axis == "x" else 0]
            ys = [0, 1.0 if axis == "y" else 0]
            zs = [0, 1.0 if axis == "z" else 0]
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=3), name=f"+{axis}",
            ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            xaxis=dict(
                range=[-x_extent, x_extent],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            yaxis=dict(
                range=[-max(y_half * 1.35, bw), max(y_half * 1.55, bw)],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            zaxis=dict(
                range=[-0.04 if dark else 0, max(config.stack_height_m + 0.35, 1.35)],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.55, y=-1.85, z=1.12), center=dict(x=0.02, y=0, z=0.14)),
        ),
        paper_bgcolor="rgba(0,0,0,0)" if dark else "white",
        plot_bgcolor="rgba(0,0,0,0)" if dark else "white",
        font=dict(color="#f5f1e7" if dark else "#2f3140"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=show_legend,
    )
    return fig


def animate_trace(
    trace,
    *,
    max_frames: int = 40,
    follow_pallet: bool = False,
    loop_cycles: int = 1,
    theme: str = "light",
    height: int = 560,
    show_bay: bool = False,
) -> go.Figure:
    """Plotly animation replaying a SimulationTrace.

    Args:
        trace: a `SimulationTrace` from `pallet_safety.solver.simulate`.
        max_frames: how many frames to show per cycle (uniformly subsampled).
        follow_pallet: if True, items render in the pallet's frame so motion
            appears as item-relative shift (good for spotting load shift / slide).
        loop_cycles: number of times to replay the simulation. Plotly has no
            native loop, so we duplicate frames. Set 1 to play once.
    """
    cfg = trace.config
    n_steps = trace.n_steps
    if n_steps < 2 or trace.n_items == 0:
        return go.Figure()

    indices = np.linspace(0, n_steps - 1, min(n_steps, max_frames), dtype=int)
    bl, bw, bh = cfg.base_dims_m

    def boxes_at(step_idx: int) -> list[go.Mesh3d]:
        ppos = trace.pallet_pos[step_idx]
        pquat = trace.pallet_quat[step_idx]
        pR = _quat_to_R(pquat)
        if follow_pallet:
            ppos_render = np.array([0.0, 0.0, ppos[2]])
            offset = ppos.copy()
            offset[2] = 0.0  # keep z; only subtract x,y
        else:
            ppos_render = ppos
            offset = np.zeros(3)

        boxes = [
            _box_mesh(
                cx=float(ppos_render[0]), cy=float(ppos_render[1]), cz=float(ppos_render[2]),
                lx=bl, ly=bw, lz=bh, rotation=pR,
                color=_BASE_COLOR, name="pallet base", opacity=0.95,
            )
        ]
        for k, item in enumerate(cfg.items):
            ipos = trace.item_world_pos[step_idx, k] - offset
            iquat = trace.item_world_quat[step_idx, k]
            iR = _quat_to_R(iquat)
            il, iw, ih = item.dims_m
            boxes.append(_box_mesh(
                cx=float(ipos[0]), cy=float(ipos[1]), cz=float(ipos[2]),
                lx=il, ly=iw, lz=ih, rotation=iR,
                color=_FRAGILITY_COLORS[item.fragility],
                name=f"{item.sku} #{k}",
            ))
        return boxes

    # Axis ranges chosen to fit the entire trajectory — including items that fly off
    if follow_pallet:
        x_min, x_max = -bl, bl
        y_min, y_max = -bw, bw
    else:
        # Use item world positions to bound view (they may fly past the pallet edge!)
        all_x = trace.item_world_pos[:, :, 0]
        all_y = trace.item_world_pos[:, :, 1]
        x_min = float(min(trace.pallet_pos[:, 0].min(), all_x.min()) - 0.5)
        x_max = float(max(trace.pallet_pos[:, 0].max(), all_x.max()) + 0.5)
        y_min = float(min(-bw / 2, all_y.min()) - 0.3)
        y_max = float(max(bw / 2, all_y.max()) + 0.3)
    dark = theme == "dark"
    if show_bay and dark:
        x_pad = max(bl * 1.25, 1.8)
        x_min = min(x_min, -x_pad)
        x_max = max(x_max, x_pad)
        y_half = max(bw * 0.75, 0.55, abs(y_min), abs(y_max))
        y_min = min(y_min, -y_half * 1.35)
        y_max = max(y_max, y_half * 1.55)
        static_traces = _bay_traces(
            x_min=x_min,
            x_max=x_max,
            y_half=y_half,
            show_flow=True,
        )
    else:
        static_traces = []
    initial = [*static_traces, *boxes_at(int(indices[0]))]
    base_frames = [
        go.Frame(data=[*static_traces, *boxes_at(int(i))], name=f"t={trace.times[i]:.2f}s")
        for i in indices
    ]
    # Loop trick: duplicate frames; each duplicate needs a unique name.
    if loop_cycles > 1:
        frames = []
        for c in range(loop_cycles):
            for fr in base_frames:
                frames.append(go.Frame(
                    data=fr.data, name=f"c{c}_{fr.name}",
                ))
    else:
        frames = base_frames
    tallest_item = max(it.dims_m[2] for it in cfg.items)
    z_max = float(trace.item_world_pos[:, :, 2].max() + max(0.3, tallest_item))

    fig = go.Figure(data=initial, frames=frames)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            xaxis=dict(
                range=[x_min, x_max],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            yaxis=dict(
                range=[y_min, y_max],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            zaxis=dict(
                range=[0, z_max],
                backgroundcolor="rgba(20, 21, 18, 0.92)" if dark else "white",
                gridcolor="rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.14)",
                zerolinecolor="rgba(236,174,70,0.35)" if dark else "rgba(0,0,0,0.25)",
                color="rgba(245,241,231,0.78)" if dark else None,
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        paper_bgcolor="rgba(0,0,0,0)" if dark else "white",
        plot_bgcolor="rgba(0,0,0,0)" if dark else "white",
        font=dict(color="#f5f1e7" if dark else "#2f3140"),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "direction": "right",
            "y": -0.06,
            "x": 0.02,
            "yanchor": "bottom",
            "xanchor": "left",
            "pad": {"r": 8, "t": 0, "b": 0},
            "font": {"size": 11, "color": "#f5f1e7" if dark else "#2f3140"},
            "bgcolor": "rgba(31, 32, 27, 0.96)" if dark else "rgba(255, 255, 255, 0.96)",
            "bordercolor": "rgba(245, 241, 231, 0.35)" if dark else "rgba(0, 0, 0, 0.25)",
            "borderwidth": 1,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 80, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "activebgcolor": "#eaae46",
            "bgcolor": "rgba(245, 241, 231, 0.12)" if dark else "rgba(0, 0, 0, 0.08)",
            "bordercolor": "rgba(245, 241, 231, 0.25)" if dark else "rgba(0, 0, 0, 0.18)",
            "borderwidth": 1,
            "currentvalue": {
                "prefix": "t = ",
                "xanchor": "left",
                "offset": 12,
                "font": {"size": 12, "color": "#f5f1e7" if dark else "#2f3140"},
            },
            "font": {"size": 10, "color": "#f5f1e7" if dark else "#2f3140"},
            "x": 0.02,
            "y": -0.18,
            "len": 0.96,
            "pad": {"t": 18, "b": 0},
            "steps": [
                {"args": [[fr.name],
                          {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 "label": fr.name.split("=")[-1] if "=" in fr.name else fr.name,
                 "method": "animate"}
                for fr in frames
            ],
        }],
        height=height,
        margin=dict(l=0, r=0, t=10, b=118),
    )
    return fig
