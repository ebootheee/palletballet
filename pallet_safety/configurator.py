"""RawInputs → PalletConfig.

The Configurator is pure business logic. It looks up SKUs in the catalog,
computes physical item positions for the given stack pattern, and assembles
a validated PalletConfig.

Stack-pattern positioning uses the pallet base footprint (default EUR 1.2x0.8m)
and the *largest* item dimension as the cell size, which keeps small items
centered within their cell rather than packed edge-to-edge. Real warehouse
pallets vary; this is a reasonable first-order approximation suitable for
physics simulation. Refinement would come from depth-camera point clouds.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from .catalog import get as get_template
from .inputs.base import RawInputs, StackPattern, VisionLayout
from .models import EnvCondition, Item, PalletConfig, Vec3, WrapType


class StackSpec(BaseModel):
    """One column-stack on a pallet: an SKU placed at a grid cell, N items tall."""
    sku: str
    grid_row: int = Field(ge=0, le=5)
    grid_col: int = Field(ge=0, le=5)
    height: int = Field(gt=0, le=10)


def compute_grid_shape(
    stacks: list[StackSpec], base_dims_m: Vec3, fallback: tuple[int, int] = (2, 3),
) -> tuple[int, int]:
    """Smallest (rows, cols) grid such that every stack's item fits in one cell.

    For mixed-SKU pallets, the grid is sized for the LARGEST item across all
    stacks so every item fits without overlapping its neighbors.
    """
    if not stacks:
        return fallback
    max_l = max(get_template(s.sku).dims_m[0] for s in stacks)
    max_w = max(get_template(s.sku).dims_m[1] for s in stacks)
    cols = max(1, int(base_dims_m[0] // max_l))
    rows = max(1, int(base_dims_m[1] // max_w))
    return (rows, cols)


def build_from_stacks(
    stacks: list[StackSpec],
    *,
    pallet_id: str,
    env: EnvCondition,
    body_temp_c: float,
    wrap: WrapType = WrapType.STRETCH,
    base_type: str = "EUR",
    base_dims_m: Vec3 = (1.2, 0.8, 0.15),
    base_mass_kg: float = 25.0,
    grid_shape: tuple[int, int] | None = None,
    seconds_since_temp_change: float = 3600.0,
) -> PalletConfig:
    """Build a PalletConfig from a list of StackSpec.

    Each stack is a homogeneous column of items in one grid cell. Supports
    uneven heights across cells and mixed SKUs per cell.

    `grid_shape` defaults to one auto-computed from the items' actual sizes —
    this guarantees adjacent items can never penetrate each other (which would
    cause MuJoCo to violently expel them at sim start). Pass an explicit
    grid_shape only when you know what you're doing (e.g., regression tests).
    """
    if grid_shape is None:
        grid_shape = compute_grid_shape(stacks, base_dims_m)
    rows, cols = grid_shape
    cell_l = base_dims_m[0] / cols
    cell_w = base_dims_m[1] / rows
    items: list[Item] = []
    occupied: set[tuple[int, int]] = set()
    for spec in stacks:
        tpl = get_template(spec.sku)
        # Skip stacks whose cell index is outside the (auto-sized) grid OR
        # would land on an already-occupied cell. Both conditions guarantee
        # items never start in penetration with each other.
        if spec.grid_row >= rows or spec.grid_col >= cols:
            continue
        cell_key = (spec.grid_row, spec.grid_col)
        if cell_key in occupied:
            continue
        occupied.add(cell_key)
        x = -base_dims_m[0] / 2.0 + (spec.grid_col + 0.5) * cell_l
        y = -base_dims_m[1] / 2.0 + (spec.grid_row + 0.5) * cell_w
        z = base_dims_m[2]
        for _ in range(spec.height):
            items.append(Item(
                sku=tpl.sku, weight_kg=tpl.weight_kg, dims_m=tpl.dims_m,
                fragility=tpl.fragility,
                position=(x, y, z), orientation_deg=0.0,
            ))
            z += tpl.dims_m[2]
    return PalletConfig(
        pallet_id=pallet_id, base_pallet_type=base_type,
        base_dims_m=base_dims_m, base_mass_kg=base_mass_kg,
        items=items, wrap=wrap, env=env, body_temp_c=body_temp_c,
        seconds_since_temp_change=seconds_since_temp_change,
    )


@dataclass
class Configurator:
    base_pallet_dims_m: Vec3 = (1.2, 0.8, 0.15)
    base_pallet_mass_kg: float = 25.0
    default_wrap: WrapType = WrapType.STRETCH

    def build(self, raw: RawInputs) -> PalletConfig:
        items = self._lay_out_items(raw.vision, raw.barcode_skus)
        return PalletConfig(
            pallet_id=raw.pallet_id,
            base_pallet_type=raw.base_pallet_type,
            base_dims_m=self.base_pallet_dims_m,
            base_mass_kg=self.base_pallet_mass_kg,
            items=items,
            wrap=self.default_wrap,
            env=raw.env,
            body_temp_c=raw.body_temp_c,
            seconds_since_temp_change=raw.seconds_since_temp_change,
        )

    # ---- layout logic ----

    def _lay_out_items(self, vision: VisionLayout, skus: list[str]) -> list[Item]:
        if not skus:
            return []
        layers = vision.layers
        per_layer = vision.items_per_layer
        expected = layers * per_layer
        # Tolerate over/under-supplied SKUs by clamping.
        actual_skus = (skus * (1 + expected // max(1, len(skus))))[:expected]

        items: list[Item] = []
        z_cursor = self.base_pallet_dims_m[2]  # start atop pallet base
        for layer_idx in range(layers):
            layer_skus = actual_skus[layer_idx * per_layer : (layer_idx + 1) * per_layer]
            layer_items, layer_height = self._lay_out_layer(
                layer_skus, layer_idx, vision, z_cursor,
            )
            items.extend(layer_items)
            z_cursor += layer_height
        return items

    def _lay_out_layer(
        self,
        skus: list[str],
        layer_idx: int,
        vision: VisionLayout,
        z_base: float,
    ) -> tuple[list[Item], float]:
        templates = [get_template(s) for s in skus]
        # Natural cell size = max item dim across this layer (per axis)
        nat_l = max(t.dims_m[0] for t in templates)
        nat_w = max(t.dims_m[1] for t in templates)
        layer_height = max(t.dims_m[2] for t in templates)

        # Bound the cell size to the pallet footprint so the grid can't span
        # beyond the base. Brick patterns shift odd layers by cell_l/2, so we
        # reserve that extra half-cell of width when computing the bound.
        cols, rows = self._grid_dims(len(templates), vision.pattern)
        effective_cols = cols + (0.5 if vision.pattern == StackPattern.BRICK else 0.0)
        max_l = self.base_pallet_dims_m[0] / max(effective_cols, 1.0)
        max_w = self.base_pallet_dims_m[1] / max(rows, 1)
        cell_l = min(nat_l, max_l)
        cell_w = min(nat_w, max_w)

        positions = self._cell_positions(
            len(templates), vision.pattern, layer_idx, cell_l, cell_w, cols, rows,
        )

        # Apply lean and overhang as anomalies
        lean_offset_x = (vision.lean_angle_deg / 8.0) * 0.05 * (layer_idx + 1)
        overhang_offset_x = vision.max_overhang_m if layer_idx == vision.layers - 1 else 0.0

        items: list[Item] = []
        for tpl, (cx, cy) in zip(templates, positions, strict=True):
            x = cx + lean_offset_x + overhang_offset_x
            y = cy
            items.append(Item(
                sku=tpl.sku,
                weight_kg=tpl.weight_kg,
                dims_m=tpl.dims_m,
                fragility=tpl.fragility,
                position=(x, y, z_base),
                orientation_deg=self._orientation(vision.pattern, layer_idx),
            ))
        return items, layer_height

    def _grid_dims(self, n: int, pattern: StackPattern) -> tuple[int, int]:
        """Return (cols, rows) for laying out n items in the chosen pattern.

        Column pattern stacks all items at the same (cols=rows=1) cell.
        Other patterns use a roughly-square grid.
        """
        if n <= 1 or pattern == StackPattern.COLUMN:
            return (1, 1)
        cols = max(1, int(round(n ** 0.5)))
        rows = (n + cols - 1) // cols
        return (cols, rows)

    def _cell_positions(
        self,
        n: int,
        pattern: StackPattern,
        layer_idx: int,
        cell_l: float,
        cell_w: float,
        cols: int,
        rows: int,
    ) -> list[tuple[float, float]]:
        if n == 0:
            return []
        if pattern == StackPattern.COLUMN:
            return [(0.0, 0.0)] * n
        x_offset_per_layer = 0.0
        if pattern == StackPattern.BRICK and layer_idx % 2 == 1:
            x_offset_per_layer = cell_l / 2.0
        positions: list[tuple[float, float]] = []
        x0 = -cell_l * (cols - 1) / 2.0 + x_offset_per_layer
        y0 = -cell_w * (rows - 1) / 2.0
        for i in range(n):
            r = i // cols
            c = i % cols
            positions.append((x0 + c * cell_l, y0 + r * cell_w))
        return positions

    def _orientation(self, pattern: StackPattern, layer_idx: int) -> float:
        if pattern == StackPattern.PINWHEEL and layer_idx % 2 == 1:
            return 90.0
        return 0.0
