"""Canonical domain types. Pydantic-validated, JSON-serializable.

Coordinate convention:
    Origin is the center of the BOTTOM face of the pallet base.
    +X = belt-travel direction, +Y = lateral, +Z = up.
    Item.position is the center of the item's bottom face, in pallet coordinates.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

Vec3 = tuple[float, float, float]


class EnvCondition(str, Enum):
    FROZEN = "frozen"               # ≤ -15 °C
    REFRIGERATED = "refrigerated"   # -5 to +5 °C
    THAWED = "thawed"               # ≥ +10 °C
    TRANSITIONING = "transitioning" # frost-melt zone, recently moved


class FragilityClass(str, Enum):
    RIGID = "rigid"             # carton, can, drum
    SEMI_RIGID = "semi_rigid"   # bagged frozen goods
    DEFORMABLE = "deformable"   # fresh produce, soft packaging


class WrapType(str, Enum):
    NONE = "none"
    SHRINK = "shrink"
    STRETCH = "stretch"
    BANDED = "banded"


class FailureMode(str, Enum):
    NO_FAILURE = "no_failure"
    TIP_OVER = "tip_over"
    TOP_ITEM_SLIDE = "top_item_slide"
    PALLET_SLIP = "pallet_slip"
    LOAD_SHIFT = "load_shift"


class Item(BaseModel):
    """A single item placed on a pallet."""

    model_config = ConfigDict(frozen=True)

    sku: str
    weight_kg: float = Field(gt=0, le=500)
    dims_m: Vec3 = Field(description="(length, width, height) in meters")
    fragility: FragilityClass = FragilityClass.RIGID
    position: Vec3 = Field(description="center of bottom face in pallet coords")
    orientation_deg: float = Field(default=0.0, ge=-180, le=180)

    @field_validator("dims_m", "position")
    @classmethod
    def _no_nan(cls, v: Vec3) -> Vec3:
        if any(x != x for x in v):  # NaN check
            raise ValueError("Vec3 components cannot be NaN")
        return v

    @field_validator("dims_m")
    @classmethod
    def _positive_dims(cls, v: Vec3) -> Vec3:
        if any(x <= 0 for x in v):
            raise ValueError("dims_m components must be positive")
        return v

    @computed_field
    @property
    def center_of_mass(self) -> Vec3:
        """Geometric center (assumes uniform density)."""
        x, y, z = self.position
        return (x, y, z + self.dims_m[2] / 2.0)


class PalletConfig(BaseModel):
    """A configured pallet: base + items + environmental state."""

    pallet_id: str
    base_pallet_type: str = "EUR"
    base_dims_m: Vec3 = (1.2, 0.8, 0.15)
    base_mass_kg: float = Field(default=25.0, gt=0, le=100)
    items: list[Item]
    wrap: WrapType = WrapType.STRETCH
    env: EnvCondition
    body_temp_c: float = Field(ge=-40, le=40)
    seconds_since_temp_change: float = Field(default=3600.0, ge=0)
    notes: str | None = None

    @computed_field
    @property
    def total_mass_kg(self) -> float:
        return self.base_mass_kg + sum(i.weight_kg for i in self.items)

    @computed_field
    @property
    def composite_com_m(self) -> Vec3:
        """Mass-weighted center of mass of (base + all items)."""
        bx, by, bz = 0.0, 0.0, self.base_dims_m[2] / 2.0
        m_total = self.base_mass_kg
        cx = self.base_mass_kg * bx
        cy = self.base_mass_kg * by
        cz = self.base_mass_kg * bz
        for item in self.items:
            ix, iy, iz = item.center_of_mass
            cx += item.weight_kg * ix
            cy += item.weight_kg * iy
            cz += item.weight_kg * iz
            m_total += item.weight_kg
        return (cx / m_total, cy / m_total, cz / m_total)

    @computed_field
    @property
    def stack_height_m(self) -> float:
        """Top of the highest item (or pallet top if no items)."""
        if not self.items:
            return self.base_dims_m[2]
        return max(i.position[2] + i.dims_m[2] for i in self.items)

    @computed_field
    @property
    def overhang_m(self) -> float:
        """Maximum horizontal distance any item extends beyond the pallet edge.

        Positive = item overhangs. Zero or negative = within footprint.
        """
        half_l = self.base_dims_m[0] / 2.0
        half_w = self.base_dims_m[1] / 2.0
        max_overhang = 0.0
        for item in self.items:
            ix, iy, _ = item.position
            il, iw, _ = item.dims_m
            x_extent = abs(ix) + il / 2.0
            y_extent = abs(iy) + iw / 2.0
            max_overhang = max(max_overhang, x_extent - half_l, y_extent - half_w)
        return max_overhang


class SafetyResult(BaseModel):
    """Output of the threshold solver. Phase D will populate this."""

    pallet_id: str
    max_speed_mps: float
    max_accel_mps2: float
    max_decel_mps2: float
    max_lateral_g: float
    dominant_failure_mode: FailureMode
    margin_pct: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    sim_runtime_ms: float
    config_hash: str
