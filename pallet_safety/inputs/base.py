"""Input adapter contract: anything that can deliver `RawInputs` is an adapter.

A `RawInputs` carries everything the system needs to *configure* a pallet,
in raw sensor-output form. The Configurator turns this into a `PalletConfig`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..models import EnvCondition


class StackPattern(str, Enum):
    COLUMN = "column"
    BRICK = "brick"
    PINWHEEL = "pinwheel"
    IRREGULAR = "irregular"


class VisionLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern: StackPattern
    layers: int = Field(ge=0, le=20)
    items_per_layer: int = Field(ge=0, le=20)
    lean_angle_deg: float = Field(default=0.0, ge=0, le=30)
    max_overhang_m: float = Field(default=0.0, ge=0, le=0.5)


class RawInputs(BaseModel):
    model_config = ConfigDict(frozen=True)

    barcode_skus: list[str]
    vision: VisionLayout
    env: EnvCondition
    body_temp_c: float = Field(ge=-40, le=40)
    seconds_since_temp_change: float = Field(default=3600.0, ge=0)
    pallet_id: str = "P-anon"
    base_pallet_type: str = "EUR"

    def summary(self) -> str:
        return (
            f"{self.pallet_id}: {len(self.barcode_skus)} SKUs, "
            f"{self.vision.pattern.value} {self.vision.layers}L×{self.vision.items_per_layer}, "
            f"{self.env.value} @ {self.body_temp_c:.1f}°C"
        )


class InputAdapter(ABC):
    @abstractmethod
    def read(self) -> RawInputs:
        ...
