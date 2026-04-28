"""Input adapter layer. Decouples the system from the source of pallet data.

Real adapters (camera/barcode/probe) and mock adapters (UI/random/batch) all
implement the same `InputAdapter` interface. The downstream pipeline (configurator
→ physics → API) doesn't know or care which one is wired in.
"""

from .base import InputAdapter, RawInputs, StackPattern, VisionLayout
from .mock_random import (
    MAX_PALLET_WEIGHT_KG,
    MAX_STACK_HEIGHT_M,
    Homogeneity,
    MockRandomAdapter,
)

__all__ = [
    "Homogeneity",
    "InputAdapter",
    "MAX_PALLET_WEIGHT_KG",
    "MAX_STACK_HEIGHT_M",
    "MockRandomAdapter",
    "RawInputs",
    "StackPattern",
    "VisionLayout",
]
