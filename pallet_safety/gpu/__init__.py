"""GPU-batched simulation backend (optional: `pallet-safety[gpu]`).

Import `pallet_safety.gpu.grid` directly; it raises ImportError with install
guidance when mujoco-warp is missing. `HAS_GPU` is the cheap capability probe.
"""

from importlib.util import find_spec

HAS_GPU = find_spec("mujoco_warp") is not None and find_spec("warp") is not None
