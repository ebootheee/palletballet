"""Phase D — binary-search the maximum safe operating envelope for a pallet.

Given a `PalletConfig`, find the highest sustained conveyor speed (and highest
acceleration) that doesn't trigger any failure mode. Returns a `SafetyResult`
that the inference API returns to the caller.

Assumptions (documented so consumers know the limits):
  - Failure is monotone in speed/accel. A pallet that fails at V is assumed to
    fail at all V' > V. This is generally true for friction-driven slip and
    for accel-driven tip, but NOT strictly true for resonance effects (which
    we don't currently model).
  - The search uses a fixed ramp accel for the speed search and a fixed target
    speed for the accel search. They don't interact in v1.
  - Caching is by SHA256 of the PalletConfig JSON — physically identical pallets
    with different pallet_id strings will still get separate cache entries
    (change `_config_fingerprint` to strip the id if that matters).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from time import perf_counter

import mujoco
from pydantic import BaseModel, ConfigDict, Field

from .failures import FailureThresholds, first_failure
from .friction import DEFAULT_PAIR
from .models import FailureMode, PalletConfig, SafetyResult
from .solver import ConveyorProfile, build_model, simulate

DEFAULT_SETTLE_S = 0.15  # items drop ~10cm during settle — sufficient


class SearchConfig(BaseModel):
    """Bounds and precision for the binary-search sweep.

    Defaults tuned for realistic conveyor ranges: pallets move 0.3–1.5 m/s in
    normal distribution ops, up to 2.0 m/s in high-throughput lines. Acceleration
    is typically 0.5–3 m/s². Searching beyond these adds latency for ranges the
    customer would never operate in. Override if your target deployment differs.
    """
    model_config = ConfigDict(frozen=True)

    speed_min_mps: float = Field(default=0.1, gt=0, le=5.0)
    speed_max_mps: float = Field(default=2.0, gt=0, le=5.0)
    accel_min_mps2: float = Field(default=0.1, gt=0, le=10.0)
    accel_max_mps2: float = Field(default=5.0, gt=0, le=10.0)
    precision_mps: float = Field(default=0.1, gt=0, le=0.5)
    precision_mps2: float = Field(default=0.2, gt=0, le=1.0)
    hold_s: float = Field(default=0.25, gt=0, le=3.0)
    speed_search_accel: float = Field(default=1.0, gt=0, le=10.0)
    accel_search_target_speed: float = Field(default=1.0, gt=0, le=5.0)


@dataclass
class AnalysisResult:
    """Rich output of a full safety analysis — wraps SafetyResult with diagnostics."""
    result: SafetyResult
    sims_run: int
    cache_hits: int = 0
    # Tuples of (axis, value, safe, failure_mode). axis in {"speed", "accel"}.
    sweep_points: list[tuple[str, float, bool, FailureMode]] = field(default_factory=list)


def _config_fingerprint(config: PalletConfig) -> str:
    """Stable hash of physically-relevant config fields (excluding notes)."""
    blob = config.model_dump_json(
        exclude={"notes"},
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ThresholdAnalyzer:
    """Runs safety analyses for pallets. Caches compiled MuJoCo models and results.

    Instantiate once, call `analyze()` many times. Thread-safe for read-only
    methods; caches are simple dicts and not meant for cross-thread writes.
    """

    def __init__(
        self,
        thresholds: FailureThresholds | None = None,
        search: SearchConfig | None = None,
        surface_pair: tuple[str, str] = DEFAULT_PAIR,
        max_model_cache: int = 64,
        max_result_cache: int = 512,
    ):
        self.thresholds = thresholds or FailureThresholds()
        self.search = search or SearchConfig()
        self.surface_pair = surface_pair
        self._model_cache: dict[str, mujoco.MjModel] = {}
        self._result_cache: dict[str, AnalysisResult] = {}
        self._max_models = max_model_cache
        self._max_results = max_result_cache

    # ---- public entry points ----

    def analyze(self, config: PalletConfig) -> AnalysisResult:
        """Full safety envelope for a pallet. Cached by config fingerprint."""
        h = _config_fingerprint(config)
        if h in self._result_cache:
            cached = self._result_cache[h]
            return AnalysisResult(
                result=cached.result, sims_run=0, cache_hits=1,
                sweep_points=cached.sweep_points,
            )

        t0 = perf_counter()
        model = self._get_model(config)
        max_speed, speed_fail_mode, n_speed, speed_sweep = self._binary_search_speed(config, model)
        max_accel, accel_fail_mode, n_accel, accel_sweep = self._binary_search_accel(config, model)

        # Margins relative to the operating envelope (clamped 0..1)
        speed_margin = max(0.0, (max_speed - self.search.speed_min_mps) / (
            self.search.speed_max_mps - self.search.speed_min_mps
        ))
        accel_margin = max(0.0, (max_accel - self.search.accel_min_mps2) / (
            self.search.accel_max_mps2 - self.search.accel_min_mps2
        ))

        # Dominant failure = whichever dimension is the tighter constraint
        dominant = speed_fail_mode if speed_margin < accel_margin else accel_fail_mode

        # Margin pct = the weaker of the two axes; confidence scales with headroom
        margin_pct = max(0.0, min(100.0, min(speed_margin, accel_margin) * 100.0))
        confidence = max(0.0, min(1.0, min(speed_margin, accel_margin) * 1.25))

        runtime_ms = (perf_counter() - t0) * 1000.0
        result = SafetyResult(
            pallet_id=config.pallet_id,
            max_speed_mps=max_speed,
            max_accel_mps2=max_accel,
            max_decel_mps2=max_accel,  # symmetric approx; v2 could search separately
            max_lateral_g=0.3,  # placeholder until we model curves
            dominant_failure_mode=dominant,
            margin_pct=margin_pct,
            confidence=confidence,
            sim_runtime_ms=runtime_ms,
            config_hash=h,
        )
        tagged_sweep = (
            [("speed", v, s, m) for v, s, m in speed_sweep]
            + [("accel", v, s, m) for v, s, m in accel_sweep]
        )
        analysis = AnalysisResult(
            result=result,
            sims_run=n_speed + n_accel,
            cache_hits=0,
            sweep_points=tagged_sweep,
        )
        self._store_result(h, analysis)
        return analysis

    def max_safe_speed(self, config: PalletConfig) -> tuple[float, FailureMode, int]:
        """Just the speed search, for callers that don't need the full envelope."""
        model = self._get_model(config)
        speed, mode, n_sims, _ = self._binary_search_speed(config, model)
        return speed, mode, n_sims

    # ---- internals ----

    def _binary_search_speed(
        self, config: PalletConfig, model: mujoco.MjModel,
    ) -> tuple[float, FailureMode, int, list[tuple[float, bool, FailureMode]]]:
        lo = self.search.speed_min_mps
        hi = self.search.speed_max_mps
        last_fail = FailureMode.NO_FAILURE
        sweep: list[tuple[float, bool, FailureMode]] = []
        n = 0

        # Check the high bound first — if safe, no search needed
        safe_hi, mode_hi = self._run_speed(config, model, hi)
        sweep.append((hi, safe_hi, mode_hi))
        n += 1
        if safe_hi:
            return hi, FailureMode.NO_FAILURE, n, sweep

        # Check the low bound — if not safe, the pallet is fragile; return min
        safe_lo, mode_lo = self._run_speed(config, model, lo)
        sweep.append((lo, safe_lo, mode_lo))
        n += 1
        if not safe_lo:
            return 0.0, mode_lo, n, sweep

        last_fail = mode_hi
        while hi - lo > self.search.precision_mps:
            mid = (lo + hi) / 2.0
            safe, mode = self._run_speed(config, model, mid)
            sweep.append((mid, safe, mode))
            n += 1
            if safe:
                lo = mid
            else:
                hi = mid
                last_fail = mode
        return lo, last_fail, n, sweep

    def _binary_search_accel(
        self, config: PalletConfig, model: mujoco.MjModel,
    ) -> tuple[float, FailureMode, int, list[tuple[float, bool, FailureMode]]]:
        lo = self.search.accel_min_mps2
        hi = self.search.accel_max_mps2
        last_fail = FailureMode.NO_FAILURE
        sweep: list[tuple[float, bool, FailureMode]] = []
        n = 0
        target = self.search.accel_search_target_speed

        safe_hi, mode_hi = self._run_accel(config, model, target, hi)
        sweep.append((hi, safe_hi, mode_hi))
        n += 1
        if safe_hi:
            return hi, FailureMode.NO_FAILURE, n, sweep

        safe_lo, mode_lo = self._run_accel(config, model, target, lo)
        sweep.append((lo, safe_lo, mode_lo))
        n += 1
        if not safe_lo:
            return 0.0, mode_lo, n, sweep

        last_fail = mode_hi
        while hi - lo > self.search.precision_mps2:
            mid = (lo + hi) / 2.0
            safe, mode = self._run_accel(config, model, target, mid)
            sweep.append((mid, safe, mode))
            n += 1
            if safe:
                lo = mid
            else:
                hi = mid
                last_fail = mode
        return lo, last_fail, n, sweep

    def _run_speed(
        self, config: PalletConfig, model: mujoco.MjModel, speed: float,
    ) -> tuple[bool, FailureMode]:
        accel = self.search.speed_search_accel
        ramp_time = speed / accel
        duration = ramp_time + self.search.hold_s
        profile = ConveyorProfile(
            target_speed_mps=speed, accel_mps2=accel, duration_s=duration,
        )
        trace = simulate(
            config, profile, surface_pair=self.surface_pair,
            settle_s=DEFAULT_SETTLE_S, model=model,
            fail_fast_tip_deg=self.thresholds.tip_angle_deg,
            fail_fast_slide_m=self.thresholds.item_slide_m,
        )
        mode, _ = first_failure(trace, self.thresholds)
        return mode == FailureMode.NO_FAILURE, mode

    def _run_accel(
        self, config: PalletConfig, model: mujoco.MjModel,
        target_speed: float, accel: float,
    ) -> tuple[bool, FailureMode]:
        ramp_time = target_speed / accel
        duration = ramp_time + self.search.hold_s
        profile = ConveyorProfile(
            target_speed_mps=target_speed, accel_mps2=accel, duration_s=duration,
        )
        trace = simulate(
            config, profile, surface_pair=self.surface_pair,
            settle_s=DEFAULT_SETTLE_S, model=model,
            fail_fast_tip_deg=self.thresholds.tip_angle_deg,
            fail_fast_slide_m=self.thresholds.item_slide_m,
        )
        mode, _ = first_failure(trace, self.thresholds)
        return mode == FailureMode.NO_FAILURE, mode

    def _get_model(self, config: PalletConfig) -> mujoco.MjModel:
        h = _config_fingerprint(config)
        if h in self._model_cache:
            return self._model_cache[h]
        if len(self._model_cache) >= self._max_models:
            # Simple eviction — drop the oldest entry
            self._model_cache.pop(next(iter(self._model_cache)))
        m = build_model(config, surface_pair=self.surface_pair)
        self._model_cache[h] = m
        return m

    def _store_result(self, h: str, analysis: AnalysisResult) -> None:
        if len(self._result_cache) >= self._max_results:
            self._result_cache.pop(next(iter(self._result_cache)))
        self._result_cache[h] = analysis


# Module-level singleton for the API to use — avoids re-caching on every request
_default_analyzer: ThresholdAnalyzer | None = None


def default_analyzer() -> ThresholdAnalyzer:
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = ThresholdAnalyzer()
    return _default_analyzer


def analyze(config: PalletConfig) -> AnalysisResult:
    """Convenience entry point using the process-wide default analyzer."""
    return default_analyzer().analyze(config)
