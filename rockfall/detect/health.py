"""Station health checks on raw (unfiltered) data windows.

Order matters, the first failing check names the fault:

* DATA_GAP  - NaN / missing samples above ``gap_frac``
* DEAD      - standard deviation below ``dead_std`` (cable cut, sensor unpowered, zeros)
* STUCK     - more than ``stuck_frac`` of consecutive samples identical (frozen ADC / buffer)
* CLIPPING  - more than ``clip_frac`` of samples at the ADC rail (gain too high, saturated)

A station that fails any check is excluded from association.  When too few stations
remain healthy the alarm state machine goes DEGRADED (see ``alarm.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rockfall.config import HealthCfg

OK, DATA_GAP, DEAD, STUCK, CLIPPING = "OK", "DATA_GAP", "DEAD", "STUCK", "CLIPPING"


@dataclass(frozen=True)
class HealthResult:
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


def check_window(x: np.ndarray, cfg: HealthCfg) -> HealthResult:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return HealthResult(DATA_GAP, "no samples")
    nan_frac = float(np.mean(~np.isfinite(x)))
    if nan_frac > cfg.gap_frac:
        return HealthResult(DATA_GAP, f"{nan_frac:.0%} missing samples")
    x = x[np.isfinite(x)]
    std = float(np.std(x))
    if std < cfg.dead_std:
        return HealthResult(DEAD, f"std {std:.4f} < {cfg.dead_std}")
    flat = float(np.mean(np.diff(x) == 0.0)) if len(x) > 1 else 0.0
    if flat > cfg.stuck_frac:
        return HealthResult(STUCK, f"{flat:.0%} repeated samples")
    clipped = float(np.mean(np.abs(x) >= cfg.clip_level))
    if clipped > cfg.clip_frac:
        return HealthResult(CLIPPING, f"{clipped:.1%} of samples at rail")
    return HealthResult(OK)


def health_timeline(x: np.ndarray, fs: float, cfg: HealthCfg) -> list[tuple[float, HealthResult]]:
    """Evaluate consecutive windows; returns ``(window_end_s, result)``."""
    w = int(cfg.window_s * fs)
    return [((i + w) / fs, check_window(x[i:i + w], cfg)) for i in range(0, len(x) - w + 1, w)]
