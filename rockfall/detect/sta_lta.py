"""Recursive STA/LTA trigger with on/off thresholds.

For the energy series ``e[n] = x[n]**2`` the short- and long-term averages are the
exponentially weighted recursions of the classic recursive STA/LTA::

    STA[n] = STA[n-1] + (e[n] - STA[n-1]) / N_s
    LTA[n] = LTA[n-1] + (e[n] - LTA[n-1]) / N_l
    R[n]   = STA[n] / LTA[n]

A trigger switches on when ``R > on`` and off when ``R < off`` (hysteresis).  Both
recursions are first-order IIR filters, so they run through ``scipy.signal.lfilter``
instead of a Python loop.

With ``freeze_lta`` the LTA is held at its value from the trigger-on sample until the
trigger ends (or ``max_trigger_s`` elapses).  Otherwise a long emergent signal such as a
debris-flow rumble drags the LTA up and the ratio collapses mid-event.  The LTA then
resumes from the frozen value.  Causal: the trigger-on time at sample ``n`` depends only
on samples ``<= n``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from rockfall.config import DetectorCfg

EPS = 1e-12


@dataclass(frozen=True)
class Trigger:
    station: str
    on_s: float
    off_s: float
    ongoing: bool  # still on at end of record
    peak_ratio: float


def recursive_sta_lta(x: np.ndarray, ns: int, nl: int) -> np.ndarray:
    """Plain recursive STA/LTA ratio (no freezing). The first ``nl`` samples (warm-up) are 0."""
    e = np.asarray(x, dtype=float) ** 2
    cs, cl = 1.0 / ns, 1.0 / nl
    sta = lfilter([cs], [1.0, -(1.0 - cs)], e)
    lta = lfilter([cl], [1.0, -(1.0 - cl)], e)
    ratio = sta / (lta + EPS)
    ratio[:nl] = 0.0
    return ratio


def _hysteresis(ratio: np.ndarray, start: int, on: float, off: float) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i, n = start, len(ratio)
    while i < n:
        up = np.flatnonzero(ratio[i:] > on)
        if not up.size:
            break
        a = i + int(up[0])
        down = np.flatnonzero(ratio[a:] < off)
        b = a + int(down[0]) if down.size else n
        out.append((a, b))
        i = b
    return out


def detect_triggers(x: np.ndarray, fs: float, cfg: DetectorCfg, station: str = "") -> list[Trigger]:
    """Return merged triggers for one conditioned (high-passed) trace."""
    e = np.asarray(x, dtype=float) ** 2
    n = len(e)
    ns, nl = max(1, round(cfg.sta_s * fs)), max(2, round(cfg.lta_s * fs))
    cs, cl = 1.0 / ns, 1.0 / nl
    sta = lfilter([cs], [1.0, -(1.0 - cs)], e)
    spans: list[tuple[int, int]] = []
    if not cfg.freeze_lta:
        lta = lfilter([cl], [1.0, -(1.0 - cl)], e)
        spans = _hysteresis(sta / (lta + EPS), nl, cfg.on, cfg.off)
    else:
        lta = np.empty(n)
        pos, zi = 0, np.zeros(1)
        while pos < n:
            seg, _ = lfilter([cl], [1.0, -(1.0 - cl)], e[pos:], zi=zi)
            lta[pos:] = seg
            first = max(pos, nl)
            if first >= n:
                break
            hit = np.flatnonzero(sta[first:] / (lta[first:] + EPS) > cfg.on)
            if not hit.size:
                break
            a = first + int(hit[0])
            frozen = lta[a]
            limit = min(n, a + round(cfg.max_trigger_s * fs))
            below = np.flatnonzero(sta[a:limit] < cfg.off * (frozen + EPS))
            b = a + int(below[0]) if below.size else limit
            lta[a:b] = frozen
            spans.append((a, b))
            pos, zi = b, np.array([(1.0 - cl) * frozen])
    ratio = sta / (lta + EPS)
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and (a - merged[-1][1]) / fs < cfg.merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [
        Trigger(station, a / fs, b / fs, b >= n, float(ratio[a:b].max()))
        for a, b in merged
        if (b - a) / fs >= cfg.min_trigger_s
    ]
