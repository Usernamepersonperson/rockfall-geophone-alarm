"""Per-station event features, computed on the conditioned trace between trigger-on and "now".

duration_s    trigger-on to min(trigger-off, evaluation time)
peak_amp      max |x|
dom_freq_hz   argmax of the Welch PSD in 1-80 Hz
centroid_hz   PSD-weighted mean frequency in 1-80 Hz
peakiness     max PSD / mean PSD in band (large = narrowband, e.g. engine harmonics)
rise_time_s   trigger-on until the 0.25 s envelope first reaches 90 % of its maximum
crest_factor  peak / RMS (large = impulsive)
env_cv        std / mean of the 0.25 s envelope (large = bursty, small = steady rumble)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import welch

BAND = (1.0, 80.0)
ENV_S = 0.25


@dataclass(frozen=True)
class Features:
    duration_s: float
    ongoing: bool
    peak_amp: float
    dom_freq_hz: float
    centroid_hz: float
    peakiness: float
    rise_time_s: float
    crest_factor: float
    env_cv: float

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["ongoing"] = float(self.ongoing)
        return d


def _envelope(x: np.ndarray, fs: float) -> np.ndarray:
    k = max(1, round(ENV_S * fs))
    c = np.cumsum(np.insert(np.abs(x), 0, 0.0))
    return (c[k:] - c[:-k]) / k if len(x) >= k else np.array([np.mean(np.abs(x))])


def compute_features(x: np.ndarray, fs: float, on_s: float, end_s: float, ongoing: bool = False) -> Features | None:
    """Features of ``x`` between ``on_s`` and ``end_s``; ``None`` if the window is under 1 s."""
    a, b = int(on_s * fs), int(end_s * fs)
    seg = np.asarray(x[a:b], dtype=float)
    if len(seg) < fs:
        return None
    freqs, psd = welch(seg, fs=fs, nperseg=min(len(seg), 512))
    m = (freqs >= BAND[0]) & (freqs <= BAND[1])
    f, p = freqs[m], psd[m]
    total = float(p.sum()) + 1e-30
    env = _envelope(seg, fs)
    rms = float(np.sqrt(np.mean(seg**2))) + 1e-30
    peak = float(np.max(np.abs(seg)))
    rise = float(np.argmax(env >= 0.9 * env.max()) / fs)
    return Features(
        duration_s=(end_s - on_s), ongoing=ongoing, peak_amp=peak,
        dom_freq_hz=float(f[int(np.argmax(p))]),
        centroid_hz=float((f * p).sum() / total),
        peakiness=float(p.max() / (p.mean() + 1e-30)),
        rise_time_s=rise, crest_factor=peak / rms,
        env_cv=float(env.std() / (env.mean() + 1e-30)),
    )


def aggregate(per_station: list[Features]) -> dict[str, float]:
    """Combine station features into one event vector: median for shape, max for size/duration."""
    if not per_station:
        return {}
    med = lambda attr: float(np.median([getattr(f, attr) for f in per_station]))  # noqa: E731
    return {
        "duration_s": max(f.duration_s for f in per_station),
        "peak_amp": max(f.peak_amp for f in per_station),
        "dom_freq_hz": med("dom_freq_hz"),
        "centroid_hz": med("centroid_hz"),
        "peakiness": med("peakiness"),
        "rise_time_s": med("rise_time_s"),
        "crest_factor": med("crest_factor"),
        "env_cv": med("env_cv"),
    }
