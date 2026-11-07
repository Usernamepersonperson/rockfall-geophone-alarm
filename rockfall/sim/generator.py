"""Synthetic geophone generator.  Everything is an invented toy model, not a calibrated one.

Units are arbitrary "counts" with background noise sigma ~ 1 and ADC full scale +-1000.
Amplitude at distance r from a source: ``A * R0/(r+R0) * exp(-r/L)`` (spreading + damping).

Sources
-------
rockfall   impulsive impacts (damped broadband bursts) along a downslope path, 5-40 m/s.
debrisflow emergent band-limited rumble (2-15 Hz), amplitude grows then decays, front moves downslope 2-6 m/s.
traffic    engine harmonics (narrowband), vehicle moves along the road at ~15 m/s.
quake      distant earthquake / blast: decaying 1-f_hi Hz wave train reaching every station almost at once.
wind       gust: a shared envelope modulating independent 1-12 Hz noise at every station.
animal     a few footsteps right next to one station.
Faults: dead (zeros), stuck (last value frozen), clipping (huge gain into the ADC rails).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from rockfall.site import ROAD_Y, Site

FULL_SCALE = 1000.0
R0, DAMP_L, V_SEISMIC = 15.0, 200.0, 600.0


@dataclass(frozen=True)
class TruthEvent:
    kind: str  # rockfall | debrisflow | traffic | wind | animal | quake
    t_start: float
    t_end: float
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def hazardous(self) -> bool:
        return self.kind in ("rockfall", "debrisflow")


@dataclass(frozen=True)
class Fault:
    station: str
    kind: str  # dead | stuck | clipping
    t_start: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float
    events: tuple[TruthEvent, ...] = ()
    faults: tuple[Fault, ...] = ()
    noise_sigma: float = 1.0
    fs: int = 200
    seed: int = 0


def _atten(r: np.ndarray | float, damp: float = DAMP_L) -> np.ndarray | float:
    return R0 / (r + R0) * np.exp(-r / damp)


def _bandpass(rng: np.random.Generator, n: int, lo: float, hi: float, fs: int) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    y = sosfilt(sos, rng.normal(size=n + fs))[fs:]
    return y / (y.std() + 1e-12)


def _add_rockfall(sig: np.ndarray, ev: TruthEvent, st_xy: tuple[float, float], rng: np.random.Generator, fs: int) -> None:
    p = ev.params
    wl = int(0.6 * fs)
    tt = np.arange(wl) / fs
    for ti, (px, py), amp in zip(p["times"], p["positions"], p["amps"]):
        r = float(np.hypot(st_xy[0] - px, st_xy[1] - py))
        arrive = ti + r / V_SEISMIC
        i = int(arrive * fs)
        if i + wl >= len(sig):
            continue
        sig[i:i + wl] += amp * _atten(r, 120.0) * rng.normal(size=wl) * np.exp(-tt / 0.12)


def _add_debris(sig: np.ndarray, ev: TruthEvent, st_xy: tuple[float, float], rng: np.random.Generator, fs: int) -> None:
    p = ev.params
    i0, i1 = int(ev.t_start * fs), min(len(sig), int(ev.t_end * fs))
    t = np.arange(i1 - i0) / fs
    front_y = p["y0"] - p["speed"] * t
    r = np.hypot(st_xy[0] - p["x"], st_xy[1] - front_y)
    u = t / (p["peak_frac"] * (ev.t_end - ev.t_start))
    env = u * np.exp(1 - u)  # grows to 1 at peak_frac, then decays
    sig[i0:i1] += p["amp"] * env * _atten(r, 80.0) * _bandpass(rng, i1 - i0, 2.0, 15.0, fs)


def _add_traffic(sig: np.ndarray, ev: TruthEvent, st_xy: tuple[float, float], rng: np.random.Generator, fs: int) -> None:
    p = ev.params
    i0, i1 = int(ev.t_start * fs), min(len(sig), int(ev.t_end * fs))
    t = np.arange(i1 - i0) / fs
    veh_x = p["x0"] + p["dir"] * p["speed"] * t
    r = np.hypot(st_xy[0] - veh_x, st_xy[1] - ROAD_Y)
    f0 = p["f0"] * (1 + 0.03 * np.sin(2 * np.pi * 0.1 * t + p["phase"]))
    phase = 2 * np.pi * np.cumsum(f0) / fs
    wave = sum(w * np.sin(k * phase + p["phase"] * k) for k, w in ((1, 1.0), (2, 0.6), (3, 0.35)))
    sig[i0:i1] += p["amp"] * _atten(r, 150.0) * wave


def _add_wind(sigs: dict[str, np.ndarray], ev: TruthEvent, rng: np.random.Generator, fs: int) -> None:
    n = len(next(iter(sigs.values())))
    i0, i1 = int(ev.t_start * fs), min(n, int(ev.t_end * fs))
    env = np.hanning(i1 - i0) ** 0.5 * (ev.params["gain"] - 1.0)
    for sig in sigs.values():
        sig[i0:i1] += env * _bandpass(rng, i1 - i0, 1.0, 12.0, fs) * rng.lognormal(0, 0.2)


def _add_quake(sigs: dict[str, np.ndarray], ev: TruthEvent, site: Site, rng: np.random.Generator, fs: int) -> None:
    p = ev.params
    for st in site.stations:
        sig = sigs[st.name]
        i0 = int((ev.t_start + float(np.hypot(st.x, st.y)) / 3000.0) * fs)
        i1 = min(len(sig), int(ev.t_end * fs))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / fs
        sig[i0:i1] += p["amp"] * (1 - np.exp(-t / 0.3)) * np.exp(-t / p["tau"]) * _bandpass(rng, i1 - i0, 1.5, p["f_hi"], fs)


def _add_animal(sig: np.ndarray, ev: TruthEvent, st_xy: tuple[float, float], rng: np.random.Generator, fs: int, near: str, me: str) -> None:
    if near != me:
        return
    wl = int(0.3 * fs)
    tt = np.arange(wl) / fs
    for ti in ev.params["steps"]:
        i = int(ti * fs)
        if i + wl < len(sig):
            sig[i:i + wl] += ev.params["amp"] * rng.normal(size=wl) * np.exp(-tt / 0.06)


def simulate(scenario: Scenario, site: Site) -> dict[str, np.ndarray]:
    """Raw ADC counts per station for the scenario."""
    fs, n = scenario.fs, int(scenario.duration_s * scenario.fs)
    rng = np.random.default_rng(scenario.seed)
    t = np.arange(n) / fs
    sigs: dict[str, np.ndarray] = {}
    for st in site.stations:
        micro = 3.0 * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28))  # microseism, removed by the 1 Hz high-pass
        sigs[st.name] = rng.normal(0.0, scenario.noise_sigma, n) + micro
    for ev in scenario.events:
        if ev.kind == "wind":
            _add_wind(sigs, ev, rng, fs)
            continue
        if ev.kind == "quake":
            _add_quake(sigs, ev, site, rng, fs)
            continue
        for st in site.stations:
            xy = (st.x, st.y)
            if ev.kind == "rockfall":
                _add_rockfall(sigs[st.name], ev, xy, rng, fs)
            elif ev.kind == "debrisflow":
                _add_debris(sigs[st.name], ev, xy, rng, fs)
            elif ev.kind == "traffic":
                _add_traffic(sigs[st.name], ev, xy, rng, fs)
            elif ev.kind == "animal":
                _add_animal(sigs[st.name], ev, xy, rng, fs, ev.params["station"], st.name)
    for f in scenario.faults:
        sig, i = sigs[f.station], int(f.t_start * fs)
        if f.kind == "dead":
            sig[i:] = 0.0
        elif f.kind == "stuck":
            sig[i:] = sig[max(i - 1, 0)]
        elif f.kind == "clipping":
            sig[i:] *= 600.0
    return {k: np.clip(v, -FULL_SCALE, FULL_SCALE) for k, v in sigs.items()}
