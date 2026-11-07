"""Scripted and randomised scenarios with ground truth."""
from __future__ import annotations

import numpy as np

from rockfall.sim.generator import Fault, Scenario, TruthEvent

NAMES = ("rockfall", "debrisflow", "traffic", "mixed", "dead-station", "degraded")
GULLY_X = 0.0
TOP_Y = 250.0


def make_rockfall(rng: np.random.Generator, t0: float, size: float | None = None) -> TruthEvent:
    """Rock released above the top station; 80-300 m path, 5-40 m/s, 2-20 s of impacts."""
    y0 = float(rng.uniform(235, 320))
    path = float(rng.uniform(150, 330))
    duration = float(rng.uniform(4, 20))
    speed = float(np.clip(path / duration, 8, 40))
    duration = path / speed
    rate = float(rng.uniform(3, 8))
    times = np.sort(rng.uniform(0, duration, int(duration * rate)))
    positions = [(GULLY_X + float(rng.normal(0, 8)), y0 - speed * ti) for ti in times]
    a0 = (size if size is not None else float(rng.lognormal(np.log(80), 1.0)))
    amps = a0 * rng.lognormal(0, 0.7, len(times))
    return TruthEvent("rockfall", t0, t0 + duration, {
        "times": (t0 + times).tolist(), "positions": positions, "amps": amps.tolist(),
        "speed": speed, "size": a0,
    })


def make_debris(rng: np.random.Generator, t0: float, amp: float | None = None) -> TruthEvent:
    speed = float(rng.uniform(2.5, 6.0))
    total = (330.0 + 80.0) / speed + 20.0  # front travels from 330 m to -80 m, then the tail fades
    return TruthEvent("debrisflow", t0, t0 + total, {
        "y0": 330.0, "x": GULLY_X, "speed": speed, "peak_frac": 0.4,
        "amp": amp if amp is not None else float(rng.uniform(25, 250)),
    })


def make_traffic(rng: np.random.Generator, t0: float) -> TruthEvent:
    speed = float(rng.uniform(12, 18))
    direction = float(rng.choice([-1, 1]))
    half = float(rng.uniform(120, 300))  # vehicle drives from -half to +half (or reverse)
    return TruthEvent("traffic", t0, t0 + 2 * half / speed, {
        "x0": -direction * half, "dir": direction, "speed": speed,
        "f0": float(rng.uniform(9, 14)), "phase": float(rng.uniform(0, 6.28)),
        "amp": float(rng.uniform(80, 200)),
    })


def make_wind(rng: np.random.Generator, t0: float) -> TruthEvent:
    d = float(rng.uniform(8, 40))
    return TruthEvent("wind", t0, t0 + d, {"gain": float(rng.uniform(3, 9))})


def make_quake(rng: np.random.Generator, t0: float) -> TruthEvent:
    tau = float(rng.uniform(3, 8))
    return TruthEvent("quake", t0, t0 + 4 * tau, {"tau": tau, "f_hi": float(rng.uniform(8, 32)), "amp": float(rng.uniform(30, 300))})


def make_animal(rng: np.random.Generator, t0: float, station: str) -> TruthEvent:
    steps = (t0 + np.cumsum(rng.uniform(0.4, 1.0, int(rng.integers(2, 6))))).tolist()
    return TruthEvent("animal", t0, steps[-1], {"steps": steps, "station": station, "amp": float(rng.uniform(25, 60))})


def named_scenario(name: str, seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    if name == "rockfall":
        return Scenario(name, 150.0, (make_rockfall(rng, 70.0),), seed=seed)
    if name == "debrisflow":
        ev = make_debris(rng, 70.0)
        return Scenario(name, ev.t_end + 30.0, (ev,), seed=seed)
    if name == "traffic":
        return Scenario(name, 200.0, (make_traffic(rng, 60.0), make_traffic(rng, 120.0)), seed=seed)
    if name == "dead-station":  # one dead + one clipping station, rockfall still has to be caught
        return Scenario(name, 150.0, (make_rockfall(rng, 70.0),),
                        (Fault("G2", "dead"), Fault("G5", "clipping")), seed=seed)
    if name == "degraded":  # three of six stations fail -> DEGRADED, never silently green
        return Scenario(name, 150.0, (make_rockfall(rng, 90.0),),
                        (Fault("G1", "dead"), Fault("G3", "stuck", 40.0), Fault("G6", "clipping", 40.0)), seed=seed)
    if name == "mixed":
        deb = make_debris(rng, 700.0)
        events = (make_traffic(rng, 60.0), make_wind(rng, 150.0), make_animal(rng, 230.0, "G3"),
                  make_rockfall(rng, 300.0), make_traffic(rng, 480.0), deb)
        return Scenario(name, deb.t_end + 30.0, events, seed=seed)
    raise ValueError(f"unknown scenario {name!r}; choose from {NAMES}")


BENCH_KINDS = ("rockfall", "debrisflow", "traffic", "wind", "animal", "quake", "quiet")
BENCH_WEIGHTS = (0.25, 0.10, 0.17, 0.17, 0.08, 0.10, 0.13)


def random_scenario(kind: str, seed: int) -> Scenario:
    """One randomised scenario of a given kind; ~10 % get a dead station, ~20 % get background wind."""
    rng = np.random.default_rng(seed)
    t0 = float(rng.uniform(50, 90))
    events: list[TruthEvent] = []
    if kind == "rockfall":
        events.append(make_rockfall(rng, t0))
        dur = 150.0
    elif kind == "debrisflow":
        events.append(make_debris(rng, t0))
        dur = events[0].t_end + 30.0
    elif kind == "traffic":
        events += [make_traffic(rng, t0), make_traffic(rng, t0 + float(rng.uniform(50, 150)))]
        dur = 400.0
    elif kind == "wind":
        events += [make_wind(rng, t0), make_wind(rng, t0 + float(rng.uniform(60, 200)))]
        dur = 400.0
    elif kind == "animal":
        events.append(make_animal(rng, t0, f"G{int(rng.integers(1, 7))}"))
        dur = 200.0
    elif kind == "quake":
        events += [make_quake(rng, t0), make_quake(rng, t0 + float(rng.uniform(80, 200)))]
        dur = 400.0
    elif kind == "quiet":
        dur = 600.0
    else:
        raise ValueError(kind)
    if kind in ("rockfall", "debrisflow") and rng.random() < 0.2:
        events.append(make_wind(rng, t0 + float(rng.uniform(-30, 20))))
    faults = (Fault(f"G{int(rng.integers(1, 7))}", "dead"),) if rng.random() < 0.1 else ()
    return Scenario(kind, dur, tuple(events), faults, noise_sigma=float(rng.uniform(0.7, 1.5)), seed=seed)
