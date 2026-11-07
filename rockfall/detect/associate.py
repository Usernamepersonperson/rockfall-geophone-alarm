"""Network association: k-of-n coincidence plus an apparent-velocity fit.

Each station contributes its trigger-on time ``t_i``.  A plane-wave / moving-front model

    t_i = t0 + s . (x_i, y_i)

is fitted by least squares for the slowness vector ``s`` (s/m).  Apparent speed is
``1/|s|`` and the direction ``s/|s|`` is compared with the downslope unit vector
(``cos_downslope`` = 1 means the front moves straight downhill).

* With >= 4 stations and a genuinely 2-D array (minor/major aperture >= 0.4) the full 2-D
  fit (3 parameters) is used.
* Otherwise (a gully-shaped array like the demo site) the arrival times are projected on the
  downslope axis and fitted in 1-D (2 parameters).  That keeps one degree of freedom for
  3 stations, so the residual still says something about consistency, and it avoids
  inventing a cross-slope direction that a narrow array cannot resolve.

Association is causal: triggers are consumed in order of trigger-on time, and an event is
declared at the on-time of the trigger that completes the first consistent group of
``min_stations``.  Later consistent triggers are absorbed into the same event.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from rockfall.config import AssociationCfg
from rockfall.detect.sta_lta import Trigger
from rockfall.site import Site

MIN_2D_ASPECT = 0.4  # minor/major aperture ratio needed before the cross-slope component is trusted
SPEED_CAP = 1e6


@dataclass(frozen=True)
class Fit:
    t0: float
    slowness: tuple[float, float]  # (sx, sy) in s/m
    speed_ms: float
    cos_downslope: float
    rms_s: float
    n: int
    dof: int

    def predict(self, xy: np.ndarray) -> np.ndarray:
        return self.t0 + xy @ np.array(self.slowness)

    def plausible(self, cfg: AssociationCfg) -> bool:
        """Moving downslope at a believable apparent speed."""
        return cfg.min_speed_ms <= self.speed_ms <= cfg.max_speed_ms and self.cos_downslope >= cfg.min_cos_downslope


def fit_arrivals(xy: np.ndarray, t: np.ndarray, downslope: tuple[float, float]) -> Fit:
    """Least-squares apparent velocity from arrival times at known positions."""
    xy, t = np.asarray(xy, float), np.asarray(t, float)
    d = np.asarray(downslope, float)
    d = d / np.linalg.norm(d)
    n = len(t)
    centred = xy - xy.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)
    if n >= 4 and sv[-1] >= MIN_2D_ASPECT * sv[0]:
        design = np.column_stack([np.ones(n), xy])
        sol, *_ = np.linalg.lstsq(design, t, rcond=None)
        t0, s = float(sol[0]), np.array(sol[1:])
        dof = n - 3
    else:
        u = xy @ d
        design = np.column_stack([np.ones(n), u])
        sol, *_ = np.linalg.lstsq(design, t, rcond=None)
        t0, s = float(sol[0]), float(sol[1]) * d
        dof = n - 2
    resid = t - (t0 + xy @ s)
    rms = float(np.sqrt(np.sum(resid**2) / max(dof, 1))) if dof > 0 else 0.0
    mag = float(np.linalg.norm(s))
    speed = SPEED_CAP if mag < 1.0 / SPEED_CAP else 1.0 / mag
    cos = float(s @ d / mag) if mag > 0 else 0.0
    return Fit(t0, (float(s[0]), float(s[1])), speed, cos, rms, n, dof)


@dataclass
class AssocEvent:
    """A group of consistent triggers.  ``t_assoc`` is when the k-th station completed it."""

    event_id: int
    t_assoc: float
    members: dict[str, Trigger]
    fit: Fit
    span_s: float = 0.0
    fit0: Fit | None = None  # fit at the moment of association (before later stations were absorbed)

    @property
    def first_on(self) -> float:
        return min(m.on_s for m in self.members.values())


def _tol(cfg: AssociationCfg, times: np.ndarray) -> float:
    return cfg.rms_tol_base_s + cfg.rms_tol_frac * float(times.max() - times.min())


def associate(triggers: list[Trigger], site: Site, cfg: AssociationCfg) -> list[AssocEvent]:
    """Group triggers from all (healthy) stations into events, causally."""
    by_name = {s.name: np.array([s.x, s.y]) for s in site.stations}
    ordered = sorted(triggers, key=lambda tr: tr.on_s)
    events: list[AssocEvent] = []
    used: set[int] = set()
    for idx, trg in enumerate(ordered):
        # 1) does it extend an existing event?
        for ev in events:
            if trg.station in ev.members or trg.on_s - ev.first_on > cfg.max_span_s:
                continue
            members = {**ev.members, trg.station: trg}
            names = list(members)
            times = np.array([members[n].on_s for n in names])
            fit = fit_arrivals(np.array([by_name[n] for n in names]), times, site.downslope)
            if fit.rms_s <= _tol(cfg, times):
                ev.members, ev.fit, ev.span_s = members, fit, float(times.max() - times.min())
                used.add(idx)
                break
        if idx in used:
            continue
        # 2) newest unused trigger per station inside the span window
        pool: dict[str, tuple[int, Trigger]] = {}
        for j in range(idx + 1):
            other = ordered[j]
            if j not in used and trg.on_s - other.on_s <= cfg.max_span_s:
                pool[other.station] = (j, other)
        if len(pool) < cfg.min_stations or trg.station not in pool:
            continue
        best: tuple[tuple[bool, int, float], dict[str, Trigger], Fit] | None = None
        others = [n for n in pool if n != trg.station]
        for size in range(len(pool), cfg.min_stations - 1, -1):
            for combo in combinations(others, size - 1):
                names = [*combo, trg.station]
                times = np.array([pool[n][1].on_s for n in names])
                fit = fit_arrivals(np.array([by_name[n] for n in names]), times, site.downslope)
                if fit.rms_s > _tol(cfg, times):
                    continue
                key = (fit.plausible(cfg), size, -fit.rms_s)
                if best is None or key > best[0]:
                    best = (key, {n: pool[n][1] for n in names}, fit)
            if best is not None and best[0][1] == size:
                break
        if best is None:
            continue
        _, members, fit = best
        for n in members:
            used.add(pool[n][0])
        times = np.array([m.on_s for m in members.values()])
        events.append(AssocEvent(len(events) + 1, trg.on_s, members, fit, float(times.max() - times.min()), fit))
    return events
