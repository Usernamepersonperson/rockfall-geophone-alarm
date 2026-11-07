"""End-to-end chain: raw traces -> health -> STA/LTA -> association -> classification -> alarm.

The run is a replay with explicit event time.  Every decision carries the timestamp at which
the data it needs became available, plus the fixed latency overhead (acquisition block,
telemetry, processing, actuation) from the config:

    t_YELLOW = t_association(k-th station trigger) + overhead
    t_RED    = t_association + classification window + overhead

All decisions are then fed to the state machine in chronological order.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from rockfall.alarm import AlarmStateMachine, State, Transition
from rockfall.config import Config
from rockfall.detect.associate import AssocEvent, associate, fit_arrivals
from rockfall.detect.classify import UNCLASSIFIED, Classification, classify
from rockfall.detect.features import Features, aggregate, compute_features
from rockfall.detect.health import HealthResult, health_timeline
from rockfall.detect.sta_lta import Trigger, detect_triggers
from rockfall.sinks import Sink
from rockfall.site import Site

HAZARD_LABELS = ("rockfall", "debris_flow")


@dataclass
class EventReport:
    event_id: int
    stations: list[str]
    first_on_s: float
    t_assoc_s: float
    speed_ms: float
    cos_downslope: float
    fit_rms_s: float
    plausible: bool
    label: str = UNCLASSIFIED
    rule: str = ""
    t_class_s: float | None = None
    t_yellow_s: float | None = None
    t_red_s: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def latency_breakdown(self, overhead: dict[str, float]) -> dict[str, float] | None:
        """Where the time to RED went (seconds after the first station trigger)."""
        if self.t_red_s is None or self.t_class_s is None:
            return None
        return {
            "association_wait": round(self.t_assoc_s - self.first_on_s, 2),
            "classification_window": round(self.t_class_s - self.t_assoc_s, 2),
            **{k: round(v, 2) for k, v in overhead.items()},
            "total_after_first_trigger": round(self.t_red_s - self.first_on_s, 2),
        }


@dataclass
class RunResult:
    events: list[EventReport]
    transitions: list[Transition]
    records: list[dict[str, Any]]
    health: dict[str, list[tuple[float, HealthResult]]]
    triggers: dict[str, list[Trigger]]
    duration_s: float

    @property
    def states(self) -> list[str]:
        return [t.to_state.value for t in self.transitions]

    @property
    def final_state(self) -> State:
        return self.transitions[-1].to_state

    def first_time(self, state: State) -> float | None:
        return next((t.t for t in self.transitions if t.to_state is state), None)


class Pipeline:
    def __init__(self, cfg: Config, site: Site, sinks: Iterable[Sink] = ()) -> None:
        self.cfg, self.site, self.sinks = cfg, site, list(sinks)
        self._hp = butter(4, cfg.highpass_hz, btype="high", fs=cfg.fs, output="sos")
        lat = cfg.latency
        self.overhead = {"acquisition_block": lat.acquisition_block_s, "telemetry": lat.telemetry_s,
                         "processing": lat.processing_s, "actuation": lat.actuation_s}

    # ------------------------------------------------------------------
    def run(self, data: dict[str, np.ndarray], ack_times: Iterable[float] = (), flush: bool = True) -> RunResult:
        cfg, fs = self.cfg, self.cfg.fs
        n = len(next(iter(data.values())))
        duration = n / fs
        hp = {k: sosfilt(self._hp, np.nan_to_num(v)) for k, v in data.items()}
        health = {k: health_timeline(v, fs, cfg.health) for k, v in data.items()}
        triggers = {k: detect_triggers(hp[k], fs, cfg.detector, k) for k in data}

        def healthy_at(station: str, t: float) -> bool:
            ok = True
            for t_end, res in health[station]:
                if t_end > t:
                    break
                ok = res.ok
            return ok

        usable = [tr for k, trs in triggers.items() for tr in trs if healthy_at(k, tr.on_s)]
        reports: list[EventReport] = []
        for ev in associate(usable, self.site, cfg.association):
            reports.append(self._evaluate(ev, hp, duration))

        # ---- chronological feed to the state machine -------------------
        ov = cfg.latency.fixed_overhead_s
        feed: list[tuple[float, int, str, Any]] = []
        windows = sorted({t for series in health.values() for t, _ in series})
        for t_w in windows:
            ok = sum(1 for k in health if next(r for t, r in health[k] if t == t_w).ok)
            feed.append((t_w + ov, 0, "health", (ok, len(health))))
        for rep in reports:
            if rep.t_yellow_s is not None:
                feed.append((rep.t_yellow_s, 1, "candidate", rep))
            if rep.t_red_s is not None:
                feed.append((rep.t_red_s, 2, "hazard", rep))
        feed += [(t, 3, "ack", None) for t in ack_times]
        feed.sort(key=lambda f: (f[0], f[1]))

        records: list[dict[str, Any]] = []
        ctx: dict[str, Any] = {}

        def on_transition(tr: Transition) -> None:
            rec = {"t": round(tr.t, 3), "site": cfg.site, "state": tr.to_state.value, "previous": tr.from_state.value,
                   "light": tr.light, "road_closed": tr.road_closed, "reason": tr.reason, "event": dict(ctx) or None}
            records.append(rec)
            for sink in self.sinks:
                sink.emit(rec)

        sm = AlarmStateMachine(cfg.alarm, on_transition)
        for t, _, kind, payload in feed:
            ctx.clear()
            if kind == "health":
                sm.health(t, *payload)
            elif kind == "ack":
                sm.ack(t)
            else:
                rep: EventReport = payload
                ctx.update(event_id=rep.event_id, label=rep.label, rule=rep.rule, stations=rep.stations,
                           speed_ms=round(rep.speed_ms, 1), cos_downslope=round(rep.cos_downslope, 2))
                if kind == "candidate":
                    sm.candidate(t, f"event {rep.event_id}: coincident downslope trigger on {len(rep.stations)} stations")
                else:
                    ctx["latency"] = rep.latency_breakdown(self.overhead)
                    sm.hazard(t, f"event {rep.event_id}: {rep.label} ({rep.explanation})")
        ctx.clear()
        if flush and feed:  # assume a quiet tail so hold times can expire
            sm.advance(feed[-1][0] + cfg.alarm.red_hold_s + cfg.alarm.yellow_hold_s + cfg.alarm.degraded_recovery_s)
        return RunResult(reports, sm.transitions, records, health, triggers, duration)

    # ------------------------------------------------------------------
    def _evaluate(self, ev: AssocEvent, hp: dict[str, np.ndarray], duration: float) -> EventReport:
        cfg, fs = self.cfg, self.cfg.fs
        ov = cfg.latency.fixed_overhead_s
        fit0 = ev.fit0 or ev.fit
        rep = EventReport(ev.event_id, sorted(ev.members), ev.first_on, ev.t_assoc, fit0.speed_ms,
                          fit0.cos_downslope, fit0.rms_s, fit0.plausible(cfg.association))
        if rep.plausible:
            rep.t_yellow_s = ev.t_assoc + ov
        c = cfg.classification
        attempt = ev.t_assoc + c.window_s
        last: Classification | None = None
        while True:
            tc = min(attempt, duration)
            members = [m for m in ev.members.values() if m.on_s <= tc - 1.0]
            if len(members) >= cfg.association.min_stations:
                names = [m.station for m in members]
                fit = fit_arrivals(self.site.xy(names), np.array([m.on_s for m in members]), self.site.downslope)
                feats: list[Features] = []
                for m in members:
                    f = compute_features(hp[m.station], fs, m.on_s, min(m.off_s, tc), ongoing=m.off_s > tc)
                    if f is not None:
                        feats.append(f)
                agg = aggregate(feats)
                agg.update(speed_ms=fit.speed_ms, cos_downslope=fit.cos_downslope, n_stations=float(len(members)),
                           fit_rms_s=fit.rms_s)
                last = classify(agg, c)
                rep.features, rep.t_class_s = agg, tc
                rep.speed_ms, rep.cos_downslope, rep.fit_rms_s = fit.speed_ms, fit.cos_downslope, fit.rms_s
                if last.label in HAZARD_LABELS:
                    break
            still_on = any(m.off_s > tc for m in members)
            if tc >= duration or attempt >= ev.t_assoc + c.max_s or not still_on:
                break  # (a non-hazard verdict is only final once the event is over or the retry budget is spent)
            attempt += c.retry_s
        if last is not None:
            rep.label, rep.rule, rep.explanation = last.label, last.rule, last.explain() if last.rule else "no rule matched"
        if rep.label in HAZARD_LABELS and rep.t_class_s is not None:
            rep.t_red_s = rep.t_class_s + ov
        return rep
