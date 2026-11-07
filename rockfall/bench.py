"""Randomised benchmark: confusion matrix, false alarms per simulated day, misses, latency."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rockfall.alarm import State
from rockfall.config import Config
from rockfall.pipeline import Pipeline, RunResult
from rockfall.sim.generator import Scenario, simulate
from rockfall.sim.scenarios import BENCH_KINDS, BENCH_WEIGHTS, random_scenario
from rockfall.site import Site

TRUTH_TO_LABEL = {"rockfall": "rockfall", "debrisflow": "debris_flow"}
SIZE_BINS = ((30.0, "<30"), (100.0, "30-100"), (300.0, "100-300"), (float("inf"), ">300"))  # impact amplitude, counts
COLUMNS = ("rockfall", "debris_flow", "traffic", "noise", "unclassified", "none")


@dataclass
class BenchResult:
    n: int = 0
    seconds: float = 0.0
    confusion: dict[str, Counter] = field(default_factory=dict)
    hazard_total: Counter = field(default_factory=Counter)
    hazard_detected: Counter = field(default_factory=Counter)
    latencies: dict[str, list[float]] = field(default_factory=dict)
    in_budget: Counter = field(default_factory=Counter)
    quiet_seconds: float = 0.0
    false_red: int = 0
    false_yellow: int = 0
    never_green_violations: int = 0
    by_size: dict[str, list[int]] = field(default_factory=dict)  # bin -> [detected, total]


def predicted_label(run: RunResult) -> str:
    red = [e for e in run.events if e.t_red_s is not None]
    if red:
        return min(red, key=lambda e: e.t_red_s or 0.0).label
    if not run.events:
        return "none"
    return max(run.events, key=lambda e: len(e.stations)).label


def score(sc: Scenario, run: RunResult, cfg: Config, res: BenchResult) -> None:
    hazards = [e for e in sc.events if e.hazardous]
    truth = hazards[0].kind if hazards else (sc.events[0].kind if sc.events else "quiet")
    res.confusion.setdefault(truth, Counter())[predicted_label(run)] += 1
    reds = [t.t for t in run.transitions if t.to_state is State.RED]
    yellows = [t.t for t in run.transitions if t.to_state is State.YELLOW]
    if hazards:
        onset = hazards[0].t_start
        res.hazard_total[truth] += 1
        hit = [t for t in reds if t >= onset - 1.0]
        if truth == "rockfall":
            size = hazards[0].params["size"]
            label = next(lbl for lim, lbl in SIZE_BINS if size < lim)
            cell = res.by_size.setdefault(label, [0, 0])
            cell[0] += bool(hit)
            cell[1] += 1
        if hit:
            lat = hit[0] - onset
            res.hazard_detected[truth] += 1
            res.latencies.setdefault(truth, []).append(lat)
            if lat <= cfg.latency.budget_s.get(TRUTH_TO_LABEL[truth], float("inf")):
                res.in_budget[truth] += 1
    else:
        res.quiet_seconds += sc.duration_s
        res.false_red += len(reds)
        res.false_yellow += len(yellows)
    # fail-safe invariant: with >= half of the stations faulty we must not end GREEN
    healthy = len(run.health) - len({f.station for f in sc.faults})
    if healthy < cfg.alarm.min_healthy_stations and run.final_state is State.GREEN:
        res.never_green_violations += 1


def run_bench(cfg: Config, site: Site, n: int = 120, seed: int = 1, progress: bool = False) -> BenchResult:
    rng = np.random.default_rng(seed)
    kinds = rng.choice(len(BENCH_KINDS), size=n, p=BENCH_WEIGHTS)
    pipe = Pipeline(cfg, site)
    res = BenchResult(n=n)
    t0 = time.time()
    for i, k in enumerate(kinds):
        sc = random_scenario(BENCH_KINDS[int(k)], seed=int(seed * 100_000 + i))
        score(sc, pipe.run(simulate(sc, site)), cfg, res)
        if progress and (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{n}", flush=True)
    res.seconds = time.time() - t0
    return res


def format_report(res: BenchResult) -> str:
    out = [f"scenarios: {res.n}   wall time: {res.seconds:.0f}s", "", "Confusion matrix (rows = truth, columns = classifier verdict of the event chain)"]
    out.append(f"{'':>12}" + "".join(f"{c:>14}" for c in COLUMNS))
    for truth in ("rockfall", "debrisflow", "traffic", "wind", "animal", "quake", "quiet"):
        row = res.confusion.get(truth, Counter())
        out.append(f"{truth:>12}" + "".join(f"{row.get(c, 0):>14}" for c in COLUMNS))
    out.append("")
    for truth in ("rockfall", "debrisflow"):
        tot, det = res.hazard_total[truth], res.hazard_detected[truth]
        lats = res.latencies.get(truth, [])
        p = (f"p50 {np.percentile(lats, 50):.1f}s  p95 {np.percentile(lats, 95):.1f}s  max {max(lats):.1f}s"
             if lats else "n/a")
        out.append(f"{truth:>11}: detected {det}/{tot}  missed {tot - det}  onset->RED latency {p}  within budget {res.in_budget[truth]}/{det}")
    allat = [x for v in res.latencies.values() for x in v]
    if allat:
        out.append(f"{'all hazards':>11}: onset->RED latency p50 {np.percentile(allat, 50):.1f}s  p95 {np.percentile(allat, 95):.1f}s")
    if res.by_size:
        out.append("rockfall detection by median impact amplitude (counts; noise sigma ~1): " + "  ".join(
            f"{lbl}: {res.by_size[lbl][0]}/{res.by_size[lbl][1]}" for _, lbl in SIZE_BINS if lbl in res.by_size))
    days = res.quiet_seconds / 86400.0
    out.append("")
    out.append(f"non-hazard simulated time: {res.quiet_seconds / 3600:.1f} h ({days:.3f} day)")
    out.append(f"false RED   : {res.false_red}  ->  {res.false_red / days:.1f} per simulated day (resolution: 1 event = {1 / days:.1f}/day)")
    out.append(f"false YELLOW: {res.false_yellow}  ->  {res.false_yellow / days:.1f} per simulated day")
    out.append(f"fail-safe violations (>= half the stations faulty yet final state GREEN): {res.never_green_violations}")
    return "\n".join(out)
