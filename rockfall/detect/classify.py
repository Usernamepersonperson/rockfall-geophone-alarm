"""Transparent rule-based event classifier.  No learned model: every decision can be audited.

Rules live in ``config/default.json`` and are evaluated top to bottom; the first rule whose
conditions *all* hold wins.  Each condition is ``feature: [lo, hi]`` (null = open end).
The result keeps every individual check, so an operator can see why an event was (not)
called a rockfall.  No rule matches -> ``unclassified`` (the alarm logic treats that as
"keep YELLOW", never as "ignore").

Default rules (thresholds in the config):

1. ``simultaneous`` -> noise.        apparent speed >= 150 m/s: all stations trip together, so it is
                                     not a front moving on the slope (wind gust, regional quake, EMI).
2. ``traffic`` -> traffic.           narrowband (peakiness >= 12), dominant 6-45 Hz, 5-60 s, smooth
                                     envelope (env_cv <= 0.8): engine harmonics of a passing vehicle.
3. ``debris_flow`` -> debris_flow.   >= 25 s, spectral centroid <= 20 Hz, non-bursty envelope,
                                     apparent speed 1-12 m/s downslope.
4. ``rockfall`` -> rockfall.         1-40 s, crest factor >= 6, bursty envelope (env_cv >= 0.9),
                                     not narrowband, apparent speed 5-150 m/s downslope.
"""
from __future__ import annotations

from dataclasses import dataclass

from rockfall.config import ClassificationCfg, Rule

UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Check:
    feature: str
    value: float | None
    lo: float | None
    hi: float | None
    passed: bool


@dataclass(frozen=True)
class Classification:
    label: str
    rule: str
    checks: tuple[Check, ...]
    near_miss: str = ""  # best non-matching rule, for diagnostics

    def explain(self) -> str:
        parts = []
        for c in self.checks:
            rng = f"[{'-inf' if c.lo is None else c.lo}, {'inf' if c.hi is None else c.hi}]"
            val = "n/a" if c.value is None else f"{c.value:.3g}"
            parts.append(f"{c.feature}={val} {'in' if c.passed else 'NOT in'} {rng}")
        return f"{self.label} via '{self.rule}': " + "; ".join(parts)


def evaluate_rule(rule: Rule, features: dict[str, float]) -> tuple[bool, tuple[Check, ...]]:
    checks = []
    for feat, (lo, hi) in rule.conditions.items():
        v = features.get(feat)
        ok = v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
        checks.append(Check(feat, v, lo, hi, ok))
    return all(c.passed for c in checks), tuple(checks)


def classify(features: dict[str, float], cfg: ClassificationCfg) -> Classification:
    """``features`` = aggregated station features plus ``speed_ms`` and ``cos_downslope``."""
    best_miss, best_frac = "", -1.0
    for rule in cfg.rules:
        ok, checks = evaluate_rule(rule, features)
        if ok:
            return Classification(rule.label, rule.name, checks)
        frac = sum(c.passed for c in checks) / len(checks)
        if frac > best_frac:
            best_miss, best_frac = rule.name, frac
    return Classification(UNCLASSIFIED, "", (), near_miss=best_miss)
