"""Typed configuration loaded from JSON. All thresholds live in ``config/default.json``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).parent / "config" / "default.json"


@dataclass(frozen=True)
class DetectorCfg:
    sta_s: float
    lta_s: float
    on: float
    off: float
    freeze_lta: bool
    max_trigger_s: float
    merge_gap_s: float
    min_trigger_s: float


@dataclass(frozen=True)
class AssociationCfg:
    min_stations: int
    max_span_s: float
    rms_tol_base_s: float
    rms_tol_frac: float
    min_speed_ms: float
    max_speed_ms: float
    min_cos_downslope: float


@dataclass(frozen=True)
class Rule:
    """One transparent classifier rule: all conditions must hold. ``[lo, hi]``, null = open."""

    name: str
    label: str
    doc: str
    conditions: dict[str, tuple[float | None, float | None]]


@dataclass(frozen=True)
class ClassificationCfg:
    window_s: float
    retry_s: float
    max_s: float
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class HealthCfg:
    window_s: float
    dead_std: float
    stuck_frac: float
    clip_level: float
    clip_frac: float
    gap_frac: float


@dataclass(frozen=True)
class AlarmCfg:
    yellow_hold_s: float
    red_hold_s: float
    require_ack: bool
    min_healthy_stations: int
    degraded_recovery_s: float
    degraded_closes_road: bool


@dataclass(frozen=True)
class LatencyCfg:
    acquisition_block_s: float
    telemetry_s: float
    processing_s: float
    actuation_s: float
    budget_s: dict[str, float]

    @property
    def fixed_overhead_s(self) -> float:
        """Latency added to every decision that is not caused by the signal itself."""
        return self.acquisition_block_s + self.telemetry_s + self.processing_s + self.actuation_s


@dataclass(frozen=True)
class Config:
    site: str
    fs: int
    highpass_hz: float
    detector: DetectorCfg
    association: AssociationCfg
    classification: ClassificationCfg
    health: HealthCfg
    alarm: AlarmCfg
    latency: LatencyCfg

    def __post_init__(self) -> None:
        if self.alarm.min_healthy_stations < self.association.min_stations:
            raise ValueError("min_healthy_stations must be >= association.min_stations")
        if self.detector.off >= self.detector.on:
            raise ValueError("detector.off must be below detector.on (hysteresis)")


def from_dict(raw: dict[str, Any]) -> Config:
    cls_raw = raw["classification"]
    rules = tuple(
        Rule(r["name"], r["label"], r.get("doc", ""),
             {k: (v[0], v[1]) for k, v in r["conditions"].items()})
        for r in cls_raw["rules"]
    )
    return Config(
        site=raw["site"], fs=int(raw["fs"]), highpass_hz=float(raw["highpass_hz"]),
        detector=DetectorCfg(**raw["detector"]),
        association=AssociationCfg(**raw["association"]),
        classification=ClassificationCfg(cls_raw["window_s"], cls_raw["retry_s"], cls_raw["max_s"], rules),
        health=HealthCfg(**raw["health"]),
        alarm=AlarmCfg(**raw["alarm"]),
        latency=LatencyCfg(**raw["latency"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        return from_dict(json.load(fh))
