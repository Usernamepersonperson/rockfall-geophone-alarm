import pytest

from rockfall.config import Config
from rockfall.detect.classify import UNCLASSIFIED, classify

BASE = dict(duration_s=10.0, peak_amp=100.0, dom_freq_hz=20.0, centroid_hz=30.0, peakiness=4.0, rise_time_s=2.0,
            crest_factor=10.0, env_cv=1.3, speed_ms=60.0, cos_downslope=1.0, n_stations=5.0)
DEBRIS = dict(duration_s=60.0, centroid_hz=9.0, dom_freq_hz=7.0, crest_factor=4.5, env_cv=0.5, peakiness=9.0,
              speed_ms=4.0, cos_downslope=1.0)
TRAFFIC = dict(peakiness=60.0, dom_freq_hz=11.0, crest_factor=3.0, env_cv=0.3, duration_s=20.0, speed_ms=15.0,
               centroid_hz=15.0)


def verdict(cfg: Config, **kw: float) -> tuple[str, str]:
    c = classify({**BASE, **kw}, cfg.classification)
    return c.label, c.rule


def test_rockfall_moving(cfg: Config) -> None:
    assert verdict(cfg) == ("rockfall", "rockfall_moving")


def test_rockfall_impact_when_onsets_are_simultaneous(cfg: Config) -> None:
    assert verdict(cfg, speed_ms=500.0, cos_downslope=-1.0) == ("rockfall", "rockfall_impact")


def test_simultaneous_but_not_impulsive_is_noise(cfg: Config) -> None:
    assert verdict(cfg, speed_ms=500.0, crest_factor=4.0, env_cv=0.3) == ("noise", "simultaneous")


def test_low_frequency_quake_is_not_a_rockfall(cfg: Config) -> None:
    assert verdict(cfg, speed_ms=3000.0, centroid_hz=9.0)[0] == "noise"


def test_traffic_is_narrowband(cfg: Config) -> None:
    assert verdict(cfg, **TRAFFIC) == ("traffic", "traffic")


@pytest.mark.parametrize("override", [{"duration_s": 2.0}, {"duration_s": 120.0}, {"peakiness": 5.0},
                                      {"dom_freq_hz": 2.0}, {"env_cv": 1.5}])
def test_traffic_each_condition_matters(cfg: Config, override: dict[str, float]) -> None:
    assert verdict(cfg, **{**TRAFFIC, **override})[0] != "traffic"


def test_debris_flow(cfg: Config) -> None:
    assert verdict(cfg, **DEBRIS) == ("debris_flow", "debris_flow")


@pytest.mark.parametrize("override", [{"duration_s": 10.0}, {"speed_ms": 40.0}, {"cos_downslope": -1.0},
                                      {"centroid_hz": 30.0}, {"env_cv": 1.4}])
def test_debris_flow_each_condition_matters(cfg: Config, override: dict[str, float]) -> None:
    assert verdict(cfg, **{**DEBRIS, **override})[0] != "debris_flow"


@pytest.mark.parametrize("override", [{"crest_factor": 3.0}, {"env_cv": 0.4}, {"cos_downslope": -0.9},
                                      {"speed_ms": 2.0}, {"duration_s": 90.0}, {"peakiness": 40.0},
                                      {"centroid_hz": 10.0}])
def test_rockfall_each_condition_matters(cfg: Config, override: dict[str, float]) -> None:
    assert verdict(cfg, **override)[0] != "rockfall"


def test_unclassified_and_explanation(cfg: Config) -> None:
    c = classify({**BASE, "crest_factor": 3.0, "env_cv": 0.5, "duration_s": 15.0}, cfg.classification)
    assert c.label == UNCLASSIFIED and c.near_miss
    ok = classify(BASE, cfg.classification)
    assert "crest_factor=10 in [6.0, inf]" in ok.explain()


def test_missing_feature_fails_the_rule(cfg: Config) -> None:
    feats = {k: v for k, v in BASE.items() if k != "crest_factor"}
    assert classify(feats, cfg.classification).label == UNCLASSIFIED
