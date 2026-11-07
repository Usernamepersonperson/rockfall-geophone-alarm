import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from rockfall.alarm import State
from rockfall.config import DEFAULT_PATH, Config, from_dict, load_config
from rockfall.pipeline import Pipeline, RunResult
from rockfall.sim.generator import Fault, Scenario, TruthEvent, simulate
from rockfall.sim.scenarios import make_traffic, named_scenario
from rockfall.sinks import JsonlSink, verify_audit_log
from rockfall.site import Site


def run(pipeline: Pipeline, site: Site, name: str, seed: int = 0) -> tuple[Scenario, RunResult]:
    sc = named_scenario(name, seed)
    return sc, pipeline.run(simulate(sc, site))


def hazard_sequence(states: list[str]) -> list[str]:
    """Collapse the start-up self-test and the optional YELLOW step."""
    return [s for s in states if s != "YELLOW"][1:]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rockfall_goes_red_then_green_after_hold(pipeline: Pipeline, site: Site, seed: int) -> None:
    sc, res = run(pipeline, site, "rockfall", seed)
    assert hazard_sequence(res.states) == ["GREEN", "RED", "GREEN"]
    onset = sc.events[0].t_start
    assert onset < res.first_time(State.RED) < onset + 30  # inside the 30 s budget
    assert next(t for t in res.transitions if t.to_state is State.RED).road_closed
    (ev,) = res.events
    assert ev.label == "rockfall" and ev.t_red_s is not None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_debris_flow_yellow_then_red(pipeline: Pipeline, site: Site, seed: int) -> None:
    sc, res = run(pipeline, site, "debrisflow", seed)
    assert res.states == ["DEGRADED", "GREEN", "YELLOW", "RED", "GREEN"]
    assert res.first_time(State.RED) - sc.events[0].t_start < 90
    assert res.first_time(State.YELLOW) < res.first_time(State.RED)


def test_traffic_never_goes_red(pipeline: Pipeline, site: Site) -> None:
    for seed in range(4):
        _, res = run(pipeline, site, "traffic", seed)
        assert "RED" not in res.states
        assert res.final_state is State.GREEN


def test_wind_gust_never_goes_red(pipeline: Pipeline, site: Site) -> None:
    sc = Scenario("wind", 200.0, (TruthEvent("wind", 60.0, 100.0, {"gain": 8.0}),), seed=5)
    assert "RED" not in pipeline.run(simulate(sc, site)).states


def test_dead_and_clipping_station_still_alarm(pipeline: Pipeline, site: Site) -> None:
    _, res = run(pipeline, site, "dead-station")
    last = {k: v[-1][1].status for k, v in res.health.items()}
    assert last["G2"] == "DEAD" and last["G5"] == "CLIPPING"
    assert hazard_sequence(res.states) == ["GREEN", "RED", "GREEN"]
    assert all("G2" not in e.stations and "G5" not in e.stations for e in res.events)


def test_too_many_faults_end_degraded_never_green(pipeline: Pipeline, site: Site) -> None:
    _, res = run(pipeline, site, "degraded")
    assert res.final_state is State.DEGRADED
    assert "RED" in res.states  # a real rockfall is still reported, even half blind
    assert res.states[-1] == "DEGRADED" and res.states[-2] == "RED"


def test_mixed_day_sequence(pipeline: Pipeline, site: Site) -> None:
    _, res = run(pipeline, site, "mixed")
    assert res.states.count("RED") == 2  # the rockfall and the debris flow, nothing else
    labels = {e.label for e in res.events}
    assert {"rockfall", "debris_flow"} <= labels


def test_require_ack_holds_red(site: Site, cfg: Config) -> None:
    strict = dataclasses.replace(cfg, alarm=dataclasses.replace(cfg.alarm, require_ack=True))
    data = simulate(named_scenario("rockfall", 0), site)
    assert Pipeline(strict, site).run(data).final_state is State.RED
    acked = Pipeline(strict, site).run(data, ack_times=[200.0])
    assert acked.final_state is State.GREEN and acked.transitions[-1].t >= 200.0


def test_audit_log_written_and_verifiable(site: Site, cfg: Config, tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    res = Pipeline(cfg, site, [JsonlSink(log)]).run(simulate(named_scenario("debrisflow", 1), site))
    assert verify_audit_log(log)
    assert len(log.read_text().splitlines()) == len(res.transitions)
    red = next(r for r in res.records if r["state"] == "RED")
    lat = red["event"]["latency"]
    assert lat["total_after_first_trigger"] > lat["classification_window"] > 0
    assert red["road_closed"] is True


def test_simulation_is_deterministic(site: Site) -> None:
    a = simulate(named_scenario("rockfall", 3), site)
    b = simulate(named_scenario("rockfall", 3), site)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_faults_shape_the_raw_signal(site: Site) -> None:
    sc = Scenario("f", 60.0, (), (Fault("G1", "dead"), Fault("G2", "clipping"), Fault("G3", "stuck", 30.0)), seed=1)
    d = simulate(sc, site)
    assert np.all(d["G1"] == 0) and np.mean(np.abs(d["G2"]) >= 1000) > 0.05
    assert len(np.unique(d["G3"][30 * 200:])) == 1


def test_traffic_signal_stands_out_of_the_noise(site: Site) -> None:
    ev = make_traffic(np.random.default_rng(0), 10.0)
    d = simulate(Scenario("t", 60.0, (ev,), seed=0), site)
    assert np.abs(d["G6"]).max() > 5 * np.abs(d["G6"][:1000]).std()


def test_config_validation() -> None:
    raw = json.loads(DEFAULT_PATH.read_text())
    raw["alarm"]["min_healthy_stations"] = 2
    with pytest.raises(ValueError):
        from_dict(raw)
    assert load_config().classification.rules[0].conditions
