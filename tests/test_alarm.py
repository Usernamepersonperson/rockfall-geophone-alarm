import dataclasses

from rockfall.alarm import AlarmStateMachine, State
from rockfall.config import Config


def machine(cfg: Config, **over: object) -> AlarmStateMachine:
    sm = AlarmStateMachine(dataclasses.replace(cfg.alarm, **over))  # type: ignore[arg-type]
    sm.health(10.0, 6, 6)
    return sm


def seq(sm: AlarmStateMachine) -> list[str]:
    return [t.to_state.value for t in sm.transitions]


def test_starts_degraded_not_green(cfg: Config) -> None:
    sm = AlarmStateMachine(cfg.alarm)
    assert sm.state is State.DEGRADED
    sm.health(10.0, 6, 6)
    assert sm.state is State.GREEN and seq(sm) == ["DEGRADED", "GREEN"]


def test_yellow_then_red_then_back_after_hold(cfg: Config) -> None:
    sm = machine(cfg)
    sm.candidate(100.0, "coincidence")
    sm.hazard(110.0, "rockfall")
    assert sm.state is State.RED and sm.road_closed()
    sm.advance(110.0 + cfg.alarm.red_hold_s - 1)
    assert sm.state is State.RED  # hold not over
    sm.advance(110.0 + cfg.alarm.red_hold_s + 1)
    assert sm.state is State.GREEN
    assert seq(sm) == ["DEGRADED", "GREEN", "YELLOW", "RED", "GREEN"]
    assert sm.transitions[-1].t == 110.0 + cfg.alarm.red_hold_s  # exact expiry time, not poll time


def test_new_evidence_extends_red_hold(cfg: Config) -> None:
    sm = machine(cfg)
    sm.hazard(100.0, "a")
    sm.hazard(300.0, "b")
    sm.advance(100.0 + cfg.alarm.red_hold_s + 5)
    assert sm.state is State.RED
    sm.advance(300.0 + cfg.alarm.red_hold_s + 1)
    assert sm.state is State.GREEN


def test_yellow_expires_without_confirmation(cfg: Config) -> None:
    sm = machine(cfg)
    sm.candidate(100.0, "x")
    sm.advance(100.0 + cfg.alarm.yellow_hold_s + 1)
    assert seq(sm)[-2:] == ["YELLOW", "GREEN"]


def test_red_needs_ack_when_required(cfg: Config) -> None:
    sm = machine(cfg, require_ack=True)
    sm.hazard(100.0, "x")
    sm.advance(100.0 + 10 * cfg.alarm.red_hold_s)
    assert sm.state is State.RED  # hold elapsed, but nobody acknowledged
    sm.ack(5000.0)
    assert sm.state is State.GREEN


def test_early_ack_waits_for_hold_and_new_hazard_clears_it(cfg: Config) -> None:
    sm = machine(cfg, require_ack=True)
    sm.hazard(100.0, "x")
    sm.ack(150.0)
    assert sm.state is State.RED  # hold still running
    sm.advance(100.0 + cfg.alarm.red_hold_s + 1)
    assert sm.state is State.GREEN  # ack + hold both satisfied

    sm2 = machine(cfg, require_ack=True)
    sm2.hazard(100.0, "x")
    sm2.ack(150.0)
    sm2.hazard(200.0, "again")  # fresh evidence invalidates the old ack
    sm2.advance(200.0 + cfg.alarm.red_hold_s + 10)
    assert sm2.state is State.RED
    sm2.ack(1000.0)
    assert sm2.state is State.GREEN


def test_degraded_when_too_few_stations_and_recovery_delay(cfg: Config) -> None:
    sm = machine(cfg)
    sm.health(20.0, 3, 6)
    assert sm.state is State.DEGRADED and sm.transitions[-1].light == "yellow_flashing"
    sm.health(30.0, 6, 6)
    assert sm.state is State.DEGRADED  # not trusted again yet
    sm.advance(30.0 + cfg.alarm.degraded_recovery_s + 1)
    assert sm.state is State.GREEN


def test_unhealthy_blip_resets_recovery_timer(cfg: Config) -> None:
    sm = machine(cfg)
    sm.health(20.0, 2, 6)
    sm.health(30.0, 6, 6)
    sm.health(60.0, 2, 6)
    sm.health(70.0, 6, 6)
    sm.advance(30.0 + cfg.alarm.degraded_recovery_s + 1)
    assert sm.state is State.DEGRADED
    sm.advance(70.0 + cfg.alarm.degraded_recovery_s + 1)
    assert sm.state is State.GREEN


def test_hazard_while_degraded_is_red_and_returns_to_degraded_not_green(cfg: Config) -> None:
    sm = machine(cfg)
    sm.health(20.0, 3, 6)
    sm.hazard(50.0, "x")
    assert sm.state is State.RED
    sm.advance(50.0 + cfg.alarm.red_hold_s + 1)
    assert sm.state is State.DEGRADED  # still blind: never silently green
    sm.advance(1e6)
    assert sm.state is State.DEGRADED


def test_yellow_during_degraded_shows_degraded(cfg: Config) -> None:
    sm = machine(cfg)
    sm.health(20.0, 3, 6)
    sm.candidate(30.0, "x")
    assert sm.state is State.DEGRADED


def test_degraded_road_policy(cfg: Config) -> None:
    open_ = machine(cfg)
    open_.health(20.0, 1, 6)
    closed = machine(cfg, degraded_closes_road=True)
    closed.health(20.0, 1, 6)
    assert not open_.road_closed() and closed.road_closed()


def test_transition_callback_receives_each_change(cfg: Config) -> None:
    got: list[str] = []
    sm = AlarmStateMachine(cfg.alarm, lambda t: got.append(t.to_state.value))
    sm.health(10.0, 6, 6)
    sm.hazard(20.0, "x")
    assert got == ["DEGRADED", "GREEN", "RED"]
