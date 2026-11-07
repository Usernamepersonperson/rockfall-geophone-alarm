"""Alarm state machine: GREEN -> YELLOW -> RED, plus DEGRADED (fail-safe).

Two independent things are tracked and then composed:

* ``level``  GREEN / YELLOW / RED from event evidence, with hold times.
* ``degraded``  True when too few stations are healthy (and at start-up until the first
  health check passes).

Composed state precedence: RED > DEGRADED > YELLOW > GREEN.  So a broken network is never
reported as GREEN; a real hazard still shows RED even while degraded.

Rules
-----
* ``candidate()``  (associated, plausible downslope event) -> at least YELLOW, hold ``yellow_hold_s``.
* ``hazard()``     (classified rockfall / debris flow)       -> RED, hold ``red_hold_s``; new evidence extends the hold.
* RED returns to GREEN only when the hold has elapsed AND, if ``require_ack``, an operator acknowledged.
  An ack given early is remembered and takes effect when the hold ends; fresh hazard evidence clears it.
* DEGRADED clears after the network was healthy for ``degraded_recovery_s`` (immediately for the start-up check).
* ``degraded_closes_road`` decides whether DEGRADED also closes the road (site policy).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from rockfall.config import AlarmCfg


class State(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    DEGRADED = "DEGRADED"


LIGHT = {State.GREEN: "green", State.YELLOW: "yellow", State.RED: "red", State.DEGRADED: "yellow_flashing"}


@dataclass(frozen=True)
class Transition:
    t: float
    from_state: State
    to_state: State
    reason: str
    light: str
    road_closed: bool


class AlarmStateMachine:
    def __init__(self, cfg: AlarmCfg, on_transition: Callable[[Transition], None] | None = None) -> None:
        self.cfg = cfg
        self.transitions: list[Transition] = []
        self._cb = on_transition
        self._level = State.GREEN
        self._yellow_until = 0.0
        self._red_until = 0.0
        self._acked_at: float | None = None
        self._degraded = True  # fail-safe: not trusted until the first health check passes
        self._startup = True
        self._recover_at: float | None = None
        self._state = State.GREEN  # placeholder so the start-up transition is GREEN -> DEGRADED
        self._now = 0.0
        self._emit(0.0, State.DEGRADED, "start-up: waiting for first health check")

    # ---- public API -------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    def road_closed(self) -> bool:
        return self._state is State.RED or (self._state is State.DEGRADED and self.cfg.degraded_closes_road)

    def advance(self, t: float) -> None:
        """Apply every hold / recovery expiry up to time ``t`` in chronological order."""
        while True:
            nxt = self._next_deadline()
            if nxt is None or nxt[0] > t:
                break
            when, kind = nxt
            if kind == "yellow":
                self._level = State.GREEN
                self._update(when, f"yellow hold {self.cfg.yellow_hold_s:.0f}s elapsed")
            elif kind == "red":
                self._level = State.GREEN
                self._update(when, "red hold elapsed" + (" and acknowledged" if self.cfg.require_ack else ""))
            else:
                self._degraded = False
                self._recover_at = None
                self._update(when, "start-up health check passed" if self._startup else
                             f"network healthy for {self.cfg.degraded_recovery_s:.0f}s")
        self._now = max(self._now, t)

    def candidate(self, t: float, reason: str) -> None:
        self.advance(t)
        if self._level is State.GREEN:
            self._level = State.YELLOW
        self._yellow_until = max(self._yellow_until, t + self.cfg.yellow_hold_s)
        self._update(t, reason)

    def hazard(self, t: float, reason: str) -> None:
        self.advance(t)
        self._level = State.RED
        self._red_until = max(self._red_until, t + self.cfg.red_hold_s)
        self._acked_at = None
        self._update(t, reason)

    def ack(self, t: float) -> None:
        self.advance(t)
        if self._level is State.RED:
            self._acked_at = t
            self.advance(t)  # releases immediately if the hold is already over

    def health(self, t: float, healthy: int, total: int) -> None:
        self.advance(t)
        if healthy < self.cfg.min_healthy_stations:
            self._degraded = True
            self._recover_at = None
            self._update(t, f"only {healthy}/{total} stations healthy (< {self.cfg.min_healthy_stations})")
        elif self._degraded and self._recover_at is None:
            self._recover_at = t if self._startup else t + self.cfg.degraded_recovery_s
            self.advance(t)
        if healthy >= self.cfg.min_healthy_stations:
            self._startup = False

    # ---- internals --------------------------------------------------
    def _next_deadline(self) -> tuple[float, str] | None:
        cands: list[tuple[float, str]] = []
        if self._level is State.YELLOW:
            cands.append((self._yellow_until, "yellow"))
        if self._level is State.RED:
            if not self.cfg.require_ack:
                cands.append((self._red_until, "red"))
            elif self._acked_at is not None:
                cands.append((max(self._red_until, self._acked_at), "red"))
        if self._degraded and self._recover_at is not None:
            cands.append((self._recover_at, "recover"))
        return min(cands) if cands else None

    def _update(self, t: float, reason: str) -> None:
        if self._level is State.RED:
            new = State.RED
        elif self._degraded:
            new = State.DEGRADED
        else:
            new = self._level
        if new is not self._state:
            self._emit(t, new, reason)

    def _emit(self, t: float, new: State, reason: str) -> None:
        old = self._state
        self._state = new
        tr = Transition(t, old, new, reason, LIGHT[new], self.road_closed())
        self.transitions.append(tr)
        if self._cb:
            self._cb(tr)
