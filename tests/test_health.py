import numpy as np

from rockfall.config import Config
from rockfall.detect.health import CLIPPING, DATA_GAP, DEAD, OK, STUCK, check_window, health_timeline

RNG = np.random.default_rng(0)


def test_healthy_noise_is_ok(cfg: Config) -> None:
    assert check_window(RNG.normal(size=2000), cfg.health).status == OK


def test_dead_zeros_and_tiny_noise(cfg: Config) -> None:
    assert check_window(np.zeros(2000), cfg.health).status == DEAD
    assert check_window(RNG.normal(0, 0.001, 2000), cfg.health).status == DEAD


def test_stuck_value(cfg: Config) -> None:
    x = RNG.normal(size=2000)
    x[300:] = x[299]
    assert check_window(x, cfg.health).status == STUCK


def test_clipping(cfg: Config) -> None:
    x = np.clip(RNG.normal(0, 800, 2000), -1000, 1000)
    res = check_window(x, cfg.health)
    assert res.status == CLIPPING and "rail" in res.detail


def test_a_few_clipped_samples_in_a_real_event_are_tolerated(cfg: Config) -> None:
    x = RNG.normal(size=2000)
    x[100:110] = 1000.0
    assert check_window(x, cfg.health).status == OK


def test_missing_data(cfg: Config) -> None:
    x = RNG.normal(size=2000)
    x[:400] = np.nan
    assert check_window(x, cfg.health).status == DATA_GAP
    assert check_window(np.array([]), cfg.health).status == DATA_GAP


def test_timeline_detects_failure_time(cfg: Config) -> None:
    x = RNG.normal(size=200 * 60)
    x[200 * 25:] = 0.0  # cable dies at t = 25 s
    status = {t: r.status for t, r in health_timeline(x, 200, cfg.health)}
    assert status[10.0] == OK and status[20.0] == OK and status[40.0] == DEAD  # flagged within one window of the failure
